import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from lib import utils
from lib.model import ModelInterface
from lib.discriminators import ProjectedDiscriminator
from core.loss import MyModelLoss
from core.nets import MyGenerator, S
from innerverz import DECA
from lib.utils import get_convexhull_mask
from SyncNet import SyncNet


class MyModel(ModelInterface):
    def declare_networks(self):
        # init
        self.G = MyGenerator(
            skip=self.CONFIG["NETWORK"]["ADD_SKIP"],
            ref_input=self.CONFIG["NETWORK"]["REF_INPUT"],
        ).cuda()
        self.D = ProjectedDiscriminator().cuda()
        self.S = SyncNet()

        self.set_networks_train_mode()
        self.DC = DECA()

    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)

    def set_networks_eval_mode(self):
        self.G.eval()
        self.D.eval()

    def go_step(self):
        self.batch_data_names = [
            "gt_img",
            "guide_img",
            "ref_img",
            "flame_params",
            "lipsync_feature",
            "syncnet_feature",
        ]
        self.saving_data_names = [
            "gt_img",
            "input",
            "ref_img",
            "output",
            "result_img",
            "crop_img",
        ]

        batch_data_bundle = self.load_next_batch(
            self.train_dataloader, self.train_iterator, "train"
        )

        for data_name, batch_data in zip(self.batch_data_names, batch_data_bundle):
            self.train_dict[data_name] = batch_data

        # run G
        self.run_G(self.train_dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        self.update_net(self.opt_G, loss_G)

        # run D
        self.run_D(self.train_dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        self.update_net(self.opt_D, loss_D)

        # print images
        self.train_images = [
            self.train_dict[data_name] for data_name in self.saving_data_names
        ]

    def run_G(self, run_dict):
        with torch.no_grad():
            # merge B & T / (B, T, ...) -> (B*T, ...)
            for key in run_dict["flame_params"].keys():
                run_dict["flame_params"][key] = torch.cat(
                    [
                        run_dict["flame_params"][key][:, i]
                        for i in range(run_dict["flame_params"][key].shape[1])
                    ],
                    dim=0,
                )
            run_dict["gt_img"] = torch.cat(
                [run_dict["gt_img"][:, i] for i in range(run_dict["gt_img"].shape[1])],
                dim=0,
            )
            run_dict["guide_img"] = torch.cat(
                [
                    run_dict["guide_img"][:, i]
                    for i in range(run_dict["guide_img"].shape[1])
                ],
                dim=0,
            )
            run_dict["ref_img"] = torch.cat(
                [
                    run_dict["ref_img"][:, i]
                    for i in range(run_dict["ref_img"].shape[1])
                ],
                dim=0,
            )
            run_dict["lipsync_feature"] = torch.cat(
                [
                    run_dict["lipsync_feature"][:, i]
                    for i in range(run_dict["lipsync_feature"].shape[1])
                ],
                dim=0,
            ).unsqueeze(1)

            # make masked face
            vis_dict = self.DC.decode(
                run_dict["flame_params"],
                original_image=run_dict["gt_img"],
                tform_invs=run_dict["flame_params"]["tform_inv"],
            )
            run_dict["mask"] = get_convexhull_mask(
                run_dict["gt_img"], vis_dict["landmarks2d_points"] / 2
            )  # mask = polygon2mask((H,W), np.array([x1,y1],[x2,y2],[x3,y3]))
            run_dict["masked_face"] = run_dict["gt_img"] * (1 - run_dict["mask"])
            if self.CONFIG["BASE"]["NO_LMKS"]:
                run_dict["input"] = (
                    torch.zeros_like(run_dict["guide_img"]) * run_dict["mask"]
                    + run_dict["masked_face"]
                )
            else:
                run_dict["input"] = (
                    run_dict["guide_img"] * run_dict["mask"] + run_dict["masked_face"]
                )

        # run Lipsync model
        run_dict["output"] = self.G(
            run_dict["input"], run_dict["ref_img"], run_dict["lipsync_feature"]
        )
        run_dict["result_img"] = run_dict["output"] * run_dict["mask"] + run_dict[
            "gt_img"
        ] * (
            1 - run_dict["mask"]
        )  # (B, T, 3, H, W)

        # run syncnet model
        BT = run_dict["result_img"].shape[0]
        img_size = run_dict["result_img"].shape[2]
        sync_imgs, sync_imgs_256 = [], []
        for i in range(BT):
            # crop images
            sync_img = run_dict["result_img"][i][
                :, img_size // 2 :, img_size // 4 : img_size // 4 + img_size // 2
            ].unsqueeze(0)
            sync_imgs.append(
                F.interpolate(
                    sync_img, size=(img_size // 4, img_size // 2), mode="bilinear"
                ).squeeze()
            )
            sync_imgs_256.append(
                F.interpolate(
                    sync_img, size=(img_size, img_size), mode="bilinear"
                ).squeeze()
            )
        run_dict["crop_img"] = torch.stack(sync_imgs_256, dim=0)  # B*T, 3, 256, 256
        sync_img = torch.stack(sync_imgs, dim=0)  # B*T, 3, H, W
        sync_img = torch.flip(sync_img, [1])  # rgb to bgr
        sync_img = torch.split(sync_img, sync_img.shape[0] // 5, dim=0)
        sync_img = torch.stack(sync_img, dim=2) * 0.5 + 0.5  # B, 3, T, H, W
        sync_img = torch.cat(
            [sync_img[:, :, i] for i in range(5)], dim=1
        )  # B,3*T, H, W
        print(sync_img.shape)
        run_dict["mel_embedding"], run_dict["img_embedding"] = self.S(
            run_dict["syncnet_feature"], sync_img
        )

        g_pred_fake, feat_fake = self.D(run_dict["result_img"], None)
        feat_real = self.D.get_feature(run_dict["gt_img"])

        run_dict["g_feat_fake"] = feat_fake
        run_dict["g_feat_real"] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_real, _ = self.D(run_dict["gt_img"], None)
        d_pred_fake, _ = self.D(run_dict["result_img"].detach(), None)

        run_dict["d_pred_real"] = d_pred_real
        run_dict["d_pred_fake"] = d_pred_fake

    @property
    def loss_collector(self):
        return self._loss_collector

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)
