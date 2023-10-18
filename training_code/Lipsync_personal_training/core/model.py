import torch
import torch.nn.functional as F
from core.loss import MyModelLoss
from core.nets import MyGenerator
from innerverz import DECA
from lib import utils
from lib.discriminators import ProjectedDiscriminator
from lib.model import ModelInterface
from SyncNet import SyncNet


class MyModel(ModelInterface):
    def declare_networks(self):
        # INIT
        self.G_pre = MyGenerator(
            skip=self.CONFIG["NETWORK"]["ADD_SKIP"],
            ref_input=self.CONFIG["NETWORK"]["REF_INPUT"],
        ).cuda()
        self.G = MyGenerator(
            skip=self.CONFIG["NETWORK"]["ADD_SKIP"],
            ref_input=self.CONFIG["NETWORK"]["REF_INPUT"],
        ).cuda()
        self.D = ProjectedDiscriminator().cuda()
        self.S = SyncNet()

        # LOAD CKPT
        ckpt_dict_G = torch.load(
            self.CONFIG["CKPT"]["G_PRE_PATH"], map_location=torch.device("cuda")
        )
        ckpt_dict_D = torch.load(
            self.CONFIG["CKPT"]["D_PRE_PATH"], map_location=torch.device("cuda")
        )
        self.G_pre.load_state_dict(ckpt_dict_G["model"], strict=False)
        self.G.load_state_dict(ckpt_dict_G["model"], strict=False)
        self.D.load_state_dict(ckpt_dict_D["model"], strict=False)

        # FREEZE SOME NET
        self.G_pre.eval()
        self.S.eval()

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
            "reg_flame_params",
            "hubert_feature",
            "reg_hubert_feature",
            "mel_feature",
            "reg_mel_feature",
        ]
        self.saving_data_names = [
            "gt_img",
            "input",
            "ref_img",
            "output",
            "result_img",
            "crop_img",
            "reg_input",
            "reg_result_img",
        ]

        batch_data_bundle = self.load_next_batch(
            self.train_dataloader, self.train_iterator, "train"
        )

        for data_name, batch_data in zip(self.batch_data_names, batch_data_bundle):
            self.train_dict[data_name] = batch_data

        # run G
        self.run_G(self.train_dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(
            self.train_dict, model=self.G, prev_model=self.G_pre
        )
        self.update_net(self.opt_G, loss_G)

        # run D
        self.run_D(self.train_dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        self.update_net(self.opt_D, loss_D)

        # print images
        self.train_images = [self.train_dict[data_name] for data_name in self.saving_data_names]

    def run_G(self, run_dict):
        with torch.no_grad():
            # for calculation reshape tensor (B*T, 3, H, W)
            for key in run_dict["flame_params"].keys():
                run_dict["flame_params"][key] = torch.cat(
                    [
                        run_dict["flame_params"][key][:, i]
                        for i in range(run_dict["flame_params"][key].shape[1])
                    ],
                    dim=0,
                )
                run_dict["reg_flame_params"][key] = torch.cat(
                    [
                        run_dict["reg_flame_params"][key][:, i]
                        for i in range(run_dict["reg_flame_params"][key].shape[1])
                    ],
                    dim=0,
                )
            run_dict["gt_img"] = torch.cat(
                [run_dict["gt_img"][:, i] for i in range(run_dict["gt_img"].shape[1])],
                dim=0,
            )
            run_dict["guide_img"] = torch.cat(
                [run_dict["guide_img"][:, i] for i in range(run_dict["guide_img"].shape[1])],
                dim=0,
            )
            run_dict["ref_img"] = torch.cat(
                [run_dict["ref_img"][:, i] for i in range(run_dict["ref_img"].shape[1])],
                dim=0,
            )
            run_dict["hubert_feature"] = torch.cat(
                [
                    run_dict["hubert_feature"][:, i]
                    for i in range(run_dict["hubert_feature"].shape[1])
                ],
                dim=0,
            ).unsqueeze(1)
            run_dict["reg_hubert_feature"] = torch.cat(
                [
                    run_dict["reg_hubert_feature"][:, i]
                    for i in range(run_dict["reg_hubert_feature"].shape[1])
                ],
                dim=0,
            ).unsqueeze(1)

            # make masked face
            vis_dict = self.DC.decode(
                run_dict["flame_params"],
                original_image=run_dict["gt_img"],
                tform_invs=run_dict["flame_params"]["tform_inv"],
            )
            reg_vis_dict = self.DC.decode(
                run_dict["reg_flame_params"],
                original_image=run_dict["gt_img"],
                tform_invs=run_dict["flame_params"]["tform_inv"],
            )
            run_dict["reg_guide_imgs"] = utils.draw_landmarks_batch(
                reg_vis_dict["landmarks2d_points"] / 2, color=(1, 1, 1), size=2
            )
            run_dict["mask"] = utils.get_convexhull_mask(
                run_dict["gt_img"], vis_dict["landmarks2d_points"] / 2
            )  # mask = polygon2mask((H,W), np.array([x1,y1],[x2,y2],[x3,y3]))
            run_dict["reg_mask"] = utils.get_convexhull_mask(
                run_dict["gt_img"],
                reg_vis_dict["landmarks2d_points"] / 2,
                or_mask=run_dict["mask"],
            )  # mask = polygon2mask((H,W), np.array([x1,y1],[x2,y2],[x3,y3]))
            blur_mask = utils.blur_mask(run_dict["mask"])
            reg_blur_mask = utils.blur_mask(run_dict["reg_mask"] | run_dict["mask"])

            run_dict["masked_face"] = run_dict["gt_img"] * (1 - blur_mask)
            run_dict["reg_masked_face"] = run_dict["gt_img"] * (1 - reg_blur_mask)
            if self.CONFIG["BASE"]["NO_LMKS"]:
                run_dict["input"] = (
                    torch.zeros_like(run_dict["guide_img"]) * blur_mask + run_dict["masked_face"]
                )
                run_dict["reg_input"] = (
                    torch.zeros_like(run_dict["reg_guide_imgs"]) * reg_blur_mask
                    + run_dict["reg_masked_face"]
                )
            else:
                run_dict["input"] = run_dict["guide_img"] * blur_mask + run_dict["masked_face"]
                run_dict["reg_input"] = (
                    run_dict["reg_guide_imgs"] * reg_blur_mask + run_dict["reg_masked_face"]
                )

            run_dict["input"] = run_dict["guide_img"] * blur_mask + run_dict["masked_face"]
            run_dict["reg_input"] = (
                run_dict["reg_guide_imgs"] * reg_blur_mask + run_dict["reg_masked_face"]
            )

        run_dict["output"] = self.G(
            run_dict["input"], run_dict["ref_img"], run_dict["hubert_feature"]
        )
        run_dict["reg_output"] = self.G(
            run_dict["reg_input"], run_dict["ref_img"], run_dict["reg_hubert_feature"]
        )
        run_dict["result_img"] = run_dict["output"] * blur_mask + run_dict["gt_img"] * (
            1 - blur_mask
        )  # (B, T, 3, H, W)
        run_dict["reg_result_img"] = run_dict["reg_output"] * reg_blur_mask + run_dict["gt_img"] * (
            1 - reg_blur_mask
        )  # (B, T, 3, H, W)

        BT = run_dict["result_img"].shape[0]  # B*T
        img_size = run_dict["result_img"].shape[2]
        sync_imgs, sync_imgs_256 = [], []
        reg_sync_imgs, reg_sync_imgs_256 = [], []
        for i in range(BT):
            sync_img = run_dict["result_img"][i][
                :, img_size // 2 :, img_size // 4 : img_size // 4 + img_size // 2
            ].unsqueeze(0)
            reg_sync_img = run_dict["reg_result_img"][i][
                :, img_size // 2 :, img_size // 4 : img_size // 4 + img_size // 2
            ].unsqueeze(0)
            sync_imgs.append(
                F.interpolate(
                    sync_img, size=(img_size // 2, img_size // 2), mode="bilinear"
                ).squeeze()
            )
            reg_sync_imgs.append(
                F.interpolate(
                    reg_sync_img, size=(img_size // 2, img_size // 2), mode="bilinear"
                ).squeeze()
            )
            sync_imgs_256.append(
                F.interpolate(sync_img, size=(img_size, img_size), mode="bilinear").squeeze()
            )
            reg_sync_imgs_256.append(
                F.interpolate(reg_sync_img, size=(img_size, img_size), mode="bilinear").squeeze()
            )
        run_dict["crop_img"] = torch.stack(sync_imgs_256, dim=0)  # 10, 3, 256, 256
        run_dict["reg_crop_img"] = torch.stack(reg_sync_imgs_256, dim=0)  # 10, 3, 256, 256
        sync_img = torch.stack(sync_imgs, dim=0)  # 10, 3, 224, 224
        reg_sync_img = torch.stack(reg_sync_imgs, dim=0)  # 10, 3, 224, 224
        # sync_img = sync_img[:, :, sync_img.shape[2]//2:]
        sync_img = torch.split(sync_img, sync_img.shape[0] // 5, dim=0)
        reg_sync_img = torch.split(reg_sync_img, reg_sync_img.shape[0] // 5, dim=0)
        sync_img = torch.stack(sync_img, dim=2) * 0.5 + 0.5  # *127.5 + 127.5 # 2, 3, 5, 224, 224
        reg_sync_img = (
            torch.stack(reg_sync_img, dim=2) * 0.5 + 0.5
        )  # *127.5 + 127.5 # 2, 3, 5, 224, 224
        sync_img = torch.cat([sync_img[:, :, i] for i in range(5)], dim=1)  # 2, 15, 224, 224
        reg_sync_img = torch.cat(
            [reg_sync_img[:, :, i] for i in range(5)], dim=1
        )  # 2, 15, 224, 224
        run_dict["mel_embedding"], run_dict["img_embedding"] = self.S(
            run_dict["mel_feature"], sync_img
        )
        run_dict["reg_mel_embedding"], run_dict["reg_img_embedding"] = self.S(
            run_dict["reg_mel_feature"], reg_sync_img
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
