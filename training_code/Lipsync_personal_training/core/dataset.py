import os
import random

import numpy as np
import torch
from lib import utils
from lib.dataset import DatasetInterface
from PIL import Image
from torchvision import transforms


class MyDataset(DatasetInterface):
    def __init__(self, CONFIG, dataset_path_list):
        super(MyDataset, self).__init__(CONFIG)
        self.set_tf()
        self.frame_num_per_video = CONFIG["BASE"]["FRAME_NUM_PER_VIDEO"]
        self.same_prob = CONFIG["BASE"]["SAME_PROB"]
        self.forbidden_list = [
            "9MgNeIQ3dYE",
            "XpeWSDxWDnE",
            "jtYhopgMK14",
            "BQtK9CX6L1Y",
        ]  # , 'p5FGIF2LEvQ']
        self.opt_urls = CONFIG["BASE"]["OPT_URL"]
        self.opt_gt_img_list = []
        for gt_img_path in dataset_path_list["gt_img_path_list"]:
            for opt_url in self.opt_urls:
                if opt_url in gt_img_path:
                    self.opt_gt_img_list.append(gt_img_path)
        print("opt frame num :", len(self.opt_gt_img_list))
        self.multi_det_list = []
        f = open(CONFIG["DATASET"]["MULTI_DET_VIDEOS"], "r")
        lines = f.readlines()
        for line in lines:
            self.multi_det_list.append(line)
        f.close()
        self.fps = 25  # frame per second
        self.mps = 80  # mel per second
        self.mpts = 16  # mel per timestep
        self.image_ratio = 512 // CONFIG["BASE"]["IMG_SIZE"]
        for name, list in dataset_path_list.items():
            self.__setattr__(name, list)

    def __getitem__(self, index):
        T = 5  # time step
        is_opt = random.randint(0, 1)
        FLAG = False
        while not FLAG:
            try:
                index_t = random.randint(0, self.__len__() - 1)
                # except selected videos
                gt_img_path = self.gt_img_path_list[index_t]
                if is_opt:
                    gt_img_path = random.choice(self.opt_gt_img_list)
                reg_gt_img_path = random.choice(self.gt_img_path_list)
                reg_video_url_name = reg_gt_img_path.split("/")[-4]
                video_name = gt_img_path.split("/")[-3]
                video_url_name = gt_img_path.split("/")[-4]
                file_name = int(gt_img_path.split("/")[-1].split(".")[0])
                reg_file_name = int(reg_gt_img_path.split("/")[-1].split(".")[0])
                if (
                    video_url_name in self.forbidden_list
                    or reg_video_url_name in self.forbidden_list
                    or file_name + 5 > self.frame_num_per_video
                    or reg_file_name + 5 > self.frame_num_per_video
                    or os.path.dirname(os.path.dirname(gt_img_path)) in self.multi_det_list
                ):
                    continue

                # data path list with time step T
                video_frames_path_list = [
                    path
                    for path in self.gt_img_path_list
                    if (video_url_name in path and video_name in path)
                ]
                ref_img_path = random.choice(video_frames_path_list[:-4])
                ref_file_name = int(ref_img_path.split("/")[-1].split(".")[0])
                gt_img_paths, param_paths, reg_param_paths, ref_img_paths, lmk_paths = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                hubert_path = os.path.join(
                    os.path.dirname(gt_img_path.replace("aligned_imgs", "mel")),
                    "000000.npy",
                )
                reg_hubert_path = os.path.join(
                    os.path.dirname(reg_gt_img_path.replace("aligned_imgs", "mel")),
                    "000000.npy",
                )
                # ref_idx = random.randint(0,len(video_frames_path_list)-6)
                for i in range(T):
                    gt_img_paths.append(
                        gt_img_path.replace("%06d" % file_name, "%06d" % (file_name + i))
                    )
                    lmk_paths.append(
                        os.path.splitext(
                            gt_img_path.replace("aligned_imgs", "2d_lmks").replace(
                                "%06d" % file_name, "%06d" % (file_name + i)
                            )
                        )[0]
                        + ".npy"
                    )
                    param_paths.append(
                        os.path.splitext(
                            gt_img_path.replace("aligned_imgs", "flame_params").replace(
                                "%06d" % file_name, "%06d" % (file_name + i)
                            )
                        )[0]
                        + ".npy"
                    )
                    reg_param_paths.append(
                        os.path.splitext(
                            reg_gt_img_path.replace("aligned_imgs", "flame_params").replace(
                                "%06d" % reg_file_name, "%06d" % (reg_file_name + i)
                            )
                        )[0]
                        + ".npy"
                    )
                    ref_img_paths.append(
                        ref_img_path.replace("%06d" % ref_file_name, "%06d" % (ref_file_name + i))
                    )

                # init data list
                flame_params = {}
                reg_flame_params = {}
                reg_crossid_params = {}
                (
                    gt_imgs,
                    guide_imgs,
                    ref_imgs,
                    hubert_features,
                    reg_hubert_features,
                    mel_features,
                    reg_mel_features,
                ) = ([], [], [], [], [], [], [])
                for key in np.load(param_paths[0], allow_pickle=True).item().keys():
                    flame_params[key] = []
                    reg_flame_params[key] = []
                    reg_crossid_params[key] = []
                jaw_noise = random.uniform(0, 0.2) if random.randint(0, 1) else 0

                # put datas in data list
                hubert_all = torch.tensor(
                    np.load(hubert_path, allow_pickle=True).transpose(1, 0)
                ).float()
                reg_hubert_all = torch.tensor(
                    np.load(reg_hubert_path, allow_pickle=True).transpose(1, 0)
                ).float()
                mel_features.append(
                    hubert_all[
                        :, int(file_name / 25 * 80) : int(file_name / 25 * 80) + 16
                    ].unsqueeze(0)
                )
                reg_mel_features.append(
                    reg_hubert_all[
                        :,
                        int(reg_file_name / 25 * 80) : int(reg_file_name / 25 * 80) + 16,
                    ].unsqueeze(0)
                )
                if (
                    file_name == 0
                    or mel_features[0].shape[2] != 16
                    or reg_mel_features[0].shape[2] != 16
                ):
                    continue

                for i in range(T):
                    lmk_image = np.repeat(
                        utils.draw_landmarks(
                            np.ones((256, 256)) * 127.5,
                            np.load(lmk_paths[i]) / 2,
                            color=(255, 255, 255),
                            size=2,
                        )[:, :, np.newaxis],
                        3,
                        -1,
                    )
                    lmk_image = Image.fromarray(lmk_image.astype(np.uint8)).convert("RGB")
                    guide_imgs.append(self.tf_color(lmk_image).unsqueeze(0))
                    gt_imgs.append(self.pp_image(gt_img_paths[i]).unsqueeze(0))
                    ref_imgs.append(self.pp_image(ref_img_paths[i]).unsqueeze(0))
                    start_id = int(file_name / 25 * 80)  # melmel
                    reg_start_id = int(reg_file_name / 25 * 80)  # melmel
                    hubert_features.append(
                        hubert_all[:, (start_id - 1 + i) : (start_id - 1 + i) + 16].unsqueeze(0)
                    )  # melmel
                    reg_hubert_features.append(
                        reg_hubert_all[
                            :, (reg_start_id - 1 + i) : (reg_start_id - 1 + i) + 16
                        ].unsqueeze(0)
                    )  # melmel
                    tmp_params = np.load(param_paths[i], allow_pickle=True).item()
                    reg_tmp_params = np.load(reg_param_paths[i], allow_pickle=True).item()
                    for key in flame_params.keys():
                        flame_params[key].append(torch.from_numpy(tmp_params[key]).unsqueeze(0))
                        reg_flame_params[key].append(
                            torch.from_numpy(reg_tmp_params[key]).unsqueeze(0)
                        )
                        reg_crossid_params[key].append(flame_params[key][-1].clone())
                    reg_crossid_params["pose"][i][:, 3] = reg_flame_params["pose"][i][:, 3]
                    reg_crossid_params["exp"][i] = reg_flame_params["exp"][i]
                    flame_params["pose"][i][:, 3] = flame_params["pose"][i][:, 3] + jaw_noise

                gt_imgs = torch.cat(gt_imgs, dim=0)  # T, 3, H, W
                guide_imgs = torch.cat(guide_imgs, dim=0)  # T, 3, H, W
                ref_imgs = torch.cat(ref_imgs, dim=0)  # T, 3, H, W
                hubert_features = torch.cat(hubert_features, dim=0)  # T, 80, 16
                reg_hubert_features = torch.cat(reg_hubert_features, dim=0)  # T, 80, 16
                mel_features = torch.cat(mel_features, dim=0)  # 1, 80, 16
                reg_mel_features = torch.cat(reg_mel_features, dim=0)  # 1, 80, 16
                for key in flame_params.keys():
                    flame_params[key] = torch.cat(flame_params[key], dim=0)  # T, param_dim
                    reg_crossid_params[key] = torch.cat(
                        reg_crossid_params[key], dim=0
                    )  # T, param_dim
                if hubert_features.shape[2] != 16:
                    continue

                FLAG = True
            except:
                continue
        return [
            gt_imgs,
            guide_imgs,
            ref_imgs,
            flame_params,
            reg_crossid_params,
            hubert_features,
            reg_hubert_features,
            mel_features,
            reg_mel_features,
        ]

    def __len__(self):
        return len(self.gt_img_path_list)

    # override
    def set_tf(self):
        self.tf_gray = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        self.tf_color = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def divide_datasets(model, CONFIG):
    gt_img_path_list = sorted(
        utils.get_all_images(
            CONFIG["DATASET"]["TRAIN_PATH"]["GT_IMG"],
            CONFIG["BASE"]["FRAME_NUM_PER_VIDEO"],
        )
    )
    opt_img_path_list = sorted(
        utils.get_all_images(
            CONFIG["DATASET"]["TRAIN_PATH"]["OPT_IMG"],
            CONFIG["BASE"]["FRAME_NUM_PER_VIDEO"],
        )
    )
    train_gt_img_path_list, valid_gt_img_path_list = utils.split_dataset(
        gt_img_path_list,
        CONFIG["BASE"]["FRAME_NUM_PER_VIDEO"],
        CONFIG["BASE"]["VAL_SIZE"],
    )
    train_opt_img_path_list, valid_opt_img_path_list = utils.split_dataset(
        opt_img_path_list,
        CONFIG["BASE"]["FRAME_NUM_PER_VIDEO"],
        CONFIG["BASE"]["VAL_SIZE"],
    )

    model.train_dataset_dict = {
        "gt_img_path_list": train_gt_img_path_list + train_opt_img_path_list,
    }
    model.valid_dataset_dict = {
        "gt_img_path_list": valid_gt_img_path_list + valid_opt_img_path_list,
    }
