import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, image_paths, lmk_paths, dict_paths, img_size=None):
        self.img_size = img_size
        self.image_paths = image_paths
        self.lmk_paths = lmk_paths
        self.dict_paths = dict_paths

    def __getitem__(self, idx):
        # path setting
        image_path = self.image_paths[idx]
        lmk_path = self.lmk_paths[idx]
        dict_path = self.dict_paths[idx]
        fidx = int(image_path.split("/")[-1].split(".")[0])

        # load data
        image = cv2.imread(image_path)
        insight_lmk = torch.tensor(np.load(lmk_path, allow_pickle=True))
        image_dict = np.load(dict_path, allow_pickle=True).item()

        input_image = torch.tensor(image_dict["image"]).squeeze().float()
        tform_inv = torch.tensor(image_dict["tform_inv"]).squeeze().float()

        return image, input_image, insight_lmk, tform_inv, image_path

    def __len__(self):
        return len(self.image_paths)


def get_batch_dataloader(img_paths, lmk_paths, dict_paths, batch_size, num_worker, img_size=None):
    dataset = SimpleDataset(img_paths, lmk_paths, dict_paths, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    dataloader_iter = iter(dataloader)
    return dataloader_iter


def dict_setting(code_dict, input_images, tform_invs):
    new_image_dict = {}
    new_image_dict["image"] = input_images.cuda()
    new_image_dict["tform_inv"] = tform_invs.cuda()

    new_code_dict = code_dict.copy()
    del new_code_dict["images"]
    del new_code_dict["detail"]

    return new_image_dict, new_code_dict
