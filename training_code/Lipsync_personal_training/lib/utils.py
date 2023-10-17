import glob
import json
import os

import _jsonnet
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml


def print_dict(dict):
    print(json.dumps(dict, sort_keys=True, indent=4))


def load_jsonnet(load_path):
    return json.loads(_jsonnet.evaluate_file(load_path))


def save_json(save_path, dict):
    with open(save_path + ".jsonnet", "w") as f:
        json.dump(dict, f, indent=4, sort_keys=True)


def load_yaml(load_path):
    with open(load_path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_yaml(save_path, dict):
    with open(save_path, "w") as f:
        yaml.dump(dict, f)


def make_dirs(CONFIG):
    os.makedirs(CONFIG["BASE"]["SAVE_ROOT"], exist_ok=True)

    train_result_dirs = os.listdir(CONFIG["BASE"]["SAVE_ROOT"])
    if train_result_dirs:
        last_train_index = sorted(train_result_dirs)[-1][:3]
        CONFIG["BASE"][
            "RUN_ID"
        ] = f"{str(int(last_train_index)+1).zfill(3)}_{CONFIG['BASE']['RUN_ID']}"

    else:
        CONFIG["BASE"]["RUN_ID"] = f"000_{CONFIG['BASE']['RUN_ID']}"

    CONFIG["BASE"][
        "SAVE_ROOT_RUN"
    ] = f"{CONFIG['BASE']['SAVE_ROOT']}/{CONFIG['BASE']['RUN_ID']}"
    os.makedirs(CONFIG["BASE"]["SAVE_ROOT_RUN"], exist_ok=True)

    CONFIG["BASE"]["SAVE_ROOT_CKPT"] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/ckpt"
    CONFIG["BASE"]["SAVE_ROOT_IMGS"] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/imgs"
    CONFIG["BASE"]["SAVE_ROOT_CODE"] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/code"
    os.makedirs(CONFIG["BASE"]["SAVE_ROOT_CKPT"], exist_ok=True)
    os.makedirs(CONFIG["BASE"]["SAVE_ROOT_IMGS"], exist_ok=True)
    os.makedirs(CONFIG["BASE"]["SAVE_ROOT_CODE"], exist_ok=True)


def get_all_images(dataset_root_list):
    image_paths = []

    for dataset_root in dataset_root_list:
        image_paths += glob.glob(f"{dataset_root}/*.*g")
        for root, dirs, _ in os.walk(dataset_root):
            for dir in dirs:
                image_paths += glob.glob(f"{root}/{dir}/*.*g")

    return sorted(image_paths)


def get_all_features(dataset_root_list):
    feature_paths = []

    for dataset_root in dataset_root_list:
        feature_paths += glob.glob(f"{dataset_root}/*.npy")
        for root, dirs, _ in os.walk(dataset_root):
            for dir in dirs:
                feature_paths += glob.glob(f"{root}/{dir}/*.npy")

    return sorted(feature_paths)


def split_dataset(dataset_path_list, val_size):
    train_list = []
    valid_list = []

    for dataset_path in dataset_path_list:
        video_name = dataset_path.split("/")[-3]
        file_name = int(dataset_path.split("/")[-1].split(".")[0])
        if "id999" not in dataset_path:
            if file_name < len(dataset_path_list) - val_size:
                train_list.append(dataset_path)
            else:
                valid_list.append(dataset_path)
        else:
            train_list.append(dataset_path)
    return train_list, valid_list


def delete_given_name(dataset_path_list):
    for dataset_root in dataset_path_list:
        feature_paths += glob.glob(f"{dataset_root}/*.npy")
        for root, dirs, _ in os.walk(dataset_root):
            for dir in dirs:
                feature_paths += glob.glob(f"{root}/{dir}/*.npy")

    return sorted(feature_paths)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


# def update_net(model, optimizer, loss, use_mGPU=False):
#     optimizer.zero_grad()
#     loss.backward()
#     if use_mGPU:
#         size = float(torch.distributed.get_world_size())
#         for param in model.parameters():
#             if param.grad == None:
#                 continue
#             torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
#             param.grad.data /= size
#     optimizer.step()

# def setup_ddp(gpu, ngpus_per_node, PORT):
#     torch.distributed.init_process_group(
#             backend='nccl',
#             init_method=f'tcp://127.0.0.1:{PORT}',
#             world_size=ngpus_per_node,
#             rank=gpu)


def make_grid_image(images_list):
    grid_rows = []

    for images in images_list:
        # images = images[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(images, nrow=images.shape[0]) * 0.5 + 0.5
        grid_rows.append(grid_row)
    if grid_rows[0].shape[1] > grid_rows[0].shape[2]:
        grid = torch.cat(grid_rows, dim=2)
    else:
        grid = torch.cat(grid_rows, dim=1)
    grid = grid.detach().cpu().numpy().transpose([1, 2, 0])[:, :, ::-1] * 255
    return grid
    # return Image.fromarray(grid.astype(np.uint8))


def stack_image_grid(batch_data_items: list, target_image: list):
    column = []
    for item in batch_data_items:
        column.append(item)
    target_image.append(torch.cat(column, dim=-2))


def get_convexhull_mask(faces, lmks, or_mask=None):
    faces = faces.clone().detach().cpu().numpy()
    lmks = lmks.clone().detach().cpu().numpy()
    masks = []
    for i in range(faces.shape[0]):
        face = faces[i].transpose(1, 2, 0)
        lmk = lmks[i]
        kernel = np.ones((3, 3), np.uint8)
        skin_canvas = np.zeros_like(face.copy()).astype(np.uint8)
        nose_canvas = np.zeros_like(face.copy()).astype(np.uint8)
        skin_points = np.array(
            [
                lmk[2],
                lmk[3],
                lmk[4],
                lmk[5],
                lmk[6],
                lmk[7],
                lmk[8],
                lmk[9],
                lmk[10],
                lmk[11],
                lmk[12],
                lmk[13],
                lmk[14],
                (lmk[35] + lmk[47]) / 2,
                lmk[35],
                lmk[33],
                lmk[31],
                (lmk[40] + lmk[31]) / 2,
            ],
            np.int32,
        )
        nose_points = cv2.convexHull(
            np.array(
                [
                    lmk[31] + [0, -10],
                    lmk[33] + [0, -10],
                    lmk[35] + [0, -10],
                    lmk[27],
                    lmk[30],
                ],
                np.int32,
            )
        )
        skin_mask = cv2.fillPoly(skin_canvas, [skin_points], (1, 1, 1))
        nose_mask = cv2.fillConvexPoly(nose_canvas, points=nose_points, color=(1, 1, 1))
        dilation_skin_mask = cv2.dilate(skin_mask, kernel, iterations=5)
        dilation_nose_mask = cv2.dilate(nose_mask, kernel, iterations=8)
        dilation_mask = dilation_skin_mask & (dilation_nose_mask * (-1) + 1)
        # dilation_mask = cv2.blur(dilation_mask*127.5+127.5, (5, 5))
        # dilation_mask = dilation_mask/127.5-1
        masks.append(
            np.expand_dims(dilation_mask.transpose(2, 0, 1).astype(np.float32), axis=0)
        )
    masks = np.concatenate(masks, axis=0).astype(np.int32)
    return torch.from_numpy(masks).cuda()


def blur_mask(mask):
    faces = mask.clone().detach().cpu().numpy()
    masks = []
    for i in range(faces.shape[0]):
        face = faces[i].transpose(1, 2, 0).astype(np.float32)
        dilation_mask = cv2.blur(face * 127.5 + 127.5, (5, 5))
        dilation_mask = dilation_mask / 127.5 - 1
        masks.append(
            np.expand_dims(dilation_mask.transpose(2, 0, 1).astype(np.float32), axis=0)
        )
    masks = np.concatenate(masks, axis=0)
    return torch.from_numpy(masks).cuda()


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    return model


def draw_landmarks(face, lmks, color=(255, 0, 0), size=2):
    face_ = face.copy()
    indexes = [27, 28, 29, 30, 31, 33, 35, 49, 51, 53, 61, 63, 67, 65, 55, 57, 59]
    # lmks = np.concatenate([lmks[:17], lmks[27:]], axis=0)
    lmks = np.concatenate([lmks[:17], lmks[indexes]], axis=0)
    for lmk in lmks:
        # cv2.putText(face_, str(i), (int(lmk[0])-10, int(lmk[1])-10), font, 0.5, color, 2)
        cv2.circle(face_, (int(lmk[0]), int(lmk[1])), 1, color, size)
    return face_


def draw_landmarks_batch(lmks, color=(1, 1, 1), size=2):
    lmks = lmks.clone().detach().cpu().numpy()

    guide_imgs = []
    for i in range(lmks.shape[0]):
        face = np.zeros((256, 256))
        lmk = lmks[i]

        lmk_image = np.repeat(
            draw_landmarks(face, lmk, color=color, size=size)[:, :, np.newaxis], 3, -1
        )
        guide_imgs.append(np.expand_dims(lmk_image.transpose(2, 0, 1), axis=0))
    guide_imgs = np.concatenate(guide_imgs, axis=0).astype(np.float32)
    return torch.from_numpy(guide_imgs).cuda()
