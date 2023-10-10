import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torchvision import transforms


def landmark_smoothing(landmarks):
    sm_landmarks = gaussian_filter1d(landmarks, sigma=1.0, axis=0)
    sm_landmarks = np.reshape(sm_landmarks, (-1, 68, 2))
    return sm_landmarks


def get_lmk_imgs(
    batch_lmk, color=(1, 1, 1), size=2, image_size=256, types="full", device="cuda"
):
    lmk_imgs = []
    for lmk in batch_lmk:
        canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        if types == "full":
            for lmk_ in lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        else:
            for lmk_ in lmk[:17]:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

            indexes = [
                27,
                28,
                29,
                30,
                31,
                33,
                35,
                49,
                51,
                53,
                61,
                63,
                67,
                65,
                55,
                57,
                59,
            ]
            _lmk = lmk[indexes, :]
            for lmk_ in _lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        lmk_imgs.append(np.expand_dims(canvas.transpose(2, 0, 1), axis=0))
    lmk_imgs = np.concatenate(lmk_imgs, axis=0)
    return torch.from_numpy(lmk_imgs).to(device)


def transfer_lip_params(s_flame_params, d_flame_params):
    lipysnc_flame_params = {}
    for key in s_flame_params.keys():
        lipysnc_flame_params[key] = s_flame_params[key].clone()

    lipysnc_flame_params["pose"][0][3] = d_flame_params["pose"][0][3]
    # lipysnc_flame_params["pose"][0][4] = d_flame_params["pose"][0][4]
    # lipysnc_flame_params["pose"][0][5] = d_flame_params["pose"][0][5]
    lipysnc_flame_params["exp"] = d_flame_params["exp"]
    # import pdb;pdb.set_trace()
    return lipysnc_flame_params




def get_convexhull_mask(
    batch_lmk, skin_dilate_iter=5, nose_dilate_iter=8, image_size=256, device="cuda"
):
    masks = []
    for lmk in batch_lmk:
        kernel = np.ones((3, 3), np.uint8)
        skin_canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        skin_points = np.array(
            [
                # lmk[1],
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
                # lmk[15],
                (lmk[35] + lmk[47]) / 2,
                lmk[35],
                lmk[33],
                lmk[31],
                (lmk[40] + lmk[31]) / 2,
            ],
            np.int32,
        )
        skin_mask = cv2.fillPoly(skin_canvas, [skin_points], (1, 1, 1))
        dilation_skin_mask = cv2.dilate(skin_mask, kernel, iterations=skin_dilate_iter)

        if nose_dilate_iter:
            nose_canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
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
            nose_mask = cv2.fillConvexPoly(
                nose_canvas, points=nose_points, color=(1, 1, 1)
            )

            dilation_nose_mask = cv2.dilate(
                nose_mask, kernel, iterations=nose_dilate_iter
            )
            dilation_mask = dilation_skin_mask & (dilation_nose_mask * (-1) + 1)
        else:
            dilation_mask = dilation_skin_mask

        masks.append(np.expand_dims(dilation_mask.transpose(2, 0, 1), axis=0))
    masks = np.concatenate(masks, axis=0)
    return torch.from_numpy(masks).to(device)

# def get_convexhull_mask(batch_lmk, dilate_iter=5, image_size=256, device="cuda"):
#     masks = []
#     for lmk in batch_lmk:
#         kernel = np.ones((3, 3), np.uint8)
#         canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
#         points = np.array(lmk[1:16], np.int32)
#         skin_mask = cv2.fillConvexPoly(canvas, points=points, color=(1, 1, 1))
#         dilation_skin_mask = cv2.dilate(skin_mask, kernel, iterations=dilate_iter)
#         masks.append(np.expand_dims(dilation_skin_mask.transpose(2, 0, 1), axis=0))
#     masks = np.concatenate(masks, axis=0)
#     return torch.from_numpy(masks).to(device)


def get_batch_image_from_path(opts, image_path_list):
    tf_color = transforms.Compose(
        [
            transforms.Resize((opts.image_size, opts.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    batch_image = []
    for image_path in image_path_list:
        img = Image.open(image_path).convert("RGB")
        img = tf_color(img)
        batch_image.append(img.unsqueeze(0))
    _batch_image = torch.cat(batch_image, dim=0)
    return _batch_image


def get_lmk_imgs(
    batch_lmk, color=(1, 1, 1), size=2, image_size=256, types="full", device="cuda"
):
    lmk_imgs = []
    for lmk in batch_lmk:
        canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        if types == "full":
            for lmk_ in lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        else:
            for lmk_ in lmk[:17]:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

            indexes = [
                27,
                28,
                29,
                30,
                31,
                33,
                35,
                49,
                51,
                53,
                61,
                63,
                67,
                65,
                55,
                57,
                59,
            ]
            _lmk = lmk[indexes, :]
            for lmk_ in _lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        lmk_imgs.append(np.expand_dims(canvas.transpose(2, 0, 1), axis=0))
    lmk_imgs = np.concatenate(lmk_imgs, axis=0)
    return torch.from_numpy(lmk_imgs).to(device)


def get_deca_lmks(DC, flame_params):
    trans_landmarks2d_list = []
    for flame_param in flame_params:
        _, _, _, _, trans_landmarks2d, _, _, _ = DC.get_lmks_from_params(
            flame_param, tform_invs=flame_param["tform_inv"]
        )

        trans_landmarks2d_list.append(trans_landmarks2d.clone().detach().cpu().numpy())
    trans_landmark2d_np = np.concatenate(trans_landmarks2d_list, axis=0)
    return trans_landmark2d_np
