import argparse
import os
import random
from glob import glob
from os.path import basename, dirname, isfile, join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import audio
import wandb
from hparams import hparams
from innerverz import FaceAligner
from models import SyncNet_color as SyncNet
from PIL import Image
from torch import nn, optim
from torch.utils import data as data_utils
from tqdm import tqdm

FA_3D = FaceAligner(size=256, lmk_type="3D")

run = wandb.init(
    # Set the project where this run will be logged
    project="SyncNet - Wav2Lip",
    # Track hyperparameters and run metadata
    config={
        "dataset": "Voxceleb2",
    },
    tags={"bbox_face", "mel"},
)
parser = argparse.ArgumentParser(description="Code to train the expert lip-sync discriminator")

parser.add_argument(
    "--train_data_root",
    help="Root folder of the preprocessed LRS2 dataset",
    default="/ssd2t/DATASET/VoxCeleb2/1url_1video/512over",
)
parser.add_argument(
    "--test_data_root",
    help="Root folder of the preprocessed LRS2 dataset",
    default="/ssd2t/DATASET/VoxCeleb2/1url_1video/512over",
)

parser.add_argument(
    "--checkpoint_dir",
    help="Save checkpoints to this directory",
    default="./train_results_add_dataset_demo_Oct6",
    type=str,
)
parser.add_argument(
    "--checkpoint_path", help="Resumed from this checkpoint", default=None, type=str
)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(object):
    def __init__(self, data_root, type="train"):
        self.data_root = data_root
        self.all_videos = self.get_image_list(data_root, type)

    def get_image_list(self, data_root, type):
        face_folder_list, errors = [], ""
        count = 0

        if type == "train":
            id_names = sorted(os.listdir(data_root))[5:]
            for id_name in tqdm(id_names):
                url_ids = sorted(os.listdir(os.path.join(data_root, id_name)))
                for url_id in url_ids:
                    trim_list = sorted(os.listdir(os.path.join(data_root, id_name, url_id)))
                    for trim in trim_list:
                        # try:
                        frames_folder_path = os.path.join(data_root, id_name, url_id, trim)
                        face_path_list = sorted(
                            glob(os.path.join(frames_folder_path, "aligned_imgs/*.*"))
                        )

                        frames_num = len(face_path_list)
                        for face_path in face_path_list:
                            face_folder_list.append([face_path, frames_num])

                        count += frames_num
        else:
            # WARNING
            id_names = sorted(os.listdir(data_root))[:5]

            for id_name in tqdm(id_names):
                url_ids = sorted(os.listdir(os.path.join(data_root, id_name)))
                for url_id in url_ids:
                    trim_list = sorted(os.listdir(os.path.join(data_root, id_name, url_id)))
                    for trim in trim_list:
                        # try:
                        frames_folder_path = os.path.join(data_root, id_name, url_id, trim)
                        face_path_list = sorted(
                            glob(os.path.join(frames_folder_path, "aligned_imgs/*.*"))
                        )

                        frames_num = len(face_path_list)
                        for face_path in face_path_list:
                            face_folder_list.append([face_path, frames_num])

                        count += 1

        return face_folder_list

    def get_frame_id(self, frame):
        return int(basename(frame).split(".")[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, "{}.jpg".format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx:end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            try:
                idx = random.randint(0, len(self.all_videos) - 1)
                face_path, image_amount = self.all_videos[idx]
                folder_path = face_path.split("/")[:-2]
                folder_path = "/".join(folder_path)
                # audio
                # 1s 25fps / 0.2 16 -> 1 80
                mel_path = os.path.join(folder_path, "mel", "000000.npy")
                mel = np.load(mel_path)

                # check min duration
                video_duration = image_amount / 25
                audio_duration = mel.shape[0] / 80
                min_duration = min(video_duration, audio_duration)

                image_amount = int(min_duration * 25)

                image_num = random.choice(range(image_amount - 10))

                start_mel_id = int(image_num / 25 * 80)
                _mel = mel[start_mel_id : start_mel_id + 16]  # 1 80 16 / 1 80 0
                _mel = torch.FloatTensor(_mel.T).unsqueeze(0)
                assert _mel.shape[-1] == 16

                # frames

                if random.choice([True, False]):
                    y = torch.ones(1).float()
                    chosen_image_num = image_num
                else:
                    wrong_image_num = random.choice(range(image_amount - 6))
                    while wrong_image_num == image_num:
                        wrong_image_num = random.choice(range(image_amount - 6))

                    y = torch.zeros(1).float()
                    chosen_image_num = wrong_image_num

                image_list = []
                for i in range(5):
                    chosen_image_file = str(chosen_image_num + i).zfill(6) + ".png"
                    file_path = os.path.join(folder_path, "aligned_imgs", chosen_image_file)
                    face = cv2.imread(file_path)
                    face = cv2.resize(face, (256, 256))
                    face = face[:, 128 - 64 : 128 + 64]
                    _face = cv2.resize(face, (hparams.img_size, hparams.img_size))
                    image_list.append(_face)

                x = np.concatenate(image_list, axis=2) / 255.0
                x = x.transpose(2, 0, 1)
                x = x[:, x.shape[1] // 2 :]
                x = torch.FloatTensor(x)

                # if _mel.shape[-1] == 0:
                # print(image_amount, image_name, start_mel_id, mel.shape, _mel.shape)
                return x, _mel, y
            except:
                continue


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(
    device,
    model,
    train_data_loader,
    test_data_loader,
    optimizer,
    checkpoint_dir=None,
    checkpoint_interval=None,
    nepochs=None,
):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()
            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)

            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description("Loss: {}".format(running_loss / (step + 1)))

            if global_step % 1000 == 0:
                batch_images = (
                    x[0]
                    .reshape(5, 3, 64, 128)
                    .contiguous()
                    .clone()
                    .detach()
                    .permute([0, 2, 3, 1])
                    .cpu()
                    .numpy()[:, :, :, ::-1]
                    * 255
                )
                image_list = []
                for i, images in enumerate(batch_images):
                    _images = np.array(images, dtype=np.uint8)
                    _images = Image.fromarray(_images)
                    _images = wandb.Image(_images, caption=f"{i} image")
                    image_list.append(_images)

                wandb.log({"examples": image_list})

            if step % 10 == 0:
                wandb.log({"train_loss": round(running_loss / (step + 1), 5)})
        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print("Evaluating for {} steps".format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps:
                break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)
        wandb.log({"validation_loss": round(averaged_loss, 5)})

        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        },
        checkpoint_path,
    )
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset(args.train_data_root, "train")
    test_dataset = Dataset(args.test_data_root, "valid")

    train_data_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size, num_workers=8
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print(
        "total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=hparams.syncnet_lr
    )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(
        device,
        model,
        train_data_loader,
        test_data_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs,
    )
