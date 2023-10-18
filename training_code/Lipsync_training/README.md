# Lipsync_training

## Dataset setting
We use **VoxCeleb2** dataset

VoxCeleb2 video dataset is stored in the following paths

    /data1/PUBLIC/Voxceleb2

VoxCeleb2 preprocessing dataset is stored in the following paths

    /data1/LipSync-dataset/Voxceleb2


'aligned_imgs', 'mel', 'flame_params', '2d_lmks_smooth' folders are should be contained in preprocessing dataset folder

It is recommended that you place the dataset in your local folder.


After placing the dataset, fix config file DATASET option.

## Assets
Assets can be downloaded from here

https://drive.google.com/drive/folders/1LTExg6NZBzsvtgvpqI3THQBclIE7nijw?usp=sharing


Syncnet model checkpoints in the following path

    SyncNet/ckpt


General Lipsync model checkpoints in the following path

    ckpt

## How to Run
For example, below code is for training with 2gpu(gpu 0,1)

details in here : https://github.com/Innerverz-AI/Training_Template_GANs

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --gpu_ids=0,1 --main_process_port=3457 scripts/train.py 