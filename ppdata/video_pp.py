import argparse
import glob
import multiprocessing
import os
import subprocess
import sys

import cv2
import numpy as np
import parmap
import torch
from innerverz import DECA, FaceAligner
from pp_utils import dict_setting, get_batch_dataloader

sys.path.append('../')
from utils import Sync_tool
from utils.audio import get_mel

FA_3D = FaceAligner(size=512, lmk_type='3D')
DC = DECA(device='cpu')
DC_CUDA = DECA(device='cuda')
synctool = Sync_tool()


def get_video_frames(inputs):
    video_name, video_path, save_path = inputs
    frame_save_path = os.path.join(save_path, 'frames')
    mel_save_path = os.path.join(save_path, 'mel')
    os.makedirs(frame_save_path, exist_ok=True)
    os.makedirs(mel_save_path, exist_ok=True)
    
    # frames
    command = f'ffmpeg -y -i {video_path} -f image2 -vb 20M -start_number 0 {os.path.join(frame_save_path, "%06d.png")}'
    subprocess.call(command, shell=True, stdout=None)
    
    # audio - wav
    command = f'ffmpeg -y -i {video_path} -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(save_path, f"{video_name}.wav")}'
    subprocess.call(command, shell=True, stdout=None)
    
    # audio - mel
    mel = get_mel(video_path)
    np.save(os.path.join(mel_save_path, '000000.npy'), mel)
    return
    

def get_faces(inputs):
    video_name, frame_path, save_path = inputs
    frame_name = os.path.basename(frame_path)
    
    image = cv2.imread(frame_path)
    face_dict = FA_3D.get_face(image)
    if not face_dict['facebool']: 
        f = open('no_det_videos.txt', 'a')
        f.write(f'{video_path}, {frame_name}\n') # video path vs frame path
        f.close()
        return
    aligned_img = face_dict['aligned_face']
    aligned_lmks = FA_3D.detect_lmk(aligned_img)['lmks_68']
    
    image_dict = DC.data_preprocess(aligned_img, aligned_lmks)
    
    save_dict = {}
    for att in ['image','tform_inv']:
        save_dict[att] = image_dict[att].cpu().numpy().copy()
    
    save_path_aligned_img = os.path.join(save_path, 'aligned_imgs', frame_name)
    cv2.imwrite(save_path_aligned_img, aligned_img)
    save_path_2d_lmks = os.path.splitext(os.path.join(save_path, 'insight_lmks', frame_name))[0]+'.npy'
    np.save(save_path_2d_lmks, aligned_lmks)
    save_path_flame_params = os.path.splitext(os.path.join(save_path, 'tmp_dicts', frame_name))[0]+'.npy'
    np.save(save_path_flame_params, save_dict)
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_num', type=int, default=99000)
    # start id num / fps / sr / save path
    parser.add_argument('--video_folder_path', type=str, default='../assets/dataset_videos')
    parser.add_argument('--synced_video_folder_path', type=str, default='./assets/sync_dataset_videos')
    parser.add_argument('--save_folder_path', type=str, default='./assets/dataset_videos_pp')
    parser.add_argument('--list', nargs='+', default=(10,12), type=int, help='the indices of transparent classes')
    parser.add_argument('--gen_sync_video', default=True, type=bool)
    
    # GPU process options
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--num_iter', type=int, default=10)
    
    
    args = parser.parse_args()
    
    # sync video & 
    save_list = []
    video_path_list = sorted(glob.glob(os.path.join(args.video_folder_path,'*.*')))
    for i, video_file_path in enumerate(video_path_list):
        id_name = f'id{str(args.id_num - i).zfill(5)}'
        video_file = os.path.basename(video_file_path)

        if args.gen_sync_video:
            os.makedirs(args.synced_video_folder_path, exist_ok=True)
            args.video_folder_path = args.synced_video_folder_path
            synced_video_path = os.path.join(args.video_folder_path, video_file)
            synctool.forward(video_file_path, synced_video_path)
        save_list.append([id_name, video_file])
    
    for id_name, video_file in save_list:
        video_name = os.path.basename(video_file).split('.')[0]
        video_path = os.path.join(args.video_folder_path, video_file)
        save_path = os.path.join(args.save_folder_path, id_name, video_name)#, 'frames')

        get_video_frames([video_name, video_path, save_path])
        
    for id_name, video_file in save_list:
        video_name = os.path.basename(video_file).split('.')[0]
        video_path = os.path.join(args.video_folder_path, video_file)
        frame_paths = sorted(glob.glob(os.path.join(args.save_folder_path, id_name, video_name, 'frames/*.*')))
        save_path = os.path.join(args.save_folder_path, id_name, video_name)#, 'frames')
        face_save_path = os.path.join(save_path, 'aligned_imgs')
        lmk_save_path = os.path.join(save_path, 'insight_lmks')
        tmp_save_path = os.path.join(save_path, 'tmp_dicts')
        os.makedirs(face_save_path, exist_ok=True)
        os.makedirs(lmk_save_path, exist_ok=True)
        os.makedirs(tmp_save_path, exist_ok=True)
        
        for frame_path in frame_paths:
            get_faces([video_name, frame_path, save_path])
            
            
    # deca lmk(GPU process)
    for id_name, video_file in save_list:
        video_name = os.path.basename(video_file).split('.')[0]
        save_dir = os.path.join(args.save_folder_path, id_name, video_name)
        save_dir_flame_params = os.path.join(save_dir, 'flame_params')
        os.makedirs(save_dir_flame_params, exist_ok=True)
        save_dir_2d_lmks = os.path.join(save_dir, '2d_lmks')
        os.makedirs(save_dir_2d_lmks, exist_ok=True)
        
        img_paths = sorted(glob.glob(f'{save_dir}/aligned_imgs/*.*'))
        lmk_paths = sorted(glob.glob(f'{save_dir}/insight_lmks/*.*'))
        dict_paths = sorted(glob.glob(f'{save_dir}/tmp_dicts/*.*'))
        
        
        assert len(img_paths)==len(dict_paths) and len(img_paths)==len(lmk_paths), 'img_length is different with dict_length'
        dataloader = get_batch_dataloader(img_paths, lmk_paths, dict_paths, batch_size=args.batch_size, img_size=512, num_worker=args.num_worker)
        for batch_idx, batch in enumerate(dataloader):
            images, input_images, insight_lmks, tform_invs, image_paths = batch
            code_dict = DC_CUDA.encode(input_images.cuda())
            
            # fail to detect face
            if 0 in insight_lmks:
                f = open(f'no_lmks_videos.txt', 'a')
                f.write(f'{save_dir}\n')
                f.close()
                break
            
            new_image_dict, new_code_dict = dict_setting(code_dict, input_images, tform_invs)
            
            insight_lmks = insight_lmks.cuda()
            
            # z calibration
            _, _, _, _, trans_landmarks2ds, trans_landmarks3ds, trans_landmarks_756_3ds, _ = DC_CUDA.get_lmks_from_params(code_dict, tform_invs=new_image_dict['tform_inv'])
            
            z_calibration_t = (insight_lmks[:, :, 2]-trans_landmarks3ds[:, :, 2]).mean(1, keepdim=True)
            z_calibration_s = torch.ones_like(z_calibration_t)
            code_dict['z_calibration'] = torch.cat([z_calibration_t, z_calibration_s], dim=1)
            
            # optimization
            optimized_code_dict = DC_CUDA.optimize_per_batch(code_dict, insight_lmks, new_image_dict['tform_inv'], num_iter=args.num_iter, lmk_type='3D')
            _, _, _, _, trans_landmarks2ds, trans_landmarks3ds, trans_landmarks_756_3ds, _ = DC_CUDA.get_lmks_from_params(optimized_code_dict, tform_invs=new_image_dict['tform_inv'])
            
            for tform_inv, shape, tex, exp, pose, cam, light, insight_lmk, trans_landmarks2d, image_path in \
                zip(new_image_dict['tform_inv'], optimized_code_dict['shape'], optimized_code_dict['tex'], optimized_code_dict['exp'], optimized_code_dict['pose'], optimized_code_dict['cam'], optimized_code_dict['light'], insight_lmks, trans_landmarks2ds, image_paths):
                
                opt_code_dict = {}
                opt_code_dict['shape'] = shape.detach().cpu().numpy()
                opt_code_dict['tex'] = tex.detach().cpu().numpy()
                opt_code_dict['exp'] = exp.detach().cpu().numpy()
                opt_code_dict['pose'] = pose.detach().cpu().numpy()
                opt_code_dict['cam'] = cam.detach().cpu().numpy()
                opt_code_dict['light'] = light.detach().cpu().numpy()
                opt_code_dict['tform_inv'] = tform_inv.cpu().numpy()
                opt_code_dict['aligned_lmk_3D'] = insight_lmk.cpu().numpy()

                save_path_flame_params = os.path.splitext(os.path.join(save_dir_flame_params, os.path.basename(image_path)))[0]+'.npy'
                np.save(save_path_flame_params, opt_code_dict)
                save_path_2d_lmks = os.path.splitext(os.path.join(save_dir_2d_lmks, os.path.basename(image_path)))[0]+'.npy'
                np.save(save_path_2d_lmks, trans_landmarks2d.detach().cpu().numpy())
            print(f'{batch_idx}/{len(dataloader)}')