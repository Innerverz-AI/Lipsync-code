import os, glob, cv2, torch, time
import numpy as np
import parmap
import multiprocessing
import argparse
from tqdm import tqdm
from innerverz import FaceAligner, DECA

FA_3D = FaceAligner(size=512, lmk_type='3D')
DC = DECA(device='cpu')

def save_aligned_image(image_path, args):
    save_dir = video_path.replace('.mp4', '')
    image = cv2.imread(image_path)
    
    face_dict = FA_3D.get_face(image)
    # fail to detect face
    if 'aligned_face' not in face_dict.keys():
        f = open('no_det_videos.txt', 'a')
        f.write(f'{video_path}\n')
        f.close()
    # save data
    else:
        aligned_img = face_dict['aligned_face']
        aligned_lmks = FA_3D.detect_lmk(aligned_img)['lmks_68']
        image_dict = DC.data_preprocess(aligned_img, aligned_lmks)
        
        save_dict = {}
        for k in ['image','tform_inv']:
            save_dict[k] = image_dict[k].cpu().numpy().copy()
        
        save_path_aligned_img = os.path.join(save_dir, 'aligned_imgs', os.path.basename(image_path))
        cv2.imwrite(save_path_aligned_img, aligned_img)
        save_path_2d_lmks = os.path.splitext(os.path.join(save_dir, 'insight_lmks', os.path.basename(image_path)))[0]+'.npy'
        np.save(save_path_2d_lmks, aligned_lmks)
        save_path_flame_params = os.path.splitext(os.path.join(save_dir, 'tmp_dicts', os.path.basename(image_path)))[0]+'.npy'
        np.save(save_path_flame_params, save_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/ssd2t/DATASET/VoxCeleb2/time_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2t/DATASET/VoxCeleb2/time_test')
    
    args = parser.parse_args()
    
    video_paths = sorted(glob.glob(os.path.join(args.dataset_dir, '*/*/*.mp4')))

    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        save_dir = video_path.replace(args.dataset_dir, args.save_dir).replace('.mp4', '')
        
        save_dir_aligned_img = os.path.join(save_dir, 'aligned_imgs')
        os.makedirs(save_dir_aligned_img, exist_ok=True)
        save_dir_aligned_img = os.path.join(save_dir, 'insight_lmks')
        os.makedirs(save_dir_aligned_img, exist_ok=True)
        save_dir_aligned_img = os.path.join(save_dir, 'tmp_dicts')
        os.makedirs(save_dir_aligned_img, exist_ok=True)
        
        img_paths = sorted(glob.glob(f'{save_dir}/frames/*.*'))
        
        # # slicing img_paths
        # if len(img_paths) > 100:
        #     img_paths = img_paths[:100]
        num_cores = multiprocessing.cpu_count()
        parmap.map(save_aligned_image, img_paths, args, pm_pbar=True, pm_processes=int(num_cores * .55))
