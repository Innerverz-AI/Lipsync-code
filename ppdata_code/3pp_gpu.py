import os, glob, cv2, torch, time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from innerverz import DECA

DC = DECA(device='cuda')

def get_batch_dataloader(img_paths, lmk_paths, dict_paths, batch_size, num_worker, img_size=None):
    dataset = SimpleDataset(img_paths, lmk_paths, dict_paths, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    dataloader_iter = iter(dataloader)
    return dataloader_iter

class SimpleDataset(Dataset):
    def __init__(self, image_paths, lmk_paths, dict_paths, img_size=None):
        self.img_size = img_size
        self.image_paths = image_paths
        self.lmk_paths = lmk_paths
        self.dict_paths = dict_paths

    def __getitem__(self,idx):
        # path setting
        image_path  = self.image_paths[idx]
        lmk_path = self.lmk_paths[idx]
        dict_path = self.dict_paths[idx]
        fidx = int(image_path.split('/')[-1].split('.')[0])
        
        # load data
        image = cv2.imread(image_path)
        insight_lmk = torch.tensor(np.load(lmk_path, allow_pickle=True))
        image_dict = np.load(dict_path, allow_pickle=True).item()
        
        input_image = torch.tensor(image_dict['image']).squeeze().float()
        tform_inv = torch.tensor(image_dict['tform_inv']).squeeze().float()
        
        return image, input_image, insight_lmk, tform_inv, image_path

    def __len__(self):
        return len(self.image_paths)

root_path = "/ssd2t/DATASET/VoxCeleb2/demo_driving_video_100iter/512over"
video_paths = sorted(glob.glob(os.path.join(root_path, '*/*/*.mp4')))

# TODO: use argparser to controll multi-GPU 
batch_size = 16
num_worker = 16
num_iter = 10
for i in tqdm(range(len(video_paths))):
    video_path = video_paths[i]
    save_dir = video_path.replace('.mp4', '')
    
    save_dir_flame_params = os.path.join(save_dir, 'flame_params')
    os.makedirs(save_dir_flame_params, exist_ok=True)
    save_dir_2d_lmks = os.path.join(save_dir, '2d_lmks')
    os.makedirs(save_dir_2d_lmks, exist_ok=True)
    
    img_paths = sorted(glob.glob(f'{save_dir}/aligned_imgs/*.*'))
    lmk_paths = sorted(glob.glob(f'{save_dir}/insight_lmks/*.*'))
    dict_paths = sorted(glob.glob(f'{save_dir}/tmp_dicts/*.*'))
    
    # slicing data
    # if len(img_paths) > 100:
    #     img_paths = img_paths[:100]
    # if len(lmk_paths) > 100:
    #     lmk_paths = lmk_paths[:100]
    # if len(dict_paths) > 100:
    #     dict_paths = dict_paths[:100]
    
    assert len(img_paths)==len(dict_paths) and len(img_paths)==len(lmk_paths), 'img_length is different with dict_length'
    dataloader = get_batch_dataloader(img_paths, lmk_paths, dict_paths, batch_size=batch_size, img_size=512, num_worker=num_worker)
    for batch_idx, batch in enumerate(dataloader):

        images, input_images, insight_lmks, tform_invs, image_paths = batch
        code_dict = DC.encode(input_images.cuda())
        
        # fail to detect face
        if 0 in insight_lmks:
            f = open(f'no_lmks_videos.txt', 'a')
            f.write(f'{save_dir}\n')
            f.close()
            break
        
        # setting data
        image_dict = {}
        image_dict['image'] = input_images.cuda()
        image_dict['tform_inv'] = tform_invs.cuda()
        
        code_dict['shape'] = code_dict['shape'].cuda()
        code_dict['tex'] = code_dict['tex'].cuda()
        code_dict['exp'] = code_dict['exp'].cuda()
        code_dict['pose'] = code_dict['pose'].cuda()
        code_dict['cam'] = code_dict['cam'].cuda()
        code_dict['light'] = code_dict['light'].cuda()
        
        insight_lmks = insight_lmks.cuda()
        
        # z calibration
        _, _, _, _, trans_landmarks2ds, trans_landmarks3ds, trans_landmarks_756_3ds, _ = DC.get_lmks_from_params(code_dict, tform_invs=image_dict['tform_inv'])
        z_calibration_t = (insight_lmks[:, :, 2]-trans_landmarks3ds[:, :, 2]).mean(1, keepdim=True)
        z_calibration_s = torch.ones_like(z_calibration_t)
        code_dict['z_calibration'] = torch.cat([z_calibration_t, z_calibration_s], dim=1)
        
        # optimization
        optimized_code_dict = DC.optimize_per_batch(code_dict, insight_lmks, image_dict['tform_inv'], num_iter=num_iter, lmk_type='3D')
        _, _, _, _, trans_landmarks2ds, trans_landmarks3ds, trans_landmarks_756_3ds, _ = DC.get_lmks_from_params(optimized_code_dict, tform_invs=image_dict['tform_inv'])
        
        for tform_inv, shape, tex, exp, pose, cam, light, insight_lmk, trans_landmarks2d, image_path in \
            zip(image_dict['tform_inv'], optimized_code_dict['shape'], optimized_code_dict['tex'], optimized_code_dict['exp'], optimized_code_dict['pose'], optimized_code_dict['cam'], optimized_code_dict['light'], insight_lmks, trans_landmarks2ds, image_paths):
            
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