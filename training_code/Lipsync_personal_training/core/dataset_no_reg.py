import os, torch, random
import numpy as np
from lib import utils
from lib.dataset import DatasetInterface
from torchvision import transforms

class MyDataset(DatasetInterface):
    def __init__(self, CONFIG, dataset_path_list):
        super(MyDataset, self).__init__(CONFIG)
        self.set_tf()
        self.frame_num_per_video = CONFIG['BASE']['FRAME_NUM_PER_VIDEO']
        self.same_prob = CONFIG['BASE']['SAME_PROB']
        self.forbidden_list = ['9MgNeIQ3dYE', 'XpeWSDxWDnE','jtYhopgMK14', 'BQtK9CX6L1Y', 'p5FGIF2LEvQ']
        self.opt_url = CONFIG['BASE']['OPT_URL']
        self.opt_gt_img_list = []
        for gt_img_path in dataset_path_list['gt_img_path_list']:
            if self.opt_url in gt_img_path:
                self.opt_gt_img_list.append(gt_img_path)
        for name, list in dataset_path_list.items() : self.__setattr__(name, list)
        
    def __getitem__(self, index):
        T = 5 # time step
        is_opt = random.randint(0,1)
        FLAG = False
        while not FLAG:
            try:
                index_t = random.randint(0, self.__len__()-1)
                # except selected videos
                gt_img_path = self.gt_img_path_list[index_t]
                if is_opt:
                    gt_img_path = random.choice(self.opt_bbox_list)
                video_url_name = gt_img_path.split('/')[-4]
                file_name = int(gt_img_path.split('/')[-1].split('.')[0])
                if video_url_name in self.forbidden_list or file_name+5 > self.frame_num_per_video:
                    continue
                
                # data path list with time step T
                video_frames_path_list = [path for path in self.gt_img_path_list if video_url_name in path]
                gt_img_paths, param_paths, ref_img_paths, lmk_paths = [], [], [], []
                hubert_path = os.path.join(os.path.dirname(gt_img_path.replace('aligned_imgs', 'mel')), '000000.npy')
                # ref_idx = random.randint(0,len(video_frames_path_list)-6)
                for i in range(T):
                    gt_img_paths.append(gt_img_path.replace('%06d'%file_name, '%06d'%(file_name+i)))
                    lmk_paths.append(os.path.splitext(gt_img_path.replace('aligned_imgs', '2d_lmks').replace('%06d'%file_name, '%06d'%(file_name+i)))[0] + '.npy')
                    param_paths.append(os.path.splitext(gt_img_path.replace('aligned_imgs', 'flame_params').replace('%06d'%file_name, '%06d'%(file_name+i)))[0] + '.npy')
                    ref_img_paths.append(random.choice(video_frames_path_list[:-4]))
                
                # init data list
                flame_params = {}
                gt_imgs, guide_imgs, ref_imgs, hubert_features, mel_features = [], [], [], [], []
                for key in np.load(param_paths[0], allow_pickle=True).item().keys():
                    flame_params[key] = []
                jaw_noise = random.uniform(0,0.2) if random.randint(0,1) else 0
                
                # put datas in data list
                hubert_all = torch.tensor(np.load(hubert_path, allow_pickle=True).transpose(1,0)).float()
                mel_features.append(hubert_all[:,int(file_name/25*80):int(file_name/25*80)+16].unsqueeze(0))
                if file_name == 0 or mel_features[0].shape[2] != 16:
                    continue
                
                for i in range(T):
                    lmk_image = np.repeat(utils.draw_landmarks(np.ones((256,256))*127.5, np.load(lmk_paths[i])/2, color=(255,255,255), size=2)[:,:,np.newaxis],3,-1)
                    lmk_image = Image.fromarray(lmk_image.astype(np.uint8)).convert("RGB")
                    guide_imgs.append(self.tf_color(lmk_image).unsqueeze(0))
                    gt_imgs.append(self.pp_image(gt_img_paths[i]).unsqueeze(0))
                    ref_imgs.append(self.pp_image(ref_img_paths[i]).unsqueeze(0))
                    start_id = int(file_name/25*80) # melmel
                    hubert_features.append(hubert_all[:,(start_id-1+i):(start_id-1+i)+16].unsqueeze(0)) # melmel
                    tmp_params = np.load(param_paths[i], allow_pickle=True).item()
                    for key in flame_params.keys():
                        flame_params[key].append(torch.from_numpy(tmp_params[key]).unsqueeze(0))
                    flame_params['pose'][i][:,3] = flame_params['pose'][i][:,3] + jaw_noise
                
                gt_imgs = torch.cat(gt_imgs, dim=0) # T, 3, H, W
                guide_imgs = torch.cat(guide_imgs, dim=0) # T, 3, H, W
                ref_imgs = torch.cat(ref_imgs, dim=0) # T, 3, H, W
                hubert_features = torch.cat(hubert_features, dim=0) # T, 80, 16
                mel_features = torch.cat(mel_features, dim=0) # 1, 80, 16
                for key in flame_params.keys():
                    flame_params[key] = torch.cat(flame_params[key], dim=0) # T, param_dim

                if hubert_features.shape[2] != 16:
                    continue
                
                FLAG = True
            except:
                continue
            
        return [gt_imgs, ref_imgs, flame_params, hubert_features, mel_features]

    def __len__(self):
        return len(self.gt_img_path_list)

    # override
    def set_tf(self):

        self.tf_gray = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.tf_color = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def divide_datasets(model, CONFIG):
    gt_img_path_list = sorted(utils.get_all_images(CONFIG['DATASET']['TRAIN_PATH']['GT_IMG'], CONFIG['BASE']['FRAME_NUM_PER_VIDEO']))
    train_gt_img_path_list, valid_gt_img_path_list = utils.split_dataset(gt_img_path_list, CONFIG['BASE']['FRAME_NUM_PER_VIDEO'], CONFIG['BASE']['VAL_SIZE'])
    
    model.train_dataset_dict = {
            'gt_img_path_list' : train_gt_img_path_list,
        }  
    model.valid_dataset_dict = {
            'gt_img_path_list' : valid_gt_img_path_list,
        }