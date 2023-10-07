### for demo video
### make correct directory tree
import cv2, os, glob
from tqdm import tqdm


dataset_base_dir2 = "/ssd2t/DATASET/VoxCeleb2/demo_driving_video/512over"
video_paths = sorted(glob.glob(os.path.join(dataset_base_dir2, "*.mp4")))  # 20

i = 0
for video_path in video_paths:
    video_name = video_path.split('/')[-1].split('.')[0]
    video_id = 'id'+str(99946 - i)
    video_url = video_name.lower()
    print(video_id, video_url, video_name)
    save_path = os.path.join(os.path.dirname(video_path), video_id, video_url)
    os.makedirs(save_path, exist_ok=True)
    os.system(f'mv {video_path} {save_path}')
    i += 1
    