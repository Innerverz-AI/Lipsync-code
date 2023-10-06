
import glob
import os

from utils import Sync_tool, get_audio_features

# # get audio features
# audio_path = './assets/GU_1.wav' # have to check file extension with '.wav'
# audio_features_dict = get_audio_features(audio_path)

# get synced videos

video_path = './assets/videos/GU_1.mp4'
sync_tool = Sync_tool()
os.makedirs('./assets/synced_videos/', exist_ok=True)
offset = sync_tool.forward(video_path, './assets/synced_videos/GU_1.mp4')
    
print("YES")
