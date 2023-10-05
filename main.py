
import glob
import os

from utils import Sync_tool, get_audio_features

# # get audio features
# audio_path = './assets/GU_1.wav' # have to check file extension with '.wav'
# audio_features_dict = get_audio_features(audio_path)

# get synced videos
video_path = ''
sync_tool = Sync_tool()
offset = sync_tool.forward(video_path)
    
print("YES")
