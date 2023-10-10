import glob, os, cv2, librosa, subprocess, torch
import numpy as np
import python_speech_features
from utils import audio
import parmap
import multiprocessing
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.io import wavfile
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

dataset_base_dir2 = "/ssd2t/DATASET/VoxCeleb2/Oct1_sync_crop_part2/*/*/"
video_paths = sorted(glob.glob(os.path.join(dataset_base_dir2, "*.mp4")))  # 20

def pp_from_video(video_path):
    # path setting
    video_name = os.path.basename(video_path).split(".")[0]
    save_dir = video_path.replace('.mp4', '')
    frame_save_dir =os.path.join(save_dir, "frames")
    mel_save_path = os.path.join(save_dir, "mel")
    os.makedirs(frame_save_dir, exist_ok=True)
    os.makedirs(mel_save_path, exist_ok=True)
    
    # save frames
    total_num = 0
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        ret, frame = cap.read(image=None)
        if frame is None:
            break
        cv2.imwrite(f'{save_dir}/%06d.png'%total_num, frame)
        total_num += 1
    cap.release()
    
    # save audio
    audio_path = os.path.join(save_dir, f"{video_name}.wav")
    command = "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (
        video_path,
        audio_path,
    )
    output = subprocess.call(command, shell=True, stdout=None)
    
    # no audio file exception
    if not os.path.isfile(audio_path):
        f = open('no_audio_videos.txt', 'a')
        f.write(f'{video_path}\n')
        f.close()
        continue
        
    # pp mel
    with VideoFileClip(video_path) as video_clip:
        video_duration = video_clip.duration

    audio_data, _ = librosa.load(audio_path, sr=sr)
    audio_duration = len(audio_data) / sr

    wav = audio.load_wav(audio_path, sr)
    origin_mel = audio.melspectrogram(wav).T

    min_duration = min(video_duration, audio_duration)
    
    crop_duration = max_crop_duration
    if crop_duration > min_duration:
        crop_duration = int(min_duration)
    
    v_start = 0
    a_start_sample = int(v_start * sr)
    a_end_sample = int((v_start + crop_duration) * sr)
    mel_start_idx = int(80.0 * v_start)  # 1sec -> 80
    mel_end_idx = int(mel_start_idx + 80 * crop_duration)  # 0.2sec 16 -> 10sec : 800
    mel = origin_mel[mel_start_idx:mel_end_idx, :]
    
    # mel
    # 전체 변환 -> 16개 만큼 자르기
    # 오디오 -> 자르고 -> 변환 -> 17, 80
    np.save(
        os.path.join(mel_save_path, f"{str(int(v_start * fps)).zfill(6)}.npy"),
        mel,
    )  # 800(t) 80

fps = 25
frame_duration = 1 / fps
frame_amount = 5
sr = 16000
max_crop_duration = 6000 # second

num_cores = multiprocessing.cpu_count()
print(num_cores)
parmap.map(pp_from_video, video_paths, pm_pbar=True, pm_processes=int(num_cores * .55))
