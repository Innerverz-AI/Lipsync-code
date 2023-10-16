import glob, os, cv2, librosa, subprocess, torch, sys
import numpy as np
import python_speech_features
import parmap
import multiprocessing
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.io import wavfile
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
sys.path.append('../')
from utils.audio import get_mel

def pp_from_video(video_path, args):
    # path setting
    video_name = os.path.basename(video_path).split(".")[0]
    save_dir = video_path.replace(args.dataset_dir, args.save_dir).replace('.mp4', '')
    frame_save_dir =os.path.join(save_dir, "frames")
    mel_save_path = os.path.join(save_dir, "mel")
    mfcc_save_path = os.path.join(save_dir, "mfcc")
    os.makedirs(frame_save_dir, exist_ok=True)
    os.makedirs(mel_save_path, exist_ok=True)
    os.makedirs(mfcc_save_path, exist_ok=True)
    
    # save frames
    total_num = 0
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        ret, frame = cap.read(image=None)
        if frame is None:
            break
        cv2.imwrite(f'{frame_save_dir}/%06d.png'%total_num, frame)
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
        return
        
    # pp mel
    with VideoFileClip(video_path) as video_clip:
        video_duration = video_clip.duration

    audio_data, _ = librosa.load(audio_path, sr=args.sr)
    audio_duration = len(audio_data) / args.sr

    origin_mel = get_mel(audio_path, args.sr).T
    
    # mfcc
    mfcc_sr, mfcc_wav = wavfile.read(audio_path)
    mfcc = zip(*python_speech_features.mfcc(mfcc_wav, args.sr))
    mfcc = np.stack([np.array(i) for i in mfcc])
    
    min_duration = min(video_duration, audio_duration)
    
    crop_duration = args.max_crop_duration
    if crop_duration > min_duration:
        crop_duration = min_duration
    
    v_start = 0
    a_start_sample = int(v_start * args.sr)
    a_end_sample = int((v_start + crop_duration) * args.sr)
    mel_start_idx = int(80.0 * v_start)  # 1sec -> 80
    mel_end_idx = int(mel_start_idx + 80 * crop_duration)  # 0.2sec 16 -> 10sec : 800
    mel = origin_mel[mel_start_idx:mel_end_idx, :]
    
    # mfcc
    mfcc_wav_segment = mfcc[
        :, int(v_start * args.fps * 4) : int(v_start * args.fps * 4) + int(crop_duration * 100)
    ]  # 0.2sec 20개 -> 1sec : 100 -> 10sec : 1000

    # mel
    # 전체 변환 -> 16개 만큼 자르기
    # 오디오 -> 자르고 -> 변환 -> 17, 80
    np.save(
        os.path.join(mel_save_path, f"{str(int(v_start * args.fps)).zfill(6)}.npy"),
        mel,
    )  # 800(t) 80
    np.save(
        os.path.join(mfcc_save_path, f"{str(int(v_start * args.fps)).zfill(6)}.npy"),
        mfcc_wav_segment,
    )  # 13, 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='z')
    parser.add_argument('--save_dir', type=str, default='z')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--max_crop_duration', type=int, default=6000)
    
    args = parser.parse_args()
    
    video_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "*/*/*.mp4")))  # 20
    num_cores = multiprocessing.cpu_count()
    parmap.map(pp_from_video, video_paths, args, pm_pbar=True, pm_processes=int(num_cores * .55))