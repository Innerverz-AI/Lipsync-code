import argparse
import glob
import math
import os
import subprocess

import cv2
import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d

from packages.detectors import S3FD
from packages.syncnet import SyncNet

from .audio import get_mfcc
from .util import bb_intersection_over_union, calc_pdist


class Sync_tool():
    def __init__(self):
        parser = argparse.ArgumentParser()
        # detector options
        parser.add_argument('--facedet_scale',  type=float, default=0.25)
        parser.add_argument('--crop_scale',     type=float, default=0.40)
        parser.add_argument('--iouThres',      type=float, default=0.5)
        parser.add_argument('--min_track',      type=int, default=100)
        parser.add_argument('--frame_rate',     type=int, default=25)
        parser.add_argument('--num_failed_det', type=int, default=25)
        parser.add_argument('--min_face_size',  type=int, default=100)
        
        
        # video options
        parser.add_argument('--duration',  type=int, default=4)
        
        # syncnet model options
        parser.add_argument('--batch_size',  type=int, default=100)
        parser.add_argument('--vshift',  type=int, default=10)
        
        # save options
        parser.add_argument('--tmp_save_root',  type=str, default='.sync')
        parser.add_argument('--save_root',  type=str, default='./synced_videos')
        parser.add_argument('--delete_tmp',  type=bool, default=True)
        
        self.opts = parser.parse_args()
        
        # imports
        self.DET = S3FD(device='cuda')
        self.syncnet = SyncNet()
        
    def forward(self, video_path, save_path):
        self.separate_info(video_path)
        frames_dets = self.get_bbox()
        track_dets = self.tracking(frames_dets)
        track_dets = self.crop_face(track_dets)
        
        mfcc_feature = get_mfcc(self.opts.audio_save_path)
        
        im_feats, ad_feats = self.get_feats(track_dets, mfcc_feature)
        
        offset, conf = self.get_offset(im_feats, ad_feats)
        
        self.get_synced_video(video_path, save_path, offset)
        
        if self.opts.delete_tmp:
            os.system(f'rm -r {self.opts.tmp_save_path}')
        
        return offset
        
    
    def separate_info(self, video_path):
        self.opts.video_file = os.path.basename(video_path)
        self.opts.video_name = self.opts.video_file.split('.')[0]
        self.opts.tmp_save_path = os.path.join(self.opts.tmp_save_root, self.opts.video_name)
        self.opts.frame_save_path = self.opts.tmp_save_path + "/frames"
        self.opts.audio_save_path = self.opts.tmp_save_path + "/audio.wav"
        
        os.makedirs(self.opts.tmp_save_path, exist_ok=True)
        os.makedirs(self.opts.save_root, exist_ok=True)
        os.makedirs(self.opts.frame_save_path, exist_ok=True)
        
        frame_command = f'ffmpeg -y -i {video_path} -ss 0 -to {0 + self.opts.duration} -r {self.opts.frame_rate} -vb 20M -f image2 -start_number 0 {self.opts.frame_save_path + "/%06d.png"}'
        subprocess.call(frame_command, shell=True, stdout=None)
        
        audio_command = f'ffmpeg -y -i {video_path} -ss 0 -to {0 + self.opts.duration} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {self.opts.audio_save_path}'
        subprocess.call(audio_command, shell=True, stdout=None)

    def get_bbox(self):
        dets = []
        frame_path_list = sorted(glob.glob(self.opts.tmp_save_path + "/frames/*.*"))
        for idx, fpath in enumerate(frame_path_list):
            frame = cv2.imread(fpath)
            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(frame_np, conf_th=0.9, scales=[self.opts.facedet_scale])
            
            frame_dets = []
            for bbox in bboxes:
                frame_dets.append({'index':idx, 'frame':frame, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
            dets.append(frame_dets)
            
        return dets
    
    def tracking(self, frames_dets):
        
        tracks = []
        for index, faces in enumerate(frames_dets):
            for det_idx, face in enumerate(faces):
                if len(tracks) == 0:
                    tracks.append(face)
                    faces.remove(face)
                elif face['index'] - tracks[-1]['index'] <= self.opts.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], tracks[-1]['bbox'])
                    if iou > self.opts.iouThres:
                        tracks.append(face)
                        faces.remove(face)
                        continue
                else:
                    break

        return tracks
        
    def crop_face(self, frames_dets):
        
        # get box position
        dets = {'x': [], 'y': [], 's':[]}
        for frame_dets in frames_dets:
            det = frame_dets['bbox']
            dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y
            
        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

        for frame_dets in frames_dets:
            index = frame_dets['index']
            bs  = dets['s'][index]   # Detection box size
            cs = self.opts.crop_scale
            bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

            image = frame_dets['frame']
            frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
            my  = dets['y'][index]+bsi  # BBox center Y
            mx  = dets['x'][index]+bsi  # BBox center X

            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            _face = cv2.resize(face, (224, 224))
            frame_dets['face'] = _face
        return frames_dets
    
    def get_feats(self, track_dets, mfcc):
        
        # video frame numpy to tensor
        faces = []
        for track_det in track_dets:
            faces.append(track_det['face'])
        im_np = np.stack(faces,axis=3)
        im_np = np.expand_dims(im_np,axis=0)
        im_ts = torch.tensor(np.transpose(im_np,(0,3,4,1,2)), dtype=torch.float32) # shape = (b, c, t, 224, 224)

        cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        # 1sec -> 100
        minframe = min(math.floor(len(track_dets)), math.floor(mfcc.shape[-1]/100*self.opts.frame_rate))
        lastframe = minframe - 5
        
        im_feats, cc_feats, batch_size = [], [], self.opts.batch_size
        for i in range(0, lastframe, batch_size):
            im_batch = [ im_ts[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+batch_size)) ]
            im_in = torch.cat(im_batch,0)

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            
            im_out, cc_out = self.syncnet(im_in.cuda(), cc_in.cuda())
            im_feats.append(im_out.data.cpu())
            cc_feats.append(cc_out.data.cpu())

        im_feats = torch.cat(im_feats, 0)
        cc_feats = torch.cat(cc_feats, 0)
        
        return im_feats, cc_feats
    
    def get_offset(self, im_feats, ad_feats):
        dists = calc_pdist(im_feats, ad_feats, vshift = self.opts.vshift)
        
        mdist = torch.mean(torch.stack(dists,1),1)
        minval, minidx = torch.min(mdist,0)
        offset = self.opts.vshift-minidx
        conf   = torch.median(mdist) - minval
        return offset , conf
    
    def get_synced_video(self, video_path, save_path, offset):
        command = f'ffmpeg -y -i {video_path} -itsoffset {offset/self.opts.frame_rate} -i {video_path} -vb 20M -map 0:v -map 1:a -r {self.opts.frame_rate} {save_path}'
        os.system(command)
    