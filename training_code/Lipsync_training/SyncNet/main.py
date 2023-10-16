
import torch
import torch.nn as nn

from .syncnet import SyncNet_color_1024

class SyncNet(nn.Module):
    def __init__(self, ckpt_path = 'SyncNet/ckpt/VoxCeleb2_add_dataset_110k.pth'):
        super(SyncNet, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.syncnet = SyncNet_color_1024().to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device)["state_dict"]
        
        self.syncnet.load_state_dict(ckpt)
        for param in self.syncnet.parameters():
            param.requires_grad = False
        self.syncnet.eval()
        del ckpt

        self.logloss = nn.BCELoss()

    # shape
    #   audio : (b, 1, 80, 13)
    #   video : (b, 15, 64, 128)
    def forward(self, audio, video):
        a, v = self.syncnet(audio, video)
        
        return a, v

    def get_loss(self, audio, fake_video):
        a, v = self.syncnet(audio, fake_video)
        y = torch.ones(fake_video.size(0), 1).float().to(a.device)
        d = nn.functional.cosine_similarity(a, v)
        loss = self.logloss(d.unsqueeze(1), y)
        return loss
    
# if __name__ == "__main__":
#     videos = torch.randn((8,15,64,128)).to('cuda')
#     audios = torch.randn((8,1,80,13)).to('cuda')
    
#     syncnet = SyncNet()
#     loss = syncnet.get_loss(audios, videos)