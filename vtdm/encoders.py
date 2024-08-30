import torch
from annotator.midas.api import MiDaSInference
from einops import rearrange, repeat
from sgm.modules.encoders.modules import AbstractEmbModel
from tools.aes_score import MLP, normalized
import clip
from sgm.modules.diffusionmodules.util import timestep_embedding
from sgm.util import autocast, instantiate_from_config
from torchvision.models.optical_flow import raft_large
from typing import Any, Dict, List, Tuple, Union
from tools.softmax_splatting.softsplat import softsplat
from vtdm.model import create_model, load_state_dict


class DepthEmbedder(AbstractEmbModel):
    def __init__(self, freeze=True, use_3d=False, shuffle_size=3, scale_factor=2.6666):
        super().__init__()
        self.model = MiDaSInference(model_type="dpt_hybrid", model_path="ckpts/dpt_hybrid_384.pt").cuda()
        self.use_3d = use_3d
        self.shuffle_size = shuffle_size
        self.scale_factor = scale_factor
        if freeze:
            self.freeze()
        
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @autocast
    @torch.no_grad()
    def forward(self, x):
        if len(x.shape) == 4:       # (16, 3, 512, 512)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=16)
        B, C, T, H, W = x.shape   # (1, 3, 16, 1024, 1024)
        
        sH = int(H / self.scale_factor / 32) * 32
        sW = int(W / self.scale_factor / 32) * 32
        
        y = rearrange(x, 'b c t h w -> (b t) c h w')
        y = torch.nn.functional.interpolate(y, [sH, sW], mode='bilinear')
        # y = torch.nn.functional.interpolate(y, [576, 1024], mode='bilinear')
        
        y = self.model(y)
        y = rearrange(y, 'b h w -> b 1 h w')
        y = torch.nn.functional.interpolate(y, [H // 8 * self.shuffle_size, W // 8 * self.shuffle_size], mode='bilinear')
        for i in range(y.shape[0]):
            y[i] -= torch.min(y[i])
            y[i] /= max(torch.max(y[i]).item(), 1e-6)
        y = rearrange(y, 'b c (h h0) (w w0) -> b (c h0 w0) h w', h0=self.shuffle_size, w0=self.shuffle_size)
        if self.use_3d:
            y = rearrange(y, '(b t) c h w -> b c t h w', t=T)
        return y
        

class AesEmbedder(AbstractEmbModel):
    def __init__(self, freeze=True):
        super().__init__()
        aesthetic_model, _ = clip.load("ckpts/ViT-L-14.pt")
        del aesthetic_model.transformer
        self.aesthetic_model = aesthetic_model
        self.aesthetic_mlp = MLP(768)
        self.aesthetic_mlp.load_state_dict(torch.load("ckpts/metric_models/sac+logos+ava1-l14-linearMSE.pth"))

        if freeze:
            self.freeze()
        
    def freeze(self):
        self.aesthetic_model = self.aesthetic_model.eval()
        self.aesthetic_mlp = self.aesthetic_mlp.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @autocast
    @torch.no_grad()
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        y = x[:, :, T//2]
        y = torch.nn.functional.interpolate(y, [224, 384], mode='bilinear')
        y = y[:, :, :, 80:304]
        y = (y + 1) * 0.5
        y[:, 0] = (y[:, 0] - 0.48145466) / 0.26862954
        y[:, 1] = (y[:, 1] - 0.4578275) / 0.26130258
        y[:, 2] = (y[:, 2] - 0.40821073) / 0.27577711
        
        image_features = self.aesthetic_model.encode_image(y)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        aesthetic = self.aesthetic_mlp(torch.from_numpy(im_emb_arr).to('cuda').type(torch.cuda.FloatTensor))
        
        return torch.cat([aesthetic, timestep_embedding(aesthetic[:, 0] * 100, 255)], 1)