import torch
import torch.nn as nn
from . import loss_fn as L
import lpips

class LossWarper(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg["device"]

        if "recon_lpips_loss" in self.cfg.keys():
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)

    
    def forward(self, x, y):
        loss_info = {}
        loss = 0.0

        for loss_name, loss_kwargs in self.cfg.items():
            if "loss" not in loss_name:
                continue

            if "recon_lpips_loss" == loss_name:
                loss_info[loss_name] =  getattr(L, loss_name)(x, y, loss_model=self.lpips_model) * loss_kwargs['scale']
            elif "recon_watson_fft_loss" == loss_name:
                loss_info[loss_name] =  getattr(L, loss_name)(x, y, loss_model=self.watson_fft_model) * loss_kwargs['scale']
            else:
                loss_info[loss_name] = getattr(L, loss_name)(x, y, **loss_kwargs['kwargs']) * loss_kwargs['scale']
        
        if isinstance(loss, float):
            loss = torch.tensor(0, device=self.device)
        return loss_info
    
    def get_loss_fn(self, loss_name):
        return getattr(L, loss_name)