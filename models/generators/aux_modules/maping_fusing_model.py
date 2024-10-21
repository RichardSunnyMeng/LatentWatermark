import torch as th
from .unet import UNetModel
from utils.fp16_util import convert_module_to_f16, convert_module_to_f32



class MappingFusingUNet(th.nn.Module):
    def __init__(self, dtype=th.float16, z_channel: list=[0], msg_channel=1) -> None:
        super().__init__()
        self.dtype = dtype
        self.z_channel = z_channel
        self.mapping = UNetModel(in_channels=len(z_channel), out_channels=msg_channel, use_fp16=(self.dtype==th.float16))
        self.fusing = UNetModel(in_channels=len(z_channel)+msg_channel, out_channels=len(z_channel), use_fp16=(self.dtype==th.float16))
        self.msg_channel = msg_channel
    
    def fuse(self, z1, z2):
        B, C, W, H = z1.shape
        z2 = z2.reshape(B, -1, W, H)
        z1_split = z1.split(1, 1)
        
        to_fuse = [z2]
        for c in self.z_channel:
            to_fuse.append(z1[:, c:c+1, :, :])
        to_fuse = th.cat(to_fuse, 1)
        fused = self.fusing(to_fuse)

        z = []
        for i in range(z1.shape[1]):
            if i in self.z_channel:
                idx = self.z_channel.index(i)
                z.append(fused[:, idx: idx + 1, :, :])
            else:
                z.append(z1_split[i])
        z = th.cat(z, 1)
        return z
    
    def map(self, z):
        z = self.mapping(z[:, self.z_channel, :, :])
        return None, z
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.mapping.apply(convert_module_to_f16)
        self.fusing.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.mapping.apply(convert_module_to_f32)
        self.fusing.apply(convert_module_to_f32)

