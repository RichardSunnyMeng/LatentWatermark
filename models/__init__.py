import os
import torch
import copy
from glob import glob

import utils
from loss import LossWarper
from .MessageModel import NaiveMessageModel
from .generators import LatentDiffusion, DDIMSampler
from .generators.aux_modules import MappingFusingUNet



class ModelWarper():
    def __init__(self, name:str, config:dict, logger=None, pertrained_model=None) -> None:
        self.name = name
        self.logger = logger
        self.device = config['device']

        self.model = NaiveMessageModel(**config['model']['kwargs']).to(device=self.device)
        self.model.device = self.device

        self.init(pertrained_model)

    def init(self, pretrain_model_path=None):
        # load pretrained model
        if pretrain_model_path is not None:
            if os.path.exists(pretrain_model_path):
                state_dict = torch.load(pretrain_model_path, map_location=torch.device('cpu'))["model"]
                missing, unexpected = self.model.load_state_dict(state_dict=state_dict, strict=False)
                self.logger.info("Load pretrained model {} success!".format(self.name))
                self.logger.info("Load the pretrained model {}".format(pretrain_model_path))
                self.logger.info("Missing Module: {}".format(missing))
                self.logger.info("Unexpected Module: {}".format(unexpected))
            else:
                raise FileNotFoundError
    
    def save_checkpoint(self, epoch, path):
        path = os.path.join(path, "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save({'model': self.model.state_dict()}, os.path.join(path, self.name + "_epoch_{}.pth".format(epoch)))
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def train_encode(self, B, message=None, **kwargs):
        if message is None:
            message, z = self.model.encode(B=B)
        else:
            _, z = self.model.encode(message=message.to(self.device))
        return {"msg_z": z, "message": message}
    
    def train_decode(self, msg_z, **kwargs):
        return {"msg_dec": self.model.decode(msg_z.to(torch.float32))}

    @torch.no_grad()
    def eval_encode(self, message, **kwargs):
        _, z = self.model.encode(message=message.to(self.device))
        return {"msg_z": z}

    @torch.no_grad()
    def eval_decode(self, msg_z, **kwargs):
        return {"msg_dec": self.model.decode(msg_z.to(torch.float32))}



class StabelWarper():
    def __init__(self, name:str, config:dict, logger=None, pertrained_model=None) -> None:
        self.name = name
        self.logger = logger
        self.device = config['device']

        self.model = MappingFusingUNet(dtype=torch.float16,)
        
        self.diffusion = LatentDiffusion(**config["model"]["kwargs"])
        self.ddim_sampler = DDIMSampler(model=self.diffusion, **config["ddim"])

        self.init(config, pertrained_model)
        
        self.diffusion.eval()
        self.diffusion.requires_grad_(False)
        self.diffusion.convert_to_fp16()
        self.diffusion.to(self.device)
        self.model.convert_to_fp16()
        self.model.to(self.device)
        self.data_type = self.diffusion.data_type
        
    def init(self, config, pretrain_model_path=None):
        # load stable diffusion
        state_dict = torch.load(config['sd_ckpt'], map_location=torch.device('cpu'))
        missing, unexpected = self.diffusion.load_state_dict(state_dict=state_dict, strict=False)
        self.logger.info("Load pretrained SD successfully!")
        self.logger.info("Load the state dict {}".format(config['sd_ckpt']))
        self.logger.info("Missing Module: {}".format(missing))
        self.logger.info("Unexpected Module: {}".format(unexpected))

        # load pretrained model
        if pretrain_model_path is not None:
            if os.path.exists(pretrain_model_path):
                state_dict = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
                missing, unexpected = self.model.load_state_dict(state_dict=state_dict, strict=False)
                self.logger.info("Load pretrained model {} success!".format(self.name))
                self.logger.info("Load the pretrained model {}".format(pretrain_model_path))
                self.logger.info("Missing Module: {}".format(missing))
                self.logger.info("Unexpected Module: {}".format(unexpected))
            else:
                raise FileNotFoundError
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def save_checkpoint(self, mp_trainer, epoch, path):
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
        state_dict = mp_trainer.master_params_to_state_dict(mp_trainer.master_params)
        filename = os.path.join(path, "checkpoints", self.name + "_epoch_{}.pth".format(epoch))
        torch.save(state_dict, filename)

    def train_iter(self, imgs, txt, msg_z, fusing_only=False, **kwargs):
        inputs = {"image": imgs, "txt": txt}
        utils.toDevice(self.device, inputs)
        self.diffusion.convert_batch_to_dtype(inputs)
        
        # encode images
        z, _ = self.diffusion.get_input(inputs, self.diffusion.first_stage_key)
        z = z.to(self.data_type).detach()

        # inject messages
        z_m = self.model.fuse(z, msg_z)

        if fusing_only:
            return {"z_m": z_m, "z_0": z}

        out = self.diffusion.differentiable_decode_first_stage(z_m)

        # encode images
        z_rec = self.diffusion.differentiable_encode_first_stage(out).mean
        _, z_rec = self.model.map(z_rec)

        return {"z_rec": z_rec, "z_m": z_m, "z_0": z, "x_rec": out, "x_0": imgs}
    

    @torch.no_grad()
    def eval_enc(self, imgs, **kwargs):
        imgs = imgs.to(self.device).to(self.data_type)
        z_enc = self.diffusion.differentiable_encode_first_stage(imgs).mean
        _, z_enc = self.model.map(z_enc)
        return z_enc

    @torch.no_grad()
    def eval_latent(self, imgs, **kwargs):
        imgs = imgs.to(self.device).to(self.data_type)
        z = self.diffusion.differentiable_encode_first_stage(imgs).mean
        _, z_enc = self.model.map(z)
        return z, z_enc

    @torch.no_grad()
    def eval_iter(self, y, msg_z=None, **kwargs):
        uc = self.diffusion.get_learned_conditioning(len(y) * [""])
        c = self.diffusion.get_learned_conditioning(y)

        # generate z
        sample_fn = self.ddim_sampler.sample
        sample, _ = sample_fn(batch_size=len(y), conditioning=c, unconditional_conditioning=uc)

        # inject messages
        sample = self.model.fuse(sample, msg_z)

        # decode z
        sample = self.diffusion.decode_first_stage(sample.float())

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1) # RGB
        sample = sample.contiguous().detach().cpu().numpy()
        return sample
    
    
    def encode_delta(self, x):
        x = x.half().to(self.device)
        encoder_posterior = self.diffusion.encode_first_stage(x)
        z = self.diffusion.get_first_stage_encoding(encoder_posterior).detach()
        return z

    @torch.no_grad()
    def get_z(self, x):
        x = x.half().to(self.device)
        return self.diffusion.encode_first_stage(x).mean
    

