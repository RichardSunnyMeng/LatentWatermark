import torch
import copy
import json
import cv2
import os
from glob import glob
import blobfile as bf
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from . import tools as Tls
from . import transformers as Tfs




class InjectDataset(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.preprocess = cfg['preprocess']
        self.data_aug = cfg['data_aug']
        self.data_list = open(cfg['data_json'], 'r').readlines()


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        info = json.loads(self.data_list[index])

        img = Image.open(info['img_path']).convert("RGB")
        img = np.array(img)

        assert img is not None, "Img read error at {}".format(info['img_path'])

        for name, kwarg in self.data_aug.items():
            img = getattr(Tfs, name)(img, **kwarg)
        for name, kwarg in self.preprocess.items():
            img = getattr(Tls, name)(img, **kwarg)

        info['imgs'] = img

        Tls.cvtTensor(info)
        return info
