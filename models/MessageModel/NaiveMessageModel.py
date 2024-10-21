from copy import deepcopy
import torch
import torch.nn as nn
from torchvision.models import resnet18

import numpy as np


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, stride=1, padding=0),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
    

class NaiveMessageModel(nn.Module):
    def __init__(self, bit_num, latent_dim, enc_dim, mode="regression"):
        super(NaiveMessageModel, self).__init__()

        self.bit_num = bit_num
        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.device = "cpu"
        self.mode = mode

        self.encoder = nn.Sequential(
                nn.Linear(bit_num, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, enc_dim),
            )

        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 1024),
                nn.Tanh(),

                nn.Linear(1024, 1024),
                nn.Tanh(),
            
                nn.Linear(1024, 1024),
                nn.Tanh(),

                nn.Linear(1024, bit_num),
            )
        if self.mode == "logistic":
            self.decoder.append(nn.Sigmoid())

    def encode(self, B=None, message=None):
        '''
        B: number of messages. Using this parameter to generate random messages.
        message:  Encode given messages instead of random messages.
        '''
        assert B is not None or message is not None
        if message is None:
            message = torch.zeros((B, self.bit_num), device=self.device)

            pos_p = torch.rand((B, self.bit_num), device=self.device)
            message[pos_p > 0.5] = 1

            if self.mode == "regression":
                message = 2 * (message - 0.5)

        return message, self.encoder(message)
    
    def decode(self, z):
        z = z.reshape(z.shape[0], -1)
        if z.shape[-1] > self.latent_dim:
            z = torch.nn.functional.adaptive_avg_pool1d(z, (self.latent_dim,))
        z = self.decoder(z)
        return z
