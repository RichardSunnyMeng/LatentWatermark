import torch
import torch.nn.functional as F
import lpips
import numpy as np


def msg_lse_loss(x, y):
    msg = y["m_out"]["message"]
    msg_pred = y["m_out"]["msg_dec"]
    lse_loss = ((msg - msg_pred) ** 2).exp().sum(-1).log().mean()
    return lse_loss

def msg_bce_loss(x, y):
    msg = y["m_out"]["message"]
    msg_pred = y["m_out"]["msg_dec"]
    loss = (- msg * torch.log(msg_pred) - (1 - msg) * torch.log(1 - msg_pred)).mean()
    return loss

def msg_reg_loss(x, y):
    msg = y["m_out"]["message"]
    msg_pred = y["m_out"]["msg_dec"]
    loss = ((msg - msg_pred) ** 2).mean()
    return loss

def msg_bit_loss(x, y, mode):
    if mode == "regression":
        return msg_reg_loss(x, y)
    elif mode == "logistic":
        return msg_bce_loss(x, y)


def recon_lpips_loss(x, y, loss_model, thr=0):
    t, s = y["g_out"]["x_0"], y["g_out"]["x_rec"]
    t = t.detach().to(s.device)
    loss = loss_model(t, s).mean()
    loss = loss * (loss > thr)
    return loss


def recon_latent_l2_loss(x, y):
    t, s = y["g_out"]["z_0"], y["g_out"]["z_m"]
    t = t.detach().to(s.device)
    loss = (t - s) ** 2
    return loss.mean()