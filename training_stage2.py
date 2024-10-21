import os
import datetime
import cv2
import json
import torch
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import utils
from data import build_dataset
from models import ModelWarper, StabelWarper
from loss import LossWarper
from utils import log_creator, \
                    print_args, \
                    print_training_log, \
                    write_scalars, \
                    MixedPrecisionTrainer

IMAGENETCLASSNUM = 1000


def train_pipeline(args, cfg):
    logger = log_creator(
        os.path.join(cfg['task_cfg']['log_path'], 
                        "train." + 
                        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log")
    )
    writter = SummaryWriter(cfg['task_cfg']['log_path'])
    print_args(cfg, logger)


    # message_model
    message_model = ModelWarper("message_model", cfg["message_model"], logger, args.pretrained_message_model)
    message_model.eval()

    # generator
    generator = StabelWarper("generator", cfg["generator"], logger)
    generator.train()

    pre_mp_trainer = MixedPrecisionTrainer(model=generator.model, **cfg["generator"]['mix_precision'])
    pre_g_optimizer = getattr(optim, cfg["generator"]["pretrain"]['optimizer']['type'])(
        pre_mp_trainer.master_params, 
        **cfg["generator"]["pretrain"]['optimizer']['kwargs']
    )
    pre_schedulers = {}
    if "scheduler" in cfg["generator"]["pretrain"].keys():
        for scheduler_name, scheduler_cfg in cfg["generator"]["pretrain"]["scheduler"].items():
            pre_schedulers[scheduler_name] = getattr(
                lr_scheduler, 
                cfg["generator"]["pretrain"]['scheduler'][scheduler_name]['type'])\
                                    (
                                        pre_g_optimizer, 
                                        **scheduler_cfg['kwargs'],
                                    )


    # data
    dataset = build_dataset(cfg['training_data']['dataset'])
    cfg['training_data']['dataloader']["collate_fn"] = getattr(\
        utils, 
        cfg['training_data']['dataloader']["collate_fn"]
    )
    dataloader = DataLoader(dataset, **cfg['training_data']['dataloader'])


    # pretrain fusing and mapping layer
    B = cfg["training_data"]["dataloader"]["batch_size"]
    thr = cfg['task_cfg']['thr_stage2']
    cnt = 0
    loss_ema = 1
    while loss_ema > thr:
        for batch in dataloader:
            cnt += 1

            m_out = message_model.train_encode(B)
            g_out = generator.train_iter(**batch, **m_out, fusing_only=True)
            
            # recon latent l2
            # loss re-weight
            
            s, t = g_out["z_m"], g_out["z_0"]
            t = t.detach().to(s.device)
            loss = ((s - t) ** 2).mean()

            pre_mp_trainer.zero_grad()
            loss.backward()
            pre_mp_trainer.optimize(pre_g_optimizer)
            for _, scheduler in pre_schedulers.items():
                scheduler.step()

            loss_ema = 0.99 * loss_ema + 0.01 * loss.detach().item()
            writter.add_scalar("pretrain_loss_latent_l2", loss, cnt)
            writter.add_scalar("pretrain_loss_latent_l2_ema", loss_ema, cnt)
            
            print_training_log(
                logger, 
                {"Training stage 2": {"loss": loss}},
                0, cnt, cnt, 
                {"g": pre_g_optimizer}, 
            )
            for i in range(len(pre_g_optimizer.param_groups)):
                writter.add_scalar("pretrain_g_" + "lr_group_" + str(i), pre_g_optimizer.param_groups[i]['lr'], cnt)

            if loss_ema <= thr:
                break

        generator.save_checkpoint(pre_mp_trainer, -1, cfg['task_cfg']['log_path'])


if __name__ == "__main__":
    utils.set_seed(2024)

    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        default="",
        type=str,
    )
    args.add_argument(
        "--pretrained_message_model",
        default="",
        type=str,
    )
    args.add_argument(
        "--pretrained_fusing_model",
        default="",
        type=str,
    )
    args = args.parse_args()

    f = open(args.config, "r")
    cfg = json.load(f)["inject"]

    train_pipeline(args, cfg)

