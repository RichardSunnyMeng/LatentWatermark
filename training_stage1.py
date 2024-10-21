import os
import datetime
import json
import torch
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import utils
from models import ModelWarper
from utils import log_creator, \
                    print_args, \
                    print_training_log


def train_pipeline(args, cfg):
    os.makedirs(cfg['task_cfg']['log_path'], exist_ok=True)
    logger = log_creator(
        os.path.join(cfg['task_cfg']['log_path'], 
                        "train." + 
                        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log")
    )
    writter = SummaryWriter(cfg['task_cfg']['log_path'])
    print_args(cfg, logger)

    # message_model
    message_model = ModelWarper("message_model", cfg["message_model"], logger)
    message_model.train()
    pre_optimizer = getattr(optim, cfg["message_model"]["pretrain"]['optimizer']['type'])(
        message_model.model.parameters(), 
        **cfg["message_model"]["pretrain"]['optimizer']['kwargs']
    )
    pre_schedulers = {}
    if "scheduler" in cfg["message_model"]["pretrain"].keys():
        for scheduler_name, scheduler_cfg in cfg["message_model"]["pretrain"]["scheduler"].items():
            pre_schedulers[scheduler_name] = \
                getattr(
                    lr_scheduler, 
                    cfg["message_model"]["pretrain"]['scheduler'][scheduler_name]['type'])\
                        (
                            pre_optimizer, 
                            **scheduler_cfg['kwargs'],
                        )


    # pretrain msg model
    B = cfg["training_data"]["dataloader"]["batch_size"]
    thr = cfg['task_cfg']['thr_stage1']
    bit_acc_ema = 0.5
    cnt = 0
    while bit_acc_ema < thr:
        m_out = message_model.train_encode(B)
        m_out.update(message_model.train_decode(**m_out))
        
        msg = m_out["message"]
        msg_pred = m_out["msg_dec"]

        # loss
        if message_model.model.mode == "regression":
            loss = ((msg - msg_pred) ** 2).mean()
        elif message_model.model.mode == "logistic":
            loss = (- msg * torch.log(msg_pred) - (1 - msg) * torch.log(1 - msg_pred)).mean()
        else:
            raise NotImplementedError
        writter.add_scalar("pretrain_loss_bit_acc", loss, cnt)

        pre_optimizer.zero_grad()
        loss.backward()
        pre_optimizer.step()
        for _, scheduler in pre_schedulers.items():
            scheduler.step()

        # bit acc
        m = msg.detach().cpu().numpy()
        p = msg_pred.detach().cpu().numpy()
        if message_model.model.mode == "regression":
            p[p > 0.5] = 1
            p[p < -0.5] = -1
        elif message_model.model.mode == "logistic":
            p[p > 0.75] = 1
            p[p < 0.25] = 0

        bit_acc = (p == m).sum() / (m.shape[0] * m.shape[1])
        bit_acc_ema = 0.99 * bit_acc_ema + 0.01 * bit_acc
        writter.add_scalar("pretrain_bit_acc", bit_acc, cnt)
        writter.add_scalar("pretrain_bit_acc_ema", bit_acc_ema, cnt)

        print_training_log(
            logger, 
            {"Training stage 1": {"loss": loss}},
            0, cnt, cnt, 
            {"m": pre_optimizer}, 
        )

        for i in range(len(pre_optimizer.param_groups)):
            writter.add_scalar("pretrain_m_" + "lr_group_" + str(i), pre_optimizer.param_groups[i]['lr'], cnt)
        cnt += 1

    message_model.save_checkpoint(-1, cfg['task_cfg']['log_path'])

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
        
