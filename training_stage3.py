import os
import datetime
import json
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import utils
from data import build_dataset
from models import StabelWarper, ModelWarper
from loss import LossWarper
from utils import log_creator, \
                    print_args, \
                    print_training_log, \
                    write_scalars, \
                    MixedPrecisionTrainer


def train_pipeline(args, cfg):
    logger = log_creator(
        os.path.join(cfg['task_cfg']['log_path'], 
                        "train." + 
                        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log")
    )
    writter = SummaryWriter(cfg['task_cfg']['log_path'])
    print_args(cfg, logger)

    # task cfg
    epoch_num = cfg['task_cfg']['running_epoch']

    # message_model
    message_model = ModelWarper(
        "message_model", 
        cfg["message_model"], 
        logger, 
        args.pretrained_message_model
    )
    message_model.train()
    
    m_optimizer = getattr(optim, cfg["message_model"]['optimizer']['type'])(
        message_model.model.parameters(), 
        **cfg["message_model"]['optimizer']['kwargs']
    )
    m_schedulers = {}
    if "scheduler" in cfg["message_model"].keys():
        for scheduler_name, scheduler_cfg in cfg["message_model"]["scheduler"].items():
            m_schedulers[scheduler_name] = getattr(
                lr_scheduler, 
                cfg["message_model"]['scheduler'][scheduler_name]['type'])(
                                        m_optimizer, 
                                        **scheduler_cfg['kwargs'],
                                    )

    # generator
    generator = StabelWarper(
        "generator", 
        cfg["generator"], 
        logger, 
        args.pretrained_fusing_model
    )
    generator.train()

    mp_trainer = MixedPrecisionTrainer(model=generator.model, **cfg["generator"]['mix_precision'])
    g_optimizer = getattr(optim, cfg["generator"]['optimizer']['type'])(
        mp_trainer.master_params, 
        **cfg["generator"]['optimizer']['kwargs']
    )
    g_schedulers = {}
    if "scheduler" in cfg["generator"].keys():
        for scheduler_name, scheduler_cfg in cfg["generator"]["scheduler"].items():
            g_schedulers[scheduler_name] = getattr(lr_scheduler, 
                                      cfg["generator"]['scheduler'][scheduler_name]['type'])(
                                      g_optimizer, 
                                      **scheduler_cfg['kwargs'],
                                      )

    # data
    dataset = build_dataset(cfg['training_data']['dataset'])
    cfg['training_data']['dataloader']["collate_fn"] = getattr(\
        utils, 
        cfg['training_data']['dataloader']["collate_fn"]
    )
    dataloader = DataLoader(dataset, **cfg['training_data']['dataloader'])

    # loss
    loss_warper = LossWarper(cfg['loss'])

    # curriculum learning
    B = cfg["training_data"]["dataloader"]["batch_size"]
    thr = cfg['task_cfg']['thr_stage3']
    cnt = 0
    bit_acc_ema = 0.5
    while bit_acc_ema < thr:
        _, batch = enumerate(dataloader).__next__()
        m_out = message_model.train_encode(B)
        g_out = generator.train_iter(**batch, **m_out)
        m_out.update(message_model.train_decode(**batch, **g_out, **m_out))

        loss_info = loss_warper(batch, {"m_out": m_out, "g_out": g_out})

        # loss re-weight
        loss_recon = 0
        for name, val in loss_info.items():
            if name.startswith("recon_"):
                loss_recon = loss_recon + val
        loss_msg = loss_info["msg_bit_loss"]
        loss = loss_recon * 0.1 + loss_msg
        loss_info["loss"] = loss

        mp_trainer.zero_grad()
        m_optimizer.zero_grad()
        loss.backward()

        m_optimizer.step()
        mp_trainer.optimize(g_optimizer)

        print_training_log(
            logger, 
            {"Training stage 3": loss_info},
            0, 0, len(dataloader), 
            {"g": g_optimizer, "m": m_optimizer}, 
        )
                
        write_scalars(
            writter, cnt, 
            {"stage3": loss_info},
            {"g": g_optimizer, "m": m_optimizer}, 
        )

        # bit acc
        msg = m_out["message"]
        msg_pred = m_out["msg_dec"]
        m = msg.detach().cpu().numpy()
        p = msg_pred.detach().cpu().numpy()
        if message_model.model.mode == "regression":
            p[p > 0] = 1
            p[p <= 0] = -1
        elif message_model.model.mode == "logistic":
            p[p > 0.5] = 1
            p[p <= 0.5] = 0
        bit_acc = (p == m).sum() / (m.shape[0] * m.shape[1])
        bit_acc_ema = 0.99 * bit_acc_ema + 0.01 * bit_acc
        writter.add_scalar("bit_acc", bit_acc, cnt)
        writter.add_scalar("bit_acc_ema", bit_acc_ema, cnt)

        if bit_acc_ema > 0.9:
            generator.save_checkpoint(mp_trainer, 0, cfg['task_cfg']['log_path'])
            message_model.save_checkpoint(0, cfg['task_cfg']['log_path'])
        
        cnt += 1

    generator.save_checkpoint(mp_trainer, 0, cfg['task_cfg']['log_path'])
    message_model.save_checkpoint(0, cfg['task_cfg']['log_path'])

    # formal training
    for epoch in range(epoch_num):
        # train
        for idx, batch in enumerate(dataloader):
            m_out = message_model.train_encode(B)
            g_out = generator.train_iter(**batch, **m_out)
            m_out.update(message_model.train_decode(**batch, **g_out, **m_out))
            loss_info = loss_warper(batch, {"m_out": m_out, "g_out": g_out})

            # bit acc
            msg = m_out["message"]
            msg_pred = m_out["msg_dec"]
            m = msg.detach().cpu().numpy()
            p = msg_pred.detach().cpu().numpy()
            if message_model.model.mode == "regression":
                p[p > 0] = 1
                p[p <= 0] = -1
            elif message_model.model.mode == "logistic":
                p[p > 0.5] = 1
                p[p <= 0.5] = 0
            bit_acc = (p == m).sum() / (m.shape[0] * m.shape[1])
            bit_acc_ema = 0.99 * bit_acc_ema + 0.01 * bit_acc
            writter.add_scalar("bit_acc", bit_acc, cnt)
            writter.add_scalar("bit_acc_ema", bit_acc_ema, cnt)

            # loss re-weight
            loss_recon = 0
            for name, val in loss_info.items():
                if name.startswith("recon"):
                    loss_recon = loss_recon + val
            loss_msg = loss_info["msg_bit_loss"] * (bit_acc_ema < 0.9) + loss_info["msg_lse_loss"] * (bit_acc_ema >= 0.9)
            loss = loss_recon + loss_msg
            loss_info["loss"] = loss

            mp_trainer.zero_grad()
            m_optimizer.zero_grad()
            loss.backward()
            mp_trainer.optimize(g_optimizer)
            m_optimizer.step()

            for _, scheduler in g_schedulers.items():
                scheduler.step()
            for _, scheduler in m_schedulers.items():
                scheduler.step()

            print_training_log(
                logger, 
                {"Training stage 3": loss_info},
                epoch, idx, len(dataloader), 
                {"g": g_optimizer, "m": m_optimizer}, 
            )
            write_scalars(
                writter, cnt, 
                {"stage3": loss_info},
                {"g": g_optimizer, "m": m_optimizer}, 
            )
            cnt += 1
        
        generator.save_checkpoint(mp_trainer, 1 + epoch, cfg['task_cfg']['log_path'])
        message_model.save_checkpoint(1 + epoch, cfg['task_cfg']['log_path'])



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
        
