import os
import datetime
import cv2
import json
import torch
import argparse
import blobfile as bf

import numpy as np
from glob import glob
from PIL import Image

import utils
from models import StabelWarper, ModelWarper
from utils import log_creator, get_prompts


def load_and_precess_imgs(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = np.array(img)
        img = (img / 255.0 - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        imgs.append(img.unsqueeze(0))
    return torch.cat(imgs, dim=0).to(torch.float32)


@torch.no_grad()
def inject_pipeline(args, cfg):
    logger = log_creator(
        os.path.join(cfg['task_cfg']['log_path'], 
                        "eval.extract." + 
                        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log")
    )

    message_model = ModelWarper(
        "message_model", 
        cfg["message_model"], 
        logger, 
        args.message_model_ckpt
    )
    message_model.eval()

    generator = StabelWarper(
        "generator", 
        cfg["generator"], 
        logger, 
        args.fusing_mapping_model_ckpt
    )
    generator.model.convert_to_fp16()
    generator.eval()

    batch_size = cfg['task_cfg']['generation_cfg']['batch_size']
    thr = cfg['task_cfg']['thr_eval']

    msg = np.load(cfg['task_cfg']['generation_cfg']['msg_path'])
    if message_model.model.mode == "regression":
        msg = 2 * (msg - 0.5)

    save_path = cfg['task_cfg']['generation_cfg']['save_path']
    img_path_list = glob(os.path.join(save_path, "images", "*.png"))
    img_path_list.sort(key=lambda x: int(bf.basename(x).split(".png")[0]))

    idx = 0
    dec_msgs = []
    while idx < len(img_path_list):
        imgs_path = img_path_list[idx: min(idx + batch_size, len(img_path_list))]
        imgs = load_and_precess_imgs(imgs_path)
        msg_z = generator.eval_enc(imgs)
        msg_dec = message_model.eval_decode(msg_z=msg_z)["msg_dec"].detach().cpu().numpy()
        dec_msgs.append(msg_dec)
        idx = idx + batch_size
    dec_msgs = np.concatenate(dec_msgs, axis=0)

    if message_model.model.mode == "regression":
        dec_msgs[dec_msgs > 0] = 1
        dec_msgs[dec_msgs <= 0] = 0
    elif message_model.model.mode == "logistic":
        dec_msgs[dec_msgs > 0.5] = 1
        dec_msgs[dec_msgs <= 0.5] = 0

    np.save(os.path.join(save_path, "decoded_msgs"), dec_msgs)

    rec_flag = (dec_msgs == msg[:dec_msgs.shape[0]])
    bit_acc = rec_flag.sum() / rec_flag.reshape(-1).shape[0]
    det_acc = (rec_flag.sum(axis=-1) > thr).sum() / rec_flag.shape[0]

    logger.info("Decoded messages are saved in {}.".format(os.path.join(save_path, "decoded_msgs")))
    logger.info("Bit Acc: {}, Dec Acc: {}".format(bit_acc, det_acc))


if __name__ == "__main__":
    utils.set_seed(2024)

    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        default="",
        type=str,
    )
    args.add_argument(
        "--message_model_ckpt",
        default="",
        type=str,
    )
    args.add_argument(
        "--fusing_mapping_model_ckpt",
        default="",
        type=str,
    )
    args = args.parse_args()

    f = open(args.config, "r")
    cfg = json.load(f)["inject"]

    inject_pipeline(args, cfg)
        
