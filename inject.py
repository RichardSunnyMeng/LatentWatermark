import os
import datetime
import cv2
import json
import torch
import argparse
import numpy as np

import utils
from models import StabelWarper, ModelWarper
from utils import log_creator, get_prompts


@torch.no_grad()
def inject_pipeline(args, cfg):
    logger = log_creator(
        os.path.join(cfg['task_cfg']['log_path'], 
                        "eval.inject." + 
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
    num_per_class = cfg['task_cfg']['generation_cfg']['num_per_class']
    idxes, prompts = get_prompts(cfg['task_cfg']['generation_cfg']["prompts"], num_per_class)

    msg = np.load(cfg['task_cfg']['generation_cfg']['msg_path'])
    if message_model.model.mode == "regression":
        msg = 2 * (msg - 0.5)

    save_path = os.path.join(cfg['task_cfg']['generation_cfg']['save_path'], "images")
    os.makedirs(save_path,exist_ok=True)

    for i in range(0, len(idxes), batch_size):
        prompt = prompts[i: min(i + batch_size, len(prompts))]
        idx = idxes[i: min(i + batch_size, len(idxes))]

        msg_batch = msg[i: i + len(prompt)]
        msg_batch = torch.from_numpy(msg_batch).to(torch.float32)
        msg_z = message_model.eval_encode(message=msg_batch)["msg_z"]  
        samples = generator.eval_iter(y=prompt, msg_z=msg_z)

        for ii in range(samples.shape[0]):
            img = samples[ii]
            label = idx[ii]
            cv2.imwrite(os.path.join(save_path, "{}.png".format(str(i + ii))), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            logger.info("create {}-th sample for prompt [{}].".format(i + ii, label))


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
        
