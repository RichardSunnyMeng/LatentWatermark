import cv2
import torch
from math import ceil
import numpy as np

def norm(img, mean=0, std=1):
    img = (img / 255.0 - mean) / std
    return img
    
def random_crop(img, target_size):
    x, y = img.shape[:2]
    edge_min = min(x, y)
    if edge_min < target_size:
        x_new = ceil((x / edge_min * target_size))
        y_new = ceil((y / edge_min * target_size))
        img = cv2.copyMakeBorder(img, 0, x_new - x, 0, y_new - y, cv2.BORDER_WRAP)
        x = x_new
        y = y_new
    c_x = np.random.randint(0, x - target_size) if x > target_size else 0
    c_y = np.random.randint(0, y - target_size) if y > target_size else 0
    img = img[c_x: c_x + target_size, c_y: c_y + target_size, :]
    return img

def center_crop(img, target_size):
    x, y = img.shape[:2]
    edge_min = min(x, y)
    if edge_min < target_size:
        x_new = ceil((x / edge_min * target_size))
        y_new = ceil((y / edge_min * target_size))
        img = cv2.copyMakeBorder(img, 0, x_new - x, 0, y_new - y, cv2.BORDER_WRAP)
        x = x_new
        y = y_new
    c_x = (x - target_size) // 2
    c_y = (y - target_size) // 2
    img = img[c_x: c_x + target_size, c_y: c_y + target_size, :]
    return img

def img_resize(img, target_size):
    img = cv2.resize(img, (target_size, target_size))
    return img

def rescale(img, min_size):
    x, y = img.shape[:2]
    min_edge = min(x, y)
    scale = min_size / min_edge
    img = cv2.resize(img, (int(y * scale), int(x * scale)))
    return img

def cvtTensor(info: dict) -> None:
    keys = info.keys()
    for key in keys:
        if 'label' in key or key == 'y':
            tpe = torch.long
        elif 'img' in key:
            tpe = torch.float
        else:
            tpe = type(info[key])
        
        if "np" in key:
            continue

        if isinstance(info[key], int):
            info[key] = torch.tensor(info[key], dtype=tpe)
        if isinstance(info[key], np.ndarray):
            info[key] = torch.from_numpy(info[key]).to(tpe).permute(2, 0, 1)
    return