
import numpy as np


def get_prompts(path, num_per_class=1):
    idx = []
    captions = []
    f = open(path, "r").readlines()
    for c in f:
        for _ in range(num_per_class):
            captions.append(c.strip())
            idx.append(c.strip().replace(" ", "_"))
    return idx, captions
