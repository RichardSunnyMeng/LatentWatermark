from . import dataset as D

def build_dataset(cfg, **kwargs):
    dataset = getattr(D, cfg['type'])(cfg, **kwargs)
    return dataset