from .standard import *
from .customized import *

__factory = {
    # standard
    'RN50x4': CLIPRN50X4,
    'RN101': CLIPRN101,
    'ViT-B/32': CLIPViTB32,
    'ViT-B/16': CLIPViTB16,
    'ViT-L/14': CLIPViTL14,

    # custom
    'RN101_448': CLIPRN101_448,
    'ViT-B/32_448': CLIPViTB32_448
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown clip model:", name)
    return __factory[name](*args, **kwargs)