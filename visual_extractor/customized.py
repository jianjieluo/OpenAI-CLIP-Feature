from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from PIL import Image
import torch.nn as nn

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from timm.models.vision_transformer import resize_pos_embed

from clip.clip import _convert_image_to_rgb
from .standard import CLIPRN101, CLIPViTB32

__all__ = ["CLIPRN101_448", "CLIPViTB32_448"]

transform = Compose([
            Resize((448, 448), interpolation=Image.BICUBIC),
            CenterCrop((448, 448)),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])


class CLIPRN101_448(CLIPRN101):

    def __init__(self, args, src_list, dst_list):
        super(CLIPRN101_448, self).__init__(args, src_list, dst_list)

        # larger resolution
        self.transform = transform

        # resize CNN visual.attnpool.positional_embedding for larger resolution
        num_patches = 196
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.model.visual.attnpool.positional_embedding.size(-1), device=self.device),)
        resized_pos_embed_weight = resize_pos_embed(self.model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
        pos_embed = nn.Parameter(resized_pos_embed_weight.squeeze(0),)
        self.model.visual.attnpool.positional_embedding = pos_embed

        # downsample feature map
        self.pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    
class CLIPViTB32_448(CLIPViTB32):

    def __init__(self, args, src_list, dst_list):
        super(CLIPViTB32_448, self).__init__(args, src_list, dst_list)

        # larger resolution
        self.transform = transform

        # resize ViT visual.positional_embedding for larger resolution
        num_patches = 196
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.model.visual.positional_embedding.size(-1),  device=self.device),)
        resized_pos_embed_weight = resize_pos_embed(self.model.visual.positional_embedding.unsqueeze(0), pos_embed)
        pos_embed = nn.Parameter(resized_pos_embed_weight.squeeze(0),)
        self.model.visual.positional_embedding = pos_embed

        # downsample feature map
        self.pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=2)