from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .basic import BaseClipVisualExtractor

__all__ = ["CLIPRN50X4", "CLIPRN101", "CLIPViTB32", "CLIPViTB16", "CLIPViTL14"]

class CLIPRN50X4(BaseClipVisualExtractor):

    def __init__(self, args, src_list, dst_list):
        super(CLIPRN50X4, self).__init__(args, src_list, dst_list)


class CLIPRN101(BaseClipVisualExtractor):

    def __init__(self, args, src_list, dst_list):
        super(CLIPRN101, self).__init__(args, src_list, dst_list)


class CLIPViTB32(BaseClipVisualExtractor):

    def __init__(self, args, src_list, dst_list):
        super(CLIPViTB32, self).__init__(args, src_list, dst_list)


class CLIPViTB16(BaseClipVisualExtractor):

    def __init__(self, args, src_list, dst_list):
        super(CLIPViTB16, self).__init__(args, src_list, dst_list)
        # downsample feature map
        self.pool2d = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)


class CLIPViTL14(BaseClipVisualExtractor):

    def __init__(self, args, src_list, dst_list):
        super(CLIPViTL14, self).__init__(args, src_list, dst_list)
        # downsample feature map
        self.pool2d = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)