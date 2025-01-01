from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import PIL
from abc import ABCMeta

import clip

class BaseClipVisualExtractor(object, metaclass=ABCMeta):
    
    def __init__(self, args, src_list, dst_list, pool2d=None):
        self.args = args
        self.src_list = src_list
        self.dst_list = dst_list
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = clip.load(args.model_type_or_path, jit=False, device=self.device)
        self.pool2d = pool2d
        
        assert len(src_list) == len(dst_list)

    def downsample_feature(self, x):
        # x shape: (WxH) x D
        s = int(np.sqrt(x.size(0)))
        assert x.size(0) == (s * s)
        x = x.permute(1, 0).view(1, -1, s, s).contiguous()

        x = self.pool2d(x)
        x = x[0].permute(1, 2, 0)
        return x

    def extract(self):
        self.model.eval()
        
        pool = list(zip(self.src_list, self.dst_list))
        random.shuffle(pool)
        corrupt_img_ids = []

        for (img_path, dst_path) in tqdm(pool):
            if os.path.isfile(dst_path):
                continue

            if self.args.skip_miss_img and os.path.exists(img_path) == False:
                continue
            
            # load the image
            with torch.no_grad():
                if self.args.debug:
                    img_path = './example/example.jpg'
                
                try:
                    image = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device=self.device)
                except PIL.UnidentifiedImageError:
                    print("Corrupt image: ", img_path)
                    corrupt_img_ids.append(img_path)
                    continue

                local_feat, global_feat = self.model.encode_image(image)

            if len(global_feat.shape) == 1:
                global_feat = global_feat.unsqueeze(0)

            local_feat = local_feat.squeeze(0)
            if len(local_feat.shape) == 3:
                # CNN output D x W x H -> token output (WxH) x D
                local_feat = local_feat.permute(1, 2, 0)
                local_feat = local_feat.view(-1, local_feat.size(-1)).contiguous()

            if self.pool2d:
                local_feat = self.downsample_feature(local_feat)
            
            # final output feature
            local_feat = local_feat.view(-1, local_feat.size(-1)).data.cpu().float().numpy()
            global_feat = global_feat.data.cpu().float().numpy()

            if self.args.debug:
                print('Local feature shape: ', local_feat.shape)     
                print('Global feature shape: ', global_feat.shape)     
                break

            np.savez_compressed(
                dst_path, 
                features=local_feat,
                g_feature=global_feat
            )

        print('#Corrupt_imgs: ', len(corrupt_img_ids))
        with open('corrupt_img_ids.txt', 'w') as f:
            for x in corrupt_img_ids:
                f.write(x+'\n')
