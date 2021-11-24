from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from tqdm import tqdm
import numpy as np
import torch

import clip
import basic_utils as utils
from params import parse_args

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.model_type_or_path, jit=False, device=device)

    # load annotations
    imgs = utils.load_json(args.anno)['images']
    random.shuffle(imgs)

    for img in tqdm(imgs):
        image_id = img['cocoid']
        dst_path = os.path.join(args.output_dir, str(image_id)+'.npz')
        if os.path.isfile(dst_path):
            continue

        # iter over the sentences
        sents = [sent['raw'].lower().strip().strip('.') for sent in img['sentences']]
        sents = clip.tokenize(sents).to(device)

        with torch.no_grad():
            text_feat = model.encode_text(sents)
        text_feat = text_feat.data.cpu().float().numpy()
        
        if args.debug:
            print('Text feature shape: ', text_feat.shape) 
            break

        np.savez_compressed(
            dst_path, 
            g_feature=text_feat
        )


if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        utils.mkdirp(args.output_dir)
    main(args)