from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import visual_extractor
import basic_utils as utils
from params import parse_args



def main(args):
    # load image list
    imgs = utils.read_lines(args.image_list)
    imgs = [x.split(', ') for x in imgs]
    
    # build src_list & dst_list
    src_list = [os.path.join(args.image_dir, img[1]) for img in imgs]
    dst_list = [os.path.join(args.output_dir, img[0]+'.npz') for img in imgs]
    src_list, dst_list = utils.shuffle_list(src_list, dst_list)

    # extract
    worker = visual_extractor.create(args.ve_name, args, src_list, dst_list)
    worker.extract()


if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        utils.mkdirp(args.output_dir)
    main(args)
