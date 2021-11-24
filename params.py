import argparse
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--image_list', help='Path of the image_list file', default=None)
    parser.add_argument('--image_dir', help='Root dir of the image path in image_list file', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--anno', default=None)

    # options
    parser.add_argument('--ve_name', type=str, default=None, choices=['RN101', 'ViT-B/32', 'ViT-B/16', 'RN101_448', 'ViT-B/32_448'])
    parser.add_argument('--model_type_or_path', default="RN101", type=str, help='model type from original CLIP or model path offline')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print('Called with args:')
    pprint(vars(args), indent=2)

    return args