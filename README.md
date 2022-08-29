# Extracting OpenAI CLIP (Global/Grid) Features from Image and Text

This repo aims at providing an easy to use and efficient code for extracting image & text features using the [official OpenAI CLIP models](https://github.com/openai/CLIP), which is also optimized for multi processing GPU feature extraction.

The [official OpenAI CLIP repo](https://github.com/openai/CLIP) only supports extracting global visual features, while the local grid features from CLIP visual models may also contain more detailed semantic information which can benefit multi visual-and-language downstream tasks[\[1\]](#1)[\[2\]](#2). As an alternative, this repo encapsulates minor-modified CLIP code in order to extract **not only global visual features but also local grid visual features** from different CLIP visual models. What's more, this repo is designed in a user-friendly object-oriented fashion, allowing users to add their customized `visual_extractor` classes **easily to customize different input and output grid resolution**.

To verify the semantic meaning of the extracted visual grid features, we also applied the extracted visual grid features of MSCOCO images from different official CLIP models for standard image captioning task. We got comparable or superior results in transformer baseline **easily without hard-tuning hyperparameters**, via simply replacing [BUTD features](https://github.com/peteanderson80/bottom-up-attention) with the extracted CLIP gird features. Surprisingly, we got `116.9` CIDEr score in teacher-forcing setting and `129.6` in reinforcement learning setting when using `ViT-B/32` CLIP model, which **conflicts with** the experiment results in [CLIP-ViL paper](https://arxiv.org/pdf/2107.06383.pdf)[\[1\]](#1) where the authors observed that CLIP-ViT-B with grid features has a large performance degradation compared with other models (`58.0` CIDEr score in `CLIP-ViT-B_Transformer` setting in COCO Captioning).

We provide supported CLIP models, results on MSCOCO image captioning, and other information below. We believe **this repo can facilitate the usage of powerful CLIP models**.

## 1. Supported CLIP Models

Currently this repo supports five visual extractor settings, including three **standard** pipelines used in official OpenAI CLIP repo and two additional **customized** pipelines supporting larger input resolution. You can refer to [this file](visual_extractor/customized.py) for more details about customizing your own visual backbones for different input and output resolution. In order to imporve training efficiency in image captioning task, we apply `AvgPool2d` to the output feature map to reduce grid features size in some settings without large performance degradation. We will support more CLIP models in the future.

<table>
<thead>
  <tr>
    <th></th>
    <th>Visual Backbone</th>
    <th>CLIP Model</th>
    <th>Input Resolution</th>
    <th>Output Resolution</th>
    <th>Feature Map Downsample</th>
    <th>Grid Feature Shape</th>
    <th>Global Feature Shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">Standard</td>
    <td>RN101</td>
    <td>RN101</td>
    <td>224 x 224</td>
    <td>7 x 7</td>
    <td>None</td>
    <td>49 x 2048</td>
    <td>1 x 512</td>
  </tr>
  <tr>
    <td>ViT-B/32</td>
    <td>ViT-B/32</td>
    <td>224 x 224</td>
    <td>7 x 7</td>
    <td>None</td>
    <td>49 x 768</td>
    <td>1 x 512</td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>ViT-B/16</td>
    <td>224 x 224</td>
    <td>14 x 14</td>
    <td>AvgPool2d(kernel_size=(2,2), stride=2)</td>
    <td>49 x 768</td>
    <td>1 x 512</td>
  </tr>
  <tr>
    <td rowspan="2">Customized</td>
    <td>RN101_448</td>
    <td>RN101</td>
    <td>448 x 448</td>
    <td>14 x 14</td>
    <td>AvgPool2d(kernel_size=(2,2), stride=2)</td>
    <td>49 x 2048</td>
    <td>1 x 512</td>
  </tr>
  <tr>
    <td>ViT-B/32_448</td>
    <td>ViT-B/32</td>
    <td>448 x 448</td>
    <td>14 x 14</td>
    <td>AvgPool2d(kernel_size=(2,2), stride=2)</td>
    <td>49 x 768</td>
    <td>1 x 512</td>
  </tr>
</tbody>
</table>

## 2. Results on MSCOCO Image Captioning (Karpathy's Splits)

We ran image captioning experiments on [X-modaler](https://github.com/YehLi/xmodaler) with the extracted CLIP grid features. We easily got comparable or superior results in transformer baseline using the default hyperparameters in X-modaler's transformer baseline, except for `SOLVER.BASE_LR=2e-4` in `ViT-B/16` and `ViT-B/32_448` teacher-forcing settings. The performance of transformer baseline using [BUTD features](https://github.com/peteanderson80/bottom-up-attention) is taken from [X-modaler's paper](https://arxiv.org/pdf/2108.08217.pdf).

### 2.1 Teacher-forcing

| Name         | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | ROUGE-L | CIDEr-D | SPICE |
| :---:        | :---:  | :---:  | :---:  | :---:  | :---:  | :---:   | :---:   | :---: |
| BUTD_feat    | 76.4   | 60.3   | 46.5   | 35.8   | 28.2   | 56.7    | 116.6   | 21.3  |
| RN101        | 77.3   | 61.3   | 47.7   | 36.9   | 28.7   | 57.5    | 120.6   | 21.8  |
| ViT-B/32     | 76.4   | 60.3   | 46.5   | 35.6   | 28.1   | 56.7    | 116.9   | 21.2  |
| ViT-B/16     | 78.0   | 62.1   | 48.2   | 37.2   | 28.8   | 57.6    | 122.3   | 22.1  |
| RN101_448    | 78.0   | 62.4   | 48.9   | 38.0   | 29.0   | 57.9    | 123.6   | 22.1  |
| ViT-B/32_448 | 75.8   | 59.6   | 45.9   | 35.1   | 27.8   | 56.3    | 114.2   | 21.0  |

### 2.2 Self-critical Reinforcement Learning

| Name         | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | ROUGE-L | CIDEr-D | SPICE |
| :---:        | :---:  | :---:  | :---:  | :---:  | :---:  | :---:   | :---:   | :---: |
| BUTD_feat    | 80.5   | 65.4   | 51.1   | 39.2   | 29.1   | 58.7    | 130.0   | 23.0  |
| RN101        | 81.3   | 66.4   | 52.1   | 40.3   | 29.6   | 59.6    | 134.2   | 23.4  |
| ViT-B/32     | 79.9   | 64.6   | 50.4   | 38.5   | 29.0   | 58.6    | 129.6   | 22.8  |
| ViT-B/16     | 82.0   | 67.3   | 53.1   | 41.1   | 29.9   | 59.8    | 136.6   | 23.8  |
| RN101_448    | 81.6   | 66.9   | 52.6   | 40.6   | 29.9   | 59.8    | 136.2   | 23.9  |
| ViT-B/32_448 | 79.9   | 64.6   | 50.4   | 38.7   | 28.8   | 58.4    | 127.8   | 22.6  |

## 3. Get Started

**Note**: The extracted feature files **are compatible with** [X-modaler](https://github.com/YehLi/xmodaler), where you can setup your experiments about cross-modal analytics conveniently.

### 3.1 Requirements

- PyTorch ≥ 1.9 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
- timm ≥ 0.4.5

### 3.2 Examples

1. Use CLIP `ViT-B/32` model to extract global textual features of MSCOCO sentences from `dataset_coco.json` in [Karpathy's released annotations](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

```bash
CUDA_VISIBLE_DEVICES=0 python3 clip_textual_feats.py \
    --anno dataset_coco.json \
    --output_dir ${TXT_OUTPUT_DIR} \
    --model_type_or_path 'ViT-B/32'
```

2. Use CLIP `ViT-B/16` model to extract global and grid visual features of MSCOCO images.

```bash
CUDA_VISIBLE_DEVICES=0 python3 clip_visual_feats.py \
    --image_list 'example/MSCOCO/image_list_2017.txt' \
    --image_dir ${IMG_DIR} \
    --output_dir ${IMG_OUTPUT_DIR} \
    --ve_name 'ViT-B/16' \
    --model_type_or_path 'ViT-B/16'
```

3. Use CLIP `RN101` model to extract global and grid visual features of MSCOCO images.

```bash
CUDA_VISIBLE_DEVICES=0 python3 clip_visual_feats.py \
    --image_list 'example/MSCOCO/image_list_2017.txt' \
    --image_dir ${IMG_DIR} \
    --output_dir ${IMG_OUTPUT_DIR} \
    --ve_name 'RN101' \
    --model_type_or_path 'RN101'
```

4. Use CLIP `RN101` model to extract global and grid visual features of MSCOCO images **with 448 x 448 resolution**.

```bash
CUDA_VISIBLE_DEVICES=0 python3 clip_visual_feats.py \
    --image_list 'example/MSCOCO/image_list_2017.txt' \
    --image_dir ${IMG_DIR} \
    --output_dir ${IMG_OUTPUT_DIR} \
    --ve_name 'RN101_448' \
    --model_type_or_path 'RN101'
```

### 3.3 Speeding up feature extraction with Multiple GPUs

You can run the same script with same input list (i.e. `--image_list` or `--anno`) on another GPU (that can be from a different machine, provided that the disk to output the features is shared between the machines). The script will create a new feature extraction process that will only focus on processing the items that have not been processed yet, without overlapping with the other extraction process already running.

## 4. License

MIT

## 5. Acknowledgement

This repo used resources from [OpenAI CLIP](https://github.com/openai/CLIP), [timm](https://github.com/rwightman/pytorch-image-models), [CLIP-ViL](https://github.com/clip-vil/CLIP-ViL), [X-modaler](https://github.com/YehLi/xmodaler). The repo is implemented using PyTorch. We thank the authors for open-sourcing their awesome projects.

## 6. References

<p id="1">[1] How Much Can CLIP Benefit Vision-and-Language Tasks? Sheng Shen, Liunian Harold Li, Hao Tan,  Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, Kurt Keutzer. In Arxiv2021.</p>

<p id="2">[2] In Defense of Grid Features for Visual Question Answering. Huaizu Jiang, Ishan Misra, Marcus Rohrbach, Erik Learned-Miller, Xinlei Chen. In CVPR2020.</p>

<p id="3">[3] X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics. Yehao Li, Yingwei Pan, Jingwen Chen, Ting Yao, Tao Mei. In ACMMM2021 Open Source Software Competition.</p>