# Mirrored X-Net
This repository contains pytorch supported code and configuration of the proposed weakly supervised model for geographic atrophy (GA) lesions segmentation:

## Contents
- [Introduction](#Introduction)
- [Dataset](#Dataset)
- [Running](#Running)
- [Requirements](#Requirements)

## Introduction
- TO DO: Introduce our work.

## Dataset
We utlize two different GA datasets (Dataset 1 and Dataset 2) and one normal (Dataset 3) dataset to evaluate the model. Each dataset contains SD-OCT cubes with three dimensions. All the cubes contain the advanced non-exudative AMD with GA.
> Our datasets were acquired with a Cirrus OCT device from Stanford University.
### The summarizations of data information on three datasets

|             |   Dataset 1  |   Dataset 2  |   Dataset 3  |
| ----------- | ------------ | ------------ | ------------ |
|     Size    | 512×128×1024 | 200×200×1024 | 512×128×1024 |
|    Cubes    |      51      |      54      |      63      |
| Individuals |      8       |      54      |      38      |
|     Eyes    |      12      |      54      |      62      |
|   Category  |      GA      |      GA      |    Normal    |


## Running

### Data preparation

- Our file structure looks like:
```bash
$ tree data
├── dataset
│   ├── CubeScan
│   │   ├── eye1_cube_z
│   │   │   ├── 1.bmp
│   │   │   ├── 2.bmp
│   │   │   ├── 3.bmp
│   │   ├── eye2_cube_z
│   │   ├── eye3_cube_z
│   ├── SegmentationBScan
│   │   ├── eye1_cube_z
│   │   │   ├── 1.bmp
│   │   │   ├── 2.bmp
│   │   │   ├── 3.bmp
│   │   ├── eye2_cube_z
│   │   ├── eye3_cube_z
│   ├── Segmentation2D
│   │   ├── eye1_cube_z.bmp
│   │   ├── eye2_cube_z.bmp
│   │   ├── eye3_cube_z.bmp
│   ├── statistics.xls
```

**Notes:**
- Especially, GA is defined as a 2D lesion manifestation. The training datas are 3D cubes, the ground truths are 2D projections. We use B-scans in random order as input for training phase, all B-scans of a cube as input for test phase to produce a projection attention map.
- Normal datasets, e.g. Dataset 3, can be introduced as additional normal images.

### Configurations

- The codes are follow `pythorch`
    - The path of model building is `models\`
    - The path of dataset building is `datasets\`
    - The running file is `main.py`

- The hyper-parameter formats follow `mmcv`.
    - The path of model configuration is `configs\models\mxnet.py`.
    - The paths of dataset configuration are `configs\datasets\GA_128_bscan.py` and `configs\datasets\GA_200_bscan.py`.
    - The paths of cross validation configuration are `configs\cross_validation_128\` and `configs\cross_validation_200\`.


### Training and Testing
```
# single-gpu running
CUDA_VISIBLE_DEVICE=0 python main.py <CONFIG_FILE> --work-dir <WORK_DIR> --show-dir <SHOW_DIR>

# example for one fold of cross-validation
CUDA_VISIBLE_DEVICE=0 python main.py conifgs/cross_validation_128/1.py --work-dir output/128/1 --show-dir ./show/128/1
```

**Notes:**
- This version of code does not support distributed training and testing for now. 
- This version of code does not support loading preatrained parameters for now. 

## Requirements
* My running environments, based on Ubuntu 18.04:
    * torch
    * mmcv-full
    * timm
    * loguru
    * skimage
    * pydensecrf

## Citing
If this project is help for you, please cite it.
```
@article{ji2024mirrored,
  title={Mirrored X-Net: Joint classification and contrastive learning for weakly supervised GA segmentation in SD-OCT},
  author={Ji, Zexuan and Ma, Xiao and Leng, Theodore and Rubin, Daniel L and Chen, Qiang},
  journal={Pattern Recognition},
  volume={153},
  pages={110507},
  year={2024},
  publisher={Elsevier}
}
```
