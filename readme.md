# GaussianAD: Gaussian-Centric End-to-End Autonomous Driving
### [Paper](https://arxiv.org/pdf/)  | [Project Page](https://wzzheng.net/GaussianAD)  | [Code](https://github.com/wzzheng/GaussianAD)
![logo](./assets/logo.jpg)

Check out our [Large Driving Model](https://github.com/wzzheng/LDM/) Series!

> GaussianAD: Gaussian-Centric End-to-End Autonomous Driving

> [Wenzhao Zheng](https://wzzheng.net/)\* $\dagger$, [Junjie Wu]()\*, [Yao Zheng]()\*, [Sicheng Zuo](https://github.com/zuosc19), [Zixun Xie](), [Longchao Yang](), [Yong Pan](), [Zhihui Hao](), [Peng Jia](),[XianPeng Lang](),[Shanghang Zhang](https://www.shanghangzhang.com/)

\* Equal contribution $\dagger$ Project leader

GaussianAD is a Gaussian-centric end-to-end framework which employs sparse yet comprehensive 3D Gaussians to pass information through the pipeline to efficiently preserve more details.

## News

- **[2024/12/16]** Paper released on [arXiv](https://arxiv.org/abs/).

## Demo

![demo](./assets/demo.gif)


## Overview

![overview](./assets/overview.png)


## Getting Started

### Installation
Follow instructions [HERE](docs/installation.md) to prepare the environment.
<!-- The environment is almost the same as [SelfOcc](https://github.com/huang-yh/SelfOcc) except for two additional CUDA operations.

```
1. Follow instructions in SelfOcc to prepare the environment. Not that we do not need packages related to NeRF, so feel safe to skip them.
2. cd model/encoder/gaussian_encoder/ops && pip install -e .  # deformable cross attention with image features
3. cd model/head/localagg && pip install -e .  # Gaussian-to-Voxel splatting
``` -->

### Data Preparation
1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

2. Download the occupancy annotations from SurroundOcc [HERE](https://github.com/weiyithu/SurroundOcc) and unzip it.

**Folder structure**
```
GaussianAD
├── ...
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
│   ├── nuscenes_cam/
│   │   ├── nuscenes_infos_train_gauddisn_ad.pkl
│   │   ├── nuscenes_infos_val_gauddisn_ad.pkl
│   ├── surroundocc/
│   │   ├── train_samples/
│   │   |   ├── xxxxxxxx.pcd.bin.npy
│   │   |   ├── ...
│   │   ├── val_samples/
│   │   |   ├── xxxxxxxx.pcd.bin.npy
│   │   |   ├── ...
```

### Train
Run the following command to launch your training process. 🚀

Download the pretrained weights for the image backbone [HERE](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) and put it inside ckpts, our models are trained on 32 A100
GPUs with a batch size of 8 for 20 epochs, the learning rate begins with 8e-4
```bash
python train.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600
```

For ddp training, you can use the following command:
```bash
bash ddp_train.sh --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600
```

### Inference
1. The [checkpoint](ckpts/state_dict.pth) that reproduces the result in Table.1 of our paper.

```
CUDA_VISIBLE_DEVICES=0 python test.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600/ --resume-from ckpts/state_dict.pth

```

Stay tuned for more exciting work and models!🤗

## Related Projects

Our code is based on the excellent work [GaussianFormer](https://github.com/huang-yh/GaussianFormer).

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{gaussianad,
    title={GaussianAD: Gaussian-Centric End-to-End Autonomous Driving},
    author={Wenzhao Zheng, Junjie Wu, Yao Zheng, Sicheng Zuo, Zixun Xie, Longchao Yang, Yong Pan, Zhihui Hao, Peng Jia, XianPeng Lang, Shanghang Zhang},
    journal={arXiv preprint arXiv: xxx},
    year={2024}
}
```
