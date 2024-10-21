# Latent Watermark (IEEE T-MM 2024)
[![arXiv](https://img.shields.io/badge/arXiv-2404.00230-b31b1b.svg)](https://arxiv.org/abs/2404.00230)

Latent Watermark: Inject and Detect Watermarks in Latent Diffusion Space

Authors: [Zheling Meng](https://richardsunnymeng.github.io/), Bo Peng, Jing Dong

New Laboratory of Pattern Recognition (NLPR), Institute of Automation, CAS

![Framework](./assets/fig2.png "The framework of Latent Watermark (a) and the progressive training strategy (b).")



## Setup

### Install packages

```bash
conda create -n LWenv python=3.7.16
conda activate LWenv
pip install -r ./requirements.txt
```

### Train the modules

0. Prepare for training

a. Training data
Please prepare the training data before start. You can download LAION-Aesthetics-5+ same as our paper, or use your own data. And write your data path in the key "data_json" in the config file.

b. Pretrained SD
Please download the pretrained v1.4 model into the folder "sd_ckpts" from [Stable Diffusion](https://github.com/CompVis/stable-diffusion). If you want to use other versions, modify the related configs in the config file as well.


1. Training stage 1
```bash
sh ./scripts/training_stage1.sh $cuda_device
```

2. Training stage 2
```bash
sh ./scripts/training_stage2.sh $cuda_device
```

3. Training stage 3
```bash
sh ./scripts/training_stage3.sh $cuda_device
```

### Generate images

1. Inject watermarks
```bash
sh ./scripts/inject.sh $cuda_device
```

2. Extract watermarks
```bash
sh ./scripts/extract.sh $cuda_device
```

### Evaluate the performance
Please refer to the repo [WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker).


# Acknowledgments
The code is built upon [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

# Reference
Please cite our paper if you use our models in your works:

```bibtex
@article{meng2024latent,
  title={Latent Watermark: Inject and Detect Watermarks in Latent Diffusion Space},
  author={Meng, Zheling and Peng, Bo and Dong, Jing},
  journal={arXiv preprint arXiv:2404.00230},
  year={2024}
}