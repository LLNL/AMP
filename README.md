# AMP
Code and models for _Out of Distribution Detection with Neural Network Anchoring_
<img src=figs/teaser.png width="800">

### Dependencies
This package was built and tested using
* Python `3.7.9`
* Pytorch `1.11.0`
* Torchvision `0.12.0`
* Numpy `1.19.2`
For logging and Config files we use yaml (`5.4.1`) and logging (`0.5.1.2`).  
All of these can be installed (recommend a custom environment) using `pip install -r requirements.txt`.


### Checkpoints and pre-trained models
Pre-trained (cifar10/100: ResNet34, WRN) to reproduce experiments from the paper can be downloaded from the [Google Drive Link](https://drive.google.com/drive/folders/1Pdh693qjUsF_BUtfIQtKpV-QNVyVEA_H). The code assumes checkpoints are placed as follows: `ckpt_save/in_dataset/modeltype_seed/model_name` so for example, `ckpts/cifar100/WideResNet_seed_1/ckpt-199.pth`. The tarball containing checkpoints already preserves this directory structure, and its location must be specified in the `config.yml` before evaluating.

### LSUN Resizing Benchmark
We provide a new benchmark to test OOD robutness to resizing artifacts. This can be found in  [`resize_ood/resize_benchmark.tar.gz`](resize_ood/). To use it, extract the dataset from the tarball and point to them in the `config.yml` file, before executing the `main.py`.
<img src=figs/table7.png width="750">

### Reproducibility
We provide easy bash scripts to reproduce different tables/figures in the paper. These can be found and executed in [`reproducibility/`](reproducibility/). These depend on the pre-trained checkpoints provided, so they must first be downloaded. We also provide a separate config file with the exact settings used for our experiments.

### Citation
If you use this code, please consider citing our paper as follows:
```
@article{anirudh2022out,
  title={Out of Distribution Detection via Neural Network Anchoring},
  author={Anirudh, Rushil and Thiagarajan, Jayaraman J},
  journal={arXiv preprint arXiv:2207.04125},
  year={2022}
}

```
### License
This code is distributed under the terms of the GPL-2.0 license. All new contributions must be made under this license.
LLNL-CODE-838619
SPDX-License-Identifier: GPL-2.0
