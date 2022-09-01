### LSUN Resizing Benchmark
We provide a new benchmark to test OOD robustness to resizing artifacts. Each resizing method contains `10000` images from the original LSUN dataset resized using a different interpolation scheme.

### License
This is a modification of the original LSUN dataset [[original website]](https://www.yf.io/p/lsun), and can be downloaded from [[LSUN_SCENE_URL]](http://dl.yf.io/lsun/scenes/%s_%s_lmdb.zip). In particular, we modify the copy of LSUN obtained from the [[SCOOD github repository]](https://github.com/Jingkang50/ICCV21_SCOOD) as part of their code release.

We attribute the same license to this benchmark containing the processed LSUN as the original dataset:

```
@article{yu2015lsun,
  title={Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop},
  author={Yu, Fisher and Seff, Ari and Zhang, Yinda and Song, Shuran and Funkhouser, Thomas and Xiao, Jianxiong},
  journal={arXiv preprint arXiv:1506.03365},
  year={2015}
}
```

### Creating the dataset
We resize the original LSUN images to be the same size as CIFAR-10/100 datasets (32x32x3) using four resizing techniques: `nearest, bilinear, bicubic,lanczos` using the Pillow Imaging Library ([https://python-pillow.org/](https://python-pillow.org/)). We also provide the script in [`create_dataset.py`](create_dataset.py) that was used to create the dataset for future research. Note the script uses OpenCV version `3.4.16-dev` and PIL version `9.0.1`.
