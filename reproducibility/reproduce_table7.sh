#!/bin/bash
logdir=Table7
mkdir $logdir
mkdir $logdir/cifar10
mkdir $logdir/cifar100

for iid in cifar10 cifar100; do
  for ood in resize_bicubic resize_lanczos resize_nearest resize_bilinear; do
    sbatch driver.sh $iid $ood resnet34 reproducibility/$logdir
  done
done
