#!/bin/bash
logdir=Table12
mkdir $logdir
mkdir $logdir/cifar10
mkdir $logdir/cifar100

for iid in cifar10 cifar100; do
  for ood in isun lsun lsun_r imagenet_r imagenet svhn; do
    sbatch driver.sh $iid $ood resnet34 reproducibility/$logdir
  done
done
