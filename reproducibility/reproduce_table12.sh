#!/bin/bash
for iid in cifar10 cifar100; do
  for ood in isun lsun lsun_r imagenet_r imagenet svhn; do
    sbatch driver.sh $iid $ood resnet34 reproducibility/Table12
  done
done
