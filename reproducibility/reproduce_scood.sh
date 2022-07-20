#!/bin/bash
logdir=SCOOD
mkdir $logdir/cifar10
mkdir $logdir/cifar100

for iid in cifar10 cifar100; do
  for ood in texture svhn tin lsun places365; do
    sbatch driver.sh $iid scood_$ood resnet18 reproducibility/$logdir
  done
done
