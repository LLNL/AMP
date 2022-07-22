#!/bin/bash
logdir=Table2
mkdir $logdir
mkdir $logdir/cifar10
mkdir $logdir/cifar100

for iid in cifar10 cifar100; do
  for ood in isun lsun lsun_r places365 texture svhn; do
    sbatch driver.sh $iid $ood wideresnet reproducibility/$logdir
  done
done
