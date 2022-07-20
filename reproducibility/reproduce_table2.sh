for iid in cifar10 cifar100; do
  for ood in isun lsun lsun_r places365 texture svhn; do
    sbatch driver.sh $iid $ood wideresnet Table2
  done
done
