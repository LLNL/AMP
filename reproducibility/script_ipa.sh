for iid in cifar10 cifar100; do
  for ood in isun lsun lsun_r imagenet_r imagenet svhn; do
    sbatch Reproduce_Table_2.sh $iid $ood resnet34 reproducibility/Table12
  done
done
