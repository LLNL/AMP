'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from PIL import Image
import natsort
import random
# from scood_data import utils as scood

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
])


class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, corruption='gaussian_blur', transform=None,level=0):
        numpy_path = f'../CIFAR/pytorch-cifar/data/CIFAR-10-c/{corruption}.npy'
        t = 10000
        self.transform = transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:]
        self.data = self.data_[:t,:,:,:]
        self.targets_ = np.load('../CIFAR/pytorch-cifar/data/CIFAR-10-c/labels.npy')
        self.targets = self.targets_[:t]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image)
        targets = self.targets[idx]
        return image, targets

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def get_training_dataloader(transforms,batch_size=128, num_workers=2, shuffle=True,root = None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    cifar100_training = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transforms)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(transforms,batch_size=512, num_workers=2, shuffle=True,root = None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transforms)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader



def get_ood_dataloader(transforms,datapath,batch_size=128):
    ood_dataset = torchvision.datasets.ImageFolder(root=datapath, transform=transforms)
    ood_data_loader = torch.utils.data.DataLoader(ood_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  pin_memory=True)

    return ood_data_loader

def get_ood_dataloader_v2(transforms,datapath,batch_size=128):
    ood_dataset = CustomDataSet(main_dir=datapath, transform=transforms)
    ood_data_loader = torch.utils.data.DataLoader(ood_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  pin_memory=True)

    return ood_data_loader

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image,torch.tensor(0)

def fetch_dataloaders(args,cfg):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    corruption = args.corruption
    clevel = args.clevel

    n_batch = cfg['data']['batch_size']
    n_tr_batch = cfg['data']['tr_batch_size']
    n_workers = cfg['data']['n_workers']
    if in_dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root = cfg['data']['in_dataset']['cifar10'], train=True, download=False, transform=data_transforms)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=n_tr_batch, shuffle=True, num_workers=n_workers)

        testset = torchvision.datasets.CIFAR10(
            root = cfg['data']['in_dataset']['cifar10'], train=False, download=False, transform=data_transforms)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=n_batch, shuffle=False, num_workers=n_workers)

    elif in_dataset=='cifar100':
        trainloader = get_training_dataloader(data_transforms,root=cfg['data']['in_dataset']['cifar100'],batch_size=n_tr_batch)
        testloader = get_test_dataloader(data_transforms,root=cfg['data']['in_dataset']['cifar100'],batch_size=n_batch)

    if out_dataset in ['imagenet_r','imagenet','svhn','lsun_r','lsun','texture','places365','isun']:
        datapath = cfg['data']['ood_benchmark'][out_dataset]
        if out_dataset =='svhn':
            ood_dataset = torchvision.datasets.SVHN(datapath, split="test", transform=data_transforms, download=False)
            ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=n_batch, shuffle=False, num_workers=n_workers)
        else:
            ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)

    elif 'resize' in out_dataset:
        alg = out_dataset.split('_')[-1]
        datapath = cfg['data']['resizing_benchmark'][alg]
        ood_loader = get_ood_dataloader_v2(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='lsun_native':
        datapath = cfg['data']['resizing_benchmark'][out_dataset]
        ood_loader = get_ood_dataloader_v2(data_transforms,datapath,batch_size=n_batch)

    elif out_dataset=='cifar100':
        ood_loader = get_test_dataloader(data_transforms,root=cfg['data']['in_dataset']['cifar100'],batch_size=n_batch)
    elif out_dataset=='cifar10':
        ood_dataset = torchvision.datasets.CIFAR10(
            root = cfg['data']['in_dataset']['cifar10'], train=False, download=False, transform=data_transforms)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=n_batch, shuffle=False, num_workers=n_workers)
    elif out_dataset=='cifar10c':
        ood_dataset = CIFAR10C(corruption=corruption,level=clevel,transform=data_transforms)
        ood_loader = torch.utils.data.DataLoader(ood_dataset,shuffle=False,batch_size=n_batch,num_workers=n_workers)

    elif 'scood' in out_dataset:
        if in_dataset=='cifar10':
            num_classes=10
        else:
            num_classes=100

        get_dataloader_default = partial(scood.get_dataloader,
            root_dir=cfg['data']['scood_benchmark']['root'],
            benchmark=in_dataset,
            num_classes=num_classes)
        ood_loader = get_dataloader_default(name=out_dataset.split('_')[1],stage="test",
           batch_size=n_batch,
           shuffle=False,
           num_workers=n_workers,)
        trainloader = get_dataloader_default(name=in_dataset,stage="train",
          batch_size=n_tr_batch,
          shuffle=False,
          num_workers=n_workers,)

        testloader = get_dataloader_default(name=in_dataset,stage="test",
         batch_size=n_batch,
         shuffle=False,
         num_workers=n_workers,)
    return trainloader, testloader, ood_loader
