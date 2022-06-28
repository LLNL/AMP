import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import SVHN


import torchvision
import torchvision.transforms as transforms

import numpy as np
import glob
from functools import partial

from models import *
from models.resnetv2 import resnet20
from scood_data import utils as scood

from AnchorTraining import ANT
from sklearn.metrics import roc_auc_score
from utils import get_test_dataloader, get_training_dataloader, get_ood_dataloader, get_ood_dataloader_v2, CIFAR10C
from display_results import get_measures, get_measures_v2
from calculate_log import compute_metric
import pickle as pkl

# def combine_mu_sigma(mu,y):
    # c = torch.sigmoid(-y)
    # return torch.mul(mu,c)

def combine_mu_sigma(mu,y):
    c = torch.mean(y,1)
    c = c.unsqueeze(1).expand(mu.shape)
    return torch.div(mu,1+torch.exp(c))


def entropy(prob):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)


def ood_score(mc_preds,score_type='energy'):
    to_np = lambda x: x.data.cpu().numpy()

    if score_type=='ent':
        return entropy(to_np(F.softmax(mc_preds,1).detach()))
    elif score_type=='xent':
        return to_np((mc_preds.mean(1) - torch.logsumexp(mc_preds, dim=1)))
    elif score_type=='energy':
        return -to_np((1*torch.logsumexp(mc_preds / 1, dim=1)))
    elif score_type=='msp':
        smax = to_np(F.softmax(mc_preds, dim=1))
        return -np.max(smax, axis=1)

def fetch_dataloaders(data_transforms,in_dataset='cifar10',out_dataset='svhn',corruption=None,clevel=None):
    n_batch = 256
    if in_dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root = '../../CIFAR/pytorch-cifar/data/', train=True, download=False, transform=data_transforms)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=n_batch, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root = '../../CIFAR/pytorch-cifar/data/', train=False, download=False, transform=data_transforms)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=n_batch, shuffle=False, num_workers=2)

    elif in_dataset=='cifar100':
        trainloader = get_training_dataloader(data_transforms,batch_size=n_batch)
        testloader = get_test_dataloader(data_transforms,batch_size=n_batch)

    if out_dataset=='svhn':
        ood_dataset = torchvision.datasets.SVHN("../CIFAR/pytorch-cifar/data/SVHN-dataset", split="test", transform=data_transforms, download=False)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=n_batch, shuffle=False, num_workers=2)

    elif out_dataset=='imagenet_r':
        datapath = './odin/data/Imagenet_resize/'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='imagenet':
        datapath = './odin/data/Imagenet'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='lsun_r':
        datapath = './odin/data/LSUN_resize/'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif 'rcv2' in out_dataset:
        alg = out_dataset.split('_')[-1]
        datapath = f'/usr/workspace/anirudh1/delUQ/LSUN_resize/cv2/{alg}'
        ood_loader = get_ood_dataloader_v2(data_transforms,datapath,batch_size=n_batch)
    elif 'rpil' in out_dataset:
        alg = out_dataset.split('_')[-1]
        datapath = f'/usr/workspace/anirudh1/delUQ/LSUN_resize/pil/{alg}'
        ood_loader = get_ood_dataloader_v2(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='lsun_native':
        datapath = '/usr/workspace/anirudh1/delUQ/data/images/lsun'
        ood_loader = get_ood_dataloader_v2(data_transforms,datapath,batch_size=n_batch)

    elif out_dataset =='texture':
        datapath = '/usr/workspace/anirudh1/delUQ/data/images/texture'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset == 'places365':
        datapath = '/usr/workspace/anirudh1/delUQ/data/images/places365'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='lsun':
        datapath = './odin/data/LSUN/'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='isun':
        datapath = './odin/data/iSUN/'
        ood_loader = get_ood_dataloader(data_transforms,datapath,batch_size=n_batch)
    elif out_dataset=='cifar100':
        ood_loader = get_test_dataloader(data_transforms,batch_size=n_batch)
    elif out_dataset=='cifar10':
        ood_dataset = torchvision.datasets.CIFAR10(
            root = '../../CIFAR/pytorch-cifar/data/', train=False, download=False, transform=data_transforms)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=n_batch, shuffle=False, num_workers=2)
    elif out_dataset=='cifar10c':
        ood_dataset = CIFAR10C(corruption=corruption,level=clevel,transform=data_transforms)
        ood_loader = torch.utils.data.DataLoader(ood_dataset,shuffle=False,batch_size=n_batch,num_workers=2)

    elif 'scood' in out_dataset:
        if in_dataset=='cifar10':
            num_classes=10
        else:
            num_classes=100

        get_dataloader_default = partial(scood.get_dataloader,
            root_dir='/usr/workspace/anirudh1/delUQ/data/',
            benchmark=in_dataset,
            num_classes=num_classes)
        ood_loader = get_dataloader_default(name=out_dataset.split('_')[1],stage="test",
           batch_size=n_batch,
           shuffle=False,
           num_workers=2,)
        trainloader = get_dataloader_default(name=in_dataset,stage="train",
          batch_size=n_batch,
          shuffle=False,
          num_workers=2,)

        testloader = get_dataloader_default(name=in_dataset,stage="test",
         batch_size=n_batch,
         shuffle=False,
         num_workers=2,)
    return trainloader, testloader, ood_loader


def run_ood(args,data_transforms):
    modeltype = args.nn
    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    score_type = args.score_type
    seed = args.seed
    type1 = args.type1
    nref = args.nref

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    in_dataloader, in_test_dataloader, out_dataloader = fetch_dataloaders(data_transforms,in_dataset,out_dataset,args.corruption,args.clevel)

    if in_dataset=='cifar10':
        nclass=10
    elif in_dataset=='cifar100':
        nclass=100

    if modeltype=='resnet34':
        # net = ResNet34(nc=6,num_classes=nclass)
        if 'scood' in out_dataset:
            modelname = 'SCOOD_ResNet34'
        else:
            modelname = 'ResNet34'
    elif modeltype=='resnet18':
        # net = ResNet18(nc=6,num_classes=nclass)
        if 'scood' in out_dataset:
            modelname = 'SCOOD_ResNet18'
        else:
            modelname = 'ResNet18'
    elif modeltype=='resnet20':
        # net = resnet20(nc=6,num_classes=nclass)
        modelname = 'ResNet20'
    elif modeltype=='wideresnet':
        # net = WideResNet(nc=6,num_classes=nclass)
        modelname = 'WideResNet'
    elif modeltype=='resnet18v2':
        modelname = 'ResNet18'
        # net = delUQNetv2(modeltype,nc=6,num_classes = nclass)
    elif modeltype=='wideresnetv2':
        modelname = 'WideResNet'
        # net = delUQNetv2(nc=6,num_classes = nclass)
    elif modeltype=='wideresnet28':
        modelname = 'SCOOD_WideResNet28'
        # net = WideResNet28(nc_in=6,num_classes=num_classes)
    else:
        print('************ WARNING modeltype not understood, using resnet20 ************')
        # net = resnet20(nc=6,num_classes=nclass)
        modelname = 'ResNet20'

    modelname = modelname + f'_seed_{seed}'

    if 'scood' in out_dataset:
        sub = 'seed_*'
    # elif modeltype =='resnet18':
        # sub = 'seed_*_level3_*'
    elif args.baseline:
        sub = 'seed_*_baseline_*'
    else:
        sub = 'seed_*_txs5_ANT_API_*'
        # sub = 'repulsivev2_1e-2_uniform_*'

    ckpt_files = glob.glob(f'./chkpts/{in_dataset}/{modelname}/{sub}/ckpt-{199}.pth')
    modelpath = ckpt_files[-1]


    base_net = ResNet34(nc=6,num_classes=nclass)
    delta_model = ANT(base_network=base_net)
    delta_model = torch.nn.DataParallel(delta_model)
    checkpoint = torch.load(modelpath)
    delta_model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {modelpath} with test accuracy (with 1 anchor) --- {best_acc:.4f}')

    delta_model.eval()
    delta_model.cuda()
    print(torch.cuda.memory_allocated()/1024**2)

    pytorch_total_params = np.sum(p.numel() for p in delta_model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {pytorch_total_params}')

    preds_cifar= []
    uncs_cifar= []
    sclabels = []
    labels = []

    total=0
    correct=0
    refs_ = next(iter(in_dataloader))
    T = []
    with torch.no_grad():
        for iid_batch in in_test_dataloader:
            if 'scood' not in out_dataset:
                inputs, targets = iid_batch
                refs,_ = refs_
            else:
                inputs, targets = iid_batch['data'], iid_batch['label']
                refs = refs_['data']
                sclabels.append(iid_batch['sc_label'])
                # labels.append(iid_batch['label'])
            #
            refs = refs.to(device)
            inputs, targets = inputs.to(device), targets.to(device)

            mu0, std0 = delta_model(inputs,anchors=refs,n_anchors=nref,return_std=True)
            T.append(targets)
            preds_cifar.append(mu0)
            uncs_cifar.append(std0)

    mean_preds = torch.cat(preds_cifar,0)
    all_targets = torch.cat(T,0)
    _, predicted = mean_preds.max(1)
    correct = predicted.eq(all_targets).sum().item()/all_targets.shape[0]
    print(f'CIFAR-10 Test Accuracy with {nref} anchors: {100*correct:.3f}')
    uncs = torch.cat(uncs_cifar,0)

    # mu0_ = combine_mu_sigma(mean_preds,uncs)
    mu0_ = mean_preds
    if 'scood' in out_dataset:
        sclabel0 = np.where(torch.cat(sclabels,0)>=0,1,0)

    uncs_cifar0 = ood_score(mean_preds,score_type=score_type)
    uncs_cifar1 = ood_score(mu0_,score_type=score_type)
    uncs_cifar2 = torch.sum(uncs,1).cpu().numpy()
    if torch.sum(uncs,1).isnan().any():
        print('!!Uncertainty tensor has NaNs!!')
    #
    preds_svhn = []
    uncs_svhn = []
    sclabels_svhn = []
    with torch.no_grad():

        for ood_batch in out_dataloader:
            refs_ = next(iter(in_dataloader))
            if 'scood' not in out_dataset:
                svhn_batch, svhn_targets = ood_batch
                refs,_ = refs_

            else:
                svhn_batch, svhn_targets = ood_batch['data'], ood_batch['label']
                refs = refs_['data']
                sclabels_svhn.append(ood_batch['sc_label'])

            refs = refs.to(device)
            svhn_batch, svhn_targets = svhn_batch.to(device), svhn_targets.to(device)

            mu1, std1 = delta_model(svhn_batch,anchors=refs,n_anchors=nref,return_std=True)

            preds_svhn.append(mu1)
            uncs_svhn.append(std1)

    mean_preds_svhn = torch.cat(preds_svhn,0)
    uncs_ = torch.cat(uncs_svhn,0)
    # mu1_ = combine_mu_sigma(mean_preds_svhn,uncs_)
    mu1_ = mean_preds_svhn
    if 'scood' in out_dataset:
        sclabel1 = np.where(torch.cat(sclabels_svhn,0)>=0,1,0)
        print(f'IID: {np.sum(sclabel0)}, SC_IID: {np.sum(sclabel1)}')

    uncs_svhn0 = ood_score(mean_preds_svhn,score_type=score_type)
    uncs_svhn1 = ood_score(mu1_,score_type=score_type)
    uncs_svhn2 = torch.sum(uncs_,1).cpu().numpy()

    if torch.sum(uncs_,1).isnan().any():
        print('!!Uncertainty tensor has NaNs!!')

    y_pred0 = np.concatenate([uncs_cifar0,uncs_svhn0],0)
    y_pred1 = np.concatenate([uncs_cifar1,uncs_svhn1],0)
    y_pred2 = np.concatenate([uncs_cifar2,uncs_svhn2],0)

    if not 'scood' in out_dataset:
        y_true = np.concatenate([np.ones_like(uncs_cifar1),np.zeros_like(uncs_svhn1)])

        auroc, aupr, fpr = [],[],[]
        scores = [y_pred0,y_pred1,y_pred2]
        for i in range(3):
            _auroc,_aupr,_fpr = get_measures(y_true,-scores[i])
            # auroc.append(_auroc)
            aupr.append(_aupr)
            fpr.append(_fpr)


        dtac, aupr_in, aupr_out, tnr = [],[],[], []
        iids = [uncs_cifar0, uncs_cifar1, uncs_cifar2]
        oods = [uncs_svhn0, uncs_svhn1, uncs_svhn2]
        for i in range(3):
            results  = compute_metric(-iids[i][:], -oods[i][:])
            dtac.append(results["DTACC"])
            tnr.append(results["TNR"])
            auroc.append(results["AUROC"])
            aupr_in.append(results["AUIN"])
            aupr_out.append(results["AUOUT"])

        if type1:
            return auroc,aupr_in, fpr,aupr_out
        else:
            return tnr, auroc,dtac

    elif 'scood'in out_dataset:
        y_true = np.concatenate([sclabel0,sclabel1])

        auroc, aupr, fpr = [],[],[]
        scores = [y_pred0,y_pred1,y_pred2]
        for i in range(3):
            _auroc,_aupr,_fpr = get_measures(y_true,-scores[i])
            # auroc.append(_auroc)
            aupr.append(_aupr)
            fpr.append(_fpr)


        dtac, aupr_in, aupr_out, auroc = [],[],[], []
        iids = [uncs_cifar0, uncs_cifar1, uncs_cifar2]
        oods = [uncs_svhn0, uncs_svhn1, uncs_svhn2]
        for i in range(3):
            scood_iid = np.concatenate([iids[i],oods[i][np.where(sclabel1==1)]],0)
            scood_ood = oods[i][np.where(sclabel1==0)]
            print(scood_iid.shape, scood_ood.shape)
            results  = compute_metric(-scood_iid, -scood_ood)
            dtac.append(results["DTACC"])
            # fpr.append(1-results["TNR"])
            auroc.append(results["AUROC"])
            aupr_in.append(results["AUIN"])
            aupr_out.append(results["AUOUT"])
            # fpr.append(results["FPR"])

        if type1:
            return auroc,aupr_in, fpr,aupr_out
        else:
            return auroc,aupr,fpr
        #
