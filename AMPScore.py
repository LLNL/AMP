import torch
import numpy as np
import glob
import pickle as pkl
import random

from utils import *
from AnchoringModel import ANT

def run_ood(**kwargs):
    args = kwargs['args']
    logger = kwargs['logger']
    cfg = kwargs['cfg']

    modeltype = args.nn
    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    score_type = args.score_type
    seed = args.seed
    nref = args.nref

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    in_dataloader, in_test_dataloader, out_dataloader = fetch_dataloaders(args,cfg)
    modelname, net = fetch_model(args)

    ckpt_folder = cfg['models']['ckpt_save']
    model_name = cfg['models']['model_name']


    ckpt_path = f'./{ckpt_folder}/{in_dataset}/{modelname}/{model_name}'
    ckpt_files = glob.glob(ckpt_path)
    logger.debug(f'Reading model weights from {ckpt_path}')

    modelpath = ckpt_files[-1]
    checkpoint = torch.load(modelpath)

    delta_model = ANT(base_network=net)
    delta_model.net = torch.nn.DataParallel(delta_model.net)
    delta_model.net.load_state_dict(checkpoint['net'])

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    logger.debug(f'Checkpoint loaded from {modelpath} with test accuracy (with 1 anchor) --- {best_acc:.4f}')

    delta_model.eval()
    delta_model.cuda()
    logger.debug(f"Size of Model {torch.cuda.memory_allocated()/1024**2:.3f}MB")

    pytorch_total_params = np.sum(p.numel() for p in delta_model.parameters() if p.requires_grad)
    logger.debug(f'Total Number of Trainable Parameters: {pytorch_total_params}')

    preds_cifar= []
    uncs_cifar= []
    sclabels = []
    labels = []

    total=0
    correct=0
    T = []
    with torch.no_grad():
        for iid_batch in in_test_dataloader:
            refs_ = next(iter(in_dataloader))

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
    logger.info(f'seed {seed}-{in_dataset}: test accuracy with {nref} anchors: {100*correct:.3f}')
    uncs = torch.cat(uncs_cifar,0)

    mu0_ = mean_preds
    if 'scood' in out_dataset:
        sclabel0 = np.where(torch.cat(sclabels,0)>=0,1,0)

    uncs_cifar0 = ood_score(mean_preds,score_type=score_type)
    uncs_cifar1 = ood_score(mu0_,score_type=score_type)
    uncs_cifar2 = torch.sum(uncs,1).cpu().numpy()
    if torch.sum(uncs,1).isnan().any():
        logger.warning('!!Uncertainty tensor has NaNs!!')
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

    mu1_ = mean_preds_svhn
    if 'scood' in out_dataset:
        sclabel1 = np.where(torch.cat(sclabels_svhn,0)>=0,1,0)
        logger.debug(f'IID: {np.sum(sclabel0)}, SC_IID: {np.sum(sclabel1)}')

    uncs_svhn0 = ood_score(mean_preds_svhn,score_type=score_type)
    uncs_svhn1 = ood_score(mu1_,score_type=score_type)
    uncs_svhn2 = torch.sum(uncs_,1).cpu().numpy()

    if torch.sum(uncs_,1).isnan().any():
        logger.warning('!!Uncertainty tensor has NaNs!!')

    y_pred0 = np.concatenate([uncs_cifar0,uncs_svhn0],0)
    y_pred1 = np.concatenate([uncs_cifar1,uncs_svhn1],0)
    y_pred2 = np.concatenate([uncs_cifar2,uncs_svhn2],0)

    if not 'scood' in out_dataset:
        y_true = np.concatenate([np.ones_like(uncs_cifar1),np.zeros_like(uncs_svhn1)])

        iids = [uncs_cifar0, uncs_cifar1, uncs_cifar2]
        oods = [uncs_svhn0, uncs_svhn1, uncs_svhn2]
        results  = compute_metric(-iids[1][:], -oods[1][:])

        return results

    elif 'scood'in out_dataset:
        y_true = np.concatenate([sclabel0,sclabel1])

        iids = [uncs_cifar0, uncs_cifar1, uncs_cifar2]
        oods = [uncs_svhn0, uncs_svhn1, uncs_svhn2]

        scood_iid = np.concatenate([iids[1],oods[1][np.where(sclabel1==1)]],0)
        scood_ood = oods[1][np.where(sclabel1==0)]
        logger.debug(scood_iid.shape, scood_ood.shape)
        results  = compute_metric(-scood_iid, -scood_ood)

        return results
