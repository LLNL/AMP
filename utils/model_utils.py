import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import MODEL_DICT
__all__=['fetch_model','display_log']

def fetch_model(args):
    modeltype = args.nn

    if modeltype not in MODEL_DICT:
        raise ValueError('************ modeltype not understood ************')

    if args.in_dataset=='cifar10':
        nclass=10
    elif args.in_dataset=='cifar100':
        nclass=100
    modelname, model = MODEL_DICT[modeltype]
    if 'scood' in args.out_dataset:
        modelname = 'SCOOD_'+modelname
    modelname = modelname + f'_seed_{args.seed}'
    net = model(nc=6,num_classes=nclass)
    return modelname, net

def display_log(**kwargs):
    args = kwargs['args']
    logger = kwargs['logger']
    fpr_ = kwargs['FPR']
    auroc_ = kwargs['AUROC']
    dtac_ = kwargs['DTAC']
    aupr_in_ = kwargs['AUPR_IN']
    aupr_out_ =  kwargs['AUPR_OUT']
    cfg = kwargs['cfg']

    logger.info("\n")
    logger.info("{:31}{:>22}".format("Neural network architecture:", args.nn))
    logger.info("{:31}{:>22}".format("In-distribution dataset:", args.in_dataset))
    logger.info("{:31}{:>22}\n".format("Number of Anchors:", args.nref))

    if args.out_dataset == 'cifar10c':
        logger.info("{:31}{:>22}{:>22}{}{:>22}{}".format(f"Out-of-distribution dataset:",args.out_dataset,"corruption:",args.corruption,"intensity:",args.clevel))
    else:
        logger.info("{:31}{:>22}".format("Out-of-distribution dataset:", args.out_dataset))

    logger.info(f"OOD Detection Performance averaged across seeds: {cfg['models']['seeds']}")
    logger.info("{:>34}{:>13}{:>13}{:>15}{:>16}".format("FPR95","AUROC","DTACC","AUPR-IN","AUPR-OUT"))
    logger.info("{:20}{:13.2f}%{:>13.2f}%{:>13.2f}%{:>13.2f}%{:>13.2f}".format("AMP:",100-fpr_*100,auroc_*100,dtac_*100,100*aupr_in_, 100*aupr_out_))

    return
