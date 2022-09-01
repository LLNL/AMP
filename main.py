# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: GPL-2.0
import argparse
import os
import time
import datetime

import torch
import numpy as np
import random

import yaml
import logging

from lib.utils import display_log
import lib.AMPScore as ood



parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--nn', default="wideresnet", type=str,
                    help='neural network name and training set')
parser.add_argument('--in_dataset', default="cifar10", type=str,
                    help='in-distribution dataset')
parser.add_argument('--out_dataset', default="svhn", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--score_type', default="xent", type=str,
                    help='Type of OOD Score ent/xent/energy')
parser.add_argument('--seed', default=1, type=int,
                    help='model training seed')
parser.add_argument('--nref', default=5, type=int,
                    help='number of anchors at test time')
parser.add_argument('--clevel', default=0, type=int,help='corruption level')
parser.add_argument('--corruption', default='gaussian_blur', type=str,help='corruption')

parser.add_argument('--cfg_path', default='./config.yml', type=str,help='path to config file')
parser.add_argument('--log_path', default='./logs', type=str,help='path to logs')
parser.add_argument('--debug', action="store_true",help='determines logger info level')

parser.add_argument('--baseline', action='store_true',default=False,
                    help='model training seed')

parser.set_defaults(argument=True)

def main(**kwargs):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    results = ood.run_ood(**kwargs)

    return results

if __name__ == '__main__':

    args = parser.parse_args()
    cfg_path = args.cfg_path
    log_path = args.log_path
    log_path = f'{log_path}/{args.in_dataset}/{args.nn}/'
    logfile = f'{log_path}/{args.out_dataset}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'

    if not os.path.exists(log_path):
        os.makedirs(log_path,exist_ok = True)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    logging.basicConfig(level=loglevel,filename=logfile, filemode='w', format='%(levelname)s - %(message)s')
    logger = logging.getLogger()

    if not os.path.exists(cfg_path):
        logger.error("Missing config file")

    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    seeds = cfg['models']['seeds']
    start = time.time()
    AUROC = []
    AUPR_IN = []
    AUPR_OUT = []
    DTAC = []
    FPR = []

    for s in seeds:
        args.seed = s
        results = main(args=args, cfg=cfg, logger=logger)

        AUROC.append(results['AUROC'])
        AUPR_IN.append(results['AUIN'])
        AUPR_OUT.append(results['AUOUT'])
        DTAC.append(results['DTACC'])
        FPR.append(results['TNR'])

    fpr = np.mean(FPR)
    auroc = np.mean(AUROC)
    aupr_in = np.mean(AUPR_IN)
    aupr_out = np.mean(AUPR_OUT)
    dtac = np.mean(DTAC)

    display_log(args=args,logger=logger,cfg=cfg,
                FPR = fpr,
                AUROC = auroc,
                AUPR_IN = aupr_in,
                AUPR_OUT = aupr_out,
                DTAC = dtac)

    stop = time.time()
    logger.info(f'Time Elapsed:{stop-start:.3f} seconds')
