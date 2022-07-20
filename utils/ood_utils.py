import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
