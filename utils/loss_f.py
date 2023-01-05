import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def BCEDICE_loss(target, true):
    bce = torch.nn.BCELoss()
    bce_loss = bce(target, true)

    true_u = true.unsqueeze(1)
    target_u = target.unsqueeze(1)

    inter = (true * target).sum()
    eps = 1e-7
    dice_loss = (2 * inter + eps) / (true.sum() + target.sum() + eps)

    return bce_loss + 1 - dice_loss







