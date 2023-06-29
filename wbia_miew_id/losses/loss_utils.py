from .focal_loss import FocalLoss
from torch import nn

def fetch_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
    elif name == 'focal_loss':
        loss = FocalLoss()
    else: 
        raise NotImplementedError

    return loss

