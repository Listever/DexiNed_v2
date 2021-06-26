import torch
import torch.nn as nn

def bdcn_loss(inputs, targets, l_weight = 1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs, targets.float())
    return l_weight * cost

def get_loss(model):
    if model['g_loss'] == 'bdcn_loss':
        g_loss = bdcn_loss
    else:
        raise ValueError("GAN Loss [%s] not recognized." % model['g_loss'])
    return g_loss
