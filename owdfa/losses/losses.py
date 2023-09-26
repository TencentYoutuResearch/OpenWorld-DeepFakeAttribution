#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    eps = 1e-8
    x_ = torch.clamp(x, min=eps)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        return - b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


def symmetric_mse_loss(input1, input2):
    """
    Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.sum((input1 - input2)**2)
