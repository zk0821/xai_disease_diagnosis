import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ClassBalancedCrossEntropy(nn.Module):

    def __init__(self, beta=0.999, samples_per_class=None, use_weights=True):
        super(ClassBalancedCrossEntropy, self).__init__()
        self.beta = beta
        self.samples_per_class = samples_per_class
        self.use_weights = use_weights

    def forward(self, x, y):
        batch_size = x.size(0)
        num_classes = x.size(1)
        labels_one_hot = F.one_hot(y, num_classes).float()

        if self.use_weights:
            # class balance
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights /= np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=x.device).float()
        else:
            weights = None

        # cross entropy
        ce_loss = F.cross_entropy(x, y, reduction="mean", weight=weights)

        return ce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        log_prob = F.log_softmax(x, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            y,
            weight=self.alpha,
            reduction = self.reduction
        )


class ClassBalancedFocalLoss(nn.Module):

    def __init__(self, beta=0.999, gamma=2, samples_per_class=None, reduction="mean", use_weights=True):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        self.reduction = reduction
        self.use_weights = use_weights

    def forward(self, x, y):
        # class balance
        if self.use_weights:
            num_classes = x.size(1)
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights /= np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=x.device).float()
        else:
            weights = None
        # focal loss
        log_prob = F.log_softmax(x, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            y,
            weight=weights,
            reduction = self.reduction
        )


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
