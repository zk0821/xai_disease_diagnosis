import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cvxopt import matrix, spdiag, solvers


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        log_prob = F.log_softmax(x, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 - prob) ** self.gamma) * log_prob, y, weight=self.alpha, reduction=self.reduction)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, weight, max_m=0.5, s=30):
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


def LOW_compute_weights(lossgrad, lamb):

    device = lossgrad.get_device()
    lossgrad = lossgrad.data.cpu().numpy()

    # Compute Optimal sample Weights
    aux = -(lossgrad**2 + lamb)
    sz = len(lossgrad)
    P = 2 * matrix(lamb * np.identity(sz))
    q = matrix(aux.astype(np.double))
    A = spdiag(matrix(-1.0, (1, sz)))
    b = matrix(0.0, (sz, 1))
    Aeq = matrix(1.0, (1, sz))
    beq = matrix(1.0 * sz)
    solvers.options["show_progress"] = False
    solvers.options["maxiters"] = 20
    solvers.options["abstol"] = 1e-4
    solvers.options["reltol"] = 1e-4
    solvers.options["feastol"] = 1e-4
    sol = solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol["x"])

    return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))


class LOWLoss(torch.nn.Module):
    def __init__(self, weight, lamb=0.2):
        super(LOWLoss, self).__init__()
        self.lamb = lamb  # higher lamb means more smoothness -> weights closer to 1
        self.weight = weight

    def forward(self, logits, target):
        if logits.requires_grad:
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction="none")
            # Compute loss gradient norm
            output_d = logits.detach()
            loss_d = torch.mean(self.loss(output_d.requires_grad_(True), target), dim=0)
            loss_d.backward(torch.ones_like(loss_d))
            lossgrad = torch.norm(output_d.grad, 2, 1)

            # Computed weighted loss
            weights = LOW_compute_weights(lossgrad, self.lamb)
            loss = self.loss(logits, target)
            loss = torch.mean(torch.mul(loss, weights), dim=0)
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction="mean")
            loss = self.loss(logits, target)

        return loss
