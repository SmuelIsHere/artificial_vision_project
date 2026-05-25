import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class BinaryAsymmetricLoss(nn.Module):
    # https://arxiv.org/abs/2009.14119

    def __init__(self, gamma_neg: float, gamma_pos: float, eps: float = 1e-8, ignore_index=None, device=None) -> None:

        super(BinaryAsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg if isinstance(gamma_neg, torch.Tensor) else torch.tensor(gamma_neg, dtype=torch.float32)
        self.gamma_pos = gamma_pos if isinstance(gamma_pos, torch.Tensor) else torch.tensor(gamma_pos, dtype=torch.float32)
        self.eps = eps
        self.ignore_index = ignore_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma_neg = self.gamma_neg.to(self.device)
        self.gamma_pos = self.gamma_pos.to(self.device)

    def forward(self, prob: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        targets = targets.float().to(self.device)

        if self.ignore_index is not None:

            mask = torch.ones_like(targets, dtype=torch.bool)
            mask[targets == self.ignore_index] = 0
            targets = targets[mask]
            prob = prob[mask]

        prob = prob.squeeze(1)
        weight = torch.where(targets == 0, self.gamma_neg, self.gamma_pos).to(self.device)
        loss = F.binary_cross_entropy(prob, targets, weight=weight, reduction='mean')
        return loss
