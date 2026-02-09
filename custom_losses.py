import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super(LDAMLoss, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = s
        self.weight = weight

    def forward(self, logits, targets, weight=None):
        margins = self.m_list[targets]
        
        logits = logits.clone()
        logits[range(len(targets)), targets] -= margins
        
        final_weight = weight if weight is not None else self.weight
        
        return F.cross_entropy(self.s * logits, targets, weight=final_weight, reduction="mean")
