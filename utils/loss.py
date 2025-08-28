"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理类别不平衡"""
    
    def __init__(self, gamma=2.0, alpha=None, temperature=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 温度缩放
        if self.temperature != 1.0:
            inputs = inputs / self.temperature
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 应用focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用alpha权重（如果提供）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """带类别权重的Focal Loss"""
    
    def __init__(self, class_weights, gamma=2.0, temperature=1.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 温度缩放
        if self.temperature != 1.0:
            inputs = inputs / self.temperature
        
        # 计算加权交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 应用focal weight
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss