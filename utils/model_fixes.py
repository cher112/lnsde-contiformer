"""
修复LinearNoiseSDEContiformer，确保只返回logits
"""

import torch
import torch.nn as nn
from typing import Optional


class FixedLinearNoiseSDEContiformer(nn.Module):
    """修复版本：确保forward只返回logits tensor"""
    
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        
    def forward(self, time_series: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        只返回logits，不返回tuple
        Args:
            time_series: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
        Returns:
            logits: (batch, num_classes) - 纯tensor
        """
        # 调用原始模型
        result = self.model(time_series, mask)
        
        # 确保只返回logits
        if isinstance(result, tuple):
            logits = result[0]  # 取第一个元素（logits）
        else:
            logits = result
            
        return logits


def patch_model_forward(model_class):
    """猴子补丁：修改模型的forward方法"""
    original_forward = model_class.forward
    
    def new_forward(self, time_series, mask=None, return_features=False):
        # 调用原始forward
        result = original_forward(self, time_series, mask, return_stability_info=False)
        
        # 处理返回值
        if isinstance(result, tuple):
            logits = result[0]
            features = result[1] if len(result) > 1 else None
        else:
            logits = result
            features = None
        
        # 根据需要返回
        if return_features and features is not None:
            return logits, features
        else:
            return logits
    
    model_class.forward = new_forward
    return model_class


# 导出
__all__ = ['FixedLinearNoiseSDEContiformer', 'patch_model_forward']