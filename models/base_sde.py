"""
Base SDE Model class for all Neural SDE architectures
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/root/autodl-tmp/torchsde')
sys.path.append('/root/autodl-tmp/PhysioPro')
import torchsde
from abc import ABC, abstractmethod


class BaseSDEModel(nn.Module, ABC):
    """
    基础SDE模型类，定义了所有SDE模型的通用接口
    """
    def __init__(self, input_channels, hidden_channels, output_channels, sde_type='ito'):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.sde_type = sde_type
        self.noise_type = 'diagonal'  # 默认对角噪声
        
    @abstractmethod
    def f(self, t, y):
        """漂移函数"""
        pass
        
    @abstractmethod
    def g(self, t, y):
        """扩散函数"""
        pass
        
    def forward(self, ts, batch_size):
        """前向传播，求解SDE"""
        y0 = torch.randn(batch_size, self.hidden_channels, 
                        device=ts.device, dtype=ts.dtype)
        return torchsde.sdeint(self, y0, ts)


class MaskedSequenceProcessor:
    """
    处理mask序列的工具类
    """
    @staticmethod
    def apply_mask(sequence, mask):
        """
        应用mask到序列
        Args:
            sequence: (batch, seq_len, features)
            mask: (batch, seq_len) - True表示有效位置
        """
        masked_seq = sequence.clone()
        masked_seq[~mask] = 0.0
        return masked_seq
        
    @staticmethod
    def get_last_valid_output(sequence, mask):
        """
        获取每个序列的最后一个有效输出
        Args:
            sequence: (batch, seq_len, features)
            mask: (batch, seq_len)
        """
        batch_size = sequence.size(0)
        last_indices = mask.sum(dim=1) - 1  # 最后有效位置的索引
        last_indices = last_indices.clamp(min=0)  # 防止负数索引
        
        # 使用高级索引获取最后有效输出
        batch_indices = torch.arange(batch_size, device=sequence.device)
        return sequence[batch_indices, last_indices]