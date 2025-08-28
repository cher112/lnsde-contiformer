"""
Preprocessing utilities for light curve data
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class LombScargleProcessor:
    """Lomb-Scargle周期图处理器（占位符）"""
    def __init__(self):
        pass
    
    def process(self, times, mags):
        """处理光变曲线数据"""
        return times, mags


class TimeSeriesNormalizer:
    """时间序列标准化器"""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        """拟合标准化参数"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        
    def transform(self, data):
        """标准化数据"""
        if self.mean is None or self.std is None:
            raise ValueError("请先调用fit方法")
        return (data - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, data):
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)


class MaskGenerator:
    """生成mask序列"""
    @staticmethod
    def generate_mask(seq_len, valid_len):
        """生成mask"""
        mask = np.zeros(seq_len, dtype=bool)
        mask[:valid_len] = True
        return mask