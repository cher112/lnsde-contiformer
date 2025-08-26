"""
系统相关工具函数
"""

import os
import random
import torch
import numpy as np
import gc

def set_seed(seed):
    """设置随机种子，确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_device(device_arg):
    """获取计算设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"使用GPU: {gpu_name}")
        else:
            device = torch.device('cpu')
            print("使用CPU")
    else:
        device = torch.device(device_arg)
        if device_arg == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            device = torch.device('cpu')
    
    return device