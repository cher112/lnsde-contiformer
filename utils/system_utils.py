"""
系统相关工具函数
"""

import os
import random
import torch
import numpy as np
import gc
import subprocess

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


def get_gpu_memory_usage():
    """获取所有GPU的内存使用情况（通过nvidia-smi）"""
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    
    try:
        # 使用nvidia-smi获取实际的GPU内存使用情况
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_id = int(parts[0])
                        gpu_name = parts[1]
                        total_mb = float(parts[2])
                        used_mb = float(parts[3])
                        free_mb = float(parts[4])
                        
                        total_gb = total_mb / 1024
                        used_gb = used_mb / 1024
                        usage_percent = (used_gb / total_gb) * 100
                        
                        gpu_info.append({
                            'id': gpu_id,
                            'name': gpu_name,
                            'allocated_gb': used_gb,  # 实际已使用内存
                            'total_gb': total_gb,
                            'usage_percent': usage_percent,
                            'available_gb': free_mb / 1024
                        })
        else:
            print(f"nvidia-smi命令执行失败: {result.stderr}")
            
    except Exception as e:
        print(f"无法通过nvidia-smi获取GPU信息: {e}")
        # 回退到PyTorch方法
        return get_pytorch_gpu_info()
    
    return gpu_info


def get_pytorch_gpu_info():
    """回退方法：使用PyTorch获取GPU信息"""
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            usage_percent = (allocated / total) * 100
            
            gpu_info.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'allocated_gb': allocated,
                'total_gb': total,
                'usage_percent': usage_percent,
                'available_gb': total - allocated
            })
        except Exception as e:
            print(f"无法获取GPU {i} 信息: {e}")
    
    return gpu_info


def find_best_gpu(usage_threshold=15.0):
    """
    找到最适合的GPU
    优先选择内存使用率低于阈值的GPU，避免与其他进程竞争
    
    Args:
        usage_threshold: 内存使用率阈值（百分比），低于此值认为GPU空闲
    
    Returns:
        最佳GPU的ID，-1表示没有合适的GPU
    """
    gpu_info = get_gpu_memory_usage()
    if not gpu_info:
        return -1
    
    print("=== GPU状态（基于nvidia-smi实际使用情况）===")
    
    # 首先尝试找到完全空闲的GPU（使用率 < usage_threshold）
    available_gpus = []
    busy_gpus = []
    
    for gpu in gpu_info:
        status_emoji = ""
        status_text = ""
        
        if gpu['usage_percent'] < usage_threshold:
            status_emoji = "🟢"
            status_text = "空闲可用"
            available_gpus.append(gpu)
        elif gpu['usage_percent'] < 50:
            status_emoji = "🟡"
            status_text = "轻度使用"
            busy_gpus.append(gpu)
        elif gpu['usage_percent'] < 80:
            status_emoji = "🟠"
            status_text = "中度使用"
            busy_gpus.append(gpu)
        else:
            status_emoji = "🔴"
            status_text = "高负载"
            busy_gpus.append(gpu)
        
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  内存: {gpu['allocated_gb']:.1f}GB / {gpu['total_gb']:.1f}GB ({gpu['usage_percent']:.1f}%) {status_emoji} {status_text}")
    
    # 选择策略
    if available_gpus:
        # 有空闲GPU，选择使用率最低的
        best_gpu = min(available_gpus, key=lambda x: x['usage_percent'])
        print(f"\n✅ 选择空闲GPU {best_gpu['id']}: {best_gpu['name']} (使用率: {best_gpu['usage_percent']:.1f}%)")
        return best_gpu['id']
    elif busy_gpus:
        # 没有完全空闲的GPU，选择使用率最低的忙碌GPU
        best_gpu = min(busy_gpus, key=lambda x: x['usage_percent'])
        print(f"\n⚠️ 所有GPU都在使用中，选择负载最低的GPU {best_gpu['id']}: {best_gpu['name']} (使用率: {best_gpu['usage_percent']:.1f}%)")
        print("   建议：监控训练过程中的内存使用，可能会与其他进程竞争")
        return best_gpu['id']
    else:
        print("\n❌ 没有找到可用的GPU")
        return -1


def get_device(device_arg, gpu_id=-1):
    """获取计算设备，支持GPU选择"""
    if device_arg == 'auto':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            print("CUDA不可用，使用CPU")
            return device
        
        # 根据gpu_id参数选择GPU
        if gpu_id >= 0:
            # 检查指定的GPU是否存在
            if gpu_id >= torch.cuda.device_count():
                print(f"指定的GPU {gpu_id} 不存在，自动选择最佳GPU")
                gpu_id = find_best_gpu()
            else:
                print(f"使用指定的GPU {gpu_id}")
        else:
            # 自动选择最佳GPU
            gpu_id = find_best_gpu()
        
        if gpu_id >= 0:
            device = torch.device(f'cuda:{gpu_id}')
            # 设置当前GPU
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device('cpu')
            print("无可用GPU，使用CPU")
    else:
        device = torch.device(device_arg)
        if 'cuda' in device_arg and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            device = torch.device('cpu')
    
    return device