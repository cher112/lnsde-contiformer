"""
内存监控和OOM保护工具
"""

import torch
import psutil
import gc


def get_gpu_memory_info():
    """获取GPU内存使用信息"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    cached = torch.cuda.memory_reserved() / 1024**3     # GB
    max_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return {
        'allocated': allocated,
        'cached': cached,
        'total': max_mem,
        'free': max_mem - allocated,
        'usage_percent': (allocated / max_mem) * 100
    }


def check_memory_pressure():
    """检查是否存在内存压力"""
    gpu_info = get_gpu_memory_info()
    
    if gpu_info is None:
        return False
    
    # 如果GPU内存使用超过85%，认为有内存压力
    return gpu_info['usage_percent'] > 85.0


def emergency_memory_cleanup():
    """紧急内存清理"""
    # Python垃圾回收
    gc.collect()
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("执行紧急内存清理")


def safe_forward_pass(model, inputs, max_retries=3):
    """安全的前向传播，包含OOM保护"""
    
    for attempt in range(max_retries):
        try:
            # 检查内存压力
            if check_memory_pressure():
                emergency_memory_cleanup()
            
            return model(*inputs)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM detected, attempt {attempt + 1}/{max_retries}")
                emergency_memory_cleanup()
                
                if attempt == max_retries - 1:
                    print("多次尝试后仍然OOM，跳过此批次")
                    return None
            else:
                raise e
    
    return None


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, warning_threshold=80, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_warning_batch = -1
        
    def check_and_warn(self, batch_idx):
        """检查内存并发出警告"""
        gpu_info = get_gpu_memory_info()
        
        if gpu_info is None:
            return
        
        usage = gpu_info['usage_percent']
        
        if usage > self.critical_threshold:
            print(f"\n🚨 临界内存警告 (批次{batch_idx}): GPU内存使用{usage:.1f}%")
            print("执行强制内存清理...")
            emergency_memory_cleanup()
            
        elif usage > self.warning_threshold and batch_idx - self.last_warning_batch > 10:
            print(f"\n⚠️ 内存警告 (批次{batch_idx}): GPU内存使用{usage:.1f}%")
            self.last_warning_batch = batch_idx
    
    def get_memory_summary(self):
        """获取内存使用摘要"""
        gpu_info = get_gpu_memory_info()
        
        if gpu_info is None:
            return "GPU不可用"
        
        return (f"GPU内存: {gpu_info['allocated']:.1f}GB/"
                f"{gpu_info['total']:.1f}GB ({gpu_info['usage_percent']:.1f}%)")


# 全局内存监控器
memory_monitor = MemoryMonitor()