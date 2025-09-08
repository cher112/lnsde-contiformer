#!/usr/bin/env python3
"""
GPU使用率优化分析和解决方案
"""

import torch
import time
import psutil
from torch.profiler import profile, record_function, ProfilerActivity

def analyze_gpu_bottlenecks():
    """分析GPU使用率瓶颈"""
    print("="*60)
    print("GPU使用率瓶颈分析")
    print("="*60)
    
    # GPU信息
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = gpu.total_memory / 1024**3
        
        print(f"GPU型号: {gpu.name}")
        print(f"内存: {allocated:.1f}GB/{total:.1f}GB (使用{allocated/total*100:.1f}%)")
        print(f"缓存: {reserved:.1f}GB")
        print(f"计算能力: {gpu.major}.{gpu.minor}")
        print(f"多处理器: {gpu.multi_processor_count}")
        
        # 内存利用率分析
        memory_util = allocated / total * 100
        if memory_util < 70:
            print(f"🔍 内存利用率低 ({memory_util:.1f}%) - 可以增大批次大小")
        elif memory_util > 90:
            print(f"⚠️ 内存接近满载 ({memory_util:.1f}%) - 需要内存优化")
        else:
            print(f"✅ 内存利用适中 ({memory_util:.1f}%)")
    
    # CPU分析
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"\nCPU核心: {cpu_count}")
    print(f"系统内存: {memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB")


def gpu_optimization_strategies():
    """GPU使用率优化策略"""
    print("\n" + "="*60)
    print("GPU使用率优化策略")
    print("="*60)
    
    print("\n1. 🚀 批次大小优化")
    print("   当前可能问题: batch_size过小，GPU未充分利用")
    print("   解决方案:")
    print("   - 动态批次大小: 根据内存自动调整")
    print("   - 梯度累积: 模拟更大批次而不超出内存")
    print("   - 建议测试: batch_size=32,64,128")
    
    print("\n2. ⚡ 数据加载优化")
    print("   当前可能问题: CPU数据预处理成为瓶颈")
    print("   解决方案:")
    print("   - 增加DataLoader workers")
    print("   - 启用pin_memory")
    print("   - 数据预缓存到GPU")
    print("   - 异步数据传输")
    
    print("\n3. 🔧 计算优化")
    print("   当前可能问题: SDE求解串行计算")
    print("   解决方案:")
    print("   - 向量化SDE求解")
    print("   - 并行ContiFormer层")
    print("   - 启用cudNN基准模式")
    print("   - 优化attention计算")
    
    print("\n4. 💾 内存优化")
    print("   当前可能问题: 内存碎片和释放不及时")
    print("   解决方案:")
    print("   - 梯度检查点")
    print("   - 激活重计算")
    print("   - 及时清理中间变量")
    print("   - 混合精度训练")


def create_optimized_config():
    """创建GPU优化配置"""
    print("\n" + "="*60)
    print("GPU优化配置建议")
    print("="*60)
    
    # RTX 4090 优化配置
    rtx4090_configs = [
        {
            'name': '高吞吐量配置',
            'batch_size': 64,
            'num_workers': 8,
            'pin_memory': True,
            'gradient_accumulation_steps': 1,
            'use_amp': True,
            'torch_compile': True,
            'expected_gpu_util': '70-85%'
        },
        {
            'name': '内存平衡配置', 
            'batch_size': 32,
            'num_workers': 6,
            'pin_memory': True,
            'gradient_accumulation_steps': 2,
            'use_amp': True,
            'torch_compile': False,
            'expected_gpu_util': '60-75%'
        },
        {
            'name': '稳定性优先配置',
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'gradient_accumulation_steps': 4,
            'use_amp': True,
            'torch_compile': False,
            'expected_gpu_util': '50-65%'
        }
    ]
    
    for config in rtx4090_configs:
        print(f"\n📋 {config['name']}:")
        print(f"   批次大小: {config['batch_size']}")
        print(f"   数据加载进程: {config['num_workers']}")
        print(f"   梯度累积: {config['gradient_accumulation_steps']}")
        print(f"   混合精度: {config['use_amp']}")
        print(f"   预期GPU利用率: {config['expected_gpu_util']}")
        
        # 生成命令
        cmd = f"""python main.py \\
  --dataset MACHO \\
  --batch_size {config['batch_size']} \\
  --num_workers {config['num_workers']} \\
  --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
  {'--amp' if config['use_amp'] else '--no_amp'} \\
  {'--torch_compile' if config.get('torch_compile', False) else ''} \\
  --learning_rate 1e-5 \\
  --gradient_clip 1.0 \\
  --epochs 3"""
        
        print(f"   命令: {cmd}")


def benchmark_batch_sizes():
    """批次大小性能测试"""
    print("\n" + "="*60)
    print("批次大小性能基准测试")
    print("="*60)
    
    test_script = '''
import torch
import time
from torch.cuda.amp import autocast

# 模拟模型前向传播
def benchmark_forward(batch_size, seq_len=200, hidden_dim=128):
    device = torch.device('cuda')
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, 3, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    
    # 模拟复杂计算
    with autocast():
        # ContiFormer attention
        attn_weights = torch.matmul(x, x.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_output = torch.matmul(attn_weights, x)
        
        # SDE求解模拟
        dt = 0.01
        steps = 10
        y = x
        for _ in range(steps):
            dy = torch.randn_like(y) * dt
            y = y + dy
        
        output = torch.mean(y, dim=1)
    
    return output

# 测试不同批次大小
batch_sizes = [8, 16, 32, 64, 128]
results = []

for bs in batch_sizes:
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()
        
        # 多次运行取平均
        for _ in range(5):
            output = benchmark_forward(bs)
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        avg_time = (end_time - start_time) / 5
        memory_used = (end_memory - start_memory) / 1024**2  # MB
        throughput = bs / avg_time  # samples/sec
        
        results.append({
            'batch_size': bs,
            'time_per_batch': avg_time,
            'memory_mb': memory_used,
            'throughput': throughput
        })
        
        print(f"Batch {bs:3d}: {avg_time:.3f}s, {memory_used:4.0f}MB, {throughput:5.1f} samples/s")
        
    except torch.cuda.OutOfMemoryError:
        print(f"Batch {bs:3d}: OOM")
        break
    except Exception as e:
        print(f"Batch {bs:3d}: Error - {e}")

# 找最优批次大小
if results:
    best = max(results, key=lambda x: x['throughput'])
    print(f"\\n最优批次大小: {best['batch_size']} (吞吐量: {best['throughput']:.1f} samples/s)")
'''
    
    print("运行基准测试:")
    print("python -c \"")
    for line in test_script.strip().split('\n'):
        print(line)
    print("\"")


if __name__ == "__main__":
    analyze_gpu_bottlenecks()
    gpu_optimization_strategies()
    create_optimized_config()
    benchmark_batch_sizes()