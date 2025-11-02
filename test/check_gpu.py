#!/usr/bin/env python3
"""
GPU状态查看工具
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.system_utils import get_gpu_memory_usage, find_best_gpu

def main():
    print("=== GPU状态查看工具 ===")
    
    gpu_info = get_gpu_memory_usage()
    
    if not gpu_info:
        print("❌ 未检测到可用GPU")
        return
    
    print(f"检测到 {len(gpu_info)} 个GPU:\n")
    
    # 显示所有GPU状态
    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  总内存: {gpu['total_gb']:.1f} GB")
        print(f"  已使用: {gpu['allocated_gb']:.1f} GB ({gpu['usage_percent']:.1f}%)")
        print(f"  可用内存: {gpu['available_gb']:.1f} GB")
        
        # 状态指示
        if gpu['usage_percent'] < 15:
            status = "🟢 空闲可用"
        elif gpu['usage_percent'] < 50:
            status = "🟡 轻度使用"
        elif gpu['usage_percent'] < 80:
            status = "🟠 中度使用"
        else:
            status = "🔴 高负载"
        print(f"  状态: {status}")
        print()
    
    # 推荐最佳GPU
    print("=" * 50)
    best_gpu_id = find_best_gpu()
    
    print("\n📝 使用说明:")
    print("  自动选择GPU: python main.py --dataset 3")
    print(f"  指定GPU:     python main.py --gpu_id {best_gpu_id if best_gpu_id >= 0 else '0'} --dataset 3")
    print("  使用CPU:     python main.py --device cpu --dataset 3")
    
    print("\n💡 多GPU并行训练建议:")
    available_count = sum(1 for gpu in gpu_info if gpu['usage_percent'] < 15)
    if available_count >= 2:
        print(f"  检测到 {available_count} 个空闲GPU，可以同时运行多个训练任务:")
        available_gpus = [gpu['id'] for gpu in gpu_info if gpu['usage_percent'] < 15]
        for i, gpu_id in enumerate(available_gpus[:3]):  # 最多显示3个建议
            datasets = ['ASAS', 'LINEAR', 'MACHO']
            if i < len(datasets):
                print(f"    终端{i+1}: python main.py --gpu_id {gpu_id} --dataset {i+1}  # 训练{datasets[i]}")
    else:
        print(f"  当前只有 {available_count} 个空闲GPU，建议单任务训练")

if __name__ == "__main__":
    main()