#!/usr/bin/env python3
"""
简化并行重采样脚本 - 去掉复杂的进度监控
"""

import os
import sys
import numpy as np
import torch
import pickle
from multiprocessing import Pool
from datetime import datetime
import argparse
from collections import Counter
import psutil

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import HybridResampler, configure_chinese_font

# 配置中文字体
configure_chinese_font()

def load_dataset_simple(dataset_name, data_dir='/root/autodl-fs/lnsde-contiformer/data'):
    """简化的数据加载函数"""
    print(f"[{dataset_name}] 🔍 开始加载数据...")
    
    file_path = os.path.join(data_dir, f'{dataset_name}_original.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        n_samples = len(data)
        seq_len = len(data[0]['time'])
        n_features = 3
        
        X = np.zeros((n_samples, seq_len, n_features))
        y = np.zeros(n_samples, dtype=int)
        times = np.zeros((n_samples, seq_len))
        masks = np.ones((n_samples, seq_len), dtype=bool)
        
        print(f"[{dataset_name}] 🔄 转换{n_samples}个样本...")
        for i, sample in enumerate(data):
            X[i, :, 0] = sample['time']
            X[i, :, 1] = sample['mag'] 
            X[i, :, 2] = sample['errmag']
            y[i] = sample['label']
            times[i] = sample['time']
            if 'mask' in sample:
                masks[i] = sample['mask']
        
        print(f"[{dataset_name}] ✅ 数据加载完成: {X.shape}, 分布: {Counter(y)}")
        return X, y, times, masks
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")

def simple_resample_worker(dataset_name):
    """简化的重采样工作函数"""
    try:
        print(f"\n{'='*60}")
        print(f"[{dataset_name}] 🚀 开始重采样进程")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # 1. 加载数据
        X, y, times, masks = load_dataset_simple(dataset_name)
        
        # 2. 创建重采样器
        print(f"[{dataset_name}] 🔧 初始化重采样器...")
        resampler = HybridResampler(
            smote_k_neighbors=5,
            enn_n_neighbors=3,
            sampling_strategy='balanced',
            synthesis_mode='hybrid',
            noise_level=0.05,
            apply_enn=True,
            random_state=535411460
        )
        
        # 3. 执行重采样
        print(f"[{dataset_name}] ⚡ 执行混合模式重采样...")
        X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
            X, y, times, masks
        )
        
        # 4. 保存结果
        print(f"[{dataset_name}] 💾 保存重采样数据...")
        save_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_resampled.pkl'
        
        resampled_data = {
            'X': X_resampled,
            'y': y_resampled,
            'times': times_resampled,
            'masks': masks_resampled,
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'distribution': dict(Counter(y_resampled)),
            'synthesis_mode': 'hybrid',
            'original_distribution': dict(Counter(y))
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(resampled_data, f)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            'dataset': dataset_name,
            'status': 'success',
            'original_samples': len(y),
            'resampled_samples': len(y_resampled),
            'original_distribution': dict(Counter(y)),
            'resampled_distribution': dict(Counter(y_resampled)),
            'processing_time': processing_time,
            'save_path': save_path
        }
        
        print(f"[{dataset_name}] ✅ 重采样完成! 用时: {processing_time:.1f}秒")
        print(f"[{dataset_name}] 📊 {len(y):,} → {len(y_resampled):,} 样本")
        print(f"[{dataset_name}] 💾 保存至: {save_path}")
        
        return result
        
    except Exception as e:
        print(f"[{dataset_name}] ❌ 重采样失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e),
            'processing_time': 0
        }

def simple_parallel_resample(datasets=['ASAS', 'LINEAR', 'MACHO'], n_processes=None):
    """简化的并行重采样"""
    
    # 自动选择进程数
    if n_processes is None:
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        max_by_memory = int(memory_gb / 4)  # 每个进程4GB内存
        max_by_cpu = min(cpu_count, len(datasets))
        n_processes = min(max_by_memory, max_by_cpu, len(datasets))
        n_processes = max(1, n_processes)
        
        print(f"🖥️  CPU: {cpu_count}核心, 内存: {memory_gb:.1f}GB")
        print(f"⚡ 自动选择进程数: {n_processes}")
    
    print(f"\n{'='*80}")
    print(f"🚀 简化并行重采样 - {len(datasets)}个数据集")
    print(f"数据集: {', '.join(datasets)}")
    print(f"并行进程数: {n_processes}")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    
    # 并行处理
    if n_processes == 1:
        # 单进程处理
        print("📝 使用单进程模式...")
        results = []
        for dataset in datasets:
            result = simple_resample_worker(dataset)
            results.append(result)
    else:
        # 多进程处理
        print(f"🔥 启动{n_processes}个并行进程...")
        with Pool(processes=n_processes) as pool:
            results = pool.map(simple_resample_worker, datasets)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("🎯 最终结果汇总")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    total_original = 0
    total_resampled = 0
    
    for result in results:
        if result['status'] == 'success':
            successful += 1
            original = result['original_samples']
            resampled = result['resampled_samples']
            total_original += original
            total_resampled += resampled
            
            print(f"✅ {result['dataset']}: 成功")
            print(f"   📊 {original:,} → {resampled:,} 样本 (+{resampled-original:,})")
            print(f"   ⏱️  用时: {result['processing_time']:.1f}秒")
            print(f"   💾 保存: {result['save_path']}")
        else:
            failed += 1
            print(f"❌ {result['dataset']}: 失败 - {result.get('error', '未知错误')}")
    
    print(f"\n🏆 总结:")
    print(f"   ✅ 成功: {successful}/{len(datasets)} 个数据集")
    print(f"   ❌ 失败: {failed}/{len(datasets)} 个数据集")
    print(f"   📈 总样本增长: {total_original:,} → {total_resampled:,} (+{((total_resampled/max(total_original,1)-1)*100):.1f}%)")
    print(f"   ⏱️  总用时: {total_time:.1f}秒 (平均 {total_time/len(datasets):.1f}秒/数据集)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='简化并行重采样脚本')
    parser.add_argument('--datasets', nargs='+', default=['ASAS', 'LINEAR', 'MACHO'],
                        help='要处理的数据集列表')
    parser.add_argument('--processes', type=int, default=None,
                        help='并行进程数 (None=自动优化)')
    
    args = parser.parse_args()
    
    results = simple_parallel_resample(
        datasets=args.datasets,
        n_processes=args.processes
    )
    
    print(f"\n🎉 所有重采样任务完成！")
    return results

if __name__ == "__main__":
    main()