#!/usr/bin/env python3
"""
多核并行重采样脚本 - 对ASAS, LINEAR, MACHO三个数据集进行混合模式重采样
使用多进程并行处理，提高效率
"""

import os
import sys
import numpy as np
import torch
import pickle
from multiprocessing import Pool, Manager
from datetime import datetime
import argparse
from collections import Counter
from tqdm import tqdm
import time
import psutil

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import HybridResampler, save_resampled_data, configure_chinese_font

# 配置中文字体
configure_chinese_font()

def load_dataset(dataset_name, data_dir='/root/autodl-fs/lnsde-contiformer/data'):
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 ('ASAS', 'LINEAR', 'MACHO')
        data_dir: 数据目录
    
    Returns:
        X, y, times, masks 数据
    """
    # 使用统一的文件名格式，优先使用original版本进行重采样
    dataset_files = {
        'ASAS': [
            f'{dataset_name}_original.pkl',
            f'{dataset_name}_fixed.pkl'
        ],
        'LINEAR': [
            f'{dataset_name}_original.pkl',
            f'{dataset_name}_fixed.pkl'
        ],
        'MACHO': [
            f'{dataset_name}_original.pkl',
            f'{dataset_name}_fixed.pkl'
        ]
    }
    
    if dataset_name not in dataset_files:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 优先使用fixed版本（样本数稍少，处理更快）
    file_path = None
    for filename in dataset_files[dataset_name]:
        candidate_path = os.path.join(data_dir, filename)
        if os.path.exists(candidate_path):
            file_path = candidate_path
            break
    
    if file_path is None:
        available_files = os.listdir(data_dir)
        raise FileNotFoundError(f"找不到{dataset_name}数据集文件。可用文件: {available_files}")
    
    print(f"加载{dataset_name}数据集: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 处理列表格式的数据（每个样本都是字典）
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        n_samples = len(data)
        
        # 获取序列长度（假设所有样本具有相同长度）
        seq_len = len(data[0]['time'])
        n_features = 3  # time, mag, errmag
        
        # 初始化数组
        X = np.zeros((n_samples, seq_len, n_features))
        y = np.zeros(n_samples, dtype=int)
        times = np.zeros((n_samples, seq_len))
        masks = np.zeros((n_samples, seq_len), dtype=bool)
        
        # 提取数据
        for i, sample in enumerate(data):
            X[i, :, 0] = sample['time']
            X[i, :, 1] = sample['mag'] 
            X[i, :, 2] = sample['errmag']
            y[i] = sample['label']
            times[i] = sample['time']
            masks[i] = sample['mask']
            
        print(f"从列表格式转换完成")
        
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    print(f"{dataset_name} 数据形状: X={X.shape}, y={len(y)}")
    print(f"{dataset_name} 类别分布: {Counter(y)}")
    
    return X, y, times, masks


def resample_dataset_worker(args):
    """
    单个数据集的重采样工作函数
    
    Args:
        args: (dataset_name, config, shared_results, progress_dict)
    
    Returns:
        结果信息
    """
    dataset_name, config, shared_results, progress_dict = args
    
    try:
        # 更新进度：开始加载数据
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"📂 加载{dataset_name}数据集..."
        
        print(f"\n{'='*60}")
        print(f"开始处理 {dataset_name} 数据集重采样")
        print(f"{'='*60}")
        
        # 加载数据
        X, y, times, masks = load_dataset(dataset_name)
        
        # 更新进度：数据加载完成
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"🔧 初始化{dataset_name}重采样器..."
        
        # 创建重采样器
        resampler = HybridResampler(
            smote_k_neighbors=config['k_neighbors'],
            enn_n_neighbors=config['enn_neighbors'],
            sampling_strategy=config['sampling_strategy'],
            synthesis_mode=config['synthesis_mode'],
            noise_level=config['noise_level'],
            apply_enn=config['apply_enn'],
            random_state=config['random_state']
        )
        
        # 更新进度：开始重采样
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"⚡ {dataset_name}混合模式重采样中..."
        
        # 执行重采样
        start_time = datetime.now()
        X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
            X, y, times, masks
        )
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 统计结果
        original_counts = Counter(y)
        resampled_counts = Counter(y_resampled)
        
        # 更新进度：保存数据
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"💾 保存{dataset_name}重采样数据..."
        
        # 保存重采样数据
        save_dir = config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # 直接保存到data目录，使用统一命名
        resampled_filename = f'{dataset_name}_resampled.pkl'
        resampled_path = os.path.join('/root/autodl-fs/lnsde-contiformer/data', resampled_filename)
        
        # 保存数据
        resampled_data = {
            'X': X_resampled,
            'y': y_resampled,
            'times': times_resampled,
            'masks': masks_resampled,
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'distribution': dict(resampled_counts),
            'synthesis_mode': config['synthesis_mode'],
            'original_distribution': dict(original_counts)
        }
        
        with open(resampled_path, 'wb') as f:
            pickle.dump(resampled_data, f)
        
        print(f"重采样数据已保存至: {resampled_path}")
        
        # 生成可视化
        if config['generate_plots']:
            # 更新进度：生成可视化
            with progress_dict.lock:
                progress_dict.status[dataset_name] = f"📊 生成{dataset_name}可视化图表..."
            
            plots_dir = os.path.join(config['plots_dir'], dataset_name)
            os.makedirs(plots_dir, exist_ok=True)
            
            # 类别分布图
            distribution_path = os.path.join(plots_dir, f'{dataset_name}_resampling_distribution.png')
            resampler.visualize_distribution(save_path=distribution_path)
            
            # 合成效果对比图
            if len(np.unique(y)) <= 3 and len(y) >= 10:  # 只对小类别数且有足够样本的数据集生成
                comparison_path = os.path.join(plots_dir, f'{dataset_name}_synthesis_comparison.png')
                try:
                    resampler.smote.visualize_synthesis_comparison(
                        X, y, n_examples=2, save_path=comparison_path
                    )
                except Exception as e:
                    print(f"生成{dataset_name}合成对比图失败: {e}")
        
        # 更新进度：完成
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"✅ {dataset_name}重采样完成！"
        
        result = {
            'dataset': dataset_name,
            'status': 'success',
            'original_samples': len(y),
            'resampled_samples': len(y_resampled),
            'original_distribution': dict(original_counts),
            'resampled_distribution': dict(resampled_counts),
            'processing_time': processing_time,
            'save_path': resampled_path,
            'synthesis_mode': config['synthesis_mode']
        }
        
        # 存储到共享结果中
        with shared_results.lock:
            shared_results.results[dataset_name] = result
        
        print(f"\n✓ {dataset_name} 重采样完成！")
        print(f"处理时间: {processing_time:.2f}秒")
        print(f"原始样本: {len(y)} -> 重采样后: {len(y_resampled)}")
        print(f"保存路径: {resampled_path}")
        
        return result
        
    except Exception as e:
        # 更新进度：失败
        with progress_dict.lock:
            progress_dict.status[dataset_name] = f"❌ {dataset_name}重采样失败"
        
        error_msg = f"{dataset_name} 重采样失败: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        result = {
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e),
            'processing_time': 0
        }
        
        with shared_results.lock:
            shared_results.results[dataset_name] = result
            
        return result


def parallel_resample_datasets(datasets=None, config=None, n_processes=None):
    """
    并行重采样多个数据集 - 优化CPU利用率
    
    Args:
        datasets: 数据集列表，默认['ASAS', 'LINEAR', 'MACHO']
        config: 配置字典
        n_processes: 并行进程数，None时自动优化选择
    
    Returns:
        结果汇总
    """
    if datasets is None:
        datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    # 自动优化进程数选择
    if n_processes is None:
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
        logical_count = psutil.cpu_count(logical=True)  # 逻辑核心数
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 基于CPU和内存优化进程数
        # 每个重采样进程大约需要2-4GB内存
        max_by_memory = int(memory_gb / 3)
        max_by_cpu = min(logical_count, len(datasets) * 2)  # 每个数据集最多2个进程
        
        n_processes = min(max_by_memory, max_by_cpu, len(datasets))
        n_processes = max(1, n_processes)  # 至少1个进程
        
        print(f"🖥️  CPU信息: {cpu_count}物理核心, {logical_count}逻辑核心")
        print(f"💾 内存信息: {memory_gb:.1f}GB")
        print(f"⚡ 自动选择进程数: {n_processes} (基于{len(datasets)}个数据集)")
    
    if config is None:
        config = {
            'k_neighbors': 5,
            'enn_neighbors': 3,
            'sampling_strategy': 'balanced',
            'synthesis_mode': 'hybrid',
            'noise_level': 0.05,
            'apply_enn': True,
            'random_state': 535411460,
            'save_dir': '/root/autodl-fs/lnsde-contiformer/data/resampled',
            'plots_dir': '/root/autodl-tmp/lnsde-contiformer/results/pics',
            'generate_plots': True
        }
    
    print(f"\n{'='*80}")
    print(f"多核并行重采样 - {len(datasets)}个数据集")
    print(f"数据集: {', '.join(datasets)}")
    print(f"合成模式: {config['synthesis_mode']}")
    print(f"并行进程数: {n_processes}")
    print(f"{'='*80}")
    
    # 创建输出目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    
    # 创建管理器和共享结果存储
    manager = Manager()
    shared_results = manager.Namespace()
    shared_results.lock = manager.Lock()
    shared_results.results = manager.dict()
    
    # 创建进度跟踪字典
    progress_dict = manager.Namespace()
    progress_dict.lock = manager.Lock()
    progress_dict.status = manager.dict()
    
    # 初始化进度状态
    for dataset in datasets:
        progress_dict.status[dataset] = f"⏳ {dataset}等待开始..."
    
    # 准备参数
    args_list = [(dataset, config, shared_results, progress_dict) for dataset in datasets]
    
    # 并行处理
    start_time = datetime.now()
    
    print(f"\n🚀 启动并行重采样进程...")
    
    # 创建进度条
    def show_progress():
        """实时显示进度状态"""
        last_status = {}
        completed_count = 0
        
        while completed_count < len(datasets):
            try:
                time.sleep(3)  # 每3秒检查一次
                
                with progress_dict.lock:
                    current_status = dict(progress_dict.status)
                
                # 检查状态变化
                for dataset, status in current_status.items():
                    if dataset not in last_status or last_status[dataset] != status:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
                        last_status[dataset] = status
                
                # 统计完成数量
                completed_count = sum(1 for status in current_status.values() 
                                    if "完成" in status or "✅" in status)
                
                if completed_count > 0:
                    progress_percent = (completed_count / len(datasets)) * 100
                    progress_bar = "█" * int(progress_percent / 5) + "░" * (20 - int(progress_percent / 5))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 总进度: [{progress_bar}] {progress_percent:.1f}% ({completed_count}/{len(datasets)})")
                    
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 进度监控错误: {e}")
                time.sleep(1)
    
    # 启动进度监控线程
    import threading
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()
    
    with Pool(processes=min(n_processes, len(datasets))) as pool:
        pool_results = pool.map(resample_dataset_worker, args_list)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 等待进度线程结束
    progress_thread.join(timeout=1)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 所有进程完成，汇总结果...")
    
    # 汇总结果
    results_summary = {
        'total_datasets': len(datasets),
        'successful': 0,
        'failed': 0,
        'total_processing_time': total_time,
        'results': dict(shared_results.results)
    }
    
    print(f"\n{'='*80}")
    print("🎯 并行重采样最终结果")
    print(f"{'='*80}")
    
    total_original = 0
    total_resampled = 0
    
    for dataset in datasets:
        result = shared_results.results.get(dataset, {})
        if result.get('status') == 'success':
            results_summary['successful'] += 1
            original = result.get('original_samples', 0)
            resampled = result.get('resampled_samples', 0)
            processing_time = result.get('processing_time', 0)
            
            total_original += original
            total_resampled += resampled
            
            print(f"✅ {dataset}: 重采样成功!")
            print(f"   📊 样本数: {original:,} → {resampled:,} (+{resampled-original:,})")
            print(f"   ⏱️  用时: {processing_time:.1f}秒")
            print(f"   🎯 类别平衡: {result.get('resampled_distribution', {})}")
            print()
        else:
            results_summary['failed'] += 1
            print(f"❌ {dataset}: 重采样失败")
            print(f"   ⚠️  错误: {result.get('error', '未知错误')}")
            print()
    
    # 性能统计
    avg_time_per_dataset = total_time / len(datasets) if datasets else 0
    speedup = sum(result.get('processing_time', 0) for result in shared_results.results.values()) / max(total_time, 0.001)
    
    print(f"🏆 性能统计:")
    print(f"   ✅ 成功: {results_summary['successful']}/{len(datasets)} 个数据集")
    print(f"   ❌ 失败: {results_summary['failed']}/{len(datasets)} 个数据集")
    print(f"   📈 总样本增长: {total_original:,} → {total_resampled:,} (+{((total_resampled/max(total_original,1)-1)*100):.1f}%)")
    print(f"   ⏱️  总用时: {total_time:.1f}秒 (平均 {avg_time_per_dataset:.1f}秒/数据集)")
    print(f"   🚀 并行加速: {speedup:.1f}x")
    print(f"   💾 合成模式: {config['synthesis_mode'].upper()}")
    
    if results_summary['successful'] > 0:
        print(f"\n💾 重采样文件保存位置:")
        for dataset in datasets:
            result = shared_results.results.get(dataset, {})
            if result.get('status') == 'success':
                print(f"   {dataset}: /root/autodl-fs/lnsde-contiformer/data/{dataset}_resampled.pkl")
    
    # 保存汇总结果
    summary_path = os.path.join(config['save_dir'], f'parallel_resampling_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"结果汇总已保存: {summary_path}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='多核并行重采样脚本')
    parser.add_argument('--datasets', nargs='+', default=['ASAS', 'LINEAR', 'MACHO'],
                        help='要处理的数据集列表')
    parser.add_argument('--processes', type=int, default=None,
                        help='并行进程数 (None=自动优化)')
    parser.add_argument('--mode', choices=['interpolation', 'warping', 'hybrid'], default='hybrid',
                        help='合成模式')
    parser.add_argument('--noise-level', type=float, default=0.05,
                        help='噪声水平')
    parser.add_argument('--no-plots', action='store_true',
                        help='不生成可视化图片')
    parser.add_argument('--no-enn', action='store_true',
                        help='不使用ENN清理')
    
    args = parser.parse_args()
    
    config = {
        'k_neighbors': 5,
        'enn_neighbors': 3,
        'sampling_strategy': 'balanced',
        'synthesis_mode': args.mode,
        'noise_level': args.noise_level,
        'apply_enn': not args.no_enn,
        'random_state': 535411460,
        'save_dir': '/root/autodl-fs/lnsde-contiformer/data/resampled',
        'plots_dir': '/root/autodl-tmp/lnsde-contiformer/results/pics',
        'generate_plots': not args.no_plots
    }
    
    results = parallel_resample_datasets(
        datasets=args.datasets,
        config=config,
        n_processes=args.processes
    )
    
    print(f"\n🎉 并行重采样完成！")
    return results


if __name__ == "__main__":
    main()