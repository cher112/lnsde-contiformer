#!/usr/bin/env python3
"""
提取3个数据集3个方法(lnsde, geosde, langevin)的所有实验结果
取前3名计算平均值±标准差
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict


def extract_all_best_metrics():
    """提取所有实验的最优指标"""
    
    # 方法映射
    method_mapping = {
        'linear_noise': 'lnsde',
        'geometric': 'geosde', 
        'langevin': 'langevin'
    }
    
    # 存储所有结果：dataset -> method -> [实验结果列表]
    all_results = defaultdict(lambda: defaultdict(list))
    
    # 查找所有日志文件
    log_files = glob.glob("results/**/*.log", recursive=True)
    
    print("=== 查找所有实验结果 ===")
    print("数据集: ASAS, LINEAR, MACHO") 
    print("方法: lnsde(linear_noise), geosde(geometric), langevin")
    print(f"找到 {len(log_files)} 个日志文件")
    print()
    
    # 处理每个日志文件
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset = data.get('dataset', '')
            model_type = data.get('model_type', '')
            
            # 过滤我们需要的数据集和方法
            if dataset not in ['ASAS', 'LINEAR', 'MACHO']:
                continue
                
            if model_type not in ['linear_noise', 'geometric', 'langevin']:
                continue
            
            method_name = method_mapping[model_type]
            
            # 找到最优验证准确率的epoch
            best_val_acc = 0
            best_epoch_data = None
            
            for epoch_data in data.get('epochs', []):
                val_acc = epoch_data.get('val_acc', 0)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch_data = epoch_data
            
            if best_epoch_data and best_val_acc > 0:
                # 提取指标
                metrics = {
                    'epoch': best_epoch_data.get('epoch', 0),
                    'train_acc': best_epoch_data.get('train_acc', 0),
                    'val_acc': best_epoch_data.get('val_acc', 0),
                    'train_f1': best_epoch_data.get('train_f1', 0),
                    'val_f1': best_epoch_data.get('val_f1', 0),
                    'train_recall': best_epoch_data.get('train_recall', 0),
                    'val_recall': best_epoch_data.get('val_recall', 0),
                    'file': os.path.basename(log_file)
                }
                
                all_results[dataset][method_name].append(metrics)
                print(f"找到 {dataset}-{method_name}: 验证准确率 {best_val_acc:.2f}% 来自 {os.path.basename(log_file)}")
                    
        except Exception as e:
            print(f"处理文件 {log_file} 时出错: {e}")
    
    # 计算每个组合的前3名平均值和标准差
    print("\n" + "=" * 80)
    print("📊 前3名结果统计 (平均值±标准差)")
    print("=" * 80)
    
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    methods = ['lnsde', 'geosde', 'langevin']
    
    summary_results = {}
    
    for dataset in datasets:
        print(f"\n🔸 {dataset} 数据集:")
        print("-" * 60)
        
        summary_results[dataset] = {}
        
        for method in methods:
            results = all_results[dataset][method]
            
            if len(results) == 0:
                print(f"\n  {method.upper()}: 无数据")
                summary_results[dataset][method] = None
                continue
            
            # 按验证准确率排序，取前3名
            results.sort(key=lambda x: x['val_acc'], reverse=True)
            top3 = results[:3]
            
            print(f"\n  {method.upper()}: 找到 {len(results)} 个实验，取前3名:")
            
            if len(top3) > 0:
                # 显示前3名详情
                for i, result in enumerate(top3, 1):
                    print(f"    #{i}: {result['val_acc']:.2f}% (F1:{result['val_f1']:.1f}, Recall:{result['val_recall']:.1f}) - {result['file']}")
                
                # 计算平均值和标准差
                val_accs = [r['val_acc'] for r in top3]
                val_f1s = [r['val_f1'] for r in top3]  
                val_recalls = [r['val_recall'] for r in top3]
                train_accs = [r['train_acc'] for r in top3]
                train_f1s = [r['train_f1'] for r in top3]
                train_recalls = [r['train_recall'] for r in top3]
                
                stats = {
                    'train_acc_mean': np.mean(train_accs),
                    'train_acc_std': np.std(train_accs, ddof=1) if len(train_accs) > 1 else 0,
                    'val_acc_mean': np.mean(val_accs),
                    'val_acc_std': np.std(val_accs, ddof=1) if len(val_accs) > 1 else 0,
                    'train_f1_mean': np.mean(train_f1s),
                    'train_f1_std': np.std(train_f1s, ddof=1) if len(train_f1s) > 1 else 0,
                    'val_f1_mean': np.mean(val_f1s),
                    'val_f1_std': np.std(val_f1s, ddof=1) if len(val_f1s) > 1 else 0,
                    'train_recall_mean': np.mean(train_recalls),
                    'train_recall_std': np.std(train_recalls, ddof=1) if len(train_recalls) > 1 else 0,
                    'val_recall_mean': np.mean(val_recalls),
                    'val_recall_std': np.std(val_recalls, ddof=1) if len(val_recalls) > 1 else 0,
                    'count': len(top3)
                }
                
                summary_results[dataset][method] = stats
                
                print(f"    统计结果 (n={len(top3)}):")
                print(f"      训练准确率: {stats['train_acc_mean']:.2f}±{stats['train_acc_std']:.2f}%")
                print(f"      验证准确率: {stats['val_acc_mean']:.2f}±{stats['val_acc_std']:.2f}%")
                print(f"      训练加权F1: {stats['train_f1_mean']:.2f}±{stats['train_f1_std']:.2f}")
                print(f"      验证加权F1: {stats['val_f1_mean']:.2f}±{stats['val_f1_std']:.2f}")
                print(f"      训练加权Recall: {stats['train_recall_mean']:.2f}±{stats['train_recall_std']:.2f}")
                print(f"      验证加权Recall: {stats['val_recall_mean']:.2f}±{stats['val_recall_std']:.2f}")
            else:
                summary_results[dataset][method] = None
    
    # 生成最终汇总表
    print("\n" + "=" * 100)
    print("📋 最终汇总表 (前3名平均±标准差)")
    print("=" * 100)
    print(f"{'数据集':<8} {'方法':<10} {'验证准确率':<15} {'验证F1':<15} {'验证Recall':<15}")
    print("-" * 100)
    
    for dataset in datasets:
        for method in methods:
            if summary_results[dataset][method] is not None:
                stats = summary_results[dataset][method]
                val_acc_str = f"{stats['val_acc_mean']:.2f}±{stats['val_acc_std']:.2f}"
                val_f1_str = f"{stats['val_f1_mean']:.2f}±{stats['val_f1_std']:.2f}"  
                val_recall_str = f"{stats['val_recall_mean']:.2f}±{stats['val_recall_std']:.2f}"
                print(f"{dataset:<8} {method.upper():<10} {val_acc_str:<15} {val_f1_str:<15} {val_recall_str:<15}")
            else:
                print(f"{dataset:<8} {method.upper():<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 100)


if __name__ == "__main__":
    extract_all_best_metrics()