#!/usr/bin/env python3
"""
提取3个数据集2个方法(lnsde, geosde)的最优训练/验证准确率、F1、Recall
"""

import json
import glob
import os
from collections import defaultdict


def extract_best_metrics():
    """提取最优指标"""
    
    # 方法映射
    method_mapping = {
        'linear_noise': 'lnsde',
        'geometric': 'geosde'
    }
    
    # 存储结果
    results = defaultdict(lambda: defaultdict(dict))
    
    # 查找所有日志文件
    log_files = glob.glob("/autodl-fs/data/lnsde-contiformer/results/**/*.log", recursive=True)
    
    print("=== 3个数据集2个方法的最优指标 ===")
    print("数据集: ASAS, LINEAR, MACHO")
    print("方法: lnsde(linear_noise), geosde(geometric)")
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
                
            if model_type not in ['linear_noise', 'geometric']:
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
            
            if best_epoch_data:
                # 提取指标
                metrics = {
                    'epoch': best_epoch_data.get('epoch', 0),
                    'train_acc': best_epoch_data.get('train_acc', 0),
                    'val_acc': best_epoch_data.get('val_acc', 0),
                    'train_f1': best_epoch_data.get('train_f1', 0),
                    'val_f1': best_epoch_data.get('val_f1', 0),
                    'train_recall': best_epoch_data.get('train_recall', 0),
                    'val_recall': best_epoch_data.get('val_recall', 0),
                    'file': log_file
                }
                
                # 如果已有该组合，选择验证准确率更高的
                if method_name not in results[dataset] or metrics['val_acc'] > results[dataset][method_name]['val_acc']:
                    results[dataset][method_name] = metrics
                    
        except Exception as e:
            print(f"处理文件 {log_file} 时出错: {e}")
    
    # 输出结果
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    methods = ['lnsde', 'geosde']
    
    print("=" * 80)
    for dataset in datasets:
        print(f"\n📊 {dataset} 数据集:")
        print("-" * 60)
        
        for method in methods:
            if method in results[dataset]:
                data = results[dataset][method]
                print(f"\n🔹 {method.upper()}:")
                print(f"  最优epoch: {data['epoch']}")
                print(f"  训练准确率: {data['train_acc']:.2f}%")
                print(f"  验证准确率: {data['val_acc']:.2f}%")
                print(f"  训练加权F1: {data['train_f1']:.2f}")
                print(f"  验证加权F1: {data['val_f1']:.2f}")
                print(f"  训练加权Recall: {data['train_recall']:.2f}")
                print(f"  验证加权Recall: {data['val_recall']:.2f}")
                print(f"  日志文件: {os.path.basename(data['file'])}")
            else:
                print(f"\n🔹 {method.upper()}: 暂无数据")
    
    print("\n" + "=" * 80)
    
    # 生成对比表格
    print("\n📋 汇总对比表:")
    print("-" * 80)
    print(f"{'数据集':<8} {'方法':<8} {'验证准确率':<10} {'验证F1':<10} {'验证Recall':<10}")
    print("-" * 80)
    
    for dataset in datasets:
        for method in methods:
            if method in results[dataset]:
                data = results[dataset][method]
                print(f"{dataset:<8} {method.upper():<8} {data['val_acc']:<10.2f} {data['val_f1']:<10.2f} {data['val_recall']:<10.2f}")
            else:
                print(f"{dataset:<8} {method.upper():<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 80)


if __name__ == "__main__":
    extract_best_metrics()