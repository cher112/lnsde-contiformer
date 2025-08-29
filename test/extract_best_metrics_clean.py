#!/usr/bin/env python3
"""
从训练日志中提取每个数据集的最佳原始F1分数、召回率和准确率
基于混淆矩阵计算，而非macro平均
"""

import os
import json
import glob
from typing import Dict, List
import numpy as np

def calculate_metrics_from_cm(cm: List[List]) -> Dict[str, float]:
    """从混淆矩阵计算二分类指标（将多分类转换为二分类）"""
    try:
        cm_array = np.array(cm)
        
        # 对于多分类问题，我们将第一类（通常是变星类）作为正类，其他作为负类
        if cm_array.shape[0] > 2:
            # 将多分类转换为二分类
            # 正类（变星）：第一类
            # 负类：其他所有类
            tp = cm_array[0, 0]  # 变星被正确分类为变星
            fn = np.sum(cm_array[0, 1:])  # 变星被错误分类为其他类
            fp = np.sum(cm_array[1:, 0])  # 其他类被错误分类为变星
            tn = np.sum(cm_array[1:, 1:])  # 其他类被正确分类为其他类
        else:
            # 二分类情况
            tn, fp, fn, tp = cm_array.ravel()
        
        # 计算准确率
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # 计算召回率 (真正率)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    except Exception as e:
        print(f"计算混淆矩阵指标出错: {e}")
        return {}

def extract_metrics_from_json_log(log_file: str) -> List[Dict]:
    """从JSON格式的日志文件中提取所有epoch的指标"""
    metrics_list = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'epochs' in data:
            for epoch_data in data['epochs']:
                if 'confusion_matrix' in epoch_data:
                    cm = epoch_data['confusion_matrix']
                    metrics = calculate_metrics_from_cm(cm)
                    
                    if metrics:
                        metrics['epoch'] = epoch_data.get('epoch', 0)
                        metrics['log_file'] = os.path.basename(log_file)
                        metrics['val_acc_reported'] = epoch_data.get('val_acc', 0) / 100.0  # 转换为小数
                        metrics['val_f1_reported'] = epoch_data.get('val_f1', 0) / 100.0   # 转换为小数
                        metrics['val_recall_reported'] = epoch_data.get('val_recall', 0) / 100.0  # 转换为小数
                        metrics_list.append(metrics)
    
    except Exception as e:
        print(f"处理JSON日志文件 {log_file} 时出错: {e}")
    
    return metrics_list

def find_best_metrics_for_dataset(dataset: str) -> Dict:
    """查找指定数据集的最佳指标"""
    log_pattern = f"/root/autodl-tmp/lnsde-contiformer/results/**/{dataset}/**/logs/*.log"
    log_files = glob.glob(log_pattern, recursive=True)
    
    print(f"找到 {dataset} 数据集的日志文件: {len(log_files)} 个")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    all_metrics = []
    for log_file in log_files:
        metrics = extract_metrics_from_json_log(log_file)
        print(f"从 {os.path.basename(log_file)} 提取到 {len(metrics)} 个epoch的数据")
        all_metrics.extend(metrics)
    
    if not all_metrics:
        return {'dataset': dataset, 'status': 'no_data'}
    
    # 按F1分数排序找最佳
    best_by_f1 = max(all_metrics, key=lambda x: x['f1'])
    best_by_accuracy = max(all_metrics, key=lambda x: x['accuracy'])
    best_by_recall = max(all_metrics, key=lambda x: x['recall'])
    
    return {
        'dataset': dataset,
        'total_experiments': len(all_metrics),
        'best_f1': {
            'f1': best_by_f1['f1'],
            'recall': best_by_f1['recall'],
            'accuracy': best_by_f1['accuracy'],
            'precision': best_by_f1['precision'],
            'epoch': best_by_f1['epoch'],
            'log_file': best_by_f1['log_file'],
            'confusion_matrix': {
                'tp': int(best_by_f1['tp']),
                'tn': int(best_by_f1['tn']), 
                'fp': int(best_by_f1['fp']),
                'fn': int(best_by_f1['fn'])
            }
        },
        'best_accuracy': {
            'accuracy': best_by_accuracy['accuracy'],
            'f1': best_by_accuracy['f1'],
            'recall': best_by_accuracy['recall'],
            'precision': best_by_accuracy['precision'],
            'epoch': best_by_accuracy['epoch'],
            'log_file': best_by_accuracy['log_file']
        },
        'best_recall': {
            'recall': best_by_recall['recall'],
            'f1': best_by_recall['f1'],
            'accuracy': best_by_recall['accuracy'],
            'precision': best_by_recall['precision'],
            'epoch': best_by_recall['epoch'],
            'log_file': best_by_recall['log_file']
        }
    }

def main():
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    print("="*80)
    print("数据集最佳指标汇总报告")
    print("="*80)
    print()
    
    results_summary = {}
    
    for dataset in datasets:
        print(f"📊 {dataset} 数据集分析")
        print("-" * 40)
        
        results = find_best_metrics_for_dataset(dataset)
        results_summary[dataset] = results
        
        if results.get('status') == 'no_data':
            print(f"❌ 未找到 {dataset} 数据集的有效数据")
            print()
            continue
        
        print(f"📈 总实验次数: {results['total_experiments']}")
        print()
        
        # 最佳F1分数结果
        best_f1 = results['best_f1']
        print("🏆 最佳F1分数结果:")
        print(f"   F1 Score: {best_f1['f1']:.4f}")
        print(f"   召回率 (Recall): {best_f1['recall']:.4f}")
        print(f"   准确率 (Accuracy): {best_f1['accuracy']:.4f}")
        print(f"   精确率 (Precision): {best_f1['precision']:.4f}")
        print(f"   Epoch: {best_f1['epoch']}")
        print(f"   日志文件: {best_f1['log_file']}")
        print(f"   混淆矩阵: TP={best_f1['confusion_matrix']['tp']}, TN={best_f1['confusion_matrix']['tn']}, FP={best_f1['confusion_matrix']['fp']}, FN={best_f1['confusion_matrix']['fn']}")
        print()
        
        print()
    
    # 汇总表格
    print("="*80)
    print("三个数据集最佳指标汇总表")
    print("="*80)
    print(f"{'数据集':<10} {'最佳F1':<10} {'最佳召回率':<12} {'最佳准确率':<12}")
    print("-" * 50)
    
    for dataset in datasets:
        results = results_summary.get(dataset, {})
        if results.get('status') != 'no_data' and 'best_f1' in results:
            best_f1 = results['best_f1']['f1']
            best_recall = max(results['best_f1']['recall'], results['best_recall']['recall'])
            best_acc = max(results['best_f1']['accuracy'], results['best_accuracy']['accuracy'])
            print(f"{dataset:<10} {best_f1:<10.4f} {best_recall:<12.4f} {best_acc:<12.4f}")

if __name__ == '__main__':
    main()