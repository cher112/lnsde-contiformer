#!/usr/bin/env python3
"""
从混淆矩阵重新计算多分类F1和Recall指标
分别分析LNSDE和GeoSDE两种方法
"""

import os
import json
import glob
from typing import Dict, List
import numpy as np

def calculate_multiclass_metrics(cm: List[List]) -> Dict[str, float]:
    """从多分类混淆矩阵计算准确率、macro平均和micro平均F1和Recall"""
    try:
        cm_array = np.array(cm, dtype=float)
        n_classes = cm_array.shape[0]
        
        # 计算总体准确率
        total_samples = np.sum(cm_array)
        correct_predictions = np.trace(cm_array)  # 对角线元素之和
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # 为每个类别计算precision, recall, f1
        class_precisions = []
        class_recalls = []
        class_f1s = []
        
        # 计算micro平均的总体TP, FP, FN
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i in range(n_classes):
            # 第i类的true positives
            tp = cm_array[i, i]
            
            # 第i类的false positives (其他类被错误预测为第i类)
            fp = np.sum(cm_array[:, i]) - tp
            
            # 第i类的false negatives (第i类被错误预测为其他类)
            fn = np.sum(cm_array[i, :]) - tp
            
            # 累积用于micro平均
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # 计算precision, recall, f1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1s.append(f1)
        
        # 计算macro平均
        macro_precision = np.mean(class_precisions)
        macro_recall = np.mean(class_recalls)
        macro_f1 = np.mean(class_f1s)
        
        # 计算micro平均 (对于多分类，micro F1 = micro Recall = micro Precision = Accuracy)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall, 
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'class_precisions': class_precisions,
            'class_recalls': class_recalls,
            'class_f1s': class_f1s,
            'n_classes': n_classes
        }
    except Exception as e:
        print(f"计算多分类指标出错: {e}")
        return {}

def extract_corrected_metrics(log_file: str) -> Dict:
    """提取并重新计算指标"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        method = data.get('model_type', 'unknown')
        dataset = data.get('dataset', 'unknown')
        
        metrics_list = []
        if 'epochs' in data:
            for epoch_data in data['epochs']:
                if 'confusion_matrix' in epoch_data:
                    cm = epoch_data['confusion_matrix']
                    
                    # 重新计算指标
                    metrics = calculate_multiclass_metrics(cm)
                    
                    if metrics:
                        metrics['epoch'] = epoch_data.get('epoch', 0)
                        metrics['method'] = method
                        metrics['dataset'] = dataset
                        
                        # 保留原始报告的指标用于对比
                        metrics['original_val_acc'] = epoch_data.get('val_acc', 0) / 100.0
                        metrics['original_val_f1'] = epoch_data.get('val_f1', 0) / 100.0
                        metrics['original_val_recall'] = epoch_data.get('val_recall', 0) / 100.0
                        
                        metrics_list.append(metrics)
        
        return {
            'method': method,
            'dataset': dataset, 
            'metrics': metrics_list,
            'log_file': os.path.basename(log_file)
        }
    
    except Exception as e:
        print(f"处理文件 {log_file} 出错: {e}")
        return {}

def analyze_dataset_corrected(dataset: str) -> Dict:
    """重新分析数据集指标"""
    log_pattern = f"/autodl-fs/data/lnsde-contiformer/results/**/{dataset}/**/logs/*.log"
    log_files = glob.glob(log_pattern, recursive=True)
    
    print(f"\n📊 重新分析 {dataset} 数据集")
    print(f"找到日志文件 {len(log_files)} 个")
    
    results_by_method = {
        'linear_noise': {'metrics': [], 'files': []},
        'geometric': {'metrics': [], 'files': []}
    }
    
    for log_file in log_files:
        result = extract_corrected_metrics(log_file)
        if result and result['metrics']:
            method = result['method']
            if method in results_by_method:
                results_by_method[method]['metrics'].extend(result['metrics'])
                results_by_method[method]['files'].append(result['log_file'])
                print(f"  {method}: {result['log_file']} - {len(result['metrics'])} epochs")
    
    # 为每种方法找最佳结果
    final_results = {}
    for method, data in results_by_method.items():
        if data['metrics']:
            best_f1 = max(data['metrics'], key=lambda x: x['macro_f1'])
            best_acc = max(data['metrics'], key=lambda x: x['accuracy']) 
            best_recall = max(data['metrics'], key=lambda x: x['macro_recall'])
            best_micro_f1 = max(data['metrics'], key=lambda x: x['micro_f1'])
            best_micro_recall = max(data['metrics'], key=lambda x: x['micro_recall'])
            
            final_results[method] = {
                'total_epochs': len(data['metrics']),
                'files': data['files'],
                'best_f1': best_f1,
                'best_accuracy': best_acc,
                'best_recall': best_recall,
                'best_micro_f1': best_micro_f1,
                'best_micro_recall': best_micro_recall
            }
        else:
            final_results[method] = {'total_epochs': 0, 'files': [], 'status': 'no_data'}
    
    return final_results

def main():
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    print("="*80)
    print("重新计算的多分类指标分析")
    print("="*80)
    
    all_results = {}
    
    for dataset in datasets:
        results = analyze_dataset_corrected(dataset)
        all_results[dataset] = results
        
        print(f"\n🎯 {dataset} 数据集结果汇总:")
        print("-" * 50)
        
        for method in ['linear_noise', 'geometric']:
            method_name = 'LNSDE' if method == 'linear_noise' else 'GeoSDE'
            print(f"\n📈 {method_name} 方法:")
            
            if results[method]['total_epochs'] == 0:
                print("   ❌ 无有效数据")
                continue
            
            print(f"   总训练轮次: {results[method]['total_epochs']}")
            print(f"   使用文件: {', '.join(results[method]['files'])}")
            
            best_f1 = results[method]['best_f1']
            print(f"   🏆 最佳Macro F1: {best_f1['macro_f1']:.4f} (Epoch {best_f1['epoch']})")
            print(f"      Macro召回率: {best_f1['macro_recall']:.4f}")
            print(f"      Micro F1: {best_f1['micro_f1']:.4f}")
            print(f"      Micro召回率: {best_f1['micro_recall']:.4f}")
            print(f"      准确率: {best_f1['accuracy']:.4f}")
            print(f"      Macro精确率: {best_f1['macro_precision']:.4f}")
            print(f"      分类数: {best_f1['n_classes']}")
            
            # 显示与原始指标的对比
            print(f"   📝 与原始报告对比:")
            print(f"      原始val_acc: {best_f1['original_val_acc']:.4f} -> 重算accuracy: {best_f1['accuracy']:.4f}")
            print(f"      原始val_f1: {best_f1['original_val_f1']:.4f} -> 重算macro_f1: {best_f1['macro_f1']:.4f}")
            print(f"      原始val_recall: {best_f1['original_val_recall']:.4f} -> 重算macro_recall: {best_f1['macro_recall']:.4f}")
    
    # 汇总表格
    print("\n" + "="*80)
    print("重新计算的方法对比汇总表")
    print("="*80)
    print(f"{'数据集':<10} {'方法':<10} {'Macro F1':<12} {'Micro F1':<12} {'Macro Recall':<15} {'Micro Recall':<15} {'准确率':<12}")
    print("-" * 95)
    
    for dataset in datasets:
        for method in ['linear_noise', 'geometric']:
            method_name = 'LNSDE' if method == 'linear_noise' else 'GeoSDE'
            results = all_results[dataset][method]
            
            if results['total_epochs'] > 0:
                macro_f1 = results['best_f1']['macro_f1']
                micro_f1 = results['best_micro_f1']['micro_f1']
                macro_recall = results['best_recall']['macro_recall']
                micro_recall = results['best_micro_recall']['micro_recall']
                best_acc = results['best_accuracy']['accuracy']
                print(f"{dataset:<10} {method_name:<10} {macro_f1:<12.4f} {micro_f1:<12.4f} {macro_recall:<15.4f} {micro_recall:<15.4f} {best_acc:<12.4f}")
            else:
                print(f"{dataset:<10} {method_name:<10} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12}")

if __name__ == '__main__':
    main()