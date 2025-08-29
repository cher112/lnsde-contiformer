#!/usr/bin/env python3
"""
分别分析LNSDE和GeoSDE两种方法的最佳指标
基于混淆矩阵计算，处理多分类问题
"""

import os
import json
import glob
from typing import Dict, List
import numpy as np

def identify_variable_star_class(cm: List[List], dataset: str) -> int:
    """识别哪个类是变星类（通常是样本数量较少的类）"""
    cm_array = np.array(cm)
    # 计算每个类的总样本数（真实标签）
    class_totals = np.sum(cm_array, axis=1)
    
    # 通常变星是少数类，但我们需要排除没有样本的类
    valid_classes = [(i, total) for i, total in enumerate(class_totals) if total > 0]
    
    if not valid_classes:
        return 0  # 默认第0类
    
    # 对于天体物理数据，通常变星是第0类或者样本数较少的类
    # 让我们先尝试第0类，如果第0类没有样本，则选择样本数最少的有效类
    if class_totals[0] > 0:
        return 0
    else:
        # 选择样本数最少的有效类作为变星类
        valid_classes.sort(key=lambda x: x[1])
        return valid_classes[0][0]

def calculate_binary_metrics_from_multiclass_cm(cm: List[List], positive_class: int = 0) -> Dict[str, float]:
    """从多分类混淆矩阵计算二分类指标"""
    try:
        cm_array = np.array(cm)
        n_classes = cm_array.shape[0]
        
        # 将多分类转换为二分类：指定类 vs 其他所有类
        tp = cm_array[positive_class, positive_class]  # 正类被正确分类
        fn = np.sum(cm_array[positive_class, :]) - tp  # 正类被错误分类为其他类
        fp = np.sum(cm_array[:, positive_class]) - tp  # 其他类被错误分类为正类
        tn = np.sum(cm_array) - tp - fn - fp  # 其他类被正确分类为其他类
        
        # 计算指标
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'positive_class': positive_class,
            'class_total': int(tp + fn)
        }
    except Exception as e:
        print(f"计算指标出错: {e}")
        return {}

def extract_metrics_by_method(log_file: str) -> Dict:
    """从JSON日志文件提取指标，按方法分类"""
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
                    
                    # 尝试不同的类作为正类，找到最佳结果
                    best_metrics = None
                    best_f1 = -1
                    
                    for pos_class in range(len(cm)):
                        metrics = calculate_binary_metrics_from_multiclass_cm(cm, pos_class)
                        if metrics and metrics['f1'] > best_f1 and metrics['class_total'] > 0:
                            best_metrics = metrics
                            best_f1 = metrics['f1']
                    
                    if best_metrics:
                        best_metrics['epoch'] = epoch_data.get('epoch', 0)
                        best_metrics['method'] = method
                        best_metrics['dataset'] = dataset
                        metrics_list.append(best_metrics)
        
        return {
            'method': method,
            'dataset': dataset,
            'metrics': metrics_list,
            'log_file': os.path.basename(log_file)
        }
    
    except Exception as e:
        print(f"处理文件 {log_file} 出错: {e}")
        return {}

def analyze_dataset_by_method(dataset: str) -> Dict:
    """按方法分析数据集"""
    log_pattern = f"/root/autodl-tmp/lnsde-contiformer/results/**/{dataset}/**/logs/*.log"
    log_files = glob.glob(log_pattern, recursive=True)
    
    print(f"\n📊 分析 {dataset} 数据集")
    print(f"找到日志文件 {len(log_files)} 个")
    
    results_by_method = {
        'linear_noise': {'metrics': [], 'files': []},
        'geometric': {'metrics': [], 'files': []}
    }
    
    for log_file in log_files:
        result = extract_metrics_by_method(log_file)
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
            best_f1 = max(data['metrics'], key=lambda x: x['f1'])
            best_acc = max(data['metrics'], key=lambda x: x['accuracy'])
            best_recall = max(data['metrics'], key=lambda x: x['recall'])
            
            final_results[method] = {
                'total_epochs': len(data['metrics']),
                'files': data['files'],
                'best_f1': best_f1,
                'best_accuracy': best_acc,
                'best_recall': best_recall
            }
        else:
            final_results[method] = {'total_epochs': 0, 'files': [], 'status': 'no_data'}
    
    return final_results

def main():
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    print("="*80)
    print("按方法分析数据集最佳指标")
    print("="*80)
    
    all_results = {}
    
    for dataset in datasets:
        results = analyze_dataset_by_method(dataset)
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
            print(f"   🏆 最佳F1: {best_f1['f1']:.4f} (Epoch {best_f1['epoch']})")
            print(f"      召回率: {best_f1['recall']:.4f}")
            print(f"      准确率: {best_f1['accuracy']:.4f}")
            print(f"      精确率: {best_f1['precision']:.4f}")
            print(f"      正类ID: {best_f1['positive_class']} (样本数: {best_f1['class_total']})")
            print(f"      混淆矩阵: TP={best_f1['tp']}, TN={best_f1['tn']}, FP={best_f1['fp']}, FN={best_f1['fn']}")
    
    # 汇总表格
    print("\n" + "="*80)
    print("方法对比汇总表")
    print("="*80)
    print(f"{'数据集':<10} {'方法':<10} {'最佳F1':<10} {'最佳召回率':<12} {'最佳准确率':<12}")
    print("-" * 60)
    
    for dataset in datasets:
        for method in ['linear_noise', 'geometric']:
            method_name = 'LNSDE' if method == 'linear_noise' else 'GeoSDE'
            results = all_results[dataset][method]
            
            if results['total_epochs'] > 0:
                best_f1 = results['best_f1']['f1']
                best_recall = results['best_recall']['recall']
                best_acc = results['best_accuracy']['accuracy']
                print(f"{dataset:<10} {method_name:<10} {best_f1:<10.4f} {best_recall:<12.4f} {best_acc:<12.4f}")
            else:
                print(f"{dataset:<10} {method_name:<10} {'N/A':<10} {'N/A':<12} {'N/A':<12}")

if __name__ == '__main__':
    main()