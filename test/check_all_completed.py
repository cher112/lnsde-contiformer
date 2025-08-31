#!/usr/bin/env python3
"""提取所有已完成任务的完整指标"""

import json
import os
from glob import glob

def get_all_completed_metrics():
    """获取所有已完成任务的完整指标"""
    
    base_dir = "/root/autodl-tmp/lnsde-contiformer/results/20250830"
    
    # 查找所有相关日志文件
    patterns = [
        # Linear Noise SDE
        "**/ASAS_modellinear_noise_sde_only*.log",
        "**/LINEAR_modellinear_noise_sde_only*.log",
        "**/MACHO_modellinear_noise_sde_only*.log",
        # ContiFormer only  
        "**/ASAS_contiformer_only*.log",
        "**/LINEAR_contiformer_only*.log",
        "**/MACHO_contiformer_only*.log",
    ]
    
    results = []
    
    for pattern in patterns:
        log_files = glob(os.path.join(base_dir, pattern), recursive=True)
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                    
                    # 提取文件名信息
                    filename = os.path.basename(log_file)
                    dataset = filename.split('_')[0]
                    
                    if 'linear_noise' in filename:
                        model_type = "Linear Noise SDE"
                    elif 'contiformer_only' in filename:
                        model_type = "ContiFormer only"
                    else:
                        continue
                    
                    if 'epochs' in data and len(data['epochs']) > 0:
                        # 获取最佳准确率及其对应的F1和Recall
                        best_acc = 0
                        best_epoch_data = None
                        
                        for e in data['epochs']:
                            if e.get('val_acc', 0) > best_acc:
                                best_acc = e.get('val_acc', 0)
                                best_epoch_data = e
                        
                        if best_epoch_data:
                            results.append({
                                'dataset': dataset,
                                'model': model_type,
                                'acc': best_acc,
                                'f1': best_epoch_data.get('val_f1', 0),
                                'recall': best_epoch_data.get('val_recall', 0),
                                'epoch': best_epoch_data.get('epoch', '?')
                            })
                        
                except Exception as e:
                    print(f"处理 {log_file} 时出错: {e}")
    
    # 打印结果
    print("=" * 80)
    print("已完成的6个任务最终指标汇总")
    print("=" * 80)
    print()
    print(f"{'数据集':<10} {'模型':<20} {'准确率':>10} {'加权F1':>10} {'加权Recall':>12} {'最佳Epoch':>10}")
    print("-" * 80)
    
    # 按数据集和模型排序
    for r in sorted(results, key=lambda x: (x['dataset'], x['model'])):
        print(f"{r['dataset']:<10} {r['model']:<20} {r['acc']:>9.2f}% {r['f1']:>10.1f} {r['recall']:>12.1f} {r['epoch']:>10}")
    
    print("=" * 80)
    
    # 分组显示
    print("\n按模型类型分组：")
    print("-" * 80)
    
    print("\n【Linear Noise SDE-only】")
    for r in results:
        if r['model'] == "Linear Noise SDE":
            print(f"  {r['dataset']:<10}: 准确率={r['acc']:.2f}%, F1={r['f1']:.1f}, Recall={r['recall']:.1f}")
    
    print("\n【ContiFormer-only】")
    for r in results:
        if r['model'] == "ContiFormer only":
            print(f"  {r['dataset']:<10}: 准确率={r['acc']:.2f}%, F1={r['f1']:.1f}, Recall={r['recall']:.1f}")
    
    print("=" * 80)

if __name__ == "__main__":
    get_all_completed_metrics()