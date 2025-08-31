#!/usr/bin/env python3
"""提取已完成任务的最终指标"""

import json
import os
from glob import glob

def get_completed_metrics():
    """获取已完成任务的最终指标"""
    
    # 已完成的任务日志
    completed_logs = {
        # Linear Noise SDE-only
        "ASAS Linear Noise SDE": "**/ASAS_linear_noise_sde_only*.log",
        "LINEAR Linear Noise SDE": "**/LINEAR_linear_noise_sde_only*.log", 
        "MACHO Linear Noise SDE": "**/MACHO_linear_noise_sde_only*.log",
        # ContiFormer-only
        "ASAS ContiFormer": "**/ASAS_contiformer_only*.log",
        "LINEAR ContiFormer": "**/LINEAR_contiformer_only*.log",
        "MACHO ContiFormer": "**/MACHO_contiformer_only*.log",
    }
    
    base_dir = "/root/autodl-tmp/lnsde-contiformer/results/20250830"
    
    print("=" * 80)
    print("已完成的6个任务最终指标")
    print("=" * 80)
    
    results = []
    
    for task_name, pattern in completed_logs.items():
        log_files = glob(os.path.join(base_dir, pattern), recursive=True)
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'epochs' in data and len(data['epochs']) > 0:
                        # 获取最佳结果
                        best_acc = max([e.get('val_acc', 0) for e in data['epochs']])
                        best_f1 = max([e.get('val_f1', 0) for e in data['epochs']])
                        best_recall = max([e.get('val_recall', 0) for e in data['epochs']])
                        
                        # 找到最佳准确率的epoch
                        best_epoch = None
                        for e in data['epochs']:
                            if e.get('val_acc', 0) == best_acc:
                                best_epoch = e
                                break
                        
                        if best_epoch:
                            epoch_num = best_epoch.get('epoch', '?')
                            # 使用最佳epoch的F1和Recall
                            epoch_f1 = best_epoch.get('val_f1', 0)
                            epoch_recall = best_epoch.get('val_recall', 0)
                            
                            results.append({
                                'name': task_name,
                                'acc': best_acc,
                                'f1': epoch_f1,
                                'recall': epoch_recall,
                                'epoch': epoch_num
                            })
                            
                            print(f"\n【{task_name}】")
                            print(f"  最佳Epoch: {epoch_num}")
                            print(f"  准确率: {best_acc:.2f}%")
                            print(f"  加权F1: {epoch_f1:.1f}")
                            print(f"  加权Recall: {epoch_recall:.1f}")
                        
                except Exception as e:
                    print(f"  处理 {task_name} 时出错: {e}")
    
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)
    print(f"{'模型':<30} {'准确率':>10} {'F1':>10} {'Recall':>10}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x['name']):
        print(f"{r['name']:<30} {r['acc']:>9.2f}% {r['f1']:>10.1f} {r['recall']:>10.1f}")
    
    print("=" * 80)

if __name__ == "__main__":
    get_completed_metrics()