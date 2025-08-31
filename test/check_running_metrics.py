#!/usr/bin/env python3
"""检查当前运行任务的最新指标"""

import json
import os
from glob import glob
from datetime import datetime

def get_latest_metrics():
    """获取最新的训练指标"""
    
    # 目标日志文件模式
    target_patterns = [
        # Langevin SDE-only
        "**/ASAS_langevin_sde_only*.log",
        "**/LINEAR_langevin_sde_only*.log", 
        "**/MACHO_langevin_sde_only*.log",
        # Geometric SDE-only
        "**/ASAS_geometric_sde_only*.log",
        "**/LINEAR_geometric_sde_only*.log",
        "**/MACHO_geometric_sde_only*.log",
    ]
    
    base_dir = "/autodl-fs/data/lnsde-contiformer/results/20250830"
    
    print("=" * 80)
    print("当前运行的6个任务最新指标")
    print("=" * 80)
    
    for pattern in target_patterns:
        log_files = glob(os.path.join(base_dir, pattern), recursive=True)
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                    
                    # 提取文件名信息
                    filename = os.path.basename(log_file)
                    dataset = filename.split('_')[0]
                    model_type = filename.split('_')[1]
                    
                    if 'epochs' in data and len(data['epochs']) > 0:
                        # 获取最新的epoch数据
                        last_epoch = data['epochs'][-1]
                        epoch_num = last_epoch.get('epoch', '?')
                        
                        # 获取最佳结果
                        best_acc = max([e.get('val_acc', 0) for e in data['epochs']])
                        best_f1 = max([e.get('val_f1', 0) for e in data['epochs']])
                        best_recall = max([e.get('val_recall', 0) for e in data['epochs']])
                        
                        # 当前epoch结果
                        current_acc = last_epoch.get('val_acc', 0)
                        current_f1 = last_epoch.get('val_f1', 0)
                        current_recall = last_epoch.get('val_recall', 0)
                        
                        print(f"\n【{dataset} - {model_type.upper()} SDE-only】")
                        print(f"  进度: Epoch {epoch_num}/50")
                        print(f"  当前: Acc={current_acc:.2f}%, F1={current_f1:.1f}, Recall={current_recall:.1f}")
                        print(f"  最佳: Acc={best_acc:.2f}%, F1={best_f1:.1f}, Recall={best_recall:.1f}")
                        
                except Exception as e:
                    pass
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    get_latest_metrics()