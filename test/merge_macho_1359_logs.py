#!/usr/bin/env python3
"""
MACHO 1359数据集日志合并脚本
合并MACHO 1359的训练日志，从21到50 epoch
"""
import json
import os
from datetime import datetime

def merge_macho_1359_logs():
    """合并MACHO 1359的训练日志文件"""
    log_file = "/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/logs/MACHO_geometric_config1.log"
    
    if not os.path.exists(log_file):
        print(f"未找到日志文件: {log_file}")
        return None
    
    print(f"读取日志文件: {log_file}")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # 创建合并后的数据结构
    merged_data = {
        "experiment_info": {
            "dataset": "MACHO",
            "model_type": "geometric", 
            "sde_config": 1,
            "start_time": data.get("start_time", "20250829_135920"),
            "date": data.get("date", "20250829"),
            "log_file": "MACHO_1359_geometric_config1_merged.log",
            "note": "包含epochs 21-50的训练数据"
        },
        "training_history": {
            "epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "train_f1": [],
            "train_recall": [],
            "val_f1": [],
            "val_recall": [],
            "class_accuracy_history": [],
            "learning_rates": [],
            "epoch_times": [],
            "confusion_matrices": []
        },
        "best_metrics": {
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
            "best_val_f1": 0.0,
            "best_train_accuracy": 0.0,
            "best_train_f1": 0.0
        },
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 处理训练历史数据
    epochs_data = data.get("epochs", [])
    
    for epoch_info in epochs_data:
        epoch = epoch_info.get("epoch")
        merged_data["training_history"]["epochs"].append(epoch)
        merged_data["training_history"]["train_loss"].append(epoch_info.get("train_loss", 0))
        merged_data["training_history"]["train_accuracy"].append(epoch_info.get("train_acc", 0))
        merged_data["training_history"]["val_loss"].append(epoch_info.get("val_loss", 0))
        merged_data["training_history"]["val_accuracy"].append(epoch_info.get("val_acc", 0))
        merged_data["training_history"]["train_f1"].append(epoch_info.get("train_f1", 0))
        merged_data["training_history"]["train_recall"].append(epoch_info.get("train_recall", 0))
        merged_data["training_history"]["val_f1"].append(epoch_info.get("val_f1", 0))
        merged_data["training_history"]["val_recall"].append(epoch_info.get("val_recall", 0))
        merged_data["training_history"]["learning_rates"].append(epoch_info.get("learning_rate", 0))
        merged_data["training_history"]["epoch_times"].append(epoch_info.get("epoch_time", 0))
        merged_data["training_history"]["confusion_matrices"].append(epoch_info.get("confusion_matrix", []))
        
        # 处理类别准确率
        if "class_accuracies" in epoch_info:
            merged_data["training_history"]["class_accuracy_history"].append(epoch_info["class_accuracies"])
        else:
            merged_data["training_history"]["class_accuracy_history"].append({})
    
    # 更新最佳指标
    val_accuracies = merged_data["training_history"]["val_accuracy"]
    val_f1_scores = merged_data["training_history"]["val_f1"]
    train_accuracies = merged_data["training_history"]["train_accuracy"] 
    train_f1_scores = merged_data["training_history"]["train_f1"]
    
    if val_accuracies:
        best_val_acc_idx = val_accuracies.index(max(val_accuracies))
        merged_data["best_metrics"]["best_val_accuracy"] = max(val_accuracies)
        merged_data["best_metrics"]["best_epoch"] = merged_data["training_history"]["epochs"][best_val_acc_idx]
    
    if val_f1_scores:
        merged_data["best_metrics"]["best_val_f1"] = max(val_f1_scores)
    
    if train_accuracies:
        merged_data["best_metrics"]["best_train_accuracy"] = max(train_accuracies)
        
    if train_f1_scores:
        merged_data["best_metrics"]["best_train_f1"] = max(train_f1_scores)
    
    # 保存合并后的数据
    output_dir = "/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/logs"
    merged_file_path = os.path.join(output_dir, "MACHO_1359_geometric_config1_merged.log")
    
    with open(merged_file_path, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"合并完成，保存到: {merged_file_path}")
    print(f"训练epochs范围: {min(merged_data['training_history']['epochs'])} - {max(merged_data['training_history']['epochs'])}")
    print(f"总epochs: {len(merged_data['training_history']['epochs'])}")
    print(f"最佳验证准确率: {merged_data['best_metrics']['best_val_accuracy']:.2f}%")
    print(f"最佳验证F1: {merged_data['best_metrics']['best_val_f1']:.2f}")
    
    return merged_data

if __name__ == "__main__":
    merge_macho_1359_logs()