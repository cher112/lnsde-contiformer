#!/usr/bin/env python3
import json
import os
from datetime import datetime

def merge_macho_logs():
    log_dir = "/autodl-fs/data/lnsde-contiformer/results/logs/MACHO"
    
    # 读取三个MACHO log文件
    log_files = [
        "MACHO_linear_noise_config1_20250824_135530.log",
        "MACHO_linear_noise_config1_20250825_101031.log", 
        "MACHO_linear_noise_config1_20250825_212939.log"
    ]
    
    merged_data = None
    
    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if merged_data is None:
            # 初始化合并数据结构
            merged_data = {
                "experiment_info": {
                    "dataset": "MACHO",
                    "model_type": "linear_noise",
                    "sde_config": 1,
                    "start_time": "20250824_135530",  # 使用第一个文件的开始时间
                    "log_file": "MACHO_linear_noise_config1_merged.log"
                },
                "training_history": {
                    "epochs": [],
                    "train_loss": [],
                    "train_accuracy": [],
                    "val_loss": [],
                    "val_accuracy": [],
                    "class_accuracy_history": [],
                    "learning_rates": []
                },
                "best_metrics": {
                    "best_val_accuracy": 0.0,
                    "best_epoch": 0,
                    "best_class_accuracy": {}
                },
                "last_update": ""
            }
        
        # 合并训练历史数据
        merged_data["training_history"]["epochs"].extend(data["training_history"]["epochs"])
        merged_data["training_history"]["train_loss"].extend(data["training_history"]["train_loss"])
        merged_data["training_history"]["train_accuracy"].extend(data["training_history"]["train_accuracy"])
        merged_data["training_history"]["val_loss"].extend(data["training_history"]["val_loss"])
        merged_data["training_history"]["val_accuracy"].extend(data["training_history"]["val_accuracy"])
        merged_data["training_history"]["class_accuracy_history"].extend(data["training_history"]["class_accuracy_history"])
        merged_data["training_history"]["learning_rates"].extend(data["training_history"]["learning_rates"])
        
        # 更新最佳指标
        if "best_metrics" in data and data["best_metrics"]["best_val_accuracy"] > merged_data["best_metrics"]["best_val_accuracy"]:
            merged_data["best_metrics"] = data["best_metrics"]
        
        # 更新最后更新时间
        merged_data["last_update"] = data["last_update"]
    
    # 保存合并后的数据
    merged_file_path = os.path.join(log_dir, "MACHO_linear_noise_config1_merged.log")
    with open(merged_file_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"合并完成，保存到: {merged_file_path}")
    print(f"总epochs: {len(merged_data['training_history']['epochs'])}")
    print(f"最佳验证准确率: {merged_data['best_metrics']['best_val_accuracy']:.2f}%")
    
    return merged_data

if __name__ == "__main__":
    merge_macho_logs()