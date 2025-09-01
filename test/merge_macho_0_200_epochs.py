#!/usr/bin/env python3
import json
import os
from datetime import datetime

def merge_macho_0_200_epochs():
    """合并MACHO 0-100和101-200 epoch的训练日志"""
    
    # 定义日志文件路径
    log1_path = "/root/autodl-tmp/lnsde-contiformer/results/20250901/MACHO/2046/logs/MACHO_modellinear_noise_sde_cf_config1_lr1e-04_bs80_hc128_cd256.log"
    log2_path = "/root/autodl-tmp/lnsde-contiformer/results/20250901/MACHO/2205/logs/MACHO_modellinear_noise_sde_cf_config1_lr1e-04_bs80_hc128_cd256.log"
    
    # 读取第一个日志文件 (epoch 1-100)
    print("读取第一个日志文件 (epoch 1-100)...")
    with open(log1_path, 'r') as f:
        data1 = json.load(f)
    
    # 读取第二个日志文件 (epoch 101-200)
    print("读取第二个日志文件 (epoch 101-200)...")
    with open(log2_path, 'r') as f:
        data2 = json.load(f)
    
    # 创建合并后的数据结构
    merged_data = {
        "dataset": data1["dataset"],
        "model_type": data1["model_type"],
        "sde_config": data1["sde_config"],
        "start_time": data1["start_time"],
        "date": data1["date"],
        "epochs": []
    }
    
    # 合并epochs数据
    print("合并epochs数据...")
    merged_data["epochs"].extend(data1["epochs"])
    merged_data["epochs"].extend(data2["epochs"])
    
    # 验证epoch连续性
    print("验证epoch连续性...")
    epochs = [ep["epoch"] for ep in merged_data["epochs"]]
    expected_epochs = list(range(1, 201))
    
    if epochs == expected_epochs:
        print("✓ Epoch连续性验证通过 (1-200)")
    else:
        print("⚠ Epoch连续性验证失败")
        print(f"期望: {expected_epochs[:5]}...{expected_epochs[-5:]}")
        print(f"实际: {epochs[:5]}...{epochs[-5:]}")
    
    # 保存合并后的文件
    output_dir = "/root/autodl-tmp/lnsde-contiformer/results/20250901/MACHO/"
    output_file = "MACHO_0_200_epochs_merged.log"
    output_path = os.path.join(output_dir, output_file)
    
    print(f"保存合并后的数据到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # 统计信息
    print("\n=== 合并完成 ===")
    print(f"总epochs: {len(merged_data['epochs'])}")
    print(f"数据集: {merged_data['dataset']}")
    print(f"模型类型: {merged_data['model_type']}")
    print(f"开始时间: {merged_data['start_time']}")
    
    # 显示训练进展统计
    first_epoch = merged_data['epochs'][0]
    last_epoch = merged_data['epochs'][-1]
    print(f"\n第1个epoch - 训练准确率: {first_epoch['train_acc']:.2f}%, 验证准确率: {first_epoch['val_acc']:.2f}%")
    print(f"第200个epoch - 训练准确率: {last_epoch['train_acc']:.2f}%, 验证准确率: {last_epoch['val_acc']:.2f}%")
    
    return merged_data, output_path

if __name__ == "__main__":
    merge_macho_0_200_epochs()