#!/usr/bin/env python3
"""
简化评估脚本 - 直接处理原始数据
"""

import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sys
import os

sys.path.append('/root/autodl-tmp/lnsde-contiformer')
from models import LangevinSDEContiformer, LinearNoiseSDEContiformer, GeometricSDEContiformer

def create_simple_dataset(data_path):
    """创建简单的数据集，生成缺失的mask"""
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    processed_data = []
    labels = []
    
    for item in raw_data:
        # 提取原始数据
        times = np.array(item['time'], dtype=np.float32)
        mags = np.array(item['mag'], dtype=np.float32)
        errmags = np.array(item['errmag'], dtype=np.float32)
        
        # 生成mask - 假设所有数据都是有效的
        mask = np.ones(len(times), dtype=bool)
        
        # 构建时间序列特征 [time, mag, errmag]
        features = np.column_stack([times, mags, errmags])  # [seq_len, 3]
        
        processed_data.append({
            'features': torch.tensor(features, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': item['label']
        })
        labels.append(item['label'])
    
    # 重新映射标签到0开始的连续整数
    unique_labels = sorted(list(set(labels)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    for data_item in processed_data:
        data_item['label'] = label_map[data_item['label']]
    
    return processed_data, len(unique_labels)

def evaluate_single_model(model_path, data_path, dataset_name, model_name):
    """评估单个模型"""
    print(f"\n=== {model_name} 在 {dataset_name} 上的评估 ===")
    
    try:
        # 1. 加载和预处理数据
        dataset, num_classes = create_simple_dataset(data_path)
        print(f"数据集: {dataset_name}")
        print(f"样本数: {len(dataset)}")
        print(f"类别数: {num_classes}")
        
        # 2. 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # 3. 创建模型
        input_dim = 3  # time, mag, errmag
        
        if 'langevin' in model_path.lower():
            model = LangevinSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=num_classes,
                dropout=0.1
            )
        elif 'geometric' in model_path.lower():
            model = GeometricSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=num_classes,
                dropout=0.1
            )
        else:  # linear_noise
            model = LinearNoiseSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=num_classes,
                dropout=0.1
            )
        
        # 4. 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 5. 批量预测
        all_preds = []
        all_labels = []
        batch_size = 32
        
        print("开始预测...")
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_data = dataset[i:i+batch_size]
                
                # 准备批次数据
                features = torch.stack([item['features'] for item in batch_data]).to(device)
                labels = torch.tensor([item['label'] for item in batch_data]).to(device)
                
                # 模型预测
                outputs = model(features)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"已处理 {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size} 批次")
        
        # 6. 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"\n结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"加权F1: {f1_weighted:.4f}")
        print(f"加权召回率: {recall_weighted:.4f}")
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'samples': len(all_labels),
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'recall_weighted': recall_weighted
        }
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始模型评估...")
    
    # 数据路径
    data_paths = {
        'ASAS': '/root/autodl-tmp/db/ASAS/ASAS_processed.pkl',
        'LINEAR': '/root/autodl-tmp/db/LINEAR/LINEAR_processed.pkl',
        'MACHO': '/root/autodl-tmp/db/MACHO/MACHO_processed.pkl'
    }
    
    # 模型路径
    models = [
        ('/root/autodl-tmp/lnsde-contiformer/results/20250828/ASAS/2116/models/ASAS_linear_noise_best.pth', 'ASAS_linear_noise_0828'),
        ('/root/autodl-tmp/lnsde-contiformer/results/20250828/LINEAR/2116/models/LINEAR_linear_noise_best.pth', 'LINEAR_linear_noise_0828'),
        ('/root/autodl-tmp/lnsde-contiformer/results/20250828/MACHO/2116/models/MACHO_linear_noise_best.pth', 'MACHO_linear_noise_0828'),
        ('/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/models/MACHO_geometric_best.pth', 'MACHO_geometric_0829'),
        ('/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1756/models/MACHO_langevin_best.pth', 'MACHO_langevin_0829')
    ]
    
    results = []
    
    # 评估所有模型
    for model_path, model_name in models:
        for dataset_name, data_path in data_paths.items():
            result = evaluate_single_model(model_path, data_path, dataset_name, model_name)
            if result:
                results.append(result)
                print("-" * 60)
    
    # 汇总结果
    print("\n" + "="*80)
    print("最终结果汇总")
    print("="*80)
    
    for result in results:
        print(f"{result['model']} 在 {result['dataset']} 上:")
        print(f"  样本数: {result['samples']}")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  加权F1: {result['f1_weighted']:.4f}")
        print(f"  加权召回率: {result['recall_weighted']:.4f}")
        print()
    
    print("评估完成!")

if __name__ == "__main__":
    main()