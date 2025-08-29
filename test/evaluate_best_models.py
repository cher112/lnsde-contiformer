#!/usr/bin/env python3
"""
评估最佳模型脚本
用于测试0828和0829训练的最好模型在整个数据集上的表现
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sys
import os

sys.path.append('/root/autodl-tmp/lnsde-contiformer')
from utils import LightCurveDataset
from models import LangevinSDEContiformer, LinearNoiseSDEContiformer, GeometricSDEContiformer
from torch.utils.data import DataLoader


def load_model_and_evaluate(model_path, dataset_name, data_path):
    """加载模型并在指定数据集上评估"""
    print(f"\n=== 评估模型: {model_path} ===")
    print(f"数据集: {dataset_name}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return None
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 - {data_path}")
        return None
    
    try:
        # 加载数据集
        test_dataset = LightCurveDataset(data_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取第一个batch来确定输入维度
        first_batch = next(iter(test_loader))
        input_dim = first_batch[0].shape[-1]
        
        # 根据模型路径确定模型类型
        if 'linear_noise' in model_path:
            model = LinearNoiseSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=3,
                dropout=0.1
            )
        elif 'geometric' in model_path:
            model = GeometricSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=3,
                dropout=0.1
            )
        elif 'langevin' in model_path:
            model = LangevinSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=3,
                dropout=0.1
            )
        else:
            # 默认使用LinearNoiseSDEContiformer
            model = LinearNoiseSDEContiformer(
                input_dim=input_dim,
                hidden_channels=128,
                contiformer_dim=128,
                n_heads=8,
                n_layers=6,
                num_classes=3,
                dropout=0.1
            )
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # 进行预测
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                
                # 模型预测
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n结果:")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"加权平均F1分数: {f1_weighted:.4f}")
        print(f"加权平均召回率: {recall_weighted:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        return {
            'model_path': model_path,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'recall_weighted': recall_weighted,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(test_dataset)
        }
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("开始评估最佳模型...")
    
    # 数据文件路径
    data_files = {
        'ASAS': '/root/autodl-tmp/db/ASAS/ASAS_processed.pkl',
        'LINEAR': '/root/autodl-tmp/db/LINEAR/LINEAR_processed.pkl', 
        'MACHO': '/root/autodl-tmp/db/MACHO/MACHO_processed.pkl'
    }
    
    # 0828年训练的最佳模型路径
    models_0828 = {
        'ASAS': '/root/autodl-tmp/lnsde-contiformer/results/20250828/ASAS/2116/models/ASAS_linear_noise_best.pth',
        'LINEAR': '/root/autodl-tmp/lnsde-contiformer/results/20250828/LINEAR/2116/models/LINEAR_linear_noise_best.pth',
        'MACHO': '/root/autodl-tmp/lnsde-contiformer/results/20250828/MACHO/2116/models/MACHO_linear_noise_best.pth'
    }
    
    # 0829年MACHO训练的最佳模型路径
    models_0829 = {
        'MACHO_geometric': '/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/models/MACHO_geometric_best.pth',
        'MACHO_langevin': '/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1756/models/MACHO_langevin_best.pth'
    }
    
    results = []
    
    print("\n" + "="*60)
    print("测试 0828 训练的最佳模型")
    print("="*60)
    
    # 评估0828模型
    for dataset_name, model_path in models_0828.items():
        for test_dataset, data_path in data_files.items():
            result = load_model_and_evaluate(model_path, f"{dataset_name}_model_on_{test_dataset}", data_path)
            if result:
                results.append(result)
    
    print("\n" + "="*60)
    print("测试 0829 MACHO 训练的最佳模型")
    print("="*60)
    
    # 评估0829模型
    for model_name, model_path in models_0829.items():
        for test_dataset, data_path in data_files.items():
            result = load_model_and_evaluate(model_path, f"{model_name}_on_{test_dataset}", data_path)
            if result:
                results.append(result)
    
    # 汇总结果
    print("\n" + "="*80)
    print("所有模型评估结果汇总")
    print("="*80)
    
    for result in results:
        print(f"\n模型: {os.path.basename(result['model_path'])}")
        print(f"数据集: {result['dataset']}")
        print(f"样本数: {result['total_samples']}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"加权F1: {result['f1_weighted']:.4f}")
        print(f"加权召回率: {result['recall_weighted']:.4f}")
        print("-" * 40)
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()