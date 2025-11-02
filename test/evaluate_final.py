#!/usr/bin/env python3
"""
正确的模型评估脚本
基于main.py的数据处理方式
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sys
import os

sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders
from models import LangevinSDEContiformer, LinearNoiseSDEContiformer, GeometricSDEContiformer

def evaluate_model(model_path, data_path, dataset_name, model_name):
    """评估单个模型"""
    print(f"\n=== 评估 {model_name} 在 {dataset_name} 数据集上的表现 ===")
    
    try:
        # 1. 加载数据
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=data_path, 
            batch_size=64, 
            num_workers=4,
            random_seed=535411460
        )
        
        print(f"数据集: {dataset_name}")
        print(f"类别数: {num_classes}")
        print(f"测试样本数: {len(test_loader.dataset)}")
        
        # 2. 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取模型参数
        if 'model_params' in checkpoint:
            model_params = checkpoint['model_params']
        else:
            model_params = {
                'hidden_channels': 128,
                'contiformer_dim': 128, 
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1
            }
        
        # 从数据加载器获取第一个batch来确定输入维度
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['features'].shape[-1]  # 应该是3: time, mag, errmag
        
        # 3. 创建对应的模型
        if 'langevin' in model_path.lower():
            model = LangevinSDEContiformer(
                input_dim=input_dim,
                hidden_channels=model_params.get('hidden_channels', 128),
                contiformer_dim=model_params.get('contiformer_dim', 128),
                n_heads=model_params.get('n_heads', 8),
                n_layers=model_params.get('n_layers', 6),
                num_classes=num_classes,
                dropout=model_params.get('dropout', 0.1)
            )
        elif 'geometric' in model_path.lower():
            model = GeometricSDEContiformer(
                input_dim=input_dim,
                hidden_channels=model_params.get('hidden_channels', 128),
                contiformer_dim=model_params.get('contiformer_dim', 128),
                n_heads=model_params.get('n_heads', 8),
                n_layers=model_params.get('n_layers', 6),
                num_classes=num_classes,
                dropout=model_params.get('dropout', 0.1)
            )
        else:  # linear_noise
            model = LinearNoiseSDEContiformer(
                input_dim=input_dim,
                hidden_channels=model_params.get('hidden_channels', 128),
                contiformer_dim=model_params.get('contiformer_dim', 128),
                n_heads=model_params.get('n_heads', 8),
                n_layers=model_params.get('n_layers', 6),
                num_classes=num_classes,
                dropout=model_params.get('dropout', 0.1)
            )
        
        # 4. 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 5. 预测
        all_preds = []
        all_labels = []
        
        print("开始预测...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 提取数据
                features = batch['features'].to(device)  # [batch, seq_len, 3]
                labels = batch['label'].to(device)
                
                # 模型预测
                outputs = model(features)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
        
        # 6. 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n结果:")
        print(f"样本数: {len(all_labels)}")
        print(f"准确率: {accuracy:.4f}")
        print(f"加权F1: {f1_weighted:.4f}")
        print(f"加权召回率: {recall_weighted:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'samples': len(all_labels),
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'recall_weighted': recall_weighted,
            'confusion_matrix': cm.tolist()
        }
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始模型评估...")
    
    # 数据路径配置
    data_paths = {
        'ASAS': '/root/autodl-tmp/db/ASAS/ASAS_processed.pkl',
        'LINEAR': '/root/autodl-tmp/db/LINEAR/LINEAR_processed.pkl',
        'MACHO': '/root/autodl-tmp/db/MACHO/MACHO_processed.pkl'
    }
    
    # 模型路径配置
    models_0828 = [
        ('/autodl-fs/data/lnsde-contiformer/results/20250828/ASAS/2116/models/ASAS_linear_noise_best.pth', 'ASAS_linear_noise_0828'),
        ('/autodl-fs/data/lnsde-contiformer/results/20250828/LINEAR/2116/models/LINEAR_linear_noise_best.pth', 'LINEAR_linear_noise_0828'),
        ('/autodl-fs/data/lnsde-contiformer/results/20250828/MACHO/2116/models/MACHO_linear_noise_best.pth', 'MACHO_linear_noise_0828')
    ]
    
    models_0829 = [
        ('/autodl-fs/data/lnsde-contiformer/results/20250829/MACHO/1359/models/MACHO_geometric_best.pth', 'MACHO_geometric_0829'),
        ('/autodl-fs/data/lnsde-contiformer/results/20250829/MACHO/1756/models/MACHO_langevin_best.pth', 'MACHO_langevin_0829')
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("测试 0828 训练的最佳模型")
    print("="*60)
    
    # 评估0828模型
    for model_path, model_name in models_0828:
        for dataset_name, data_path in data_paths.items():
            result = evaluate_model(model_path, data_path, dataset_name, model_name)
            if result:
                results.append(result)
                print("-" * 60)
    
    print("\n" + "="*60)
    print("测试 0829 MACHO 训练的最佳模型")
    print("="*60)
    
    # 评估0829模型
    for model_path, model_name in models_0829:
        for dataset_name, data_path in data_paths.items():
            result = evaluate_model(model_path, data_path, dataset_name, model_name)
            if result:
                results.append(result)
                print("-" * 60)
    
    # 汇总结果
    print("\n" + "="*80)
    print("最终结果汇总")
    print("="*80)
    
    for result in results:
        print(f"模型: {result['model']}")
        print(f"数据集: {result['dataset']}")
        print(f"样本数: {result['samples']}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"加权F1: {result['f1_weighted']:.4f}")
        print(f"加权召回率: {result['recall_weighted']:.4f}")
        print("-" * 50)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()