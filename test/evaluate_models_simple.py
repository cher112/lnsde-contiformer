#!/usr/bin/env python3
"""
简化的模型评估脚本
直接加载模型和数据进行评估
"""

import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sys
import os

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

def load_data(file_path):
    """加载pkl数据文件"""
    print(f"加载数据: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 处理字典列表格式
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # 提取fit_lc作为特征，label作为标签
        features = []
        labels = []
        
        for item in data:
            if 'fit_lc' in item and 'label' in item:
                features.append(item['fit_lc'])
                labels.append(item['label'])
        
        # 转换为numpy数组
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"成功加载数据: features={features.shape}, labels={labels.shape}")
        print(f"标签分布: {np.bincount(labels)}")
        
        return features, labels
    
    # 其他格式
    elif isinstance(data, tuple):
        features, labels = data
    elif isinstance(data, dict):
        features = data.get('features', data.get('X', None))
        labels = data.get('labels', data.get('y', None))
    else:
        print(f"未支持的数据格式: {type(data)}")
        return None, None
    
    return features, labels

def evaluate_model_simple(model_path, data_path, model_type='linear_noise'):
    """简化的模型评估函数"""
    print(f"\n=== 评估模型: {os.path.basename(model_path)} ===")
    print(f"数据文件: {os.path.basename(data_path)}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在")
        return None
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在")
        return None
    
    try:
        # 加载数据
        features, labels = load_data(data_path)
        if features is None or labels is None:
            print("错误: 无法解析数据文件")
            return None
        
        # 转换为torch tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        
        print(f"数据形状: features={features.shape}, labels={labels.shape}")
        print(f"标签分布: {torch.bincount(labels)}")
        
        # 加载模型检查点
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"检查点键: {list(checkpoint.keys())}")
        
        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 从状态字典推断模型架构
        input_dim = None
        num_classes = 3  # 默认类别数
        
        # 检查 feature_encoder 第一层 
        if 'feature_encoder.0.weight' in state_dict:
            encoder_input_dim = state_dict['feature_encoder.0.weight'].shape[1]
            print(f"特征编码器期望输入维度: {encoder_input_dim}")
            
            # 如果数据维度不匹配，需要调整
            if encoder_input_dim == 3 and features.shape[1] == 256:
                print("数据维度不匹配，使用前3个特征")
                features = features[:, :3]  # 只使用前3个特征
                input_dim = 3
            else:
                input_dim = encoder_input_dim
        else:
            print("错误: 无法找到feature_encoder参数")
            return None
        
        # 从分类器层推断类别数
        if 'classifier.3.weight' in state_dict:
            num_classes = state_dict['classifier.3.weight'].shape[0]
            print(f"推断类别数: {num_classes}")
        
        print(f"最终输入维度: {input_dim}, 类别数: {num_classes}")
        
        # 导入模型类
        from models import LangevinSDEContiformer, LinearNoiseSDEContiformer, GeometricSDEContiformer
        
        # 创建模型实例
        if 'langevin' in model_path.lower():
            model_class = LangevinSDEContiformer
        elif 'geometric' in model_path.lower():
            model_class = GeometricSDEContiformer
        else:
            model_class = LinearNoiseSDEContiformer
        
        model = model_class(
            input_dim=input_dim,
            hidden_channels=128,
            contiformer_dim=128,
            n_heads=8,
            n_layers=6,
            num_classes=num_classes,
            dropout=0.1
        )
        
        # 加载权重
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # 准备数据
        features = features.to(device)
        labels = labels.to(device)
        
        # 批量预测
        batch_size = 64
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                # 模型预测
                outputs = model(batch_features)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"加权F1: {f1_weighted:.4f}")
        print(f"加权召回率: {recall_weighted:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        return {
            'model': os.path.basename(model_path),
            'data': os.path.basename(data_path),
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'recall_weighted': recall_weighted,
            'samples': len(all_labels)
        }
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始评估模型...")
    
    # 数据文件路径 (使用fixed数据)
    data_files = {
        'ASAS': '/root/autodl-tmp/db/ASAS/ASAS_processed.pkl',
        'LINEAR': '/root/autodl-tmp/db/LINEAR/LINEAR_processed.pkl', 
        'MACHO': '/root/autodl-tmp/db/MACHO/MACHO_processed.pkl'
    }
    
    # 0828年训练的最佳模型
    models_0828 = [
        '/root/autodl-tmp/lnsde-contiformer/results/20250828/ASAS/2116/models/ASAS_linear_noise_best.pth',
        '/root/autodl-tmp/lnsde-contiformer/results/20250828/LINEAR/2116/models/LINEAR_linear_noise_best.pth',
        '/root/autodl-tmp/lnsde-contiformer/results/20250828/MACHO/2116/models/MACHO_linear_noise_best.pth'
    ]
    
    # 0829年MACHO训练的最佳模型
    models_0829 = [
        '/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/models/MACHO_geometric_best.pth',
        '/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1756/models/MACHO_langevin_best.pth'
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("测试 0828 训练的最佳模型")
    print("="*60)
    
    # 评估0828模型
    for model_path in models_0828:
        for dataset_name, data_path in data_files.items():
            result = evaluate_model_simple(model_path, data_path)
            if result:
                results.append(result)
                print("-" * 40)
    
    print("\n" + "="*60)
    print("测试 0829 MACHO 训练的最佳模型")
    print("="*60)
    
    # 评估0829模型
    for model_path in models_0829:
        for dataset_name, data_path in data_files.items():
            result = evaluate_model_simple(model_path, data_path)
            if result:
                results.append(result)
                print("-" * 40)
    
    # 汇总结果
    print("\n" + "="*80)
    print("所有模型评估结果汇总")
    print("="*80)
    
    for result in results:
        print(f"模型: {result['model']}")
        print(f"数据集: {result['data']}")
        print(f"样本数: {result['samples']}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"加权F1: {result['f1_weighted']:.4f}")
        print(f"加权召回率: {result['recall_weighted']:.4f}")
        print("-" * 40)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()