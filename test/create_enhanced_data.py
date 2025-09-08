#!/usr/bin/env python3
"""
方案3：混合数据补充策略
- 从original恢复高质量样本
- 对少数类进行SMOTE重采样
- 保持训练速度的同时最大化信息
"""

import pickle
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import os
import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

def recover_high_quality_samples(original_data, fixed_data, dataset_name):
    """从original中恢复高质量样本"""
    
    # 获取fixed中的file_ids
    fixed_ids = set(s['file_id'] for s in fixed_data)
    
    # 找出被移除的样本
    removed_samples = []
    for sample in original_data:
        if sample['file_id'] not in fixed_ids:
            removed_samples.append(sample)
    
    if not removed_samples:
        return []
    
    # 根据数据集设置恢复策略
    if dataset_name == 'MACHO':
        # MACHO: 重点恢复LPV(3)和MOA(4)类
        target_classes = {3: 150, 4: 100}  # LPV恢复150个，MOA恢复100个
        recovered = []
        
        for class_label, target_count in target_classes.items():
            class_samples = [s for s in removed_samples if s['label'] == class_label]
            
            # 按质量指标排序
            class_samples.sort(key=lambda x: (
                x.get('coverage', 0),  # 覆盖率优先
                len(x['time']),  # 序列长度次之
                -np.mean(x['errmag']) if np.any(x['errmag'] > 0) else float('inf')  # 误差合理性
            ), reverse=True)
            
            # 选择高质量样本
            for sample in class_samples[:target_count]:
                # 修复零误差问题
                if np.any(sample['errmag'] == 0):
                    sample['errmag'] = np.where(sample['errmag'] == 0, 0.01, sample['errmag'])
                
                # 只选择质量较好的
                if sample.get('coverage', 0) > 0.5 and len(sample['time']) > 100:
                    recovered.append(sample)
                    if len(recovered) >= target_count:
                        break
        
        print(f"MACHO: 从{len(removed_samples)}个被移除样本中恢复{len(recovered)}个高质量样本")
        
    elif dataset_name == 'ASAS':
        # ASAS: 主要恢复DSCT类
        recovered = []
        dsct_samples = [s for s in removed_samples if s['label'] == 3]  # DSCT类
        
        # 按质量排序
        dsct_samples.sort(key=lambda x: (
            x.get('coverage', 0),
            len(x['time'])
        ), reverse=True)
        
        # 恢复50个高质量DSCT样本
        for sample in dsct_samples[:50]:
            if np.any(sample['errmag'] == 0):
                sample['errmag'] = np.where(sample['errmag'] == 0, 0.01, sample['errmag'])
            
            if sample.get('coverage', 0) > 0.7 and len(sample['time']) > 150:
                recovered.append(sample)
        
        print(f"ASAS: 从{len(removed_samples)}个被移除样本中恢复{len(recovered)}个高质量样本")
        
    else:  # LINEAR
        recovered = []
        print(f"LINEAR: 无需恢复（无样本损失）")
    
    return recovered


def smote_augment(data, target_classes, n_samples_per_class):
    """对指定类别进行SMOTE增强"""
    
    augmented_samples = []
    
    for class_label, n_new in n_samples_per_class.items():
        if class_label not in target_classes:
            continue
            
        # 获取该类别的样本
        class_samples = [s for s in data if s['label'] == class_label]
        
        if len(class_samples) < 2:
            continue
        
        print(f"  为类别{class_label}生成{n_new}个SMOTE样本（原有{len(class_samples)}个）")
        
        # 提取特征用于KNN
        features = []
        for s in class_samples:
            # 使用统计特征
            feat = [
                np.mean(s['mag']),
                np.std(s['mag']),
                np.mean(s['errmag']),
                s.get('period', 0),
                len(s['time']),
                s.get('coverage', 1)
            ]
            features.append(feat)
        
        features = np.array(features)
        
        # KNN找最近邻
        k = min(5, len(class_samples) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
        
        # 生成新样本
        for _ in range(n_new):
            # 随机选择一个样本
            idx = np.random.randint(len(class_samples))
            base_sample = class_samples[idx]
            
            # 找最近邻
            distances, indices = nbrs.kneighbors([features[idx]])
            neighbor_idx = np.random.choice(indices[0][1:])  # 排除自己
            neighbor_sample = class_samples[neighbor_idx]
            
            # 插值生成新样本
            alpha = np.random.random()
            new_sample = {
                'time': base_sample['time'].copy(),
                'mag': base_sample['mag'] * alpha + neighbor_sample['mag'][:len(base_sample['mag'])] * (1-alpha),
                'errmag': base_sample['errmag'] * alpha + neighbor_sample['errmag'][:len(base_sample['errmag'])] * (1-alpha),
                'mask': base_sample['mask'].copy(),
                'period': base_sample['period'] * alpha + neighbor_sample['period'] * (1-alpha),
                'label': class_label,
                'file_id': f"SMOTE_{class_label}_{len(augmented_samples)}",
                'original_length': base_sample['original_length'],
                'valid_points': base_sample['valid_points'],
                'coverage': base_sample['coverage'],
                'class_name': base_sample['class_name']
            }
            
            # 添加少量噪声
            noise_level = 0.01
            new_sample['mag'] += np.random.normal(0, noise_level, len(new_sample['mag']))
            
            augmented_samples.append(new_sample)
    
    return augmented_samples


def create_enhanced_datasets():
    """创建增强数据集"""
    
    print("="*70)
    print("创建增强数据集 (方案3: 混合策略)")
    print("="*70)
    
    datasets = ['MACHO', 'LINEAR', 'ASAS']
    
    for dataset_name in datasets:
        print(f"\n处理 {dataset_name} 数据集...")
        print("-"*50)
        
        # 加载数据
        original_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_original.pkl'
        fixed_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_fixed.pkl'
        enhanced_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_enhanced.pkl'
        
        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)
        with open(fixed_path, 'rb') as f:
            fixed_data = pickle.load(f)
        
        # 从fixed数据开始
        enhanced_data = fixed_data.copy()
        
        # 步骤1: 恢复高质量样本
        recovered = recover_high_quality_samples(original_data, fixed_data, dataset_name)
        enhanced_data.extend(recovered)
        
        # 步骤2: SMOTE增强少数类
        if dataset_name == 'MACHO':
            # 统计当前类别分布
            labels = [s['label'] for s in enhanced_data]
            counts = Counter(labels)
            print(f"\n当前类别分布: {dict(counts)}")
            
            # SMOTE增强
            target_classes = {0, 5}  # Be(0)和QSO(5)
            n_samples = {0: 20, 5: 15}  # Be生成20个，QSO生成15个
            
            print("\nSMOTE增强:")
            augmented = smote_augment(enhanced_data, target_classes, n_samples)
            enhanced_data.extend(augmented)
        
        # 保存增强数据集
        with open(enhanced_path, 'wb') as f:
            pickle.dump(enhanced_data, f)
        
        # 统计信息
        print(f"\n最终统计:")
        print(f"  Original: {len(original_data)} 样本")
        print(f"  Fixed: {len(fixed_data)} 样本")
        print(f"  Enhanced: {len(enhanced_data)} 样本")
        print(f"  增幅: +{(len(enhanced_data)/len(fixed_data)-1)*100:.1f}%")
        
        # 类别分布
        final_labels = [s['label'] for s in enhanced_data]
        final_counts = Counter(final_labels)
        print(f"  最终类别分布: {dict(sorted(final_counts.items()))}")
    
    print("\n" + "="*70)
    print("增强数据集创建完成！")
    print("="*70)
    
    print("\n使用方法:")
    print("python main.py --dataset 3 --enhanced  # 使用MACHO增强数据集")
    print("\n或手动指定:")
    print("python main.py --data_path /root/autodl-fs/lnsde-contiformer/data/MACHO_enhanced.pkl")


if __name__ == "__main__":
    create_enhanced_datasets()