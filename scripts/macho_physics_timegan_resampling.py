#!/usr/bin/env python3
"""
为MACHO数据集应用物理约束TimeGAN重采样
生成高质量合成样本以提升分类准确率
"""

import sys
import os
import numpy as np
import pickle
import torch
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import HybridResampler


def load_macho_data():
    """加载MACHO原始数据"""
    data_path = "/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl"
    
    print(f"🔄 加载MACHO数据: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 加载完成: {len(data)}个样本")
    
    # 分析类别分布
    labels = [item['label'] for item in data]
    class_counts = Counter(labels)
    class_names = {item['label']: item['class_name'] for item in data}
    
    print(f"\n📊 MACHO原始数据分布:")
    total_samples = sum(class_counts.values())
    
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        class_name = class_names.get(label, f'Unknown_{label}')
        percentage = count / total_samples * 100
        print(f"  类别 {label} ({class_name}): {count:3d} 样本 ({percentage:5.1f}%)")
    
    print(f"  总计: {total_samples} 样本")
    
    return data


def convert_to_training_format(data):
    """将MACHO数据转换为训练格式"""
    print(f"\n🔄 转换数据格式...")
    
    X_list = []
    y_list = []
    times_list = []
    masks_list = []
    periods_list = []
    
    for item in data:
        # 提取时间序列数据 (seq_len, 3) [time, mag, errmag]
        time_data = item['time'].astype(np.float32)
        mag_data = item['mag'].astype(np.float32)
        errmag_data = item['errmag'].astype(np.float32)
        mask_data = item['mask'].astype(bool)
        
        # 构建特征矩阵
        features = np.column_stack([time_data, mag_data, errmag_data])
        X_list.append(features)
        y_list.append(item['label'])
        times_list.append(time_data)
        masks_list.append(mask_data)
        periods_list.append(item['period'])
    
    # 转换为numpy数组
    X = np.array(X_list)
    y = np.array(y_list)
    times = np.array(times_list)
    masks = np.array(masks_list)
    periods = np.array(periods_list)
    
    print(f"✅ 转换完成:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  数据类型: {X.dtype}")
    
    return X, y, times, masks, periods


def design_optimal_resampling_strategy(class_counts):
    """设计最优的重采样策略，平衡效果和训练效率"""
    
    print(f"\n🎯 设计重采样策略...")
    
    # 计算基础统计
    total_samples = sum(class_counts.values())
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    print(f"  最大类别: {max_count} 样本")
    print(f"  最小类别: {min_count} 样本") 
    print(f"  不平衡率: {max_count/min_count:.2f}")
    
    # 设计策略：
    # 1. 对于极少数类（<100），扩展到200-300样本
    # 2. 对于中等少数类（100-400），扩展到400-500样本  
    # 3. 对于多数类，保持原样或适度减少
    
    target_strategy = {}
    
    for label, count in class_counts.items():
        if count < 100:
            # 极少数类：大幅扩展
            target = min(300, max_count * 0.8)  # 不超过最大类的80%
        elif count < 400:
            # 中等少数类：适度扩展
            target = min(500, max_count * 0.9)
        else:
            # 多数类：保持不变
            target = count
            
        target_strategy[label] = int(target)
    
    print(f"\n📋 重采样策略:")
    for label in sorted(target_strategy.keys()):
        original = class_counts[label]
        target = target_strategy[label]
        change = target - original
        change_pct = (change / original * 100) if original > 0 else 0
        
        status = "增加" if change > 0 else ("减少" if change < 0 else "保持")
        print(f"  类别 {label}: {original:3d} → {target:3d} ({status} {abs(change):3d}, {change_pct:+5.1f}%)")
    
    total_target = sum(target_strategy.values())
    print(f"  总样本: {total_samples} → {total_target} (增加 {total_target - total_samples})")
    
    return target_strategy


def apply_physics_timegan_resampling(X, y, times, masks, periods, target_strategy):
    """应用物理约束TimeGAN重采样"""
    
    print(f"\n🧬 开始物理约束TimeGAN重采样...")
    print("=" * 60)
    
    # 创建重采样器 - 使用适中的参数平衡效果和速度
    resampler = HybridResampler(
        smote_k_neighbors=5,
        enn_n_neighbors=3,
        sampling_strategy=target_strategy,
        synthesis_mode='physics_timegan',  # 使用物理约束TimeGAN
        apply_enn=False,  # 暂时禁用ENN以加快速度
        noise_level=0.05,
        physics_weight=0.2,  # 适中的物理约束权重
        random_state=535411460
    )
    
    # 执行重采样
    try:
        X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
            X, y, times, masks
        )
        
        print("✅ 物理约束TimeGAN重采样成功完成!")
        
        # 统计重采样结果
        final_counts = Counter(y_resampled.tolist() if torch.is_tensor(y_resampled) else y_resampled)
        
        print(f"\n📊 重采样结果统计:")
        for label in sorted(final_counts.keys()):
            count = final_counts[label]
            target = target_strategy.get(label, 0)
            diff = count - target
            print(f"  类别 {label}: {count:3d} 样本 (目标: {target}, 差异: {diff:+d})")
        
        total_final = sum(final_counts.values())
        print(f"  最终总样本: {total_final}")
        
        return X_resampled, y_resampled, times_resampled, masks_resampled, final_counts
        
    except Exception as e:
        print(f"❌ 物理约束TimeGAN重采样失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def convert_back_to_original_format(X_resampled, y_resampled, times_resampled, masks_resampled, original_data):
    """将重采样数据转换回原始MACHO数据格式"""
    
    print(f"\n🔄 转换回原始数据格式...")
    
    # 构建类别到类别名的映射
    label_to_class_name = {}
    for item in original_data:
        label_to_class_name[item['label']] = item['class_name']
    
    resampled_data = []
    n_samples = len(y_resampled)
    
    for i in range(n_samples):
        # 提取重采样数据
        if torch.is_tensor(X_resampled):
            features = X_resampled[i].cpu().numpy()
            label = y_resampled[i].cpu().numpy().item()
        else:
            features = X_resampled[i]
            label = y_resampled[i]
            
        time_data = features[:, 0]
        mag_data = features[:, 1] 
        errmag_data = features[:, 2]
        
        # 获取对应的时间和掩码
        if times_resampled is not None:
            if torch.is_tensor(times_resampled):
                time_data = times_resampled[i].cpu().numpy()
            else:
                time_data = times_resampled[i]
                
        if masks_resampled is not None:
            if torch.is_tensor(masks_resampled):
                mask_data = masks_resampled[i].cpu().numpy().astype(bool)
            else:
                mask_data = masks_resampled[i].astype(bool)
        else:
            # 基于时间数据生成掩码
            mask_data = (time_data > -1000) & (time_data < 1e10)
        
        # 数据质量修正
        # 1. 确保时间数据合理
        valid_mask = mask_data.astype(bool)
        time_data[~valid_mask] = -1e9
        mag_data[~valid_mask] = 0.0
        
        # 2. 确保误差数据非负且合理
        errmag_data = np.abs(errmag_data)
        errmag_data = np.clip(errmag_data, 0.001, 1.0)  # 限制在合理范围
        errmag_data[~valid_mask] = 0.0
        
        # 计算统计信息
        valid_points = valid_mask.sum()
        
        # 随机选择一个同类别的原始样本作为模板（用于period等参数）
        same_class_samples = [s for s in original_data if s['label'] == label]
        if same_class_samples:
            template_sample = np.random.choice(same_class_samples)
            period = template_sample['period']
            coverage = valid_points / len(mask_data)
        else:
            period = np.float64(1.0)  # 默认周期
            coverage = valid_points / len(mask_data)
        
        # 构建与原始格式完全一致的样本
        resampled_sample = {
            # 核心数据
            'time': time_data.astype(np.float64),
            'mag': mag_data.astype(np.float64),
            'errmag': errmag_data.astype(np.float64),
            'mask': mask_data.astype(bool),
            'period': np.float64(period),
            'label': int(label),
            
            # 元数据
            'file_id': f'timegan_resampled_{i:06d}.dat',
            'original_length': int(valid_points),
            'valid_points': np.int64(valid_points),
            'coverage': np.float64(coverage),
            'class_name': label_to_class_name.get(label, f'class_{label}')
        }
        
        resampled_data.append(resampled_sample)
    
    print(f"✅ 格式转换完成: {len(resampled_data)} 个样本")
    
    return resampled_data


def save_resampled_data(resampled_data, save_path):
    """保存重采样数据"""
    
    print(f"\n💾 保存重采样数据...")
    
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump(resampled_data, f)
    
    print(f"✅ 重采样数据已保存至: {save_path}")
    
    # 验证保存的数据
    print(f"🔍 验证保存的数据...")
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    print(f"  验证: 保存了{len(saved_data)}个样本")
    
    if saved_data:
        # 检查第一个样本的格式
        sample = saved_data[0]
        print(f"  样本键: {list(sample.keys())}")
        
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: {type(value).__name__} {value.shape} {value.dtype}")
            else:
                print(f"    {key}: {type(value).__name__} = {value}")
    
    # 统计最终分布
    final_counts = Counter([s['label'] for s in saved_data])
    print(f"\n📊 最终保存的类别分布:")
    for label in sorted(final_counts.keys()):
        count = final_counts[label]
        class_name = saved_data[0]['class_name'] if saved_data else 'Unknown'
        # 找到对应类别名
        for s in saved_data:
            if s['label'] == label:
                class_name = s['class_name']
                break
        print(f"  类别 {label} ({class_name}): {count} 样本")
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")
    
    return save_path


def main():
    """主函数"""
    print("🚀 MACHO数据集物理约束TimeGAN重采样")
    print("=" * 60)
    
    try:
        # 1. 加载MACHO数据
        original_data = load_macho_data()
        
        # 2. 转换为训练格式
        X, y, times, masks, periods = convert_to_training_format(original_data)
        
        # 3. 分析并设计重采样策略
        class_counts = Counter(y)
        target_strategy = design_optimal_resampling_strategy(class_counts)
        
        # 4. 应用物理约束TimeGAN重采样
        X_resampled, y_resampled, times_resampled, masks_resampled, final_counts = apply_physics_timegan_resampling(
            X, y, times, masks, periods, target_strategy
        )
        
        if X_resampled is None:
            print("❌ 重采样失败，程序退出")
            return
        
        # 5. 转换回原始格式
        resampled_data = convert_back_to_original_format(
            X_resampled, y_resampled, times_resampled, masks_resampled, original_data
        )
        
        # 6. 保存到指定位置
        save_path = "/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl"
        saved_path = save_resampled_data(resampled_data, save_path)
        
        print(f"\n🎉 MACHO物理约束TimeGAN重采样完成!")
        print("=" * 60)
        print(f"✅ 原始数据: {len(original_data)} 样本")
        print(f"✅ 重采样后: {len(resampled_data)} 样本")
        print(f"✅ 保存位置: {saved_path}")
        print(f"✅ 可直接用于训练，预期分类准确率显著提升")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()