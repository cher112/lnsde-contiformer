#!/usr/bin/env python3
"""
LINEAR数据集快速重采样 - 专门优化类别0和1
使用混合重采样避免TimeGAN的复杂度问题
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import HybridResampler

def configure_chinese_font():
    """配置中文字体显示"""
    import matplotlib.font_manager as fm
    
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return True

configure_chinese_font()

def pad_sequences_to_fixed_length(X_list, max_length=512):
    """将变长序列填充到固定长度"""
    print(f"📏 将序列统一填充到长度: {max_length}")
    
    n_features = X_list[0].shape[1]  # 通常是3 [time, mag, error]
    padded_X = np.zeros((len(X_list), max_length, n_features), dtype=np.float32)
    
    for i, seq in enumerate(X_list):
        seq_len = min(len(seq), max_length)
        padded_X[i, :seq_len, :] = seq[:seq_len]
    
    return padded_X

def apply_linear_hybrid_resampling():
    """应用LINEAR数据集混合重采样，专门优化类别0和1"""
    print(f"🚀 LINEAR混合重采样 - 专门优化类别0和1")
    print("=" * 50)
    
    # 加载原始数据
    data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据: {len(data)}样本")
    
    # 转换为训练格式
    X_list, y_list = [], []
    
    for item in data:
        # 提取特征
        mask = item['mask'].astype(bool)
        times = item['time'][mask]
        mags = item['mag'][mask]
        errors = item['errmag'][mask]
        
        if len(times) < 10:  # 过滤太短的序列
            continue
        
        # 构建特征矩阵 [time, mag, error]
        features = np.column_stack([times, mags, errors])
        
        X_list.append(features)
        y_list.append(item['label'])
    
    # 统一序列长度
    X = pad_sequences_to_fixed_length(X_list, max_length=512)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"有效样本: {len(X)}个")
    print(f"数据形状: {X.shape}")
    print(f"类别分布: {Counter(y.tolist())}")
    
    # 分析类别分布
    class_counts = Counter(y.tolist())
    print(f"\n📊 原始类别分布:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        class_name_map = {0: 'Beta_Persei', 1: 'Delta_Scuti', 2: 'RR_Lyrae_FM', 3: 'RR_Lyrae_FO', 4: 'W_Ursae_Maj'}
        name = class_name_map.get(cls, f'Class_{cls}')
        print(f"  类别{cls} ({name}): {count}样本")
    
    # 重采样策略 - 重点增强类别0和1
    target_strategy = {
        0: 400,   # Beta_Persei: 291 → 400 (+37%)
        1: 300,   # Delta_Scuti: 70 → 300 (+328%) 重点增强  
        2: 2234,  # RR_Lyrae_FM: 保持不变
        3: 749,   # RR_Lyrae_FO: 保持不变
        4: 1860   # W_Ursae_Maj: 保持不变
    }
    
    print(f"\n🎯 重采样目标:")
    for cls, target_count in target_strategy.items():
        current = class_counts.get(cls, 0)
        increase = target_count - current
        increase_rate = increase / current if current > 0 else 0
        print(f"  类别{cls}: {current} → {target_count} (+{increase}, +{increase_rate:.0%})")
    
    # 使用混合重采样器 - 避免TimeGAN复杂度
    resampler = HybridResampler(
        synthesis_mode='hybrid',  # 使用混合模式而非TimeGAN
        smote_k_neighbors=8,      # 增加邻居数提升质量
        noise_level=0.02,         # 低噪声保持数据质量
        sampling_strategy=target_strategy,
        apply_enn=True,           # 应用ENN清理
        random_state=42
    )
    
    print(f"\n📈 开始混合重采样...")
    print(f"配置: 混合模式, 8邻居, ENN清理")
    
    # 执行重采样 - HybridResampler返回4个值
    X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(X, y)
    
    print(f"\n✅ 重采样完成!")
    print(f"重采样后样本数: {len(X_resampled)}")
    print(f"重采样后类别分布: {Counter(y_resampled)}")
    
    # 统计效果
    original_counts = Counter(y)
    resampled_counts = Counter(y_resampled)
    
    print(f"\n📊 增强效果统计:")
    for cls in sorted(set(y)):
        orig_count = original_counts[cls]
        resamp_count = resampled_counts[cls]
        increase = resamp_count - orig_count
        increase_rate = increase / orig_count if orig_count > 0 else 0
        
        print(f"  类别{cls}: {orig_count} → {resamp_count} (+{increase}, +{increase_rate:.0%})")
    
    return X_resampled, y_resampled

def convert_to_standard_format(X_resampled, y_resampled, original_data):
    """将重采样数据转换为标准pkl格式"""
    print(f"\n🔄 转换为标准数据格式...")
    
    # 创建类别名称映射
    class_name_mapping = {}
    for item in original_data:
        class_name_mapping[item['label']] = item['class_name']
    
    print(f"类别映射: {class_name_mapping}")
    
    # 转换重采样数据
    converted_data = []
    
    for i, (features, label) in enumerate(zip(X_resampled, y_resampled)):
        # features: [512, 3] - [time, mag, error]，但只有前面部分是有效的
        
        # 找到有效数据的结束位置（假设时间为0表示padding）
        valid_mask = features[:, 0] != 0  # 时间不为0的点是有效的
        if not np.any(valid_mask):
            valid_mask[0] = True  # 至少保留一个点
        
        times = features[valid_mask, 0].astype(np.float32)
        mags = features[valid_mask, 1].astype(np.float32)
        errors = features[valid_mask, 2].astype(np.float32)
        
        seq_len = len(times)
        
        # 创建mask
        mask = np.ones(seq_len, dtype=bool)
        
        # 估计周期
        if seq_len > 1:
            time_span = times.max() - times.min()
            estimated_period = time_span / max(1, seq_len // 10)
        else:
            estimated_period = 1.0
        
        # 构建标准格式的数据项
        data_item = {
            'time': times,
            'mag': mags,
            'errmag': errors,
            'mask': mask,
            'period': np.float32(estimated_period),
            'label': int(label),
            'class_name': class_name_mapping.get(label, f'Class_{label}'),
            'file_id': f'hybrid_generated_{i}',
            'original_length': seq_len,
            'valid_points': seq_len,
            'coverage': 1.0
        }
        
        converted_data.append(data_item)
    
    print(f"✅ 转换完成: {len(converted_data)}样本")
    return converted_data

def save_linear_resampled_data(converted_data):
    """保存LINEAR重采样数据"""
    print(f"\n💾 保存LINEAR重采样数据...")
    
    # 使用和TimeGAN相同的命名模式以便兼容
    output_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_resample_hybrid.pkl'
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)
    
    print(f"✅ 数据已保存至: {output_path}")
    
    # 验证保存的数据
    with open(output_path, 'rb') as f:
        verified_data = pickle.load(f)
    
    print(f"验证: 加载了{len(verified_data)}样本")
    
    # 检查类别分布
    labels = [item['label'] for item in verified_data]
    class_counts = Counter(labels)
    print(f"最终类别分布: {dict(class_counts)}")
    
    return output_path

def main():
    """主函数"""
    print("🚀 LINEAR混合重采样 - 专门优化类别0和1")
    print("=" * 60)
    
    # 1. 应用混合重采样
    X_resampled, y_resampled = apply_linear_hybrid_resampling()
    
    # 2. 转换为标准格式
    original_data = []
    with open('/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    converted_data = convert_to_standard_format(X_resampled, y_resampled, original_data)
    
    # 3. 保存数据
    output_path = save_linear_resampled_data(converted_data)
    
    print(f"\n🎉 LINEAR重采样完成!")
    print(f"📁 数据文件: {output_path}")
    print(f"\n🎯 主要改进:")
    print(f"  • 类别0 (Beta_Persei): 291 → 400样本 (+37%)")
    print(f"  • 类别1 (Delta_Scuti): 70 → 300样本 (+328%)")
    print(f"  • 使用混合重采样技术提升类别区分性")
    
    print(f"\n💡 使用方法:")
    print(f"python main.py --dataset 2 --use_resampling --resampled_data_path {output_path}")

if __name__ == "__main__":
    main()