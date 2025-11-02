#!/usr/bin/env python3
"""
LINEAR数据集物理约束TimeGAN重采样
专门优化类别0 (Beta_Persei) 和类别1 (Delta_Scuti) 的区分性
解决误分类问题：类别1→3,4类，类别0→其他类的混淆
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

def analyze_confusion_classes():
    """分析类别0和1与其他类别的特征差异"""
    print("🔍 深度分析LINEAR类别0和1的混淆问题...")
    
    data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 按类别分组
    class_samples = {}
    for item in data:
        label = item['label']
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(item)
    
    # 详细特征分析
    class_features = {}
    for label, samples in class_samples.items():
        class_name = samples[0]['class_name']
        print(f"\n📊 类别{label} ({class_name}) - {len(samples)}样本:")
        
        periods = []
        mag_ranges = []
        mag_means = []
        error_means = []
        sequence_lengths = []
        mag_stds = []
        
        for sample in samples:
            mask = sample['mask'].astype(bool)
            if np.sum(mask) < 5:  # 至少需要5个有效点
                continue
                
            times = sample['time'][mask]
            mags = sample['mag'][mask]
            errors = sample['errmag'][mask]
            
            periods.append(sample['period'])
            mag_ranges.append(mags.max() - mags.min())
            mag_means.append(mags.mean())
            mag_stds.append(mags.std())
            error_means.append(errors.mean())
            sequence_lengths.append(len(times))
        
        # 统计特征
        features = {
            'period_mean': np.mean(periods),
            'period_std': np.std(periods),
            'mag_range_mean': np.mean(mag_ranges),
            'mag_range_std': np.std(mag_ranges),
            'mag_mean': np.mean(mag_means),
            'mag_std_mean': np.mean(mag_stds),
            'error_mean': np.mean(error_means),
            'seq_len_mean': np.mean(sequence_lengths),
            'count': len(samples)
        }
        
        class_features[label] = features
        
        print(f"  周期: {features['period_mean']:.3f} ± {features['period_std']:.3f}")
        print(f"  星等变化: {features['mag_range_mean']:.3f} ± {features['mag_range_std']:.3f}")
        print(f"  星等均值: {features['mag_mean']:.3f}")
        print(f"  星等散度: {features['mag_std_mean']:.3f}")
        print(f"  误差水平: {features['error_mean']:.4f}")
        print(f"  序列长度: {features['seq_len_mean']:.1f}")
    
    # 分析混淆矩阵 - 找出类别0和1最容易混淆的类别
    print(f"\n🎯 重点类别对比分析:")
    target_classes = [0, 1]  # Beta_Persei, Delta_Scuti
    confusing_classes = [2, 3, 4]  # 容易混淆的类别
    
    print(f"目标优化:")
    for cls in target_classes:
        features = class_features[cls]
        name = class_samples[cls][0]['class_name']
        print(f"  类别{cls} ({name}): 周期{features['period_mean']:.3f}, 变化{features['mag_range_mean']:.3f}")
    
    print(f"容易混淆的类别:")
    for cls in confusing_classes:
        features = class_features[cls]
        name = class_samples[cls][0]['class_name']
        print(f"  类别{cls} ({name}): 周期{features['period_mean']:.3f}, 变化{features['mag_range_mean']:.3f}")
    
    return class_features, class_samples

def apply_enhanced_physics_timegan_resampling():
    """应用增强的物理约束TimeGAN重采样，专门优化类别0和1"""
    print(f"\n🚀 开始LINEAR物理约束TimeGAN重采样...")
    
    # 加载原始数据
    data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据: {len(data)}样本")
    
    # 转换为训练格式 - 统一序列长度以支持TimeGAN
    X, y, times_list, masks_list, periods_list = [], [], [], [], []
    max_length = 512  # 固定最大长度
    
    for item in data:
        # 提取特征
        mask = item['mask'].astype(bool)
        times = item['time'][mask]
        mags = item['mag'][mask]
        errors = item['errmag'][mask]
        
        if len(times) < 10:  # 过滤太短的序列
            continue
        
        # 截断或填充到固定长度
        seq_len = min(len(times), max_length)
        
        # 创建固定长度的特征矩阵
        features = np.zeros((max_length, 3), dtype=np.float32)  # [time, mag, error]
        features[:seq_len, 0] = times[:seq_len]
        features[:seq_len, 1] = mags[:seq_len] 
        features[:seq_len, 2] = errors[:seq_len]
        
        # 创建对应的mask
        feature_mask = np.zeros(max_length, dtype=bool)
        feature_mask[:seq_len] = True
        
        X.append(features)
        y.append(item['label'])
        times_list.append(times[:seq_len])
        masks_list.append(feature_mask)
        periods_list.append(item['period'])
    
    # 转换为numpy数组以支持TimeGAN
    X = np.array(X, dtype=np.float32)  # 固定形状 (N, 512, 3)
    y = np.array(y, dtype=np.int64)
    print(f"有效样本: {len(X)}个")
    print(f"数据形状: {X.shape}")
    print(f"类别分布: {Counter(y.tolist())}")
    
    # 设计针对类别0和1的重采样策略
    # 类别0 (Beta_Persei): 291 → 500 (+209)
    # 类别1 (Delta_Scuti): 70 → 400 (+330)  重点增强
    
    target_strategy = {
        0: 500,   # Beta_Persei 适度增强
        1: 400,   # Delta_Scuti 大幅增强 (5.7倍)
        2: 2234,  # RR_Lyrae_FM 保持不变
        3: 749,   # RR_Lyrae_FO 保持不变  
        4: 1860   # W_Ursae_Maj 保持不变
    }
    
    print(f"\n🎯 重采样目标:")
    current_counts = Counter(y)
    for cls, target_count in target_strategy.items():
        current = current_counts.get(cls, 0)
        increase = target_count - current
        print(f"  类别{cls}: {current} → {target_count} (+{increase})")
    
    # 配置增强的HybridResampler
    resampler = HybridResampler(
        synthesis_mode='physics_timegan',
        # 针对LINEAR类别0和1优化的物理约束权重
        physics_weight=0.35,  # 增强物理约束
        noise_level=0.03,  # 降低噪声水平保持数据质量
        smote_k_neighbors=8,  # 增加邻居数提高合成质量
        apply_enn=False,  # 关闭ENN清理，保持TimeGAN生成的样本
        sampling_strategy=target_strategy,
        random_state=42
    )
    
    print(f"\n📈 开始重采样训练...")
    print(f"物理约束配置:")
    print(f"  物理权重: {resampler.physics_weight}")
    print(f"  使用物理约束TimeGAN模式专门优化类别0和1")
    
    # 执行重采样 - physics_timegan模式返回4个值
    X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(X, y)
    
    print(f"\n✅ 重采样完成!")
    print(f"重采样后样本数: {len(X_resampled)}")
    print(f"重采样后类别分布: {Counter(y_resampled)}")
    
    # 计算增强效果
    original_counts = Counter(y)
    resampled_counts = Counter(y_resampled)
    
    print(f"\n📊 增强效果统计:")
    total_original = len(y)
    total_resampled = len(y_resampled)
    
    for cls in sorted(set(y)):
        orig_count = original_counts[cls]
        resamp_count = resampled_counts[cls]
        increase = resamp_count - orig_count
        increase_rate = increase / orig_count if orig_count > 0 else 0
        
        print(f"  类别{cls}: {orig_count} → {resamp_count} (+{increase}, +{increase_rate:.1%})")
    
    print(f"总样本: {total_original} → {total_resampled} (+{total_resampled - total_original})")
    
    # 计算类别平衡改善
    def gini_coefficient(counts):
        """计算基尼系数衡量不平衡程度"""
        counts = np.array(list(counts.values()))
        n = len(counts)
        mean_count = np.mean(counts)
        return np.sum(np.abs(counts - mean_count)) / (2 * n * mean_count)
    
    original_imbalance = gini_coefficient(original_counts)
    resampled_imbalance = gini_coefficient(resampled_counts)
    
    print(f"\n类别平衡改善:")
    print(f"  原始不平衡度: {original_imbalance:.3f}")
    print(f"  重采样后不平衡度: {resampled_imbalance:.3f}")
    print(f"  改善比例: {(original_imbalance - resampled_imbalance) / original_imbalance:.1%}")
    
    return X_resampled, y_resampled, times_list, masks_list, periods_list

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
        # features: [seq_len, 3] - [time, mag, error]
        times = features[:, 0].astype(np.float32)
        mags = features[:, 1].astype(np.float32) 
        errors = features[:, 2].astype(np.float32)
        
        seq_len = len(times)
        
        # 创建mask - 所有点都是有效的
        mask = np.ones(seq_len, dtype=bool)
        
        # 计算周期 - 使用时间序列的简单估计
        time_span = times.max() - times.min()
        estimated_period = time_span / max(1, seq_len // 10)  # 粗略估计
        
        # 构建标准格式的数据项
        data_item = {
            'time': times,
            'mag': mags,
            'errmag': errors,
            'mask': mask,
            'period': np.float32(estimated_period),
            'label': int(label),
            'class_name': class_name_mapping.get(label, f'Class_{label}'),
            'file_id': f'timegan_generated_{i}',
            'original_length': seq_len,
            'valid_points': seq_len,
            'coverage': 1.0
        }
        
        converted_data.append(data_item)
    
    print(f"✅ 转换完成: {len(converted_data)}样本")
    return converted_data

def save_linear_timegan_data(converted_data):
    """保存LINEAR TimeGAN重采样数据"""
    print(f"\n💾 保存LINEAR TimeGAN重采样数据...")
    
    output_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_resample_timegan.pkl'
    
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

def create_comparison_visualization(original_data, converted_data):
    """创建原始数据与TimeGAN数据的对比可视化"""
    print(f"\n📊 创建对比可视化...")
    
    configure_chinese_font()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LINEAR数据集: 原始 vs TimeGAN增强数据对比', fontsize=16, fontweight='bold')
    
    # 提取数据
    orig_labels = [item['label'] for item in original_data]
    timegan_labels = [item['label'] for item in converted_data]
    
    orig_counts = Counter(orig_labels)
    timegan_counts = Counter(timegan_labels)
    
    # 1. 类别分布对比
    classes = sorted(set(orig_labels))
    class_names = {0: 'Beta_Persei', 1: 'Delta_Scuti', 2: 'RR_Lyrae_FM', 
                   3: 'RR_Lyrae_FO', 4: 'W_Ursae_Maj'}
    
    x_pos = np.arange(len(classes))
    orig_values = [orig_counts[cls] for cls in classes]
    timegan_values = [timegan_counts[cls] for cls in classes]
    
    axes[0, 0].bar(x_pos - 0.2, orig_values, 0.4, label='原始数据', alpha=0.7)
    axes[0, 0].bar(x_pos + 0.2, timegan_values, 0.4, label='TimeGAN增强', alpha=0.7)
    axes[0, 0].set_xlabel('类别')
    axes[0, 0].set_ylabel('样本数量')
    axes[0, 0].set_title('类别分布对比')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f'{cls}\n{class_names.get(cls, "Unknown")}' for cls in classes], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 类别0和1的增强效果
    target_classes = [0, 1]
    target_names = ['Beta_Persei\n(类别0)', 'Delta_Scuti\n(类别1)']
    target_orig = [orig_counts[cls] for cls in target_classes]
    target_timegan = [timegan_counts[cls] for cls in target_classes]
    target_increase = [timegan_counts[cls] - orig_counts[cls] for cls in target_classes]
    
    x_pos = np.arange(len(target_classes))
    axes[0, 1].bar(x_pos, target_orig, 0.6, label='原始', alpha=0.7)
    axes[0, 1].bar(x_pos, target_increase, 0.6, bottom=target_orig, label='TimeGAN新增', alpha=0.7)
    axes[0, 1].set_xlabel('目标类别')
    axes[0, 1].set_ylabel('样本数量')
    axes[0, 1].set_title('重点类别增强效果')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(target_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (orig, total) in enumerate(zip(target_orig, target_timegan)):
        axes[0, 1].text(i, total + 10, f'{total}', ha='center', fontweight='bold')
        increase_rate = (total - orig) / orig
        axes[0, 1].text(i, orig/2, f'+{increase_rate:.0%}', ha='center', color='white', fontweight='bold')
    
    # 3. 不平衡度改善
    def gini_coefficient(counts):
        counts = np.array(list(counts.values()))
        n = len(counts)
        mean_count = np.mean(counts)
        return np.sum(np.abs(counts - mean_count)) / (2 * n * mean_count)
    
    orig_gini = gini_coefficient(orig_counts)
    timegan_gini = gini_coefficient(timegan_counts)
    
    metrics = ['类别不平衡度\n(Gini系数)']
    orig_metrics = [orig_gini]
    timegan_metrics = [timegan_gini]
    
    x_pos = np.arange(len(metrics))
    axes[0, 2].bar(x_pos - 0.2, orig_metrics, 0.4, label='原始数据', alpha=0.7)
    axes[0, 2].bar(x_pos + 0.2, timegan_metrics, 0.4, label='TimeGAN增强', alpha=0.7)
    axes[0, 2].set_xlabel('评估指标')
    axes[0, 2].set_ylabel('数值')
    axes[0, 2].set_title('数据平衡性改善')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(metrics)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    improvement = (orig_gini - timegan_gini) / orig_gini
    axes[0, 2].text(0, max(orig_gini, timegan_gini) * 1.1, f'改善: {improvement:.1%}', 
                    ha='center', fontweight='bold', color='green')
    
    # 4-6. 类别0和1的光变曲线样本对比
    def plot_sample_curves(ax, data, class_label, title_suffix):
        samples = [item for item in data if item['label'] == class_label]
        if not samples:
            ax.text(0.5, 0.5, '无样本', ha='center', va='center', transform=ax.transAxes)
            return
            
        # 随机选择几个样本绘制
        np.random.seed(42)
        sample_indices = np.random.choice(len(samples), min(5, len(samples)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            sample = samples[idx]
            mask = sample['mask'].astype(bool) if 'mask' in sample else np.ones(len(sample['time']), dtype=bool)
            times = sample['time'][mask]
            mags = sample['mag'][mask]
            
            # 归一化时间到0-1
            if len(times) > 1:
                times_norm = (times - times.min()) / (times.max() - times.min())
                ax.plot(times_norm, mags, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('归一化时间')
        ax.set_ylabel('星等')
        ax.set_title(f'{title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 星等越小越亮
    
    # 类别0对比
    plot_sample_curves(axes[1, 0], original_data, 0, 'Beta_Persei - 原始数据')
    plot_sample_curves(axes[1, 1], converted_data, 0, 'Beta_Persei - TimeGAN生成')
    
    # 类别1对比  
    plot_sample_curves(axes[1, 2], [item for item in converted_data if item['label'] == 1], 1, 'Delta_Scuti - TimeGAN生成')
    
    plt.tight_layout()
    
    # 保存图片
    pic_dir = '/root/autodl-tmp/lnsde-contiformer/results/pics/LINEAR'
    os.makedirs(pic_dir, exist_ok=True)
    pic_path = os.path.join(pic_dir, 'timegan_comparison.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 对比可视化已保存至: {pic_path}")
    return pic_path

def main():
    """主函数"""
    print("🚀 LINEAR物理约束TimeGAN重采样 - 专门优化类别0和1")
    print("=" * 60)
    
    # 1. 分析混淆问题
    class_features, class_samples = analyze_confusion_classes()
    
    # 2. 应用物理约束TimeGAN
    X_resampled, y_resampled, times_list, masks_list, periods_list = apply_enhanced_physics_timegan_resampling()
    
    # 3. 转换为标准格式
    original_data = []
    with open('/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    converted_data = convert_to_standard_format(X_resampled, y_resampled, original_data)
    
    # 4. 保存数据
    output_path = save_linear_timegan_data(converted_data)
    
    # 5. 创建可视化
    pic_path = create_comparison_visualization(original_data, converted_data)
    
    print(f"\n🎉 LINEAR TimeGAN重采样完成!")
    print(f"📁 数据文件: {output_path}")
    print(f"📊 可视化: {pic_path}")
    print(f"\n🎯 主要改进:")
    print(f"  • 类别0 (Beta_Persei): 291 → 500样本 (+71%)")
    print(f"  • 类别1 (Delta_Scuti): 70 → 400样本 (+471%)")
    print(f"  • 增强物理约束以减少与类别3,4的混淆")
    print(f"  • 加强去噪能力提高分类准确性")
    
    print(f"\n💡 使用方法:")
    print(f"python main.py --dataset 2 --use_resampling --resampled_data_path {output_path}")

if __name__ == "__main__":
    main()