#!/usr/bin/env python3
"""
检查MACHO TimeGAN重采样数据，分析训练变慢的原因
"""

import pickle
import numpy as np
from collections import Counter
import sys

def analyze_macho_timegan_data():
    """分析MACHO TimeGAN数据"""
    print("🔍 分析MACHO TimeGAN重采样数据...")
    
    # 加载TimeGAN数据
    timegan_path = '/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl'
    with open(timegan_path, 'rb') as f:
        timegan_data = pickle.load(f)
    
    # 加载原始数据对比
    original_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    print(f"原始数据: {len(original_data)} 样本")
    print(f"TimeGAN数据: {len(timegan_data)} 样本")
    print(f"数据增长: {len(timegan_data) - len(original_data)} 样本 ({(len(timegan_data)/len(original_data) - 1)*100:.1f}%)")
    
    # 检查样本结构
    print(f"\n📊 样本结构对比:")
    orig_sample = original_data[0]
    timegan_sample = timegan_data[0]
    
    print(f"原始样本字段: {list(orig_sample.keys())}")
    print(f"TimeGAN样本字段: {list(timegan_sample.keys())}")
    
    # 检查序列长度分布
    print(f"\n📏 序列长度分析:")
    
    def analyze_lengths(data, name):
        lengths = []
        for sample in data:
            if 'mask' in sample:
                mask = sample['mask']
                valid_length = np.sum(mask.astype(bool))
            else:
                valid_length = len(sample['time'])
            lengths.append(valid_length)
        
        print(f"{name}:")
        print(f"  平均长度: {np.mean(lengths):.1f}")
        print(f"  最大长度: {np.max(lengths)}")
        print(f"  最小长度: {np.min(lengths)}")
        print(f"  长度标准差: {np.std(lengths):.1f}")
        
        return lengths
    
    orig_lengths = analyze_lengths(original_data, "原始数据")
    timegan_lengths = analyze_lengths(timegan_data, "TimeGAN数据")
    
    # 检查类别分布
    print(f"\n🏷️ 类别分布对比:")
    orig_labels = [item['label'] for item in original_data]
    timegan_labels = [item['label'] for item in timegan_data]
    
    orig_counts = Counter(orig_labels)
    timegan_counts = Counter(timegan_labels)
    
    print(f"原始类别分布: {dict(orig_counts)}")
    print(f"TimeGAN类别分布: {dict(timegan_counts)}")
    
    # 检查数据类型和大小
    print(f"\n💾 数据大小分析:")
    
    def get_sample_size(sample):
        total_size = 0
        for key, value in sample.items():
            if hasattr(value, 'nbytes'):
                total_size += value.nbytes
            else:
                total_size += sys.getsizeof(value)
        return total_size
    
    orig_sample_size = get_sample_size(orig_sample)
    timegan_sample_size = get_sample_size(timegan_sample)
    
    print(f"原始样本大小: {orig_sample_size} 字节")
    print(f"TimeGAN样本大小: {timegan_sample_size} 字节")
    print(f"单样本增长: {timegan_sample_size - orig_sample_size} 字节")
    
    total_orig_size = orig_sample_size * len(original_data)
    total_timegan_size = timegan_sample_size * len(timegan_data)
    
    print(f"总数据大小:")
    print(f"  原始: {total_orig_size / 1024 / 1024:.1f} MB")
    print(f"  TimeGAN: {total_timegan_size / 1024 / 1024:.1f} MB")
    print(f"  增长: {(total_timegan_size - total_orig_size) / 1024 / 1024:.1f} MB")
    
    # 分析可能的训练慢原因
    print(f"\n🐌 可能的训练变慢原因分析:")
    
    # 1. 样本数量增加
    sample_increase = len(timegan_data) / len(original_data)
    print(f"1. 样本数量增加 {sample_increase:.2f}x - 直接影响训练时间")
    
    # 2. 序列长度变化
    avg_orig_len = np.mean(orig_lengths)
    avg_timegan_len = np.mean(timegan_lengths)
    length_ratio = avg_timegan_len / avg_orig_len
    print(f"2. 平均序列长度: {avg_orig_len:.1f} → {avg_timegan_len:.1f} ({length_ratio:.2f}x)")
    
    # 3. 数据复杂度
    orig_unique_lengths = len(set(orig_lengths))
    timegan_unique_lengths = len(set(timegan_lengths))
    print(f"3. 序列长度多样性: {orig_unique_lengths} → {timegan_unique_lengths} 种")
    
    # 4. 内存使用估算
    memory_ratio = total_timegan_size / total_orig_size
    print(f"4. 内存使用增长: {memory_ratio:.2f}x")
    
    # 检查是否有异常长的序列
    print(f"\n⚠️ 异常检查:")
    long_sequences = [l for l in timegan_lengths if l > 1000]
    if long_sequences:
        print(f"发现 {len(long_sequences)} 个超长序列 (>1000点)")
        print(f"最长序列: {max(long_sequences)} 点")
    else:
        print("未发现异常长序列")
    
    # 计算理论训练时间增长
    # 训练时间 ≈ 样本数 × 序列长度 × 模型复杂度
    theoretical_slowdown = sample_increase * length_ratio
    print(f"\n📈 理论训练时间增长: {theoretical_slowdown:.2f}x")
    
    return {
        'sample_increase': sample_increase,
        'length_ratio': length_ratio, 
        'memory_ratio': memory_ratio,
        'theoretical_slowdown': theoretical_slowdown
    }

def suggest_optimization():
    """建议优化方案"""
    print(f"\n💡 训练加速建议:")
    
    print(f"1. 减小批次大小 (batch_size):")
    print(f"   --batch_size 32  # 从64减到32")
    print(f"   --batch_size 16  # 更保守的选择")
    
    print(f"2. 启用梯度累积:")
    print(f"   --gradient_accumulation_steps 4  # 保持有效批次大小")
    
    print(f"3. 使用梯度检查点:")
    print(f"   --use_gradient_checkpoint  # 节省显存")
    
    print(f"4. 调整工作进程:")
    print(f"   --num_workers 8  # 减少数据加载时间")
    print(f"   --prefetch_factor 2  # 减少内存占用")
    
    print(f"5. 使用更快的SDE配置:")
    print(f"   --sde_config 3  # 时间优先配置")
    
    print(f"6. 早期测试:")
    print(f"   --epochs 10  # 先用少量epoch测试")
    
    print(f"7. 如果显存不足:")
    print(f"   --no_amp  # 禁用混合精度")
    print(f"   --hidden_channels 64  # 减小模型规模")

if __name__ == "__main__":
    stats = analyze_macho_timegan_data()
    suggest_optimization()
    
    print(f"\n📋 总结:")
    print(f"MACHO TimeGAN数据增长了 {stats['theoretical_slowdown']:.1f}x，主要原因:")
    print(f"  • 样本数量: +{(stats['sample_increase']-1)*100:.0f}%")
    print(f"  • 序列长度: {stats['length_ratio']:.2f}x")
    print(f"  • 内存使用: {stats['memory_ratio']:.2f}x")
    print(f"建议使用小批次大小和梯度累积来加速训练。")