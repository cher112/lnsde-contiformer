#!/usr/bin/env python3
"""
NaN样本分析和统计工具
"""

import os
import re
from collections import defaultdict, Counter


def analyze_nan_log():
    """分析nan_samples.log文件，生成详细统计"""
    
    log_file = 'nan_samples.log'
    if not os.path.exists(log_file):
        print("没有找到nan_samples.log文件")
        return
    
    print("="*70)
    print("NaN样本过滤统计分析")
    print("="*70)
    
    # 统计数据结构
    epoch_stats = defaultdict(lambda: {'total': 0, 'batches': set(), 'samples': []})
    batch_stats = defaultdict(lambda: {'epoch': 0, 'samples': []})
    sample_frequency = Counter()
    total_filtered = 0
    
    # 解析日志文件
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or 'filtered out' not in line:
                continue
            
            # 解析: "Epoch 5, Batch 123, Sample 7 filtered out (NaN loss)"
            match = re.search(r'Epoch (\d+), Batch (\d+), Sample (\d+)', line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                sample = int(match.group(3))
                
                # 更新统计
                epoch_stats[epoch]['total'] += 1
                epoch_stats[epoch]['batches'].add(batch)
                epoch_stats[epoch]['samples'].append(sample)
                
                batch_stats[f"{epoch}_{batch}"]['epoch'] = epoch
                batch_stats[f"{epoch}_{batch}"]['samples'].append(sample)
                
                sample_frequency[f"E{epoch}_B{batch}_S{sample}"] += 1
                total_filtered += 1
    
    # 显示统计结果
    print(f"\n📊 总体统计:")
    print(f"  总过滤样本数: {total_filtered}")
    print(f"  涉及轮次数: {len(epoch_stats)}")
    print(f"  涉及批次数: {len(batch_stats)}")
    
    if len(epoch_stats) > 0:
        print(f"\n📈 按轮次统计:")
        for epoch in sorted(epoch_stats.keys()):
            stats = epoch_stats[epoch]
            print(f"  Epoch {epoch}: {stats['total']} 个样本, {len(stats['batches'])} 个批次")
        
        print(f"\n🎯 最频繁出现NaN的批次:")
        batch_sample_counts = {}
        for key, stats in batch_stats.items():
            epoch, batch = key.split('_')
            batch_sample_counts[key] = len(stats['samples'])
        
        top_batches = sorted(batch_sample_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for batch_key, sample_count in top_batches:
            epoch, batch = batch_key.split('_')
            print(f"  Epoch {epoch}, Batch {batch}: {sample_count} 个样本")
        
        # 分析样本在批次中的位置分布
        print(f"\n📍 样本位置分布分析:")
        all_sample_positions = []
        for stats in epoch_stats.values():
            all_sample_positions.extend(stats['samples'])
        
        if all_sample_positions:
            position_counter = Counter(all_sample_positions)
            print(f"  最常出现NaN的样本位置:")
            top_positions = position_counter.most_common(10)
            for pos, count in top_positions:
                print(f"    位置 {pos}: {count} 次")
        
        # 检查是否有重复出现的样本
        print(f"\n🔄 重复出现的样本:")
        repeated = [(k, v) for k, v in sample_frequency.items() if v > 1]
        if repeated:
            repeated.sort(key=lambda x: x[1], reverse=True)
            for sample, count in repeated[:10]:
                print(f"  {sample}: {count} 次")
        else:
            print("  没有样本多次出现NaN")
    
    print(f"\n💡 建议:")
    if total_filtered > 0:
        filter_rate = total_filtered / max(len(batch_stats) * 20, 1) * 100  # 假设每批次20个样本
        print(f"  样本过滤率约: {filter_rate:.2f}%")
        
        if filter_rate > 5:
            print("  ⚠️  过滤率较高，建议检查:")
            print("    - 模型参数是否合适")
            print("    - 学习率是否过大")
            print("    - 数据预处理是否正确")
        elif filter_rate > 1:
            print("  ✅ 过滤率适中，模型训练基本稳定")
        else:
            print("  🎉 过滤率很低，模型训练非常稳定")
    
    print("="*70)


def clean_nan_log():
    """清理过旧的日志文件"""
    log_file = 'nan_samples.log'
    if os.path.exists(log_file):
        # 备份旧日志
        backup_file = f"{log_file}.backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(log_file, backup_file)
        print(f"已备份旧日志到: {backup_file}")
        print("新的训练将创建新的日志文件")
    else:
        print("没有找到需要清理的日志文件")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        clean_nan_log()
    else:
        analyze_nan_log()