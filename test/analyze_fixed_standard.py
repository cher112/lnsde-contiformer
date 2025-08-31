#!/usr/bin/env python3
"""
分析实际的fixed数据集类别分布
使用学界标准指标
"""

import numpy as np
from tabulate import tabulate
from scipy.stats import entropy
import pickle


def load_and_analyze_fixed_data():
    """加载并分析fixed数据集"""
    
    datasets_info = {}
    
    # ASAS fixed
    print("\n加载ASAS fixed数据...")
    with open('/autodl-fs/data/lnsde-contiformer/ASAS_folded_512_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 分析padding情况
    total_samples = len(data)
    padded_samples = 0
    valid_samples = []
    
    for sample in data:
        # 检查是否全是padding（时间全为0或mask全为False）
        if 'mask' in sample and np.all(~sample['mask']):
            padded_samples += 1
        elif 'time' in sample and np.all(sample['time'] == 0):
            padded_samples += 1
        else:
            valid_samples.append(sample['label'])
    
    unique_labels, counts = np.unique(valid_samples, return_counts=True)
    
    print(f"  总样本: {total_samples}")
    print(f"  Padding样本: {padded_samples}")
    print(f"  有效样本: {len(valid_samples)}")
    print(f"  类别分布: {dict(zip(unique_labels, counts))}")
    
    datasets_info['ASAS'] = {
        'counts': counts,
        'n_classes': len(unique_labels),
        'total': len(valid_samples),
        'padded': padded_samples
    }
    
    # LINEAR fixed
    print("\n加载LINEAR fixed数据...")
    with open('/autodl-fs/data/lnsde-contiformer/LINEAR_folded_512_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    total_samples = len(data)
    padded_samples = 0
    valid_samples = []
    
    for sample in data:
        if 'mask' in sample and np.all(~sample['mask']):
            padded_samples += 1
        elif 'time' in sample and np.all(sample['time'] == 0):
            padded_samples += 1
        else:
            valid_samples.append(sample['label'])
    
    unique_labels, counts = np.unique(valid_samples, return_counts=True)
    
    print(f"  总样本: {total_samples}")
    print(f"  Padding样本: {padded_samples}")
    print(f"  有效样本: {len(valid_samples)}")
    print(f"  类别分布: {dict(zip(unique_labels, counts))}")
    
    datasets_info['LINEAR'] = {
        'counts': counts,
        'n_classes': len(unique_labels),
        'total': len(valid_samples),
        'padded': padded_samples
    }
    
    # MACHO fixed
    print("\n加载MACHO fixed数据...")
    with open('/autodl-fs/data/lnsde-contiformer/MACHO_folded_512_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    total_samples = len(data)
    padded_samples = 0
    valid_samples = []
    
    for sample in data:
        if 'mask' in sample and np.all(~sample['mask']):
            padded_samples += 1
        elif 'time' in sample and np.all(sample['time'] == 0):
            padded_samples += 1
        else:
            valid_samples.append(sample['label'])
    
    unique_labels, counts = np.unique(valid_samples, return_counts=True)
    
    print(f"  总样本: {total_samples}")
    print(f"  Padding样本: {padded_samples}")
    print(f"  有效样本: {len(valid_samples)}")
    print(f"  类别分布: {dict(zip(unique_labels, counts))}")
    
    datasets_info['MACHO'] = {
        'counts': counts,
        'n_classes': len(unique_labels),
        'total': len(valid_samples),
        'padded': padded_samples
    }
    
    return datasets_info


def calculate_standard_metrics(counts, n_classes):
    """计算学界标准的不均衡指标"""
    
    metrics = {}
    normalized = counts / counts.sum()
    
    # 1. Imbalance Ratio (IR) - 最基础的指标
    metrics['IR'] = counts.max() / counts.min()
    
    # 2. Imbalance Degree (ID) - Ortigosa-Hernández et al. 2017
    # 多类不均衡的标准度量
    K = n_classes
    n = counts.sum()
    ni_n = counts / n
    ID = 0
    for p in ni_n:
        ID += p * min(1, K * p)
    metrics['ID'] = 1 - ((K-1)/K) * ID
    
    # 3. Shannon Entropy - 信息论指标
    max_entropy = np.log(n_classes)
    actual_entropy = entropy(normalized)
    metrics['Entropy'] = actual_entropy
    metrics['Normalized_Entropy'] = actual_entropy / max_entropy
    
    # 4. Gini Index - 经济学借鉴的指标
    sorted_p = np.sort(normalized)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    metrics['Gini'] = (n + 1 - 2 * np.sum(cumsum)) / n
    
    # 5. Coefficient of Variation (CV)
    metrics['CV'] = np.std(counts) / np.mean(counts)
    
    # 6. Multi-class Imbalance Ratio (MIR) - Tanwani & Farooq 2010
    mir_sum = 0
    for i in range(len(counts)):
        for j in range(i+1, len(counts)):
            if counts[i] > 0 and counts[j] > 0:
                ratio = max(counts[i]/counts[j], counts[j]/counts[i])
                mir_sum += ratio
    num_pairs = n_classes * (n_classes - 1) / 2
    metrics['MIR'] = mir_sum / num_pairs if num_pairs > 0 else 0
    
    # 基础统计
    metrics['n_classes'] = n_classes
    metrics['min_samples'] = counts.min()
    metrics['max_samples'] = counts.max()
    metrics['total_samples'] = counts.sum()
    metrics['min_ratio'] = (counts.min() / counts.sum()) * 100
    
    return metrics


def main():
    """主分析函数"""
    
    print("="*100)
    print("📊 Fixed数据集分析（删除padding，使用学界标准指标）")
    print("="*100)
    
    # 加载数据
    datasets = load_and_analyze_fixed_data()
    
    # 计算标准指标
    results = {}
    for name in ['ASAS', 'LINEAR', 'MACHO']:
        metrics = calculate_standard_metrics(
            datasets[name]['counts'],
            datasets[name]['n_classes']
        )
        results[name] = metrics
    
    # 性能数据
    performance = {
        'ASAS': {'acc': 96.57, 'f1': 95.33},
        'LINEAR': {'acc': 89.43, 'f1': 86.87},
        'MACHO': {'acc': 81.52, 'f1': 80.17}
    }
    
    print("\n" + "="*100)
    print("📈 类别分布详细统计")
    print("="*100)
    
    for name in ['ASAS', 'LINEAR', 'MACHO']:
        counts = datasets[name]['counts']
        total = counts.sum()
        print(f"\n{name}数据集:")
        print(f"  有效样本: {total}")
        print(f"  类别数: {len(counts)}")
        print(f"  各类别样本数: {counts}")
        print(f"  各类别占比: {np.round(counts/total*100, 1)}%")
        print(f"  最大/最小: {counts.max()}/{counts.min()} = {counts.max()/counts.min():.1f}")
    
    print("\n" + "="*100)
    print("📊 学界标准不均衡指标")
    print("="*100)
    
    # 主表格
    table = []
    for name in ['ASAS', 'LINEAR', 'MACHO']:
        r = results[name]
        p = performance[name]
        table.append([
            name,
            r['n_classes'],
            r['total_samples'],
            f"{p['acc']:.1f}",
            f"{p['f1']:.1f}",
            f"{r['IR']:.1f}",
            f"{r['ID']:.3f}",
            f"{r['MIR']:.2f}",
            f"{r['Gini']:.3f}"
        ])
    
    headers = ['数据集', '类别', '样本', '准确率%', 'F1', 'IR↑', 'ID↑', 'MIR↑', 'Gini↑']
    print(tabulate(table, headers=headers, tablefmt='pipe'))
    
    print("\n标准指标说明：")
    print("• IR: Imbalance Ratio (最大类/最小类)")
    print("• ID: Imbalance Degree (Ortigosa-Hernández 2017)")
    print("• MIR: Multi-class Imbalance Ratio (Tanwani 2010)")
    print("• Gini: Gini系数 (0=均衡, 1=极端不均衡)")
    
    # 附加指标表
    print("\n" + "="*100)
    print("📊 附加统计指标")
    print("="*100)
    
    table2 = []
    for name in ['ASAS', 'LINEAR', 'MACHO']:
        r = results[name]
        table2.append([
            name,
            f"{r['Entropy']:.3f}",
            f"{r['Normalized_Entropy']:.3f}",
            f"{r['CV']:.3f}",
            f"{r['min_samples']}",
            f"{r['min_ratio']:.1f}%"
        ])
    
    headers2 = ['数据集', 'Entropy', 'Norm_Entropy', 'CV', '最小类', '最小类%']
    print(tabulate(table2, headers=headers2, tablefmt='pipe'))
    
    # 排名分析
    print("\n" + "="*100)
    print("🏆 不均衡度排名（基于多个标准指标）")
    print("="*100)
    
    # 各指标排名
    for metric in ['IR', 'ID', 'MIR', 'Gini', 'CV']:
        sorted_by_metric = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
        print(f"\n{metric}排名:")
        for i, (name, r) in enumerate(sorted_by_metric, 1):
            print(f"  {i}. {name}: {metric}={r[metric]:.3f}")
    
    # 综合评估
    print("\n" + "="*100)
    print("📌 综合分析")
    print("="*100)
    
    print("\n根据标准指标：")
    print("• IR (不均衡比): LINEAR(35.8) > ASAS(12.6) > MACHO(10.3)")
    print("• MIR (多类不均衡): LINEAR > ASAS > MACHO")
    print("• 但MACHO是7分类问题，固有复杂度更高")
    print("• MACHO准确率最低(81.52%)，反映了综合难度")
    
    print("\n结论：")
    print("• 纯不均衡度: LINEAR最高")
    print("• 考虑分类复杂度: MACHO挑战最大(7类 vs 5类)")
    print("• 实际表现: MACHO准确率最低，验证了其综合难度")


if __name__ == "__main__":
    main()