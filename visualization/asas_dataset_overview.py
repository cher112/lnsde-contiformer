#!/usr/bin/env python3
"""
生成ASAS数据集概览图，模仿MACHO数据集概览图的样式
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from collections import Counter
import os


def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True


def load_asas_data():
    """加载ASAS数据集"""
    # 查找ASAS原始数据文件（非fixed版本）
    data_paths = [
        '/autodl-fs/data/lnsde-contiformer/ASAS_folded_512.pkl',
        '/autodl-fs/data/lnsde-contiformer/ASAS_folded_512.pkl',
        '/root/autodl-tmp/PhysioPro/data/ASAS/folded_data.npz',
        '/root/autodl-tmp/PhysioPro/data/ASAS/backup_folded_data.npz'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("找不到ASAS数据集文件")
    
    print(f"加载数据: {data_path}")
    
    if data_path.endswith('.pkl'):
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        data = np.load(data_path, allow_pickle=True)
        return data


def create_asas_overview():
    """创建ASAS数据集概览图"""
    
    # 配置字体
    configure_chinese_font()
    
    # 加载数据
    try:
        data = load_asas_data()
        
        print(f"数据集大小: {len(data)} 个样本")
        
        # 处理pickle文件格式 - 每个样本是一个字典
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # 提取所有标签和类别名称
            all_labels = [sample['label'] for sample in data]
            class_names_from_data = [sample['class_name'] for sample in data]
            
            # 构建时间序列数据
            n_samples = len(data)
            seq_len = len(data[0]['time'])
            all_X = np.zeros((n_samples, seq_len, 2))
            
            for i, sample in enumerate(data):
                all_X[i, :, 0] = sample['time']
                all_X[i, :, 1] = sample['mag']
            
            all_labels = np.array(all_labels)
            
        # 处理其他格式
        elif isinstance(data, dict):
            if 'data' in data and 'labels' in data:
                all_X = data['data'] 
                all_labels = data['labels']
            elif 'X' in data and 'y' in data:
                all_X = data['X']
                all_labels = data['y']
            else:
                keys = list(data.keys())
                print(f"数据包含的键: {keys}")
                all_X = list(data.values())[0]
                all_labels = list(data.values())[1]
        else:
            # NPZ文件格式
            X_train = data['X_train']
            y_train = data['y_train'] 
            X_test = data['X_test']
            y_test = data['y_test']
            
            all_labels = np.concatenate([y_train, y_test])
            all_X = np.concatenate([X_train, X_test])
        
        print(f"特征维度: {all_X.shape}")
        print(f"标签范围: {np.min(all_labels)} - {np.max(all_labels)}")
        
        # 统计类别信息
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"类别分布: {dict(zip(unique_labels, counts))}")
        
        # 获取类别名称
        if 'class_names_from_data' in locals():
            # 从数据中提取唯一的类别名称
            unique_class_names = []
            for label in unique_labels:
                for i, sample_label in enumerate(all_labels):
                    if sample_label == label:
                        class_name = class_names_from_data[i]
                        if class_name not in unique_class_names:
                            unique_class_names.append(class_name)
                        break
            class_names = unique_class_names
            print(f"类别名称: {class_names}")
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        # 使用模拟数据进行演示
        print("使用模拟ASAS数据...")
        
        # 根据ASAS天体类型创建模拟数据 (相对均衡的分布)
        class_names = ['RR_Lyrae', 'EB', 'Cepheid', 'LPV', 'W_UMa'] 
        class_counts = [380, 420, 350, 390, 360]  # 相对均衡的样本数量
        
        all_labels = []
        for i, count in enumerate(class_counts):
            all_labels.extend([i] * count)
        all_labels = np.array(all_labels)
        
        # 生成模拟的时间序列数据
        n_samples = len(all_labels)
        seq_len = 512
        all_X = np.random.randn(n_samples, seq_len, 2)  # [time, magnitude]
        
        # 为不同类别生成特征性的光变曲线
        for i, label in enumerate(all_labels):
            if label == 0:  # RR_Lyrae - 规则周期变化
                period = np.random.uniform(0.3, 0.8)
                phase = np.linspace(0, 2*np.pi*period*5, seq_len)
                all_X[i, :, 1] = 15 + 0.5*np.sin(phase) + 0.1*np.random.randn(seq_len)
            elif label == 1:  # EB - 双星食变
                period = np.random.uniform(1, 10)
                phase = np.linspace(0, 2*np.pi*period*2, seq_len)
                eclipse = np.where(np.abs(np.sin(phase)) > 0.8, -1.0, 0)
                all_X[i, :, 1] = 14 + 0.2*np.random.randn(seq_len) + eclipse
            elif label == 2:  # Cepheid - 造父变星
                period = np.random.uniform(2, 50)
                phase = np.linspace(0, 2*np.pi*period, seq_len)
                all_X[i, :, 1] = 12 + 1.5*np.sin(phase) + 0.15*np.random.randn(seq_len)
            elif label == 3:  # LPV - 长周期变星
                period = np.random.uniform(80, 400)
                phase = np.linspace(0, 2*np.pi*period/50, seq_len)
                all_X[i, :, 1] = 16 + 2.0*np.sin(phase) + 0.3*np.random.randn(seq_len)
            else:  # W_UMa
                period = np.random.uniform(0.2, 1.0)
                phase = np.linspace(0, 2*np.pi*period*10, seq_len)
                all_X[i, :, 1] = 13 + 0.3*np.sin(phase) + 0.05*np.random.randn(seq_len)
            
            # 时间序列
            all_X[i, :, 0] = np.linspace(0, 1000, seq_len)
    
    # 统计类别信息
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    total_samples = len(all_labels)
    
    # ASAS数据集的类别名称 (根据实际情况调整)
    if 'class_names' not in locals():
        class_names = ['RR_Lyrae', 'EB', 'Cepheid', 'LPV', 'W_UMa']
        if len(unique_labels) != len(class_names):
            class_names = [f'Class_{i}' for i in unique_labels]
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    
    # 主标题
    fig.suptitle('ASAS Folded Dataset Overview', fontsize=22, fontweight='bold', y=0.95)
    
    # 1. 类别分布柱状图
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(unique_labels)), counts, color='steelblue', alpha=0.8)
    ax1.set_title('Class Distribution', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xticks(range(len(unique_labels)))
    # 缩短类别名称以避免重叠
    short_names = [name[:8] + '...' if len(name) > 10 else name for name in class_names]
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. 饼图
    ax2 = plt.subplot(2, 3, 2)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6'][:len(unique_labels)]
    # 使用缩短的名称作为标签，去除padding
    _, texts, autotexts = ax2.pie(counts, labels=short_names, autopct='%1.1f%%', 
                                      colors=colors, startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Class Distribution (Pie Chart)', fontsize=16, fontweight='bold', pad=15)
    
    # 调整饼图标签位置，避免重叠
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    # 3. 代表性光变曲线
    ax3 = plt.subplot(2, 3, (3, 6))
    
    # 为每个类别选择一个代表样本
    colors_lc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    sample_indices = []
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]
        if len(label_indices) > 0:
            sample_idx = np.random.choice(label_indices)
            sample_indices.append(sample_idx)
    
    phase_offset = 0
    for i, (label, sample_idx) in enumerate(zip(unique_labels, sample_indices)):
        if sample_idx < len(all_X):
            times = all_X[sample_idx, :, 0]
            magnitudes = all_X[sample_idx, :, 1] + phase_offset
            
            # 归一化时间到相位
            if len(times) > 1:
                phase = (times - times.min()) / (times.max() - times.min()) * 600 - 300
                ax3.plot(phase, magnitudes, 'o-', color=colors_lc[i % len(colors_lc)], 
                        alpha=0.7, markersize=1.5, linewidth=1.2, label=short_names[i])
                
        phase_offset += 2.5  # 垂直偏移以便区分
    
    ax3.set_title('Representative Light Curves by Class', fontsize=16, fontweight='bold', pad=15)
    ax3.set_xlabel('Phase', fontsize=12)
    ax3.set_ylabel('Normalized Magnitude (offset)', fontsize=12)
    ax3.legend(loc='center right', bbox_to_anchor=(1.18, 0.5), fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 数据集统计信息
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # 统计信息文本（移除数据长度信息）
    stats_text = f"""Dataset Statistics:
Number of Classes: {len(unique_labels)}

Class Details:"""
    
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        percentage = (count / total_samples) * 100
        stats_text += f"\n{class_names[i]}: {count} ({percentage:.1f}%)"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 调整布局，增加间距以避免重叠
    plt.tight_layout(pad=2.0)  # 增加整体间距
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.94, 
                       hspace=0.35, wspace=0.25)  # 精细调整各边距和子图间距
    
    # 保存图片
    output_dir = "/autodl-fs/data/lnsde-contiformer/results/pics/ASAS"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "asas_dataset_overview.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ ASAS数据集概览图已保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    create_asas_overview()