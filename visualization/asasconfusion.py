import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义所有混淆矩阵
confusion_matrices = {
    "Dataset 1 (5-class)": np.array([
        [65, 1, 1, 0, 0],
        [0, 23, 0, 0, 0],
        [9, 0, 150, 0, 3],
        [6, 1, 0, 7, 1],
        [2, 0, 2, 0, 324]
    ]),
    "Dataset 2 (5-class)": np.array([
        [0, 0, 28, 26, 8],
        [0, 19, 0, 0, 0],
        [0, 0, 405, 12, 3],
        [0, 0, 5, 157, 2],
        [0, 1, 0, 15, 360]
    ]),
    "Dataset 3 (7-class)": np.array([
        [20, 0, 0, 0, 4, 0, 0],
        [0, 20, 2, 0, 0, 0, 0],
        [0, 7, 33, 1, 5, 0, 9],
        [5, 0, 0, 14, 1, 0, 0],
        [4, 0, 1, 0, 72, 0, 5],
        [1, 0, 2, 0, 6, 0, 0],
        [0, 0, 0, 0, 3, 0, 107]
    ])
}

# 设置配色方案
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 绘制所有混淆矩阵
for idx, (name, cm) in enumerate(confusion_matrices.items()):
    n_classes = cm.shape[0]
    labels = [f'C{i}' for i in range(n_classes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title(f'Raw Confusion Matrix', fontsize=14)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # 处理除零情况
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax2)
    ax2.set_title(f'Normalized Confusion Matrix', fontsize=14)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    
    # 计算并显示准确率
    accuracy = np.trace(cm) / np.sum(cm)
    fig.suptitle(f'{name} - Accuracy: {accuracy:.3f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{idx+1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印基本信息
    print(f"\n{name}:")
    print(f"Total samples: {np.sum(cm)}")
    print(f"Correct predictions: {np.trace(cm)}")
    print(f"Overall accuracy: {accuracy:.3f}")
