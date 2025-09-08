#!/usr/bin/env python3
"""
调试F1和Recall计算问题
"""

import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support

def test_metrics_calculation():
    """测试指标计算的正确性"""
    print("=" * 60)
    print("F1和Recall计算调试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_classes = 7  # MACHO数据集有7个类别
    
    # 生成随机标签和预测
    true_labels = np.random.randint(0, n_classes, n_samples)
    
    # 创建不同准确率的预测结果进行测试
    accuracies_to_test = [0.5, 0.7, 0.8, 0.9]
    
    for target_acc in accuracies_to_test:
        print(f"\n测试目标准确率: {target_acc:.1%}")
        print("-" * 40)
        
        # 生成指定准确率的预测
        predictions = true_labels.copy()
        n_wrong = int(n_samples * (1 - target_acc))
        wrong_indices = np.random.choice(n_samples, n_wrong, replace=False)
        
        for idx in wrong_indices:
            # 随机选择错误的类别
            wrong_class = np.random.randint(0, n_classes)
            while wrong_class == true_labels[idx]:
                wrong_class = np.random.randint(0, n_classes)
            predictions[idx] = wrong_class
        
        # 计算各种指标
        acc = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        macro_recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        weighted_recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        
        print(f"实际准确率:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"宏平均F1:      {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        print(f"加权F1:        {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
        print(f"宏平均Recall:  {macro_recall:.4f} ({macro_recall*100:.2f}%)")
        print(f"加权Recall:    {weighted_recall:.4f} ({weighted_recall*100:.2f}%)")
        
        # 分析差异
        f1_diff = abs(weighted_f1 - acc)
        recall_diff = abs(weighted_recall - acc)
        
        print(f"加权F1与准确率差异:    {f1_diff:.4f}")
        print(f"加权Recall与准确率差异: {recall_diff:.4f}")
        
        if f1_diff > 0.01 or recall_diff > 0.01:
            print("⚠️  差异过大，可能存在计算问题")
        else:
            print("✅ 指标计算正常")

def test_real_scenario():
    """测试实际场景中的不平衡数据"""
    print("\n" + "=" * 60)
    print("不平衡数据场景测试")
    print("=" * 60)
    
    # 模拟MACHO数据集的类别分布
    # 0:'Be'(128), 1:'CEPH'(101), 2:'EB'(255), 3:'LPV'(365), 4:'MOA'(582), 5:'QSO'(59), 6:'RRL'(610)
    class_counts = [128, 101, 255, 365, 582, 59, 610]
    class_names = ['Be', 'CEPH', 'EB', 'LPV', 'MOA', 'QSO', 'RRL']
    
    # 生成不平衡的测试数据
    true_labels = []
    for class_id, count in enumerate(class_counts):
        true_labels.extend([class_id] * count)
    true_labels = np.array(true_labels)
    
    # 生成80%准确率的预测
    predictions = true_labels.copy()
    n_total = len(true_labels)
    n_wrong = int(n_total * 0.2)  # 20%错误
    wrong_indices = np.random.choice(n_total, n_wrong, replace=False)
    
    for idx in wrong_indices:
        wrong_class = np.random.randint(0, 7)
        while wrong_class == true_labels[idx]:
            wrong_class = np.random.randint(0, 7)
        predictions[idx] = wrong_class
    
    # 计算指标
    acc = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    macro_recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    weighted_recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    print(f"数据总数: {n_total}")
    print(f"类别分布: {dict(zip(class_names, class_counts))}")
    print()
    
    print(f"准确率:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"宏平均F1:      {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"加权F1:        {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"宏平均Recall:  {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"加权Recall:    {weighted_recall:.4f} ({weighted_recall*100:.2f}%)")
    
    print("\n📊 详细分析:")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    for i, (name, p, r, f, s) in enumerate(zip(class_names, precision, recall, f1, support)):
        print(f"  {name:<6}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, 支持={s}")

def debug_current_implementation():
    """调试当前实现的问题"""
    print("\n" + "=" * 60)
    print("当前实现调试")
    print("=" * 60)
    
    # 模拟当前training_utils.py中的计算过程
    np.random.seed(42)
    
    # 生成示例数据
    all_predictions = np.random.randint(0, 7, 500)
    all_labels = np.random.randint(0, 7, 500)
    
    # 确保有一定的准确率
    correct_mask = np.random.random(500) < 0.75  # 75%准确率
    all_predictions[correct_mask] = all_labels[correct_mask]
    
    # 按照当前实现计算
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # 计算准确率 (按照当前main loop的方式)
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = 100. * correct / total
    
    # 计算F1和Recall
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    print(f"模拟当前实现结果:")
    print(f"准确率:        {accuracy:.2f}%")
    print(f"宏平均F1:      {macro_f1*100:.2f}%")
    print(f"加权F1:        {weighted_f1*100:.2f}%")
    print(f"宏平均Recall:  {macro_recall*100:.2f}%")
    print(f"加权Recall:    {weighted_recall*100:.2f}%")
    
    print(f"\n差异分析:")
    print(f"加权F1 vs 准确率: {abs(weighted_f1*100 - accuracy):.2f}% 差异")
    print(f"加权Recall vs 准确率: {abs(weighted_recall*100 - accuracy):.2f}% 差异")
    
    if abs(weighted_f1*100 - accuracy) < 2 and abs(weighted_recall*100 - accuracy) < 2:
        print("✅ 计算正常，差异在合理范围内")
    else:
        print("❌ 计算异常，需要检查实现")
        
    return accuracy, weighted_f1*100, weighted_recall*100

if __name__ == "__main__":
    test_metrics_calculation()
    test_real_scenario()
    accuracy, wf1, wr = debug_current_implementation()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("理论上:")
    print("• 加权F1应该非常接近准确率（差异<2%）")
    print("• 加权Recall应该等于准确率（多分类情况下）")
    print("• 宏平均指标可能偏低（受类别不平衡影响）")
    
    print("\n如果实际训练中看到:")
    print("• 准确率80%，但加权F1只有20% → 可能是单位问题（0.2 vs 20%）")
    print("• 或者数据收集有问题（predictions和labels不匹配）")
    print("• 或者计算时机有问题（在错误的地方计算）")