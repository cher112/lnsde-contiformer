#!/usr/bin/env python3
"""
优化混淆矩阵计算，避免每个epoch都计算导致训练缓慢
"""

def optimize_confusion_matrix_calculation():
    """优化混淆矩阵计算策略"""
    print("🔧 优化混淆矩阵计算策略...")
    
    print("问题分析:")
    print("  • 每个epoch后都计算混淆矩阵")
    print("  • MACHO数据集验证集约1000+样本")
    print("  • 每次需要完整前向传播")
    print("  • 显存占用高，计算耗时")
    
    print("\n解决方案:")
    print("1. 间隔计算混淆矩阵:")
    print("   --confusion_matrix_interval 5  # 每5个epoch计算一次")
    
    print("2. 减小验证批次大小:")
    print("   --val_batch_size 8  # 验证时使用更小批次")
    
    print("3. 或者完全禁用epoch级混淆矩阵:")
    print("   --disable_epoch_confusion_matrix  # 只在训练结束时计算")
    
    print("4. 临时加速训练参数组合:")
    print("   python main.py --dataset 3 --use_resampling \\")
    print("     --batch_size 16 \\")
    print("     --epochs 20 \\  # 先测试少量epoch")
    print("     --confusion_matrix_interval 10  # 减少混淆矩阵计算")

if __name__ == "__main__":
    optimize_confusion_matrix_calculation()