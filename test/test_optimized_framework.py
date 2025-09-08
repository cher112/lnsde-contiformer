#!/usr/bin/env python3
"""
快速测试优化后的框架
验证稳定性修复是否有效
"""

import torch
import sys
import time
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from models import LinearNoiseSDEContiformer
from utils import get_device, set_seed
from utils.stability_fixes import StableTrainingManager, create_optimized_training_config
import pickle
import numpy as np

def quick_test():
    """快速测试训练稳定性"""
    print("="*60)
    print("框架优化测试")
    print("="*60)
    
    # 配置
    config = create_optimized_training_config()
    device = get_device('auto')
    set_seed(42)
    
    print(f"\n优化配置:")
    print(f"  Lion学习率: {config['learning_rate']:.2e}")
    print(f"  梯度裁剪: {config['gradient_clip']}")
    print(f"  标签平滑: {config['label_smoothing']}")
    print(f"  修复零误差: {config['fix_zero_errors']}")
    print(f"  修复时间单调性: {config['fix_time_monotonicity']}")
    
    # 创建最简单的模型
    model = LinearNoiseSDEContiformer(
        input_dim=3,
        hidden_channels=32,
        num_classes=7,
        contiformer_dim=64,
        n_heads=2,
        n_layers=1,
        use_sde=False,  # 先禁用SDE
        use_contiformer=False,
        use_cga=False
    ).to(device)
    
    print(f"\n模型配置:")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  设备: {device}")
    
    # 创建训练管理器
    manager = StableTrainingManager(model, config)
    
    # 加载少量数据测试
    print("\n加载测试数据...")
    data_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 只使用前100个样本
    data = data[:100]
    
    # 准备批次
    batch_size = 8
    batches = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        
        # 提取数据
        times_list = []
        values_list = []
        labels_list = []
        max_len = 0
        
        for sample in batch_data:
            t = sample['time'][:200]  # 限制长度
            m = sample['mag'][:200]
            e = sample['errmag'][:200]
            
            times_list.append(t)
            values_list.append(np.stack([m, e], axis=-1))
            labels_list.append(sample['label'])
            max_len = max(max_len, len(t))
        
        # Padding
        times_tensor = torch.zeros(len(batch_data), max_len)
        values_tensor = torch.zeros(len(batch_data), max_len, 2)
        mask_tensor = torch.zeros(len(batch_data), max_len, dtype=torch.bool)
        
        for j, (t, v) in enumerate(zip(times_list, values_list)):
            length = len(t)
            times_tensor[j, :length] = torch.tensor(t, dtype=torch.float32)
            values_tensor[j, :length] = torch.tensor(v, dtype=torch.float32)
            mask_tensor[j, :length] = True
        
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        batches.append((times_tensor, values_tensor, labels_tensor, mask_tensor))
        
        if len(batches) >= 5:  # 只用5个批次测试
            break
    
    print(f"准备了 {len(batches)} 个批次")
    
    # 训练测试
    print("\n开始训练测试...")
    start_time = time.time()
    
    losses = []
    accs = []
    
    for i, batch in enumerate(batches):
        result = manager.train_step(batch)
        
        if not result['skipped']:
            losses.append(result['loss'])
            accs.append(result['acc'])
            print(f"Batch {i+1}: Loss={result['loss']:.4f}, Acc={result['acc']:.4f}")
        else:
            print(f"Batch {i+1}: SKIPPED (NaN detected)")
    
    elapsed = time.time() - start_time
    
    # 统计
    print(f"\n测试完成 (耗时: {elapsed:.2f}秒)")
    if losses:
        print(f"平均Loss: {np.mean(losses):.4f}")
        print(f"平均Acc: {np.mean(accs):.4f}")
        print(f"NaN批次: {len(batches) - len(losses)}")
        
        if np.mean(losses) < 10 and len(losses) == len(batches):
            print("\n✅ 框架优化成功！")
            print("  - 无NaN Loss")
            print("  - 数值稳定")
            print("  - GPU运算正常")
        else:
            print("\n⚠️ 仍有稳定性问题")
    else:
        print("❌ 所有批次都失败了")
    
    # 测试带SDE的版本
    print("\n" + "="*40)
    print("测试SDE组件...")
    
    model_sde = LinearNoiseSDEContiformer(
        input_dim=3,
        hidden_channels=32,
        num_classes=7,
        contiformer_dim=64,
        n_heads=2,
        n_layers=1,
        use_sde=True,  # 启用SDE
        use_contiformer=False,
        use_cga=False,
        dt=0.1,  # 更大的步长
        sde_method='euler'
    ).to(device)
    
    manager_sde = StableTrainingManager(model_sde, config)
    
    # 测试一个批次
    batch = batches[0]
    result = manager_sde.train_step(batch)
    
    if not result['skipped']:
        print(f"✅ SDE测试成功: Loss={result['loss']:.4f}")
    else:
        print("❌ SDE仍有NaN问题")
    
    return len(losses) == len(batches)


if __name__ == "__main__":
    success = quick_test()
    
    print("\n" + "="*60)
    print("优化总结")
    print("="*60)
    
    print("""
关键修复:
1. ✅ Mask除零错误 - 添加eps避免除零
2. ✅ Lion学习率 - 降低到5e-6（原来的1/20）
3. ✅ 零误差处理 - clamp到最小值1e-6
4. ✅ 时间单调性 - 自动修复非递增序列
5. ✅ 纯Tensor返回 - 避免tuple，统一GPU运算

测试命令:
# 最稳定配置（推荐）
python main.py --use_stable_training --use_sde 0 --use_contiformer 0 --epochs 1 --batch_size 8

# 逐步启用组件
python main.py --use_stable_training --use_sde 1 --use_contiformer 0 --epochs 1

# 完整配置
python main.py --use_stable_training --learning_rate 5e-6 --gradient_clip 0.5 --epochs 1
""")
    
    if success:
        print("🎉 框架优化完成，可以稳定训练！")
    else:
        print("⚠️ 请进一步调试")