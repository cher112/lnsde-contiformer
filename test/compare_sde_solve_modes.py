#!/usr/bin/env python3
"""
比较两种SDE求解模式的差异
- 模式0: 逐步求解（当前实现）
- 模式1: 一次性求解整个轨迹（demo.ipynb方式）
"""

import torch
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.linear_noise_sde import LinearNoiseSDEContiformer


def create_test_data(batch_size=4, seq_len=50, device='cpu'):
    """创建测试数据"""
    # 模拟不规则时间序列
    time_series = torch.randn(batch_size, seq_len, 3).to(device)  # (batch, seq_len, 3)

    # 创建不规则时间点
    for i in range(batch_size):
        # 生成递增的时间序列（模拟不规则采样）
        times = torch.cumsum(torch.rand(seq_len) * 0.5 + 0.1, dim=0)
        time_series[i, :, 0] = times

    # 创建mask（模拟不同长度的序列）
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    for i in range(batch_size):
        # 随机截断长度
        valid_len = np.random.randint(seq_len // 2, seq_len)
        mask[i, valid_len:] = False

    return time_series, mask


def compare_solve_modes(device='cpu', num_trials=3):
    """比较两种求解模式"""
    print("=" * 80)
    print("SDE求解模式对比测试")
    print("=" * 80)

    # 模型参数
    model_params = {
        'input_dim': 3,
        'hidden_channels': 32,  # 较小的维度以加快测试
        'contiformer_dim': 64,
        'num_classes': 7,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1,
        'sde_method': 'euler',
        'dt': 0.05,
        'rtol': 1e-2,
        'atol': 1e-3,
        'use_sde': True,
        'use_contiformer': True,
        'use_cga': False
    }

    # 创建两个模型（参数相同，只是求解模式不同）
    print("\n创建模型...")
    model_stepwise = LinearNoiseSDEContiformer(**model_params, sde_solve_mode=0).to(device)
    model_full = LinearNoiseSDEContiformer(**model_params, sde_solve_mode=1).to(device)

    # 使用相同的初始化参数
    model_full.load_state_dict(model_stepwise.state_dict())

    model_stepwise.eval()
    model_full.eval()

    print(f"  模式0: 逐步求解")
    print(f"  模式1: 一次性求解整个轨迹")

    # 创建测试数据
    print("\n创建测试数据...")
    time_series, mask = create_test_data(batch_size=4, seq_len=50, device=device)
    print(f"  批次大小: {time_series.shape[0]}")
    print(f"  序列长度: {time_series.shape[1]}")
    print(f"  有效长度: {mask.sum(dim=1).tolist()}")

    # 多次测试取平均
    print(f"\n进行 {num_trials} 次测试...")

    stepwise_times = []
    full_times = []
    outputs_stepwise_list = []
    outputs_full_list = []

    with torch.no_grad():
        for trial in range(num_trials):
            print(f"\n--- 测试 {trial + 1}/{num_trials} ---")

            # 测试模式0: 逐步求解
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            outputs_stepwise = model_stepwise(time_series, mask)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            stepwise_time = time.time() - start_time
            stepwise_times.append(stepwise_time)
            outputs_stepwise_list.append(outputs_stepwise.cpu())
            print(f"  模式0 耗时: {stepwise_time:.4f}秒")

            # 测试模式1: 一次性求解
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            outputs_full = model_full(time_series, mask)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            full_time = time.time() - start_time
            full_times.append(full_time)
            outputs_full_list.append(outputs_full.cpu())
            print(f"  模式1 耗时: {full_time:.4f}秒")

            # 计算输出差异
            diff = torch.abs(outputs_stepwise - outputs_full)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  输出差异: 最大={max_diff:.6f}, 平均={mean_diff:.6f}")

    # 统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    avg_stepwise_time = np.mean(stepwise_times)
    avg_full_time = np.mean(full_times)

    print(f"\n【计算时间对比】")
    print(f"  模式0 (逐步求解)     : {avg_stepwise_time:.4f} ± {np.std(stepwise_times):.4f} 秒")
    print(f"  模式1 (一次性求解)   : {avg_full_time:.4f} ± {np.std(full_times):.4f} 秒")
    print(f"  速度比 (模式1/模式0) : {avg_full_time/avg_stepwise_time:.2f}x")

    # 计算所有测试的平均差异
    all_diffs = []
    for out_s, out_f in zip(outputs_stepwise_list, outputs_full_list):
        diff = torch.abs(out_s - out_f)
        all_diffs.append(diff)

    all_diffs_tensor = torch.stack(all_diffs)

    print(f"\n【输出差异统计】")
    print(f"  最大差异: {all_diffs_tensor.max().item():.6f}")
    print(f"  平均差异: {all_diffs_tensor.mean().item():.6f}")
    print(f"  中位数差异: {all_diffs_tensor.median().item():.6f}")
    print(f"  标准差: {all_diffs_tensor.std().item():.6f}")

    # 判断一致性
    print(f"\n【一致性评估】")
    threshold_close = 1e-3
    threshold_similar = 1e-2

    if all_diffs_tensor.max().item() < threshold_close:
        print(f"  ✅ 两种模式输出非常接近 (差异 < {threshold_close})")
        print(f"  → 说明两种实现在数值上等价")
    elif all_diffs_tensor.max().item() < threshold_similar:
        print(f"  ⚠️  两种模式输出相似但有差异 ({threshold_close} < 差异 < {threshold_similar})")
        print(f"  → 可能是数值精度或随机性导致的差异")
    else:
        print(f"  ❌ 两种模式输出存在显著差异 (差异 > {threshold_similar})")
        print(f"  → 需要检查实现是否正确")

    # 预测一致性检查
    print(f"\n【预测一致性】")
    preds_stepwise = torch.argmax(outputs_stepwise_list[0], dim=1)
    preds_full = torch.argmax(outputs_full_list[0], dim=1)
    agreement = (preds_stepwise == preds_full).float().mean().item()
    print(f"  预测一致性: {agreement*100:.1f}%")
    print(f"  模式0预测: {preds_stepwise.tolist()}")
    print(f"  模式1预测: {preds_full.tolist()}")

    return {
        'stepwise_time': avg_stepwise_time,
        'full_time': avg_full_time,
        'max_diff': all_diffs_tensor.max().item(),
        'mean_diff': all_diffs_tensor.mean().item(),
        'agreement': agreement
    }


def test_consistency(device='cpu'):
    """测试全局一致性：验证逐步求解是否学到统一的动力学"""
    print("\n" + "=" * 80)
    print("全局一致性测试")
    print("=" * 80)
    print("测试逐步求解是否等价于一次性求解")

    # 这个测试通过比较两种模式的输出来验证
    # 如果差异很小，说明逐步求解学到了统一的全局动力学

    results = compare_solve_modes(device=device, num_trials=5)

    print("\n【结论】")
    if results['max_diff'] < 1e-3:
        print("  ✅ 逐步求解与一次性求解高度一致")
        print("  → 模型学到了统一的全局动力学系统")
    else:
        print("  ⚠️  两种模式存在差异")
        print("  → 可能原因：")
        print("    1. 数值求解器的累积误差")
        print("    2. 浮点运算的顺序差异")
        print("    3. 布朗运动的采样方式不同")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='比较SDE求解模式')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    parser.add_argument('--trials', type=int, default=3,
                       help='测试次数')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print(f"\n使用设备: {device}")

    # 运行对比测试
    compare_solve_modes(device=device, num_trials=args.trials)

    # 运行一致性测试
    # test_consistency(device=device)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    print("\n使用方法:")
    print("  # 训练时使用模式0（逐步求解，默认）")
    print("  python main.py --sde_solve_mode 0")
    print()
    print("  # 训练时使用模式1（一次性求解）")
    print("  python main.py --sde_solve_mode 1")
