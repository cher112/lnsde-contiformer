#!/usr/bin/env python3
"""
修复梯度裁剪和内存泄漏问题
"""

print("="*60)
print("梯度裁剪和内存问题修复方案")
print("="*60)

print("\n问题1: 梯度裁剪参数未生效")
print("-"*40)
print("原因: training_utils.py中硬编码了max_norm=1.0，忽略了命令行参数")
print("位置: utils/training_utils.py:97,117")
print("解决: 需要将args.gradient_clip传递到训练函数中")

print("\n问题2: SDE求解导致内存泄漏")
print("-"*40)
print("原因: SDE数值求解产生大量中间张量，计算图没有及时释放")
print("表现: RAM增长但GPU使用率不变，卡在某个batch")
print("解决方案:")

print("\n1. 立即修复 - 启用更严格的梯度断开")
print("python main.py \\")
print("  --dataset MACHO \\")
print("  --learning_rate 1e-5 \\")
print("  --gradient_clip 0.5 \\")  
print("  --use_optimization \\")  # 这会启用梯度断开
print("  --batch_size 16 \\")
print("  --epochs 1")

print("\n2. 中期修复 - 调整SDE求解参数")
print("在models/linear_noise_sde.py中:")
print("- dt: 0.05 → 0.1 (减少求解步数)")
print("- rtol: 1e-5 → 1e-3 (放松精度)")
print("- atol: 1e-6 → 1e-4 (放松精度)")

print("\n3. 长期修复 - 模块化梯度裁剪")
print("创建统一的梯度管理器，确保参数正确传递")

print("\n临时解决RAM增长:")
print("if batch_idx % 10 == 0:")
print("    gc.collect()")
print("    torch.cuda.empty_cache()")

print("\n检测卡顿的命令:")
print("watch -n 1 'nvidia-smi | head -20; echo \"--- RAM ---\"; free -h'")

print("\n="*60)
print("推荐测试命令 (最稳定配置):")
print("="*60)
print("python main.py \\")
print("  --dataset MACHO \\")
print("  --use_sde 1 \\")
print("  --use_contiformer 0 \\")  # 先禁用ContiFormer
print("  --use_cga 0 \\")           # 先禁用CGA  
print("  --learning_rate 5e-6 \\")  # 很低的学习率
print("  --gradient_clip 0.1 \\")   # 很强的裁剪
print("  --batch_size 8 \\")        # 小批次
print("  --no_amp \\")              # 禁用混合精度
print("  --epochs 1")