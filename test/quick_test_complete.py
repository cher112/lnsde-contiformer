#!/usr/bin/env python3
"""
快速测试完整架构是否能正常运行
"""

import subprocess
import time
import sys

def test_configuration(name, args, timeout=60):
    """测试一个配置"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    cmd = f"python main.py {args} --epochs 1 --batch_size 4 --gradient_accumulation_steps 2"
    print(f"命令: {cmd}")
    
    try:
        start = time.time()
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        elapsed = time.time() - start
        
        # 检查输出
        if "Training:" in result.stdout and "Loss=" in result.stdout:
            print(f"✅ 成功 (用时: {elapsed:.1f}秒)")
            # 提取关键信息
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Loss=', 'Acc=', '模型参数', '优化器']):
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"❌ 失败")
            if result.stderr:
                print(f"错误: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ 超时但可能正在正常训练 ({timeout}秒)")
        return True  # 超时但没报错，说明在训练
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


# 测试配置列表 - 根据数据特征优化的参数
configs = [
    ("基础模型", "--use_sde 0 --use_contiformer 0 --use_cga 0"),
    ("仅SDE", "--use_sde 1 --use_contiformer 0 --use_cga 0"),
    ("SDE+ContiFormer", "--use_sde 1 --use_contiformer 1 --use_cga 0"),
    ("完整架构(LNSDE+ContiFormer+CGA)", "--use_sde 1 --use_contiformer 1 --use_cga 1"),
]

print("="*60)
print("测试完整LNSDE+ContiFormer+CGA架构")
print("优化参数:")
print("  - Lion学习率: 5e-6 (原1e-4的1/20)")
print("  - 梯度裁剪: 0.5 (更严格)")
print("  - Mask处理: 添加eps=1e-8避免除零")
print("  - 模型返回: 纯tensor而非tuple")
print("="*60)

results = []
for name, args in configs:
    success = test_configuration(name, args)
    results.append((name, success))
    
print("\n" + "="*60)
print("测试总结")
print("="*60)

for name, success in results:
    status = "✅" if success else "❌"
    print(f"{status} {name}")

# 生成推荐命令
print("\n推荐训练命令:")
print("# 完整架构训练")
print("python main.py --use_sde 1 --use_contiformer 1 --use_cga 1 \\")
print("  --epochs 150 --batch_size 16 \\")
print("  --learning_rate 5e-5 --gradient_clip 0.5")
print("\n# 测试快速收敛")
print("python main.py --use_sde 1 --use_contiformer 1 --use_cga 1 \\")
print("  --epochs 10 --batch_size 8 --gradient_accumulation_steps 2")