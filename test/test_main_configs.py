#!/usr/bin/env python3
"""
快速测试main.py的完整架构能否运行
"""

import subprocess
import time
import sys

def test_configuration(name, args, timeout=20):
    """测试一个配置"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    cmd = f"python main.py {args} --epochs 1 --batch_size 4"
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
        if "Loss" in result.stdout or "训练" in result.stdout:
            print(f"✅ 成功 (用时: {elapsed:.1f}秒)")
            # 提取关键信息
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Loss', 'Acc', '模型参数', '优化器']):
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"❌ 失败")
            if result.stderr:
                print(f"错误: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ 超时 ({timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


# 测试配置列表
configs = [
    ("基础模型", "--use_stable_training --use_sde 0 --use_contiformer 0 --use_cga 0"),
    ("LNSDE", "--use_stable_training --use_sde 1 --use_contiformer 0 --use_cga 0"),
    ("LNSDE+ContiFormer", "--use_stable_training --use_sde 1 --use_contiformer 1 --use_cga 0"),
    ("完整架构", "--use_stable_training --use_sde 1 --use_contiformer 1 --use_cga 1"),
]

print("="*60)
print("测试main.py完整架构")
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
print("\n推荐命令:")
for name, success in results:
    if success:
        args = next(a for n, a in configs if n == name)
        print(f"# {name}")
        print(f"python main.py {args} --epochs 100 --batch_size 16")
        break