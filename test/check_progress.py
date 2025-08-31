#!/usr/bin/env python3
"""检查所有运行中任务的进度"""

import subprocess
import re

# 任务列表
tasks = {
    'bash_3': 'ASAS Geometric SDE',
    'bash_6': 'LINEAR Geometric SDE', 
    'bash_9': 'MACHO Geometric SDE',
    'bash_16': 'ASAS Langevin SDE',
    'bash_17': 'LINEAR Langevin SDE',
    'bash_18': 'MACHO Langevin SDE'
}

print("=" * 60)
print("当前运行任务进度")
print("=" * 60)

for bash_id, task_name in tasks.items():
    # 获取最新的输出
    try:
        # 尝试从BashOutput获取信息
        result = subprocess.run(
            f"tail -100 /tmp/{bash_id}_output.log 2>/dev/null | grep -E 'Epoch \[' | tail -1",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout:
            # 解析epoch信息
            match = re.search(r'Epoch \[(\d+)/(\d+)\]', result.stdout)
            if match:
                current_epoch = match.group(1)
                total_epochs = match.group(2)
                progress = int(current_epoch) / int(total_epochs) * 100
                print(f"\n{task_name}:")
                print(f"  进度: Epoch {current_epoch}/{total_epochs} ({progress:.1f}%)")
            else:
                print(f"\n{task_name}: 正在训练中...")
        else:
            print(f"\n{task_name}: 无法获取进度信息")
            
    except Exception as e:
        print(f"\n{task_name}: 检查失败 - {e}")

print("\n" + "=" * 60)