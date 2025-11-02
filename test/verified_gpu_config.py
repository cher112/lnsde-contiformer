#!/usr/bin/env python3
"""
实际可用的GPU优化配置
"""

print("="*60)
print("🔧 实际可用的GPU优化配置")
print("="*60)

print("\n✅ 已验证的参数:")
print("• --num_workers (default=16)")
print("• --use_gradient_checkpoint") 
print("• --batch_size")
print("• --gradient_accumulation_steps")
print("• --amp (默认启用)")
print("• --pin_memory (default=True)")
print("• --prefetch_factor (default=4)")

print("\n🎯 测试就绪的优化配置:")

configs = [
    {
        'name': '🥉 保守配置',
        'batch_size': 16,
        'gradient_accumulation_steps': 4,
        'use_gradient_checkpoint': '',
        'num_workers': 4,
        'expected_gpu': '50-65%'
    },
    {
        'name': '🥈 平衡配置', 
        'batch_size': 24,
        'gradient_accumulation_steps': 3,
        'use_gradient_checkpoint': '--use_gradient_checkpoint',
        'num_workers': 8,
        'expected_gpu': '70-85%'
    },
    {
        'name': '🥇 激进配置',
        'batch_size': 32, 
        'gradient_accumulation_steps': 2,
        'use_gradient_checkpoint': '--use_gradient_checkpoint',
        'num_workers': 12,
        'expected_gpu': '85-95%'
    }
]

for config in configs:
    print(f"\n{config['name']} (预期GPU: {config['expected_gpu']}):")
    
    cmd_parts = [
        "python main.py \\",
        f"  --dataset MACHO \\",
        f"  --batch_size {config['batch_size']} \\",
        f"  --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\",
        f"  --num_workers {config['num_workers']} \\",
        f"  --learning_rate 1e-5 \\",
        f"  --gradient_clip 0.5 \\",
    ]
    
    if config['use_gradient_checkpoint']:
        cmd_parts.append(f"  {config['use_gradient_checkpoint']} \\")
    
    cmd_parts.extend([
        "  --epochs 3 \\",
        "  --save_best_only"
    ])
    
    for part in cmd_parts:
        print(part)

print("\n" + "="*60)
print("🚀 立即可测试命令")
print("="*60)

print("\n推荐从这个开始:")
print("python main.py \\")
print("  --dataset MACHO \\")
print("  --batch_size 24 \\")
print("  --gradient_accumulation_steps 3 \\")
print("  --use_gradient_checkpoint \\")
print("  --num_workers 8 \\")
print("  --learning_rate 1e-5 \\")
print("  --gradient_clip 0.5 \\")
print("  --epochs 1")

print("\n监控命令:")
print("watch -n 1 'nvidia-smi | head -15'")

print("\n" + "="*60)
print("参数说明:")
print("• batch_size=24: 单批次大小")
print("• gradient_accumulation_steps=3: 等效batch=72")  
print("• use_gradient_checkpoint: 节省50-80%内存")
print("• num_workers=8: 8个进程并行加载数据")
print("• 混合精度默认启用 (除非加--no_amp)")
print("="*60)