#!/usr/bin/env python3
"""
GPU使用率优化完整解决方案
"""

print("="*70)
print("🚀 GPU使用率优化完整解决方案")
print("="*70)

print("\n📊 现状分析:")
print("• 当前GPU使用率: 9% (严重不足)")  
print("• RTX 4090: 24GB VRAM, 128个SM处理器")
print("• 问题根源: SDE计算图深度 + ContiFormer注意力 + 小批次大小")

print("\n🎯 优化策略组合:")

configs = [
    {
        'name': '🥉 保守配置 (稳定优先)',
        'batch_size': 16,
        'gradient_accumulation_steps': 4,
        'use_gradient_checkpoint': False,
        'amp': True,
        'learning_rate': '1e-5',
        'gradient_clip': '1.0',
        'num_workers': 4,
        'expected_gpu_util': '50-65%',
        'expected_memory': '8-12GB',
        'risk': '低',
        'description': '最稳定，适合调试和验证'
    },
    {
        'name': '🥈 平衡配置 (推荐)',
        'batch_size': 24,
        'gradient_accumulation_steps': 3,
        'use_gradient_checkpoint': True,
        'amp': True,
        'learning_rate': '1e-5',
        'gradient_clip': '0.5',
        'num_workers': 6,
        'expected_gpu_util': '70-85%',
        'expected_memory': '14-18GB',
        'risk': '中',
        'description': '最佳性能/稳定性平衡'
    },
    {
        'name': '🥇 激进配置 (最大化GPU)',
        'batch_size': 32,
        'gradient_accumulation_steps': 2,
        'use_gradient_checkpoint': True,
        'amp': True,
        'learning_rate': '5e-6',
        'gradient_clip': '0.3',
        'num_workers': 8,
        'expected_gpu_util': '85-95%',
        'expected_memory': '18-22GB',
        'risk': '高',
        'description': '最大化GPU利用率，可能OOM'
    },
    {
        'name': '⚡ 极限配置 (实验性)',
        'batch_size': 48,
        'gradient_accumulation_steps': 1,
        'use_gradient_checkpoint': True,
        'amp': True,
        'learning_rate': '3e-6',
        'gradient_clip': '0.1',
        'num_workers': 12,
        'expected_gpu_util': '95-100%',
        'expected_memory': '20-24GB',
        'risk': '极高',
        'description': '极限测试，高概率OOM'
    }
]

for i, config in enumerate(configs):
    print(f"\n{config['name']}:")
    print(f"   批次大小: {config['batch_size']}")
    print(f"   梯度累积: {config['gradient_accumulation_steps']} (等效批次={config['batch_size']*config['gradient_accumulation_steps']})")
    print(f"   梯度检查点: {'✅' if config['use_gradient_checkpoint'] else '❌'}")
    print(f"   混合精度: {'✅' if config['amp'] else '❌'}")
    print(f"   学习率: {config['learning_rate']}")
    print(f"   梯度裁剪: {config['gradient_clip']}")
    print(f"   数据加载进程: {config['num_workers']}")
    print(f"   预期GPU利用率: {config['expected_gpu_util']}")
    print(f"   预期内存使用: {config['expected_memory']}")
    print(f"   风险等级: {config['risk']}")
    print(f"   说明: {config['description']}")

print("\n" + "="*70)
print("🛠️ 具体执行命令")
print("="*70)

# 生成具体命令
for i, config in enumerate(configs):
    print(f"\n{config['name']}:")
    
    cmd_parts = [
        "python main.py \\",
        f"  --dataset MACHO \\",
        f"  --batch_size {config['batch_size']} \\",
        f"  --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\",
        f"  --learning_rate {config['learning_rate']} \\",
        f"  --gradient_clip {config['gradient_clip']} \\",
        f"  --num_workers {config['num_workers']} \\",
    ]
    
    if config['use_gradient_checkpoint']:
        cmd_parts.append("  --use_gradient_checkpoint \\")
    
    if config['amp']:
        cmd_parts.append("  --amp \\")
    
    cmd_parts.extend([
        "  --epochs 3 \\",
        "  --save_best_only"
    ])
    
    for part in cmd_parts:
        print(part)

print("\n" + "="*70)
print("📈 性能监控")
print("="*70)

monitor_commands = {
    "实时GPU监控": "watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits'",
    "详细性能分析": "nvidia-smi dmon -s pucvmet",
    "Python内存监控": "ps aux | grep python | head -5",
    "系统资源监控": "htop"
}

for name, cmd in monitor_commands.items():
    print(f"\n{name}:")
    print(f"  {cmd}")

print("\n" + "="*70)
print("⚠️ 重要注意事项")
print("="*70)

warnings = [
    "1. 从保守配置开始测试，逐步增加批次大小",
    "2. 监控GPU内存使用，接近90%时停止增加批次",
    "3. 梯度检查点会增加15-30%计算时间但节省50-80%内存",
    "4. 混合精度训练必须启用，可节省50%内存",
    "5. 学习率需要根据等效批次大小调整",
    "6. num_workers过多可能导致CPU瓶颈",
    "7. OOM时减小batch_size而不是关闭功能"
]

for warning in warnings:
    print(f"  {warning}")

print("\n" + "="*70)
print("🎯 推荐执行顺序")
print("="*70)

steps = [
    "1. 🧪 测试保守配置验证基础功能",
    "2. 📊 运行平衡配置进行正式训练", 
    "3. ⚡ 尝试激进配置最大化性能",
    "4. 🔍 根据实际内存使用微调参数",
    "5. 📈 持续监控并记录最优配置"
]

for step in steps:
    print(f"  {step}")

print(f"\n{'='*70}")
print("✅ GPU优化方案已完成!")
print("从平衡配置开始，预期GPU使用率可达70-85%！")
print("="*70)