#!/usr/bin/env python3
"""
测试模型前向传播中的NaN/Inf问题
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_model_forward():
    """测试模型前向传播"""
    # 导入必要的模块
    from models import LinearNoiseSDEContiformer
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 200
    num_classes = 8
    
    # 创建不同类型的测试输入 (batch, seq_len, 3)
    test_inputs = {
        "正常输入": torch.randn(batch_size, seq_len, 3),
        "小值输入": torch.randn(batch_size, seq_len, 3) * 1e-5,
        "大值输入": torch.randn(batch_size, seq_len, 3) * 100,
        "包含零的输入": torch.zeros(batch_size, seq_len, 3),
        "混合输入": torch.randn(batch_size, seq_len, 3),
    }
    
    # 在混合输入中加入一些极端值
    test_inputs["混合输入"][0, :10, :] = 0  # 一些零值
    test_inputs["混合输入"][1, :10, :] = 1e-8  # 一些很小的值
    test_inputs["混合输入"][2, :10, :] = 1000  # 一些大值
    
    # 测试不同的模型配置
    configs = [
        {"use_sde": True, "use_contiformer": True, "use_cga": True, "name": "完整模型"},
        {"use_sde": False, "use_contiformer": True, "use_cga": True, "name": "无SDE"},
        {"use_sde": True, "use_contiformer": False, "use_cga": True, "name": "无Contiformer"},
        {"use_sde": True, "use_contiformer": True, "use_cga": False, "name": "无CGA"},
        {"use_sde": False, "use_contiformer": False, "use_cga": False, "name": "基础模型"},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*60}")
        
        # 创建模型
        model = LinearNoiseSDEContiformer(
            input_dim=3,
            num_classes=num_classes,
            hidden_channels=64,
            contiformer_dim=128,
            n_heads=8,
            n_layers=4,
            dropout=0.1,
            use_sde=config['use_sde'],
            use_contiformer=config['use_contiformer'],
            use_cga=config['use_cga'],
            cga_group_dim=16,
            cga_heads=4,
            cga_temperature=1.0,
            cga_gate_threshold=0.5
        ).to(device)
        
        model.eval()
        
        # 测试每种输入
        for input_name, test_input in test_inputs.items():
            test_input = test_input.to(device)
            
            print(f"\n测试输入: {input_name}")
            print(f"  输入统计: min={test_input.min():.4f}, max={test_input.max():.4f}, mean={test_input.mean():.4f}")
            
            with torch.no_grad():
                try:
                    # 记录中间输出
                    def check_hook(name):
                        def hook(module, input, output):
                            if isinstance(output, torch.Tensor):
                                has_nan = torch.isnan(output).any()
                                has_inf = torch.isinf(output).any()
                                if has_nan or has_inf:
                                    print(f"  ⚠️ {name}: NaN={has_nan}, Inf={has_inf}")
                                    if has_nan:
                                        nan_ratio = torch.isnan(output).float().mean().item()
                                        print(f"     NaN比例: {nan_ratio*100:.2f}%")
                                    if has_inf:
                                        inf_ratio = torch.isinf(output).float().mean().item()
                                        print(f"     Inf比例: {inf_ratio*100:.2f}%")
                        return hook
                    
                    # 注册hooks
                    hooks = []
                    if hasattr(model, 'embedding'):
                        hooks.append(model.embedding.register_forward_hook(check_hook('Embedding')))
                    if hasattr(model, 'contiformer') and model.use_contiformer:
                        hooks.append(model.contiformer.register_forward_hook(check_hook('Contiformer')))
                    if hasattr(model, 'lnsde') and model.use_sde:
                        hooks.append(model.lnsde.register_forward_hook(check_hook('LNSDE')))
                    if hasattr(model, 'cga') and model.use_cga:
                        hooks.append(model.cga.register_forward_hook(check_hook('CGA')))
                    
                    # 前向传播
                    output = model(test_input)
                    
                    # 检查输出
                    has_nan = torch.isnan(output).any()
                    has_inf = torch.isinf(output).any()
                    
                    if has_nan or has_inf:
                        print(f"  ❌ 输出: NaN={has_nan}, Inf={has_inf}")
                    else:
                        print(f"  ✓ 输出正常: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
                    
                    # 清理hooks
                    for hook in hooks:
                        hook.remove()
                    
                except Exception as e:
                    print(f"  ❌ 错误: {e}")

def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("测试梯度流")
    print("="*60)
    
    from models import LinearNoiseSDEContiformer
    import torch.nn as nn
    import torch.optim as optim
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单的测试场景
    model = LinearNoiseSDEContiformer(
        input_dim=3,
        num_classes=8,
        hidden_channels=64,
        contiformer_dim=128,
        n_heads=8,
        n_layers=4,
        dropout=0.0,  # 关闭dropout以减少随机性
        use_sde=True,
        use_contiformer=True,
        use_cga=True,
        cga_group_dim=16,
        cga_heads=4,
        cga_temperature=1.0,
        cga_gate_threshold=0.5
    ).to(device)
    
    model.train()
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 测试不同的输入
    test_scenarios = [
        ("正常输入", torch.randn(4, 200, 3) * 0.1),
        ("小值输入", torch.randn(4, 200, 3) * 1e-5),
        ("大值输入", torch.randn(4, 200, 3) * 10),
    ]
    
    for scenario_name, test_input in test_scenarios:
        print(f"\n测试场景: {scenario_name}")
        
        test_input = test_input.to(device)
        target = torch.randint(0, 8, (4,)).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        
        print(f"  Loss: {loss.item():.4f}")
        
        # 反向传播
        try:
            loss.backward()
            
            # 检查梯度
            total_params = 0
            nan_grad_params = 0
            inf_grad_params = 0
            zero_grad_params = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_params += param.numel()
                    
                    if torch.isnan(param.grad).any():
                        nan_grad_params += torch.isnan(param.grad).sum().item()
                        print(f"    ⚠️ NaN梯度: {name}")
                    
                    if torch.isinf(param.grad).any():
                        inf_grad_params += torch.isinf(param.grad).sum().item()
                        print(f"    ⚠️ Inf梯度: {name}")
                    
                    if (param.grad == 0).all():
                        zero_grad_params += param.numel()
            
            if nan_grad_params > 0:
                print(f"  NaN梯度参数: {nan_grad_params}/{total_params} ({100*nan_grad_params/total_params:.2f}%)")
            if inf_grad_params > 0:
                print(f"  Inf梯度参数: {inf_grad_params}/{total_params} ({100*inf_grad_params/total_params:.2f}%)")
            if zero_grad_params > 0:
                print(f"  零梯度参数: {zero_grad_params}/{total_params} ({100*zero_grad_params/total_params:.2f}%)")
            
            if nan_grad_params == 0 and inf_grad_params == 0:
                print(f"  ✓ 梯度正常")
            
        except Exception as e:
            print(f"  ❌ 反向传播失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("模型NaN/Inf调试工具")
    print("="*60)
    
    # 测试前向传播
    test_model_forward()
    
    # 测试梯度流
    test_gradient_flow()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)