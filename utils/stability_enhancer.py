"""
数值稳定性增强模块 - 用于SDE训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilizedSDE(nn.Module):
    """包装SDE模块，增加数值稳定性"""
    
    def __init__(self, sde_module):
        super().__init__()
        self.sde = sde_module
        self.eps = 1e-6
        self.max_norm = 10.0
        
    def forward(self, t, y):
        """前向传播with数值稳定性保护"""
        # 输入归一化
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        y_normalized = y / (y_norm + self.eps)
        
        # 调用原始SDE
        output = self.sde(t, y_normalized * torch.clamp(y_norm, max=self.max_norm))
        
        # 输出裁剪
        output = torch.clamp(output, -self.max_norm, self.max_norm)
        
        # NaN/Inf检测和替换
        if torch.isnan(output).any() or torch.isinf(output).any():
            mask = torch.isnan(output) | torch.isinf(output)
            output = torch.where(mask, torch.zeros_like(output), output)
            
        return output


class GradientStabilizer(nn.Module):
    """梯度稳定器 - 防止梯度爆炸"""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.gradient_scale = 0.1
        
    def forward(self, *args, **kwargs):
        # 前向传播
        output = self.module(*args, **kwargs)
        
        # 在反向传播时缩放梯度
        if self.training:
            output = output * self.gradient_scale + output.detach() * (1 - self.gradient_scale)
            
        return output


def add_stability_hooks(model):
    """为模型添加稳定性钩子"""
    
    def gradient_clipping_hook(grad):
        """梯度裁剪钩子"""
        if grad is not None:
            # 裁剪梯度范数
            grad_norm = torch.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm
            # 移除NaN/Inf
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                grad = torch.zeros_like(grad)
        return grad
    
    # 为所有参数注册钩子
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(gradient_clipping_hook)
    
    return model


def stabilize_optimizer(optimizer, max_grad_norm=0.1):
    """稳定优化器设置"""
    
    # 调整优化器参数
    for param_group in optimizer.param_groups:
        # 降低学习率
        param_group['lr'] = min(param_group['lr'], 1e-5)
        # 增加权重衰减
        if 'weight_decay' in param_group:
            param_group['weight_decay'] = max(param_group['weight_decay'], 1e-3)
        # 设置梯度裁剪
        param_group['max_grad_norm'] = max_grad_norm
        
    return optimizer


def check_model_health(model, batch_data):
    """检查模型健康状态"""
    
    health_report = {
        'has_nan_params': False,
        'has_inf_params': False,
        'has_nan_gradients': False,
        'has_inf_gradients': False,
        'max_param_norm': 0,
        'max_grad_norm': 0
    }
    
    # 检查参数
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            health_report['has_nan_params'] = True
            print(f"⚠️ NaN detected in parameter: {name}")
        if torch.isinf(param).any():
            health_report['has_inf_params'] = True
            print(f"⚠️ Inf detected in parameter: {name}")
        
        param_norm = torch.norm(param).item()
        health_report['max_param_norm'] = max(health_report['max_param_norm'], param_norm)
        
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health_report['has_nan_gradients'] = True
                print(f"⚠️ NaN detected in gradient: {name}")
            if torch.isinf(param.grad).any():
                health_report['has_inf_gradients'] = True
                print(f"⚠️ Inf detected in gradient: {name}")
            
            grad_norm = torch.norm(param.grad).item()
            health_report['max_grad_norm'] = max(health_report['max_grad_norm'], grad_norm)
    
    return health_report


class StableTrainingWrapper:
    """稳定训练包装器"""
    
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.nan_count = 0
        self.max_nan_tolerance = 5
        
    def train_step(self, batch_data, loss_fn):
        """执行一个稳定的训练步骤"""
        
        self.optimizer.zero_grad()
        
        try:
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_data)
                    loss = loss_fn(outputs, batch_data['labels'])
            else:
                outputs = self.model(**batch_data)
                loss = loss_fn(outputs, batch_data['labels'])
            
            # 检查损失
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                print(f"⚠️ NaN/Inf loss detected (count: {self.nan_count})")
                
                if self.nan_count > self.max_nan_tolerance:
                    print("❌ Too many NaN losses, stopping training")
                    raise ValueError("Training unstable")
                
                # 跳过这个批次
                return None
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            
            # 重置NaN计数器
            self.nan_count = 0
            
            return loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ CUDA OOM, clearing cache")
                torch.cuda.empty_cache()
                return None
            else:
                raise e