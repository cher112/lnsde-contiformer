"""
优化的训练配置，确保数值稳定性和GPU效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedModelWrapper(nn.Module):
    """
    优化的模型包装器
    - 正确处理模型返回的tuple (logits, sde_features)
    - 确保所有张量都在同一设备上
    - 添加数值稳定性检查
    """
    def __init__(self, model, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 数值稳定性参数
        self.eps = 1e-8
        self.max_value = 1e6
        self.min_value = -1e6
        
        # 监控统计
        self.nan_count = 0
        self.inf_count = 0
        
    def forward(self, features, times=None, mask=None):
        """
        前向传播，保持原始接口
        返回: (logits, sde_features) tuple
        """
        # 确保所有输入在正确的设备上
        features = features.to(self.device)
        if times is not None:
            times = times.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            # 确保mask是bool类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 输入稳定性检查
        features = self._stabilize_tensor(features, "input")
        
        # 调用原始模型
        if times is not None:
            outputs = self.model(features, times, mask)
        else:
            outputs = self.model(features, mask)
        
        # 处理输出
        if isinstance(outputs, tuple):
            logits, sde_features = outputs
        else:
            # 如果只返回单个张量，创建dummy sde_features
            logits = outputs
            sde_features = torch.zeros(
                (features.shape[0], features.shape[1], self.model.hidden_channels),
                device=self.device
            )
        
        # 稳定化输出
        logits = self._stabilize_tensor(logits, "logits")
        if sde_features is not None:
            sde_features = self._stabilize_tensor(sde_features, "sde_features")
        
        return logits, sde_features
    
    def _stabilize_tensor(self, tensor, name="tensor"):
        """稳定化张量，移除nan/inf"""
        if tensor is None:
            return None
            
        # 检查nan/inf
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            if has_nan:
                self.nan_count += torch.isnan(tensor).sum().item()
            if has_inf:
                self.inf_count += torch.isinf(tensor).sum().item()
            
            # 替换nan为0
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            # 裁剪inf
            tensor = torch.clamp(tensor, self.min_value, self.max_value)
            
            if self.training:
                print(f"警告: {name} 包含 NaN/Inf，已修复")
        
        return tensor
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count
        }


class StableTrainingConfig:
    """
    稳定训练配置
    """
    def __init__(self):
        # 梯度裁剪
        self.gradient_clip_val = 1.0
        self.gradient_clip_algorithm = 'norm'  # 'norm' or 'value'
        
        # 学习率调度
        self.use_lr_scheduler = True
        self.lr_scheduler_type = 'cosine'  # 'cosine', 'step', 'exponential'
        self.warmup_epochs = 5
        
        # 混合精度训练
        self.use_amp = True  # 自动混合精度
        self.amp_level = 'O1'  # O1: 混合精度, O2: 几乎全FP16
        
        # 数值稳定性
        self.eps = 1e-8
        self.label_smoothing = 0.1
        
        # GPU优化
        self.pin_memory = True
        self.non_blocking = True  # 异步数据传输
        self.num_workers = 4
        self.prefetch_factor = 2
        
        # 检查点
        self.gradient_accumulation_steps = 1
        self.val_check_interval = 0.25  # 每25%的epoch验证一次
        
    def get_optimizer_config(self, model, base_lr=1e-3):
        """获取优化器配置"""
        # 参数分组，不同层使用不同学习率
        param_groups = [
            {'params': model.embedding.parameters(), 'lr': base_lr * 0.1},
            {'params': model.contiformer.parameters(), 'lr': base_lr},
            {'params': model.lnsde.parameters(), 'lr': base_lr * 0.5},
            {'params': model.classifier.parameters(), 'lr': base_lr * 2}
        ]
        
        return param_groups
    
    def apply_gradient_clipping(self, model, optimizer):
        """应用梯度裁剪"""
        if self.gradient_clip_algorithm == 'norm':
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=self.gradient_clip_val
            )
        elif self.gradient_clip_algorithm == 'value':
            torch.nn.utils.clip_grad_value_(
                model.parameters(), 
                clip_value=self.gradient_clip_val
            )
    
    def get_loss_function(self, num_classes):
        """获取损失函数"""
        return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)


def optimize_dataloader(dataloader, device='cuda'):
    """
    优化数据加载器以提高GPU利用率
    """
    # 预取数据到GPU
    class PrefetchLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
            
        def __iter__(self):
            stream = torch.cuda.Stream()
            first = True
            
            for next_data in self.loader:
                with torch.cuda.stream(stream):
                    # 异步传输到GPU
                    if isinstance(next_data, (list, tuple)):
                        next_data = [
                            d.to(self.device, non_blocking=True) 
                            if isinstance(d, torch.Tensor) else d 
                            for d in next_data
                        ]
                    else:
                        next_data = next_data.to(self.device, non_blocking=True)
                
                if not first:
                    yield current_data
                else:
                    first = False
                    
                torch.cuda.current_stream().wait_stream(stream)
                current_data = next_data
                
            yield current_data
            
        def __len__(self):
            return len(self.loader)
    
    if device == 'cuda' and torch.cuda.is_available():
        return PrefetchLoader(dataloader, device)
    return dataloader


def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': reserved - allocated
        }
    return None


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# 使用示例配置
def get_optimized_training_setup(model, train_loader, val_loader, device='cuda'):
    """
    获取优化的训练设置
    """
    # 包装模型
    model = OptimizedModelWrapper(model, device)
    
    # 获取配置
    config = StableTrainingConfig()
    
    # 优化数据加载器
    train_loader = optimize_dataloader(train_loader, device)
    val_loader = optimize_dataloader(val_loader, device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        config.get_optimizer_config(model.model, base_lr=1e-3),
        weight_decay=1e-4,
        eps=config.eps
    )
    
    # 创建学习率调度器
    if config.use_lr_scheduler:
        if config.lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=len(train_loader) * 100,  # 100 epochs
                eta_min=1e-6
            )
        elif config.lr_scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
    else:
        scheduler = None
    
    # 损失函数
    criterion = config.get_loss_function(model.model.num_classes)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'scaler': scaler,
        'config': config,
        'train_loader': train_loader,
        'val_loader': val_loader
    }