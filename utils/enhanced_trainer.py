"""
增强的训练工具，集成数值稳定性和GPU优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time


class EnhancedTrainer:
    """
    增强的训练器，包含所有优化
    """
    def __init__(self, model, optimizer, criterion, device='cuda', config=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        
        # 混合精度训练
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度裁剪
        self.gradient_clip_val = self.config.get('gradient_clip_val', 1.0)
        
        # 梯度累积
        self.accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # 统计
        self.train_stats = {
            'loss': [],
            'nan_count': 0,
            'inf_count': 0,
            'grad_norm': []
        }
        
    def train_epoch(self, train_loader, epoch=0):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 解包数据 - 处理字典格式的batch
            if isinstance(batch, dict):
                # 字典格式（从dataloader返回）
                features = batch.get('features')
                times = batch.get('times') 
                mask = batch.get('mask')
                labels = batch.get('labels')
                
                # 如果features不存在但x存在，使用x构建features
                if features is None and 'x' in batch:
                    x = batch['x'].to(self.device)
                    batch_size, seq_len = x.shape[0], x.shape[1]
                    # 添加errmag维度（全零）
                    errmag = torch.zeros(batch_size, seq_len, 1, device=self.device)
                    features = torch.cat([x, errmag], dim=-1)  # (batch, seq_len, 3)
                    times = batch.get('time_steps')
                    mask = batch.get('attn_mask')
                    
                # 移动到设备
                if features is not None:
                    features = features.to(self.device, non_blocking=True)
                if times is not None:
                    times = times.to(self.device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)
                    # 确保mask是bool类型
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                if labels is not None:
                    labels = labels.to(self.device, non_blocking=True)
                    
            elif len(batch) == 4:
                # 元组格式
                features, times, mask, labels = batch
                features, times, mask, labels = (
                    features.to(self.device, non_blocking=True),
                    times.to(self.device, non_blocking=True),
                    mask.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True)
                )
                # 确保mask是bool类型
                if mask.dtype != torch.bool:
                    mask = mask.bool()
            elif len(batch) == 3:
                features, mask, labels = batch
                features, mask, labels = (
                    features.to(self.device, non_blocking=True),
                    mask.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True)
                )
                # 确保mask是bool类型
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                times = None
            else:
                features, labels = batch
                features, labels = (
                    features.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True)
                )
                times = None
                mask = None
            
            # 混合精度前向传播
            if self.use_amp:
                with autocast():
                    loss, acc = self._forward_step(features, times, mask, labels, batch_idx)
            else:
                loss, acc = self._forward_step(features, times, mask, labels, batch_idx)
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )
                    
                    # 更新参数
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 更新统计
            total_loss += loss.item()
            correct += acc
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'grad': f'{grad_norm:.3f}' if 'grad_norm' in locals() else 'N/A'
            })
            
            # 记录统计
            self.train_stats['loss'].append(loss.item())
            if 'grad_norm' in locals():
                self.train_stats['grad_norm'].append(grad_norm.item())
        
        return total_loss / len(train_loader), correct / total
    
    def _forward_step(self, features, times, mask, labels, batch_idx):
        """前向传播步骤"""
        # 数据增强（可选）
        if self.model.training and np.random.rand() < 0.1:  # 10%概率
            features = self._augment_data(features)
        
        # 前向传播
        if times is not None:
            logits, sde_features = self.model(features, times, mask)
        else:
            logits, sde_features = self.model(features, mask)
        
        # 检查nan/inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.train_stats['nan_count'] += torch.isnan(logits).sum().item()
            self.train_stats['inf_count'] += torch.isinf(logits).sum().item()
            print(f"警告: Batch {batch_idx} 输出包含 NaN/Inf")
            
            # 清理logits
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            logits = torch.clamp(logits, -10, 10)
        
        # 计算损失
        if hasattr(self.model.model, 'compute_loss'):
            # 使用模型自定义损失
            loss = self.model.model.compute_loss(
                logits, labels, sde_features,
                focal_gamma=2.0,
                temperature=1.0
            )
        else:
            # 标准交叉熵损失
            loss = self.criterion(logits, labels)
        
        # 计算准确率
        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == labels).sum().item()
        
        # 损失缩放（梯度累积）
        loss = loss / self.accumulation_steps
        
        return loss, acc
    
    def _augment_data(self, features):
        """简单的数据增强"""
        # 添加高斯噪声
        noise = torch.randn_like(features) * 0.01
        features = features + noise
        
        # 随机缩放
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
        features = features * scale
        
        return features
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 解包数据 - 处理字典格式的batch
                if isinstance(batch, dict):
                    # 字典格式（从dataloader返回）
                    features = batch.get('features')
                    times = batch.get('times') 
                    mask = batch.get('mask')
                    labels = batch.get('labels')
                    
                    # 如果features不存在但x存在，使用x构建features
                    if features is None and 'x' in batch:
                        x = batch['x'].to(self.device)
                        batch_size, seq_len = x.shape[0], x.shape[1]
                        # 添加errmag维度（全零）
                        errmag = torch.zeros(batch_size, seq_len, 1, device=self.device)
                        features = torch.cat([x, errmag], dim=-1)  # (batch, seq_len, 3)
                        times = batch.get('time_steps')
                        mask = batch.get('attn_mask')
                        
                    # 移动到设备
                    if features is not None:
                        features = features.to(self.device, non_blocking=True)
                    if times is not None:
                        times = times.to(self.device, non_blocking=True)
                    if mask is not None:
                        mask = mask.to(self.device, non_blocking=True)
                        # 确保mask是bool类型
                        if mask.dtype != torch.bool:
                            mask = mask.bool()
                    if labels is not None:
                        labels = labels.to(self.device, non_blocking=True)
                        
                elif len(batch) == 4:
                    # 元组格式
                    features, times, mask, labels = batch
                    features, times, mask, labels = (
                        features.to(self.device, non_blocking=True),
                        times.to(self.device, non_blocking=True),
                        mask.to(self.device, non_blocking=True),
                        labels.to(self.device, non_blocking=True)
                    )
                    # 确保mask是bool类型
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                elif len(batch) == 3:
                    features, mask, labels = batch
                    features, mask, labels = (
                        features.to(self.device, non_blocking=True),
                        mask.to(self.device, non_blocking=True),
                        labels.to(self.device, non_blocking=True)
                    )
                    # 确保mask是bool类型
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    times = None
                else:
                    features, labels = batch
                    features, labels = (
                        features.to(self.device, non_blocking=True),
                        labels.to(self.device, non_blocking=True)
                    )
                    times = None
                    mask = None
                
                # 前向传播
                if times is not None:
                    logits, _ = self.model(features, times, mask)
                else:
                    logits, _ = self.model(features, mask)
                
                # 计算损失
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def get_stats(self):
        """获取训练统计"""
        stats = self.train_stats.copy()
        
        # 添加模型统计
        if hasattr(self.model, 'get_stats'):
            stats.update(self.model.get_stats())
        
        # GPU内存统计
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        
        return stats


def train_with_optimization(model, train_loader, val_loader, epochs=100, device='cuda'):
    """
    使用所有优化进行训练
    """
    from utils.training_optimizer import (
        OptimizedModelWrapper, 
        StableTrainingConfig,
        get_optimized_training_setup
    )
    
    # 获取优化配置
    setup = get_optimized_training_setup(model, train_loader, val_loader, device)
    
    # 创建训练器
    trainer = EnhancedTrainer(
        model=setup['model'],
        optimizer=setup['optimizer'],
        criterion=setup['criterion'],
        device=device,
        config=vars(setup['config'])
    )
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        val_loss, val_acc = trainer.validate(val_loader)
        
        # 学习率调度
        if setup['scheduler'] is not None:
            setup['scheduler'].step()
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': setup['model'].state_dict(),
                'optimizer_state_dict': setup['optimizer'].state_dict(),
                'val_acc': val_acc,
            }, f'best_model_acc_{val_acc:.4f}.pth')
        
        # 打印统计
        stats = trainer.get_stats()
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            print(f"⚠️ NaN count: {stats['nan_count']}, Inf count: {stats['inf_count']}")
        
        # 清理GPU缓存
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    return trainer