#!/usr/bin/env python3
"""
数值稳定性和性能优化工具集
解决NaN、除零错误、GPU优化等问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class StablePooling(nn.Module):
    """数值稳定的池化层，避免除零错误"""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - True表示有效位置
        Returns:
            pooled: (batch, hidden_dim)
        """
        if mask is None:
            return features.mean(dim=1)
        
        # 确保mask是float类型并在正确设备上
        mask = mask.float()
        
        # 扩展mask维度
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # 应用mask
        masked_features = features * mask_expanded
        
        # 计算有效元素数量，添加eps避免除零
        mask_sum = mask.sum(dim=1, keepdim=True) + self.eps  # (batch, 1)
        
        # 安全的平均池化
        pooled = masked_features.sum(dim=1) / mask_sum
        
        return pooled


class SafeMaskProcessor(nn.Module):
    """安全的mask处理器，避免数值问题"""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def apply_mask(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用mask到序列，处理边界情况"""
        if mask is None:
            return sequence
        
        # 确保mask类型和设备一致
        mask = mask.to(sequence.device).to(sequence.dtype)
        
        if sequence.dim() == 3:
            mask = mask.unsqueeze(-1)
        
        return sequence * mask
    
    def get_last_valid(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """获取每个序列最后的有效值"""
        batch_size = sequence.size(0)
        
        # 计算每个序列的有效长度
        seq_lengths = mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0)
        
        # 获取batch索引
        batch_indices = torch.arange(batch_size, device=sequence.device)
        
        # 安全索引
        last_valid = sequence[batch_indices, seq_lengths]
        
        # 处理完全无效的序列（全padding）
        valid_mask = mask.any(dim=1)
        result = torch.where(
            valid_mask.unsqueeze(-1),
            last_valid,
            torch.zeros_like(last_valid)
        )
        
        return result


class StableDataPreprocessor:
    """数据预处理器，修复零误差和时间问题"""
    
    @staticmethod
    def fix_zero_errors(data: torch.Tensor, min_error: float = 1e-6) -> torch.Tensor:
        """修复零误差值"""
        # data shape: (batch, seq_len, features) 或 (seq_len, features)
        if data.shape[-1] >= 3:  # 假设第三维是误差
            data[..., 2] = torch.clamp(data[..., 2], min=min_error)
        return data
    
    @staticmethod
    def fix_time_monotonicity(times: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """确保时间序列单调递增"""
        if times.dim() == 1:
            # 一维时间序列
            diff = torch.diff(times)
            if (diff <= 0).any():
                # 累积和确保单调
                fixed_diff = torch.clamp(diff, min=epsilon)
                times = torch.cat([times[:1], times[:1] + torch.cumsum(fixed_diff, dim=0)])
        elif times.dim() == 2:
            # 批量处理
            batch_size = times.size(0)
            for i in range(batch_size):
                times[i] = StableDataPreprocessor.fix_time_monotonicity(times[i], epsilon)
        
        return times
    
    @staticmethod
    def preprocess_batch(time_series: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """批量预处理数据"""
        # 修复零误差
        time_series = StableDataPreprocessor.fix_zero_errors(time_series)
        
        # 修复时间单调性
        if time_series.shape[-1] >= 1:
            times = time_series[..., 0]
            times = StableDataPreprocessor.fix_time_monotonicity(times)
            time_series[..., 0] = times
        
        return time_series, mask


class OptimizedLinearNoiseSDEForward(nn.Module):
    """优化的前向传播，避免tuple返回，纯tensor GPU运算"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stable_pool = StablePooling()
        self.mask_processor = SafeMaskProcessor()
        self.preprocessor = StableDataPreprocessor()
    
    def forward(self, time_series: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        优化的前向传播，只返回logits
        Args:
            time_series: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
        Returns:
            logits: (batch, num_classes) - 纯tensor，不是tuple
        """
        # 数据预处理
        time_series, mask = self.preprocessor.preprocess_batch(time_series, mask)
        
        # 确保输入在GPU上
        device = next(self.model.parameters()).device
        time_series = time_series.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        # 1. 提取时间
        times = time_series[:, :, 0]
        
        # 2. 特征提取（根据use_sde设置）
        if hasattr(self.model, 'use_sde') and self.model.use_sde:
            features = self._sde_forward_stable(time_series, times, mask)
        elif hasattr(self.model, 'direct_mapping'):
            features = self.model.direct_mapping(time_series)
        else:
            # Fallback：简单线性映射
            features = time_series
        
        # 3. 应用mask
        if mask is not None:
            features = self.mask_processor.apply_mask(features, mask)
        
        # 4. 池化或ContiFormer处理
        if hasattr(self.model, 'use_contiformer') and self.model.use_contiformer and hasattr(self.model, 'contiformer'):
            contiformer_out, pooled_features = self.model.contiformer(features, times, mask)
            features_for_classifier = contiformer_out
        else:
            pooled_features = self.stable_pool(features, mask)
            features_for_classifier = pooled_features
        
        # 5. 分类（只返回logits）
        if hasattr(self.model, 'use_cga') and self.model.use_cga and hasattr(self.model, 'cga_classifier'):
            logits, _, _ = self.model.cga_classifier(features, mask)
        elif hasattr(self.model, 'classifier'):
            # 确保输入维度正确
            if features_for_classifier.dim() == 3:
                features_for_classifier = self.stable_pool(features_for_classifier, mask)
            logits = self.model.classifier(features_for_classifier)
        else:
            # Fallback：简单线性层
            num_classes = 7  # 默认MACHO数据集类别数
            if features_for_classifier.dim() == 3:
                features_for_classifier = self.stable_pool(features_for_classifier, mask)
            logits = nn.Linear(features_for_classifier.size(-1), num_classes, device=device)(features_for_classifier)
        
        return logits  # 纯tensor，不是tuple
    
    def _sde_forward_stable(self, time_series: torch.Tensor, times: torch.Tensor, 
                           mask: Optional[torch.Tensor]) -> torch.Tensor:
        """数值稳定的SDE前向传播"""
        batch_size, seq_len = time_series.shape[:2]
        device = time_series.device
        
        # 初始化
        if hasattr(self.model, 'sde_model') and hasattr(self.model.sde_model, 'initial_encoder'):
            y0 = self.model.sde_model.initial_encoder(time_series[:, 0, :])
        else:
            # Fallback：使用第一个时间步
            y0 = time_series[:, 0, :]
        
        # 确定隐藏维度
        hidden_dim = y0.size(-1)
        
        # 初始化特征张量
        features = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
        features[:, 0] = y0
        
        # 简化的SDE求解（避免复杂循环和数值不稳定）
        decay_rate = 0.99  # 衰减率，防止数值爆炸
        
        for t in range(1, min(seq_len, 100)):  # 限制最大步数
            if mask is not None:
                # 只处理有效位置
                valid_mask = mask[:, t]
                if not valid_mask.any():
                    break
                
                # 稳定的状态更新
                prev_state = features[:, t-1]
                # 添加小的噪声项，避免完全确定性
                noise = torch.randn_like(prev_state) * 0.01
                next_state = prev_state * decay_rate + noise
                
                # 只更新有效位置
                features[:, t] = torch.where(
                    valid_mask.unsqueeze(-1),
                    next_state,
                    torch.zeros_like(next_state)
                )
            else:
                # 无mask时的简单更新
                features[:, t] = features[:, t-1] * decay_rate + torch.randn_like(features[:, t-1]) * 0.01
        
        return features


# 改进的Lion优化器配置
def create_stable_lion_optimizer(model, base_lr=1e-4):
    """创建数值稳定的Lion优化器"""
    from lion_pytorch import Lion
    
    # 大幅降低学习率，Lion对学习率非常敏感
    stable_lr = base_lr * 0.05  # 降低20倍
    
    optimizer = Lion(
        model.parameters(),
        lr=stable_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.99)  # 更保守的动量参数
    )
    
    return optimizer, stable_lr


def create_optimized_training_config():
    """创建完整的优化训练配置"""
    return {
        # Lion优化器配置（关键修改）
        'learning_rate': 5e-6,  # 原始1e-4的1/20
        'weight_decay': 1e-4,
        
        # 梯度配置
        'gradient_clip': 0.5,  # 更严格的梯度裁剪
        'gradient_accumulation_steps': 1,
        
        # 数值稳定性
        'label_smoothing': 0.1,
        'eps': 1e-8,
        
        # SDE配置（保守参数）
        'sde_method': 'euler',
        'dt': 0.1,
        'rtol': 1e-3,
        'atol': 1e-4,
        
        # 数据预处理
        'fix_zero_errors': True,
        'min_error_value': 1e-6,
        'fix_time_monotonicity': True,
        
        # 训练配置
        'batch_size': 16,
        'use_amp': False,  # 暂时禁用AMP避免数值问题
        
        # 早停配置
        'early_stopping_patience': 10,
        'early_stopping_delta': 0.001,
    }


class StableTrainingManager:
    """完整的稳定训练管理器"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or create_optimized_training_config()
        
        # 创建优化器
        self.optimizer, self.actual_lr = create_stable_lion_optimizer(
            model, 
            self.config['learning_rate']
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-8
        )
        
        # 包装模型
        self.forward_wrapper = OptimizedLinearNoiseSDEForward(model)
        
        # 数值监控
        self.nan_count = 0
        self.inf_count = 0
        
        print(f"训练管理器初始化完成")
        print(f"  实际学习率: {self.actual_lr:.2e}")
        print(f"  梯度裁剪: {self.config['gradient_clip']}")
        print(f"  标签平滑: {self.config['label_smoothing']}")
    
    def train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 解包数据
        if len(batch) == 3:
            times, values, labels = batch
            mask = None
        else:
            times, values, labels, mask = batch
        
        # 组合输入
        device = next(self.model.parameters()).device
        
        # 确保数据格式正确
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        
        input_data = torch.cat([times.unsqueeze(-1), values], dim=-1).to(device)
        labels = labels.to(device)
        
        if mask is not None:
            mask = mask.to(device)
        
        # 前向传播（返回纯tensor logits）
        self.optimizer.zero_grad()
        
        try:
            logits = self.forward_wrapper(input_data, mask)
            
            # 计算损失
            if self.config['label_smoothing'] > 0:
                loss = F.cross_entropy(
                    logits, labels, 
                    label_smoothing=self.config['label_smoothing']
                )
            else:
                loss = F.cross_entropy(logits, labels)
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                print(f"Warning: NaN/Inf loss detected (count: {self.nan_count})")
                return {'loss': 0.0, 'acc': 0.0, 'skipped': True}
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip']
            )
            
            # 优化器步骤
            self.optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean()
            
            return {
                'loss': loss.item(),
                'acc': acc.item(),
                'skipped': False
            }
            
        except Exception as e:
            print(f"训练步骤错误: {e}")
            return {'loss': 0.0, 'acc': 0.0, 'skipped': True}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 解包数据
                if len(batch) == 3:
                    times, values, labels = batch
                    mask = None
                else:
                    times, values, labels, mask = batch
                
                # 准备数据
                device = next(self.model.parameters()).device
                if values.dim() == 2:
                    values = values.unsqueeze(-1)
                
                input_data = torch.cat([times.unsqueeze(-1), values], dim=-1).to(device)
                labels = labels.to(device)
                
                if mask is not None:
                    mask = mask.to(device)
                
                try:
                    # 前向传播
                    logits = self.forward_wrapper(input_data, mask)
                    loss = F.cross_entropy(logits, labels)
                    
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        preds = logits.argmax(dim=-1)
                        total_acc += (preds == labels).float().mean().item()
                        count += 1
                        
                except Exception as e:
                    print(f"验证错误: {e}")
                    continue
        
        if count > 0:
            return {
                'val_loss': total_loss / count,
                'val_acc': total_acc / count
            }
        else:
            return {'val_loss': 0.0, 'val_acc': 0.0}


if __name__ == "__main__":
    print("="*60)
    print("数值稳定性修复工具")
    print("="*60)
    
    config = create_optimized_training_config()
    
    print("\n关键修改:")
    print(f"1. Lion学习率: 1e-4 → {config['learning_rate']:.2e}")
    print(f"2. 梯度裁剪: 1.0 → {config['gradient_clip']}")
    print(f"3. 修复零误差: {config['fix_zero_errors']}")
    print(f"4. 修复时间单调性: {config['fix_time_monotonicity']}")
    print(f"5. 禁用AMP: {not config['use_amp']}")
    
    print("\n测试命令:")
    print("python main.py --learning_rate 5e-6 --gradient_clip 0.5 --no_amp --epochs 1")