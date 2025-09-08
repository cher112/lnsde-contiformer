"""
模型包装器修复版，处理mask类型问题
"""

import torch
import torch.nn as nn


class FixedModelWrapper(nn.Module):
    """
    修复后的模型包装器，确保mask类型正确
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
        前向传播，确保mask类型正确
        """
        # 确保所有输入在正确的设备上
        features = features.to(self.device)
        if times is not None:
            times = times.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            # 强制转换mask为bool类型
            if mask.dtype != torch.bool:
                mask = mask > 0.5  # 转换为bool
        
        # 输入稳定性检查
        features = self._stabilize_tensor(features, "input")
        
        # Hook来修复模型内部的mask使用
        def fix_mask_hook(module, args):
            """修复forward函数的参数"""
            if len(args) >= 3 and args[2] is not None:
                # args[0] = time_series, args[1] = times, args[2] = mask
                mask = args[2]
                if mask.dtype != torch.bool:
                    mask = mask > 0.5
                args = (args[0], args[1], mask) + args[3:]
            return args
        
        # 临时添加hook
        handle = self.model._forward_hook_fn = self.model.register_forward_pre_hook(fix_mask_hook)
        
        try:
            # 调用原始模型
            if times is not None:
                outputs = self.model(features, times, mask)
            else:
                outputs = self.model(features, mask)
        finally:
            # 移除hook
            handle.remove()
        
        # 处理输出
        if isinstance(outputs, tuple):
            logits, sde_features = outputs
        else:
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


# 猴子补丁：修复模型的_forward_with_sde方法
def patch_model_mask_handling(model):
    """
    通过猴子补丁修复模型中的mask处理
    """
    original_forward_with_sde = model._forward_with_sde
    
    def fixed_forward_with_sde(time_series, times, mask, return_stability_info):
        # 确保mask是bool类型
        if mask is not None and mask.dtype != torch.bool:
            mask = mask > 0.5
        return original_forward_with_sde(time_series, times, mask, return_stability_info)
    
    model._forward_with_sde = fixed_forward_with_sde
    
    # 同样修复_forward_without_mask_control如果存在
    if hasattr(model, '_forward_without_mask_control'):
        original_forward_without_mask = model._forward_without_mask_control
        
        def fixed_forward_without_mask(time_series, times, return_stability_info):
            return original_forward_without_mask(time_series, times, return_stability_info)
        
        model._forward_without_mask_control = fixed_forward_without_mask
    
    return model