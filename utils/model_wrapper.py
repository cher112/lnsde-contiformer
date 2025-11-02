"""
模型包装器，用于处理模型输出和数值稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableModelWrapper(nn.Module):
    """
    稳定的模型包装器，处理nan/inf问题
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nan_count = 0
        self.inf_count = 0
        
    def forward(self, x):
        """前向传播，处理tuple输出和nan/inf"""
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: 输入包含NaN/Inf，进行清理")
            x = self._clean_tensor(x)
        
        # 前向传播
        output = self.model(x)
        
        # 处理tuple输出
        if isinstance(output, tuple):
            output = output[0]  # 取第一个元素作为logits
        
        # 检查输出
        if torch.isnan(output).any() or torch.isinf(output).any():
            self.nan_count += torch.isnan(output).sum().item()
            self.inf_count += torch.isinf(output).sum().item()
            print(f"警告: 输出包含NaN({self.nan_count})/Inf({self.inf_count})，进行清理")
            output = self._clean_tensor(output)
        
        return output
    
    def _clean_tensor(self, tensor):
        """清理tensor中的nan/inf"""
        # 替换nan为0
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        # 替换inf为大数值
        tensor = torch.where(torch.isinf(tensor), 
                            torch.full_like(tensor, 1e6) * torch.sign(tensor), 
                            tensor)
        # 裁剪到合理范围
        tensor = torch.clamp(tensor, -1e6, 1e6)
        return tensor


class NumericallyStableTrainer:
    """
    数值稳定的训练器
    """
    def __init__(self, model, optimizer, device='cuda'):
        self.model = StableModelWrapper(model)
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_val = 1.0
        
    def train_step(self, batch_data, batch_labels):
        """单个训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # 数据预处理，确保数值稳定
        batch_data = self._preprocess_data(batch_data)
        
        # 获取输出
        outputs = self.model(batch_data)
        
        # 计算损失
        loss = self._compute_stable_loss(outputs, batch_labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        # 检查梯度
        self._check_gradients()
        
        # 更新参数
        self.optimizer.step()
        
        return loss.item(), outputs
    
    def _preprocess_data(self, data):
        """预处理数据，确保数值稳定"""
        # 标准化
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        data = (data - mean) / (std + 1e-8)
        
        # 裁剪异常值
        data = torch.clamp(data, -10, 10)
        
        return data
    
    def _compute_stable_loss(self, outputs, labels):
        """计算数值稳定的损失"""
        # 使用label smoothing
        num_classes = outputs.size(-1)
        smooth_labels = torch.zeros_like(outputs)
        smooth_labels.scatter_(1, labels.unsqueeze(1), 0.9)
        smooth_labels += 0.1 / num_classes
        
        # 计算交叉熵
        log_probs = F.log_softmax(outputs, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def _check_gradients(self):
        """检查梯度是否有nan/inf"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"警告: {name} 梯度包含NaN，置零")
                    param.grad.zero_()
                elif torch.isinf(param.grad).any():
                    print(f"警告: {name} 梯度包含Inf，裁剪")
                    param.grad = torch.clamp(param.grad, -1e3, 1e3)


def create_stable_model(model_class, *args, **kwargs):
    """
    创建数值稳定的模型
    """
    # 创建原始模型
    model = model_class(*args, **kwargs)
    
    # 初始化参数，避免过大或过小的值
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Xavier初始化
                nn.init.xavier_normal_(param, gain=0.5)
            else:
                nn.init.normal_(param, std=0.01)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    # 包装模型
    stable_model = StableModelWrapper(model)
    
    return stable_model