"""
Langevin-type SDE Model
朗之万型随机微分方程：dY_t = -∇U(Y_t)dt + g(t,Y_t)dW_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/autodl-tmp/torchsde')
import torchsde

from .base_sde import BaseSDEModel, MaskedSequenceProcessor
from .contiformer import ContiFormerModule


class LangevinSDE(BaseSDEModel):
    """
    朗之万型SDE实现
    dY_t = -∇U(Y_t)dt + σ(t,Y_t)dW_t
    其中U(Y_t)是势能函数
    """
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__(input_channels, hidden_channels, output_channels, sde_type='ito')
        self.noise_type = 'diagonal'
        
        # 势能函数网络 U(y)
        self.potential_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)  # 输出标量势能
        )
        
        # 扩散系数网络 σ(t,y)
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()  # 确保扩散系数为正
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络参数"""
        for module in [self.potential_net, self.diffusion_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def f(self, t, y):
        """
        漂移函数：-∇U(y)
        Args:
            t: (batch,) 时间
            y: (batch, hidden_channels) 状态
        """
        y = y.requires_grad_(True)
        
        # 计算势能 U(y)
        potential = self.potential_net(y).sum()  # 对批次求和以便计算梯度
        
        # 计算势能梯度 ∇U(y)
        grad = torch.autograd.grad(
            outputs=potential, 
            inputs=y,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 返回 -∇U(y)（负梯度，即势能下降方向）
        return -grad
        
    def g(self, t, y):
        """
        扩散函数：σ(t,y)
        Args:
            t: (batch,) 时间
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 扩展时间维度并与状态连接
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        ty = torch.cat([y, t_expanded], dim=1)
        
        # 计算对角扩散矩阵
        diffusion = self.diffusion_net(ty)
        
        return diffusion


class LangevinSDEContiformer(nn.Module):
    """
    Langevin SDE + ContiFormer 完整模型
    用于光变曲线分类任务
    """
    def __init__(self, 
                 # 数据参数
                 input_dim=3,  # time, mag, errmag
                 num_classes=5,  # 分类数量
                 # SDE参数
                 hidden_channels=64,
                 # ContiFormer参数
                 contiformer_dim=128,
                 n_heads=8,
                 n_layers=4,
                 # 训练参数
                 dropout=0.1,
                 # SDE求解参数
                 sde_method='euler',
                 dt=0.01,
                 rtol=1e-3,
                 atol=1e-4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        
        # 输入特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Langevin SDE模块
        self.sde_model = LangevinSDE(
            input_channels=input_dim,
            hidden_channels=hidden_channels, 
            output_channels=hidden_channels
        )
        
        # ContiFormer模块
        self.contiformer = ContiFormerModule(
            input_dim=hidden_channels,
            d_model=contiformer_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(contiformer_dim, contiformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(contiformer_dim // 2, num_classes)
        )
        
        # Mask处理器
        self.mask_processor = MaskedSequenceProcessor()
        
    def forward(self, time_series, times, mask=None):
        """
        前向传播 - 高效的批处理SDE求解版本
        Args:
            time_series: (batch, seq_len, input_dim) 光变曲线数据 [time, mag, errmag]
            times: (batch, seq_len) 时间戳
            mask: (batch, seq_len) mask, True表示有效位置
        Returns:
            logits: (batch, num_classes) 分类logits
            sde_features: (batch, seq_len, hidden_channels) SDE特征
        """
        batch_size, seq_len = time_series.shape[:2]
        
        # 1. 使用原始特征编码作为SDE的基础状态
        encoded_features = self.feature_encoder(time_series)  # (batch, seq_len, hidden_channels)
        
        # 2. 高效的批处理SDE建模 - 类似Linear Noise SDE的方式
        sde_trajectory = []
        
        # 从第一个时间点开始，逐步求解SDE
        current_state = encoded_features[:, 0]  # (batch, hidden_channels) 初始状态
        sde_solving_count = 0  # 统计实际进行SDE求解的步数
        
        for i in range(seq_len):
            # 当前时间点
            current_time = times[:, i]  # (batch,)
            
            if i == 0:
                # 第一个时间点直接使用初始状态
                sde_output = current_state
            else:
                # 从上一个状态求解到当前时间
                prev_time = times[:, i-1]
                
                # 使用mask来判断是否应该进行SDE求解
                if mask is not None:
                    # 检查当前批次中是否有有效的时间点
                    current_valid = mask[:, i]  # (batch,) 当前时间点的mask
                    prev_valid = mask[:, i-1]    # (batch,) 前一时间点的mask
                    batch_has_valid = torch.any(current_valid & prev_valid)  # 是否有样本在两个时间点都有效
                else:
                    batch_has_valid = True
                
                # 检查时间是否递增（使用第一个样本作为参考）
                prev_time_val = prev_time[0].item()
                current_time_val = current_time[0].item()
                time_increasing = current_time_val > prev_time_val + 1e-8
                
                # 决定是否进行SDE求解
                should_solve_sde = (
                    batch_has_valid and 
                    time_increasing and
                    prev_time_val > 0 and 
                    current_time_val > 0
                )
                
                if should_solve_sde:
                    t_solve = torch.stack([prev_time[0], current_time[0]])  # 使用第一个样本的时间作为参考
                    
                    try:
                        # 求解SDE - 批处理版本
                        ys = torchsde.sdeint(
                            sde=self.sde_model,
                            y0=current_state,  # (batch, hidden_channels)
                            ts=t_solve,
                            method=self.sde_method,
                            dt=self.dt,
                            rtol=self.rtol,
                            atol=self.atol
                        )
                        sde_output = ys[-1]  # (batch, hidden_channels) 取最终时间的结果
                        sde_solving_count += 1
                        
                    except Exception as e:
                        if sde_solving_count < 5:  # 只在前几次失败时打印
                            print(f"Langevin SDE求解失败 (步骤 {i}): {e}, 使用编码特征")
                        sde_output = encoded_features[:, i]
                else:
                    # 使用编码特征
                    sde_output = encoded_features[:, i]
            
            # 更新当前状态
            current_state = sde_output
            sde_trajectory.append(sde_output)
        
        # 3. 重组SDE轨迹为完整序列
        sde_features = torch.stack(sde_trajectory, dim=1)  # (batch, seq_len, hidden_channels)
        
        # 4. 应用mask
        if mask is not None:
            sde_features = self.mask_processor.apply_mask(sde_features, mask)
        
        # 5. ContiFormer处理整个SDE轨迹序列
        contiformer_out, pooled_features = self.contiformer(
            sde_features,  # 传入完整的SDE轨迹
            times, 
            mask
        )
        
        # 6. 分类
        logits = self.classifier(pooled_features)
        
        return logits, sde_features
    
    def compute_loss(self, logits, labels, sde_features=None, alpha_stability=1e-4, 
                     weight=None, focal_alpha=0.25, focal_gamma=4.0, label_smoothing=0.3, temperature=0.5):
        """
        超激进反崩塌损失函数
        """
        batch_size, num_classes = logits.shape
        
        # 0. 温度缩放和动态权重调整
        scaled_logits = logits / temperature
        pred_probs = F.softmax(scaled_logits, dim=1)
        pred_counts = pred_probs.sum(dim=0)
        dynamic_weights = 1.0 / (pred_counts + 1e-6)
        dynamic_weights = dynamic_weights / dynamic_weights.sum() * num_classes
        
        if weight is not None:
            combined_weights = 0.3 * weight.to(logits.device) + 0.7 * dynamic_weights
        else:
            combined_weights = dynamic_weights
        
        # 确保权重不需要梯度，避免梯度计算问题
        combined_weights = combined_weights.detach()
        
        # 1. 更强的Focal Loss
        ce_loss = F.cross_entropy(scaled_logits, labels, weight=combined_weights, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = p_t
        focal_loss = focal_alpha * (1-pt)**focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # 2. 增强标签平滑
        if label_smoothing > 0:
            # 创建平滑标签
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
            
            # KL散度损失
            log_probs = F.log_softmax(scaled_logits, dim=1)
            smooth_loss = -torch.sum(smooth_labels * log_probs, dim=1).mean()
            
            # 更大权重给标签平滑
            classification_loss = 0.5 * focal_loss + 0.5 * smooth_loss
        else:
            classification_loss = focal_loss
        
        # 3. 超强多样性损失
        pred_mean = pred_probs.mean(dim=0)
        
        uniform_target = torch.ones_like(pred_mean) / num_classes
        uniformity_loss = F.mse_loss(pred_mean, uniform_target) * 2.0
        
        # 计算预测熵 - 熵越低说明越集中在单一类别
        pred_entropy = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10))
        max_entropy = torch.log(torch.tensor(float(num_classes)))  # 最大熵（均匀分布）
        entropy_loss = 0.5 * (max_entropy - pred_entropy)
        
        diversity_target = torch.ones_like(pred_mean) / num_classes
        diversity_loss = F.kl_div(pred_mean.log(), diversity_target, reduction='sum') * 0.5
        
        # 4. 超严厉惩罚：40%阈值，立方惩罚
        max_class_ratio = pred_mean.max()
        if max_class_ratio > 0.4:
            single_class_penalty = 5.0 * (max_class_ratio - 0.4) ** 3
        else:
            single_class_penalty = torch.tensor(0.0, device=logits.device)
            
        pred_var = torch.var(pred_mean)
        ideal_var = (1.0/num_classes) * (1.0 - 1.0/num_classes) / num_classes
        variance_penalty = 2.0 * torch.abs(pred_var - ideal_var)
        
        # 5. 稳定性正则化损失
        stability_loss = torch.tensor(0.0, device=logits.device)
        if sde_features is not None and alpha_stability > 0:
            # L2正则化：鼓励特征平滑
            diff = sde_features[:, 1:] - sde_features[:, :-1]
            stability_loss = alpha_stability * torch.mean(diff ** 2)
        
        total_loss = (classification_loss + stability_loss + diversity_loss + entropy_loss + 
                     single_class_penalty + uniformity_loss + variance_penalty)
        
        return total_loss
    
    def get_stability_loss(self, sde_features, alpha=1e-4):
        """
        添加SDE稳定性正则化损失
        Args:
            sde_features: (batch, seq_len, hidden_channels)
            alpha: 正则化系数
        """
        # L2正则化：鼓励特征平滑
        diff = sde_features[:, 1:] - sde_features[:, :-1]
        stability_loss = alpha * torch.mean(diff ** 2)
        return stability_loss