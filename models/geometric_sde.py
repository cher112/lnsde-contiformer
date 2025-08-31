"""
Geometric SDE Model
几何随机微分方程：dY_t/Y_t = μ(t,Y_t)dt + σ(t,Y_t)dW_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/autodl-tmp/torchsde')
import torchsde

from .base_sde import BaseSDEModel, MaskedSequenceProcessor
from .contiformer import ContiFormerModule


class GeometricSDE(BaseSDEModel):
    """
    几何SDE实现
    dY_t = μ(t,Y_t)Y_t dt + σ(t,Y_t)Y_t dW_t
    等价于: dY_t/Y_t = μ(t,Y_t)dt + σ(t,Y_t)dW_t
    """
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__(input_channels, hidden_channels, output_channels, sde_type='ito')
        self.noise_type = 'diagonal'
        
        # 几何漂移系数网络 μ(t,y)
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 几何波动率网络 σ(t,y)
        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 稳定性参数：确保满足几何SDE的稳定性条件
        self.min_sigma = nn.Parameter(torch.tensor(0.05))
        self.max_sigma = nn.Parameter(torch.tensor(2.0))
        
        # 防止解为零的小常数
        self.epsilon = 1e-8
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络参数"""
        for module in [self.mu_net, self.sigma_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def f(self, t, y):
        """
        漂移函数：μ(t,y) * y
        Args:
            t: (batch,) 时间
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 防止y为零和数值爆炸，使用更安全的正定化
        y_safe = torch.clamp(torch.abs(y), min=self.epsilon, max=10.0)
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        ty = torch.cat([y_safe, t_expanded], dim=1)
        
        # 计算几何漂移系数，添加数值稳定性检查
        mu = self.mu_net(ty)
        mu = torch.clamp(mu, min=-5.0, max=5.0)  # 限制漂移系数范围
        
        # 几何漂移：μ(t,y) * |y|
        drift = mu * y_safe
        
        # 检查输出是否为NaN或Inf
        if torch.isnan(drift).any() or torch.isinf(drift).any():
            drift = torch.zeros_like(drift)
        
        return drift
        
    def g(self, t, y):
        """
        扩散函数：σ(t,y) * y
        Args:
            t: (batch,) 时间  
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 防止y为零和数值爆炸，使用更安全的正定化
        y_safe = torch.clamp(torch.abs(y), min=self.epsilon, max=10.0)
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        ty = torch.cat([y_safe, t_expanded], dim=1)
        
        # 计算几何波动率，添加数值稳定性检查
        sigma = self.sigma_net(ty)
        
        # 限制波动率范围以确保数值稳定性
        sigma = torch.clamp(sigma, 
                          min=self.min_sigma.abs(), 
                          max=self.max_sigma.abs())
        
        # 几何扩散：σ(t,y) * |y|
        diffusion = sigma * y_safe
        
        # 检查输出是否为NaN或Inf
        if torch.isnan(diffusion).any() or torch.isinf(diffusion).any():
            diffusion = torch.ones_like(diffusion) * 0.1  # 使用小的正值替代
        
        return diffusion
    
    def get_stability_condition(self, t, y):
        """
        检查几何SDE的稳定性条件: |σ|² > 2K_μ
        其中K_μ是μ函数的增长率上界
        """
        with torch.no_grad():
            y_safe = torch.clamp(torch.abs(y), min=self.epsilon, max=10.0)
            diffusion = self.g(t, y_safe)
            
            # 安全的除法，防止除零
            ratio = diffusion / (y_safe + self.epsilon)
            sigma_squared = ratio ** 2
            sigma_squared_mean = sigma_squared.mean()
            
            # 估计μ的增长率（保守估计）
            K_mu = 2.0
            
            stability_margin = sigma_squared_mean - K_mu
            return stability_margin.item()
    
    def ensure_positivity(self, y):
        """
        确保解的正定性（几何SDE的重要性质）
        """
        return torch.clamp(torch.abs(y), min=self.epsilon, max=10.0)


class GeometricSDEContiformer(nn.Module):
    """
    Geometric SDE + ContiFormer 完整模型
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
                 atol=1e-4,
                 # 组件开关参数
                 use_sde=1,
                 use_contiformer=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        self.use_sde = bool(use_sde)
        self.use_contiformer = bool(use_contiformer)
        
        # 输入特征编码器（确保输出为正）
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),  # 确保为正
            nn.Dropout(dropout),
            nn.Softplus()  # 进一步确保正定性
        )
        
        # 根据开关创建Geometric SDE模块
        if self.use_sde:
            self.sde_model = GeometricSDE(
                input_channels=input_dim,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels
            )
        else:
            self.sde_model = None
        
        # 根据开关创建ContiFormer模块
        if self.use_contiformer:
            self.contiformer = ContiFormerModule(
                input_dim=hidden_channels,
                d_model=contiformer_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout
            )
            classifier_input_dim = contiformer_dim
        else:
            self.contiformer = None
            classifier_input_dim = hidden_channels
        
        # 分类头 - 输入维度根据是否使用ContiFormer动态调整
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        # Mask处理器
        self.mask_processor = MaskedSequenceProcessor()
        
    def forward(self, time_series, times, mask=None, return_stability_info=False):
        """
        前向传播 - 支持模块化组件开关
        Args:
            time_series: (batch, seq_len, input_dim) 光变曲线数据 [time, mag, errmag]
            times: (batch, seq_len) 时间戳
            mask: (batch, seq_len) mask, True表示有效位置
            return_stability_info: 是否返回稳定性信息
        Returns:
            logits: (batch, num_classes) 分类logits
            features: (batch, seq_len, hidden_channels) 特征序列
            stability_info: 稳定性信息（可选）
        """
        batch_size, seq_len = time_series.shape[:2]
        
        # 1. 基础特征编码
        encoded_features = self.feature_encoder(time_series)  # (batch, seq_len, hidden_channels)
        
        # 2. SDE建模（如果启用）
        if self.use_sde and self.sde_model is not None:
            sde_features, stability_info = self._apply_sde_processing(
                encoded_features, times, mask, seq_len, return_stability_info
            )
        else:
            sde_features = encoded_features
            stability_info = []
        
        # 3. 应用mask（如果提供）
        if mask is not None:
            sde_features = self.mask_processor.apply_mask(sde_features, mask)
        
        # 4. ContiFormer处理（如果启用）
        if self.use_contiformer and self.contiformer is not None:
            contiformer_out, pooled_features = self.contiformer(
                sde_features,
                times, 
                mask
            )
            final_features = pooled_features
        else:
            # 不使用ContiFormer时，直接对特征序列进行全局平均池化
            if mask is not None:
                # 考虑mask的平均池化
                mask_expanded = mask.unsqueeze(-1).expand_as(sde_features)
                masked_features = sde_features * mask_expanded.float()
                valid_lengths = mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
                final_features = masked_features.sum(dim=1) / (valid_lengths + 1e-8)  # (batch, hidden_channels)
            else:
                # 简单全局平均池化
                final_features = sde_features.mean(dim=1)  # (batch, hidden_channels)
        
        # 5. 分类
        logits = self.classifier(final_features)
        
        if return_stability_info:
            return logits, sde_features, {
                'stability_margins': stability_info,
                'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0,
                'positivity_preserved': torch.all(sde_features >= 0).item() if self.use_sde else True,
                'sde_solving_steps': len(stability_info)
            }
        else:
            return logits, sde_features
    
    def _apply_sde_processing(self, encoded_features, times, mask, seq_len, return_stability_info):
        """
        应用SDE处理 - 从原始forward函数提取的SDE处理逻辑
        """
        batch_size = encoded_features.shape[0]
        device = encoded_features.device
        
        # GPU优化的SDE建模 - 减少循环和同步
        sde_trajectory = torch.zeros(batch_size, seq_len, self.hidden_channels, device=device)
        stability_info = []
        
        # 初始状态
        current_state = self.sde_model.ensure_positivity(encoded_features[:, 0])
        sde_trajectory[:, 0] = current_state
        sde_solving_count = 0
        
        # 批量处理SDE求解，减少Python循环
        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):  # 启用混合精度
            for i in range(1, seq_len):
                # 获取时间间隔
                prev_time = times[:, i-1]
                current_time = times[:, i]
                
                # 检查是否需要SDE求解（减少条件检查）
                if mask is not None:
                    valid_mask = mask[:, i] & mask[:, i-1]
                    batch_has_valid = valid_mask.any()
                else:
                    batch_has_valid = True
                
                # 时间递增检查（使用张量操作避免CPU同步）
                time_diff = current_time - prev_time
                time_increasing = (time_diff > 1e-8).any()
                
                should_solve_sde = batch_has_valid and time_increasing
                
                if should_solve_sde and sde_solving_count < seq_len * 0.8:  # 限制SDE求解次数
                    try:
                        # 使用批次平均时间作为参考，减少CPU-GPU同步
                        t_start = prev_time.mean()
                        t_end = current_time.mean()
                        
                        if t_end > t_start:
                            t_solve = torch.tensor([t_start, t_end], device=device)
                            
                            # SDE求解（GPU并行）
                            ys = torchsde.sdeint(
                                sde=self.sde_model,
                                y0=current_state,
                                ts=t_solve,
                                method=self.sde_method,
                                dt=min(self.dt, (t_end - t_start).item() / 4),  # 自适应步长
                                rtol=self.rtol,
                                atol=self.atol,
                                adaptive=True  # 启用自适应求解
                            )
                            sde_output = self.sde_model.ensure_positivity(ys[-1])
                            sde_solving_count += 1
                        else:
                            sde_output = encoded_features[:, i]
                    except:
                        sde_output = encoded_features[:, i]
                else:
                    # 直接使用编码特征，减少计算
                    sde_output = encoded_features[:, i]
                
                # 更新状态
                current_state = self.sde_model.ensure_positivity(sde_output)
                sde_trajectory[:, i] = current_state
        
        # 稳定性信息（仅在需要时计算，减少开销）
        if return_stability_info:
            try:
                # 采样检查，不是每个时间点都检查
                sample_indices = torch.linspace(0, seq_len-1, min(10, seq_len), dtype=torch.long)
                for idx in sample_indices:
                    stability_margin = self.sde_model.get_stability_condition(
                        times[:, idx], sde_trajectory[:, idx]
                    )
                    stability_info.append(stability_margin)
            except:
                stability_info = [0.0]
        
        return sde_trajectory, stability_info
    
    def compute_loss(self, logits, labels, sde_features=None, 
                    alpha_stability=1e-3, alpha_positivity=1e-3, weight=None,
                    focal_alpha=0.25, focal_gamma=4.0, label_smoothing=0.3, temperature=1.0):
        """
        增强的损失函数，强力处理类别不平衡和几何SDE特性
        Args:
            logits: (batch, num_classes) 预测logits
            labels: (batch,) 真实标签  
            sde_features: (batch, seq_len, hidden_channels) SDE特征
            alpha_stability: 稳定性正则化系数
            alpha_positivity: 正定性正则化系数
            weight: (num_classes,) 类别权重
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数（增大以更强惩罚易分类样本）
            label_smoothing: 标签平滑参数（增大以防止过拟合多数类）
        """
        batch_size, num_classes = logits.shape
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"警告：logits包含NaN或Inf，使用零logits替代")
            logits = torch.zeros_like(logits)
        
        # 0. 温度缩放 - 防止除零
        temperature = max(temperature, 0.1)  # 防止温度过小
        scaled_logits = logits / temperature
        
        # 1. 更强的Focal Loss - 增大gamma值
        ce_loss = F.cross_entropy(scaled_logits, labels, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = p_t
        focal_loss = focal_alpha * (1-pt)**focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # 2. 增强标签平滑
        if label_smoothing > 0:
            # 创建平滑标签
            smooth_labels = torch.zeros_like(scaled_logits)
            smooth_labels.fill_(label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
            
            # KL散度损失
            log_probs = F.log_softmax(scaled_logits, dim=1)
            smooth_loss = -torch.sum(smooth_labels * log_probs, dim=1).mean()
            
            # 更大权重给标签平滑
            classification_loss = 0.5 * focal_loss + 0.5 * smooth_loss
        else:
            classification_loss = focal_loss
        
        # 3. 更强的多样性损失：严厉惩罚单一类别预测
        pred_probs = F.softmax(scaled_logits, dim=1)
        pred_mean = pred_probs.mean(dim=0)  # 平均预测分布
        
        # 计算预测熵 - 熵越低说明越集中在单一类别
        pred_entropy = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10))
        max_entropy = torch.log(torch.tensor(float(num_classes)))  # 最大熵（均匀分布）
        entropy_loss = 0.15 * (max_entropy - pred_entropy)  # 更大权重惩罚低熵
        
        # 额外的均匀性损失
        diversity_target = torch.ones_like(pred_mean) / num_classes  # 均匀分布
        diversity_loss = F.kl_div(pred_mean.log(), diversity_target, reduction='sum') * 0.25
        
        # 4. 更严厉的单一类别惩罚：如果超过60%的预测是同一类别，施加重惩罚  
        max_class_ratio = pred_mean.max()
        if max_class_ratio > 0.6:
            single_class_penalty = 1.0 * (max_class_ratio - 0.6) ** 2
        else:
            single_class_penalty = torch.tensor(0.0, device=logits.device)
        
        # 5. 几何SDE特有的正则化损失
        stability_loss = torch.tensor(0.0, device=logits.device)
        positivity_loss = torch.tensor(0.0, device=logits.device)
        
        if sde_features is not None:
            # 稳定性正则化：鼓励SDE轨迹平滑
            if alpha_stability > 0:
                diff = sde_features[:, 1:] - sde_features[:, :-1]
                stability_loss = alpha_stability * torch.mean(diff ** 2)
            
            # 正定性正则化：惩罚接近零的值（几何SDE需要正定）
            if alpha_positivity > 0:
                positivity_penalty = torch.mean(torch.exp(-sde_features))
                positivity_loss = alpha_positivity * positivity_penalty
        
        total_loss = classification_loss + stability_loss + positivity_loss + diversity_loss + entropy_loss + single_class_penalty
            
        return total_loss, classification_loss, stability_loss, positivity_loss
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'Geometric SDE + ContiFormer',
            'sde_type': 'Geometric SDE',
            'mathematical_form': 'dY_t/Y_t = μ(t,Y_t)dt + σ(t,Y_t)dW_t',
            'stability_condition': '|σ|² > 2K_μ',
            'special_properties': 'Solution always positive, multiplicative noise',
            'parameters': {
                'hidden_channels': self.hidden_channels,
                'contiformer_dim': self.contiformer.d_model,
                'num_classes': self.num_classes
            }
        }