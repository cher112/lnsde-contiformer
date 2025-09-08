"""
Geometric SDE Model
几何随机微分方程：dX_t = μ(t,X_t)X_t dt + σ(t,X_t)X_t dW_t
特点是漂移和扩散项都乘以当前状态X_t，适合建模比例增长过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/autodl-tmp/torchsde')
import torchsde

from .base_sde import BaseSDEModel, MaskedSequenceProcessor
from .contiformer import ContiFormerModule
from .cga_module import CGAClassifier


class GeometricSDE(BaseSDEModel):
    """
    Geometric SDE实现
    dX_t = μ(t,X_t)X_t dt + σ(t,X_t)X_t dW_t
    其中漂移和扩散项都与状态成比例
    """
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__(input_channels, hidden_channels, output_channels, sde_type='ito')
        self.noise_type = 'diagonal'
        
        # 比例漂移系数网络 μ(t,x)
        self.drift_coeff_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 比例扩散系数网络 σ(t,x)
        self.diffusion_coeff_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(), 
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Softplus()  # 确保为正
        )
        
        # 稳定性参数
        self.min_diffusion = nn.Parameter(torch.tensor(0.05))
        self.max_drift = nn.Parameter(torch.tensor(1.0))
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络参数"""
        for module in [self.drift_coeff_net, self.diffusion_coeff_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def f(self, t, y):
        """
        漂移函数：μ(t,y) * y (比例漂移)
        Args:
            t: (batch,) 时间
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        
        ty = torch.cat([y, t_expanded], dim=1)
        
        # 计算比例漂移系数
        drift_coeff = self.drift_coeff_net(ty)  # (batch, hidden_channels)
        
        # 应用稳定性约束
        drift_coeff = torch.clamp(drift_coeff, -self.max_drift.abs(), self.max_drift.abs())
        
        # Geometric漂移：μ(t,y) * y
        drift = drift_coeff * y
        
        return drift
        
    def g(self, t, y):
        """
        扩散函数：σ(t,y) * y (比例扩散)
        Args:
            t: (batch,) 时间
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        
        ty = torch.cat([y, t_expanded], dim=1)
        
        # 计算比例扩散系数
        diffusion_coeff = self.diffusion_coeff_net(ty)  # (batch, hidden_channels)
        
        # 添加最小扩散以确保数值稳定性
        diffusion_coeff = diffusion_coeff + self.min_diffusion.abs()
        
        # Geometric扩散：σ(t,y) * y，但要防止y=0时的数值问题
        y_safe = torch.clamp(y, min=-10.0, max=10.0)  # 防止状态过大
        diffusion = diffusion_coeff * y_safe
        
        return diffusion


class GeometricSDEContiformer(nn.Module):
    """
    Geometric SDE + ContiFormer 完整模型
    用于光变曲线分类任务 - 基于lnsde-contiformer2架构
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
                 # SDE求解参数 - 优化以减少内存使用
                 sde_method='euler',
                 dt=0.05,           # 增大时间步长，减少求解步数
                 rtol=1e-2,         # 放松相对容差，减少迭代次数
                 atol=1e-3,         # 放松绝对容差，减少计算复杂度
                 # 梯度管理参数
                 enable_gradient_detach=True,
                 detach_interval=10,
                 # 消螏实验参数
                 use_sde=True,
                 use_contiformer=True,
                 # CGA参数
                 use_cga=False,
                 cga_group_dim=64,
                 cga_heads=4,
                 cga_temperature=0.1,
                 cga_gate_threshold=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        self.use_cga = use_cga
        
        # 消融实验参数
        self.use_sde = use_sde
        self.use_contiformer = use_contiformer
        
        # 梯度管理参数
        self.enable_gradient_detach = enable_gradient_detach
        self.detach_interval = detach_interval
        
        # Geometric SDE模块 - 可选
        if self.use_sde:
            self.sde_model = GeometricSDE(
                input_channels=input_dim,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels
            )
        else:
            self.sde_model = None
            # 不使用SDE时，直接使用线性映射
            self.direct_mapping = nn.Linear(input_dim, hidden_channels)
        
        # ContiFormer模块 - 可选
        if self.use_contiformer:
            self.contiformer = ContiFormerModule(
                input_dim=hidden_channels,
                d_model=contiformer_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout
            )
        else:
            self.contiformer = None
            # 不使用ContiFormer时，使用简单的全局平均池化
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头或CGA分类器
        # 决定输入维度
        classifier_input_dim = contiformer_dim if self.use_contiformer else hidden_channels
        
        if self.use_cga:
            # 使用CGA增强的分类器
            self.cga_classifier = CGAClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                cga_config={
                    'group_dim': cga_group_dim,
                    'n_heads': cga_heads,
                    'temperature': cga_temperature,
                    'gate_threshold': cga_gate_threshold
                },
                dropout=dropout
            )
            self.classifier = None
        else:
            # 普通分类头
            self.cga_classifier = None
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_input_dim // 2, num_classes)
            )
        
        # Mask处理器
        self.mask_processor = MaskedSequenceProcessor()
        
    def forward(self, time_series, mask=None, return_stability_info=False):
        """
        前向传播 - 支持消融实验的批处理
        Args:
            time_series: (batch, seq_len, input_dim) 光变曲线数据 [time, mag, errmag]
            mask: (batch, seq_len) mask, True表示有效位置
            return_stability_info: 是否返回稳定性信息
        Returns:
            logits: (batch, num_classes) 分类logits
        """
        batch_size, seq_len = time_series.shape[:2]
        
        # 1. 提取时间数据
        times = time_series[:, :, 0]  # (batch, seq_len) 时间数据
        
        # 2. 特征提取 - 根据消融实验设置
        if self.use_sde:
            # 使用SDE进行特征提取
            features, stability_info = self._forward_with_sde(time_series, times, mask, return_stability_info)
        else:
            # 不使用SDE，直接映射
            features = self._forward_without_sde(time_series)
            stability_info = {'sde_solving_steps': 0}
        
        # 3. 应用mask
        if mask is not None:
            features = self.mask_processor.apply_mask(features, mask)
        
        # 4. ContiFormer处理或简单池化
        if self.use_contiformer:
            contiformer_out, pooled_features = self.contiformer(
                features,
                times, 
                mask
            )
            final_features = contiformer_out
        else:
            # 不使用ContiFormer，直接进行全局平均池化
            if mask is not None:
                # 修复除零问题 - 添加eps
                masked_features = features * mask.unsqueeze(-1)
                mask_sum = mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
                eps = 1e-8
                pooled_features = torch.where(
                    mask_sum > eps,
                    masked_features.sum(dim=1) / mask_sum.clamp(min=eps),
                    features.mean(dim=1)  # fallback到普通均值
                )
            else:
                pooled_features = features.mean(dim=1)
            final_features = features
        
        # 5. 分类 - CGA或普通分类器
        if self.use_cga and self.cga_classifier is not None:
            # 使用CGA增强的分类器
            logits, _, _ = self.cga_classifier(final_features, mask)
        else:
            # 使用普通分类器
            logits = self.classifier(pooled_features)
        
        # 只返回logits，不返回tuple
        return logits
    
    def _forward_with_sde(self, time_series, times, mask, return_stability_info):
        """使用Geometric SDE进行特征提取"""
        batch_size, seq_len = time_series.shape[:2]
        
        # mag/errmag直接注入SDE - 不使用feature_encoder
        sde_trajectory = []
        
        # 从第一个时间点开始，逐步求解SDE
        # 初始状态：将mag/errmag转换为hidden_channels维度
        initial_features = time_series[:, 0]  # (batch, input_dim)
        current_state = torch.ones(batch_size, self.hidden_channels, device=time_series.device) * 0.1  # 初始化为小正值，避免Geometric SDE的零状态问题
        # 将mag/errmag映射到SDE隐藏状态空间，但避免零值
        current_state[:, :min(self.input_dim, self.hidden_channels)] = torch.clamp(
            initial_features[:, :min(self.input_dim, self.hidden_channels)], 
            min=0.01, max=10.0  # Geometric SDE需要正值
        )
        sde_solving_count = 0
        
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
                    current_valid = mask[:, i]
                    prev_valid = mask[:, i-1]
                    batch_has_valid = torch.any(current_valid & prev_valid)
                else:
                    batch_has_valid = True
                
                # 检查时间是否递增
                prev_time_val = prev_time[0].item()
                current_time_val = current_time[0].item()
                time_increasing = current_time_val > prev_time_val + 1e-6
                
                # 决定是否进行SDE求解
                should_solve_sde = (
                    batch_has_valid and 
                    time_increasing and
                    prev_time_val > 0 and 
                    current_time_val > 0
                )
                
                if should_solve_sde:
                    t_solve = torch.stack([prev_time[0], current_time[0]])
                    
                    try:
                        # 求解Geometric SDE
                        ys = torchsde.sdeint(
                            sde=self.sde_model,
                            y0=current_state,
                            ts=t_solve,
                            method=self.sde_method,
                            dt=self.dt,
                            rtol=self.rtol,
                            atol=self.atol
                        )
                        sde_output = ys[-1]
                        sde_solving_count += 1
                        
                        # 确保输出数值稳定
                        sde_output = torch.clamp(sde_output, min=0.01, max=10.0)
                        
                    except Exception as e:
                        if sde_solving_count < 5:
                            print(f"Geometric SDE求解失败 (步骤 {i}): {e}, 使用原始特征")
                        # 当SDE求解失败时，直接使用当前时间点的原始特征
                        current_features = time_series[:, i]
                        sde_output = torch.ones(batch_size, self.hidden_channels, device=time_series.device) * 0.1
                        sde_output[:, :min(self.input_dim, self.hidden_channels)] = torch.clamp(
                            current_features[:, :min(self.input_dim, self.hidden_channels)],
                            min=0.01, max=10.0
                        )
                else:
                    # 使用当前时间点的原始特征
                    current_features = time_series[:, i]
                    sde_output = torch.ones(batch_size, self.hidden_channels, device=time_series.device) * 0.1
                    sde_output[:, :min(self.input_dim, self.hidden_channels)] = torch.clamp(
                        current_features[:, :min(self.input_dim, self.hidden_channels)],
                        min=0.01, max=10.0
                    )
            
            # 更新当前状态
            current_state = sde_output
            sde_trajectory.append(sde_output)
        
        # 重组SDE轨迹为完整序列
        sde_features = torch.stack(sde_trajectory, dim=1)  # (batch, seq_len, hidden_channels)
        stability_info = {'sde_solving_steps': sde_solving_count}
        return sde_features, stability_info
    
    def _forward_without_sde(self, time_series):
        """不使用SDE，直接映射特征"""
        # 直接使用线性映射处理所有时间步
        batch_size, seq_len, input_dim = time_series.shape
        # 重塑为 (batch * seq_len, input_dim)，然后映射，再重塑回来
        flat_series = time_series.view(-1, input_dim)
        flat_features = self.direct_mapping(flat_series)
        features = flat_features.view(batch_size, seq_len, self.hidden_channels)
        return features
    
    def compute_loss(self, logits, labels, sde_features=None, alpha_stability=1e-3, 
                     weight=None, focal_alpha=1.0, focal_gamma=2.0, temperature=1.0):
        """
        改进的损失函数 - 支持数据集特定的超参数
        """
        batch_size, num_classes = logits.shape
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # Focal Loss
        ce_loss = F.cross_entropy(scaled_logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = focal_alpha * (1-pt)**focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # 稳定性正则化损失
        stability_loss = torch.tensor(0.0, device=logits.device)
        if sde_features is not None and alpha_stability > 0:
            diff = sde_features[:, 1:] - sde_features[:, :-1]
            stability_loss = alpha_stability * torch.mean(diff ** 2)
        
        # 轻度信息熵正则化
        pred_probs = F.softmax(scaled_logits, dim=1)
        pred_mean = pred_probs.mean(dim=0)
        pred_entropy = -torch.sum(pred_mean * torch.log(pred_mean + 1e-8))
        max_entropy = torch.log(torch.tensor(float(num_classes), device=logits.device))
        entropy_penalty = 0.1 * (max_entropy - pred_entropy)
        
        total_loss = focal_loss + stability_loss + entropy_penalty
        
        return total_loss
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'Geometric SDE + ContiFormer',
            'sde_type': 'Geometric SDE',
            'mathematical_form': 'dX_t = μ(t,X_t)X_t dt + σ(t,X_t)X_t dW_t',
            'description': 'Geometric Brownian motion with neural coefficients',
            'parameters': {
                'hidden_channels': self.hidden_channels,
                'contiformer_dim': self.contiformer.d_model,
                'num_classes': self.num_classes
            }
        }