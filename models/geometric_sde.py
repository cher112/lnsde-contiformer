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
        
        # 防止y为零，添加小的正值
        y_safe = torch.abs(y) + self.epsilon
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        ty = torch.cat([y_safe, t_expanded], dim=1)
        
        # 计算几何漂移系数
        mu = self.mu_net(ty)
        
        # 几何漂移：μ(t,y) * |y|
        drift = mu * y_safe
        
        return drift
        
    def g(self, t, y):
        """
        扩散函数：σ(t,y) * y
        Args:
            t: (batch,) 时间  
            y: (batch, hidden_channels) 状态
        """
        batch_size = y.shape[0]
        
        # 防止y为零
        y_safe = torch.abs(y) + self.epsilon
        
        # 扩展时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.unsqueeze(0).expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.view(-1, 1).expand(batch_size, 1)
        ty = torch.cat([y_safe, t_expanded], dim=1)
        
        # 计算几何波动率
        sigma = self.sigma_net(ty)
        
        # 限制波动率范围以确保数值稳定性
        sigma = torch.clamp(sigma, 
                          min=self.min_sigma.abs(), 
                          max=self.max_sigma.abs())
        
        # 几何扩散：σ(t,y) * |y|
        diffusion = sigma * y_safe
        
        return diffusion
    
    def get_stability_condition(self, t, y):
        """
        检查几何SDE的稳定性条件: |σ|² > 2K_μ
        其中K_μ是μ函数的增长率上界
        """
        with torch.no_grad():
            y_safe = torch.abs(y) + self.epsilon
            diffusion = self.g(t, y_safe)
            sigma_squared = (diffusion / y_safe) ** 2  # σ²
            sigma_squared_mean = sigma_squared.mean()
            
            # 估计μ的增长率（保守估计）
            K_mu = 2.0
            
            stability_margin = sigma_squared_mean - K_mu
            return stability_margin.item()
    
    def ensure_positivity(self, y):
        """
        确保解的正定性（几何SDE的重要性质）
        """
        return torch.abs(y) + self.epsilon


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
                 atol=1e-4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        
        # 输入特征编码器（确保输出为正）
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),  # 确保为正
            nn.Dropout(dropout),
            nn.Softplus()  # 进一步确保正定性
        )
        
        # Geometric SDE模块
        self.sde_model = GeometricSDE(
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
        
    def forward(self, time_series, times, mask=None, return_stability_info=False):
        """
        前向传播 - 高效的批处理SDE求解版本
        Args:
            time_series: (batch, seq_len, input_dim) 光变曲线数据 [time, mag, errmag]
            times: (batch, seq_len) 时间戳
            mask: (batch, seq_len) mask, True表示有效位置
            return_stability_info: 是否返回稳定性信息
        Returns:
            logits: (batch, num_classes) 分类logits
            sde_features: (batch, seq_len, hidden_channels) SDE特征
            stability_info: 稳定性信息（可选）
        """
        batch_size, seq_len = time_series.shape[:2]
        
        # 1. 特征编码（确保正定性）
        encoded_features = self.feature_encoder(time_series)
        encoded_features = self.sde_model.ensure_positivity(encoded_features)
        
        # 2. 高效的批处理SDE建模 - 类似Linear Noise SDE的方式
        sde_trajectory = []
        stability_info = []
        
        # 从第一个时间点开始，逐步求解SDE
        current_state = self.sde_model.ensure_positivity(encoded_features[:, 0])  # (batch, hidden_channels) 初始状态
        sde_solving_count = 0  # 统计实际进行SDE求解的步数
        
        for i in range(seq_len):
            # 当前时间点
            current_time = times[:, i]  # (batch,)
            
            # 检查稳定性条件
            if return_stability_info:
                try:
                    stability_margin = self.sde_model.get_stability_condition(current_time, current_state)
                    stability_info.append(stability_margin)
                except:
                    stability_info.append(0.0)
            
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
                        # 确保求解结果的正定性
                        sde_output = self.sde_model.ensure_positivity(sde_output)
                        sde_solving_count += 1
                        
                    except Exception as e:
                        if sde_solving_count < 5:  # 只在前几次失败时打印
                            print(f"Geometric SDE求解失败 (步骤 {i}): {e}, 使用编码特征")
                        sde_output = encoded_features[:, i]
                        sde_output = self.sde_model.ensure_positivity(sde_output)
                else:
                    # 使用编码特征
                    sde_output = encoded_features[:, i]
                    sde_output = self.sde_model.ensure_positivity(sde_output)
            
            # 更新当前状态
            current_state = sde_output
            sde_trajectory.append(sde_output)
        
        # 3. 重组SDE轨迹为完整序列
        sde_features = torch.stack(sde_trajectory, dim=1)  # (batch, seq_len, hidden_channels)
        
        # 4. 应用mask到SDE特征
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
        
        if return_stability_info:
            return logits, sde_features, {
                'stability_margins': stability_info,
                'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0,
                'positivity_preserved': torch.all(sde_features > 0).item(),
                'sde_solving_steps': sde_solving_count
            }
        else:
            return logits, sde_features
    
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
        
        # 0. 温度缩放
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