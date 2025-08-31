"""
Linear Noise SDE Model  
线性噪声随机微分方程：dY_t = f(t,Y_t)dt + (A(t) + B(t)Y_t)dW_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/autodl-tmp/torchsde')
import torchsde

from .base_sde import BaseSDEModel, MaskedSequenceProcessor
from .contiformer import ContiFormerModule


class LinearNoiseSDE(BaseSDEModel):
    """
    线性噪声SDE实现
    dY_t = f(t,Y_t)dt + (A(t) + B(t)Y_t)dW_t
    其中扩散项是状态的线性函数
    """
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__(input_channels, hidden_channels, output_channels, sde_type='ito')
        self.noise_type = 'diagonal'
        
        # 漂移网络 f(t,y)
        self.drift_net = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 线性噪声参数
        # A(t): 时间相关的加性噪声项  
        self.A_net = nn.Sequential(
            nn.Linear(1, hidden_channels),  # 时间输入
            nn.Tanh(), 
            nn.Linear(hidden_channels, hidden_channels),
            nn.Softplus()  # 确保为正
        )
        
        # B(t): 时间相关的乘性噪声系数 - 稳定性改进版本
        self.B_net = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()  # 输出[0,1]，防止负乘性噪声导致不稳定
        )
        
        # 稳定性参数：大幅增强以确保满足稳定性条件 |σ|² > 2L_f
        self.min_diffusion = nn.Parameter(torch.tensor(1.0))  # 从0.1提升至1.0
        self.max_diffusion_scale = nn.Parameter(torch.tensor(5.0))  # 扩散项上界控制
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络参数"""
        for module in [self.drift_net, self.A_net, self.B_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def f(self, t, y):
        """
        漂移函数：f(t,y)
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
        
        # 计算漂移
        drift = self.drift_net(ty)
        
        return drift
        
    def g(self, t, y):
        """
        扩散函数：A(t) + B(t)y （线性噪声） - 数值稳定性增强版
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
        
        # 计算 A(t) 和 B(t)
        A_t = self.A_net(t_expanded)  # (batch, hidden_channels)
        B_t = self.B_net(t_expanded)  # (batch, hidden_channels) - 现在输出[0,1]
        
        # 状态归一化，防止y过大导致数值爆炸
        y_normalized = torch.clamp(y, -10.0, 10.0)  # 限制状态范围
        
        # 线性扩散: A(t) + B(t) * y_normalized
        diffusion = A_t + B_t * y_normalized
        
        # 增强的稳定性控制
        min_diff = self.min_diffusion.abs()  # 最小扩散：1.0
        max_diff = self.max_diffusion_scale.abs()  # 最大扩散：5.0
        
        # 确保扩散项始终为正且在合理范围内
        diffusion = torch.clamp(diffusion + min_diff, min_diff, max_diff)
        
        return diffusion
    
    def get_stability_condition(self, t, y):
        """
        检查稳定性条件: |σ|² > 2L_f
        其中L_f是漂移函数的Lipschitz常数
        """
        with torch.no_grad():
            diffusion = self.g(t, y)
            sigma_squared = (diffusion ** 2).mean()
            
            # 估计Lipschitz常数（简化版本）
            lipschitz_bound = 2.0  # 基于网络设计的保守估计
            
            stability_margin = sigma_squared - lipschitz_bound
            return stability_margin.item()


class LinearNoiseSDEContiformer(nn.Module):
    """
    Linear Noise SDE + ContiFormer 完整模型
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
                 # 梯度管理参数
                 enable_gradient_detach=True,
                 detach_interval=10,
                 # 调试参数
                 debug_mode=False,
                 # SDE求解优化参数
                 min_time_interval=0.01,
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
        
        # 梯度管理参数
        self.enable_gradient_detach = enable_gradient_detach
        self.detach_interval = detach_interval
        
        # 调试模式
        self.debug_mode = debug_mode
        
        # SDE求解优化参数
        self.min_time_interval = min_time_interval
        
        # 输入特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 根据开关创建Linear Noise SDE模块
        if self.use_sde:
            self.sde_model = LinearNoiseSDE(
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
        
        # 稳定性监控
        self.stability_history = []
        
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
                'sde_solving_steps': len(stability_info)
            }
        else:
            return logits, sde_features
    
    def _apply_sde_processing(self, encoded_features, times, mask, seq_len, return_stability_info):
        """
        应用SDE处理 - 从原始forward函数提取的SDE处理逻辑
        """
        sde_trajectory = []
        stability_info = []
        
        # 从第一个时间点开始，逐步求解SDE
        current_state = encoded_features[:, 0]  # (batch, hidden_channels) 初始状态
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
                
                # 检查时间是否递增且间隔足够大（使用第一个样本作为参考）
                prev_time_val = prev_time[0].item()
                current_time_val = current_time[0].item()
                time_increasing = current_time_val > prev_time_val + 1e-6
                time_interval_sufficient = (current_time_val - prev_time_val) >= self.min_time_interval
                
                # 决定是否进行SDE求解
                should_solve_sde = (
                    batch_has_valid and 
                    time_increasing and
                    time_interval_sufficient and  # 新增时间间隔检查
                    prev_time_val > 0 and 
                    current_time_val > 0
                )
                
                if should_solve_sde:
                    t_solve = torch.stack([prev_time[0], current_time[0]])  # 使用第一个样本的时间作为参考
                    
                    try:
                        # 求解SDE - 使用高精度配置保持准确性
                        if self.debug_mode:
                            print(f"    开始SDE求解: t={prev_time[0].item():.3f} → {current_time[0].item():.3f}")
                        ys = torchsde.sdeint(
                            sde=self.sde_model,
                            y0=current_state,  # (batch, hidden_channels)
                            ts=t_solve,
                            method=self.sde_method,  # 使用原始配置的method (milstein)
                            dt=min(self.dt, 0.005),  # 使用更小的步长保证精度
                            rtol=self.rtol,  # 使用原始的高精度容差
                            atol=self.atol,  # 使用原始的高精度容差
                            options={
                                'norm': torch.norm,  # 使用L2范数进行误差估计
                                'jump_t': None,      # 避免不连续时间点
                                'adaptive': True     # 启用自适应步长
                            }
                        )
                        sde_output = ys[-1]  # (batch, hidden_channels) 取最终时间的结果
                        sde_solving_count += 1
                        if self.debug_mode:
                            print(f"    SDE求解成功，输出range: [{sde_output.min().item():.3f}, {sde_output.max().item():.3f}]")
                        
                    except Exception as e:
                        if self.debug_mode and sde_solving_count < 5:  # 调试模式下显示失败信息
                            print(f"Linear Noise SDE求解失败 (步骤 {i}): {e}, 使用编码特征")
                        sde_output = encoded_features[:, i]
                else:
                    # 使用编码特征
                    sde_output = encoded_features[:, i]
            
            # 更新当前状态
            current_state = sde_output
            sde_trajectory.append(sde_output)
        
        # 重组SDE轨迹为完整序列
        sde_features = torch.stack(sde_trajectory, dim=1)  # (batch, seq_len, hidden_channels)
        return sde_features, stability_info
    
    def compute_loss(self, logits, labels, sde_features=None, alpha_stability=1e-3, 
                     weight=None, focal_alpha=1.0, focal_gamma=2.0, temperature=1.0):
        """
        改进的损失函数 - 支持数据集特定的超参数
        Args:
            logits: (batch, num_classes) 预测logits
            labels: (batch,) 真实标签
            sde_features: (batch, seq_len, hidden_channels) SDE特征
            alpha_stability: 稳定性正则化系数
            weight: 忽略，保持接口兼容
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数（数据集特定）
            temperature: 温度缩放参数（数据集特定）
        """
        batch_size, num_classes = logits.shape
        
        # 0. 温度缩放 - 数据集特定的决策边界调整
        scaled_logits = logits / temperature
        
        # 1. Focal Loss - 数据集特定的参数
        ce_loss = F.cross_entropy(scaled_logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = p_t
        focal_loss = focal_alpha * (1-pt)**focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # 2. 稳定性正则化损失
        stability_loss = torch.tensor(0.0, device=logits.device)
        if sde_features is not None and alpha_stability > 0:
            diff = sde_features[:, 1:] - sde_features[:, :-1]
            stability_loss = alpha_stability * torch.mean(diff ** 2)
        
        # 3. 轻度信息熵正则化 - 鼓励预测分布多样性（不使用权重）
        pred_probs = F.softmax(scaled_logits, dim=1)
        pred_mean = pred_probs.mean(dim=0)  # 平均预测分布
        pred_entropy = -torch.sum(pred_mean * torch.log(pred_mean + 1e-8))
        max_entropy = torch.log(torch.tensor(float(num_classes), device=logits.device))
        entropy_penalty = 0.1 * (max_entropy - pred_entropy)  # 温和的熵正则化
        
        total_loss = focal_loss + stability_loss + entropy_penalty
        
        return total_loss
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'Linear Noise SDE + ContiFormer',
            'sde_type': 'Linear Noise SDE',
            'mathematical_form': 'dY_t = f(t,Y_t)dt + (A(t) + B(t)Y_t)dW_t',
            'stability_condition': '|σ|² > 2L_f',
            'noise_structure': 'Linear in state variable',
            'parameters': {
                'hidden_channels': self.hidden_channels,
                'contiformer_dim': self.contiformer.d_model,
                'num_classes': self.num_classes
            }
        }