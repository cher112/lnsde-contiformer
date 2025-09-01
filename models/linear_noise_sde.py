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
from .cga_module import CGAClassifier


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
        
        # B(t): 时间相关的乘性噪声系数
        self.B_net = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh()  # 可以为负，控制增长/衰减
        )
        
        # 稳定性参数：确保满足稳定性条件 |σ|² > 2L_f
        self.min_diffusion = nn.Parameter(torch.tensor(0.1))
        
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
        扩散函数：A(t) + B(t)y （线性噪声）
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
        B_t = self.B_net(t_expanded)  # (batch, hidden_channels)
        
        # 线性扩散: A(t) + B(t) * y
        diffusion = A_t + B_t * y
        
        # 添加最小扩散以确保数值稳定性
        diffusion = diffusion + self.min_diffusion.abs()
        
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
        
        # 消螏实验参数
        self.use_sde = use_sde
        self.use_contiformer = use_contiformer
        
        # 梯度管理参数
        self.enable_gradient_detach = enable_gradient_detach
        self.detach_interval = detach_interval
        
        # Linear Noise SDE模块 - 可选
        if self.use_sde:
            self.sde_model = LinearNoiseSDE(
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
        
        # 稳定性监控
        self.stability_history = []
        
    def forward(self, time_series, mask=None, return_stability_info=False):
        """
        前向传播 - 支持消螏实验的批处理
        Args:
            time_series: (batch, seq_len, input_dim) 光变曲线数据 [time, mag, errmag]
            mask: (batch, seq_len) mask, True表示有效位置
            return_stability_info: 是否返回稳定性信息
        Returns:
            logits: (batch, num_classes) 分类logits
            features: (batch, seq_len, hidden_channels) 特征
            stability_info: 稳定性信息（可选）
        """
        batch_size, seq_len = time_series.shape[:2]
        
        # 1. 提取时间数据
        times = time_series[:, :, 0]  # (batch, seq_len) 时间数据
        
        # 2. 特征提取 - 根据消螏实验设置
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
                # 考虑mask的池化
                masked_features = features * mask.unsqueeze(-1)
                pooled_features = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled_features = features.mean(dim=1)
            final_features = features
        
        # 5. 分类 - CGA或普通分类器
        if self.use_cga and self.cga_classifier is not None:
            # 使用CGA增强的分类器
            logits, cga_features, class_representations = self.cga_classifier(final_features, mask)
        else:
            # 使用普通分类器
            logits = self.classifier(pooled_features)
        
        if return_stability_info:
            return logits, features, stability_info
        else:
            return logits, features
    
    def _forward_with_sde(self, time_series, times, mask, return_stability_info):
        """使用SDE进行特征提取"""
        batch_size, seq_len = time_series.shape[:2]
        
        sde_trajectory = []
        stability_info = []
        
        # 初始状态：将mag/errmag转换为hidden_channels维度
        initial_features = time_series[:, 0]  # (batch, input_dim)
        current_state = torch.zeros(batch_size, self.hidden_channels, device=time_series.device)
        current_state[:, :min(self.input_dim, self.hidden_channels)] = initial_features[:, :min(self.input_dim, self.hidden_channels)]
        sde_solving_count = 0
        
        for i in range(seq_len):
            current_time = times[:, i]
            
            # 检查稳定性条件
            if return_stability_info and hasattr(self.sde_model, 'get_stability_condition'):
                try:
                    stability_margin = self.sde_model.get_stability_condition(current_time, current_state)
                    stability_info.append(stability_margin)
                except:
                    stability_info.append(0.0)
            
            if i == 0:
                sde_output = current_state
            else:
                prev_time = times[:, i-1]
                
                # 检查是否应该进行SDE求解
                if mask is not None:
                    current_valid = mask[:, i]
                    prev_valid = mask[:, i-1]
                    batch_has_valid = torch.any(current_valid & prev_valid)
                else:
                    batch_has_valid = True
                
                prev_time_val = prev_time[0].item()
                current_time_val = current_time[0].item()
                time_increasing = current_time_val > prev_time_val + 1e-6
                
                should_solve_sde = (
                    batch_has_valid and 
                    time_increasing and
                    prev_time_val > 0 and 
                    current_time_val > 0
                )
                
                if should_solve_sde:
                    t_solve = torch.stack([prev_time[0], current_time[0]])
                    
                    try:
                        import torchsde
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
                    except Exception as e:
                        if sde_solving_count < 5:
                            print(f"Linear Noise SDE求解失败 (步骤 {i}): {e}, 使用原始特征")
                        current_features = time_series[:, i]
                        sde_output = torch.zeros(batch_size, self.hidden_channels, device=time_series.device)
                        sde_output[:, :min(self.input_dim, self.hidden_channels)] = current_features[:, :min(self.input_dim, self.hidden_channels)]
                else:
                    current_features = time_series[:, i]
                    sde_output = torch.zeros(batch_size, self.hidden_channels, device=time_series.device)
                    sde_output[:, :min(self.input_dim, self.hidden_channels)] = current_features[:, :min(self.input_dim, self.hidden_channels)]
            
            current_state = sde_output
            sde_trajectory.append(sde_output)
        
        features = torch.stack(sde_trajectory, dim=1)
        stability_info_dict = {
            'stability_margins': stability_info,
            'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0,
            'sde_solving_steps': sde_solving_count
        }
        
        return features, stability_info_dict
        
    def _forward_without_sde(self, time_series):
        """不使用SDE，直接映射特征"""
        # 直接将输入映射到hidden_channels维度
        features = self.direct_mapping(time_series)  # (batch, seq_len, hidden_channels)
        return features
    
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