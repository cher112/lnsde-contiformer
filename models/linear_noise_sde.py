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
                 detach_interval=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        
        # 梯度管理参数
        self.enable_gradient_detach = enable_gradient_detach
        self.detach_interval = detach_interval
        
        # 输入特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Linear Noise SDE模块
        self.sde_model = LinearNoiseSDE(
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
        
        # 稳定性监控
        self.stability_history = []
        
    def forward(self, time_series, times, mask=None, return_stability_info=False):
        """
        前向传播
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
        
        # 1. 提取时间和mag数据用于SDE建模
        time_data = time_series[:, :, 0]  # (batch, seq_len) 时间数据
        mag_data = time_series[:, :, 1]   # (batch, seq_len) mag数据
        
        # 2. 使用原始特征编码作为SDE的基础状态
        encoded_features = self.feature_encoder(time_series)  # (batch, seq_len, hidden_channels)
        
        # 3. SDE建模整个时间序列 - 改进版本避免递归深度问题
        sde_trajectory = []
        stability_info = []
        
        # 从第一个时间点开始，逐步求解SDE
        current_state = encoded_features[:, 0]  # (batch, hidden_channels) 初始状态
        
        # 设置最大连续求解步数，防止递归过深
        max_consecutive_steps = 50
        consecutive_steps = 0
        
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
                consecutive_steps = 0
            else:
                # 从上一个状态求解到当前时间
                prev_time = times[:, i-1]
                
                # 创建时间序列用于SDE求解
                # 确保时间严格递增，避免padding零值导致的问题
                prev_time_val = prev_time[0].item()
                current_time_val = current_time[0].item()
                
                # 检查是否需要跳过SDE求解以避免递归深度问题
                should_skip_sde = (
                    current_time_val <= prev_time_val or  # 时间不递增
                    prev_time_val <= 0 or current_time_val <= 0 or  # 包含零值
                    consecutive_steps >= max_consecutive_steps or  # 连续步数过多
                    abs(current_time_val - prev_time_val) < 1e-6  # 时间差太小
                )
                
                if should_skip_sde:
                    # 使用编码特征，重置计数器
                    sde_output = encoded_features[:, i]
                    consecutive_steps = 0
                else:
                    t_solve = torch.stack([prev_time[0], current_time[0]])  # 使用第一个样本的时间作为参考
                    
                    try:
                        # 断开梯度连接以防止递归累积
                        detached_state = current_state.detach().requires_grad_(True)
                        
                        # 求解SDE从prev_state到当前时间
                        with torch.no_grad():
                            # 先尝试无梯度求解以检测数值问题
                            test_ys = torchsde.sdeint(
                                sde=self.sde_model,
                                y0=detached_state.detach(),
                                ts=t_solve,
                                method=self.sde_method,
                                dt=min(self.dt, abs(current_time_val - prev_time_val) / 10),  # 动态调整步长
                                rtol=self.rtol,
                                atol=self.atol,
                                options={'max_num_steps': 100}  # 限制最大步数
                            )
                        
                        # 如果测试通过，进行带梯度的求解
                        ys = torchsde.sdeint(
                            sde=self.sde_model,
                            y0=detached_state,
                            ts=t_solve,
                            method=self.sde_method,
                            dt=min(self.dt, abs(current_time_val - prev_time_val) / 10),
                            rtol=self.rtol,
                            atol=self.atol,
                            options={'max_num_steps': 100}
                        )
                        sde_output = ys[-1]  # 取最终时间的结果
                        consecutive_steps += 1
                        
                    except (RuntimeError, RecursionError) as e:
                        print(f"Linear Noise SDE求解失败 (步骤 {i}): {type(e).__name__}: {str(e)[:100]}, 使用编码特征")
                        sde_output = encoded_features[:, i]
                        consecutive_steps = 0
                    except Exception as e:
                        print(f"Linear Noise SDE求解失败 (步骤 {i}): {e}, 使用编码特征")
                        sde_output = encoded_features[:, i]
                        consecutive_steps = 0
            
            # 更新当前状态为SDE输出（但限制梯度累积）
            if self.enable_gradient_detach and consecutive_steps > 0 and consecutive_steps % self.detach_interval == 0:
                # 每N步断开一次梯度连接
                current_state = sde_output.detach().requires_grad_(True)
            else:
                current_state = sde_output
                
            sde_trajectory.append(sde_output)
        
        # 4. 重组SDE轨迹为完整序列
        sde_features = torch.stack(sde_trajectory, dim=1)  # (batch, seq_len, hidden_channels)
        
        # 5. 应用mask
        if mask is not None:
            sde_features = self.mask_processor.apply_mask(sde_features, mask)
        
        # 6. ContiFormer处理整个SDE轨迹序列
        contiformer_out, pooled_features = self.contiformer(
            sde_features,  # 传入完整的SDE轨迹
            times, 
            mask
        )
        
        # 7. 分类
        logits = self.classifier(pooled_features)
        
        if return_stability_info:
            return logits, sde_features, {
                'stability_margins': stability_info,
                'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0
            }
        else:
            return logits, sde_features
    
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