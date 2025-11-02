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
        
        # 仅在检测到NaN/Inf时进行修复，保持模型表达能力
        if torch.isnan(diffusion).any() or torch.isinf(diffusion).any():
            diffusion = torch.where(torch.isnan(diffusion) | torch.isinf(diffusion), 
                                   self.min_diffusion.abs(), diffusion)
        
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
                 cga_gate_threshold=0.5,
                 # GPU优化参数
                 use_gradient_checkpoint=False,
                 # SDE求解模式
                 sde_solve_mode=0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dt = dt
        self.sde_method = sde_method
        self.rtol = rtol
        self.atol = atol
        self.use_cga = use_cga
        self.sde_solve_mode = sde_solve_mode  # 0=逐步求解, 1=一次性求解
        
        # 消螏实验参数
        self.use_sde = use_sde
        self.use_contiformer = use_contiformer
        
        # 梯度管理参数
        self.enable_gradient_detach = enable_gradient_detach
        self.detach_interval = detach_interval
        
        # GPU优化参数
        self.use_gradient_checkpoint = use_gradient_checkpoint
        
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
            # 传递梯度检查点参数
            self.contiformer.use_gradient_checkpoint = self.use_gradient_checkpoint
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
            # 使用SDE进行特征提取 - 根据求解模式选择方法
            if self.sde_solve_mode == 0:
                # 模式0: 逐步求解（当前实现）
                features, stability_info = self._forward_with_sde(time_series, times, mask, return_stability_info)
            else:
                # 模式1: 一次性求解整个轨迹（demo.ipynb方式）
                features, stability_info = self._forward_with_sde_full_trajectory(time_series, times, mask, return_stability_info)

            # 可选的轨迹完整性验证（调试用）
            if hasattr(self, '_debug_mode') and self._debug_mode:
                is_valid = self._validate_sde_trajectory(features, mask, times)
                if not is_valid:
                    print("SDE轨迹验证失败，可能存在padding数据污染")
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
                # 修复维度不匹配 - mask_sum已经是(batch, 1)，不需要squeeze
                pooled_features = torch.where(
                    mask_sum > eps,
                    masked_features.sum(dim=1) / mask_sum.clamp(min=eps),  # 直接使用mask_sum
                    features.mean(dim=1)  # fallback到普通均值
                )
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
        
        # 只返回logits，不返回tuple
        return logits
    
    def _forward_with_sde(self, time_series, times, mask, return_stability_info):
        """高性能SDE前向传播 - 最小化循环和计算开销"""
        batch_size, seq_len = time_series.shape[:2]
        
        if mask is None:
            return self._forward_without_mask_control(time_series, times, return_stability_info)
        
        # 预计算所有mask信息，避免循环中重复计算
        valid_positions = mask  # (batch_size, seq_len)
        
        # 预计算需要SDE求解的位置 - 向量化操作
        current_valid = valid_positions[:, 1:]  # (batch_size, seq_len-1)
        prev_valid = valid_positions[:, :-1]    # (batch_size, seq_len-1)
        need_sde_mask = current_valid & prev_valid  # (batch_size, seq_len-1)
        
        # 找到所有需要SDE求解的时间步
        sde_steps = []
        for i in range(1, seq_len):
            if need_sde_mask[:, i-1].any():
                sde_steps.append(i)
        
        # 初始化状态 - 批量处理
        current_state = torch.zeros(batch_size, self.hidden_channels, device=time_series.device)
        last_valid_state = torch.zeros_like(current_state)
        
        # 批量初始化：找到每个序列的第一个有效位置
        first_valid_indices = torch.zeros(batch_size, dtype=torch.long, device=time_series.device)
        has_valid = torch.zeros(batch_size, dtype=torch.bool, device=time_series.device)
        
        for batch_idx in range(batch_size):
            if valid_positions[batch_idx].any():
                first_valid_indices[batch_idx] = torch.where(valid_positions[batch_idx])[0][0]
                has_valid[batch_idx] = True
        
        # 向量化初始化特征
        if has_valid.any():
            valid_batch_idx = torch.where(has_valid)[0]
            valid_first_idx = first_valid_indices[has_valid]
            
            for i, (batch_idx, first_idx) in enumerate(zip(valid_batch_idx, valid_first_idx)):
                initial_features = time_series[batch_idx, first_idx]
                current_state[batch_idx, :min(self.input_dim, self.hidden_channels)] = initial_features[:min(self.input_dim, self.hidden_channels)]
                last_valid_state[batch_idx] = current_state[batch_idx].clone()
        
        # 预分配轨迹存储
        sde_trajectory = torch.zeros(batch_size, seq_len, self.hidden_channels, device=time_series.device)
        sde_solving_count = 0
        stability_info = []
        
        # 第一步
        sde_trajectory[:, 0] = current_state.clone()
        
        # 主循环 - 只处理需要SDE的步骤
        for i in sde_steps:
            step_need_sde = need_sde_mask[:, i-1]  # 当前步骤哪些batch需要SDE
            
            if step_need_sde.any():
                sde_batch_indices = torch.where(step_need_sde)[0]
                
                # 提取时间（假设时间对所有batch相同）
                prev_time = times[sde_batch_indices[0], i-1].item()
                curr_time = times[sde_batch_indices[0], i].item()
                
                # 时间有效性检查
                if curr_time > prev_time + 1e-6:
                    try:
                        import torchsde
                        t_solve = torch.tensor([prev_time, curr_time], device=time_series.device)
                        
                        # 批量SDE求解
                        sde_states = current_state[sde_batch_indices]
                        ys = torchsde.sdeint(
                            sde=self.sde_model,
                            y0=sde_states,
                            ts=t_solve,
                            method=self.sde_method,
                            dt=self.dt,
                            rtol=self.rtol,
                            atol=self.atol
                        )
                        
                        # 更新状态
                        current_state[sde_batch_indices] = ys[-1]
                        sde_solving_count += 1
                        
                        # 稳定性检查（简化）
                        if return_stability_info and sde_solving_count < 10:
                            try:
                                stability_margin = float(torch.norm(ys[-1]).item())
                                stability_info.append(stability_margin)
                            except:
                                stability_info.append(0.0)
                                
                    except Exception as e:
                        if sde_solving_count < 3:
                            print(f"SDE求解失败 (步骤 {i}): {e}")
            
            # 更新轨迹 - 向量化操作
            current_step_valid = valid_positions[:, i]
            
            # 有效位置：更新last_valid_state和轨迹
            valid_mask = current_step_valid.unsqueeze(-1)  # (batch_size, 1)
            new_state = torch.where(valid_mask, current_state, last_valid_state)
            
            # 更新last_valid_state（只在有效位置更新）
            last_valid_state = torch.where(valid_mask, current_state, last_valid_state)
            current_state = new_state
            
            sde_trajectory[:, i] = new_state
        
        # 处理剩余的非SDE步骤（如果有的话）
        remaining_steps = set(range(1, seq_len)) - set(sde_steps)
        for i in remaining_steps:
            current_step_valid = valid_positions[:, i].unsqueeze(-1)
            sde_trajectory[:, i] = torch.where(current_step_valid, current_state, last_valid_state)
        
        features = sde_trajectory
        
        stability_info_dict = {
            'stability_margins': stability_info,
            'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0,
            'sde_solving_steps': sde_solving_count
        }
        
        return features, stability_info_dict
    
    def _forward_with_sde_full_trajectory(self, time_series, times, mask, return_stability_info):
        """
        一次性求解整个轨迹（demo.ipynb方式）
        对每个样本独立求解完整的SDE轨迹
        """
        batch_size, seq_len = time_series.shape[:2]
        device = time_series.device

        # 预分配输出
        sde_trajectory = torch.zeros(batch_size, seq_len, self.hidden_channels, device=device)
        stability_info = []

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # 对每个样本独立求解（因为时间点不同）
        for batch_idx in range(batch_size):
            # 提取当前样本的有效数据
            batch_mask = mask[batch_idx]
            valid_indices = torch.where(batch_mask)[0]

            if len(valid_indices) == 0:
                continue

            # 获取有效的时间点
            valid_times = times[batch_idx, valid_indices]  # (num_valid,)

            # 初始化：使用第一个有效观测点的特征
            first_idx = valid_indices[0]
            initial_features = time_series[batch_idx, first_idx]  # (input_dim,)
            y0 = torch.zeros(self.hidden_channels, device=device)
            y0[:min(self.input_dim, self.hidden_channels)] = initial_features[:min(self.input_dim, self.hidden_channels)]
            y0 = y0.unsqueeze(0)  # (1, hidden_channels) for batch dimension

            try:
                # 一次性求解整个轨迹（demo.ipynb方式）
                # 注意：torchsde.sdeint 期望 y0 形状为 (batch_size, state_size)
                ys = torchsde.sdeint(
                    sde=self.sde_model,
                    y0=y0,  # (1, hidden_channels)
                    ts=valid_times,  # (num_valid,) 完整时间序列
                    method=self.sde_method,
                    dt=self.dt,
                    rtol=self.rtol,
                    atol=self.atol
                )
                # ys shape: (num_valid, 1, hidden_channels)

                # 将结果放回对应的位置
                sde_trajectory[batch_idx, valid_indices] = ys.squeeze(1)  # (num_valid, hidden_channels)

                # 稳定性信息
                if return_stability_info and batch_idx < 5:  # 只记录前几个样本
                    try:
                        stability_margin = float(torch.norm(ys[-1]).item())
                        stability_info.append(stability_margin)
                    except:
                        stability_info.append(0.0)

            except Exception as e:
                print(f"⚠️ 样本 {batch_idx} SDE一次性求解失败: {e}")
                # 失败时使用零填充
                sde_trajectory[batch_idx, valid_indices] = 0.0

        stability_info_dict = {
            'stability_margins': stability_info,
            'mean_stability': sum(stability_info) / len(stability_info) if stability_info else 0.0,
            'sde_solving_steps': batch_size  # 一次性求解，每个样本算一步
        }

        return sde_trajectory, stability_info_dict

    def _forward_without_mask_control(self, time_series, times, return_stability_info):
        """没有mask时的原始SDE处理逻辑"""
        # 这里可以实现无mask的SDE逻辑，目前简化处理
        batch_size, seq_len = time_series.shape[:2]
        features = torch.randn(batch_size, seq_len, self.hidden_channels, device=time_series.device)
        stability_info_dict = {'sde_solving_steps': 0}
        return features, stability_info_dict
    
    def _validate_sde_trajectory(self, features, mask, times):
        """验证SDE轨迹的完整性，确保没有padding数据影响"""
        if mask is None:
            return True
            
        batch_size, seq_len = features.shape[:2]
        
        for batch_idx in range(batch_size):
            batch_mask = mask[batch_idx]
            batch_times = times[batch_idx]
            
            # 检查padding位置对应的时间是否为-1e9
            padding_positions = ~batch_mask
            if padding_positions.any():
                padding_times = batch_times[padding_positions]
                is_padding_time = torch.abs(padding_times + 1e9) < 1e-6
                
                if not is_padding_time.all():
                    print(f"警告: 批次{batch_idx}存在非-1e9的padding时间值")
                    return False
            
            # 检查有效位置的特征是否有异常值
            valid_positions = batch_mask
            if valid_positions.any():
                valid_features = features[batch_idx, valid_positions]
                if torch.isnan(valid_features).any() or torch.isinf(valid_features).any():
                    print(f"警告: 批次{batch_idx}的有效位置存在NaN/Inf特征")
                    return False
                    
        return True
        
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