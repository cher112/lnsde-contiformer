"""
ContiFormer Module Implementation
基于PhysioPro的ContiFormer实现，适配光变曲线数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('/root/autodl-tmp/PhysioPro')

try:
    from physiopro.network.contiformer import ContiFormer as OriginalContiFormer
    HAS_PHYSIOPRO = True
except ImportError:
    HAS_PHYSIOPRO = False


class PositionalEncoding(nn.Module):
    """连续时间位置编码"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        position_vec = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        self.register_buffer('position_vec', position_vec)
        
    def forward(self, time):
        """
        Args:
            time: (batch, seq_len) 时间戳
        Returns:
            (batch, seq_len, d_model) 位置编码
        """
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result = result.repeat(1, 1, self.d_model // len(self.position_vec))
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result


class ContinuousMultiHeadAttention(nn.Module):
    """连续时间多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 时间感知的线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 时间编码投影
        self.time_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_enc, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            time_enc: (batch, seq_len, d_model) 时间编码
            mask: (batch, seq_len) mask
        """
        batch_size, seq_len, d_model = x.size()
        
        # 融合时间信息
        x_with_time = x + self.time_proj(time_enc)
        
        # 投影到Q, K, V
        Q = self.w_q(x_with_time).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x_with_time).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x_with_time).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask - 使用fp16安全的掩码值
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # 使用-65504作为fp16的最小安全值，避免溢出
            mask_value = -65504.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask, mask_value)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 计算输出
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(out), attention


class ContiFormerLayer(nn.Module):
    """ContiFormer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ContinuousMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_enc, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            time_enc: (batch, seq_len, d_model)
            mask: (batch, seq_len)
        """
        # 自注意力
        attn_out, attention = self.self_attn(x, time_enc, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attention


class ContiFormerModule(nn.Module):
    """
    ContiFormer模块，专门用于光变曲线时序建模
    """
    def __init__(self, input_dim, d_model=64, n_heads=8, n_layers=6, d_ff=256, 
                 max_seq_len=512, dropout=0.1, use_physiopro=None):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        
        # 决定是否使用PhysioPro
        self.use_physiopro = use_physiopro if use_physiopro is not None else HAS_PHYSIOPRO
        
        if self.use_physiopro:
            self._init_physiopro_contiformer(d_model, n_heads, n_layers, d_ff, dropout)
        else:
            self._init_custom_contiformer(d_model, n_heads, n_layers, d_ff, dropout)
            
    def _init_physiopro_contiformer(self, d_model, n_heads, n_layers, d_ff, dropout):
        """使用PhysioPro的ContiFormer"""
        try:
            self.contiformer = OriginalContiFormer(
                d_model=d_model,
                n_head=n_heads, 
                n_layers=n_layers,
                d_inner=d_ff,
                dropout=dropout,
                linear_type=None
            )
            self.input_projection = nn.Linear(self.input_dim, d_model)
            print("使用PhysioPro ContiFormer实现")
        except Exception as e:
            print(f"PhysioPro初始化失败: {e}, 转为自定义实现")
            self.use_physiopro = False
            self._init_custom_contiformer(d_model, n_heads, n_layers, d_ff, dropout)
            
    def _init_custom_contiformer(self, d_model, n_heads, n_layers, d_ff, dropout):
        """自定义ContiFormer实现"""
        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, self.max_seq_len)
        
        self.layers = nn.ModuleList([
            ContiFormerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        print("使用自定义ContiFormer实现")
    
    def forward(self, x, times=None, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim) 输入特征
            times: (batch, seq_len) 时间戳，可选
            mask: (batch, seq_len) mask，True表示有效位置
        Returns:
            encoded: (batch, seq_len, d_model) 编码输出
            pooled: (batch, d_model) 池化输出
        """
        batch_size, seq_len = x.shape[:2]
        
        # 如果没有提供时间戳，生成默认时间
        if times is None:
            times = torch.linspace(0, 1, seq_len, device=x.device, dtype=x.dtype)
            times = times.unsqueeze(0).expand(batch_size, -1)
            
        # 投影到模型维度
        x = self.input_projection(x)
        
        if self.use_physiopro:
            return self._forward_physiopro(x, times, mask)
        else:
            return self._forward_custom(x, times, mask)
    
    def _forward_physiopro(self, x, times, mask):
        """使用PhysioPro的前向传播"""
        try:
            encoded, pooled = self.contiformer(x, times, mask)
            return encoded, pooled
        except Exception as e:
            print(f"PhysioPro前向传播失败: {e}")
            # 降级到自定义实现
            return self._forward_custom(x, times, mask)
    
    def _forward_custom(self, x, times, mask):
        """自定义前向传播 - 支持梯度检查点"""
        # 时间位置编码
        time_enc = self.pos_encoding(times)
        x = x + time_enc
        x = self.dropout(x)
        
        # 逐层处理 - 支持梯度检查点
        attentions = []
        for layer in self.layers:
            if hasattr(self, 'use_gradient_checkpoint') and self.use_gradient_checkpoint:
                # 使用梯度检查点节省内存
                x, attention = torch.utils.checkpoint.checkpoint(
                    layer, x, time_enc, mask, use_reentrant=False
                )
            else:
                x, attention = layer(x, time_enc, mask)
            attentions.append(attention)
        
        # 池化：获取最后有效位置的输出
        if mask is not None:
            # 找到每个序列的最后有效位置
            seq_lengths = mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            
            # 检查是否有完全无效的序列（全为padding）
            valid_sequences = mask.sum(dim=1) > 0
            batch_indices = torch.arange(x.size(0), device=x.device)
            
            # 对于有效序列，取最后有效位置；对于无效序列，取均值池化
            pooled = torch.zeros(x.size(0), x.size(2), device=x.device)
            if valid_sequences.any():
                valid_batch_indices = batch_indices[valid_sequences]
                valid_seq_lengths = seq_lengths[valid_sequences]
                pooled[valid_sequences] = x[valid_batch_indices, valid_seq_lengths]
            
            # 对于无效序列，使用均值池化作为fallback
            if not valid_sequences.all():
                invalid_mask = ~valid_sequences
                pooled[invalid_mask] = x[invalid_mask].mean(dim=1)
        else:
            pooled = x[:, -1]  # 使用最后一个时间步
        
        return x, pooled