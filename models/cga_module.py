"""
类别感知分组注意力(Category-aware Grouped Attention, CGA)模块
用于处理类别不平衡问题，与Contiformer串联使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CategoryAwareGroupedAttention(nn.Module):
    """
    类别感知分组注意力模块
    
    主要功能：
    1. 为每个类别构建独立的表示通道
    2. 基于语义相似度的类间信息交换
    3. 门控机制控制信息流动
    """
    
    def __init__(self, 
                 input_dim=128,           # Contiformer输出维度
                 num_classes=3,           # 类别数量
                 group_dim=64,            # 每个类别组的表示维度
                 n_heads=4,               # 注意力头数
                 dropout=0.1,             # Dropout率
                 temperature=0.1,         # 相似度温度参数
                 gate_threshold=0.5):     # 门控阈值
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.group_dim = group_dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.gate_threshold = gate_threshold
        
        # 确保group_dim能被n_heads整除
        assert group_dim % n_heads == 0, f"group_dim ({group_dim}) must be divisible by n_heads ({n_heads})"
        self.head_dim = group_dim // n_heads
        
        # 输入投影层 - 将输入映射到类别特定空间
        self.input_projection = nn.Linear(input_dim, group_dim * num_classes)
        
        # 类别特定的QKV投影 - 为每个类别创建独立的注意力参数
        self.class_q_projs = nn.ModuleList([
            nn.Linear(group_dim, group_dim) for _ in range(num_classes)
        ])
        self.class_k_projs = nn.ModuleList([
            nn.Linear(group_dim, group_dim) for _ in range(num_classes)
        ])
        self.class_v_projs = nn.ModuleList([
            nn.Linear(group_dim, group_dim) for _ in range(num_classes)
        ])
        
        # 类别特定的输出投影
        self.class_out_projs = nn.ModuleList([
            nn.Linear(group_dim, group_dim) for _ in range(num_classes)
        ])
        
        # 语义相似度计算网络
        self.semantic_network = nn.Sequential(
            nn.Linear(group_dim * 2, group_dim),
            nn.ReLU(),
            nn.Linear(group_dim, 1),
            nn.Sigmoid()
        )
        
        # 门控网络 - 控制类间信息交换
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_dim * 2, group_dim),
                nn.ReLU(),
                nn.Linear(group_dim, group_dim),
                nn.Sigmoid()
            ) for _ in range(num_classes * (num_classes - 1))
        ])
        
        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(group_dim * num_classes, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 残差连接的缩放因子
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(group_dim) for _ in range(num_classes)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_dim) - Contiformer的输出
            mask: (batch, seq_len) - 可选的mask
        
        Returns:
            output: (batch, seq_len, input_dim) - CGA处理后的特征
            class_representations: (batch, num_classes, group_dim) - 类别特定表示
        """
        batch_size, seq_len, _ = x.shape
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 输入投影到类别特定空间
        # (batch, seq_len, group_dim * num_classes)
        projected = self.input_projection(x)
        
        # 分割为每个类别的表示
        # List of (batch, seq_len, group_dim)
        class_features = torch.chunk(projected, self.num_classes, dim=-1)
        
        # 存储每个类别的注意力输出
        class_outputs = []
        
        # 对每个类别执行独立的注意力计算
        for c in range(self.num_classes):
            # 获取当前类别的特征
            feat_c = class_features[c]  # (batch, seq_len, group_dim)
            
            # 计算QKV
            Q = self.class_q_projs[c](feat_c)  # (batch, seq_len, group_dim)
            K = self.class_k_projs[c](feat_c)
            V = self.class_v_projs[c](feat_c)
            
            # 多头注意力计算
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # 计算注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用mask
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
                scores = scores.masked_fill(~mask_expanded, float('-inf'))
            
            # 计算注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力
            attn_output = torch.matmul(attn_weights, V)
            
            # 重塑并投影
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.group_dim
            )
            attn_output = self.class_out_projs[c](attn_output)
            
            # 添加残差连接和层归一化
            attn_output = self.layer_norms[c](attn_output + feat_c)
            
            class_outputs.append(attn_output)
        
        # 计算类别表示的平均池化（用于语义相似度计算）
        # (batch, num_classes, group_dim)
        class_representations = []
        for c in range(self.num_classes):
            if mask is not None:
                # 考虑mask的平均池化
                masked_output = class_outputs[c] * mask.unsqueeze(-1)
                sum_output = masked_output.sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                avg_output = sum_output / count
            else:
                avg_output = class_outputs[c].mean(dim=1)
            class_representations.append(avg_output)
        
        class_representations = torch.stack(class_representations, dim=1)
        
        # 类间信息交换（基于语义相似度）
        enhanced_outputs = []
        gate_idx = 0
        
        for i in range(self.num_classes):
            enhanced = class_outputs[i].clone()
            
            for j in range(self.num_classes):
                if i != j:
                    # 计算语义相似度
                    sim_input = torch.cat([
                        class_representations[:, i:i+1, :].expand(-1, seq_len, -1),
                        class_representations[:, j:j+1, :].expand(-1, seq_len, -1)
                    ], dim=-1)
                    
                    similarity = self.semantic_network(sim_input).squeeze(-1)  # (batch, seq_len)
                    
                    # 应用温度缩放
                    similarity = similarity / self.temperature
                    
                    # 门控机制
                    gate_input = torch.cat([class_outputs[i], class_outputs[j]], dim=-1)
                    gate = self.gate_networks[gate_idx](gate_input)
                    gate_idx += 1
                    
                    # 只有相似度高于阈值时才交换信息
                    exchange_mask = (similarity > self.gate_threshold).unsqueeze(-1)
                    gated_info = gate * class_outputs[j] * exchange_mask
                    
                    # 累加交换的信息
                    enhanced = enhanced + gated_info * similarity.unsqueeze(-1)
            
            enhanced_outputs.append(enhanced)
        
        # 融合所有类别的表示
        # (batch, seq_len, group_dim * num_classes)
        fused = torch.cat(enhanced_outputs, dim=-1)
        
        # 最终投影回原始维度
        output = self.fusion_layer(fused)
        
        # 添加缩放的残差连接
        output = output + residual * self.residual_scale
        
        return output, class_representations
    
    def get_class_attention_weights(self, x, class_idx, mask=None):
        """
        获取特定类别的注意力权重（用于可视化）
        
        Args:
            x: (batch, seq_len, input_dim) - 输入特征
            class_idx: int - 类别索引
            mask: (batch, seq_len) - 可选的mask
        
        Returns:
            attention_weights: (batch, n_heads, seq_len, seq_len) - 注意力权重
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        projected = self.input_projection(x)
        class_features = torch.chunk(projected, self.num_classes, dim=-1)
        
        # 获取指定类别的特征
        feat_c = class_features[class_idx]
        
        # 计算QK
        Q = self.class_q_projs[class_idx](feat_c)
        K = self.class_k_projs[class_idx](feat_c)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights


class CGAClassifier(nn.Module):
    """
    带有CGA模块的分类器
    用于替代原始的简单分类头
    """
    
    def __init__(self,
                 input_dim=128,
                 num_classes=3,
                 cga_config=None,
                 dropout=0.1):
        super().__init__()
        
        # CGA配置
        if cga_config is None:
            cga_config = {
                'group_dim': 64,
                'n_heads': 4,
                'temperature': 0.1,
                'gate_threshold': 0.5
            }
        
        # CGA模块
        self.cga = CategoryAwareGroupedAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
            **cga_config
        )
        
        # 分类头 - 直接从类别表示进行分类
        self.classifier = nn.Sequential(
            nn.Linear(cga_config['group_dim'], cga_config['group_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cga_config['group_dim'] // 2, 1)  # 每个类别一个输出
        )
        
        # 最终分类层
        self.final_classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_dim) - 输入特征（来自Contiformer）
            mask: (batch, seq_len) - 可选的mask
        
        Returns:
            logits: (batch, num_classes) - 分类logits
            cga_features: (batch, seq_len, input_dim) - CGA处理后的特征
            class_representations: (batch, num_classes, group_dim) - 类别表示
        """
        # CGA处理
        cga_features, class_representations = self.cga(x, mask)
        
        # 方法1：使用类别表示进行分类
        class_logits = []
        for c in range(class_representations.shape[1]):
            class_repr = class_representations[:, c, :]  # (batch, group_dim)
            class_logit = self.classifier(class_repr)  # (batch, 1)
            class_logits.append(class_logit)
        
        class_based_logits = torch.cat(class_logits, dim=1)  # (batch, num_classes)
        
        # 方法2：使用增强后的特征进行分类（全局池化）
        if mask is not None:
            # 考虑mask的池化
            masked_features = cga_features * mask.unsqueeze(-1)
            pooled_features = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled_features = cga_features.mean(dim=1)
        
        feature_based_logits = self.final_classifier(pooled_features)
        
        # 结合两种分类结果
        logits = (class_based_logits + feature_based_logits) / 2
        
        return logits, cga_features, class_representations