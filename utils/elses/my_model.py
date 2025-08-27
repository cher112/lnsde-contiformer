import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from embed import TimeEmbedding, DataEmbedding
from tcn import TCN

import warnings

warnings.filterwarnings("ignore")


class Classifier(nn.Module):

    def __init__(self, latent_dim, N=5, args=None):
        super(Classifier, self).__init__()

        self.all_feature_classifier = nn.Sequential(
            nn.Linear(latent_dim, args.nhidden),
            nn.GELU(),
            nn.Linear(args.nhidden, N)
        )

    def forward(self, all_feature):
        output = self.all_feature_classifier(all_feature)
        return output


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=4):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.GELU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, seq_len = x.size()
        y = self.gap(x)  # torch.Size([50, 16, 1])
        y = y.view(b, c)  # torch.Size([50, 16])
        y = self.fc(y)  # torch.Size([50, 16])
        y = y.view(b, c, 1)  # torch.Size([50, 16, 1])
        return x * y.expand_as(x)


class SENet(nn.Module):
    expansion = 1

    def __init__(self, d_model, nhidden, kernel_size, activate='gelu', dropout=0.):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv1d(d_model, nhidden, kernel_size=kernel_size, bias=False, padding='same')
        self.bn1 = nn.BatchNorm1d(nhidden)
        self.conv2 = nn.Conv1d(nhidden, d_model, kernel_size=kernel_size, bias=False, padding='same')
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.SE = SE_Block(d_model)
        self.activate = nn.GELU() if activate == 'gelu' else nn.ReLU()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))  # torch.Size([50, 16, 256])
        x = self.activate(self.bn2(self.conv2(x)))
        SE_out = self.SE(x)
        x = x * SE_out
        x += self.shortcut(x)
        x = self.activate(x)
        x = self.dropout(x)
        return x


class MeasurementEncoderLayer(nn.Module):
    expansion = 1

    def __init__(self, args=None):
        super(MeasurementEncoderLayer, self).__init__()
        self.args = args
        if args.cnn_type == 'senet':
            self.measurement_encoder = SENet(args.d_model, args.nhidden, args.kernel_size, args.activate
                                             , args.dropout)
        elif args.cnn_type == 'tcn':
            self.measurement_encoder = TCN(args.d_model, [args.d_model, args.d_model], args.kernel_size, args.dropout)
        elif args.cnn_type == 'bitcn':
            hidden_size = int(args.d_model / 2)
            self.measurement_encoder = TCN(args.d_model, [hidden_size, hidden_size], args.kernel_size,
                                           args.dropout, bidirectional=True)
        else:
            assert False, 'Invalid cnn type'
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.measurement_encoder(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, args):
        super(ConvModule, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv = nn.Conv1d(in_channels=1, out_channels=args.d_model,
                              kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.bn = nn.BatchNorm1d(args.d_model)
        self.activate = nn.GELU() if args.activate == 'gelu' else nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.activate(self.bn(self.conv(x)))
        return x


class IrregularTimeAttention(nn.Module):
    def __init__(self, eps=None, dropout=0.1):
        super(IrregularTimeAttention, self).__init__()
        # 使用None，在forward中根据数据类型动态设置
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'"""
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)  # torch.Size([50, 1, 256, 256, 16])
        if mask is not None:
            # 根据scores的数据类型选择安全的eps值
            eps_value = self.eps if self.eps is not None else (-65504.0 if scores.dtype == torch.float16 else -1e9)
            scores = scores.masked_fill(mask == 0, eps_value)
        p_attn = F.softmax(scores, dim=-2)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn


class IrregularTimeAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, nhidden=32, num_heads=1):
        super(IrregularTimeAttentionLayer, self).__init__()
        assert d_model % num_heads == 0
        self.embed_time = d_model
        self.embed_time_k = d_model // num_heads
        self.attention = attention
        self.h = num_heads
        self.dim = d_model
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model),
                                      nn.Linear(d_model, d_model),
                                      nn.Linear(d_model * num_heads, d_model)])
        self.layer_norm1 = nn.LayerNorm(d_model)  #
        self.dropout = nn.Dropout(0.2)  #

    def forward(self, query, key, value, mask=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all h heads.
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, attn = self.attention(query, key, value.unsqueeze(1), mask)
        # x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        x = self.linears[-1](x)
        x = self.layer_norm1(value + self.dropout(x))  # Add & Norm
        return x, attn


class IrregularTimeEncoderLayer(nn.Module):

    def __init__(self, attention, args=None, layer_num=0):
        super(IrregularTimeEncoderLayer, self).__init__()
        self.args = args
        self.attention = attention
        self.layer_num = layer_num
        self.t2v = TimeEmbedding(1, args=args)
        self.dropout = nn.Dropout(args.dropout)
        self.device = args.device
        self.activate = nn.GELU if args.activate == "gelu" else nn.ReLU
        self.self_attn = nn.MultiheadAttention(embed_dim=args.d_model, num_heads=args.n_heads, dropout=args.dropout,
                                               batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(args.d_model, args.nhidden),
            self.activate(),
            nn.Linear(args.nhidden, args.d_model))
        self.layer_norm1 = nn.LayerNorm(args.d_model)
        self.layer_norm2 = nn.LayerNorm(args.d_model)
        self.layer_norm3 = nn.LayerNorm(args.d_model)

    def forward(self, x, time_steps, attn_mask=None, time_mask=None):
        key = self.t2v(time_steps, time_mask)  # torch.Size([batch_size,seq_len,d_model])
        query = torch.linspace(0, 1., self.args.seq_len).to(self.device)
        query = query.unsqueeze(-1)
        query = self.t2v(query.unsqueeze(0))  # torch.Size([1, seq_len, d_model])
        if self.layer_num == 0:
            out, attn = self.attention(query, key, x, attn_mask)  # torch.Size([50,256,2])
        else:
            out, attn = self.attention(query, key, x, None)  # torch.Size([50,256,2])
        x = x + self.dropout(out)
        x = self.layer_norm1(x)
        y = x
        y_, _ = self.self_attn(y, y, y)
        y = self.layer_norm2(y + self.dropout(y_))
        y = self.hiddens_to_z0(y)  # torch.Size([batch_size,seq_len,d_model])
        y = (y + y.mean(dim=1, keepdim=True)) * 0.5
        x = x + self.dropout(y)
        x = self.layer_norm3(x)
        return x, time_steps, attn


class CNN2Former(nn.Module):
    def __init__(self, args=None):
        super(CNN2Former, self).__init__()
        self.args = args
        self.cnn2former = nn.MultiheadAttention(embed_dim=args.d_model, num_heads=args.n_heads, dropout=args.dropout,
                                                batch_first=True)

    def forward(self, q, k, v):
        k = k.permute(0, 2, 1)  # [b, L, C]
        v = v.permute(0, 2, 1)  # [b, L, C]

        x, attn = self.cnn2former(q, k, v)  # [b, L, C]
        return x, attn


class Former2CNN(nn.Module):
    def __init__(self, args=None):
        super(Former2CNN, self).__init__()
        self.args = args
        self.former2cnn = nn.MultiheadAttention(embed_dim=args.seq_len, num_heads=args.n_heads, dropout=args.dropout,
                                                batch_first=True)

    def forward(self, q, k, v):
        k = k.permute(0, 2, 1)  # [b, C, L]
        v = v.permute(0, 2, 1)  # [b, C, L]
        x, attn = self.former2cnn(q, k, v)  # [b, L, C]
        return x, attn


class CNNEncoder(nn.Module):
    def __init__(self, conv_layers, args=None):
        super(CNNEncoder, self).__init__()
        self.first_conv = ConvModule(args)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x, time_steps, measurement, attn_mask=None, time_mask=None):
        measurement = self.first_conv(measurement)  # [b,24,256]

        for conv_layer in self.conv_layers:
            measurement = conv_layer(measurement)  # [b, C, L]

        x = measurement.permute(0, 2, 1)  # [b, L, C]

        return x, None


class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers, cnn2former_layers, former2cnn_layers, args=None):
        super(Encoder, self).__init__()
        self.first_conv = ConvModule(args)
        self.attn_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.cnn2former_layers = nn.ModuleList(cnn2former_layers)
        self.former2cnn_layers = nn.ModuleList(former2cnn_layers)
        self.cnn2former = nn.MultiheadAttention(embed_dim=args.d_model, num_heads=args.n_heads, dropout=args.dropout)
        self.former2cnn = nn.MultiheadAttention(embed_dim=args.d_model, num_heads=args.n_heads, dropout=args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, x, time_steps, measurement, attn_mask=None, time_mask=None):
        attns = []
        measurement = self.first_conv(measurement)  # [b,24,256]

        for attn_layer, conv_layer, cnn2former_layer, former2cnn_layer in zip(self.attn_layers, self.conv_layers,
                                                                              self.cnn2former_layers,
                                                                              self.former2cnn_layers):
            # measurement_temp, _ = cnn2former_layer(x, measurement, measurement)  # [b, L, C]
            x_temp, _ = cnn2former_layer(x, measurement, measurement)  # [b, L, C]
            # x = torch.add(x, measurement_temp)  # [b, L, C]

            measurement = conv_layer(measurement)  # [b, C, L]

            x, time_steps, attn = attn_layer(x_temp, time_steps, attn_mask=attn_mask, time_mask=time_mask)  # [b, L, C]
            measurement, _ = former2cnn_layer(measurement, x, x)  # [b, C, L]
            # measurement = torch.add(measurement, measurement_temp)  # [b, C, L]

            attns.append(attn)

        measurement = measurement.permute(0, 2, 1)  # [b, L, C]
        x = torch.add(x, measurement)  # [b, L, C]
        x = self.layer_norm(x)  # Add & Norm

        return x, attns


class FeatureFusionBlock(nn.Module):
    def __init__(self, configs, class_num):
        super(FeatureFusionBlock, self).__init__()

        self.configs = configs

        # Embedding
        self.enc_embedding = DataEmbedding(configs)

        if configs.fusion_type == 'ISTA+CNN':
            self.encoder = Encoder(
                [
                    IrregularTimeEncoderLayer(
                        IrregularTimeAttentionLayer(
                            IrregularTimeAttention(eps=configs.eps, dropout=configs.dropout),
                            configs.d_model, configs.nhidden, configs.n_heads),
                        args=configs, layer_num=i
                    ).to(configs.device) for i in range(configs.e_layers)
                ],
                [MeasurementEncoderLayer(configs).to(configs.device) for _ in range(configs.e_layers)],
                [CNN2Former(configs).to(configs.device) for _ in range(configs.e_layers)],
                [Former2CNN(configs).to(configs.device) for _ in range(configs.e_layers)],
                args=configs,
            ).to(configs.device)
        elif configs.fusion_type == 'CNN':
            self.encoder = nn.ModuleList(
                [MeasurementEncoderLayer(configs).to(configs.device) for _ in range(configs.e_layers)])
        elif configs.fusion_type == 'ISTA':
            self.encoder = nn.ModuleList([
                IrregularTimeEncoderLayer(
                    IrregularTimeAttentionLayer(
                        IrregularTimeAttention(eps=configs.eps, dropout=configs.dropout),
                        configs.d_model, configs.nhidden, configs.n_heads),
                    args=configs, layer_num=i
                ).to(configs.device) for i in range(configs.e_layers)
            ]).to(configs.device)
        else:
            assert False, 'Invalid fusion type'

        self.time_encoder = TimeEmbedding(1, args=configs).to(configs.device)
        self.time_avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = Classifier(configs.seq_len + 3, class_num, configs)

    def normalization_and_minmaxdiff(self, x, mode='value'):
        """

        :param mode:
        :param x:
        """
        b, l, c = x.size()
        x_means = torch.mean(x, dim=1).unsqueeze(1)

        x_stds = torch.std(x, dim=1).unsqueeze(1)
        x_normal = (x - x_means) / x_stds
        if mode == 'value':
            x_min = torch.zeros((b, 1)).to(self.configs.device)
        elif mode == 'key':
            x_min, _ = torch.min(x, dim=1)
        else:
            assert False, 'Invalid mode'
        x_max, _ = torch.max(x, dim=1)
        x_minmaxdiff = x_max - x_min
        return x_normal, x_minmaxdiff, x_means.squeeze(1), x_stds.squeeze(1)

    def forward(self, x, time_steps, fit_x, periods, attn_mask=None):
        if self.configs.use_time_mask:
            time_mask = attn_mask
        else:
            time_mask = None

        attn_mask = torch.repeat_interleave(attn_mask, self.configs.seq_len, dim=-1)
        attn_mask = torch.unsqueeze(attn_mask, -1)
        attn_mask = torch.repeat_interleave(attn_mask, self.configs.d_model, dim=-1)  # torch.Size([50, 256, 256, 16])
        x_normal, x_minmaxdiff, x_means, x_stds = self.normalization_and_minmaxdiff(x)
        fx_normal, fx_minmaxdiff, fx_means, fx_stds = self.normalization_and_minmaxdiff(fit_x)
        x_normal = self.enc_embedding(x_normal)

        x, attns = self.encoder(x_normal, time_steps, fx_normal, attn_mask=attn_mask, time_mask=time_mask)

        feature = x
        feature = F.adaptive_avg_pool1d(feature, 1)
        feature = feature.squeeze(-1)
        t = self.time_encoder(time_steps, time_mask)
        x = torch.add(x, t)
        x = self.avgpool(x).squeeze(-1)

        x = torch.cat([x, periods, x_minmaxdiff, x_means], dim=1)
        x = self.classifier(x)

        return x, attns,feature
