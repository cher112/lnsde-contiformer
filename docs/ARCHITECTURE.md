# LNSDE-ContiFormer 项目架构详细分析

## 1. 项目概览

### 1.1 项目目标
Neural SDE + ContiFormer 框架，用于**不规则采样时间序列分类**，特别针对天文光变曲线数据。通过结合三种稳定的神经随机微分方程与连续时间Transformer，解决不规则时序建模和类别不平衡两大核心问题。

### 1.2 核心创新
1. **连续时间建模**: 神经随机微分方程 (Neural SDE) 与 ContiFormer 的协同框架
2. **类别平衡处理**: 类别感知分组注意力 (CGA) 模块 + 智能混合重采样
3. **多种SDE实现**: Langevin-type SDE、Linear Noise SDE、Geometric SDE

### 1.3 数据集支持
- **MACHO**: 天文多色光变曲线（7类，2000样本）
- **LINEAR**: 大面积巡天（类别不平衡数据）
- **ASAS**: 小孔径天文测量（不规则采样）

---

## 2. 目录结构详解

```
lnsde-contiformer-master/
├── models/                 # 模型架构核心模块
│   ├── __init__.py
│   ├── base_sde.py        # 基础SDE接口与工具类
│   ├── langevin_sde.py    # 朗之万型SDE实现 (dX = -∇U dt + σ dW)
│   ├── linear_noise_sde.py # 线性噪声SDE实现 (dX = f dt + σX dW)
│   ├── geometric_sde.py    # 几何SDE实现 (dX/X = f dt + σ dW)
│   ├── contiformer.py      # 连续时间Transformer模块
│   └── cga_module.py       # 类别感知分组注意力模块
│
├── utils/                  # 工具和辅助函数
│   ├── __init__.py
│   ├── config.py          # 配置参数管理（数据集、SDE、模型参数）
│   ├── dataloader.py      # 数据加载与预处理
│   ├── preprocessing.py    # 时间序列预处理工具
│   ├── resampling.py      # 混合重采样实现（SMOTE + ENN）
│   ├── model_utils.py     # 模型创建、保存、加载
│   ├── trainer.py         # 基础训练接口
│   ├── training_manager.py # 完整训练流程管理
│   ├── training_utils.py  # 训练epoch函数、评估指标
│   ├── loss.py            # Focal Loss 损失函数
│   ├── system_utils.py    # GPU管理、随机种子、设备选择
│   ├── logging_utils.py   # 实验日志记录
│   ├── visualization.py   # 训练过程可视化
│   ├── path_manager.py    # 路径管理与标准化
│   ├── stability_fixes.py # 数值稳定性增强
│   └── elses/             # 遗留/备用模块
│
├── test/                  # 测试与实验脚本（61个）
│   ├── merge_macho_logs.py         # 合并MACHO训练日志
│   ├── test_model_shapes.py        # 模型形状测试
│   ├── test_sde_solving.py         # SDE求解测试
│   ├── test_dataloader.py          # 数据加载测试
│   ├── fix_nan_issue.py            # NaN问题诊断与修复
│   ├── analyze_nan_filtering.py    # NaN样本过滤分析
│   ├── debug_resampling.py         # 重采样过程调试
│   ├── resample_datasets.py        # 数据集重采样生成
│   ├── parallel_resampling.py      # 并行重采样
│   ├── linear_physics_timegan_resampling.py  # TimeGAN重采样
│   └── [其他诊断和优化脚本]
│
├── visualization/         # 数据可视化脚本（19个）
│   ├── macho_dataset_overview.py              # MACHO数据集概览
│   ├── linear_dataset_overview.py             # LINEAR数据集概览
│   ├── asas_dataset_overview.py               # ASAS数据集概览
│   ├── universal_training_visualization_recalc.py  # 通用训练可视化
│   ├── macho_merged_visualization.py          # MACHO合并结果可视化
│   ├── lightcurve_resampling_comparison.py    # 重采样对比可视化
│   ├── resampling_quality_visualization.py    # 重采样质量评估
│   ├── class_imbalance_comparison.py          # 类别不平衡对比
│   ├── imbalance_performance_analysis.py      # 性能与不平衡关联分析
│   └── [其他可视化脚本]
│
├── scripts/               # 脚本工具
│   └── macho_physics_timegan_resampling.py  # 物理约束TimeGAN重采样
│
├── docs/                  # 文档与参考资料
│   ├── docs总结.md                           # 技术文献总结
│   ├── NaN样本自动过滤系统.md                 # NaN处理文档
│   ├── torchsde库分析.md                     # TorchSDE库分析
│   ├── torchdiffeq库分析.md                  # TorchDiffEq库分析
│   ├── physiopro_contiformer库分析.md        # PhysioPro分析
│   ├── 数据集版本使用指南.md                  # 数据集说明
│   ├── 训练数据.md                           # 训练配置说明
│   └── [论文PDF和研究报告]
│
├── main.py                # 主训练脚本
├── main_backup.py         # 备份版本
├── CLAUDE.md              # 项目工作要求说明
└── demo.ipynb             # 演示Notebook
```

---

## 3. 核心模块详解

### 3.1 模型架构模块 (models/)

#### 3.1.1 BaseSDEModel (base_sde.py)
**功能**: 所有SDE模型的基类，定义通用接口

```python
class BaseSDEModel(nn.Module, ABC):
    """基础SDE模型"""
    @abstractmethod
    def f(self, t, y):
        """漂移函数 (drift)"""
    
    @abstractmethod
    def g(self, t, y):
        """扩散函数 (diffusion)"""
    
    def forward(self, ts, batch_size):
        """SDE求解"""
```

**关键工具类**:
- `MaskedSequenceProcessor`: 处理padding和mask的工具

#### 3.1.2 三种SDE实现

##### LangevinSDEContiformer (langevin_sde.py)
```
方程: dX_t = -∇U(X_t)dt + σ(t)dW_t
特性: 势能函数U(x,t)通过神经网络建模
优势: 有不变测度，适合收敛系统
```

**架构**:
```
输入 (batch, seq_len, 3)
  ↓
时间编码和mask处理
  ↓
Langevin SDE 求解 (torchsde.sdeint)
  ↓
ContiFormer 处理
  ↓
CGA 模块 (可选)
  ↓
分类头 (Linear + Softmax)
  ↓
输出 (batch, num_classes)
```

##### LinearNoiseSDEContiformer (linear_noise_sde.py)
```
方程: dX_t = f(t,X_t)dt + σ(t)X_t dW_t
特性: 乘性线性噪声，有稳定性保证
稳定条件: |σ|² > 2L_f（扩散强度超过漂移常数）
```

**特殊优化**:
- 梯度断开 (gradient detach) 机制防止梯度爆炸
- 可配置的断开间隔 (detach_interval)

##### GeometricSDEContiformer (geometric_sde.py)
```
方程: dX_t/X_t = f(t,X_t)dt + σ(t)dW_t
特性: 解始终为正，0为吸收态
应用: 建模非负量（如光强、概率）
```

#### 3.1.3 ContiFormerModule (contiformer.py)
**功能**: 连续时间Transformer，处理不规则时间序列

**核心组件**:
1. **PositionalEncoding**: 连续时间位置编码
   ```python
   pe(t) = sin(ωᵢ·t)、cos(ωᵢ·t)  # 频率ωᵢ = 10000^(2i/d)
   ```

2. **ContinuousMultiHeadAttention**: 时间感知注意力
   - 查询函数使用三次样条插值
   - 键值函数通过ODE动态扩展
   - 内积积分用高斯求积近似

3. **EncoderLayer**: 标准Transformer块
   - Layer Norm → Multi-Head Attention → FFN
   - 支持residual connections

#### 3.1.4 CGAClassifier (cga_module.py)
**功能**: 类别感知分组注意力，处理类别不平衡

**工作流程**:
```
输入特征 (batch, dim)
  ↓
输入投影到类别空间 (dim → group_dim × num_classes)
  ↓
[For each class]:
  - 类别特定QKV投影
  - 多头自注意力
  - 语义相似度计算
  - 类间交互门控
  ↓
融合和输出投影
  ↓
输出 (batch, dim)
```

**关键特性**:
- 为每个类别创建独立表示通道
- 基于语义相似度的类间信息交换
- 门控机制防止过度混合

---

### 3.2 数据处理模块 (utils/)

#### 3.2.1 配置管理 (config.py)

**三个关键函数**:

1. **get_dataset_specific_params(dataset_id)**
   - 为ASAS/LINEAR/MACHO返回数据集特定参数
   - 包括: temperature、focal_gamma、min_time_interval等

2. **setup_sde_config(sde_config_id)**
   - 配置SDE求解参数: 方法、dt、rtol、atol
   - 不同SDE类型有差异化配置

3. **setup_dataset_mapping(args)**
   - 将数据集ID映射到文件路径
   - 支持: original、fixed、enhanced、resampled四种数据版本

#### 3.2.2 数据加载 (dataloader.py)

**LightCurveDataset 类**:
```python
# 数据格式 (pkl文件，包含List[Dict])
{
    'time': np.array,       # 时间点 (n,)
    'mag': np.array,        # 星等值 (n,)
    'errmag': np.array,     # 观测误差 (n,)
    'mask': np.array,       # 有效数据标记 (n,)
    'period': float,        # 周期
    'label': int,           # 类别标签 (0-6)
    'file_id': str,         # 样本ID
    'class_name': str       # 类别名称
}
```

**collate_fn 函数**:
- Padding不规则长度序列到max_seq_len
- 返回张量化数据和mask

#### 3.2.3 混合重采样 (resampling.py)
**功能**: 处理类别不平衡，生成高质量合成样本

**包含技术**:
1. **SVM-SMOTE**: 改进的少数类过采样
   - 使用DTW距离计算相似度
   - 插值时同时处理特征和时间维度

2. **Repeated ENN**: 多数类欠采样
   - 基于k-NN的样本去除
   - 保留难分类样本

3. **TimeGAN**: 时间序列生成对抗网络
   - 生成物理约束的合成光变曲线

**输出**: 重采样后的pkl文件

#### 3.2.4 模型工具 (model_utils.py)

**create_model(model_type, num_classes, args, dataset_config)**
- 根据model_type创建对应的SDE+ContiFormer模型
- 支持: 'langevin'、'linear_noise'、'geometric'
- 可选启用/禁用各子模块

**save_checkpoint / load_model_checkpoint**
- 保存/加载模型权重和优化器状态
- 支持最优模型和最新检查点

---

### 3.3 训练管理模块

#### 3.3.1 TrainingManager (training_manager.py)
**核心功能**: 整合完整的训练流程

```python
class TrainingManager:
    def run_training(self, log_path, log_data, best_val_acc, start_epoch):
        """
        For each epoch:
            1. train_epoch() - 训练阶段
            2. validate_epoch() - 验证阶段
            3. scheduler.step() - 学习率调度
            4. 保存最优模型
            5. 日志更新和可视化
        """
```

**关键特性**:
- 混合精度训练 (AMP)
- GradScaler 用于梯度缩放
- 标准化模型保存路径

#### 3.3.2 训练循环 (training_utils.py)

**train_epoch函数**:
```
For each batch:
    1. 前向传播 (支持AMP自动混合精度)
    2. NaN检查和跳过
    3. 梯度累积
    4. 自适应梯度裁剪 (clip_value = 5.0)
    5. 优化器步长
    6. 计算指标 (loss、accuracy、F1等)
```

**validate_epoch函数**:
```
无梯度计算:
    1. 前向传播
    2. 损失计算
    3. 混淆矩阵生成
    4. 各类别准确率
    5. 宏/微平均指标
```

---

## 4. 数据流图

### 4.1 训练流程数据流

```
原始数据 (MACHO/LINEAR/ASAS)
    ↓
[数据预处理]
    - 时间归一化
    - Padding到固定长度
    - 生成mask
    ↓
[可选: 混合重采样]
    - SMOTE少数类过采样
    - ENN多数类欠采样
    - 或TimeGAN生成
    ↓
DataLoader (batch处理)
    ↓
[模型前向传播]
    1. 特征编码 (Linear layer)
    2. SDE求解 (Langevin/LinearNoise/Geometric)
    3. ContiFormer处理 (CT-MHA)
    4. CGA模块 (可选)
    5. 分类头
    ↓
[损失计算与梯度反向传播]
    - CrossEntropy 或 Focal Loss
    - 梯度累积
    - 梯度裁剪
    ↓
[优化器更新]
    - AdamW优化
    - 学习率调度
    ↓
[验证和保存]
    - 评估指标计算
    - 最优模型保存
    ↓
日志和可视化
```

### 4.2 模型前向传播数据流

```
输入: features (batch, seq_len, 3)
      mask (batch, seq_len)

↓ 特征编码
  Linear(3 → hidden_channels)

↓ SDE处理 [可选]
  Langevin/LinearNoise/Geometric SDE
  (多个积分步求解连续轨迹)

↓ ContiFormer处理 [可选]
  多层 CT-MHA + FFN
  (seq_len保持不变)

↓ 序列池化
  取最后有效位置的特征

↓ CGA模块 [可选]
  类别感知分组注意力
  (为每个类别创建表示)

↓ 分类头
  Linear(contiformer_dim → num_classes)

输出: logits (batch, num_classes)
```

---

## 5. 关键参数说明

### 5.1 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_channels | 128 | SDE隐藏维度 |
| contiformer_dim | 256 | ContiFormer维度 |
| n_heads | 8 | 多头注意力头数 |
| n_layers | 6 | 编码器层数 |
| dropout | 0.1 | Dropout率 |
| cga_group_dim | 64 | CGA中每个类别组维度 |
| cga_heads | 4 | CGA注意力头数 |

### 5.2 SDE求解参数

| 参数 | 准确率优先 | 平衡 | 时间优先 |
|------|-----------|------|---------|
| dt | 0.005~0.01 | 0.025~0.03 | 0.1~0.2 |
| rtol | 1e-5~1e-6 | 1e-5~1e-4 | 1e-4~1e-3 |
| atol | 1e-6~1e-7 | 1e-5~1e-6 | 1e-4~1e-5 |
| method | milstein | euler | euler |

### 5.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 64 | 批大小 |
| learning_rate | 1e-5 | 学习率(AdamW) |
| weight_decay | 5e-4 | L2正则化 |
| gradient_clip | 5.0 | 梯度裁剪值 |
| label_smoothing | 0.1 | 标签平滑系数 |
| gradient_accumulation_steps | 2 | 梯度累积步数 |

### 5.4 数据集特定参数

#### MACHO
- temperature: 1.5
- focal_gamma: 2.5
- min_time_interval: 0.01

#### LINEAR
- temperature: 0.8
- focal_gamma: 3.0
- min_time_interval: 0.005

#### ASAS
- temperature: 1.0
- focal_gamma: 2.0
- min_time_interval: 0.01

---

## 6. 核心算法实现

### 6.1 SDE求解流程

使用torchsde库的`sdeint`函数:

```python
# 求解SDE: dy = f(t,y)dt + g(t,y)dW_t
y_trajectory = torchsde.sdeint(
    sde_model,          # SDE对象 (实现f和g)
    y0,                 # 初始条件
    ts,                 # 时间点
    method='euler',     # 求解方法
    dt=0.01,           # 时间步长
    adaptive=False,    # 非自适应
    rtol=1e-3,         # 相对容差
    atol=1e-4,         # 绝对容差
)
```

### 6.2 ContiFormer 连续时间注意力

```python
# 键值函数的ODE扩展
dki/dt = f(t, ki(t))
ki(t) = ki(ti) + ∫[ti to t] f(τ, ki(τ))dτ

# 连续内积
αi(t) = ∫[ti to t] q(τ)·ki(τ)ᵀ dτ / (t - ti)

# 高斯求积近似
αi(t) ≈ (1/2) ∑[p=1 to P] γp·q(ξp)·k(ξp)ᵀ
```

### 6.3 Focal Loss

```python
# 处理类别不平衡
L_focal = -∑ (1-pt)^γ · log(pt)

其中 pt = 经过softmax的目标类概率
     γ = focal参数(通常2-3)
```

---

## 7. 关键特性和优化

### 7.1 数值稳定性增强
- **梯度断开**: LinearNoise SDE每N步断开一次梯度
- **梯度裁剪**: 自适应裁剪，早期严格后期放宽
- **自动混合精度 (AMP)**: 减少内存，加速计算

### 7.2 类别不平衡处理
- **数据层**: 混合重采样 (SMOTE + ENN + TimeGAN)
- **模型层**: CGA分组注意力
- **损失层**: Focal Loss加权

### 7.3 内存和速度优化
- **梯度累积**: 有效批次 = batch_size × gradient_accumulation_steps
- **梯度检查点**: 存储中间激活值
- **混合精度**: float16计算，float32存储

---

## 8. 配置和使用示例

### 8.1 基础训练命令

```bash
# MACHO数据集，Linear Noise SDE，时间优先配置
python main.py \
    --dataset 3 \
    --model_type 2 \
    --sde_config 3 \
    --epochs 100 \
    --batch_size 64
```

### 8.2 启用高级功能

```bash
# 使用增强数据、启用所有组件
python main.py \
    --dataset 3 \
    --use_enhanced \
    --use_sde 1 \
    --use_contiformer 1 \
    --use_cga 1 \
    --hidden_channels 128 \
    --contiformer_dim 256
```

### 8.3 调试配置

```bash
# 快速测试，禁用高级模块
python main.py \
    --dataset 3 \
    --use_sde 0 \
    --use_contiformer 0 \
    --use_cga 0 \
    --epochs 1 \
    --batch_size 16
```

---

## 9. 输出目录结构

```
/root/autodl-tmp/lnsde-contiformer/results/
├── [DATASET]/
│   └── [TIMESTAMP]/
│       ├── models/
│       │   ├── best_model.pth
│       │   ├── latest_model.pth
│       │   └── checkpoint_epoch_XX.pth
│       ├── [DATASET]_[MODEL]_config[N]_[TIMESTAMP].log  # 训练日志
│       ├── training_curve.png                            # 损失曲线
│       ├── confusion_matrix.png                          # 混淆矩阵
│       ├── class_accuracy.png                            # 各类准确率
│       └── experiment_config.txt                         # 配置记录
```

---

## 10. 依赖库说明

### 核心库
- **torch**: 深度学习框架
- **torchsde**: SDE求解库 (`/root/autodl-tmp/torchsde`)
- **physiopro**: ContiFormer实现 (`/root/autodl-tmp/PhysioPro`)
- **scikit-learn**: 数据处理和指标计算
- **numpy**: 数值计算
- **pandas**: 数据处理
- **matplotlib/seaborn**: 可视化

### 文件系统路径
```python
# 源代码
/root/autodl-tmp/torchsde/
/root/autodl-tmp/PhysioPro/

# 数据文件
/root/autodl-fs/lnsde-contiformer/data/
  - MACHO_original.pkl / MACHO_fixed.pkl / MACHO_enhanced.pkl
  - LINEAR_original.pkl / LINEAR_fixed.pkl / LINEAR_enhanced.pkl
  - ASAS_original.pkl / ASAS_fixed.pkl / ASAS_enhanced.pkl
  - macho_resample_timegan.pkl (TimeGAN重采样)
```

---

## 11. 常见问题和调试

### 11.1 NaN Loss问题
- 降低学习率 (1e-5)
- 启用梯度裁剪 (5.0)
- 减小时间步长 (dt=0.01)
- 禁用SDE/ContiFormer测试基础模型

### 11.2 内存溢出
- 降低batch_size
- 启用梯度累积
- 启用混合精度 (AMP)
- 减小模型维度

### 11.3 收敛速度慢
- 增加学习率 (但需谨慎)
- 启用梯度检查点
- 增加batch_size
- 使用重采样数据加速

### 11.4 验证准确率不提升
- 检查数据集平衡
- 尝试不同的SDE配置
- 调整CGA参数
- 使用Focal Loss加权

---

## 12. 代码组织最佳实践

### 12.1 添加新模型
1. 在 `models/` 下创建新文件
2. 继承 `BaseSDEModel`
3. 实现 `f()` 和 `g()` 方法
4. 在 `models/__init__.py` 中导出
5. 在 `model_utils.py` 中添加创建函数

### 12.2 添加新数据集
1. 准备pkl格式数据 (List[Dict])
2. 添加到 `config.py` 的 `setup_dataset_mapping()`
3. 添加数据集特定参数到 `get_dataset_specific_params()`
4. 创建对应的可视化脚本

### 12.3 新增训练脚本
- 实验脚本放在 `test/`
- 可视化脚本放在 `visualization/`
- 遵循命名规范

---

## 附录A: 技术词汇表

| 术语 | 缩写 | 说明 |
|------|------|------|
| Stochastic Differential Equation | SDE | 随机微分方程 |
| Linear Noise SDE | LNSDE | 线性噪声随机微分方程 |
| Continuous-Time Transformer | ContiFormer | 连续时间Transformer |
| Category-aware Grouped Attention | CGA | 类别感知分组注意力 |
| Synthetic Minority Over-sampling | SMOTE | 少数类过采样技术 |
| Edited Nearest Neighbors | ENN | 近邻编辑欠采样 |
| Automatic Mixed Precision | AMP | 自动混合精度训练 |
| Dynamic Time Warping | DTW | 动态时间规整距离 |
| Time GAN | TimeGAN | 时间序列生成对抗网络 |

---

*最后更新: 2025-11-02*
*项目路径: `/Users/sunzemuzi/Downloads/lnsde-contiformer-master`*
