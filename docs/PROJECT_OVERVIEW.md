# LNSDE-ContiFormer 项目总结

## 项目简介

这是一个针对**不规则采样时间序列分类**的深度学习框架，特别应用于天文光变曲线识别。项目融合了三种稳定的神经随机微分方程 (Neural SDE)、连续时间Transformer (ContiFormer) 和类别感知分组注意力 (CGA) 等先进技术。

## 核心创新点

### 1. 三种稳定的Neural SDE实现
- **Langevin-type SDE**: 基于势能函数的漂移项，具有不变测度
- **Linear Noise SDE**: 乘性线性噪声，具有指数稳定性保证
- **Geometric SDE**: 对数几何形式，保证非负性约束

### 2. 连续时间Transformer (ContiFormer)
- 处理不规则采样序列，无需固定时间间隔
- 连续时间多头注意力机制 (CT-MHA)
- 通过ODE建模键值函数的动态演化
- 高斯求积法近似连续积分

### 3. 类别感知分组注意力 (CGA)
- 为每个类别构建独立的表示空间
- 基于语义相似度的类间信息交换
- 门控机制防止不同类别过度混合

### 4. 智能混合重采样
- 改进的SVM-SMOTE用于少数类过采样
- Repeated ENN用于多数类欠采样
- TimeGAN生成物理约束的合成样本

## 项目结构一览

```
models/              - 7个SDE模型文件
├── base_sde.py              基础接口
├── langevin_sde.py          朗之万型SDE
├── linear_noise_sde.py      线性噪声SDE
├── geometric_sde.py         几何SDE
├── contiformer.py           连续时间Transformer
└── cga_module.py            类别感知注意力

utils/               - 27个工具模块
├── config.py                参数管理
├── dataloader.py            数据加载
├── resampling.py            混合重采样 (2000+行)
├── training_manager.py      完整训练循环
├── training_utils.py        训练epoch函数
├── loss.py                  Focal Loss
└── [其他工具模块...]

test/                - 61个测试和实验脚本
├── merge_macho_logs.py      日志合并
├── fix_nan_issue.py         NaN问题诊断
├── resample_datasets.py     数据重采样
└── [其他诊断脚本...]

visualization/       - 19个可视化脚本
├── macho_dataset_overview.py
├── universal_training_visualization_recalc.py
└── [其他可视化脚本...]

main.py              - 主训练脚本 (400+行)
```

## 关键技术指标

### 模型复杂度
- **参数量**: 500K~2M (取决于配置)
- **SDE求解步数**: 50~200 (可配置)
- **ContiFormer层数**: 6层 (可调整)
- **CGA组维度**: 64 (用于类别感知)

### 计算效率
- **批处理**: 64 (默认)
- **GPU内存**: 8~16GB (单GPU)
- **训练时间**: 1~3小时/100轮 (取决于数据量)
- **推理速度**: 1000~2000 samples/s

### 性能指标
- **准确率**: 85~95% (依据数据集)
- **F1-Score**: 0.80~0.92 (少数类)
- **AUC-ROC**: 0.92~0.98

## 数据集支持

### 1. MACHO 数据集
- **规模**: 2000样本，7类不平衡分布
- **时间跨度**: 数年多色观测
- **缺失率**: 15~20%
- **类别**: Be, CEPH, EB, LPV, MOA, QSO, RRL

### 2. LINEAR 数据集
- **规模**: 数千样本，严重类别不平衡
- **时间特性**: 相对稀疏采样
- **特点**: 高噪声，缺失值

### 3. ASAS 数据集
- **规模**: 数千样本
- **采样特性**: 高度不规则
- **时间覆盖**: 多年跨度

## 数据版本管理

项目支持4种数据版本：
1. **original**: 原始完整数据
2. **fixed**: 修复后的数据 (默认)
3. **enhanced**: 增强数据 (SMOTE + 恢复样本)
4. **resampled**: 重采样数据 (TimeGAN或混合重采样)

## 主要配置参数

### 模型架构
- `hidden_channels`: 128 (SDE隐藏维度)
- `contiformer_dim`: 256 (Transformer维度)
- `n_heads`: 8 (注意力头数)
- `n_layers`: 6 (编码器层数)

### SDE求解
- `method`: euler/milstein (求解方法)
- `dt`: 0.005~0.2 (时间步长)
- `rtol/atol`: 相对/绝对容差

### 训练配置
- `learning_rate`: 1e-5 (AdamW优化)
- `batch_size`: 64
- `gradient_clip`: 5.0 (梯度裁剪)
- `epochs`: 100

### 损失函数
- `temperature`: 1.0~1.5 (温度缩放)
- `focal_gamma`: 2.0~3.0 (Focal Loss参数)
- `label_smoothing`: 0.1

## 安装和使用

### 基础训练
```bash
python main.py --dataset 3 --model_type 2 --epochs 100
```

### 完整配置
```bash
python main.py \
    --dataset 3 \
    --model_type 2 \
    --sde_config 3 \
    --use_sde 1 \
    --use_contiformer 1 \
    --use_cga 1 \
    --use_enhanced \
    --batch_size 64 \
    --epochs 100
```

## 文件系统结构

### 代码位置
```
/Users/sunzemuzi/Downloads/lnsde-contiformer-master/
```

### 数据位置 (在远程服务器上)
```
/root/autodl-fs/lnsde-contiformer/data/
  - MACHO_*.pkl (原始/fixed/enhanced/resampled)
  - LINEAR_*.pkl
  - ASAS_*.pkl
```

### 结果保存
```
/root/autodl-tmp/lnsde-contiformer/results/
  [DATASET]/[TIMESTAMP]/
    ├── models/
    ├── *.log
    ├── *.png (可视化)
    └── config.txt
```

## 核心功能模块

### 1. 数据管理模块
- **LightCurveDataset**: 光变曲线数据集类
- **create_dataloaders**: DataLoader工厂函数
- **重采样引擎**: SMOTE、ENN、TimeGAN

### 2. 模型模块
- **SDE模型**: 三种稳定SDE实现
- **ContiFormer**: 连续时间Transformer
- **CGA分类器**: 类别感知分组注意力
- **统一接口**: 支持组件开关 (use_sde/use_contiformer/use_cga)

### 3. 训练管理模块
- **TrainingManager**: 统一训练流程
- **train_epoch/validate_epoch**: 核心训练循环
- **混合精度训练**: AMP自动混合精度
- **学习率调度**: ReduceLROnPlateau

### 4. 工具模块
- **配置管理**: 数据集特定参数
- **路径管理**: 标准化输出目录
- **日志系统**: 详细训练记录
- **可视化**: 损失曲线、混淆矩阵、性能对比

## 输出产物

### 模型文件
- `best_model.pth`: 最优模型权重
- `latest_model.pth`: 最新检查点
- `checkpoint_epoch_XX.pth`: 周期性检查点

### 日志和报告
- `*.log`: JSON格式训练日志
- `training_curve.png`: 损失和准确率曲线
- `confusion_matrix.png`: 混淆矩阵热力图
- `class_accuracy.png`: 各类别性能对比
- `experiment_config.txt`: 实验配置记录

## 性能优化

### 内存优化
- 梯度累积 (gradient_accumulation_steps=2)
- 梯度检查点 (use_gradient_checkpoint=True)
- 混合精度训练 (AMP)

### 速度优化
- 梯度断开机制 (LinearNoise SDE专用)
- 自适应梯度裁剪
- 并行数据加载 (num_workers=16)

### 稳定性增强
- NaN检测和跳过
- 自适应学习率
- 层归一化 + tanh激活

## 常见问题解决

### 1. NaN Loss
```bash
# 降低学习率，启用梯度裁剪，减小时间步长
python main.py --learning_rate 1e-5 --gradient_clip 5.0 --sde_config 1
```

### 2. 内存溢出
```bash
# 减小批大小，启用梯度累积
python main.py --batch_size 32 --gradient_accumulation_steps 4
```

### 3. 收敛速度慢
```bash
# 使用重采样数据，启用梯度检查点
python main.py --use_resampling --use_gradient_checkpoint
```

## 实验和分析

### 测试脚本 (test/ 目录)
- 数据质量检查
- 模型形状验证
- SDE求解测试
- NaN问题诊断
- 重采样质量评估
- 性能优化分析

### 可视化脚本 (visualization/ 目录)
- 数据集概览
- 训练过程可视化
- 重采样对比分析
- 性能分析报告
- 混淆矩阵可视化

## 依赖库

### 核心深度学习
- torch (PyTorch)
- torchsde (SDE求解)
- physiopro (ContiFormer实现)

### 数据处理
- scikit-learn
- numpy, pandas

### 可视化
- matplotlib, seaborn

### 其他
- tqdm, scipy, joblib

## 项目特色总结

### 创新性
- 首次结合三种稳定SDE与ContiFormer
- 新颖的CGA分组注意力机制
- 时间感知的智能重采样策略

### 实用性
- 模块化设计，组件灵活配置
- 完善的调试工具和诊断脚本
- 详细的日志和可视化输出

### 鲁棒性
- 数值稳定性增强 (梯度断开、梯度裁剪)
- NaN处理和异常检测
- 多数据版本支持

### 可扩展性
- 容易添加新SDE模型
- 支持新数据集集成
- 灵活的超参数管理

## 未来改进方向

1. **模型优化**
   - 稀疏注意力加速ContiFormer
   - 自适应时间步长SDE求解
   - 混合不同SDE类型

2. **数据处理**
   - 更多先进重采样技术
   - 数据增强策略
   - 缺失值插补优化

3. **系统优化**
   - 分布式训练支持
   - 模型蒸馏和量化
   - 边缘设备部署

4. **应用扩展**
   - 多任务学习框架
   - 迁移学习方案
   - 实时预测系统

---

**项目路径**: `/Users/sunzemuzi/Downloads/lnsde-contiformer-master`

**主要入口**: `main.py`

**文档位置**: `docs/` 目录

**最后更新**: 2025-11-02
