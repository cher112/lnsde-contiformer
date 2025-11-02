# LNSDE-ContiFormer 项目文档索引

## 欢迎使用项目文档

本目录包含关于LNSDE-ContiFormer项目的全面文档，帮助用户快速理解和使用该框架。

## 文档导航

### 1. 快速开始 (推荐首先阅读)

**[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** ⭐ 必读
- 项目快速概览
- 快速启动命令
- 核心文件导读
- 参数速查表
- 常见操作指南
- 故障排查流程

**推荐用途**: 初次使用、快速查询

---

### 2. 项目理解

**[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** 
- 项目简介和核心创新
- 三种稳定SDE实现
- 完整的架构说明
- 数据集支持说明
- 关键技术指标
- 常见问题解决

**推荐用途**: 理解项目整体设计、学习技术亮点

---

### 3. 深度技术分析

**[ARCHITECTURE.md](ARCHITECTURE.md)**
- 完整项目架构解析 (676行)
- 详细模块说明
  - 模型架构模块 (models/)
  - 数据处理模块 (utils/)
  - 训练管理模块
- 数据流详细说明
- 关键参数详解
- 算法实现细节
- 配置和使用示例
- 性能优化指南

**推荐用途**: 深入理解代码实现、源代码级开发

---

### 4. 技术参考 (原始文档)

**[docs总结.md](docs总结.md)**
- ContiFormer论文详细总结 (650行)
- Stable Neural SDE论文总结
- 理论基础和数学推导
- 实验设计方案

**推荐用途**: 学习底层理论、论文研究

**[torchsde库分析.md](torchsde库分析.md)**
- SDE求解库详细说明
- API参考
- 使用示例

**推荐用途**: SDE求解实现细节

**[physiopro_contiformer库分析.md](physiopro_contiformer库分析.md)**
- ContiFormer实现说明
- 模块组织
- 关键接口

**推荐用途**: ContiFormer实现细节

**[torchdiffeq库分析.md](torchdiffeq库分析.md)**
- ODE求解库说明

**[数据集版本使用指南.md](数据集版本使用指南.md)**
- 数据格式说明
- 版本选择指南

**[训练数据.md](训练数据.md)**
- 训练配置说明

**[NaN样本自动过滤系统.md](NaN样本自动过滤系统.md)**
- NaN处理机制

---

## 按使用场景推荐

### 场景1: 我是新用户，想快速上手

**阅读顺序**:
1. QUICK_REFERENCE.md - 快速了解和运行
2. PROJECT_OVERVIEW.md - 理解项目特色
3. 根据需要查询ARCHITECTURE.md中的具体模块

**预计时间**: 30分钟

---

### 场景2: 我想理解项目架构并进行开发

**阅读顺序**:
1. PROJECT_OVERVIEW.md - 项目整体概览
2. ARCHITECTURE.md (全文) - 深入了解各模块
3. 对应的源代码文件
4. docs总结.md - 理论基础

**预计时间**: 2-3小时

---

### 场景3: 我遇到了问题，需要快速解决

**查询步骤**:
1. 查看QUICK_REFERENCE.md的"故障排查速查表"
2. 根据问题在QUICK_REFERENCE.md中查找常用命令
3. 如需更深入，参考ARCHITECTURE.md的相关章节

**预计时间**: 5-15分钟

---

### 场景4: 我想研究论文和理论细节

**阅读顺序**:
1. docs总结.md - 论文总结
2. ARCHITECTURE.md第6章 - 核心算法实现
3. 原始论文PDF (docs/目录下)

**预计时间**: 1-2小时

---

### 场景5: 我想添加新功能或修改参数

**查询步骤**:
1. QUICK_REFERENCE.md - 参数速查表
2. ARCHITECTURE.md第3和5章 - 模块和参数详解
3. 对应源代码实现

**预计时间**: 30分钟-1小时

---

## 文档内容一览

| 文档文件 | 行数 | 大小 | 主要内容 | 优先级 |
|---------|------|------|---------|--------|
| QUICK_REFERENCE.md | 371 | 9.1K | 快速参考和操作指南 | ⭐⭐⭐ |
| PROJECT_OVERVIEW.md | 341 | 8.6K | 项目总体介绍 | ⭐⭐⭐ |
| ARCHITECTURE.md | 676 | 19K | 详细技术架构 | ⭐⭐ |
| docs总结.md | 650 | 16K | 论文和理论总结 | ⭐ |
| torchsde库分析.md | 200 | 8.6K | SDE库参考 | ⭐ |
| 数据集版本使用指南.md | 150 | 5.3K | 数据格式说明 | ⭐ |
| 训练数据.md | 150 | 5.0K | 训练配置 | ⭐ |
| NaN样本自动过滤系统.md | 100 | 4.2K | NaN处理 | ⭐ |

---

## 核心概念速览

### 不规则时间序列
不规则采样间隔的时间序列数据，例如天文光变曲线。特点是采样时间点不均匀分布。

### Neural SDE (神经随机微分方程)
将随机微分方程与神经网络结合，用于建模连续时间动态过程。项目实现了三种稳定的SDE:
- **Langevin-type SDE**: 基于势能函数
- **Linear Noise SDE**: 乘性线性噪声
- **Geometric SDE**: 对数几何形式

### ContiFormer (连续时间Transformer)
专门处理不规则时间序列的Transformer变体，使用连续时间多头注意力机制 (CT-MHA)。

### CGA (分组注意力)
类别感知分组注意力，为处理类别不平衡问题而设计。

### 混合重采样
结合SMOTE、ENN和TimeGAN等多种技术处理类别不平衡。

---

## 常见文件位置

### 源代码
```
models/          - 模型实现 (7个文件)
utils/           - 工具模块 (27个文件)
main.py          - 主入口脚本
```

### 测试和实验
```
test/            - 测试脚本 (61个)
visualization/   - 可视化脚本 (19个)
scripts/         - 工具脚本
```

### 数据
```
/root/autodl-fs/lnsde-contiformer/data/  - 数据文件
```

### 结果
```
/root/autodl-tmp/lnsde-contiformer/results/  - 训练结果
```

---

## 快速命令参考

### 基础运行
```bash
python main.py
```

### 指定配置
```bash
python main.py --dataset 3 --model_type 2 --sde_config 2
```

### 查看帮助
```bash
python main.py --help
```

更多命令见QUICK_REFERENCE.md

---

## 获取帮助

### 问题排查
1. 查看QUICK_REFERENCE.md的"故障排查速查表"
2. 查看ARCHITECTURE.md的"关键特性和优化"章节
3. 查看对应模块的源代码注释

### 查询参数
1. 使用QUICK_REFERENCE.md的"参数速查表"
2. 查看ARCHITECTURE.md的"关键参数说明"
3. 使用`python main.py --help`查看命令行选项

### 理解算法
1. 查看PROJECT_OVERVIEW.md的"核心创新点"
2. 查看ARCHITECTURE.md的"核心算法实现"
3. 查看docs总结.md的论文总结

---

## 文档更新日志

### 最新更新 (2025-11-02)
- 新增ARCHITECTURE.md (完整架构分析)
- 新增PROJECT_OVERVIEW.md (项目总体介绍)
- 新增QUICK_REFERENCE.md (快速参考指南)
- 创建本README.md (文档导航)

---

## 关键统计信息

### 代码规模
- 总代码行数: 15,000+
- 模型代码: 2,000+行
- 工具模块: 5,000+行
- 测试脚本: 61个
- 可视化脚本: 19个

### 文档规模
- 总文档行数: 2,500+
- 新增文档: 1,388行
- 原有文档: 1,100+行

### 性能指标
- 准确率: 85-95%
- F1-Score: 0.80-0.92
- 每个epoch耗时: 2-5分钟

---

## 联系和反馈

如有问题或建议，请参考:
- 项目主页: CLAUDE.md (项目工作要求)
- 代码注释: 各源代码文件内的详细注释
- 日志输出: 训练结果目录下的日志文件

---

## 许可证和属性

项目代码: /Users/sunzemuzi/Downloads/lnsde-contiformer-master/

主要参考:
- ContiFormer: Yuqi Chen et al., NeurIPS 2023
- Stable Neural SDE: YongKyung Oh et al., ICLR 2024

---

**快速导航**: 
- 快速上手 → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- 理解架构 → [ARCHITECTURE.md](ARCHITECTURE.md)
- 项目概览 → [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**最后更新**: 2025-11-02
