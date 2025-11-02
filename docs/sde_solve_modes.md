# SDE求解模式说明

## 概述

项目现在支持两种SDE求解模式，你可以通过 `--sde_solve_mode` 参数来选择：

- **模式0**: 逐步求解（默认，当前实现）
- **模式1**: 一次性求解整个轨迹（demo.ipynb方式）

## 两种模式的对比

### 模式0: 逐步求解（Stepwise Solving）

**实现方式**：
```python
# 对于每对相邻观测点，独立求解SDE
z₀ → z₁  (求解 SDE: [t₀, t₁])
z₁ → z₂  (求解 SDE: [t₁, t₂])
z₂ → z₃  (求解 SDE: [t₂, t₃])
...
```

**特点**：
- ✅ **内存效率高**: 只存储观测点的状态
- ✅ **适合不规则采样**: 每段可以有不同的时间间隔
- ✅ **计算效率高**: 只计算观测点之间的转移
- ⚠️ **局部建模**: 每次只看相邻两点，可能忽略长期依赖
- ⚠️ **累积误差**: 误差会沿着时间序列累积

**代码位置**: `models/linear_noise_sde.py::_forward_with_sde()`

### 模式1: 一次性求解整个轨迹（Full Trajectory Solving）

**实现方式**：
```python
# 一次性求解整个时间序列
ts = [t₀, t₁, t₂, ..., tₙ]  # 完整的时间点序列
z_trajectory = sdeint(sde, z₀, ts)  # 一次求解
# 得到: [z(t₀), z(t₁), ..., z(tₙ)]
```

**特点**：
- ✅ **全局建模**: 考虑整个轨迹的连续性
- ✅ **真正的连续时间**: 符合Neural SDE的理论定义
- ✅ **可插值**: 理论上可以在任意时刻采样状态
- ⚠️ **内存占用大**: 需要存储完整的求解轨迹
- ⚠️ **计算开销大**: 特别是对于长序列

**代码位置**: `models/linear_noise_sde.py::_forward_with_sde_full_trajectory()`

## 使用方法

### 训练时选择模式

```bash
# 使用模式0（逐步求解，默认）
python main.py --dataset 3 --sde_solve_mode 0

# 使用模式1（一次性求解）
python main.py --dataset 3 --sde_solve_mode 1
```

### 在代码中使用

```python
from models.linear_noise_sde import LinearNoiseSDEContiformer

# 创建模型时指定求解模式
model = LinearNoiseSDEContiformer(
    input_dim=3,
    hidden_channels=128,
    num_classes=7,
    sde_solve_mode=0,  # 0=逐步求解, 1=一次性求解
    # ... 其他参数
)

# 前向传播会自动使用指定的求解模式
outputs = model(time_series, mask)
```

## 测试和对比

### 运行对比测试

```bash
# 在CPU上测试
python test/compare_sde_solve_modes.py --device cpu --trials 5

# 在GPU上测试
python test/compare_sde_solve_modes.py --device cuda --trials 5
```

测试脚本会比较：
1. **计算时间**: 哪种模式更快
2. **输出差异**: 两种模式的数值差异
3. **预测一致性**: 两种模式是否给出相同的预测
4. **全局一致性**: 验证逐步求解是否学到统一的动力学

### 预期结果

根据demo.ipynb和理论分析：

| 指标 | 模式0 (逐步) | 模式1 (一次性) |
|-----|------------|--------------|
| **计算速度** | 快 | 慢 |
| **内存使用** | 低 | 高 |
| **数值精度** | 中等 | 高 |
| **理论正确性** | 局部逼近 | 全局精确 |

## 技术细节

### 为什么会有差异？

即使两种模式使用相同的SDE参数，输出仍可能不同，原因包括：

1. **数值求解器的差异**
   - 模式0: 多次短区间求解，每次重新初始化
   - 模式1: 一次长区间求解，保持求解器状态

2. **浮点运算顺序**
   - 不同的计算顺序导致舍入误差累积不同

3. **布朗运动的采样**
   - 两种模式可能在内部生成不同的布朗运动样本

### 理论背景

根据 demo.ipynb 的示例，标准的Neural SDE应该使用一次性求解：

```python
# demo.ipynb 的标准做法
sde = SDE()
ts = torch.linspace(0, 1, t_size)  # 完整时间序列
y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
ys = torchsde.sdeint(sde, y0, ts, method='euler')  # 一次性求解
```

这是因为SDE的定义本身就是连续时间的：

$$
dY_t = f(t, Y_t)dt + g(t, Y_t)dW_t
$$

理论上，$Y_t$ 在整个时间区间 $[t_0, T]$ 上都有定义。

### 当前实现的权衡

**模式0（默认）**是工程上的折衷：
- 优势：内存和计算效率
- 劣势：可能只是"记忆"局部转移，而非真正"理解"全局动力学

**模式1**更接近理论定义：
- 优势：真正的连续时间建模
- 劣势：对于长时间序列和大批次，内存可能不足

## 建议使用场景

### 推荐使用模式0的情况：
- ✅ 序列很长（>200个观测点）
- ✅ 批次很大（>64）
- ✅ GPU内存有限
- ✅ 追求训练速度

### 推荐使用模式1的情况：
- ✅ 序列较短（<100个观测点）
- ✅ 追求理论正确性
- ✅ 研究连续时间动力学
- ✅ 需要插值能力
- ✅ 与Neural SDE论文对标

## 实验建议

可以尝试以下实验：

1. **消融实验**: 在相同数据集上分别训练两种模式，比较最终准确率
2. **一致性实验**: 检查两种模式的输出差异和预测一致性
3. **速度实验**: 比较训练时间和推理时间
4. **内存实验**: 监控GPU内存使用

## 参考资料

- **torchsde demo.ipynb**: 展示了标准的一次性求解方式
- **Neural SDE论文**: [arxiv.org/abs/2001.01328](https://arxiv.org/abs/2001.01328)
- **Latent ODE论文**: [arxiv.org/abs/1907.03907](https://arxiv.org/abs/1907.03907)

## 未来改进方向

1. **混合模式**: 短区间用一次性求解，长区间用逐步求解
2. **自适应模式**: 根据可用内存自动选择
3. **批量并行**: 在模式1中对多个样本并行求解（而非循环）
4. **全局约束**: 在模式0中添加长程一致性损失
