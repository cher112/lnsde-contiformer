项目名称: PhysioPro核心架构: ContiFormer (Continuous-Time
  Transformer)开发者: Microsoft Research许可证: MIT License专业领域:
  不规则时间序列建模和生理信号处理

  核心技术架构

  1. ContiFormer 主体架构

  位置: physiopro/network/contiformer.py:244-333

  关键特性:
  - 连续时间建模: 专门处理不规则时间间隔的时序数据
  - ODE 集成: 使用常微分方程进行时间演化建模
  - 插值机制: 支持线性和三次样条插值

  2. 核心组件分析

  MultiHeadAttention (Lines 20-102)

  # 关键创新: 时间感知的注意力机制
  def forward(self, q, k, v, t, mask=None):
      q = self.w_qs(q, t)  # 查询投影，时间参数 t
      k = self.w_ks(k, t)  # 键投影，集成时间信息
      v = self.w_vs(v, t)  # 值投影，ODE线性层

  技术特点:
  - 时间感知权重: Q/K/V 投影都包含时间参数 t
  - ODE 线性层: 使用 ODELinear 和 InterpLinear 进行动态权重计算
  - 插值支持: 提供 interpolate 方法处理查询时间点

  Encoder (Lines 167-224)

  def temporal_enc(self, time):
      """连续时间位置编码"""
      result = time.unsqueeze(-1) / self.position_vec.to(time.device)
      result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
      result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
      return result

  核心功能:
  - 连续时间位置编码: 基于实际时间戳而非整数位置
  - 时间编码融合: 每层都添加时间编码信息
  - 多层堆叠: 支持可配置的编码器层数

  3. ODE 集成机制

  ODELinear (physiopro/module/linear.py:200-321)

  核心功能:
  - 时变ODE求解: 使用 TimeVariableODE 建模状态演化
  - 数值积分: 支持 RK4、Euler 等求解方法
  - 高斯积分: 提供多种数值积分策略

  InterpLinear (physiopro/module/linear.py:13-198)

  核心功能:
  - 连续插值: 使用 torchcde 库进行插值
  - 多项式支持: 线性和三次样条插值
  - 时间离散化: 高效的时间网格采样

  依赖关系分析

  核心依赖库

  torch==2.0.1          # PyTorch 深度学习框架
  torchcde==0.2.5        # 连续微分方程求解
  scipy>=1.10.1          # 科学计算库
  pandas>=2.0.3          # 数据处理
  sktime==0.21.0         # 时间序列工具
  tensorboard>=2.14.0    # 实验监控
  biosppy>=1.0.0         # 生物信号处理

  技术栈特色

  - torchcde: 核心的连续微分方程库，支持插值和数值积分
  - sktime: 专业时间序列处理框架
  - biosppy: 专门的生理信号处理工具

  函数功能详解

  主要接口函数

  1. ContiFormer.forward(x, t=None, mask=None)

  def forward(self, x, t=None, mask=None):
      if t is None:   # 处理规则时间序列
          t = torch.linspace(0, 1, x.shape[1]).to(x.device)
          t = t.unsqueeze(0).repeat(x.shape[0], 1)

      if self.linear_type is not None:
          x = self.linear(x)

      enc_output = self.encoder(x, t, mask)
      return enc_output, enc_output[:, -1, :]

  功能:
  - 自动时间生成（如果未提供）
  - 特征维度映射
  - 编码器处理
  - 返回序列输出和最终隐状态

  2. MultiHeadAttention.interpolate(q, k, v, t, qt, mask=None)

  功能: 在指定查询时间 qt 进行插值预测

  3. temporal_enc(time)

  功能: 生成连续时间的位置编码，基于正弦和余弦函数

  配置参数体系

  ODE 相关参数

  actfn_ode: sigmoid           # ODE 激活函数
  layer_type_ode: concatnorm   # ODE 层类型
  atol_ode: 1e-1              # 绝对容忍度
  rtol_ode: 1e-1              # 相对容忍度
  method_ode: rk4             # 数值求解方法
  regularize: False           # L2 正则化
  approximate_method: bilinear # 近似积分方法
  interpolate_ode: cubic      # 插值方法

  模型结构参数

  d_model: 32      # 隐藏维度
  n_layers: 1      # 编码器层数
  n_head: 4        # 注意力头数
  d_k: 8           # 键维度
  d_v: 8           # 值维度
  d_inner: 128     # FFN 隐藏维度

  应用场景和案例

  1. 不规则时间序列分类

  python -m physiopro.entry.train
  docs/configs/contiformer_mask_classification.yml \
      --data.mask_ratio 0.3 --data.name Heartbeat

  2. 时间点过程建模

  python -m physiopro.entry.train docs/configs/contiformer_tpp.yml
