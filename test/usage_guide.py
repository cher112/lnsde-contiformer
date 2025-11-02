"""
使用示例和命令说明
"""

print("=== Neural SDE + ContiFormer 使用指南 ===\\n")

usage_info = """
🚀 训练命令示例:

1. 训练 Langevin SDE 模型:
   python main.py --model_type langevin --data_path /autodl-fs/data/lnsde-contiformer/ASAS_folded_512.pkl --dataset_name ASAS --epochs 50

2. 训练 Linear Noise SDE 模型:  
   python main.py --model_type linear_noise --data_path /autodl-fs/data/lnsde-contiformer/LINEAR_folded_512.pkl --dataset_name LINEAR --epochs 50

3. 训练 Geometric SDE 模型:
   python main.py --model_type geometric --data_path /autodl-fs/data/lnsde-contiformer/MACHO_folded_512.pkl --dataset_name MACHO --epochs 50

📋 重要参数说明:
   --model_type {langevin|linear_noise|geometric}  # 模型类型选择
   --data_path PATH                                # 数据文件路径  
   --batch_size 32                                # 批大小
   --learning_rate 1e-3                           # 学习率
   --hidden_channels 64                           # SDE隐藏维度
   --contiformer_dim 128                          # ContiFormer维度
   --n_heads 8                                    # 注意力头数
   --n_layers 4                                   # 编码器层数
   --sde_method euler                             # SDE求解方法
   --device auto                                  # 计算设备

🎯 模型特点对比:

1. Langevin-type SDE:
   - 数学形式: dY_t = -∇U(Y_t)dt + σ(t,Y_t)dW_t
   - 特点: 基于势能函数，物理意义明确
   - 适用场景: 具有平衡态的动态系统

2. Linear Noise SDE:
   - 数学形式: dY_t = f(t,Y_t)dt + (A(t) + B(t)Y_t)dW_t
   - 特点: 线性噪声结构，数值稳定性好
   - 适用场景: 线性增长/衰减过程

3. Geometric SDE:
   - 数学形式: dY_t/Y_t = μ(t,Y_t)dt + σ(t,Y_t)dW_t
   - 特点: 保持解的正定性，适合比例变化
   - 适用场景: 几何增长过程，金融建模

🔧 项目结构:
   ├── main.py                    # 统一训练入口
   ├── models/                    # 模型定义
   │   ├── __init__.py
   │   ├── base_sde.py           # SDE基类
   │   ├── langevin_sde.py       # Langevin SDE
   │   ├── linear_noise_sde.py   # Linear Noise SDE
   │   ├── geometric_sde.py      # Geometric SDE
   │   └── contiformer.py        # ContiFormer模块
   ├── utils/                     # 工具函数
   │   ├── __init__.py
   │   ├── dataloader.py         # 数据加载
   │   ├── preprocessing.py      # 预处理
   │   └── trainer.py            # 训练工具
   ├── test/                      # 测试文件
   │   ├── test_model_shapes.py  # 形状测试
   │   ├── test_sde_solving.py   # SDE求解测试
   │   └── usage_guide.py        # 使用指南
   ├── data/                      # 数据文件
   ├── checkpoints/               # 模型检查点
   └── logs/                      # 训练日志

📊 支持的数据格式:
   - ASAS: 3099个样本，5个类别
   - LINEAR: 5181个样本，5个类别  
   - MACHO: 2097个样本，7个类别
   - 数据格式: Lombscaler折叠光变曲线
   - 特征: time, mag, errmag
   - 自动处理不等长序列和类别不平衡

🧪 运行测试:
   cd test
   python test_model_shapes.py    # 测试模型形状
   python test_sde_solving.py     # 测试SDE求解

"""

print(usage_info)
print("✅ Neural SDE + ContiFormer 架构已完成!")
print("🎯 使用上述命令开始训练你的模型!")