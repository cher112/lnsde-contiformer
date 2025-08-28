"""
配置相关工具函数
"""

def get_dataset_specific_params(dataset_id, args):
    """获取数据集特定的配置参数"""
    # LINEAR 数据集配置
    if dataset_id == 2:  # LINEAR
        config = {
            'temperature': 0.8,
            'focal_gamma': 3.0,
            'enable_gradient_detach': False,
            'min_time_interval': 0.005,  # LINEAR相对稀疏，中等优化
        }
        print("=== LINEAR 数据集配置 ===")
    
    # ASAS 数据集配置  
    elif dataset_id == 1:  # ASAS
        config = {
            'temperature': 1.0,
            'focal_gamma': 2.0,
            'enable_gradient_detach': True,  # 默认启用梯度断开，防止训练loss为NaN
            'min_time_interval': 0.01,   # ASAS时间点相对稀疏，保守优化
        }
        print("=== ASAS 数据集配置 ===")
    
    # MACHO 数据集配置
    elif dataset_id == 3:  # MACHO
        config = {
            'temperature': 1.5,
            'focal_gamma': 2.5,
            'enable_gradient_detach': True,  # 启用梯度断开加速训练
            'min_time_interval': 0.01,  # 跳过更多密集时间点，进一步加速
        }
        print("=== MACHO 数据集配置 ===")
    
    else:
        raise ValueError(f"未知数据集ID: {dataset_id}")
    
    # 使用命令行参数覆盖默认配置
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.focal_gamma is not None:
        config['focal_gamma'] = args.focal_gamma
    if args.enable_gradient_detach:
        config['enable_gradient_detach'] = True
    
    print(f"温度参数: {config['temperature']}")
    print(f"Focal Gamma: {config['focal_gamma']}")
    print(f"梯度断开: {config['enable_gradient_detach']}")
    print(f"最小时间间隔: {config['min_time_interval']}")
    
    return config


def setup_sde_config(sde_config_id, args):
    """设置SDE求解参数"""
    sde_config_mapping = {
        1: {  # 准确率优先 - 增加计算密度
            'sde_method': 'milstein',
            'dt': 0.005,  # 减小步长，增加积分步数
            'rtol': 1e-6,
            'atol': 1e-7,
            'name': '准确率优先'
        },
        2: {  # 平衡
            'sde_method': 'euler',
            'dt': 0.025,  # 适度减小步长
            'rtol': 1e-5,
            'atol': 1e-6,
            'name': '平衡'
        },
        3: {  # 时间优先
            'sde_method': 'euler',
            'dt': 0.1,
            'rtol': 1e-4,
            'atol': 1e-5,
            'name': '时间优先'
        },
        4: {  # 极速配置 - 快速测试用
            'sde_method': 'euler',
            'dt': 0.2,  # 更大步长
            'rtol': 1e-3,
            'atol': 1e-4,
            'name': '极速配置'
        }
    }
    
    config = sde_config_mapping[sde_config_id]
    
    # 更新args参数
    args.sde_method = config['sde_method']
    args.dt = config['dt'] 
    args.rtol = config['rtol']
    args.atol = config['atol']
    args.sde_config = sde_config_id
    
    print(f"SDE配置: {config['name']} (方法:{config['sde_method']}, dt:{config['dt']}, rtol:{config['rtol']}, atol:{config['atol']})")
    
    return config


def setup_dataset_mapping(args):
    """设置数据集映射和模型类型映射"""
    # 数据集映射 - 现在主数据文件已修复
    dataset_mapping = {
        1: ('data/ASAS_folded_512_fixed.pkl', 'ASAS'),    # 使用修复后的数据
        2: ('data/LINEAR_folded_512_fixed.pkl', 'LINEAR'), # 使用修复后的数据
        3: ('data/MACHO_folded_512_fixed.pkl', 'MACHO')   # 使用修复后的数据
    }
    
    # 模型类型映射
    model_type_mapping = {
        1: 'langevin',
        2: 'linear_noise', 
        3: 'geometric'
    }
    
    args.data_path, args.dataset_name = dataset_mapping[args.dataset]
    print(f"选择数据集: {args.dataset_name} ({args.data_path})")
    
    args.model_type = model_type_mapping[args.model_type]
    print(f"选择模型类型: {args.model_type}")
    
    return args