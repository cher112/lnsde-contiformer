"""
配置相关工具函数
"""

def get_dataset_specific_params(dataset_id, args):
    """获取数据集特定的配置参数"""
    # LINEAR 数据集配置
    if dataset_id == 2:  # LINEAR
        config = {
            'temperature': 0.8,  # v1.0原始参数
            'focal_gamma': 3.0,  # v1.0原始参数
            'enable_gradient_detach': True,  # 统一启用梯度断开，确保稳定性
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
    # 修复梯度断开参数逻辑 - 只有显式禁用时才关闭
    if hasattr(args, 'enable_gradient_detach') and not args.enable_gradient_detach:
        config['enable_gradient_detach'] = False
    if args.min_time_interval is not None:
        config['min_time_interval'] = args.min_time_interval
    
    print(f"温度参数: {config['temperature']}")
    print(f"Focal Gamma: {config['focal_gamma']}")
    print(f"梯度断开: {config['enable_gradient_detach']}")
    print(f"最小时间间隔: {config['min_time_interval']}")
    
    return config


def setup_sde_config(sde_config_id, args):
    """设置SDE求解参数"""
    # 针对不同SDE类型调整参数
    if args.model_type == 1:  # Langevin SDE
        # Langevin SDE需要更大的步长和更宽松的容差来避免求解失败
        sde_config_mapping = {
            1: {  # 准确率优先 - 但对Langevin更宽松
                'sde_method': 'euler',  # 使用更稳定的euler方法
                'dt': 0.02,  # 增大步长避免积分失败
                'rtol': 1e-4,  # 放宽容差
                'atol': 1e-5,
                'name': '准确率优先(Langevin优化)'
            },
            2: {  # 平衡
                'sde_method': 'euler',
                'dt': 0.05,
                'rtol': 1e-3,
                'atol': 1e-4,
                'name': '平衡(Langevin优化)'
            },
            3: {  # 时间优先
                'sde_method': 'euler',
                'dt': 0.1,
                'rtol': 1e-3,
                'atol': 1e-4,
                'name': '时间优先(Langevin优化)'
            },
            4: {  # 极速配置
                'sde_method': 'euler',
                'dt': 0.2,
                'rtol': 1e-2,
                'atol': 1e-3,
                'name': '极速配置(Langevin优化)'
            }
        }
    elif args.model_type == 3:  # Geometric SDE
        # Geometric SDE也需要优化参数避免数值不稳定
        sde_config_mapping = {
            1: {  # 准确率优先 - Geometric优化
                'sde_method': 'euler',  # 使用euler方法
                'dt': 0.01,  # 适中的步长
                'rtol': 1e-5,  # 较宽松的容差
                'atol': 1e-6,
                'name': '准确率优先(Geometric优化)'
            },
            2: {  # 平衡
                'sde_method': 'euler',
                'dt': 0.03,
                'rtol': 1e-4,
                'atol': 1e-5,
                'name': '平衡(Geometric优化)'
            },
            3: {  # 时间优先
                'sde_method': 'euler',
                'dt': 0.08,
                'rtol': 1e-3,
                'atol': 1e-4,
                'name': '时间优先(Geometric优化)'
            },
            4: {  # 极速配置
                'sde_method': 'euler',
                'dt': 0.15,
                'rtol': 1e-2,
                'atol': 1e-3,
                'name': '极速配置(Geometric优化)'
            }
        }
    else:  # 其他SDE类型保持原有配置
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
    # 数据集映射 - 根据数据版本参数选择
    if hasattr(args, 'use_resampling') and args.use_resampling:
        # 使用重采样数据集
        if hasattr(args, 'resampled_data_path') and args.resampled_data_path:
            # 使用指定的重采样数据路径
            if args.dataset == 3:  # MACHO
                dataset_mapping = {
                    3: (args.resampled_data_path, 'MACHO_TimeGAN')
                }
                print(f"📊 使用指定的物理约束TimeGAN重采样数据: {args.resampled_data_path}")
            else:
                # 其他数据集使用标准重采样数据
                dataset_mapping = {
                    1: ('/root/autodl-fs/lnsde-contiformer/data/ASAS_resampled.pkl', 'ASAS'),
                    2: ('/root/autodl-fs/lnsde-contiformer/data/LINEAR_resampled.pkl', 'LINEAR'),
                    3: (args.resampled_data_path, 'MACHO_TimeGAN')
                }
                print(f"📊 使用指定的重采样数据: {args.resampled_data_path}")
        else:
            # 自动选择重采样数据
            dataset_mapping = {
                1: ('/root/autodl-fs/lnsde-contiformer/data/ASAS_resampled.pkl', 'ASAS'),
                2: ('/root/autodl-fs/lnsde-contiformer/data/LINEAR_resampled.pkl', 'LINEAR'),
                3: ('/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl', 'MACHO_TimeGAN')  # 默认使用TimeGAN
            }
            print("📊 使用重采样数据集 - MACHO使用物理约束TimeGAN重采样")
    elif hasattr(args, 'use_enhanced') and args.use_enhanced:
        # 使用增强数据集
        dataset_mapping = {
            1: ('/root/autodl-fs/lnsde-contiformer/data/ASAS_enhanced.pkl', 'ASAS'),
            2: ('/root/autodl-fs/lnsde-contiformer/data/LINEAR_enhanced.pkl', 'LINEAR'),
            3: ('/root/autodl-fs/lnsde-contiformer/data/MACHO_enhanced.pkl', 'MACHO')
        }
        print("📊 使用增强数据集(enhanced) - 包含恢复样本和SMOTE生成")
    elif hasattr(args, 'use_original') and args.use_original:
        # 使用原始完整数据
        dataset_mapping = {
            1: ('/root/autodl-fs/lnsde-contiformer/data/ASAS_original.pkl', 'ASAS'),
            2: ('/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl', 'LINEAR'),
            3: ('/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl', 'MACHO')
        }
        print("📊 使用原始完整数据集(original)")
    else:
        # 默认使用fixed数据
        dataset_mapping = {
            1: ('/root/autodl-fs/lnsde-contiformer/data/ASAS_fixed.pkl', 'ASAS'),
            2: ('/root/autodl-fs/lnsde-contiformer/data/LINEAR_fixed.pkl', 'LINEAR'),
            3: ('/root/autodl-fs/lnsde-contiformer/data/MACHO_fixed.pkl', 'MACHO')
        }
        print("📊 使用修复后数据集(fixed)")
    
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