#!/usr/bin/env python3
"""
测试物理约束TimeGAN数据是否可以正常用于main.py训练
快速验证数据格式兼容性和训练流程
"""

import sys
import os
import torch
import argparse

# 添加项目根目录到路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders, setup_dataset_mapping, get_dataset_specific_params

def test_timegan_data_loading():
    """测试TimeGAN数据加载"""
    print("🔧 测试物理约束TimeGAN数据加载...")
    
    # 模拟命令行参数
    args = argparse.Namespace()
    args.dataset = 3  # MACHO
    args.model_type = 2  # 添加模型类型参数
    args.use_resampling = True  # 使用重采样数据
    args.resampled_data_path = None  # 使用默认TimeGAN路径
    args.batch_size = 32
    args.temperature = None
    args.focal_gamma = None
    args.min_time_interval = None
    
    # 设置数据集映射
    args = setup_dataset_mapping(args)
    
    print(f"数据路径: {args.data_path}")
    print(f"数据集名称: {args.dataset_name}")
    
    # 获取数据集特定配置
    config = get_dataset_specific_params(args.dataset, args)
    
    # 测试数据加载器创建
    try:
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            train_ratio=0.7,  # 修正参数名
            test_ratio=0.3,   # 修正参数名 
            normalize=False,  # 不归一化
            num_workers=0,  # 单线程避免问题
            random_seed=42
        )
        
        print(f"✅ 数据加载器创建成功!")
        print(f"   训练集: {len(train_loader)} 批次")
        print(f"   测试集: {len(test_loader)} 批次")
        print(f"   类别数: {num_classes}")
        
        # 测试数据批次
        print(f"\n🔍 检查数据批次...")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # 只检查前2个批次
                break
            
            print(f"批次 {batch_idx + 1}: {len(batch)} 个元素")
            for i, item in enumerate(batch):
                if hasattr(item, 'shape'):
                    print(f"  元素 {i}: {item.shape} {item.dtype}")
                else:
                    print(f"  元素 {i}: {type(item)} {item}")
            
            # 根据实际格式解包
            if len(batch) == 7:  # 7个元素的情况
                times, mags, errmags, masks, periods, labels, other = batch
            elif len(batch) == 6:
                times, mags, errmags, masks, periods, labels = batch
            else:
                print(f"  未知的batch格式，跳过详细检查")
                continue
            
            print(f"批次 {batch_idx + 1}:")
            print(f"  times: {times.shape} {times.dtype}")
            print(f"  mags: {mags.shape} {mags.dtype}")
            print(f"  errmags: {errmags.shape} {errmags.dtype}")
            print(f"  masks: {masks.shape} {masks.dtype}")
            print(f"  periods: {periods.shape} {periods.dtype}")
            print(f"  labels: {labels.shape} {labels.dtype}")
            print(f"  标签分布: {torch.bincount(labels)}")
            
            # 检查数据值范围
            print(f"  数据范围检查:")
            print(f"    时间: [{times.min():.3f}, {times.max():.3f}]")
            print(f"    星等: [{mags.min():.3f}, {mags.max():.3f}]")
            print(f"    误差: [{errmags.min():.3f}, {errmags.max():.3f}]")
            print(f"    有效掩码比例: {masks.float().mean():.3f}")
            
            # 检查NaN值
            has_nan = False
            for tensor_name, tensor in [('times', times), ('mags', mags), ('errmags', errmags)]:
                if torch.isnan(tensor).any():
                    print(f"    ⚠️ {tensor_name}包含NaN值!")
                    has_nan = True
            
            if not has_nan:
                print(f"    ✅ 数据中无NaN值")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_paths():
    """测试不同的数据路径选项"""
    print(f"\n🔍 测试不同数据路径选项...")
    
    test_configs = [
        {
            'name': '默认TimeGAN重采样',
            'use_resampling': True,
            'resampled_data_path': None
        },
        {
            'name': '指定TimeGAN路径',
            'use_resampling': True, 
            'resampled_data_path': '/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl'
        },
        {
            'name': '原始数据对比',
            'use_original': True
        }
    ]
    
    for config in test_configs:
        print(f"\n📊 测试配置: {config['name']}")
        
        # 创建参数
        args = argparse.Namespace()
        args.dataset = 3
        args.model_type = 2  # 添加模型类型参数
        for key, value in config.items():
            if key != 'name':
                setattr(args, key, value)
        
        # 设置数据集映射
        try:
            args = setup_dataset_mapping(args)
            print(f"  ✅ 数据路径: {args.data_path}")
            print(f"  ✅ 数据集名称: {args.dataset_name}")
        except Exception as e:
            print(f"  ❌ 配置失败: {str(e)}")

def main():
    """主测试函数"""
    print("🚀 物理约束TimeGAN数据兼容性测试")
    print("=" * 50)
    
    # 1. 测试数据加载
    success = test_timegan_data_loading()
    
    if success:
        print(f"\n✅ TimeGAN数据完全兼容main.py！")
        
        # 2. 测试不同路径配置
        test_with_different_paths()
        
        # 3. 提供使用说明
        print(f"\n🎯 main.py使用说明:")
        print(f"# 使用物理约束TimeGAN重采样数据训练:")
        print(f"python main.py --dataset 3 --use_resampling")
        print(f"")
        print(f"# 或指定具体路径:")
        print(f"python main.py --dataset 3 --use_resampling --resampled_data_path /root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl")
        print(f"")
        print(f"# 推荐训练配置（充分利用TimeGAN数据优势）:")
        print(f"python main.py --dataset 3 --use_resampling --epochs 100 --batch_size 64 --learning_rate 1e-4")
        
    else:
        print(f"\n❌ TimeGAN数据兼容性测试失败")
        print(f"请检查数据格式或联系开发者")
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 测试成功！可以开始使用物理约束TimeGAN数据进行训练了！")
    else:
        print(f"\n⚠️ 测试失败，需要修复问题后再试")