#!/usr/bin/env python3
"""
生成与原始数据格式完全兼容的重采样数据
修复所有数据格式问题，让main.py无需修改即可使用
"""

import sys
import os
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import generate_compatible_resampled_data
import multiprocessing as mp


def generate_dataset_resampling(dataset_info):
    """为单个数据集生成重采样数据"""
    dataset_name, input_path, output_path = dataset_info
    
    print(f"\n{'='*60}")
    print(f"开始处理 {dataset_name} 数据集")
    print(f"{'='*60}")
    
    try:
        result_path = generate_compatible_resampled_data(
            original_data_path=input_path,
            output_path=output_path,
            sampling_strategy='balanced',  # 完全平衡
            synthesis_mode='hybrid',       # 混合模式
            apply_enn=False,              # 不过度清理，保持样本数量
            random_state=535411460
        )
        
        print(f"✅ {dataset_name} 重采样数据生成完成!")
        print(f"   输出路径: {result_path}")
        
        return True, dataset_name, result_path
        
    except Exception as e:
        print(f"❌ {dataset_name} 重采样失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, dataset_name, str(e)


def main():
    """主函数 - 生成所有数据集的兼容重采样数据"""
    
    # 定义数据集路径
    datasets = [
        ('ASAS', '/root/autodl-fs/lnsde-contiformer/data/ASAS_fixed.pkl', 
         '/root/autodl-fs/lnsde-contiformer/data/ASAS_resampled.pkl'),
        ('LINEAR', '/root/autodl-fs/lnsde-contiformer/data/LINEAR_fixed.pkl', 
         '/root/autodl-fs/lnsde-contiformer/data/LINEAR_resampled.pkl'),
        ('MACHO', '/root/autodl-fs/lnsde-contiformer/data/MACHO_fixed.pkl', 
         '/root/autodl-fs/lnsde-contiformer/data/MACHO_resampled.pkl')
    ]
    
    print("🚀 开始生成兼容格式的重采样数据...")
    print(f"处理 {len(datasets)} 个数据集")
    
    # 检查输入文件是否存在
    for dataset_name, input_path, output_path in datasets:
        if not os.path.exists(input_path):
            print(f"❌ 找不到 {dataset_name} 原始数据: {input_path}")
            return
    
    print("✅ 所有输入文件检查通过")
    
    # 串行处理（避免内存问题）
    results = []
    for dataset_info in datasets:
        result = generate_dataset_resampling(dataset_info)
        results.append(result)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("重采样生成结果汇总")
    print(f"{'='*60}")
    
    success_count = 0
    for success, dataset_name, info in results:
        if success:
            print(f"✅ {dataset_name}: 成功")
            print(f"   路径: {info}")
            success_count += 1
        else:
            print(f"❌ {dataset_name}: 失败")
            print(f"   错误: {info}")
    
    print(f"\n总结: {success_count}/{len(datasets)} 个数据集生成成功")
    
    if success_count == len(datasets):
        print("🎉 所有数据集重采样完成! 现在可以直接使用main.py训练了")
        print("\n使用方法:")
        print("  python main.py --dataset 1 --use_resampling  # ASAS重采样")
        print("  python main.py --dataset 2 --use_resampling  # LINEAR重采样") 
        print("  python main.py --dataset 3 --use_resampling  # MACHO重采样")
    else:
        print("⚠️ 部分数据集生成失败，请检查错误信息")


if __name__ == "__main__":
    main()