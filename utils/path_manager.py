#!/usr/bin/env python3
"""
结果路径管理工具
统一管理日志和图片输出路径
"""
import os
from datetime import datetime

def get_timestamp_path(base_dir, dataset_name, create_dirs=True):
    """
    获取基于时间戳的结果路径
    
    格式: base_dir/20250828/DATASET/1352/
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")  # 20250828
    time_str = now.strftime("%H%M")    # 1352
    
    # 构建路径
    result_path = os.path.join(base_dir, date_str, dataset_name, time_str)
    
    if create_dirs:
        os.makedirs(result_path, exist_ok=True)
        # 同时创建子目录
        os.makedirs(os.path.join(result_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(result_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(result_path, "plots"), exist_ok=True)
    
    return result_path

def get_log_path(dataset_name, model_type, sde_config, base_dir="./results"):
    """获取日志文件路径"""
    timestamp_dir = get_timestamp_path(base_dir, dataset_name)
    log_filename = f"{dataset_name}_{model_type}_config{sde_config}.log"
    return os.path.join(timestamp_dir, "logs", log_filename)

def get_model_path(dataset_name, model_type, model_name, base_dir="./results"):
    """获取模型文件路径"""
    timestamp_dir = get_timestamp_path(base_dir, dataset_name)
    model_filename = f"{dataset_name}_{model_type}_{model_name}.pth"
    return os.path.join(timestamp_dir, "models", model_filename)

def get_plot_path(dataset_name, plot_name, base_dir="./results"):
    """获取图片文件路径"""
    timestamp_dir = get_timestamp_path(base_dir, dataset_name)
    plot_filename = f"{dataset_name}_{plot_name}.png"
    return os.path.join(timestamp_dir, "plots", plot_filename)

def get_latest_session_path(base_dir, dataset_name):
    """获取最新的训练会话路径"""
    date_str = datetime.now().strftime("%Y%m%d")
    dataset_path = os.path.join(base_dir, date_str, dataset_name)
    
    if not os.path.exists(dataset_path):
        return None
        
    # 获取最新的时间戳目录
    sessions = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()]
    
    if not sessions:
        return None
        
    latest_session = max(sessions)
    return os.path.join(dataset_path, latest_session)

def migrate_old_files():
    """迁移现有文件到新的目录结构"""
    print("=== 迁移现有文件到新目录结构 ===")
    
    base_results = "./results"
    old_structures = [
        "checkpoints/20250828",
        "logs/20250828", 
        "pics"
    ]
    
    migrated_count = 0
    
    for old_path in old_structures:
        full_old_path = os.path.join(base_results, old_path)
        if not os.path.exists(full_old_path):
            continue
            
        print(f"处理目录: {full_old_path}")
        
        # 遍历数据集目录
        for item in os.listdir(full_old_path):
            item_path = os.path.join(full_old_path, item)
            
            if os.path.isdir(item_path) and item in ['ASAS', 'LINEAR', 'MACHO']:
                dataset_name = item
                print(f"  处理数据集: {dataset_name}")
                
                # 创建新的时间戳目录 (使用当前时间)
                new_base_path = get_timestamp_path(base_results, dataset_name)
                
                # 迁移文件
                for file_name in os.listdir(item_path):
                    old_file_path = os.path.join(item_path, file_name)
                    
                    if os.path.isfile(old_file_path):
                        # 确定目标子目录
                        if file_name.endswith('.log'):
                            target_dir = os.path.join(new_base_path, "logs")
                        elif file_name.endswith('.pth'):
                            target_dir = os.path.join(new_base_path, "models")
                        elif file_name.endswith('.png'):
                            target_dir = os.path.join(new_base_path, "plots")
                        else:
                            target_dir = new_base_path
                            
                        os.makedirs(target_dir, exist_ok=True)
                        new_file_path = os.path.join(target_dir, file_name)
                        
                        # 移动文件
                        import shutil
                        try:
                            shutil.move(old_file_path, new_file_path)
                            migrated_count += 1
                            print(f"    迁移: {file_name} -> {target_dir}")
                        except Exception as e:
                            print(f"    ⚠️  迁移失败 {file_name}: {e}")
    
    print(f"\n迁移完成，共处理 {migrated_count} 个文件")
    return migrated_count

if __name__ == "__main__":
    # 演示新路径格式
    print("=== 新路径格式演示 ===")
    
    for dataset in ['ASAS', 'LINEAR', 'MACHO']:
        print(f"\n{dataset} 数据集路径:")
        print(f"  日志: {get_log_path(dataset, 'linear_noise', 3)}")
        print(f"  模型: {get_model_path(dataset, 'linear_noise', 'best')}")
        print(f"  图片: {get_plot_path(dataset, 'training_curves')}")
    
    # 执行迁移
    migrate_old_files()