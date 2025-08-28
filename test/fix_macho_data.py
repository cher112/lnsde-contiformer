#!/usr/bin/env python3
"""
修复MACHO数据集的异常时间值
"""
import pickle
import numpy as np

def fix_macho_data():
    """修复MACHO数据集中的异常时间值"""
    print("=== 修复MACHO数据集 ===")
    
    # 加载原始数据
    with open('data/MACHO_folded_512.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 统计修复前的信息
    original_time_ranges = []
    fixed_data = []
    removed_samples = 0
    
    for i, sample in enumerate(data):
        times = sample['time']
        mask = sample.get('mask', np.ones(len(times), dtype=bool))
        
        # 只考虑有效时间点
        valid_times = times[mask]
        
        # 检查是否有异常时间值（如-1e+09）
        if len(valid_times) > 0:
            # 过滤掉明显异常的时间值
            normal_times = valid_times[valid_times > -1000]  # 过滤掉小于-1000的异常值
            
            if len(normal_times) > 10:  # 至少需要10个有效时间点
                time_range = normal_times.max() - normal_times.min()
                
                # 过滤掉时间跨度超过100天的异常样本
                if time_range <= 100:
                    # 创建新的修复样本
                    new_sample = sample.copy()
                    
                    # 只保留正常的时间点
                    valid_indices = np.where((times > -1000) & mask)[0]
                    if len(valid_indices) > 10:
                        # 重新构建时间序列，只保留前N个有效点
                        n_keep = min(len(valid_indices), 512)
                        keep_indices = valid_indices[:n_keep]
                        
                        new_times = np.full(512, times[keep_indices[-1]])  # 用最后时间点填充
                        new_times[:n_keep] = times[keep_indices]
                        
                        new_mask = np.zeros(512, dtype=bool)
                        new_mask[:n_keep] = True
                        
                        new_sample['time'] = new_times
                        new_sample['mask'] = new_mask
                        
                        # 同样处理其他特征
                        if 'magnitude' in sample:
                            new_mag = np.full(512, sample['magnitude'][keep_indices[-1]])
                            new_mag[:n_keep] = sample['magnitude'][keep_indices]
                            new_sample['magnitude'] = new_mag
                            
                        if 'error' in sample:
                            new_err = np.full(512, sample['error'][keep_indices[-1]])
                            new_err[:n_keep] = sample['error'][keep_indices]
                            new_sample['error'] = new_err
                        
                        fixed_data.append(new_sample)
                        original_time_ranges.append(time_range)
                    else:
                        removed_samples += 1
                        print(f"  移除样本{i}: 有效时间点不足 ({len(valid_indices)})")
                else:
                    removed_samples += 1
                    print(f"  移除样本{i}: 时间跨度过大 ({time_range:.1f}天)")
            else:
                removed_samples += 1
                print(f"  移除样本{i}: 正常时间点不足 ({len(normal_times)})")
        else:
            removed_samples += 1
            print(f"  移除样本{i}: 无有效时间点")
    
    print(f"\n修复结果:")
    print(f"保留样本数: {len(fixed_data)}")
    print(f"移除样本数: {removed_samples}")
    print(f"数据保留率: {len(fixed_data)/len(data)*100:.1f}%")
    
    # 统计修复后的时间分布
    if original_time_ranges:
        time_ranges = np.array(original_time_ranges)
        print(f"\n修复后时间跨度统计:")
        print(f"平均: {time_ranges.mean():.2f}天")
        print(f"中位数: {np.median(time_ranges):.2f}天")
        print(f"最大: {time_ranges.max():.2f}天")
        print(f"最小: {time_ranges.min():.2f}天")
        
        print(f"\n分布情况:")
        print(f"<1天: {np.sum(time_ranges < 1)} ({np.sum(time_ranges < 1)/len(time_ranges)*100:.1f}%)")
        print(f"1-2天: {np.sum((time_ranges >= 1) & (time_ranges < 2))} ({np.sum((time_ranges >= 1) & (time_ranges < 2))/len(time_ranges)*100:.1f}%)")
        print(f"2-5天: {np.sum((time_ranges >= 2) & (time_ranges < 5))} ({np.sum((time_ranges >= 2) & (time_ranges < 5))/len(time_ranges)*100:.1f}%)")
        print(f">5天: {np.sum(time_ranges >= 5)} ({np.sum(time_ranges >= 5)/len(time_ranges)*100:.1f}%)")
    
    # 保存修复后的数据
    output_path = 'data/MACHO_folded_512_fixed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print(f"\n修复后的数据已保存: {output_path}")
    
    return len(fixed_data)


def fix_asas_data():
    """修复ASAS数据集中的异常时间值"""
    print("=== 修复ASAS数据集 ===")
    
    # 加载原始数据
    with open('data/ASAS_folded_512.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 统计修复前的信息
    original_time_ranges = []
    fixed_data = []
    removed_samples = 0
    
    for i, sample in enumerate(data):
        times = sample['time']
        mask = sample.get('mask', np.ones(len(times), dtype=bool))
        
        # 只考虑有效时间点
        valid_times = times[mask]
        
        # 检查是否有异常时间值（如-1e+09）
        if len(valid_times) > 0:
            # 过滤掉明显异常的时间值
            normal_times = valid_times[valid_times > -1000]  # 过滤掉小于-1000的异常值
            
            if len(normal_times) > 10:  # 至少需要10个有效时间点
                time_range = normal_times.max() - normal_times.min()
                
                # 过滤掉时间跨度超过200天的异常样本（比MACHO稍松一些）
                if time_range <= 200:
                    # 创建新的修复样本
                    new_sample = sample.copy()
                    
                    # 只保留正常的时间点
                    valid_indices = np.where((times > -1000) & mask)[0]
                    if len(valid_indices) > 10:
                        # 重新构建时间序列，只保留前N个有效点
                        n_keep = min(len(valid_indices), 512)
                        keep_indices = valid_indices[:n_keep]
                        
                        new_times = np.full(512, times[keep_indices[-1]])  # 用最后时间点填充
                        new_times[:n_keep] = times[keep_indices]
                        
                        new_mask = np.zeros(512, dtype=bool)
                        new_mask[:n_keep] = True
                        
                        new_sample['time'] = new_times
                        new_sample['mask'] = new_mask
                        
                        # 同样处理其他特征
                        if 'magnitude' in sample:
                            new_mag = np.full(512, sample['magnitude'][keep_indices[-1]])
                            new_mag[:n_keep] = sample['magnitude'][keep_indices]
                            new_sample['magnitude'] = new_mag
                            
                        if 'error' in sample:
                            new_err = np.full(512, sample['error'][keep_indices[-1]])
                            new_err[:n_keep] = sample['error'][keep_indices]
                            new_sample['error'] = new_err
                        
                        fixed_data.append(new_sample)
                        original_time_ranges.append(time_range)
                    else:
                        removed_samples += 1
                        if i < 10:  # 只打印前10个
                            print(f"  移除样本{i}: 有效时间点不足 ({len(valid_indices)})")
                else:
                    removed_samples += 1
                    if i < 50:  # 只打印前50个
                        print(f"  移除样本{i}: 时间跨度过大 ({time_range:.1f}天)")
            else:
                removed_samples += 1
                if i < 10:
                    print(f"  移除样本{i}: 正常时间点不足 ({len(normal_times)})")
        else:
            removed_samples += 1
            if i < 10:
                print(f"  移除样本{i}: 无有效时间点")
    
    print(f"\n修复结果:")
    print(f"保留样本数: {len(fixed_data)}")
    print(f"移除样本数: {removed_samples}")
    print(f"数据保留率: {len(fixed_data)/len(data)*100:.1f}%")
    
    # 统计修复后的时间分布
    if original_time_ranges:
        time_ranges = np.array(original_time_ranges)
        print(f"\n修复后时间跨度统计:")
        print(f"平均: {time_ranges.mean():.2f}天")
        print(f"中位数: {np.median(time_ranges):.2f}天")
        print(f"最大: {time_ranges.max():.2f}天")
        print(f"最小: {time_ranges.min():.2f}天")
        
        print(f"\n分布情况:")
        print(f"<1天: {np.sum(time_ranges < 1)} ({np.sum(time_ranges < 1)/len(time_ranges)*100:.1f}%)")
        print(f"1-10天: {np.sum((time_ranges >= 1) & (time_ranges < 10))} ({np.sum((time_ranges >= 1) & (time_ranges < 10))/len(time_ranges)*100:.1f}%)")
        print(f"10-100天: {np.sum((time_ranges >= 10) & (time_ranges < 100))} ({np.sum((time_ranges >= 10) & (time_ranges < 100))/len(time_ranges)*100:.1f}%)")
        print(f">100天: {np.sum(time_ranges >= 100)} ({np.sum(time_ranges >= 100)/len(time_ranges)*100:.1f}%)")
    
    # 保存修复后的数据
    output_path = 'data/ASAS_folded_512_fixed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print(f"\n修复后的数据已保存: {output_path}")
    
    return len(fixed_data)


def fix_linear_data():
    """修复LINEAR数据集中的异常时间值"""
    print("=== 修复LINEAR数据集 ===")
    
    # 加载原始数据
    with open('data/LINEAR_folded_512.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 统计修复前的信息
    original_time_ranges = []
    fixed_data = []
    removed_samples = 0
    
    for i, sample in enumerate(data):
        times = sample['time']
        mask = sample.get('mask', np.ones(len(times), dtype=bool))
        
        # 只考虑有效时间点
        valid_times = times[mask]
        
        # 检查是否有异常时间值（如-1e+09）
        if len(valid_times) > 0:
            # 过滤掉明显异常的时间值
            normal_times = valid_times[valid_times > -1000]  # 过滤掉小于-1000的异常值
            
            if len(normal_times) > 10:  # 至少需要10个有效时间点
                time_range = normal_times.max() - normal_times.min()
                
                # LINEAR数据相对稀疏，允许稍大的时间跨度
                if time_range <= 300:  # 比ASAS稍松
                    # 创建新的修复样本
                    new_sample = sample.copy()
                    
                    # 只保留正常的时间点
                    valid_indices = np.where((times > -1000) & mask)[0]
                    if len(valid_indices) > 10:
                        # 重新构建时间序列，只保留前N个有效点
                        n_keep = min(len(valid_indices), 512)
                        keep_indices = valid_indices[:n_keep]
                        
                        new_times = np.full(512, times[keep_indices[-1]])  # 用最后时间点填充
                        new_times[:n_keep] = times[keep_indices]
                        
                        new_mask = np.zeros(512, dtype=bool)
                        new_mask[:n_keep] = True
                        
                        new_sample['time'] = new_times
                        new_sample['mask'] = new_mask
                        
                        # 同样处理其他特征
                        if 'magnitude' in sample:
                            new_mag = np.full(512, sample['magnitude'][keep_indices[-1]])
                            new_mag[:n_keep] = sample['magnitude'][keep_indices]
                            new_sample['magnitude'] = new_mag
                            
                        if 'error' in sample:
                            new_err = np.full(512, sample['error'][keep_indices[-1]])
                            new_err[:n_keep] = sample['error'][keep_indices]
                            new_sample['error'] = new_err
                        
                        fixed_data.append(new_sample)
                        original_time_ranges.append(time_range)
                    else:
                        removed_samples += 1
                        if removed_samples <= 10:  # 只打印前10个
                            print(f"  移除样本{i}: 有效时间点不足 ({len(valid_indices)})")
                else:
                    removed_samples += 1
                    if removed_samples <= 50:  # 只打印前50个
                        print(f"  移除样本{i}: 时间跨度过大 ({time_range:.1f}天)")
            else:
                removed_samples += 1
                if removed_samples <= 10:
                    print(f"  移除样本{i}: 正常时间点不足 ({len(normal_times)})")
        else:
            removed_samples += 1
            if removed_samples <= 10:
                print(f"  移除样本{i}: 无有效时间点")
    
    print(f"\n修复结果:")
    print(f"保留样本数: {len(fixed_data)}")
    print(f"移除样本数: {removed_samples}")
    print(f"数据保留率: {len(fixed_data)/len(data)*100:.1f}%")
    
    # 统计修复后的时间分布
    if original_time_ranges:
        time_ranges = np.array(original_time_ranges)
        print(f"\n修复后时间跨度统计:")
        print(f"平均: {time_ranges.mean():.2f}天")
        print(f"中位数: {np.median(time_ranges):.2f}天")
        print(f"最大: {time_ranges.max():.2f}天")
        print(f"最小: {time_ranges.min():.2f}天")
        
        print(f"\n分布情况:")
        print(f"<1天: {np.sum(time_ranges < 1)} ({np.sum(time_ranges < 1)/len(time_ranges)*100:.1f}%)")
        print(f"1-10天: {np.sum((time_ranges >= 1) & (time_ranges < 10))} ({np.sum((time_ranges >= 1) & (time_ranges < 10))/len(time_ranges)*100:.1f}%)")
        print(f"10-100天: {np.sum((time_ranges >= 10) & (time_ranges < 100))} ({np.sum((time_ranges >= 10) & (time_ranges < 100))/len(time_ranges)*100:.1f}%)")
        print(f">100天: {np.sum(time_ranges >= 100)} ({np.sum(time_ranges >= 100)/len(time_ranges)*100:.1f}%)")
    
    # 保存修复后的数据
    output_path = 'data/LINEAR_folded_512_fixed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print(f"\n修复后的数据已保存: {output_path}")
    
    return len(fixed_data)


if __name__ == "__main__":
    # 修复所有数据集
    print("开始修复所有数据集...")
    
    macho_count = fix_macho_data()
    print()
    
    asas_count = fix_asas_data()
    print()
    
    linear_count = fix_linear_data()
    
    print(f"\n=== 总结 ===")
    print(f"MACHO修复后样本数: {macho_count}")
    print(f"ASAS修复后样本数: {asas_count}")
    print(f"LINEAR修复后样本数: {linear_count}")
    print("所有数据集清洗完成！")