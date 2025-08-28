"""
分析MACHO数据集时间特征以确定最优时间间隔参数
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def configure_chinese_font():
    """配置中文字体显示"""
    import matplotlib.font_manager as fm
    
    # 尝试添加系统字体
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    except:
        pass
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

def analyze_macho_time_intervals():
    """分析MACHO数据集的时间间隔特征"""
    print('=== MACHO数据集时间间隔分析 ===')
    
    try:
        # 加载数据
        print("1. 加载MACHO数据集...")
        with open('data/MACHO_folded_512.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"   数据集包含 {len(data)} 个样本")
        
        # 分析时间间隔
        print("\n2. 分析时间间隔分布...")
        all_intervals = []
        valid_samples = 0
        
        for i, sample in enumerate(data[:100]):  # 分析前100个样本
            if 'time' in sample and 'mask' in sample:
                times = sample['time']
                mask = sample['mask'].astype(bool)
                
                # 获取有效时间点
                valid_times = times[mask]
                if len(valid_times) > 1:
                    # 按时间排序
                    sorted_times = np.sort(valid_times)
                    # 计算相邻时间间隔
                    intervals = np.diff(sorted_times)
                    # 过滤掉太小的间隔
                    intervals = intervals[intervals > 1e-8]
                    
                    if len(intervals) > 0:
                        all_intervals.extend(intervals)
                        valid_samples += 1
                        
                        if i < 5:  # 显示前5个样本的详细信息
                            print(f"   样本{i}: 有效时间点{len(valid_times)}, 间隔范围[{intervals.min():.6f}, {intervals.max():.6f}]")
        
        print(f"\n   分析了{valid_samples}个有效样本")
        
        if len(all_intervals) == 0:
            print("   ❌ 未找到有效的时间间隔")
            return
        
        all_intervals = np.array(all_intervals)
        
        # 统计分析
        print("\n3. 时间间隔统计:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("   百分位数分析:")
        for p in percentiles:
            val = np.percentile(all_intervals, p)
            print(f"   {p:2}%: {val:.6f}")
        
        print(f"\n   最小间隔: {all_intervals.min():.6f}")
        print(f"   最大间隔: {all_intervals.max():.6f}")
        print(f"   平均间隔: {all_intervals.mean():.6f}")
        print(f"   中位间隔: {np.median(all_intervals):.6f}")
        print(f"   标准差: {all_intervals.std():.6f}")
        
        # 建议的时间间隔阈值
        print("\n4. 推荐的最小时间间隔参数:")
        
        # 方法1: 基于百分位数
        p90_interval = np.percentile(all_intervals, 90)
        p75_interval = np.percentile(all_intervals, 75)
        p50_interval = np.percentile(all_intervals, 50)
        
        print(f"   保守策略(90%分位数): {p90_interval:.6f}")
        print(f"   平衡策略(75%分位数): {p75_interval:.6f}")
        print(f"   激进策略(50%分位数): {p50_interval:.6f}")
        
        # 方法2: 基于SDE求解频率控制
        total_intervals = len(all_intervals)
        intervals_less_than_005 = np.sum(all_intervals < 0.005)
        intervals_less_than_001 = np.sum(all_intervals < 0.001)
        intervals_less_than_01 = np.sum(all_intervals < 0.01)
        intervals_less_than_05 = np.sum(all_intervals < 0.05)
        
        print(f"\n5. SDE求解频率分析:")
        print(f"   < 0.001: {intervals_less_than_001} ({intervals_less_than_001/total_intervals*100:.1f}%)")
        print(f"   < 0.005: {intervals_less_than_005} ({intervals_less_than_005/total_intervals*100:.1f}%)")
        print(f"   < 0.01:  {intervals_less_than_01} ({intervals_less_than_01/total_intervals*100:.1f}%)")
        print(f"   < 0.05:  {intervals_less_than_05} ({intervals_less_than_05/total_intervals*100:.1f}%)")
        
        # 可视化
        print("\n6. 生成可视化图表...")
        configure_chinese_font()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MACHO数据集时间间隔分析', fontsize=16)
        
        # 直方图
        axes[0,0].hist(all_intervals, bins=100, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('时间间隔')
        axes[0,0].set_ylabel('频数')
        axes[0,0].set_title('时间间隔分布直方图')
        axes[0,0].set_yscale('log')
        
        # 对数直方图
        log_intervals = np.log10(all_intervals[all_intervals > 0])
        axes[0,1].hist(log_intervals, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('log10(时间间隔)')
        axes[0,1].set_ylabel('频数')
        axes[0,1].set_title('对数时间间隔分布')
        
        # 累积分布
        sorted_intervals = np.sort(all_intervals)
        cumulative = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
        axes[1,0].plot(sorted_intervals, cumulative)
        axes[1,0].set_xlabel('时间间隔')
        axes[1,0].set_ylabel('累积概率')
        axes[1,0].set_title('时间间隔累积分布')
        axes[1,0].set_xscale('log')
        axes[1,0].grid(True)
        
        # 添加候选阈值线
        thresholds = [0.001, 0.005, 0.01, 0.05, p50_interval, p75_interval, p90_interval]
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
        labels = ['0.001', '0.005', '0.01', '0.05', f'P50({p50_interval:.3f})', f'P75({p75_interval:.3f})', f'P90({p90_interval:.3f})']
        
        for threshold, color, label in zip(thresholds, colors, labels):
            if threshold <= sorted_intervals.max():
                axes[1,0].axvline(threshold, color=color, linestyle='--', alpha=0.7, label=label)
        axes[1,0].legend()
        
        # SDE求解频率对比
        threshold_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        skip_ratios = []
        for thresh in threshold_values:
            skipped = np.sum(all_intervals < thresh)
            skip_ratio = skipped / total_intervals * 100
            skip_ratios.append(skip_ratio)
        
        axes[1,1].bar(range(len(threshold_values)), skip_ratios, alpha=0.7)
        axes[1,1].set_xlabel('最小时间间隔阈值')
        axes[1,1].set_ylabel('跳过的SDE求解比例 (%)')
        axes[1,1].set_title('不同阈值下的SDE求解跳过比例')
        axes[1,1].set_xticks(range(len(threshold_values)))
        axes[1,1].set_xticklabels([f'{t:.3f}' for t in threshold_values])
        
        # 添加数值标签
        for i, v in enumerate(skip_ratios):
            axes[1,1].text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = '/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO/macho_time_interval_analysis.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   图表已保存至: {output_path}")
        
        # 最终推荐
        print(f"\n=== 最终推荐 ===")
        print(f"MACHO数据集最小时间间隔参数建议:")
        print(f"  - 高性能模式: {p90_interval:.6f} (跳过{np.sum(all_intervals < p90_interval)/total_intervals*100:.1f}%的密集点)")
        print(f"  - 平衡模式: {p75_interval:.6f} (跳过{np.sum(all_intervals < p75_interval)/total_intervals*100:.1f}%的密集点)")
        print(f"  - 保精度模式: {p50_interval:.6f} (跳过{np.sum(all_intervals < p50_interval)/total_intervals*100:.1f}%的密集点)")
        
        recommended_value = p75_interval
        print(f"\n🎯 推荐使用: {recommended_value:.6f} (平衡性能和精度)")
        
        return recommended_value
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    analyze_macho_time_intervals()