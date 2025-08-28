#!/usr/bin/env python3
"""
修复matplotlib字体问题
解决负号显示问题和中文字体配置
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

def fix_matplotlib_fonts():
    """修复matplotlib字体问题"""
    print("=== 修复matplotlib字体问题 ===")
    
    # 1. 清理matplotlib缓存
    try:
        cache_dir = os.path.expanduser('~/.cache/matplotlib')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("✓ 清理matplotlib缓存完成")
    except Exception as e:
        print(f"清理缓存失败: {e}")
    
    # 2. 刷新系统字体缓存
    try:
        import subprocess
        result = subprocess.run(['fc-cache', '-fv'], capture_output=True, text=True)
        print("✓ 刷新系统字体缓存完成")
    except:
        print("⚠️ 无法刷新字体缓存")
    
    # 3. 设置matplotlib配置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 关键：使用ASCII负号
    plt.rcParams['font.size'] = 10
    
    # 4. 重建字体缓存
    fm._load_fontmanager(try_read_cache=False)
    
    # 5. 测试字体配置
    print("\n=== 测试字体配置 ===")
    import numpy as np
    
    # 创建简单测试图
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(-5, 5, 100)
    y = x**2 - 10
    
    ax.plot(x, y, label='测试曲线 y=x²-10')
    ax.set_xlabel('X轴 (负数测试: -5 到 5)')
    ax.set_ylabel('Y轴 (负数测试)')
    ax.set_title('字体测试图表')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存测试图
    test_path = '/root/autodl-tmp/lnsde-contiformer/results/font_test.png'
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    try:
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        print(f"✓ 字体测试图保存成功: {test_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 保存测试图失败: {e}")
        plt.close()
    
    print("\n=== 当前字体配置 ===")
    print(f"字体族: {plt.rcParams['font.family']}")
    print(f"无衬线字体: {plt.rcParams['font.sans-serif']}")
    print(f"Unicode负号: {plt.rcParams['axes.unicode_minus']}")
    print(f"字体大小: {plt.rcParams['font.size']}")
    
    return True

if __name__ == "__main__":
    fix_matplotlib_fonts()
    print("\n字体修复完成！重新运行训练应该不会再出现负号问题。")