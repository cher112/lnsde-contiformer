# Claude 工作要求

## MCP 工具使用透明度

当使用 MCP (Model Context Protocol) 工具进行查询时，必须明确说明正在使用哪个 MCP 服务器。

### 要求：
- 使用 context7 查询库文档时，明确说明"正在使用 context7 MCP 查询..."
- 使用 filesystem MCP 访问文件时，说明"通过 filesystem MCP 访问..."
- 使用其他 MCP 工具时，同样需要明确标识

### 目的：
- 提高工作流程的透明度
- 让用户了解数据来源
- 便于调试和故障排除

### 示例：
```
正在使用 context7 MCP 查询 React 文档...
通过 filesystem MCP 读取项目文件...
```

## 脚本分类规则

### test/ 目录
用于存放测试和工具脚本：
- `merge_macho_logs.py` - 合并MACHO训练日志文件
- `test_model_shapes.py` - 模型形状测试
- `test_sde_solving.py` - SDE求解测试
- `usage_guide.py` - 使用指南

### visualization/ 目录  
用于存放数据可视化脚本：
- `macho_merged_visualization.py` - MACHO合并数据可视化
- 其他分析和可视化脚本

### 规则：
- 数据处理和合并脚本放在 `test/` 目录
- 可视化和图表生成脚本放在 `visualization/` 目录
- 保持功能模块化，每个脚本专注单一功能

## 图片输出和中文字体规范

### 图片输出目录
**所有可视化脚本生成的图片必须保存到**：`/root/autodl-tmp/lnsde-contiformer/results/pics/数据集名称/文件名.png`

**目录结构规范**：
```
results/pics/
├── ASAS/
│   └── lc.png
├── LINEAR/
│   └── lc.png
└── MACHO/
    └── lc.png
```

### 中文字体配置标准
**所有可视化脚本必须使用以下标准中文字体配置**：

#### 1. 安装系统字体
```bash
# 安装WenQuanYi中文字体包
apt-get install -y fonts-wqy-zenhei fonts-wqy-microhei
# 刷新字体缓存
fc-cache -fv
# 清理matplotlib缓存
rm -rf ~/.cache/matplotlib
```

#### 2. Python脚本配置
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

# 在脚本开始时调用
configure_chinese_font()
```

### 字体选择规则：
1. **WenQuanYi Zen Hei** - 首选中文字体
2. **WenQuanYi Micro Hei** - 备选中文字体
3. **SimHei** - 传统中文字体
4. **DejaVu Sans** - 英文后备字体

### 目的：
- 统一项目中所有图片的中文显示效果
- 确保图片集中管理和存储
- 提供字体后备机制，确保在不同环境下的兼容性

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.