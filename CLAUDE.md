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

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.