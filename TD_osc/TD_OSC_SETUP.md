# TouchDesigner OSC Counter Demo 设置指南

这是一个简单的 Python → TouchDesigner OSC 通信演示，用于测试项目可行性。

## 快速开始

### 1. Python 端设置

```bash
# 安装依赖
pip install -r requirements.txt

# 运行发送脚本
python simple_counter_demo.py
```

脚本会循环发送 1-100 的数字，间隔 0.5 秒。

### 2. TouchDesigner 端设置

#### 步骤 1：创建 OSC In DAT

1. 在 TouchDesigner 中，按 `Tab` 键打开 OP Create Dialog
2. 搜索并创建 `OSC In DAT`
3. 选中 OSC In DAT，查看参数面板（右侧）

#### 步骤 2：配置 OSC In DAT 参数

在参数面板中设置：
- **Network Port**: `9000`
- 保持其他参数为默认值

#### 步骤 3：创建显示文本

1. 创建一个 `Text TOP`（按 `Tab` → 搜索 `Text TOP`）
2. 在 Text TOP 参数面板的 **Text** 字段，输入：
   ```python
   op('oscin1').row(0)[2]
   ```
   （假设您的 OSC In DAT 名称是 `oscin1`）

#### 步骤 4：查看接收数据

运行 Python 脚本后，您应该能在：
1. **OSC In DAT** 中看到接收到的消息（每行包含：时间戳、地址、值）
2. **Text TOP** 中实时显示当前计数值

## 数据格式说明

- **OSC 地址**: `/counter`
- **数据类型**: 整数 (1-100)
- **发送频率**: 每 0.5 秒
- **循环模式**: 到达 100 后重置为 1

## 进阶用法

### 使用 CHOP 处理数字

如果您想在 TouchDesigner 中处理数字数据（而不仅仅是显示）：

1. 创建 `OSC In CHOP`（而非 DAT）
2. 设置参数：
   - **Network Port**: `9000`
   - **OSC Address**: `/counter`
3. 输出的 channel 会包含实时数值，可连接到任何参数

### 映射到可视化参数

```python
# 在 Text DAT 中创建表达式
# 例如：将计数器映射到旋转角度（0-360度）
rx = op('oscin1')[0,'val'] * 3.6
```

## 故障排查

### Python 脚本报错

**问题**: `ModuleNotFoundError: No module named 'pythonosc'`

**解决**:
```bash
pip install python-osc
```

### TouchDesigner 收不到数据

**问题**: OSC In DAT 没有显示任何数据

**检查清单**:
1. 确认 Python 脚本正在运行（终端中应该有输出）
2. 确认端口号匹配（Python 发送到 9000，TD 监听 9000）
3. 检查防火墙设置（可能阻止了本地通信）
4. 尝试重启 TouchDesigner

### 数据延迟或丢失

**问题**: 数据显示不流畅

**解决**:
- 检查 TouchDesigner 帧率（建议 60 FPS）
- 降低 Python 发送频率（修改 `time.sleep(0.5)` 为更大值）

## 项目可行性分析

### 测试通过的标志

如果您能在 TouchDesigner 中看到：
1. OSC In DAT 显示连续的消息流
2. 数字从 1 递增到 100 后循环
3. 延迟小于 100ms

则说明 **Python ↔ TouchDesigner OSC 通信可行**，可以继续开发：
- 手势数据传输
- 实时参数控制
- 双向通信（TouchDesigner → Python）

### 下一步建议

1. **单向通信测试**: 当前 demo（✓ 已完成）
2. **复杂数据测试**: 发送字典/数组数据
3. **双向通信测试**: Python 接收 TouchDesigner 的响应
4. **性能测试**: 测试高频数据发送（如 60 FPS）
5. **集成手势系统**: 将 MediaPipe 数据通过 OSC 发送

## 参考资料

- [TouchDesigner OSC 文档](https://docs.derivative.ca/OSC_In_DAT)
- [python-osc GitHub](https://github.com/attwad/python-osc)
- OSC 协议规范: http://opensoundcontrol.org/
