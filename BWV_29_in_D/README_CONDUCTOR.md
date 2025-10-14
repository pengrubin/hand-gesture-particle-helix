# BWV_29_in_D 指挥家控制系统

实时摄像头手势识别控制7声部音频播放系统

## 系统概述

这是一个基于计算机视觉和手势识别的音乐指挥控制系统，能够实时检测指挥家的手势并控制巴赫BWV 29号康塔塔的7个声部播放。系统使用MediaPipe进行手势识别，OpenCV处理摄像头输入，Pygame管理多音轨音频播放。

## 核心功能

### 🎼 音乐控制
- **7声部音频同步播放**: Tromba I+II+III, Violins, Viola, Oboe I, Continuo, Organo obligato, Timpani
- **实时音量控制**: 基于手势强度动态调整每个声部音量
- **平滑音频过渡**: 避免突兀的音量变化
- **断点续播**: 用户离开后可从暂停位置继续

### 👋 手势识别
- **区域控制**: 7个预定义区域对应不同声部
- **中央控制**: 手掌张开控制所有声部
- **精确指向**: 手指指向特定区域控制单一声部
- **专业指挥手势**: 识别forte、legato、stop等指挥动作
- **双手协调**: 支持双手指挥的复合控制

### 👤 用户检测
- **自动启动**: 检测到用户出现自动开始播放
- **智能暂停**: 用户离开超过设定时间自动暂停
- **置信度评估**: 实时评估用户存在的可信度
- **状态持久化**: 保持用户偏好和控制状态

### 🖥️ 用户界面
- **实时视频显示**: 摄像头画面与手势标注叠加
- **区域可视化**: 7个声部区域的边界和激活状态
- **性能监控**: FPS、延迟、CPU使用率等指标
- **音频电平显示**: 实时显示各声部音量
- **键盘快捷键**: 丰富的键盘控制选项

## 文件结构

```
BWV_29_in_D/
├── conductor_control.py          # 主控制程序
├── audio_controller.py           # 音频控制器
├── test_conductor_system.py      # 系统测试脚本
├── run_conductor.py              # 启动器脚本
├── README_CONDUCTOR.md           # 说明文档
├── *.mp3                         # 9个音频文件
└── conductor_control.log         # 运行日志
```

## 安装要求

### Python 版本
- Python 3.8 或更高版本

### 必需依赖
```bash
pip install opencv-python mediapipe numpy pygame
```

### 系统要求
- **摄像头**: USB摄像头或内置摄像头
- **内存**: 至少 2GB 可用内存
- **处理器**: 支持多线程的现代CPU
- **音频**: 支持多通道音频输出的声卡

### 音频文件
系统需要以下9个MP3文件（需要用户提供）：
- Tromba_I_in_D.mp3
- Tromba_II_in_D.mp3
- Tromba_III_in_D.mp3
- Violins_in_D.mp3
- Viola_in_D.mp3
- Oboe_I_in_D.mp3
- Continuo_in_D.mp3
- Organo_obligato_in_D.mp3
- Timpani_in_D.mp3

## 使用方法

### 快速启动
```bash
cd BWV_29_in_D
python run_conductor.py
```

启动器会自动检查系统要求并引导设置。

### 手动启动
```bash
python conductor_control.py [选项]
```

#### 命令行选项
- `--camera-id 0`: 摄像头设备ID (默认: 0)
- `--audio-dir ./`: 音频文件目录 (默认: 当前目录)
- `--fullscreen`: 全屏模式启动
- `--log-level INFO`: 日志级别 (DEBUG/INFO/WARNING/ERROR)

### 系统测试
```bash
python test_conductor_system.py
```

运行完整的系统测试，验证所有组件正常工作。

## 控制说明

### 键盘快捷键
| 按键 | 功能 |
|------|------|
| ESC | 退出程序 |
| SPACE | 暂停/恢复播放 |
| R | 重置手势检测器 |
| D | 切换调试信息显示 |
| F | 切换全屏模式 |
| H | 切换帮助信息显示 |
| 1-9 | 直接设置所有声部音量 |
| 0 | 所有声部静音 |

### 手势控制

#### 基本手势
- **手掌张开**: 增加音量，张开程度对应音量大小
- **握拳**: 减少音量或停止播放
- **指向**: 食指指向特定区域控制单一声部

#### 专业指挥手势
- **Forte手势**: 手掌完全张开，快速动作
- **Legato手势**: 平滑的手臂动作
- **指挥棒握持**: 食指+中指伸出，其他收拢
- **停止手势**: 紧握拳头超过0.5秒

#### 区域控制
屏幕分为7个区域，每个对应一个声部：
```
[1-Tromba] [2-Violins] [3-Viola  ]
[4-Oboe  ] [  Central ] [5-Continuo]
[6-Organo] [           ] [7-Timpani]
```

### 控制模式

#### 1. 中央控制模式
- 手放在屏幕中央区域
- 手掌张开程度控制所有声部音量
- 适合整体音量控制

#### 2. 区域选择模式
- 手指指向特定区域
- 只控制该区域对应的声部
- 适合精确的声部控制

#### 3. 双手协调模式
- 左右手同时使用
- 双手距离影响音量范围
- 主手控制表情，副手控制力度

## 性能优化

### 建议设置
- **摄像头分辨率**: 1280x720 (平衡性能和精度)
- **帧率**: 30 FPS
- **缓冲大小**: 512 samples (低延迟)
- **手势历史**: 10帧 (平滑性)

### 性能监控
系统实时显示：
- **FPS**: 当前帧率
- **延迟**: 手势识别延迟
- **CPU使用率**: 处理器占用
- **内存使用**: 内存占用
- **音频状态**: 播放状态和音量

### 故障排除

#### 常见问题

**摄像头无法打开**
```bash
# 检查摄像头设备
ls /dev/video*  # Linux
# 或尝试不同的camera-id
python conductor_control.py --camera-id 1
```

**音频播放问题**
```bash
# 检查音频文件
python test_conductor_system.py
# 或降低音频质量
```

**手势识别不准确**
- 确保良好的光照条件
- 避免复杂的背景
- 保持手在摄像头范围内
- 调整手势敏感度

**性能问题**
- 降低摄像头分辨率
- 减少同时处理的手数量
- 关闭调试信息显示
- 增加帧跳跃间隔

#### 调试模式
```bash
python conductor_control.py --log-level DEBUG
```

查看详细的调试信息和性能统计。

## 技术架构

### 核心组件

#### 1. ConductorControl (主控制器)
- 系统状态管理
- 组件协调
- 用户界面渲染
- 事件处理

#### 2. CameraManager (摄像头管理)
- 摄像头初始化和配置
- 视频帧读取和处理
- FPS计算和性能监控

#### 3. HandGestureDetector (手势检测)
- MediaPipe手部检测
- 专业指挥手势识别
- 区域检测和激活
- 手势稳定性过滤

#### 4. AudioController (音频控制)
- 多音轨音频播放
- 实时音量控制
- 平滑音频过渡
- 性能优化

#### 5. UIRenderer (界面渲染)
- 实时视频叠加
- 区域可视化
- 状态信息显示
- 性能指标显示

### 数据流

```
摄像头 → 手势检测 → 区域映射 → 音频控制 → 声部播放
   ↓         ↓         ↓         ↓         ↓
  视频显示 ← 界面渲染 ← 状态管理 ← 性能监控 ← 用户反馈
```

## 开发和扩展

### 配置系统
```python
config = {
    'frame_skip': 2,                    # 帧跳跃
    'max_num_hands': 4,                 # 最大手数
    'min_detection_confidence': 0.7,    # 检测置信度
    'gesture_history_size': 10,         # 手势历史大小
    'position_smoothing': 0.8           # 位置平滑系数
}

detector = HandGestureDetector(config)
```

### 自定义手势
```python
def custom_gesture_detector(landmarks):
    # 实现自定义手势识别逻辑
    return gesture_type, confidence

# 注册到系统
detector.register_custom_gesture(custom_gesture_detector)
```

### 音频效果扩展
```python
def custom_audio_effect(audio_data, gesture_data):
    # 实现自定义音频效果
    return modified_audio_data

# 添加到音频处理链
audio_controller.add_effect(custom_audio_effect)
```

## 版本历史

### v1.0.0 (当前)
- 完整的7声部音频控制
- 专业指挥手势识别
- 实时用户界面
- 性能监控和优化
- 完整的测试套件

## 许可证

本项目为教育和研究目的开发。音频文件版权归原作者所有。

## 贡献

欢迎提交问题报告和功能建议。请确保：
1. 提供详细的错误描述
2. 包含系统环境信息
3. 提供复现步骤
4. 遵循代码规范

## 联系信息

技术支持和问题反馈请通过GitHub Issues提交。

---

**享受指挥音乐的乐趣！** 🎼✋🎵