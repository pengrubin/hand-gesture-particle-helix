# 手势粒子螺旋项目功能指南 🎵🤚🎨

这个项目包含多个令人惊叹的音频可视化和手势控制功能。以下是所有主要功能的详细说明和使用方法。

---

## 🎯 项目概述

本项目是一个综合性的音频可视化和手势控制系统，包含以下核心功能：
- 手势控制粒子系统
- 实时音频可视化
- MIDI转3D建筑结构
- 多种音频分析工具

---

## 📋 功能列表

### 1. 🤚 手势控制粒子系统
**TouchDesigner + Python手势识别系统**

#### 功能描述
- 使用摄像头实时检测手势
- 根据手势控制3D粒子效果
- 支持开合手势、手掌旋转
- 粒子发射器位置跟随手部
- 实时音频谱分析同步

#### 运行方法
```bash
# 启动手势检测系统
python hand_gesture_detector.py

# 启动主应用（TouchDesigner集成）
python main_app.py

# 简化版本
python run.py
```

#### 核心文件
- `hand_gesture_detector.py` - MediaPipe手势识别
- `particle_system.py` - 粒子系统控制
- `audio_spectrum_analyzer.py` - 音频谱分析
- `main_app.py` - 主程序集成

---

### 2. 🎵 实时音频音高可视化
**最受欢迎的功能！实时显示音频的音高变化**

#### 功能描述
- 实时分析音频文件的音调高低
- 三个乐器分层显示（管风琴、鲁特琴、小提琴）
- 印章式画布效果，固定位置盖章
- 根据音调频率决定垂直位置
- 半透明叠加效果，无渐变变暗

#### 运行方法
```bash
# 进入音乐目录
cd music

# 运行音高可视化
python pitch_height_bars.py
```

#### 效果说明
- **蓝色管风琴**: 低音区 (0.0-0.4)
- **绿色鲁特琴**: 中音区 (0.3-0.7)  
- **红色小提琴**: 高音区 (0.6-1.0)
- **黄色虚线**: 固定印章位置
- **连续移动**: 画布向左匀速移动

#### 核心特点
- ✅ 固定印章位置，画布移动
- ✅ 音调决定垂直位置
- ✅ 半透明独立显示
- ✅ 无渐变变暗效果
- ✅ 持续盖章不停止

---

### 3. 🏗️ MIDI转3D建筑结构
**将音乐转换为三维建筑艺术**

#### 功能描述
- 将MIDI文件转换为3D方块建筑
- X轴=时间，Y轴=音高，Z轴=力度
- 支持多乐器，不同颜色
- 导出OBJ文件供3D软件使用
- matplotlib 3D预览

#### 运行方法
```bash
# 创建示例MIDI文件
python create_sample_midi.py
python create_complex_midi.py

# 转换MIDI为3D模型（演示模式）
python midi_to_3d_visualizer.py

# 转换指定MIDI文件
python midi_to_3d_visualizer.py your_music.mid

# 自定义输出文件
python midi_to_3d_visualizer.py song.mid --output song_3d.obj

# 跳过预览直接导出
python midi_to_3d_visualizer.py song.mid --no-preview

# 调整缩放参数
python midi_to_3d_visualizer.py song.mid --time-scale 8.0 --pitch-scale 1.0
```

#### 输出文件
- `*.obj` - 3D模型文件（可在Blender/Rhino打开）
- `midi_3d_preview.png` - 预览图
- `sample_music.mid` - 简单示例
- `complex_music.mid` - 复杂多声部示例

---

### 4. 🌊 多种音频可视化效果
**丰富的实时音频可视化选项**

#### 可用的可视化模式

##### 4.1 三线音高图
```bash
cd music
python three_lines_chart.py
```
- 三条线分别显示三个乐器
- 8秒滑动窗口
- 不同颜色区分乐器

##### 4.2 跳动柱状图
```bash
cd music
python bouncing_bars_chart.py
```
- 柱状图跳动效果
- MIDI特性增强
- 重力下降动画

##### 4.3 节拍推进柱状图
```bash
cd music
python beat_push_bars.py
```
- 每个节拍推进一格
- 新柱子跳动出现
- 透明度渐变

##### 4.4 纵向重叠柱状图
```bash
cd music
python stacked_beat_bars.py
```
- 三个乐器纵向重叠
- 不同宽度层次显示
- 历史透明度渐变

##### 4.5 平滑连续柱状图
```bash
cd music
python smooth_stacked_bars.py
```
- 平滑推进动画
- 连续流动感
- 缓动函数过渡

##### 4.6 印章画布柱状图
```bash
cd music
python stamp_canvas_bars.py
```
- 固定印章位置
- 画布连续左移
- 印章跳动动画

---

### 5. 🔧 辅助工具

#### 5.1 控制台音高分析
```bash
python console_pitch_analyzer.py
```
- 命令行界面音频分析
- 实时频谱显示
- 调试用途

#### 5.2 实时音频分析器
```bash
python real_audio_analyzer.py
```
- 实时麦克风输入分析
- 频谱可视化
- 音高检测

#### 5.3 音频播放启动器
```bash
python start_mp3_app.py
```
- 简化的MP3播放器
- 集成可视化效果

---

## 🎮 推荐使用流程

### 快速体验流程
```bash
# 1. 体验最受欢迎的音高可视化
cd music
python pitch_height_bars.py

# 2. 创建MIDI 3D建筑
python create_sample_midi.py
python midi_to_3d_visualizer.py

# 3. 尝试其他可视化效果
python bouncing_bars_chart.py
python smooth_stacked_bars.py
```

### 完整开发流程
```bash
# 1. 启动手势控制系统
python main_app.py

# 2. 在TouchDesigner中集成
# （需要TouchDesigner软件）

# 3. 实时音频分析
python real_audio_analyzer.py
```

---

## 📁 项目结构

```
hand-gesture-particle-helix/
├── 🤚 手势控制核心
│   ├── hand_gesture_detector.py
│   ├── particle_system.py
│   ├── main_app.py
│   └── run.py
├── 🎵 音频可视化系统
│   └── music/
│       ├── pitch_height_bars.py          ⭐ 最受欢迎
│       ├── three_lines_chart.py
│       ├── bouncing_bars_chart.py
│       ├── smooth_stacked_bars.py
│       └── stamp_canvas_bars.py
├── 🏗️ MIDI转3D系统
│   ├── midi_to_3d_visualizer.py          ⭐ 核心转换器
│   ├── create_sample_midi.py
│   ├── create_complex_midi.py
│   └── *.obj                             # 生成的3D模型
├── 🔧 辅助工具
│   ├── console_pitch_analyzer.py
│   ├── real_audio_analyzer.py
│   └── start_mp3_app.py
└── 📚 文档
    ├── README.md
    ├── README_MIDI_3D.md
    ├── FEATURES_GUIDE.md                 # 本文件
    └── CLAUDE.md
```

---

## 🎯 常用命令速查

### 最常用的3个功能

#### 1. 实时音高可视化（最受欢迎）⭐
```bash
cd music && python pitch_height_bars.py
```

#### 2. MIDI转3D建筑 ⭐
```bash
python midi_to_3d_visualizer.py
```

#### 3. 手势控制粒子 ⭐
```bash
python main_app.py
```

### 探索更多效果
```bash
# 跳动柱状图
cd music && python bouncing_bars_chart.py

# 平滑动画
cd music && python smooth_stacked_bars.py

# 印章效果
cd music && python stamp_canvas_bars.py

# 创建复杂MIDI
python create_complex_midi.py
```

---

## 🛠️ 依赖库

### 核心依赖
```bash
pip install opencv-python mediapipe numpy pandas matplotlib pygame
```

### 音频处理
```bash
pip install librosa pretty_midi mido
```

### 3D建模
```bash
pip install trimesh
```

### 完整安装
```bash
pip install -r requirements.txt
```

---

## 🎨 可视化效果预览

### 实时音高可视化
- 🔵 管风琴低音层
- 🟢 鲁特琴中音层  
- 🔴 小提琴高音层
- 🟡 固定印章位置
- 连续画布移动

### MIDI 3D建筑
- 低音 → 厚重基础建筑
- 中音 → 主体结构
- 高音 → 塔尖装饰
- 和弦 → 建筑群组

### 手势控制
- 开手 → 粒子发散
- 握拳 → 粒子收缩
- 旋转 → 3D变换
- 位置 → 发射器跟随

---

## ⚠️ 重要提醒

1. **音频文件路径**: 确保MP3文件在正确位置
2. **摄像头权限**: 手势识别需要摄像头访问权限
3. **TouchDesigner**: 完整手势控制需要TouchDesigner软件
4. **3D软件**: OBJ文件可用Blender（免费）、Rhino等打开
5. **性能**: 大型MIDI文件可能需要较长处理时间

---

## 🎉 享受音乐可视化的艺术之旅！

这个项目将音乐、科技和艺术完美结合，创造出独特的交互体验。无论是实时音高可视化的动态美感，还是MIDI转3D的建筑艺术，都将为你带来全新的音乐感受！

🎵 **记住最受欢迎的功能**: `cd music && python pitch_height_bars.py` 🎵