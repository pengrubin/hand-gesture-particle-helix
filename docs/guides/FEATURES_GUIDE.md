# 手势粒子螺旋项目功能指南 🎵🤚🎨

本项目包含两类功能：摄像头手势控制的音乐与粒子螺旋系统，以及 MIDI 转 3D 建筑结构的离线工具。以下是主要功能的说明和使用方法。

---

## 🎯 项目概述

- 手势控制粒子螺旋系统（主应用）
- MIDI 转 3D 建筑结构（`tools/midi/`）
- 浏览器简易版（`fugue-web/`，线上地址 https://hand.pengrubin.com/）

---

## 📋 功能列表

### 1. 🤚 手势控制粒子螺旋系统
**Python + MediaPipe 手势识别，pygame + OpenGL 渲染**

#### 功能描述
- 使用摄像头实时检测手势
- 1 / 2 / 3 指、张开手掌分别控制小提琴、鲁特琴、管风琴三个声部的开关
- 根据手势控制 3D 螺旋结构的扭转和连接桥
- 实时音频谱分析同步螺旋半径

#### 运行方法
```bash
# 带依赖和摄像头检查的启动器
python run.py

# 直接启动主应用
python main_app.py
```

#### 核心文件
- `gesture_detector.py` - MediaPipe 手势识别
- `particle_sphere_system.py` - 螺旋与粒子系统
- `realistic_audio_manager.py` - 三轨音频控制
- `audio_spectrum_analyzer.py` - 音频谱分析
- `main_app.py` - 主程序集成

音频文件放在 `data/audio/`，文件名见 `realistic_audio_manager.py`。

---

### 2. 🏗️ MIDI 转 3D 建筑结构
**将音乐转换为三维建筑艺术**

#### 功能描述
- 将 MIDI 文件转换为 3D 方块建筑
- X 轴 = 时间，Y 轴 = 音高，Z 轴 = 力度
- 支持多乐器，不同颜色
- 导出 OBJ 文件供 3D 软件使用
- matplotlib 3D 预览

#### 运行方法
```bash
# 创建示例 MIDI 文件
python tools/midi/create_sample_midi.py
python tools/midi/create_complex_midi.py

# 转换 MIDI 为 3D 模型（演示模式）
python tools/midi/midi_to_3d_visualizer.py

# 转换指定 MIDI 文件
python tools/midi/midi_to_3d_visualizer.py your_music.mid

# 自定义输出文件
python tools/midi/midi_to_3d_visualizer.py song.mid --output song_3d.obj

# 跳过预览直接导出
python tools/midi/midi_to_3d_visualizer.py song.mid --no-preview

# 调整缩放参数
python tools/midi/midi_to_3d_visualizer.py song.mid --time-scale 8.0 --pitch-scale 1.0
```

#### 输出文件
- `*.obj` - 3D 模型文件（可在 Blender / Rhino 打开），示例见 `data/models/`
- `sample_music.mid`、`complex_music.mid` - 示例 MIDI，见 `data/midi/`

---

## 📁 项目结构

```
hand-gesture-particle-helix/
├── 🤚 手势控制核心
│   ├── gesture_detector.py
│   ├── particle_sphere_system.py
│   ├── realistic_audio_manager.py
│   ├── main_app.py
│   └── run.py
├── 🏗️ MIDI 转 3D 工具
│   ├── tools/midi/midi_to_3d_visualizer.py   ⭐ 核心转换器
│   ├── tools/midi/create_sample_midi.py
│   └── tools/midi/create_complex_midi.py
├── 🌐 浏览器简易版
│   └── fugue-web/
├── 🎵 数据
│   ├── data/audio/    # 三个声部的 MP3（不入库）
│   ├── data/midi/
│   └── data/models/
└── 📚 文档
    ├── README.md
    ├── docs/
    └── CLAUDE.md
```

---

## 🛠️ 依赖库

### 主应用
```bash
pip install -r requirements.txt
```

### MIDI 转 3D 工具
```bash
pip install pretty_midi trimesh matplotlib
```

---

## ⚠️ 重要提醒

1. **音频文件路径**: 确保 MP3 文件在 `data/audio/`
2. **摄像头权限**: 手势识别需要摄像头访问权限
3. **TouchDesigner**: 合作者的 TD 视觉方案通过 `td_parametric_integration.py` 接入，仓库内不包含 .toe 文件
4. **3D 软件**: OBJ 文件可用 Blender（免费）、Rhino 等打开
5. **性能**: 大型 MIDI 文件可能需要较长处理时间
