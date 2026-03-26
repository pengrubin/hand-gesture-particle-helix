# 纯Python手势控制粒子波浪效果系统

这是一个完全基于Python的实时手势识别和3D粒子波浪效果应用，**无需TouchDesigner**！

支持9种不同的波浪形状，通过手势实时控制切换！

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行应用
```bash
python run.py
```

就这么简单！

## 📋 系统要求

- **Python 3.7+**
- **摄像头** (内置或外置)
- **显卡支持OpenGL**
- **macOS/Windows/Linux** 都支持

## 🎮 操作控制

### 鼠标控制
- **左键拖拽**: 旋转3D视角
- **滚轮**: 缩放 (如果支持)

### 键盘控制
- **R**: 重置视角
- **C**: 显示/隐藏摄像头窗口
- **W**: 显示/隐藏线框球体  
- **I**: 显示/隐藏性能信息
- **S**: 手动切换波浪形状 ⭐ 新功能
- **1-5**: 调整粒子数量 (20%-100%)
- **ESC**: 退出应用

### 手势控制 → 波浪形状 ⭐ 新功能
| 手势 | 波浪形状 | 效果描述 |
|------|----------|----------|
| 🤛 握拳 | 锯齿波 | 尖锐的锯齿状波浪 |
| ☝️ 1个手指 | 正弦波 | 经典的正弦波浪线 |
| ✌️ 2个手指 | 双重波浪 | 两层叠加的波浪 |
| 🤟 3个手指 | 螺旋线 | 3D螺旋曲线 |
| 🖖 4个手指 | 心形曲线 | 浪漫的心形轨迹 |
| ✋ 张开手掌 | 3D螺旋 | 立体螺旋上升 |
| 🙌 双手同时 | 多条平行线 | 5条平行波浪线 |

### 动态参数控制
| 控制方式 | 效果 |
|----------|------|
| **手势强度** | 波浪幅度、频率、速度 |
| **左手位置** | 颜色色调、粒子大小 |
| **右手位置** | 颜色变化速度 |
| **双手距离** | 波浪复杂度 |

## 🏗️ 项目结构

```
touchdesigner/
├── run.py                      # 启动脚本 ⭐
├── main_app.py                 # 主应用程序
├── gesture_detector.py         # 手势识别系统
├── render_engine.py            # OpenGL渲染引擎
├── particle_sphere_system.py   # 粒子和球形系统
├── requirements.txt            # Python依赖
└── PYTHON_README.md           # 本文档
```

## 🔧 核心组件

### 1. 手势识别 (`gesture_detector.py`)
- 使用MediaPipe进行实时手部追踪
- 识别手指数量和手势类型
- 计算手部张开程度和中心位置
- 支持双手同时检测

### 2. 3D渲染引擎 (`render_engine.py`)
- 基于Pygame + PyOpenGL
- 支持粒子渲染和球形几何
- 相机控制和光照系统
- PBR材质和透明度

### 3. 粒子球形系统 (`particle_sphere_system.py`)
- 2000个动态粒子
- 球面发射模式
- 物理模拟 (重力、湍流、吸引力)
- 动态球形变形和旋转

### 4. 主应用 (`main_app.py`)
- 整合所有系统
- 实时交互和控制
- 性能监控
- 用户界面

## 🌊 9种波浪形状

### 1. 正弦波 (sine_wave) - 1个手指
经典的平滑正弦波浪线

### 2. 余弦波 (cosine_wave) - 可按S键切换
与正弦波相位不同的波浪

### 3. 双重波浪 (double_wave) - 2个手指  
两层不同频率的波浪叠加

### 4. 螺旋线 (spiral_line) - 3个手指
平面螺旋向外扩散

### 5. 锯齿波 (zigzag_line) - 握拳
尖锐的锯齿状波浪

### 6. 心形曲线 (heart_curve) - 4个手指
美丽的心形数学曲线

### 7. 无穷符号 (infinity_curve) - 可按S键切换
∞ 形状的数学曲线

### 8. 3D螺旋 (helix_3d) - 张开手掌
立体螺旋向上延伸

### 9. 多条平行线 (multiple_lines) - 双手
5条平行的波浪线

## 🎨 效果预览

应用会创建两个窗口：

1. **3D渲染窗口**: 显示粒子波浪效果和透明球体
2. **摄像头窗口**: 显示手势识别结果和当前形状信息 (可按C键切换)

## ⚡ 性能优化

### 如果遇到性能问题:

1. **降低粒子数量**: 按数字键1-3
2. **关闭线框显示**: 按W键
3. **隐藏摄像头窗口**: 按C键
4. **调整应用参数**:
   ```python
   # 在 main_app.py 中修改
   ParticleSphereSystem(max_particles=1000)  # 降低粒子数
   ```

### 推荐配置:
- **高性能**: 1500-2000 粒子
- **中等性能**: 800-1200 粒子  
- **低性能**: 300-600 粒子

## 🐛 故障排除

### 摄像头问题
```bash
# 检查摄像头权限 (macOS)
System Preferences → Security & Privacy → Camera → Allow Python

# 测试不同摄像头ID
# 在 gesture_detector.py 中修改
self.gesture_detector.start_camera(1)  # 尝试ID 1, 2, 3...
```

### 依赖安装问题
```bash
# 更新pip
pip install --upgrade pip

# 逐个安装依赖
pip install opencv-python mediapipe numpy pygame PyOpenGL

# macOS额外步骤
brew install python-tk  # 如果需要
```

### OpenGL问题
```bash
# Linux额外依赖
sudo apt-get install python3-opengl freeglut3-dev

# 检查OpenGL支持
python -c "from OpenGL.GL import *; print('OpenGL OK')"
```

### 手势识别不准确
- ✅ 确保充足光照
- ✅ 手部完全在镜头内
- ✅ 避免复杂背景
- ✅ 保持适当距离 (30-80cm)

## 🎯 自定义调整

### 修改粒子参数
```python
# 在 particle_sphere_system.py 中
self.params = {
    'emission_rate': 100,        # 发射速率
    'particle_life': 5.0,        # 粒子生命周期
    'base_radius': 2.0,          # 发射半径
    'velocity_scale': 1.0,       # 速度缩放
    'turbulence_strength': 0.0,  # 湍流强度
}
```

### 修改球形参数
```python
# 在 particle_sphere_system.py 中
self.params = {
    'base_radius': 2.0,              # 基础半径
    'deformation_strength': 0.0,     # 变形强度
    'rotation_speed': [1.0, 0.7, 0.5], # 旋转速度
    'pulsation_amplitude': 0.2,      # 脉动幅度
}
```

### 修改渲染参数
```python
# 在 render_engine.py 中
self.width = 1400           # 窗口宽度
self.height = 900           # 窗口高度
self.camera_pos = [0, 0, 8] # 相机位置
```

## 🚧 扩展功能建议

1. **更多波浪形状**: 
   - 玫瑰曲线 (Rose Curve)
   - 利萨茹曲线 (Lissajous Curve)  
   - 龙曲线 (Dragon Curve)
   - 分形树形状

2. **录制功能**: 保存波浪粒子效果为视频

3. **预设系统**: 保存/加载不同的波浪配置

4. **音响响应**: 根据音频频谱控制波浪参数

5. **多人支持**: 检测多人手势，不同人控制不同波浪

6. **实时波浪编辑**: 
   - 手动调节波浪参数
   - 自定义波浪公式输入
   - 波浪形状混合模式

## 📞 获取帮助

如果遇到问题:

1. 运行 `python run.py --help` 查看详细说明
2. 检查依赖安装: `python run.py` 会自动检查
3. 查看终端输出的错误信息
4. 确保摄像头权限已开启

## 🎉 享受创作！

现在你有了一个完全基于Python的实时手势控制粒子系统！

用你的手势创造美妙的3D粒子艺术吧！ ✨🎨