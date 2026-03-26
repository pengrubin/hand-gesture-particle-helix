# macOS TouchDesigner 快速启动指南

## 第一步：环境准备

### 1. 安装Python依赖
```bash
pip install opencv-python mediapipe numpy
```

### 2. 确保设备
- 摄像头（内置或外置）
- TouchDesigner for macOS已安装

## 第二步：TouchDesigner项目设置

### 1. 创建新项目
1. 打开TouchDesigner
2. 创建新项目 (File → New)
3. 保存项目为 `gesture_particle_sphere.toe`

### 2. 导入脚本模块
按以下步骤创建Text DAT并导入脚本：

#### A. 手势检测模块
1. 创建 `Text DAT`，重命名为 `gesture_detector`
2. 复制 `hand_gesture_detector.py` 的全部内容到这个Text DAT
3. 右键 → Parameters → Extension → 设置为 `.py`

#### B. 数据处理模块
1. 创建 `Text DAT`，重命名为 `data_processor`
2. 复制 `td_data_processor.py` 的全部内容
3. 设置 Extension 为 `.py`

#### C. 粒子系统模块（简化版）
1. 创建 `Text DAT`，重命名为 `particle_controller`
2. 复制以下简化的粒子控制代码：

```python
"""
macOS TouchDesigner 粒子控制脚本
使用CHOP和SOP替代GPU粒子系统
"""

import math
import random

class MacOSParticleController:
    def __init__(self):
        self.particle_count = 500
        self.positions = []
        self.colors = []
        self.current_frame = 0
        
        # 控制参数
        self.params = {
            'radius': 2.0,
            'velocity': 1.0,
            'size': 1.0,
            'color_hue': 0.5,
            'turbulence': 0.0
        }
        
        self.initialize_particles()
    
    def initialize_particles(self):
        """初始化粒子位置"""
        self.positions = []
        for i in range(self.particle_count):
            # 在球面上生成随机位置
            phi = random.uniform(0, 2 * math.pi)
            theta = random.uniform(0, math.pi)
            radius = self.params['radius']
            
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            
            self.positions.extend([x, y, z])
    
    def update_particles(self, gesture_params, dt=0.016):
        """更新粒子系统参数"""
        if gesture_params:
            self.params.update({
                'radius': gesture_params.get('radius', 2.0),
                'velocity': gesture_params.get('velocity', 1.0),
                'size': gesture_params.get('size', 1.0),
                'color_hue': gesture_params.get('color_hue', 0.5),
                'turbulence': gesture_params.get('turbulence', 0.0)
            })
        
        self.current_frame += 1
        time = self.current_frame * dt
        
        # 更新粒子位置
        for i in range(0, len(self.positions), 3):
            if i + 2 < len(self.positions):
                # 添加轻微的动画效果
                noise = math.sin(time + i * 0.01) * 0.1
                self.positions[i] += noise * self.params['turbulence']
                self.positions[i+1] += math.cos(time + i * 0.02) * 0.1 * self.params['turbulence']
    
    def get_positions_for_sop(self):
        """获取位置数据用于SOP"""
        return self.positions
    
    def get_particle_table(self):
        """获取粒子参数表格"""
        return [
            ['parameter', 'value'],
            ['count', self.particle_count],
            ['radius', self.params['radius']],
            ['velocity', self.params['velocity']],
            ['size', self.params['size']],
            ['color_hue', self.params['color_hue']],
            ['turbulence', self.params['turbulence']]
        ]

# 全局实例
if not hasattr(op, 'particle_controller'):
    op.particle_controller = MacOSParticleController()

# TouchDesigner接口函数
def update_particle_system(gesture_params, dt=0.016):
    """更新粒子系统"""
    op.particle_controller.update_particles(gesture_params, dt)

def get_particle_positions():
    """获取粒子位置"""
    return op.particle_controller.get_positions_for_sop()

def get_particle_count():
    """获取粒子数量"""
    return op.particle_controller.particle_count
```

#### D. 球形渲染模块
1. 创建 `Text DAT`，重命名为 `sphere_controller`
2. 复制 `sphere_renderer.py` 的全部内容
3. 设置 Extension 为 `.py`

### 3. 创建macOS适配的网络

#### 视频输入
1. `Video In TOP` → 重命名为 `videoIn`
   - Device: 0 (默认摄像头)
   - Resolution: 640x480

#### 数据存储
1. `Table DAT` → 重命名为 `particle_params`
2. `Table DAT` → 重命名为 `sphere_params`

#### 粒子系统（使用SOP替代GPU）
1. `Add SOP` → 重命名为 `particles_sop`
   - Points: 500
   - Point Positions: Manual
2. `Instance2 COMP` → 重命名为 `particle_instances`
3. 在Instance2 COMP内部添加：
   - `Sphere SOP` (小球体作为粒子)
   - Scale设为0.02

#### 球形主体
1. `Geometry COMP` → 重命名为 `sphere_geo`
2. 在sphere_geo内部添加：
   - `Sphere SOP` (Rows: 32, Columns: 32)
3. `Camera COMP` → 重命名为 `cam1`
4. `Light COMP` → 重命名为 `light1`
5. `Render TOP` → 重命名为 `render1`

### 4. 创建macOS主控制脚本
1. 创建 `Execute DAT` → 重命名为 `main_control`
2. 在 Callbacks 中添加：

```python
def onFrameStart(frame):
    """每帧开始时执行"""
    
    # 获取摄像头数据
    video_in = op('videoIn')
    if video_in.numpyArray() is None:
        return
    
    try:
        # 初始化手势检测器
        gesture_detector = op('gesture_detector')
        if not hasattr(gesture_detector.module, 'detector'):
            gesture_detector.module.initialize_detector()
        
        # 处理当前帧
        frame_data = video_in.numpyArray()
        processed_frame = gesture_detector.module.process_camera_frame(frame_data)
        
        # 获取手势数据
        gesture_data = gesture_detector.module.get_gesture_data()
        
        if gesture_data:
            # 处理手势数据
            data_processor = op('data_processor')
            data_processor.module.process_gesture_data(gesture_data)
            
            # 更新粒子系统
            particle_params = data_processor.module.particle_params
            particle_controller = op('particle_controller')
            particle_controller.module.update_particle_system(particle_params, 1/60.0)
            
            # 更新粒子SOP位置
            update_particle_positions()
            
            # 更新球形渲染器
            sphere_params = data_processor.module.sphere_params
            sphere_controller = op('sphere_controller')
            sphere_controller.module.update_sphere_renderer(sphere_params, 1/60.0)
            
    except Exception as e:
        print(f"Frame processing error: {e}")

def update_particle_positions():
    """更新粒子SOP位置"""
    try:
        particle_controller = op('particle_controller')
        positions = particle_controller.module.get_particle_positions()
        
        # 更新Add SOP的点位置
        particles_sop = op('particles_sop')
        if particles_sop and positions:
            # 清除现有点
            particles_sop.clear()
            
            # 添加新的粒子位置
            point_count = min(len(positions) // 3, 500)
            for i in range(point_count):
                idx = i * 3
                if idx + 2 < len(positions):
                    x, y, z = positions[idx], positions[idx+1], positions[idx+2]
                    particles_sop.addPoint([x, y, z])
                    
    except Exception as e:
        print(f"Particle position update error: {e}")

def onFrameEnd(frame):
    """每帧结束时执行"""
    
    try:
        data_processor = op('data_processor')
        
        # 更新粒子参数表
        particle_table = op('particle_params')
        if hasattr(data_processor.module, 'get_particle_table_data'):
            particle_table.clear()
            for row in data_processor.module.get_particle_table_data():
                particle_table.appendRow(row)
        
        # 更新球形参数表
        sphere_table = op('sphere_params')
        if hasattr(data_processor.module, 'get_sphere_table_data'):
            sphere_table.clear()
            for row in data_processor.module.get_sphere_table_data():
                sphere_table.appendRow(row)
                
    except Exception as e:
        print(f"Table update error: {e}")
```

## 第三步：连接参数

### 1. 连接粒子参数
在 `Add SOP` 中：
- Points: `int(op('particle_params').findCell('count', 'parameter')[1] if op('particle_params').findCell('count', 'parameter') else 500)`

### 2. 连接球形参数
在 `Sphere SOP` 中：
- Scale: `float(op('sphere_params').findCell('radius', 'parameter')[1] if op('sphere_params').findCell('radius', 'parameter') else 1)`

### 3. 连接Instance2 COMP
- 将 `particles_sop` 连接到 `particle_instances` 的 SOP 输入
- 设置 Instance2 的 Scale 参数连接到粒子大小参数

### 4. 连接最终渲染
- 将 `sphere_geo` 和 `particle_instances` 都连接到 `render1`
- 连接 `cam1` (Camera) 和 `light1` (Light) 到 `render1`

## 第四步：启动系统

### 1. 检查摄像头
- 确保 `Video In TOP` 显示摄像头画面
- 如果没有画面，检查摄像头权限：
  - System Preferences → Security & Privacy → Camera → 允许TouchDesigner

### 2. 启动检测
1. 点击 TouchDesigner 的 Play 按钮
2. 将手伸到摄像头前
3. 观察参数表格的数据变化

### 3. 查看效果
- `particles_sop` 显示粒子点云
- `render1` TOP 显示最终的3D渲染结果

## macOS特别注意事项

1. **Metal渲染器**: TouchDesigner on macOS使用Metal，性能可能与Windows版本不同
2. **粒子数量**: 建议从少量粒子开始测试（200-500个）
3. **摄像头权限**: 确保在系统设置中授予TouchDesigner摄像头权限
4. **性能监控**: 使用 Performance Monitor (Window → Performance Monitor) 监控帧率

## 故障排除

### macOS特有问题：

1. **摄像头权限被拒绝**
   ```
   System Preferences → Security & Privacy → Camera → TouchDesigner ✓
   ```

2. **Metal渲染错误**
   - 检查显卡驱动是否最新
   - 尝试降低渲染复杂度

3. **Python模块导入失败**
   - 确保使用TouchDesigner内置的Python环境
   - 检查pip install是否安装到正确的Python环境

现在你可以在macOS上启动手势控制粒子球形效果了！