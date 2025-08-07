# 快速启动指南

## 第一步：环境准备

### 1. 安装Python依赖
```bash
pip install opencv-python mediapipe numpy
```

### 2. 确保设备
- 摄像头（内置或外置）
- TouchDesigner软件已安装

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
3. 设置 Extension 为 `.py`

#### B. 数据处理模块
1. 创建 `Text DAT`，重命名为 `data_processor`
2. 复制 `td_data_processor.py` 的全部内容
3. 设置 Extension 为 `.py`

#### C. 粒子系统模块
1. 创建 `Text DAT`，重命名为 `particle_controller`
2. 复制 `particle_system.py` 的全部内容
3. 设置 Extension 为 `.py`

#### D. 球形渲染模块
1. 创建 `Text DAT`，重命名为 `sphere_controller`
2. 复制 `sphere_renderer.py` 的全部内容
3. 设置 Extension 为 `.py`

### 3. 创建基础网络
按以下顺序创建操作符：

#### 视频输入
1. `Video In TOP` → 重命名为 `videoIn`
   - Device: 0 (默认摄像头)
   - Resolution: 640x480

#### 数据存储
1. `Table DAT` → 重命名为 `particle_params`
2. `Table DAT` → 重命名为 `sphere_params`

#### 粒子系统
1. `Particle GPU TOP` → 重命名为 `particles`
   - Max Particles: 2000
   - Reset Particles: Off
   - Point Size: 5

#### 球形渲染
1. `Geometry COMP` → 重命名为 `sphere_geo`
2. 在sphere_geo内部添加：
   - `Sphere SOP` (Rows: 32, Columns: 32)
3. `Camera COMP` → 重命名为 `cam1`
4. `Light COMP` → 重命名为 `light1`
5. `Render TOP` → 重命名为 `render1`

### 4. 创建主控制脚本
1. 创建 `Execute DAT` → 重命名为 `main_control`
2. 在 DAT Callbacks 中添加以下代码：

```python
def onFrameStart(frame):
    """每帧开始时执行"""
    
    # 获取摄像头数据
    video_in = op('videoIn')
    if video_in.numpyArray() is None:
        return
    
    try:
        # 初始化手势检测器（只在第一次执行）
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
            
            # 更新球形渲染器
            sphere_params = data_processor.module.sphere_params
            sphere_controller = op('sphere_controller')
            sphere_controller.module.update_sphere_renderer(sphere_params, 1/60.0)
            
    except Exception as e:
        print(f"Frame processing error: {e}")

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
在 `Particle GPU TOP` 中设置以下表达式：

```python
# Max Particles
int(op('particle_params').findCell('count', 'parameter')[1] if op('particle_params').findCell('count', 'parameter') else 1000)

# Birth Rate  
float(op('particle_params').findCell('emission_rate', 'parameter')[1] if op('particle_params').findCell('emission_rate', 'parameter') else 100)

# Point Size
float(op('particle_params').findCell('size', 'parameter')[1] if op('particle_params').findCell('size', 'parameter') else 1) * 10
```

### 2. 连接球形参数
在 `Sphere SOP` 中：

```python
# Scale
float(op('sphere_params').findCell('radius', 'parameter')[1] if op('sphere_params').findCell('radius', 'parameter') else 1)
```

### 3. 连接渲染
- 将 `sphere_geo` 连接到 `render1` 的 Geometry 输入
- 将 `cam1` 连接到 `render1` 的 Camera 输入  
- 将 `light1` 连接到 `render1` 的 Light 输入

## 第四步：启动系统

### 1. 检查摄像头
- 确保 `Video In TOP` 显示摄像头画面
- 如果没有画面，检查摄像头设备号或权限

### 2. 启动检测
1. 点击 TouchDesigner 的 Play 按钮 (或按空格键)
2. 将手伸到摄像头前
3. 观察 `particle_params` 和 `sphere_params` 表格的数据变化

### 3. 查看效果
- `particles` TOP 显示粒子效果
- `render1` TOP 显示最终的3D渲染结果

## 第五步：测试手势

尝试以下手势动作：

1. **握拳** → 粒子湍流增强，球体快速旋转
2. **张开手掌** → 粒子变大，球体轻微脉动
3. **伸出1-5个手指** → 不同的粒子和球体效果
4. **双手同时** → 粒子数量翻倍，球体变形
5. **改变手的位置** → 影响粒子颜色和球体材质

## 故障排除

### 常见问题：

1. **摄像头无画面**
   - 检查摄像头权限
   - 尝试改变 Video In TOP 的 Device 参数 (0, 1, 2...)

2. **脚本错误**
   - 确保所有Python脚本正确复制到Text DAT中
   - 检查 TextPort 窗口的错误信息

3. **手势检测不准确**
   - 确保光线充足
   - 手部完整在摄像头视野内
   - 调整摄像头角度

4. **性能问题**
   - 降低粒子数量 (Max Particles)
   - 减少球形分辨率 (Rows/Columns)

### 调试方法：

1. 创建 `Info DAT` 显示检测状态：
```python
gesture_data = op('gesture_detector').module.get_gesture_data() if hasattr(op('gesture_detector').module, 'get_gesture_data') else None
return f"Hands detected: {gesture_data.get('hands_detected', 0) if gesture_data else 0}"
```

2. 监控表格数据变化
3. 检查 TextPort 的错误输出

## 成功标志

当系统正常运行时，你会看到：
- 摄像头画面中显示手部关键点
- 参数表格实时更新数值
- 粒子效果随手势变化
- 3D球体根据手势变形和旋转

现在你可以开始体验手势控制的粒子球形效果了！