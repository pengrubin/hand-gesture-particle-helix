# TouchDesigner项目设置详细指南

本指南提供了在TouchDesigner中设置手势控制粒子球形效果的详细步骤。

## 第一步：创建基础网络

### 1. 视频输入网络

1. 创建 `Video In TOP`
   - 设置为默认摄像头 (device: 0)
   - 分辨率设为 640x480

2. 创建 `Text DAT` (名称: gesture_detector)
   - 复制 `hand_gesture_detector.py` 的内容到Text DAT中

3. 创建 `Execute DAT` (名称: camera_processor)
   - 在 frameStart 回调中添加：
   ```python
   def onFrameStart(frame):
       camera_top = op('videoIn')
       if camera_top.numpyArray() is not None:
           op('gesture_detector').module.initialize_detector()
           # 处理摄像头数据的代码在这里
   ```

### 2. 数据处理网络

1. 创建 `Text DAT` (名称: data_processor)
   - 复制 `td_data_processor.py` 的内容

2. 创建两个 `Table DAT`:
   - `particle_params` - 用于存储粒子参数
   - `sphere_params` - 用于存储球形参数

3. 创建 `Execute DAT` (名称: main_processor)
   - 设置 frameEnd 回调：
   ```python
   def onFrameEnd(frame):
       # 更新参数表
       op('data_processor').module.update_output_tables()
   ```

### 3. 粒子系统网络

1. 创建 `Text DAT` (名称: particle_controller)
   - 复制 `particle_system.py` 的内容

2. 创建 `Particle GPU TOP`
   - Reset Particles: Off
   - Max Particles: 2000
   - Point Size: 5
   - Life: 5.0

3. 创建 `GLSL TOP` (可选，用于自定义粒子渲染)
   - 使用 `particle_system.py` 中提供的着色器代码

### 4. 球形渲染网络

1. 创建 `Text DAT` (名称: sphere_controller)
   - 复制 `sphere_renderer.py` 的内容

2. 创建 `Geometry COMP`
   - 在Geometry COMP内部：
     - 添加 `Sphere SOP`
     - 设置 Rows: 32, Columns: 32

3. 创建 `Material`
   - 启用 PBR
   - 连接参数到 sphere_params Table DAT

4. 创建 `Light COMP` (定向光或点光源)

5. 创建 `Camera COMP`

6. 创建 `Render TOP`
   - 连接 Geometry, Material, Light, Camera

## 第二步：参数连接

### 连接粒子参数

使用以下表达式连接参数：

```python
# Particle GPU TOP参数
Max Particles: op('particle_params')['count', 1]
Birth Rate: op('particle_params')['emission_rate', 1]
Point Size: op('particle_params')['size', 1] * 10
Life: 5.0
```

### 连接球形参数

```python
# Sphere SOP参数
Scale: op('sphere_params')['radius', 1]

# Material参数
Metallic: op('sphere_params')['metallic', 1] if 'metallic' in op('sphere_params').col('parameter') else 0.3
Roughness: op('sphere_params')['roughness', 1] if 'roughness' in op('sphere_params').col('parameter') else 0.4
```

### 连接Transform参数

```python
# Transform参数 (Geometry COMP)
Rotate X: op('sphere_controller').module.get_shader_uniforms()['uRotation'][0]
Rotate Y: op('sphere_controller').module.get_shader_uniforms()['uRotation'][1]
Rotate Z: op('sphere_controller').module.get_shader_uniforms()['uRotation'][2]
```

## 第三步：Execute DAT主控制脚本

创建主控制 `Execute DAT`，整合所有功能：

```python
def onFrameStart(frame):
    """每帧开始时执行"""
    
    # 1. 获取摄像头数据
    video_in = op('videoIn')
    if video_in.numpyArray() is None:
        return
    
    # 2. 处理手势检测
    gesture_detector = op('gesture_detector')
    if hasattr(gesture_detector.module, 'process_frame'):
        gesture_data = gesture_detector.module.get_gesture_data()
        
        if gesture_data:
            # 3. 更新数据处理器
            data_processor = op('data_processor')
            data_processor.module.process_gesture_data(gesture_data)
            
            # 4. 获取处理后的参数
            particle_params = data_processor.module.particle_params
            sphere_params = data_processor.module.sphere_params
            
            # 5. 更新粒子系统
            particle_controller = op('particle_controller')
            particle_controller.module.update_particle_system(particle_params, 1/60.0)
            
            # 6. 更新球形渲染器
            sphere_controller = op('sphere_controller')
            sphere_controller.module.update_sphere_renderer(sphere_params, 1/60.0)

def onFrameEnd(frame):
    """每帧结束时执行"""
    
    # 更新参数表
    try:
        data_processor = op('data_processor')
        
        # 更新粒子参数表
        particle_table = op('particle_params')
        if particle_table and hasattr(data_processor.module, 'get_particle_table_data'):
            particle_table.clear()
            for row in data_processor.module.get_particle_table_data():
                particle_table.appendRow(row)
        
        # 更新球形参数表
        sphere_table = op('sphere_params')
        if sphere_table and hasattr(data_processor.module, 'get_sphere_table_data'):
            sphere_table.clear()
            for row in data_processor.module.get_sphere_table_data():
                sphere_table.appendRow(row)
                
    except Exception as e:
        print(f"Error updating tables: {e}")
```

## 第四步：优化和调试

### 性能监控

1. 开启 Performance Monitor (Alt + P)
2. 监控 GPU 内存使用
3. 检查帧率稳定性

### 调试工具

1. 创建 `Info DAT` 显示手势检测状态：
```python
# Info DAT表达式
gesture_data = op('gesture_detector').module.get_gesture_data()
return f"Hands: {gesture_data['hands_detected'] if gesture_data else 0}"
```

2. 创建监控面板：
   - 使用 `Panel Execute` 创建实时参数显示
   - 添加滑块控制关键参数

### 常见问题解决

1. **脚本错误**：
   - 检查Python缩进
   - 确保所有必要的模块都已导入
   - 使用 try-except 包装可能出错的代码

2. **性能问题**：
   - 减少粒子数量 (Max Particles)
   - 降低球形分辨率 (Rows/Columns)
   - 优化摄像头分辨率

3. **手势检测问题**：
   - 确保良好光照
   - 调整 MediaPipe 检测阈值
   - 检查摄像头是否正常工作

## 第五步：效果调节

### 粒子效果调节

在 `particle_system.py` 中调整：
- `max_particles`: 最大粒子数
- `particle_life`: 粒子生命周期
- `velocity_scale`: 速度缩放
- `turbulence_strength`: 湍流强度

### 球形效果调节

在 `sphere_renderer.py` 中调整：
- `sphere_resolution`: 球形细分度
- `deformation_strength`: 变形强度
- `pulsation_amplitude`: 脉动幅度
- `surface_noise_strength`: 表面噪声强度

### 视觉效果调节

- 调整材质的金属度和粗糙度
- 修改光照设置
- 调整摄像头位置和角度
- 设置背景颜色或纹理

## 保存和分享

1. 保存 TouchDesigner 项目文件 (.toe)
2. 导出组件为 .tox 文件以便复用
3. 创建预设保存常用的参数组合

这个设置指南应该能帮助你完整地在TouchDesigner中实现手势控制的粒子球形效果系统。