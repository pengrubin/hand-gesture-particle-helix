#!/usr/bin/env python3
"""
波浪形状测试脚本
快速预览所有9种波浪形状，无需手势控制
"""

import sys
import time
from render_engine import RenderEngine
from particle_sphere_system import ParticleSphereSystem

def test_all_shapes():
    """测试所有波浪形状"""
    print("=== 波浪形状测试模式 ===")
    print("将自动切换所有9种波浪形状，每种显示5秒")
    print("按ESC退出测试\n")
    
    # 初始化系统
    engine = RenderEngine(width=1200, height=800, title="波浪形状测试")
    particle_system = ParticleSphereSystem(max_particles=1000)
    
    # 所有形状列表
    shapes = [
        ('sine_wave', '正弦波', '经典的平滑波浪线'),
        ('cosine_wave', '余弦波', '相位偏移的波浪'),
        ('double_wave', '双重波浪', '两层叠加波浪'),
        ('spiral_line', '螺旋线', '平面向外螺旋'),
        ('zigzag_line', '锯齿波', '尖锐锯齿形状'),
        ('heart_curve', '心形曲线', '浪漫心形轨迹'),
        ('infinity_curve', '无穷符号', '∞ 形状曲线'),
        ('helix_3d', '3D螺旋', '立体螺旋上升'),
        ('multiple_lines', '多条平行线', '5条平行波浪')
    ]
    
    current_shape_index = 0
    shape_timer = time.time()
    shape_duration = 5.0  # 每个形状显示5秒
    
    print(f"正在显示: {shapes[current_shape_index][1]} - {shapes[current_shape_index][2]}")
    
    while engine.is_running:
        if not engine.handle_events():
            break
        
        current_time = time.time()
        dt = 1.0 / 60.0
        
        # 检查是否需要切换形状
        if current_time - shape_timer >= shape_duration:
            current_shape_index = (current_shape_index + 1) % len(shapes)
            shape_info = shapes[current_shape_index]
            
            # 设置新形状
            particle_system.particle_system.params['shape_mode'] = shape_info[0]
            particle_system.particle_system.params['wave_amplitude'] = 2.0
            particle_system.particle_system.params['wave_frequency'] = 1.0
            particle_system.particle_system.params['wave_speed'] = 1.0
            particle_system.particle_system.params['line_length'] = 6.0
            
            print(f"正在显示: {shape_info[1]} - {shape_info[2]}")
            shape_timer = current_time
        
        # 模拟手势数据以获得好看的效果
        mock_gesture_data = {
            'hands_detected': 1,
            'left_hand': {
                'detected': True,
                'gesture': 'open_hand',
                'openness': 0.8,
                'center': [0.3 + 0.2 * (current_time * 0.1 % 1), 0.6]
            },
            'right_hand': {'detected': False},
            'gesture_strength': 0.7 + 0.3 * abs(math.sin(current_time * 0.5)) if 'math' in globals() else 0.7
        }
        
        # 添加数学模块导入
        import math
        mock_gesture_data['gesture_strength'] = 0.7 + 0.3 * abs(math.sin(current_time * 0.5))
        
        # 更新系统
        particle_system.update(dt, mock_gesture_data)
        
        # 渲染
        engine.clear_screen()
        engine.update_camera()
        
        # 获取渲染数据
        particle_data = particle_system.get_particle_data()
        sphere_data = particle_system.get_sphere_data()
        wireframe_data = particle_system.get_wireframe_data()
        
        # 渲染粒子
        engine.render_particles(
            particle_data['positions'],
            particle_data['colors'],
            particle_data['sizes']
        )
        
        # 渲染半透明球体作为参考
        engine.render_sphere(
            radius=sphere_data['radius'] * 0.5,
            rotation=sphere_data['rotation'],
            color=[0.3, 0.5, 1.0],
            transparency=0.1
        )
        
        # 渲染线框
        if wireframe_data:
            engine.render_wireframe_sphere(
                radius=wireframe_data['radius'] * 0.8,
                rotation=wireframe_data['rotation'],
                color=[0.5, 1.0, 0.5],
                line_width=1.0
            )
        
        engine.present()
        time.sleep(dt)
    
    engine.cleanup()
    print("\n测试结束！")

def interactive_shape_test():
    """交互式形状测试"""
    print("=== 交互式波浪形状测试 ===")
    print("按数字键1-9切换不同形状：")
    print("1. 正弦波")
    print("2. 余弦波")
    print("3. 双重波浪")
    print("4. 螺旋线")
    print("5. 锯齿波")
    print("6. 心形曲线") 
    print("7. 无穷符号")
    print("8. 3D螺旋")
    print("9. 多条平行线")
    print("按ESC退出\n")
    
    # 初始化系统
    engine = RenderEngine(width=1200, height=800, title="交互式波浪形状测试")
    particle_system = ParticleSphereSystem(max_particles=1200)
    
    shapes = [
        'sine_wave', 'cosine_wave', 'double_wave', 'spiral_line',
        'zigzag_line', 'heart_curve', 'infinity_curve', 'helix_3d', 'multiple_lines'
    ]
    shape_names = [
        '正弦波', '余弦波', '双重波浪', '螺旋线',
        '锯齿波', '心形曲线', '无穷符号', '3D螺旋', '多条平行线'
    ]
    
    current_shape = 0
    print(f"当前形状: {shape_names[current_shape]}")
    
    while engine.is_running:
        # 处理事件
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                engine.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    engine.is_running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    current_shape = event.key - pygame.K_1
                    if current_shape < len(shapes):
                        particle_system.particle_system.params['shape_mode'] = shapes[current_shape]
                        print(f"切换到: {shape_names[current_shape]}")
        
        current_time = time.time()
        dt = 1.0 / 60.0
        
        # 动态参数
        import math
        wave_phase = current_time * 0.5
        mock_gesture_data = {
            'hands_detected': 1,
            'left_hand': {
                'detected': True,
                'gesture': 'open_hand',
                'openness': 0.8 + 0.2 * math.sin(wave_phase),
                'center': [0.5 + 0.2 * math.sin(wave_phase * 0.7), 0.6]
            },
            'right_hand': {'detected': False},
            'gesture_strength': 0.6 + 0.4 * abs(math.sin(wave_phase))
        }
        
        # 更新和渲染
        particle_system.update(dt, mock_gesture_data)
        
        engine.clear_screen()
        engine.update_camera()
        
        particle_data = particle_system.get_particle_data()
        sphere_data = particle_system.get_sphere_data()
        
        engine.render_particles(
            particle_data['positions'],
            particle_data['colors'],
            particle_data['sizes']
        )
        
        engine.render_sphere(
            radius=sphere_data['radius'] * 0.3,
            rotation=sphere_data['rotation'],
            color=[0.2, 0.4, 0.8],
            transparency=0.15
        )
        
        engine.present()
        time.sleep(dt)
    
    engine.cleanup()
    print("交互式测试结束！")

def main():
    """主函数"""
    print("选择测试模式：")
    print("1. 自动轮播所有形状 (推荐)")
    print("2. 交互式形状切换")
    print("3. 退出")
    
    try:
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            test_all_shapes()
        elif choice == '2':
            interactive_shape_test()
        elif choice == '3':
            print("退出测试")
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n用户中断退出")
    except Exception as e:
        print(f"测试错误: {e}")

if __name__ == "__main__":
    main()