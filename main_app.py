"""
主应用程序
整合手势识别、粒子系统和3D渲染的完整应用
"""

import cv2
import pygame
import threading
import time
import numpy as np
from gesture_detector import GestureDetector
from render_engine import RenderEngine
from particle_sphere_system import ParticleSphereSystem

class GestureParticleApp:
    def __init__(self):
        print("正在初始化手势粒子应用...")
        
        # 初始化各个系统
        self.gesture_detector = GestureDetector()
        self.render_engine = RenderEngine(width=1400, height=900, title="手势控制粒子球形效果")
        self.particle_sphere_system = ParticleSphereSystem(max_particles=1500)
        
        # 运行状态
        self.is_running = True
        self.show_camera = True  # 是否显示摄像头窗口
        
        # 性能监控
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # 应用参数
        self.params = {
            'particle_count_multiplier': 1.0,
            'sensitivity': 1.0,
            'smoothing': 0.8,
            'background_color': [0.05, 0.05, 0.1],
            'show_wireframe': True,
            'show_info': True
        }
        
        print("✓ 手势检测器初始化完成")
        print("✓ 渲染引擎初始化完成")
        print("✓ 粒子系统初始化完成")
    
    def start(self):
        """启动应用"""
        print("\n正在启动应用...")
        
        try:
            # 启动摄像头
            print("启动摄像头...")
            self.gesture_detector.start_camera(0)
            print("✓ 摄像头启动成功")
            
            # 等待摄像头稳定
            time.sleep(1.0)
            
            print("\n=== 应用启动成功！===")
            print("控制说明：")
            print("- 鼠标左键拖拽：旋转视角")
            print("- R键：重置视角")
            print("- C键：切换摄像头窗口显示")
            print("- W键：切换线框显示")
            print("- I键：切换信息显示")
            print("- S键：切换波浪形状")
            print("- ESC键：退出应用")
            print("- 数字键1-5：调整粒子数量")
            print("\n🧬 手势控制 → 螺旋结构：")
            print("- 握拳 → 龙卷风螺旋")
            print("- 1个手指 → 双螺旋结构") 
            print("- 2个手指 → 三重螺旋")
            print("- 3个手指 → DNA双螺旋(带连接桥)")
            print("- 4个手指 → 编织螺旋线")
            print("- 张开手掌 → 银河螺旋")
            print("- 双手 → 多重螺旋塔")
            print("- 手势强度：控制螺旋半径和高度") 
            print("- 手部位置：控制颜色和扭转速度")
            print("- 双手距离：控制螺旋数量和连接桥\n")
            
            # 主循环
            self.run_main_loop()
            
        except Exception as e:
            print(f"启动错误: {e}")
        finally:
            self.cleanup()
    
    def run_main_loop(self):
        """主循环"""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 处理事件
            self.handle_events()
            
            # 获取手势数据
            gesture_data = self.gesture_detector.get_gesture_data()
            
            # 更新粒子球形系统
            self.particle_sphere_system.update(dt, gesture_data)
            
            # 渲染3D场景
            self.render_3d_scene()
            
            # 显示摄像头窗口
            if self.show_camera:
                self.show_camera_window()
            
            # 更新FPS
            self.update_fps()
            
            # 限制帧率
            if dt < 1.0/60.0:
                time.sleep(1.0/60.0 - dt)
    
    def handle_events(self):
        """处理用户输入事件"""
        # 处理渲染引擎事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
        
        # 处理OpenCV窗口事件
        if self.show_camera:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
    
    def handle_keydown(self, key):
        """处理按键事件"""
        if key == pygame.K_ESCAPE:
            self.is_running = False
        elif key == pygame.K_r:
            # 重置相机
            self.render_engine.camera_yaw = 0
            self.render_engine.camera_pitch = 0
            print("视角已重置")
        elif key == pygame.K_c:
            # 切换摄像头显示
            self.show_camera = not self.show_camera
            if not self.show_camera:
                cv2.destroyAllWindows()
            print(f"摄像头显示: {'开' if self.show_camera else '关'}")
        elif key == pygame.K_w:
            # 切换线框显示
            self.params['show_wireframe'] = not self.params['show_wireframe']
            print(f"线框显示: {'开' if self.params['show_wireframe'] else '关'}")
        elif key == pygame.K_i:
            # 切换信息显示
            self.params['show_info'] = not self.params['show_info']
            print(f"信息显示: {'开' if self.params['show_info'] else '关'}")
        elif key == pygame.K_s:
            # 手动切换波浪形状
            new_shape = self.particle_sphere_system.particle_system.cycle_shape_mode()
            shape_names = {
                'sine_wave': '正弦波',
                'cosine_wave': '余弦波', 
                'double_wave': '双重波浪',
                'spiral_line': '螺旋线',
                'zigzag_line': '锯齿波',
                'heart_curve': '心形曲线',
                'infinity_curve': '无穷符号',
                'helix_3d': '3D螺旋',
                'multiple_lines': '多条平行线',
                'double_helix': '双螺旋',
                'triple_helix': '三重螺旋',
                'dna_structure': 'DNA结构',
                'twisted_ribbon': '扭转带状',
                'braided_lines': '编织线条',
                'spiral_tower': '螺旋塔',
                'coil_spring': '弹簧线圈',
                'tornado_helix': '龙卷风螺旋',
                'galaxy_spiral': '银河螺旋'
            }
            print(f"切换到波浪形状: {shape_names.get(new_shape, new_shape)}")
        elif key >= pygame.K_1 and key <= pygame.K_5:
            # 调整粒子数量
            multiplier = (key - pygame.K_1 + 1) * 0.4
            self.params['particle_count_multiplier'] = multiplier
            new_count = int(1500 * multiplier)
            print(f"粒子数量倍数: {multiplier:.1f} (约{new_count}个粒子)")
    
    def render_3d_scene(self):
        """渲染3D场景"""
        # 清屏
        self.render_engine.clear_screen()
        
        # 更新相机
        self.render_engine.update_camera()
        
        # 获取渲染数据
        particle_data = self.particle_sphere_system.get_particle_data()
        sphere_data = self.particle_sphere_system.get_sphere_data()
        wireframe_data = self.particle_sphere_system.get_wireframe_data()
        
        # 应用粒子数量倍数
        if self.params['particle_count_multiplier'] != 1.0:
            positions = particle_data['positions']
            colors = particle_data['colors']
            sizes = particle_data['sizes']
            
            # 计算要显示的粒子数量
            total_particles = len(positions) // 3
            display_count = int(total_particles * self.params['particle_count_multiplier'])
            display_count = min(display_count, total_particles)
            
            if display_count > 0:
                particle_data['positions'] = positions[:display_count * 3]
                particle_data['colors'] = colors[:display_count * 4] if colors else None
                particle_data['sizes'] = sizes[:display_count] if sizes else None
        
        # 渲染粒子
        self.render_engine.render_particles(
            particle_data['positions'],
            particle_data['colors'],
            particle_data['sizes']
        )
        
        # 渲染螺旋结构
        helix_points = self.particle_sphere_system.get_helix_points()
        if helix_points and helix_points['positions']:
            self.render_engine.render_particles(
                helix_points['positions'],
                helix_points['colors'],
                None  # 螺旋点不需要大小变化
            )
        
        # 注释掉参考球体，不需要显示
        # self.render_engine.render_sphere(
        #     radius=sphere_data['radius'] * 0.15,  # 很小的参考球体
        #     rotation=sphere_data['rotation'],
        #     color=[0.8, 0.8, 0.8],
        #     transparency=0.1
        # )
        
        # 可选：渲染线框球体作为边界参考（默认关闭）
        # if self.params['show_wireframe'] and wireframe_data:
        #     self.render_engine.render_wireframe_sphere(
        #         radius=wireframe_data['radius'],
        #         rotation=wireframe_data['rotation'],
        #         color=wireframe_data['color'],
        #         line_width=wireframe_data['line_width']
        #     )
        
        # 显示画面
        self.render_engine.present()
    
    def show_camera_window(self):
        """显示摄像头窗口"""
        frame = self.gesture_detector.get_current_frame()
        if frame is not None:
            # 添加性能信息
            if self.params['show_info']:
                self.add_performance_info(frame)
            
            # 调整窗口大小以便观看
            display_frame = cv2.resize(frame, (480, 360))
            cv2.imshow('Hand Gesture Detection', display_frame)
    
    def add_performance_info(self, frame):
        """在摄像头画面上添加性能信息"""
        h, w = frame.shape[:2]
        
        # 获取当前系统状态
        gesture_data = self.gesture_detector.get_gesture_data()
        particle_data = self.particle_sphere_system.get_particle_data()
        active_particles = len(particle_data['positions']) // 3
        
        # 获取当前波浪形状
        current_shape = self.particle_sphere_system.particle_system.params['shape_mode']
        shape_names = {
            'sine_wave': '正弦波',
            'cosine_wave': '余弦波', 
            'double_wave': '双重波浪',
            'spiral_line': '螺旋线',
            'zigzag_line': '锯齿波',
            'heart_curve': '心形曲线',
            'infinity_curve': '无穷符号',
            'helix_3d': '3D螺旋',
            'multiple_lines': '多条平行线',
            'double_helix': '双螺旋',
            'triple_helix': '三重螺旋',
            'dna_structure': 'DNA结构',
            'twisted_ribbon': '扭转带状',
            'braided_lines': '编织线条',
            'spiral_tower': '螺旋塔',
            'coil_spring': '弹簧线圈',
            'tornado_helix': '龙卷风螺旋',
            'galaxy_spiral': '银河螺旋'
        }
        shape_display = shape_names.get(current_shape, current_shape)
        
        # 性能信息背景
        cv2.rectangle(frame, (w - 280, 10), (w - 10, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, 220), (255, 255, 255), 2)
        
        # 性能信息文本
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Particles: {active_particles}",
            f"Multiplier: {self.params['particle_count_multiplier']:.1f}x",
            f"Shape: {shape_display}",
            f"Hands: {gesture_data.get('hands_detected', 0)}",
            f"Strength: {gesture_data.get('gesture_strength', 0):.2f}",
        ]
        
        # 添加手势信息
        if gesture_data.get('left_hand', {}).get('detected', False):
            left = gesture_data['left_hand']
            info_lines.append(f"L: {left['gesture']}")
        
        if gesture_data.get('right_hand', {}).get('detected', False):
            right = gesture_data['right_hand']
            info_lines.append(f"R: {right['gesture']}")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (w - 270, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def update_fps(self):
        """更新FPS计数"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        
        try:
            self.gesture_detector.stop_camera()
            print("✓ 摄像头已停止")
        except:
            pass
        
        try:
            self.render_engine.cleanup()
            print("✓ 渲染引擎已清理")
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
            print("✓ OpenCV窗口已关闭")
        except:
            pass
        
        print("✓ 应用已完全退出")

def main():
    """主函数"""
    print("=== 手势控制粒子球形效果应用 ===")
    print("Python版本 - 无需TouchDesigner")
    
    try:
        app = GestureParticleApp()
        app.start()
    except KeyboardInterrupt:
        print("\n用户中断退出")
    except Exception as e:
        print(f"应用错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()