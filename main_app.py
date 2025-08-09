"""
主应用程序
整合手势识别、粒子系统、3D渲染和音频控制的完整应用
"""

import cv2
import pygame
import threading
import time
import numpy as np
import os
from gesture_detector import GestureDetector
from render_engine import RenderEngine
from particle_sphere_system import ParticleSphereSystem
from hand_gesture_detector import HandGestureDetector
from realistic_audio_manager import RealisticAudioManager

class GestureParticleApp:
    def __init__(self):
        print("正在初始化手势粒子音频应用...")
        
        # 初始化各个系统（注意顺序很重要）
        self.gesture_detector = GestureDetector()
        # 注意：我们复用现有的手势检测器来控制音频，不需要单独的数字检测器
        
        # 初始化pygame音频系统（在RenderEngine之前）
        print("初始化pygame音频系统...")
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            print("✓ Pygame音频系统初始化完成")
        except Exception as e:
            print(f"✗ Pygame音频初始化失败: {e}")
        
        # 初始化现实主义音频管理器
        self.audio_manager = RealisticAudioManager()
        self.audio_enabled = self.audio_manager.initialize()
        
        # 最后初始化渲染引擎（可能会重新初始化pygame）
        print("初始化渲染引擎...")
        self.render_engine = RenderEngine(width=1400, height=900, title="手势控制粒子球形效果 + 音频")
        self.particle_sphere_system = ParticleSphereSystem(max_particles=1500)
        
        # 运行状态
        self.is_running = True
        self.show_camera = True  # 是否显示摄像头窗口
        self.audio_enabled = True  # 音频开关
        
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
        print("✓ 数字手势检测器初始化完成")
        print("✓ 渲染引擎初始化完成")
        print("✓ 粒子系统初始化完成")
    
    
    def convert_gesture_to_digits(self, gesture_data):
        """将现有手势数据转换为数字手势"""
        digit_gestures = []
        
        # 检查左手
        left_hand = gesture_data.get('left_hand', {})
        if left_hand.get('detected', False):
            gesture = left_hand.get('gesture', 'none')
            digits = self.gesture_name_to_digits(gesture)
            digit_gestures.extend(digits)
        
        # 检查右手
        right_hand = gesture_data.get('right_hand', {})
        if right_hand.get('detected', False):
            gesture = right_hand.get('gesture', 'none')
            digits = self.gesture_name_to_digits(gesture)
            digit_gestures.extend(digits)
        
        # 去重并排序
        return sorted(list(set(digit_gestures)))
    
    def gesture_name_to_digits(self, gesture_name):
        """将手势名称转换为数字列表"""
        gesture_map = {
            'one': [1],         # 一个手指 -> 小提琴
            'two': [2],         # 两个手指 -> 鲁特琴  
            'three': [3],       # 三个手指 -> 管风琴
            'open_hand': [1, 2, 3],  # 张开手掌 -> 所有音轨
        }
        return gesture_map.get(gesture_name, [])
    
    def update_audio_from_gestures(self, digit_gestures):
        """根据数字手势更新音频播放（支持断点续播）"""
        if not self.audio_enabled:
            return
        
        # 使用新的高级音频管理器
        self.audio_manager.update_from_gestures(digit_gestures)
    
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
            print("- R key: Reset camera view and audio position")
            print("- C key: Toggle camera window display")
            print("- W key: Toggle wireframe display")
            print("- I key: Toggle info display")
            print("- S key: Cycle through wave shapes")
            print("- M key: Toggle audio control on/off")
            print("- P key: Manual pause/resume audio playback")
            print("- T key: Toggle audio restart strategy")
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
            print("- 双手距离：控制螺旋数量和连接桥")
            
            if self.audio_enabled:
                print("\n🎵 Realistic Continuous Audio Control:")
                print("- Audio plays continuously in background")
                print("- 1 finger → Hear violin track")
                print("- 2 fingers → Hear lute track") 
                print("- 3 fingers → Hear organ track")
                print("- Open hand → Hear all tracks (full orchestra)")
                print("- No gesture → All tracks muted (still playing)")
                print("- P key: Manual pause/resume entire playback")
                print("- R key: Restart from beginning")
                print("- T key: Toggle long-pause restart behavior\n")
            else:
                print("\n⚠️ Audio functionality disabled (missing audio files)\n")
            
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
            
            # 获取手势数据（用于粒子系统）
            gesture_data = self.gesture_detector.get_gesture_data()
            
            # 将现有手势数据转换为数字手势（用于音频控制）
            if self.audio_enabled and gesture_data:
                digit_gestures = self.convert_gesture_to_digits(gesture_data)
                self.update_audio_from_gestures(digit_gestures)
            
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
        elif key == pygame.K_m:
            # 切换音频开关
            if hasattr(self, 'audio_manager') and self.audio_manager.enabled:
                self.audio_enabled = not self.audio_enabled
                if not self.audio_enabled:
                    # 暂停所有音频
                    self.audio_manager.pause_playback()
                print(f"音频控制: {'开' if self.audio_enabled else '关'}")
            else:
                print("音频系统未初始化")
        elif key == pygame.K_p:
            # P键：手动暂停/继续音频
            if hasattr(self, 'audio_manager') and self.audio_manager.enabled:
                self.audio_manager.manual_pause_resume()
            else:
                print("Audio system not initialized")
        elif key == pygame.K_r:
            # R键：重置摄像头视角和音频位置
            # 重置相机
            self.render_engine.camera_yaw = 0
            self.render_engine.camera_pitch = 0
            print("Camera view reset")
            
            # 重置音频位置
            if hasattr(self, 'audio_manager') and self.audio_manager.enabled:
                self.audio_manager.reset_position()
                print("Audio position reset")
        elif key == pygame.K_t:
            # T键：切换音频重启策略
            if hasattr(self, 'audio_manager') and self.audio_manager.enabled:
                current_strategy = self.audio_manager.restart_from_beginning
                new_strategy = not current_strategy
                self.audio_manager.set_restart_strategy(new_strategy)
                strategy_name = "Restart from beginning" if new_strategy else "Smart resume"
                print(f"Audio strategy: {strategy_name}")
            else:
                print("Audio system not initialized")
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
        """在摄像头画面上添加性能和音频信息"""
        h, w = frame.shape[:2]
        
        # 获取当前系统状态
        gesture_data = self.gesture_detector.get_gesture_data()
        particle_data = self.particle_sphere_system.get_particle_data()
        active_particles = len(particle_data['positions']) // 3
        
        # 获取音频状态（从现有手势数据转换）
        digit_gestures = []
        if gesture_data:
            digit_gestures = self.convert_gesture_to_digits(gesture_data)
        
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
        
        # 添加音频信息
        if self.audio_enabled and hasattr(self, 'audio_manager'):
            info_lines.append("--- Audio ---")
            info_lines.append(f"Digits: {digit_gestures}")
            
            # 获取音频状态信息
            audio_status_info = self.audio_manager.get_status_info()
            
            # 显示播放状态（断点续播模式）
            audio_status = []
            if audio_status_info['enabled']:
                playing_status = "PLAY" if audio_status_info['master_playing'] else "PAUSE"
                audio_status.append(f"Status: {playing_status}")
                
                # 显示播放位置
                current_pos = audio_status_info['current_position']
                audio_status.append(f"Time: {current_pos:.1f}s")
                
                # 显示各音轨状态
                for track_id, volume in audio_status_info['volumes'].items():
                    audible = volume > 0.1
                    status = "ON" if audible else "OFF"
                    audio_status.append(f"T{track_id}:{status}({volume:.1f})")
            else:
                audio_status.append("No audio manager")
            
            info_lines.extend(audio_status)
        else:
            info_lines.append("Audio: DISABLED")
        
        # 调整背景大小以容纳更多信息
        info_height = max(220, len(info_lines) * 20 + 40)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, info_height), (255, 255, 255), 2)
        
        for i, line in enumerate(info_lines):
            # 音频信息用不同颜色
            try:
                # 确保使用ASCII字符
                safe_line = line.encode('ascii', 'ignore').decode('ascii')
                color = (0, 255, 255) if "Audio" in safe_line or "T1:" in safe_line or "T2:" in safe_line or "T3:" in safe_line or "Digits:" in safe_line else (0, 255, 0)
                cv2.putText(frame, safe_line, (w - 270, 35 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                # 如果出现编码错误，显示简化信息
                cv2.putText(frame, f"Line {i}: [encoding error]", (w - 270, 35 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
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
        
        # 清理音频资源
        try:
            if hasattr(self, 'audio_manager'):
                self.audio_manager.cleanup()
                print("✓ 高级音频管理器已清理")
            
            pygame.mixer.quit()
            print("✓ 音频系统已清理")
        except:
            pass
        
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