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
        
        # 初始化音频系统
        self.init_audio_system()
        
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
    
    def init_audio_system(self):
        """初始化音频系统"""
        print("正在初始化音频系统...")
        
        # 音频文件配置（确保文件名完全匹配）
        self.audio_files = {
            1: "Fugue in G Trio violin-Violin.mp3",      # 小提琴声部
            2: "Fugue in G Trio-Tenor_Lute.mp3",        # 鲁特琴声部  
            3: "Fugue in G Trio Organ-Organ.mp3"        # 管风琴声部
        }
        
        # 调试：输出实际存在的文件
        print("检查音频文件存在性:")
        import glob
        actual_files = glob.glob("Fugue in G Trio*.mp3")
        for f in actual_files:
            print(f"  实际文件: {f}")
        
        print("期望的文件映射:")
        for track_id, filename in self.audio_files.items():
            exists = os.path.exists(filename)
            print(f"  音轨{track_id}: {filename} {'✅存在' if exists else '❌缺失'}")
        
        # 检查音频文件是否存在
        missing_files = []
        for track_id, filename in self.audio_files.items():
            if not os.path.exists(filename):
                missing_files.append(filename)
        
        if missing_files:
            print("⚠️ 部分音频文件缺失:")
            for file in missing_files:
                print(f"   - {file}")
            print("音频功能将被禁用")
            self.audio_enabled = False
            return
        
        # 加载音频文件
        self.audio_sounds = {}
        self.audio_channels = {}
        self.audio_volumes = {1: 0.0, 2: 0.0, 3: 0.0}
        self.playing_tracks = set()
        self.master_playing = False  # 主播放状态
        self.sync_start_time = None  # 同步播放开始时间
        
        for track_id, filename in self.audio_files.items():
            try:
                print(f"🔄 正在加载音轨{track_id}: {filename}")
                sound = pygame.mixer.Sound(filename)
                sound.set_volume(0.0)  # 初始音量为0
                
                # 测试文件是否真的可以播放
                length = sound.get_length()
                print(f"  📏 音频长度: {length:.2f}秒")
                
                self.audio_sounds[track_id] = sound
                self.audio_channels[track_id] = pygame.mixer.Channel(track_id - 1)
                
                print(f"✅ 音轨{track_id}加载成功: {filename}")
            except Exception as e:
                print(f"❌ 音轨{track_id}加载失败: {e}")
                # 不要完全禁用音频系统，只是跳过这个文件
                print(f"⚠️ 跳过音轨{track_id}，继续加载其他文件...")
                continue
        
        # 检查是否至少有一个文件加载成功
        if not self.audio_sounds:
            print("❌ 没有任何音频文件加载成功，禁用音频功能")
            self.audio_enabled = False
            return
        else:
            loaded_tracks = list(self.audio_sounds.keys())
            print(f"✅ 成功加载 {len(loaded_tracks)} 个音轨: {loaded_tracks}")
        
        print("✓ 音频系统初始化完成")
    
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
        """根据数字手势更新音频播放（同步播放模式）"""
        if not self.audio_enabled:
            return
        
        if not hasattr(self, 'audio_sounds') or not self.audio_sounds:
            print("⚠️ 音频系统未正确初始化")
            return
        
        active_gestures = set(digit_gestures)
        
        # 调试信息（每60帧输出一次，避免刷屏）
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 60 == 0:  # 每秒输出一次
            volumes_str = {k: f"{v:.2f}" for k, v in self.audio_volumes.items()}
            print(f"🎵 同步音频: 手势={digit_gestures}, 主播放={self.master_playing}, 音量={volumes_str}")
        
        # 检查是否需要启动或停止主播放
        has_any_gesture = len(active_gestures) > 0
        
        if has_any_gesture and not self.master_playing:
            # 启动同步播放：同时播放所有音轨
            print("🎼 启动同步播放所有音轨")
            import time
            self.sync_start_time = time.time()
            
            for track_id in self.audio_sounds.keys():
                try:
                    # 所有音轨同时开始播放，初始音量为0
                    self.audio_sounds[track_id].set_volume(0.0)
                    self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                    self.playing_tracks.add(track_id)
                except Exception as e:
                    print(f"❌ 启动音轨{track_id}失败: {e}")
            
            self.master_playing = True
            print(f"✅ 所有音轨已同步启动，播放中: {list(self.playing_tracks)}")
        
        elif not has_any_gesture and self.master_playing:
            # 停止所有播放：淡出然后停止
            print("🔇 停止同步播放")
            # 先快速淡出所有音轨
            for track_id in self.audio_sounds.keys():
                self.audio_volumes[track_id] = 0.0
                try:
                    self.audio_sounds[track_id].set_volume(0.0)
                except Exception as e:
                    print(f"❌ 设置音轨{track_id}音量失败: {e}")
            
            # 稍后停止播放（给淡出一点时间）
            import threading
            def stop_all_delayed():
                import time
                time.sleep(0.5)  # 等待0.5秒让淡出完成
                for track_id in list(self.playing_tracks):
                    try:
                        self.audio_channels[track_id].stop()
                    except Exception as e:
                        print(f"❌ 停止音轨{track_id}失败: {e}")
                self.playing_tracks.clear()
                self.master_playing = False
                print("✅ 所有音轨已停止")
            
            threading.Thread(target=stop_all_delayed, daemon=True).start()
        
        # 如果正在播放，更新各音轨的音量
        if self.master_playing:
            for track_id in self.audio_sounds.keys():
                should_be_audible = track_id in active_gestures
                
                # 计算目标音量
                target_vol = 1.0 if should_be_audible else 0.0
                
                # 平滑音量变化
                current_vol = self.audio_volumes[track_id]
                volume_change_speed = 0.15  # 调整切换速度
                new_vol = current_vol + (target_vol - current_vol) * volume_change_speed
                
                # 更新音量
                self.audio_volumes[track_id] = new_vol
                try:
                    self.audio_sounds[track_id].set_volume(new_vol)
                except Exception as e:
                    print(f"❌ 设置音轨{track_id}音量失败: {e}")
                
                # 记录音量变化（仅用于调试）
                if abs(new_vol - current_vol) > 0.01 and self._debug_counter % 30 == 0:
                    status = "📈升高" if new_vol > current_vol else "📉降低"
                    print(f"  音轨{track_id}: {status} {current_vol:.2f} → {new_vol:.2f}")
    
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
            print("- M键：切换音频开关")
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
                print("\n🎵 数字手势 → 音频控制：")
                print("- 1️⃣ 食指 → 播放小提琴声部")
                print("- 2️⃣ 食指+中指 → 播放鲁特琴声部") 
                print("- 3️⃣ 食指+中指+无名指 → 播放管风琴声部")
                print("- ✋ 张开手掌 → 播放所有声部（完整合奏）")
                print("- 可同时做多个手势创造复杂音乐组合")
                print("- 无手势时所有音轨静音\n")
            else:
                print("\n⚠️ 音频功能未启用（音频文件缺失）\n")
            
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
        elif key == pygame.K_m:
            # 切换音频开关
            if hasattr(self, 'audio_sounds') and self.audio_sounds:
                self.audio_enabled = not self.audio_enabled
                if not self.audio_enabled:
                    # 停止所有音频
                    for track_id in list(self.playing_tracks):
                        self.audio_channels[track_id].stop()
                    self.playing_tracks.clear()
                    for track_id in self.audio_volumes:
                        self.audio_volumes[track_id] = 0.0
                print(f"音频控制: {'开' if self.audio_enabled else '关'}")
            else:
                print("音频系统未初始化")
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
        if self.audio_enabled:
            info_lines.append("--- Audio ---")
            info_lines.append(f"Digits: {digit_gestures}")
            
            # 显示播放状态（同步播放模式）
            audio_status = []
            if hasattr(self, 'audio_sounds'):
                if hasattr(self, 'master_playing') and self.master_playing:
                    audio_status.append(f"SYNC: {'ON' if self.master_playing else 'OFF'}")
                    for track_id in self.audio_sounds.keys():
                        volume = self.audio_volumes.get(track_id, 0)
                        audible = volume > 0.1
                        status = "🔊" if audible else "🔇"
                        audio_status.append(f"T{track_id}:{status}({volume:.1f})")
                else:
                    audio_status.append("SYNC: STOPPED")
            else:
                audio_status.append("No audio tracks loaded")
            
            info_lines.extend(audio_status)
        else:
            info_lines.append("Audio: DISABLED")
        
        # 调整背景大小以容纳更多信息
        info_height = max(220, len(info_lines) * 20 + 40)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, info_height), (255, 255, 255), 2)
        
        for i, line in enumerate(info_lines):
            # 音频信息用不同颜色
            color = (0, 255, 255) if "Audio" in line or "T1:" in line or "T2:" in line or "T3:" in line or "Digits:" in line else (0, 255, 0)
            cv2.putText(frame, line, (w - 270, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
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
            if hasattr(self, 'master_playing') and self.master_playing:
                # 停止同步播放
                for track_id in list(self.playing_tracks):
                    self.audio_channels[track_id].stop()
                self.playing_tracks.clear()
                self.master_playing = False
                print("✓ 同步音频播放已停止")
            
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