"""
E_Major 人体姿态音频控制系统主程序
简化版应用 - 专注于音频控制，移除粒子系统

核心功能:
- 使用 MediaPipe Pose 检测人体姿态
- 识别小提琴演奏动作
- 根据检测结果控制11个音轨的播放和音量
- 实时显示摄像头窗口和骨骼点可视化
"""

import cv2
import pygame
import time
import os
import sys
import platform
import numpy as np

# 添加父目录到路径（用于导入父目录的模块）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root)

# 导入本地模块（需要创建这些模块）
try:
    from pose_body_detector import PoseBodyDetector
    from e_major_audio_controller import EMajorAudioController
    print("✓ 成功导入核心模块")
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    print("请确保 pose_body_detector.py 和 e_major_audio_controller.py 在同一目录下")
    sys.exit(1)


class EMajorApp:
    """E_Major 人体姿态音频控制应用主类"""

    def __init__(self):
        """初始化E_Major应用"""
        print("\n" + "="*60)
        print("=== E_Major 人体姿态音频控制系统 ===")
        print("="*60)
        print("\n系统功能:")
        print("• 检测人体 → 播放管弦乐（小提琴静音）")
        print("• 检测小提琴动作 → 增强小提琴声部")
        print("• 无人检测 → 自动暂停所有音轨\n")

        # 显示平台信息
        system = platform.system()
        machine = platform.machine()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        if system == "Darwin":
            processor_type = "Apple Silicon" if machine == "arm64" else "Intel"
        else:
            processor_type = machine

        print(f"运行平台: {system} {processor_type}")
        print(f"Python版本: {python_version}")
        print(f"OpenCV版本: {cv2.__version__}")
        print(f"Pygame版本: {pygame.version.ver}\n")

        # 初始化pygame（用于窗口管理和音频）
        print("正在初始化pygame系统...")
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        print("✓ Pygame系统初始化完成\n")

        # 初始化姿态检测器
        print("正在初始化姿态检测器...")
        try:
            self.pose_detector = PoseBodyDetector()
            print("✓ 姿态检测器初始化完成")
        except Exception as e:
            print(f"✗ 姿态检测器初始化失败: {e}")
            raise

        # 初始化音频控制器
        print("\n正在初始化音频控制器...")
        try:
            self.audio_controller = EMajorAudioController()
            print("✓ 音频控制器初始化完成")
        except Exception as e:
            print(f"✗ 音频控制器初始化失败: {e}")
            raise

        # 运行状态标志
        self.is_running = True
        self.show_camera = True  # 是否显示摄像头窗口
        self.show_info = True    # 是否显示信息覆盖层
        self.paused = False      # 手动暂停标志

        # 性能监控
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        # 摄像头窗口名称
        self.window_name = 'E_Major - 人体姿态音频控制'

        print("\n" + "="*60)
        print("初始化完成！准备启动应用...")
        print("="*60 + "\n")

    def start(self):
        """启动应用"""
        print("\n正在启动应用...")

        try:
            # 启动摄像头
            print("\n1. 启动摄像头...")
            try:
                if not self.pose_detector.start_camera(0):
                    print("✗ 摄像头启动失败")

                    # 提供平台特定的解决建议
                    system = platform.system()
                    if system == "Darwin":  # macOS
                        print("\n🔧 macOS 摄像头权限解决方案:")
                        print("  1. 系统偏好设置 > 安全性与隐私 > 隐私 > 相机")
                        print("  2. 确保 Terminal 或您的 Python IDE 有摄像头权限")
                        print("  3. 重新启动终端或 IDE")
                        print("  4. 确保没有其他应用正在使用摄像头")
                    elif system == "Windows":
                        print("\n🔧 Windows 摄像头权限解决方案:")
                        print("  1. 设置 > 隐私 > 相机")
                        print("  2. 确保应用有摄像头权限")
                        print("  3. 检查设备管理器中的摄像头状态")

                    return

                print("✓ 摄像头启动成功")
            except Exception as camera_error:
                print(f"✗ 摄像头启动异常: {camera_error}")
                import traceback
                traceback.print_exc()
                return

            # 初始化音频系统
            print("\n2. 初始化音频系统...")
            try:
                if not self.audio_controller.initialize():
                    print("✗ 音频初始化失败，请检查音频文件路径")
                    return
                print("✓ 音频系统就绪（11个音轨已加载）")
            except Exception as audio_error:
                print(f"✗ 音频初始化异常: {audio_error}")
                import traceback
                traceback.print_exc()
                return

            # 等待系统稳定
            print("\n3. 系统初始化中...")
            time.sleep(1.0)

            # 显示控制说明
            self.print_control_instructions()

            # 进入主循环
            print("\n" + "="*60)
            print("应用启动成功！开始运行主循环...")
            print("="*60 + "\n")

            self.run_main_loop()

        except KeyboardInterrupt:
            print("\n\n用户按下 Ctrl+C，准备退出...")
        except Exception as e:
            print(f"\n✗ 启动错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def print_control_instructions(self):
        """显示控制说明"""
        print("\n" + "="*60)
        print("=== 控制说明 ===")
        print("="*60)

        print("\n【键盘控制】")
        print("  C键 - 切换摄像头显示开/关")
        print("  I键 - 切换信息显示开/关")
        print("  P键 - 手动暂停/恢复音频")
        print("  R键 - 重置音频到起始位置")
        print("  ESC键 - 退出应用")

        print("\n【音频控制逻辑】")
        print("  • 无人检测 → 暂停所有音轨")
        print("  • 有人检测（无小提琴动作）→ 播放管弦乐（小提琴静音）")
        print("  • 有人检测 + 小提琴动作 → 全部播放（小提琴音量100%）")

        print("\n【小提琴动作识别】")
        print("  • 左手抬高（肩膀以上）")
        print("  • 右手抬高（肩膀以上）")
        print("  • 双臂呈小提琴演奏姿势")

        print("\n【11个音轨列表】")
        track_names = [
            "1: Oboe 1", "2: Oboe 2", "3: Organ", "4: Timpani",
            "5: Trumpet 1", "6: Trumpet 2", "7: Trumpet 3",
            "8: Violas", "9: Violin (主奏)", "10: Violins 1", "11: Violins 2"
        ]
        for name in track_names:
            print(f"  • Track {name}")

        print("\n" + "="*60 + "\n")

    def run_main_loop(self):
        """主循环"""
        last_time = time.time()
        frame_count = 0

        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            frame_count += 1

            # 处理键盘事件
            self.handle_events()

            # 获取摄像头帧
            frame = self.pose_detector.get_current_frame()

            if frame is not None:
                # 处理姿态检测（带骨骼点可视化）
                processed_frame = self.pose_detector.process_frame(frame, show_skeleton=True)

                # 获取姿态数据
                pose_data = self.pose_detector.get_pose_data()

                # 更新音频控制（除非手动暂停）
                if not self.paused:
                    detected_instruments = pose_data.get('detected_instruments', {})
                    self.audio_controller.update_from_instruments(
                        person_detected=pose_data['person_detected'],
                        detected_instruments=detected_instruments
                    )

                # 显示摄像头窗口
                if self.show_camera:
                    self.show_camera_window(processed_frame, pose_data)

            # 更新FPS
            self.update_fps()

            # 限制帧率到30fps（降低CPU使用率）
            target_fps = 30.0
            frame_time = 1.0 / target_fps
            if dt < frame_time:
                time.sleep(frame_time - dt)

            # 每100帧显示一次性能信息（可选）
            if frame_count % 100 == 0:
                print(f"[性能] FPS: {self.current_fps:.1f} | 运行时间: {current_time - self.fps_timer:.1f}s")

    def handle_events(self):
        """处理键盘事件"""
        if not self.show_camera:
            return

        # OpenCV窗口事件（仅在窗口显示时处理）
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # 没有按键
            return

        # ESC键 - 退出
        if key == 27:
            print("\n[用户操作] 按下ESC键，准备退出...")
            self.is_running = False

        # C键 - 切换摄像头显示
        elif key == ord('c') or key == ord('C'):
            self.show_camera = not self.show_camera
            if not self.show_camera:
                cv2.destroyAllWindows()
            status = "开启" if self.show_camera else "关闭"
            print(f"[用户操作] 摄像头显示: {status}")

        # I键 - 切换信息显示
        elif key == ord('i') or key == ord('I'):
            self.show_info = not self.show_info
            status = "开启" if self.show_info else "关闭"
            print(f"[用户操作] 信息显示: {status}")

        # P键 - 手动暂停/恢复音频
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            if self.paused:
                self.audio_controller.pause_all()
                print("[用户操作] 音频已手动暂停")
            else:
                self.audio_controller.resume_all()
                print("[用户操作] 音频已手动恢复")

        # R键 - 重置音频位置
        elif key == ord('r') or key == ord('R'):
            self.audio_controller.reset_position()
            print("[用户操作] 音频位置已重置到起点")

    def show_camera_window(self, frame, pose_data):
        """显示摄像头窗口"""
        if frame is None:
            return

        # 添加信息覆盖层
        if self.show_info:
            self.add_info_overlay(frame, pose_data)

        # 显示窗口
        cv2.imshow(self.window_name, frame)

    def add_info_overlay(self, frame, pose_data):
        """添加信息覆盖层到摄像头画面"""
        h, w = frame.shape[:2]

        # 信息面板位置和大小
        panel_width = 360
        panel_x = w - panel_width - 10
        panel_y = 10

        # 动态计算面板高度
        base_height = 420
        panel_height = base_height

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (w - 10, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制边框
        cv2.rectangle(frame, (panel_x, panel_y),
                     (w - 10, panel_y + panel_height), (255, 255, 255), 2)

        # 准备信息文本
        info_lines = []

        # 系统性能
        info_lines.append(f"FPS: {self.current_fps:.1f}")
        info_lines.append("")

        # 检测状态
        info_lines.append("=== Detection Status ===")
        person_status = "✓ Yes" if pose_data['person_detected'] else "✗ No"
        info_lines.append(f"Person: {person_status}")

        # 显示检测到的乐器
        detected_instruments = pose_data.get('detected_instruments', {})
        if detected_instruments:
            inst_list = [f"{inst.capitalize()}({conf:.2f})"
                        for inst, conf in detected_instruments.items()]
            info_lines.append(f"Detected: {', '.join(inst_list)}")
        else:
            info_lines.append("Detected: None")

        info_lines.append(f"Confidence: {pose_data['pose_confidence']:.2f}")
        info_lines.append("")

        # 音频状态
        info_lines.append("=== Audio Status ===")
        audio_status = self.audio_controller.get_status_info()

        current_state = audio_status.get('current_state', 'Unknown')
        info_lines.append(f"State: {current_state}")

        playback_pos = audio_status.get('playback_position', 0.0)
        info_lines.append(f"Position: {playback_pos:.1f}s")

        if self.paused:
            info_lines.append("Mode: Manual Pause")
        else:
            info_lines.append("Mode: Auto Control")

        info_lines.append("")

        # 激活组
        info_lines.append("=== Activated Groups ===")
        activated_groups = audio_status.get('activated_groups', [])

        if activated_groups:
            for group in sorted(activated_groups):
                info_lines.append(f"✓ {group.capitalize()}")
        else:
            info_lines.append("None")

        info_lines.append("")

        # 音轨音量（带可视化进度条）
        info_lines.append("=== Track Volumes ===")

        track_names = {
            1: "Oboe1", 2: "Oboe2", 3: "Organ", 4: "Timpani",
            5: "Trp1", 6: "Trp2", 7: "Trp3", 8: "Violas",
            9: "Violin*", 10: "Violins1", 11: "Violins2"
        }

        volumes = audio_status.get('volumes', {})

        for track_id in sorted(track_names.keys()):
            name = track_names[track_id]
            vol = volumes.get(track_id, 0.0)
            vol_percent = int(vol * 100)

            # 创建进度条
            bar_length = 10
            filled = int(bar_length * vol)
            bar = "█" * filled + "░" * (bar_length - filled)

            # 特殊标记小提琴主奏
            marker = " *" if track_id == 9 else ""
            info_lines.append(f"{name:8s}: {bar} {vol_percent:3d}%{marker}")

        # 绘制文本
        line_height = 20
        text_color = (200, 200, 200)  # 浅灰色

        for i, line in enumerate(info_lines):
            y_pos = panel_y + 25 + i * line_height

            # 根据内容设置颜色
            if "✓" in line:
                color = (0, 255, 0)  # 绿色
            elif "✗" in line:
                color = (0, 0, 255)  # 红色
            elif "===" in line:
                color = (0, 255, 255)  # 黄色（标题）
            elif "*" in line:
                color = (255, 100, 255)  # 紫色（小提琴主奏）
            else:
                color = text_color

            cv2.putText(frame, line, (panel_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def update_fps(self):
        """更新FPS计数"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time

    def cleanup(self):
        """清理资源"""
        print("\n" + "="*60)
        print("正在清理资源...")
        print("="*60 + "\n")

        # 清理音频系统
        try:
            self.audio_controller.cleanup()
            print("✓ 音频系统已清理")
        except Exception as e:
            print(f"⚠ 音频清理警告: {e}")

        # 停止摄像头
        try:
            self.pose_detector.stop_camera()
            print("✓ 摄像头已停止")
        except Exception as e:
            print(f"⚠ 摄像头停止警告: {e}")

        # 关闭OpenCV窗口
        try:
            cv2.destroyAllWindows()
            print("✓ 窗口已关闭")
        except Exception as e:
            print(f"⚠ 窗口关闭警告: {e}")

        # 退出pygame
        try:
            pygame.quit()
            print("✓ Pygame已退出")
        except Exception as e:
            print(f"⚠ Pygame退出警告: {e}")

        print("\n" + "="*60)
        print("应用已完全退出，感谢使用！")
        print("="*60 + "\n")


def main():
    """主函数入口"""
    print("\n" + "="*70)
    print(" "*15 + "E_Major 人体姿态音频控制系统")
    print(" "*20 + "版本 1.0 - 2024")
    print("="*70)

    try:
        # 创建并启动应用
        app = EMajorApp()
        app.start()

    except KeyboardInterrupt:
        print("\n\n用户中断（Ctrl+C），正在退出...")

    except Exception as e:
        print(f"\n✗ 应用错误: {e}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()

    finally:
        print("\n程序结束。")


if __name__ == "__main__":
    main()
