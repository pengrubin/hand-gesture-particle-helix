#!/usr/bin/env python3
"""
Optimized Audio Controller
优化版音频控制器 - 针对低延迟和高性能优化

主要优化策略：
1. 专用音频线程分离
2. 优化pygame混音器配置
3. 音量渐变算法优化（15FPS更新）
4. 内存预分配和对象池
5. macOS Core Audio优化

性能目标：
- 音频延迟：<50ms
- 音量更新：15FPS
- CPU使用：<30%（音频部分）
- 内存增长：<50MB/小时

Author: Performance Engineer
Date: 2025-10-05
"""

import pygame
import threading
import time
import os
import logging
import platform
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import queue
import gc
import weakref


class AudioThread(threading.Thread):
    """专用音频处理线程"""

    def __init__(self, controller: 'OptimizedAudioController'):
        super().__init__(daemon=True, name="AudioThread")
        self.controller = weakref.ref(controller)  # 弱引用避免循环引用
        self.is_running = False
        self.update_rate = 15  # 15FPS音量更新
        self.command_queue: queue.Queue = queue.Queue(maxsize=100)
        self.last_update_time = 0.0

    def run(self):
        """音频线程主循环"""
        self.is_running = True
        frame_duration = 1.0 / self.update_rate

        while self.is_running:
            frame_start = time.time()

            # 处理命令队列
            self._process_commands()

            # 更新音频状态
            controller = self.controller()
            if controller:
                controller._update_audio_internal()

            # 控制帧率
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_commands(self):
        """处理音频命令队列"""
        try:
            while True:
                command = self.command_queue.get_nowait()
                if command is None:  # 停止信号
                    self.is_running = False
                    break

                cmd_type, *args = command
                controller = self.controller()
                if not controller:
                    break

                if cmd_type == "set_volume":
                    track_name, volume = args
                    controller._set_track_volume_immediate(track_name, volume)
                elif cmd_type == "set_region_volume":
                    region_id, volume = args
                    controller._set_region_volume_immediate(region_id, volume)

        except queue.Empty:
            pass

    def send_command(self, command: Tuple[str, ...]) -> bool:
        """发送音频命令"""
        try:
            self.command_queue.put_nowait(command)
            return True
        except queue.Full:
            return False

    def stop(self):
        """停止音频线程"""
        self.send_command((None,))
        self.join(timeout=2.0)


@dataclass
class OptimizedAudioTrack:
    """优化的音频轨道"""
    name: str
    file_path: str
    sound: Optional[pygame.mixer.Sound]
    channel: Optional[pygame.mixer.Channel]
    current_volume: float
    target_volume: float
    volume_velocity: float  # 音量变化速度
    position: Tuple[float, float]
    is_active: bool
    last_update_time: float
    fade_samples: int = 0  # 渐变采样数
    priority: int = 1  # 声部优先级


class VolumeInterpolator:
    """优化的音量插值器"""

    def __init__(self, sample_rate: int = 44100, fade_duration_ms: int = 50):
        self.sample_rate = sample_rate
        self.fade_duration_ms = fade_duration_ms
        self.fade_samples = int((fade_duration_ms / 1000.0) * sample_rate)

        # 预计算插值查找表
        self.interpolation_lut = self._generate_interpolation_lut()

    def _generate_interpolation_lut(self, size: int = 1024) -> np.ndarray:
        """生成插值查找表"""
        # 使用平滑的S曲线插值
        x = np.linspace(0, 1, size)
        # Smoothstep函数：3x² - 2x³
        return 3 * x**2 - 2 * x**3

    def interpolate_volume(self, current: float, target: float, progress: float) -> float:
        """使用查找表进行快速音量插值"""
        if abs(target - current) < 0.001:
            return target

        # 限制progress到[0, 1]范围
        progress = max(0.0, min(1.0, progress))

        # 使用查找表
        lut_index = int(progress * (len(self.interpolation_lut) - 1))
        smooth_progress = self.interpolation_lut[lut_index]

        return current + smooth_progress * (target - current)


class AudioMemoryPool:
    """音频内存池管理"""

    def __init__(self, max_buffers: int = 50):
        self.max_buffers = max_buffers
        self.buffer_pool: List[np.ndarray] = []
        self.lock = threading.Lock()

    def get_buffer(self, size: int, dtype=np.float32) -> np.ndarray:
        """获取缓冲区"""
        with self.lock:
            if self.buffer_pool:
                buffer = self.buffer_pool.pop()
                if buffer.size >= size:
                    return buffer[:size].reshape(-1)

        # 创建新缓冲区
        return np.zeros(size, dtype=dtype)

    def return_buffer(self, buffer: np.ndarray):
        """归还缓冲区"""
        with self.lock:
            if len(self.buffer_pool) < self.max_buffers:
                buffer.fill(0)  # 清零
                self.buffer_pool.append(buffer)


class OptimizedAudioController:
    """优化版音频控制器"""

    def __init__(self, audio_dir: str = None, optimization_config: Optional[Dict[str, Any]] = None):
        """
        初始化优化音频控制器

        Args:
            audio_dir: 音频文件目录
            optimization_config: 优化配置
        """
        # 设置日志
        self._setup_logging()

        # 优化配置
        self.optimization_config = {
            'buffer_size': 256,  # 减小缓冲区以降低延迟
            'audio_thread_priority': True,  # 启用高优先级音频线程
            'volume_update_rate': 15,  # 15FPS音量更新
            'memory_pool_enabled': True,  # 启用内存池
            'fade_duration_ms': 30,  # 30ms音量渐变
            'preload_audio': True,  # 预加载音频
            'use_hardware_acceleration': True,  # 使用硬件加速
            'cpu_affinity': None,  # CPU亲和性设置
            'macos_core_audio': platform.system() == 'Darwin'  # macOS Core Audio
        }

        if optimization_config:
            self.optimization_config.update(optimization_config)

        # 音频文件目录
        if audio_dir is None:
            self.audio_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.audio_dir = audio_dir

        # 音轨文件定义
        self.track_files = [
            "Tromba_I_in_D.mp3",
            "Tromba_II_in_D.mp3",
            "Tromba_III_in_D.mp3",
            "Violins_in_D.mp3",
            "Viola_in_D.mp3",
            "Oboe_I_in_D.mp3",
            "Continuo_in_D.mp3",
            "Organo_obligato_in_D.mp3",
            "Timpani_in_D.mp3"
        ]

        # 7个声部区域映射
        self.region_mapping = {
            1: ["Tromba_I_in_D", "Tromba_II_in_D", "Tromba_III_in_D"],
            2: ["Violins_in_D"],
            3: ["Viola_in_D"],
            4: ["Oboe_I_in_D"],
            5: ["Continuo_in_D"],
            6: ["Organo_obligato_in_D"],
            7: ["Timpani_in_D"]
        }

        # 声部优先级（高优先级声部获得更好的音频质量）
        self.voice_priorities = {
            1: 3,  # Tromba组合 - 高优先级
            2: 3,  # Violins - 高优先级
            3: 2,  # Viola - 中优先级
            4: 2,  # Oboe - 中优先级
            5: 2,  # Continuo - 中优先级
            6: 1,  # Organo - 低优先级
            7: 3   # Timpani - 高优先级
        }

        # 初始化组件
        self.tracks: Dict[str, OptimizedAudioTrack] = {}
        self.volume_interpolator = VolumeInterpolator(
            fade_duration_ms=self.optimization_config['fade_duration_ms']
        )

        # 内存池
        self.memory_pool = AudioMemoryPool() if self.optimization_config['memory_pool_enabled'] else None

        # 播放状态
        self.playback_state = "stopped"
        self.is_initialized = False

        # 音频线程
        self.audio_thread: Optional[AudioThread] = None

        # 性能监控
        self.performance_stats = {
            "audio_latency_ms": 0.0,
            "cpu_usage_percent": 0.0,
            "memory_usage_mb": 0.0,
            "buffer_underruns": 0,
            "volume_updates_per_sec": 0.0,
            "active_channels": 0
        }

        # 线程同步
        self._volume_lock = threading.RLock()
        self._state_lock = threading.Lock()

        # 初始化pygame
        self._initialize_pygame_optimized()

    def _setup_logging(self):
        """设置优化的日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('OptimizedAudioController')

    def _initialize_pygame_optimized(self):
        """初始化优化的pygame mixer"""
        try:
            buffer_size = self.optimization_config['buffer_size']

            # macOS特定优化
            if self.optimization_config['macos_core_audio']:
                self._setup_macos_audio()

            # 设置音频驱动
            if platform.system() == "Darwin":
                os.environ['SDL_AUDIODRIVER'] = 'coreaudio'
            elif platform.system() == "Windows":
                os.environ['SDL_AUDIODRIVER'] = 'directsound'
            elif platform.system() == "Linux":
                os.environ['SDL_AUDIODRIVER'] = 'alsa'

            # 高性能pygame初始化
            pygame.mixer.pre_init(
                frequency=44100,
                size=-16,  # 16位有符号
                channels=2,  # 立体声
                buffer=buffer_size,  # 小缓冲区低延迟
                devicename=None,  # 使用默认设备
                allowedchanges=0  # 不允许格式更改以确保一致性
            )

            pygame.mixer.init()

            # 设置更多混音通道
            num_channels = max(len(self.track_files) * 2, 32)
            pygame.mixer.set_num_channels(num_channels)

            # 设置线程优先级
            if self.optimization_config['audio_thread_priority']:
                self._set_thread_priority()

            self.logger.info(f"Optimized pygame mixer initialized: buffer={buffer_size}, channels={num_channels}")

        except Exception as e:
            self.logger.error(f"Failed to initialize optimized mixer: {e}")
            raise

    def _setup_macos_audio(self):
        """设置macOS特定的音频优化"""
        try:
            # 设置Core Audio特定环境变量
            os.environ['CA_DISABLE_DENORMALS'] = '1'  # 禁用非正规化数处理
            os.environ['CA_PREFER_FIXED_LATENCY'] = '1'  # 偏好固定延迟

            # 尝试设置低延迟音频单元
            import subprocess
            subprocess.run([
                'sudo', 'sysctl', '-w', 'kern.audio.latency_us=1000'
            ], check=False, capture_output=True)

        except Exception as e:
            self.logger.warning(f"macOS audio optimization failed: {e}")

    def _set_thread_priority(self):
        """设置音频线程优先级"""
        try:
            if platform.system() == "Darwin":
                # macOS实时优先级
                import ctypes
                libc = ctypes.CDLL("libc.dylib")
                libc.setpriority(0, 0, -10)  # 高优先级
            elif platform.system() == "Linux":
                # Linux实时调度
                import os
                os.nice(-10)
        except Exception as e:
            self.logger.warning(f"Failed to set thread priority: {e}")

    def initialize(self) -> bool:
        """初始化优化音频系统"""
        try:
            self.playback_state = "loading"
            self.logger.info("Initializing optimized audio system...")

            # 预分配内存
            if self.memory_pool:
                for _ in range(20):  # 预分配20个缓冲区
                    buffer = np.zeros(4096, dtype=np.float32)
                    self.memory_pool.return_buffer(buffer)

            # 加载音频文件
            load_start = time.time()
            loaded_count = 0

            for i, filename in enumerate(self.track_files):
                track_name = filename.replace('.mp3', '')
                file_path = os.path.join(self.audio_dir, filename)

                if not os.path.exists(file_path):
                    self.logger.warning(f"Audio file not found: {file_path}")
                    continue

                try:
                    # 确定优先级
                    priority = 1
                    for region_id, tracks in self.region_mapping.items():
                        if track_name in tracks:
                            priority = self.voice_priorities.get(region_id, 1)
                            break

                    # 创建优化的音轨对象
                    track = OptimizedAudioTrack(
                        name=track_name,
                        file_path=file_path,
                        sound=None,
                        channel=None,
                        current_volume=0.0,
                        target_volume=0.0,
                        volume_velocity=0.0,
                        position=(0.5, 0.5),
                        is_active=False,
                        last_update_time=time.time(),
                        priority=priority
                    )

                    # 预加载音频
                    if self.optimization_config['preload_audio']:
                        track.sound = pygame.mixer.Sound(file_path)
                        track.channel = pygame.mixer.Channel(i)

                        # 设置音频属性
                        track.sound.set_volume(0.0)

                    self.tracks[track_name] = track
                    loaded_count += 1

                    self.logger.info(f"Loaded optimized track: {track_name} (priority: {priority})")

                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
                    continue

            load_time = time.time() - load_start

            if loaded_count < 3:
                self.logger.error(f"Insufficient tracks loaded: {loaded_count}")
                return False

            # 启动音频线程
            self.audio_thread = AudioThread(self)
            self.audio_thread.start()

            # 设置CPU亲和性
            if self.optimization_config['cpu_affinity']:
                self._set_cpu_affinity()

            self.is_initialized = True
            self.playback_state = "stopped"

            self.logger.info(f"Optimized audio controller initialized: {loaded_count} tracks in {load_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.playback_state = "stopped"
            return False

    def _set_cpu_affinity(self):
        """设置CPU亲和性"""
        try:
            if platform.system() == "Linux":
                import os
                # 绑定到最后一个CPU核心（通常较少被使用）
                cpu_count = os.cpu_count()
                if cpu_count > 1:
                    os.sched_setaffinity(0, {cpu_count - 1})
        except Exception as e:
            self.logger.warning(f"Failed to set CPU affinity: {e}")

    def start_playback(self) -> bool:
        """开始优化播放"""
        if not self.is_initialized:
            self.logger.error("Audio controller not initialized")
            return False

        try:
            with self._state_lock:
                # 同时开始播放所有音轨
                for track in self.tracks.values():
                    if track.sound and track.channel:
                        track.channel.play(track.sound, loops=-1)
                        track.channel.set_volume(0.0)
                        track.is_active = True
                        track.last_update_time = time.time()

                self.playback_state = "playing"

            self.logger.info("Optimized playback started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start playback: {e}")
            return False

    def stop_playback(self):
        """停止播放"""
        with self._state_lock:
            for track in self.tracks.values():
                if track.channel:
                    track.channel.stop()
                track.is_active = False
                track.current_volume = 0.0
                track.target_volume = 0.0

            self.playback_state = "stopped"

        self.logger.info("Playback stopped")

    def set_region_volume(self, region_id: int, volume: float):
        """设置声部区域音量（线程安全）"""
        if self.audio_thread:
            self.audio_thread.send_command(("set_region_volume", region_id, volume))

    def _set_region_volume_immediate(self, region_id: int, volume: float):
        """立即设置区域音量（在音频线程中调用）"""
        if region_id in self.region_mapping:
            track_names = self.region_mapping[region_id]
            for track_name in track_names:
                if track_name in self.tracks:
                    with self._volume_lock:
                        self.tracks[track_name].target_volume = max(0.0, min(1.0, volume))

    def set_track_volume(self, track_name: str, volume: float):
        """设置音轨音量（线程安全）"""
        if self.audio_thread:
            self.audio_thread.send_command(("set_volume", track_name, volume))

    def _set_track_volume_immediate(self, track_name: str, volume: float):
        """立即设置音轨音量（在音频线程中调用）"""
        if track_name in self.tracks:
            with self._volume_lock:
                self.tracks[track_name].target_volume = max(0.0, min(1.0, volume))

    def _update_audio_internal(self):
        """内部音频更新（在音频线程中调用）"""
        current_time = time.time()
        updates_count = 0

        with self._volume_lock:
            for track in self.tracks.values():
                if not track.is_active:
                    continue

                delta_time = current_time - track.last_update_time
                track.last_update_time = current_time

                # 计算音量插值进度
                volume_diff = abs(track.target_volume - track.current_volume)
                if volume_diff > 0.001:
                    # 使用优化的插值算法
                    fade_duration = self.optimization_config['fade_duration_ms'] / 1000.0
                    progress = min(1.0, delta_time / fade_duration)

                    new_volume = self.volume_interpolator.interpolate_volume(
                        track.current_volume, track.target_volume, progress
                    )

                    track.current_volume = new_volume

                    # 应用到pygame通道
                    if track.channel:
                        track.channel.set_volume(new_volume)

                    updates_count += 1

        # 更新性能统计
        if updates_count > 0:
            self.performance_stats["volume_updates_per_sec"] = updates_count * self.optimization_config['volume_update_rate']

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()

        # 添加实时统计
        stats["active_tracks"] = sum(1 for track in self.tracks.values() if track.is_active)
        stats["total_tracks"] = len(self.tracks)

        # 内存使用情况
        try:
            import psutil
            process = psutil.Process()
            stats["memory_usage_mb"] = process.memory_info().rss / (1024**2)
            stats["cpu_usage_percent"] = process.cpu_percent()
        except:
            pass

        # 音频延迟估算
        buffer_size = self.optimization_config['buffer_size']
        sample_rate = 44100
        estimated_latency = (buffer_size / sample_rate) * 1000  # ms
        stats["audio_latency_ms"] = estimated_latency

        return stats

    def cleanup(self):
        """清理优化音频系统"""
        self.logger.info("Cleaning up optimized audio controller...")

        # 停止播放
        self.stop_playback()

        # 停止音频线程
        if self.audio_thread:
            self.audio_thread.stop()

        # 清理pygame
        pygame.mixer.quit()

        # 清理内存池
        if self.memory_pool:
            self.memory_pool.buffer_pool.clear()

        # 清理音轨
        self.tracks.clear()
        self.is_initialized = False

        # 强制垃圾回收
        gc.collect()

        self.logger.info("Optimized audio controller cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 工厂函数
def create_optimized_audio_controller(audio_dir: str = None,
                                    optimization_level: str = "balanced") -> OptimizedAudioController:
    """
    创建优化音频控制器

    Args:
        audio_dir: 音频文件目录
        optimization_level: 优化级别 ("low_latency", "balanced", "high_quality")

    Returns:
        优化的音频控制器实例
    """
    optimization_configs = {
        "low_latency": {
            'buffer_size': 128,
            'fade_duration_ms': 20,
            'volume_update_rate': 20,
            'audio_thread_priority': True,
            'use_hardware_acceleration': True
        },
        "balanced": {
            'buffer_size': 256,
            'fade_duration_ms': 30,
            'volume_update_rate': 15,
            'audio_thread_priority': True,
            'use_hardware_acceleration': True
        },
        "high_quality": {
            'buffer_size': 512,
            'fade_duration_ms': 50,
            'volume_update_rate': 10,
            'audio_thread_priority': False,
            'use_hardware_acceleration': False
        }
    }

    config = optimization_configs.get(optimization_level, optimization_configs["balanced"])
    return OptimizedAudioController(audio_dir, config)


def main():
    """测试优化音频控制器"""
    print("Optimized Audio Controller Test")
    print("=" * 40)

    audio_dir = os.path.dirname(os.path.abspath(__file__))

    with create_optimized_audio_controller(audio_dir, "low_latency") as controller:
        if not controller.initialize():
            print("Failed to initialize optimized audio controller")
            return

        print("Optimized audio controller initialized successfully")
        print(f"Loaded {len(controller.tracks)} tracks")

        if controller.start_playback():
            print("\nPlayback started. Testing optimized controls...")

            try:
                # 性能测试序列
                test_duration = 10
                start_time = time.time()

                while time.time() - start_time < test_duration:
                    # 循环测试不同声部
                    for region_id in range(1, 8):
                        controller.set_region_volume(region_id, 0.8)
                        time.sleep(0.5)
                        controller.set_region_volume(region_id, 0.0)
                        time.sleep(0.5)

                    # 显示性能统计
                    stats = controller.get_performance_stats()
                    print(f"\nPerformance Stats:")
                    print(f"  Audio Latency: {stats.get('audio_latency_ms', 0):.1f}ms")
                    print(f"  CPU Usage: {stats.get('cpu_usage_percent', 0):.1f}%")
                    print(f"  Memory Usage: {stats.get('memory_usage_mb', 0):.1f}MB")
                    print(f"  Volume Updates/sec: {stats.get('volume_updates_per_sec', 0):.1f}")

            except KeyboardInterrupt:
                print("\nTest interrupted")

        else:
            print("Failed to start playback")

    print("\nOptimized test completed")


if __name__ == "__main__":
    main()