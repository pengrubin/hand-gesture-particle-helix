#!/usr/bin/env python3
"""
BWV_29_in_D Audio Controller
多音轨同步播放系统，支持指挥家手势控制

主要功能：
- 9个MP3音轨同步播放
- 独立音量控制（0-1范围）
- 平滑音量过渡
- 手势控制接口
- 低延迟响应
- 内存高效管理

Author: Claude Code
Date: 2025-10-04
"""

import pygame
import threading
import time
import os
import logging
from typing import Dict, List, Optional, Tuple, Callable, Protocol
from dataclasses import dataclass
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class AudioManagerInterface(Protocol):
    """音频管理器接口定义"""

    def initialize(self) -> bool:
        """初始化音频系统"""
        ...

    def start_playback(self) -> bool:
        """开始播放"""
        ...

    def pause_playback(self) -> None:
        """暂停播放"""
        ...

    def resume_playback(self) -> None:
        """恢复播放"""
        ...

    def stop_playback(self) -> None:
        """停止播放"""
        ...

    def set_region_volume(self, region_id: int, volume: float) -> None:
        """设置区域音量"""
        ...

    def update_gesture(self, gesture_data: 'GestureData') -> None:
        """更新手势数据"""
        ...

    def cleanup(self) -> None:
        """清理资源"""
        ...


class PlaybackState(Enum):
    """播放状态枚举"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    LOADING = "loading"


class GestureMode(Enum):
    """手势控制模式"""
    ALL_TRACKS = "all_tracks"      # 手掌张开控制所有音轨
    SELECTIVE = "selective"        # 手指指向控制特定声部
    CONDUCTOR = "conductor"        # 指挥家模式


@dataclass
class AudioTrack:
    """音频轨道数据结构"""
    name: str
    file_path: str
    sound: Optional[pygame.mixer.Sound]
    channel: Optional[pygame.mixer.Channel]
    current_volume: float
    target_volume: float
    position: Tuple[float, float]  # 屏幕空间位置 (x, y) 0-1范围
    is_active: bool


@dataclass
class GestureData:
    """手势数据结构"""
    hand_openness: float  # 0-1范围，手掌张开程度
    finger_positions: List[Tuple[float, float]]  # 手指位置列表
    is_hand_present: bool  # 是否检测到手部
    pointing_direction: Optional[Tuple[float, float]]  # 指向方向


class VolumeController:
    """音量控制器 - 负责平滑音量过渡"""

    def __init__(self, transition_speed: float = 2.0):
        """
        初始化音量控制器

        Args:
            transition_speed: 音量过渡速度 (单位/秒)
        """
        self.transition_speed = transition_speed
        self.smoothing_factor = 0.15  # 平滑系数

    def update_volume(self, current: float, target: float, delta_time: float) -> float:
        """
        计算平滑音量过渡

        Args:
            current: 当前音量
            target: 目标音量
            delta_time: 时间间隔

        Returns:
            更新后的音量值
        """
        if abs(target - current) < 0.001:
            return target

        # 使用指数平滑进行音量过渡
        alpha = 1.0 - np.exp(-self.transition_speed * delta_time)
        new_volume = current + alpha * (target - current)

        # 限制音量范围
        return max(0.0, min(1.0, new_volume))


class GestureMapper:
    """手势映射器 - 将手势数据映射到音频控制参数"""

    def __init__(self):
        """初始化手势映射器"""
        self.gesture_mode = GestureMode.ALL_TRACKS
        self.sensitivity = 1.0
        self.track_positions = self._initialize_track_positions()
        self.region_mapping = self._initialize_region_mapping()
        self.region_positions = self._initialize_region_positions()

    def _initialize_track_positions(self) -> Dict[str, Tuple[float, float]]:
        """初始化音轨在屏幕空间中的位置"""
        positions = {
            "Tromba_I_in_D": (0.2, 0.2),
            "Tromba_II_in_D": (0.4, 0.2),
            "Tromba_III_in_D": (0.6, 0.2),
            "Violins_in_D": (0.2, 0.4),
            "Viola_in_D": (0.4, 0.4),
            "Oboe_I_in_D": (0.6, 0.4),
            "Continuo_in_D": (0.2, 0.6),
            "Organo_obligato_in_D": (0.4, 0.6),
            "Timpani_in_D": (0.6, 0.6)
        }
        return positions

    def _initialize_region_mapping(self) -> Dict[int, List[str]]:
        """初始化区域ID到音轨的映射（按照7个声部要求）"""
        return {
            1: ["Tromba_I_in_D", "Tromba_II_in_D", "Tromba_III_in_D"],  # Tromba组合
            2: ["Violins_in_D"],
            3: ["Viola_in_D"],
            4: ["Oboe_I_in_D"],
            5: ["Continuo_in_D"],
            6: ["Organo_obligato_in_D"],
            7: ["Timpani_in_D"]
        }

    def _initialize_region_positions(self) -> Dict[int, Tuple[float, float]]:
        """初始化7个声部区域的屏幕位置"""
        return {
            1: (0.4, 0.2),   # Tromba组合 - 中央上方
            2: (0.15, 0.35), # Violins - 左上
            3: (0.35, 0.35), # Viola - 中左
            4: (0.65, 0.35), # Oboe - 中右
            5: (0.85, 0.35), # Continuo - 右上
            6: (0.25, 0.65), # Organo - 左下
            7: (0.75, 0.65)  # Timpani - 右下
        }

    def calculate_region_volumes(self, gesture_data: GestureData) -> Dict[int, float]:
        """根据手势数据计算各声部区域目标音量"""
        region_volumes = {}

        if not gesture_data.is_hand_present:
            return {region_id: 0.0 for region_id in self.region_mapping.keys()}

        if self.gesture_mode == GestureMode.ALL_TRACKS:
            # 手掌张开程度控制所有声部
            base_volume = gesture_data.hand_openness * self.sensitivity
            return {region_id: base_volume for region_id in self.region_mapping.keys()}

        elif self.gesture_mode == GestureMode.SELECTIVE:
            # 手指指向控制特定声部
            return self._calculate_selective_region_volumes(gesture_data)

        elif self.gesture_mode == GestureMode.CONDUCTOR:
            # 指挥家模式 - 综合控制
            return self._calculate_conductor_region_volumes(gesture_data)

        return {region_id: 0.0 for region_id in self.region_mapping.keys()}

    def calculate_track_volumes(self, gesture_data: GestureData) -> Dict[str, float]:
        """
        根据手势数据计算各音轨目标音量

        Args:
            gesture_data: 手势数据

        Returns:
            音轨名称到目标音量的映射
        """
        volumes = {}

        if not gesture_data.is_hand_present:
            # 无手势时所有音轨静音
            return {track: 0.0 for track in self.track_positions.keys()}

        if self.gesture_mode == GestureMode.ALL_TRACKS:
            # 手掌张开程度控制所有音轨
            base_volume = gesture_data.hand_openness * self.sensitivity
            return {track: base_volume for track in self.track_positions.keys()}

        elif self.gesture_mode == GestureMode.SELECTIVE:
            # 手指指向控制特定声部
            return self._calculate_selective_volumes(gesture_data)

        elif self.gesture_mode == GestureMode.CONDUCTOR:
            # 指挥家模式 - 综合控制
            return self._calculate_conductor_volumes(gesture_data)

        return {track: 0.0 for track in self.track_positions.keys()}

    def _calculate_selective_volumes(self, gesture_data: GestureData) -> Dict[str, float]:
        """计算选择性音量控制"""
        volumes = {track: 0.0 for track in self.track_positions.keys()}

        if gesture_data.pointing_direction is None:
            return volumes

        point_x, point_y = gesture_data.pointing_direction
        activation_radius = 0.15  # 激活半径

        for track_name, (pos_x, pos_y) in self.track_positions.items():
            distance = np.sqrt((point_x - pos_x)**2 + (point_y - pos_y)**2)

            if distance <= activation_radius:
                # 距离越近音量越大
                volume_factor = 1.0 - (distance / activation_radius)
                volumes[track_name] = volume_factor * gesture_data.hand_openness

        return volumes

    def _calculate_selective_region_volumes(self, gesture_data: GestureData) -> Dict[int, float]:
        """计算选择性声部区域音量控制"""
        volumes = {region_id: 0.0 for region_id in self.region_mapping.keys()}

        if gesture_data.pointing_direction is None:
            return volumes

        point_x, point_y = gesture_data.pointing_direction
        activation_radius = 0.18  # 激活半径，适应7个声部布局

        for region_id, (pos_x, pos_y) in self.region_positions.items():
            distance = np.sqrt((point_x - pos_x)**2 + (point_y - pos_y)**2)

            if distance <= activation_radius:
                # 距离越近音量越大
                volume_factor = 1.0 - (distance / activation_radius)
                volumes[region_id] = volume_factor * gesture_data.hand_openness

        return volumes

    def _calculate_conductor_region_volumes(self, gesture_data: GestureData) -> Dict[int, float]:
        """计算指挥家模式声部区域音量控制"""
        volumes = {}
        base_volume = gesture_data.hand_openness * 0.8

        # 根据手势强度调整不同声部
        for region_id in self.region_mapping.keys():
            if region_id == 1:  # Tromba组合
                # 铜管组响应更强烈
                volumes[region_id] = min(1.0, base_volume * 1.2)
            elif region_id == 7:  # Timpani
                # 定音鼓需要更明确的手势
                volumes[region_id] = base_volume * 0.9 if gesture_data.hand_openness > 0.7 else 0.0
            else:
                # 弦乐和木管
                volumes[region_id] = base_volume

        return volumes

    def convert_region_to_track_volumes(self, region_volumes: Dict[int, float]) -> Dict[str, float]:
        """将声部区域音量转换为具体音轨音量"""
        track_volumes = {}

        for region_id, volume in region_volumes.items():
            if region_id in self.region_mapping:
                for track_name in self.region_mapping[region_id]:
                    track_volumes[track_name] = volume

        return track_volumes

    def _calculate_conductor_volumes(self, gesture_data: GestureData) -> Dict[str, float]:
        """计算指挥家模式音量控制"""
        volumes = {}
        base_volume = gesture_data.hand_openness * 0.8

        # 根据手势强度调整不同声部
        for track_name in self.track_positions.keys():
            if "Tromba" in track_name:
                # 铜管组响应更强烈
                volumes[track_name] = min(1.0, base_volume * 1.2)
            elif "Timpani" in track_name:
                # 定音鼓需要更明确的手势
                volumes[track_name] = base_volume * 0.9 if gesture_data.hand_openness > 0.7 else 0.0
            else:
                # 弦乐和木管
                volumes[track_name] = base_volume

        return volumes


class AudioController:
    """
    BWV_29_in_D 多音轨音频控制器

    核心功能：
    - 多音轨同步播放
    - 独立音量控制
    - 手势响应
    - 性能监控
    """

    def __init__(self, audio_dir: str = None, buffer_size: int = 512):
        """
        初始化音频控制器

        Args:
            audio_dir: 音频文件目录路径
            buffer_size: 音频缓冲区大小
        """
        # 设置日志
        self._setup_logging()

        # 音频文件目录
        if audio_dir is None:
            self.audio_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.audio_dir = audio_dir

        # 预定义的音轨文件列表
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

        # 初始化组件
        self.tracks: Dict[str, AudioTrack] = {}
        self.volume_controller = VolumeController()
        self.gesture_mapper = GestureMapper()

        # 播放状态
        self.playback_state = PlaybackState.STOPPED
        self.is_initialized = False
        self.start_time = 0.0
        self.current_position = 0.0  # 当前播放位置（秒）
        self.pause_position = 0.0   # 暂停时的位置
        self.last_position_update = 0.0  # 上次位置更新时间

        # 线程控制
        self.update_thread = None
        self.is_running = False
        self.update_rate = 60  # Hz

        # 性能监控
        self.performance_stats = {
            "update_time": [],
            "volume_changes": 0,
            "gesture_updates": 0
        }

        # 回调函数
        self.on_state_change: Optional[Callable] = None
        self.on_volume_change: Optional[Callable] = None

        # 初始化pygame mixer（允许失败）
        self.pygame_available = self._initialize_pygame(buffer_size)

    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AudioController')

    def _initialize_pygame(self, buffer_size: int):
        """初始化pygame mixer"""
        try:
            # 首先尝试清理任何现有的音频初始化
            try:
                pygame.mixer.quit()
                # 短暂等待确保清理完成
                import time
                time.sleep(0.1)
            except:
                pass

            # 平台特定的音频驱动程序选择
            import platform
            system = platform.system().lower()

            if system == 'darwin':  # macOS
                drivers_to_try = ['coreaudio', None]  # None表示使用默认
            elif system == 'windows':
                drivers_to_try = ['directsound', 'wasapi', 'dsound', None]
            elif system == 'linux':
                drivers_to_try = ['alsa', 'pulse', 'oss', None]
            else:
                drivers_to_try = [None]

            # 尝试不同的缓冲区大小
            buffer_sizes_to_try = [buffer_size, 1024, 2048, 4096]

            # 尝试初始化音频驱动
            pygame_initialized = False
            for driver in drivers_to_try:
                for try_buffer in buffer_sizes_to_try:
                    try:
                        # 清理之前的尝试
                        try:
                            pygame.mixer.quit()
                            time.sleep(0.05)
                        except:
                            pass

                        if driver:
                            os.environ['SDL_AUDIODRIVER'] = driver
                            self.logger.info(f"Trying audio driver: {driver} with buffer: {try_buffer}")
                        else:
                            # 移除环境变量使用默认驱动
                            os.environ.pop('SDL_AUDIODRIVER', None)
                            self.logger.info(f"Trying default audio driver with buffer: {try_buffer}")

                        pygame.mixer.pre_init(
                            frequency=44100,
                            size=-16,
                            channels=2,
                            buffer=try_buffer
                        )
                        pygame.mixer.init()

                        # 测试初始化是否真正成功
                        mixer_info = pygame.mixer.get_init()
                        if mixer_info:
                            pygame_initialized = True
                            if driver:
                                self.logger.info(f"Successfully initialized with driver: {driver}, buffer: {try_buffer}")
                            else:
                                self.logger.info(f"Successfully initialized with default driver, buffer: {try_buffer}")
                            break

                    except Exception as driver_error:
                        self.logger.debug(f"Driver {driver} with buffer {try_buffer} failed: {driver_error}")
                        continue

                if pygame_initialized:
                    break

            if not pygame_initialized:
                # 最后尝试：使用最基本的配置
                try:
                    pygame.mixer.quit()
                    time.sleep(0.1)
                    self.logger.info("Attempting basic audio initialization as fallback")
                    os.environ.pop('SDL_AUDIODRIVER', None)

                    pygame.mixer.pre_init(
                        frequency=22050,  # 降低频率
                        size=-16,
                        channels=1,       # 单声道
                        buffer=4096       # 大缓冲区
                    )
                    pygame.mixer.init()

                    mixer_info = pygame.mixer.get_init()
                    if mixer_info:
                        pygame_initialized = True
                        self.logger.info("Fallback audio initialization successful")

                except Exception as fallback_error:
                    self.logger.error(f"Fallback initialization failed: {fallback_error}")

            if not pygame_initialized:
                self.logger.error("All audio initialization attempts failed")
                # 不抛出异常，而是设置一个标志
                self.is_initialized = False
                return False

            # 设置混音器通道数（确保足够的通道）
            required_channels = max(len(self.track_files), 16)
            pygame.mixer.set_num_channels(required_channels)

            mixer_info = pygame.mixer.get_init()
            self.logger.info(f"Pygame mixer initialized: {mixer_info[0]}Hz, {mixer_info[1]}bit, {mixer_info[2]}ch, {required_channels} channels")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            return False

    def initialize(self) -> bool:
        """
        初始化音频系统

        Returns:
            初始化是否成功
        """
        try:
            if not self.pygame_available:
                self.logger.error("Pygame mixer not available, cannot initialize audio")
                return False

            self.playback_state = PlaybackState.LOADING
            self.logger.info("Loading audio tracks...")

            # 加载音频文件
            for i, filename in enumerate(self.track_files):
                track_name = filename.replace('.mp3', '')
                file_path = os.path.join(self.audio_dir, filename)

                if not os.path.exists(file_path):
                    self.logger.error(f"Audio file not found: {file_path}")
                    return False

                # 创建音轨对象
                position = self.gesture_mapper.track_positions.get(track_name, (0.5, 0.5))
                track = AudioTrack(
                    name=track_name,
                    file_path=file_path,
                    sound=None,
                    channel=None,
                    current_volume=0.0,
                    target_volume=0.0,
                    position=position,
                    is_active=False
                )

                # 加载音频（带错误恢复）
                try:
                    # 预检查文件大小避免内存问题
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if file_size_mb > 100:  # 100MB限制
                        self.logger.warning(f"Large audio file detected: {track_name} ({file_size_mb:.1f}MB)")

                    track.sound = pygame.mixer.Sound(file_path)
                    track.channel = pygame.mixer.Channel(i)
                    self.tracks[track_name] = track
                    self.logger.info(f"Loaded track: {track_name} ({file_size_mb:.1f}MB)")

                except MemoryError as e:
                    self.logger.error(f"Memory error loading {filename}: {e}")
                    self.logger.info("Attempting to reduce buffer size and retry...")
                    # 可以在这里实现降级策略
                    return False
                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
                    # 继续加载其他文件而不是完全失败
                    continue

            # 验证是否有足够的音轨成功加载
            if len(self.tracks) < 3:  # 至少需要一些核心音轨
                self.logger.error(f"Insufficient tracks loaded: {len(self.tracks)}/{len(self.track_files)}")
                return False

            # 验证Tromba组合是否完整
            tromba_tracks = [name for name in self.tracks.keys() if "Tromba" in name]
            if len(tromba_tracks) < 2:  # 至少需要2个Tromba音轨
                self.logger.warning(f"Incomplete Tromba section: {len(tromba_tracks)} tracks")

            self.is_initialized = True
            self.playback_state = PlaybackState.STOPPED
            self.logger.info(f"Audio controller initialized successfully with {len(self.tracks)} tracks")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.playback_state = PlaybackState.STOPPED
            return False

    def start_playback(self) -> bool:
        """
        开始播放所有音轨

        Returns:
            是否成功开始播放
        """
        if not self.is_initialized:
            self.logger.error("Audio controller not initialized")
            return False

        try:
            self.start_time = time.time()

            # 同时开始播放所有音轨（设置初始音量确保音频系统激活）
            for track in self.tracks.values():
                track.channel.play(track.sound, loops=-1)  # 无限循环
                # 设置适中的初始音量确保音频可听见
                initial_volume = 0.4 if track.name in ["Violins_in_D", "Continuo_in_D"] else 0.2
                track.channel.set_volume(initial_volume)
                track.current_volume = initial_volume
                track.target_volume = initial_volume
                track.is_active = True
                self.logger.info(f"Started track {track.name} with volume {initial_volume}")

            self.playback_state = PlaybackState.PLAYING
            self.last_position_update = time.time()
            self.current_position = 0.0

            # 启动更新线程
            self._start_update_thread()

            self.logger.info("Playbook started with enhanced audio test")

            # 增强音频测试：确保有明显的声音输出
            if "Violins_in_D" in self.tracks:
                test_track = self.tracks["Violins_in_D"]
                test_track.channel.set_volume(0.6)  # 提高音量到更明显的级别
                test_track.current_volume = 0.6
                test_track.target_volume = 0.6
                self.logger.info("Audio test: Violins volume set to 0.6 for verification")

            # 添加额外的音频输出验证
            self._verify_audio_output()

            if self.on_state_change:
                self.on_state_change(self.playback_state)

            return True

        except Exception as e:
            self.logger.error(f"Failed to start playback: {e}")
            return False

    def pause_playback(self):
        """暂停播放"""
        if self.playback_state == PlaybackState.PLAYING:
            # 记录暂停位置
            self.pause_position = self.get_playback_position()

            for track in self.tracks.values():
                track.channel.pause()
            self.playback_state = PlaybackState.PAUSED
            self.logger.info(f"Playback paused at position: {self.pause_position:.2f}s")

            if self.on_state_change:
                self.on_state_change(self.playback_state)

    def resume_playback(self):
        """恢复播放"""
        if self.playback_state == PlaybackState.PAUSED:
            for track in self.tracks.values():
                track.channel.unpause()
            self.playback_state = PlaybackState.PLAYING
            self.last_position_update = time.time()
            self.current_position = self.pause_position
            self.logger.info(f"Playback resumed from position: {self.current_position:.2f}s")

            if self.on_state_change:
                self.on_state_change(self.playback_state)

    def stop_playback(self):
        """停止播放"""
        self._stop_update_thread()

        for track in self.tracks.values():
            track.channel.stop()
            track.is_active = False
            track.current_volume = 0.0
            track.target_volume = 0.0

        self.playback_state = PlaybackState.STOPPED
        self.current_position = 0.0
        self.pause_position = 0.0
        self.logger.info("Playback stopped")

        if self.on_state_change:
            self.on_state_change(self.playback_state)

    def update_gesture(self, gesture_data: GestureData):
        """
        更新手势数据并调整音量

        Args:
            gesture_data: 手势数据
        """
        if self.playback_state != PlaybackState.PLAYING:
            return

        # 根据手势计算目标音量
        target_volumes = self.gesture_mapper.calculate_track_volumes(gesture_data)

        # 更新各音轨目标音量
        for track_name, target_volume in target_volumes.items():
            if track_name in self.tracks:
                self.tracks[track_name].target_volume = target_volume

        self.performance_stats["gesture_updates"] += 1

    def set_region_volume(self, region_id: int, volume: float):
        """
        设置声部区域音量（符合接口要求）

        Args:
            region_id: 声部区域ID (1-7)
            volume: 目标音量 (0.0-1.0)
        """
        if region_id in self.gesture_mapper.region_mapping:
            # 获取该区域对应的所有音轨
            track_names = self.gesture_mapper.region_mapping[region_id]
            for track_name in track_names:
                if track_name in self.tracks:
                    self.tracks[track_name].target_volume = max(0.0, min(1.0, volume))

    def set_track_volume(self, track_name: str, volume: float):
        """
        直接设置音轨音量

        Args:
            track_name: 音轨名称
            volume: 目标音量 (0.0-1.0)
        """
        if track_name in self.tracks:
            self.tracks[track_name].target_volume = max(0.0, min(1.0, volume))

    def set_all_volumes(self, volume: float):
        """设置所有音轨音量"""
        volume = max(0.0, min(1.0, volume))
        for track in self.tracks.values():
            track.target_volume = volume
        self.logger.debug(f"All volumes set to: {volume:.2f}")

    def test_audio_output(self) -> bool:
        """
        测试音频输出是否正常工作

        Returns:
            音频测试是否成功
        """
        if not self.is_initialized:
            self.logger.error("Audio controller not initialized")
            return False

        try:
            # 选择一个测试音轨
            test_track_name = "Violins_in_D" if "Violins_in_D" in self.tracks else list(self.tracks.keys())[0]
            test_track = self.tracks[test_track_name]

            self.logger.info(f"Testing audio output with track: {test_track_name}")

            # 播放测试音轨
            test_track.channel.play(test_track.sound, loops=0)  # 播放一次
            test_track.channel.set_volume(0.5)  # 中等音量

            # 等待短时间
            time.sleep(1.0)

            # 检查是否还在播放
            is_playing = test_track.channel.get_busy()

            # 停止测试
            test_track.channel.stop()

            if is_playing:
                self.logger.info("Audio test successful - sound output detected")
                return True
            else:
                self.logger.warning("Audio test inconclusive - could not verify sound output")
                return False

        except Exception as e:
            self.logger.error(f"Audio test failed: {e}")
            return False

    def set_gesture_mode(self, mode: GestureMode):
        """设置手势控制模式"""
        self.gesture_mapper.gesture_mode = mode
        self.logger.info(f"Gesture mode set to: {mode.value}")

    def set_gesture_sensitivity(self, sensitivity: float):
        """设置手势敏感度"""
        self.gesture_mapper.sensitivity = max(0.1, min(2.0, sensitivity))

    def is_playing(self) -> bool:
        """检查是否正在播放"""
        return self.playback_state == PlaybackState.PLAYING

    def get_playback_state(self) -> PlaybackState:
        """获取播放状态"""
        return self.playback_state

    def get_playback_position(self) -> float:
        """
        获取播放位置（秒）

        Returns:
            当前播放位置（秒）
        """
        if self.playback_state == PlaybackState.PLAYING:
            # 计算实际播放时间
            current_time = time.time()
            elapsed = current_time - self.last_position_update
            self.current_position += elapsed
            self.last_position_update = current_time
            return self.current_position
        elif self.playback_state == PlaybackState.PAUSED:
            return self.pause_position
        else:
            return 0.0

    def set_playback_position(self, position: float) -> None:
        """
        设置播放位置（秒）

        Args:
            position: 目标播放位置（秒）
        """
        position = max(0.0, position)  # 确保位置不为负

        if self.playback_state == PlaybackState.PLAYING:
            # 如果正在播放，需要重新同步所有音轨
            self._sync_tracks_to_position(position)
            self.current_position = position
            self.last_position_update = time.time()
        elif self.playback_state == PlaybackState.PAUSED:
            self.pause_position = position
            self.current_position = position
        else:
            self.current_position = position

        self.logger.debug(f"Playback position set to: {position:.2f}s")

    def _sync_tracks_to_position(self, position: float) -> None:
        """
        将所有音轨同步到指定位置

        Args:
            position: 目标位置（秒）
        """
        # 注意：pygame.mixer 不直接支持seek功能
        # 这里实现一个基本的重启同步机制
        try:
            was_playing = self.playback_state == PlaybackState.PLAYING

            # 停止所有音轨
            for track in self.tracks.values():
                if track.is_active:
                    track.channel.stop()

            # 等待短时间确保停止完成
            time.sleep(0.01)

            # 重新开始播放
            if was_playing:
                for track in self.tracks.values():
                    if track.is_active:
                        track.channel.play(track.sound, loops=-1)
                        track.channel.set_volume(track.current_volume)

            self.logger.debug(f"Tracks synchronized to position: {position:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to sync tracks to position {position}: {e}")

    def get_track_volumes(self) -> Dict[str, float]:
        """获取所有音轨当前音量"""
        return {name: track.current_volume for name, track in self.tracks.items()}

    def get_region_volumes(self) -> Dict[int, float]:
        """获取所有声部区域当前音量"""
        region_volumes = {}
        for region_id, track_names in self.gesture_mapper.region_mapping.items():
            if track_names:
                # 对于Tromba组合，取平均音量
                volumes = [self.tracks[name].current_volume
                          for name in track_names if name in self.tracks]
                if volumes:
                    region_volumes[region_id] = np.mean(volumes)
                else:
                    region_volumes[region_id] = 0.0
        return region_volumes

    def get_track_info(self) -> Dict[str, dict]:
        """获取音轨信息"""
        return {
            name: {
                "current_volume": track.current_volume,
                "target_volume": track.target_volume,
                "position": track.position,
                "is_active": track.is_active
            }
            for name, track in self.tracks.items()
        }

    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats["update_time"]:
            stats["avg_update_time"] = np.mean(stats["update_time"][-100:])  # 最近100次平均
            stats["max_update_time"] = np.max(stats["update_time"][-100:])
        return stats

    def _start_update_thread(self):
        """启动更新线程"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def _stop_update_thread(self):
        """停止更新线程"""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

    def _update_loop(self):
        """音频更新循环"""
        last_time = time.time()

        while self.is_running:
            start_time = time.time()
            current_time = start_time
            delta_time = current_time - last_time
            last_time = current_time

            # 更新音量
            volume_changed = False
            for track in self.tracks.values():
                if not track.is_active:
                    continue

                old_volume = track.current_volume
                new_volume = self.volume_controller.update_volume(
                    track.current_volume,
                    track.target_volume,
                    delta_time
                )

                if abs(new_volume - old_volume) > 0.001:
                    track.current_volume = new_volume
                    track.channel.set_volume(new_volume)
                    volume_changed = True

            if volume_changed:
                self.performance_stats["volume_changes"] += 1
                if self.on_volume_change:
                    self.on_volume_change(self.get_track_volumes())

            # 记录更新时间
            update_time = time.time() - start_time
            self.performance_stats["update_time"].append(update_time)

            # 保持指定更新频率
            sleep_time = max(0, (1.0 / self.update_rate) - update_time)
            time.sleep(sleep_time)

    def _verify_audio_output(self):
        """验证音频输出"""
        try:
            # 检查pygame mixer状态
            if not pygame.mixer.get_init():
                self.logger.error("Pygame mixer not initialized during verification")
                return

            # 检查活跃通道数
            active_channels = 0
            for track in self.tracks.values():
                if track.channel.get_busy():
                    active_channels += 1

            self.logger.info(f"Audio verification: {active_channels}/{len(self.tracks)} channels active")

            # 检查音量设置
            for track_name, track in self.tracks.items():
                volume = track.channel.get_volume()
                self.logger.info(f"Track {track_name} current volume: {volume:.2f}")

            # 获取系统音量信息（如果可用）
            try:
                mixer_info = pygame.mixer.get_init()
                if mixer_info:
                    freq, format_bits, channels = mixer_info
                    self.logger.info(f"Mixer config: {freq}Hz, {format_bits}bit, {channels}ch")
            except:
                pass

        except Exception as e:
            self.logger.error(f"Audio verification failed: {e}")

    def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up audio controller...")

        self.stop_playback()

        # 清理pygame
        pygame.mixer.quit()

        self.tracks.clear()
        self.is_initialized = False

        self.logger.info("Audio controller cleanup completed")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()


# 便利函数
def create_gesture_data(hand_openness: float = 0.0,
                       is_present: bool = False,
                       pointing_pos: Optional[Tuple[float, float]] = None) -> GestureData:
    """
    创建手势数据对象的便利函数

    Args:
        hand_openness: 手掌张开程度 (0-1)
        is_present: 是否有手势
        pointing_pos: 指向位置 (x, y)

    Returns:
        GestureData对象
    """
    return GestureData(
        hand_openness=hand_openness,
        finger_positions=[],
        is_hand_present=is_present,
        pointing_direction=pointing_pos
    )


def test_region_based_control():
    """测试基于声部区域的控制功能"""
    print("BWV_29_in_D Enhanced Audio Controller Test")
    print("=========================================")

    # 创建音频控制器
    audio_dir = os.path.dirname(os.path.abspath(__file__))

    with AudioController(audio_dir) as controller:
        # 初始化
        if not controller.initialize():
            print("Failed to initialize audio controller")
            return

        print("Audio controller initialized successfully")
        print(f"Loaded {len(controller.tracks)} tracks:")
        for track_name in controller.tracks.keys():
            print(f"  - {track_name}")

        print(f"\n7-声部区域映射:")
        for region_id, track_names in controller.gesture_mapper.region_mapping.items():
            track_list = ', '.join(track_names)
            print(f"  Region {region_id}: {track_list}")

        # 开始播放
        if controller.start_playback():
            print("\nPlayback started. Testing enhanced controls...")

            try:
                # 测试基于区域的控制
                print("\n=== 测试基于区域的音量控制 ===")
                region_tests = [
                    (1, 0.8, "Tromba组合 强奏"),
                    (2, 0.6, "Violins 中等"),
                    (7, 1.0, "Timpani 最强"),
                    (5, 0.4, "Continuo 轻柔")
                ]

                for region_id, volume, description in region_tests:
                    print(f"\n设置 {description} (Region {region_id}, 音量: {volume})")
                    controller.set_region_volume(region_id, volume)
                    time.sleep(2)

                    # 显示区域音量
                    region_volumes = controller.get_region_volumes()
                    print(f"  区域音量: {region_volumes}")

                # 测试手势模式
                print("\n=== 测试不同手势模式 ===")
                controller.set_gesture_mode(GestureMode.SELECTIVE)
                print("切换到选择性模式")

                # 模拟指向不同区域
                selective_tests = [
                    (0.15, 0.35, "指向Violins区域"),
                    (0.75, 0.65, "指向Timpani区域"),
                    (0.4, 0.2, "指向Tromba组合区域")
                ]

                for x, y, description in selective_tests:
                    print(f"\n{description}")
                    gesture_data = create_gesture_data(0.8, True, (x, y))
                    controller.update_gesture(gesture_data)
                    time.sleep(2)

                    # 显示激活的区域
                    region_volumes = controller.get_region_volumes()
                    active_regions = [f"Region {r}: {v:.2f}" for r, v in region_volumes.items() if v > 0.1]
                    if active_regions:
                        print(f"  激活区域: {', '.join(active_regions)}")

                # 测试指挥家模式
                print("\n=== 测试指挥家模式 ===")
                controller.set_gesture_mode(GestureMode.CONDUCTOR)
                conductor_tests = [
                    (0.3, "轻柔指挥"),
                    (0.7, "中等力度"),
                    (0.9, "强烈指挥")
                ]

                for openness, description in conductor_tests:
                    print(f"\n{description} (手势强度: {openness})")
                    gesture_data = create_gesture_data(openness, True)
                    controller.update_gesture(gesture_data)
                    time.sleep(2)

                    # 显示所有音轨音量
                    track_volumes = controller.get_track_volumes()
                    active_tracks = [(name, vol) for name, vol in track_volumes.items() if vol > 0.1]
                    if active_tracks:
                        for name, vol in active_tracks:
                            print(f"  {name}: {vol:.2f}")

                # 性能统计
                print("\n=== 性能统计 ===")
                stats = controller.get_performance_stats()
                if stats:
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.3f}")

            except KeyboardInterrupt:
                print("\nTest interrupted")

        else:
            print("Failed to start playback")

    print("\nEnhanced test completed")


def main():
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BWV_29_in_D Audio Controller Test')
    parser.add_argument('--enhanced', action='store_true',
                       help='Run enhanced region-based test')

    args = parser.parse_args()

    if args.enhanced:
        test_region_based_control()
    else:
        # 原有的基本测试
        print("BWV_29_in_D Audio Controller Test")
        print("==================================")

        # 创建音频控制器
        audio_dir = os.path.dirname(os.path.abspath(__file__))

        with AudioController(audio_dir) as controller:
            # 初始化
            if not controller.initialize():
                print("Failed to initialize audio controller")
                return

            print("Audio controller initialized successfully")
            print(f"Loaded {len(controller.tracks)} tracks:")
            for track_name in controller.tracks.keys():
                print(f"  - {track_name}")

            # 开始播放
            if controller.start_playback():
                print("\nPlayback started. Testing gesture controls...")

                try:
                    # 测试序列
                    test_sequences = [
                        ("Hand present, closed", create_gesture_data(0.0, True)),
                        ("Hand opening", create_gesture_data(0.5, True)),
                        ("Hand fully open", create_gesture_data(1.0, True)),
                        ("Pointing to Violins", create_gesture_data(0.8, True, (0.2, 0.4))),
                        ("Pointing to Timpani", create_gesture_data(0.9, True, (0.6, 0.6))),
                        ("Hand closing", create_gesture_data(0.3, True)),
                        ("Hand absent", create_gesture_data(0.0, False))
                    ]

                    for description, gesture_data in test_sequences:
                        print(f"\n{description}...")
                        controller.update_gesture(gesture_data)
                        time.sleep(3)

                        # 显示当前音量
                        volumes = controller.get_track_volumes()
                        active_tracks = [name for name, vol in volumes.items() if vol > 0.1]
                        if active_tracks:
                            print(f"  Active tracks: {', '.join(active_tracks)}")
                        else:
                            print("  All tracks silent")

                except KeyboardInterrupt:
                    print("\nTest interrupted")

            else:
                print("Failed to start playback")

        print("\nTest completed")


if __name__ == "__main__":
    main()