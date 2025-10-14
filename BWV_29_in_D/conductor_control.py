#!/usr/bin/env python3
"""
BWV_29_in_D 指挥家控制主程序
实时摄像头手势识别控制7声部音频播放系统

核心功能：
- 摄像头输入管理 (OpenCV)
- 实时手势识别 (MediaPipe + HandGestureDetector)
- 多声部音频控制 (AudioController)
- 用户状态检测 (出现/离开)
- 实时可视化界面
- 键盘快捷键控制
- 性能监控和调试信息

使用方法：
python conductor_control.py [--camera-id 0] [--audio-dir ./] [--fullscreen]

键盘控制：
- ESC: 退出程序
- SPACE: 暂停/恢复播放
- R: 重置手势检测器
- D: 切换调试信息显示
- F: 切换全屏模式
- 1-9: 直接控制声部音量
- 0: 所有声部静音

Author: Claude Code
Date: 2025-10-05
"""

import cv2
import numpy as np
import time
import sys
import os
import argparse
import threading
import logging
import json
import pickle
import pygame
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
from collections import deque
import queue
from contextlib import contextmanager

# 添加父目录到Python路径以导入HandGestureDetector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_gesture_detector import HandGestureDetector
from audio_controller import AudioController, GestureData, create_gesture_data
from professional_ui_renderer import ProfessionalUIRenderer


class SystemState(Enum):
    """系统状态枚举"""
    INITIALIZING = "initializing"
    WAITING_USER = "waiting_user"      # 等待用户出现
    USER_DETECTED = "user_detected"    # 用户已检测到
    AUTO_STARTING = "auto_starting"    # 自动开始播放
    CONDUCTING = "conducting"          # 指挥进行中
    USER_LEFT = "user_left"           # 用户离开
    PAUSED = "paused"                 # 暂停状态
    USER_RETURNED = "user_returned"   # 用户返回
    RESUMING = "resuming"             # 恢复播放
    ERROR = "error"                   # 错误状态
    RECOVERING = "recovering"         # 错误恢复中
    SHUTDOWN = "shutdown"             # 关闭状态


class GestureMode(Enum):
    """手势控制模式"""
    CENTRAL_CONTROL = "central_control"  # 中央控制模式
    REGION_CONTROL = "region_control"    # 区域控制模式
    MIXED_CONTROL = "mixed_control"      # 混合控制模式
    IDLE = "idle"                        # 空闲模式


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    fps: float = 0.0
    frame_time: float = 0.0
    gesture_latency: float = 0.0
    audio_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class UserPresence:
    """用户存在检测数据"""
    is_present: bool = False
    confidence: float = 0.0
    last_seen_time: float = 0.0
    first_seen_time: float = 0.0
    detection_threshold: float = 0.5  # 提高检测阈值
    timeout_seconds: float = 5.0  # 增加超时时间
    stable_presence_time: float = 2.0  # 增加稳定存在时间
    confidence_history: deque = None
    presence_confirmed: bool = False
    # 新增去抖动相关属性
    last_state_change_time: float = 0.0
    state_change_cooldown: float = 2.0  # 状态变化冷却时间
    consecutive_detections: int = 0
    consecutive_absences: int = 0
    required_consecutive_count: int = 15  # 需要连续检测的帧数

    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=60)  # 增加历史记录保持60帧


@dataclass
class PlaybackState:
    """播放状态数据"""
    current_position: float = 0.0
    last_pause_position: float = 0.0
    play_start_time: float = 0.0
    total_duration: float = 0.0
    volume_settings: Dict[str, float] = None
    gesture_mode: GestureMode = GestureMode.IDLE
    auto_play_enabled: bool = True

    def __post_init__(self):
        if self.volume_settings is None:
            self.volume_settings = {}


@dataclass
class ErrorRecoveryState:
    """错误恢复状态"""
    camera_errors: int = 0
    audio_errors: int = 0
    last_error_time: float = 0.0
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_cooldown: float = 10.0  # 恢复冷却时间


class StateTransition:
    """状态转换管理器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.transition_callbacks: Dict[Tuple[SystemState, SystemState], List[Callable]] = {}
        self.transition_history: deque = deque(maxlen=50)

    def register_callback(self, from_state: SystemState, to_state: SystemState, callback: Callable):
        """注册状态转换回调"""
        key = (from_state, to_state)
        if key not in self.transition_callbacks:
            self.transition_callbacks[key] = []
        self.transition_callbacks[key].append(callback)

    def execute_transition(self, from_state: SystemState, to_state: SystemState, context: Dict[str, Any] = None):
        """执行状态转换"""
        try:
            transition_time = time.time()
            self.transition_history.append({
                'from': from_state,
                'to': to_state,
                'time': transition_time,
                'context': context or {}
            })

            self.logger.info(f"State transition: {from_state.value} -> {to_state.value}")

            key = (from_state, to_state)
            if key in self.transition_callbacks:
                for callback in self.transition_callbacks[key]:
                    try:
                        callback(from_state, to_state, context)
                    except Exception as e:
                        self.logger.error(f"State transition callback error: {e}")

        except Exception as e:
            self.logger.error(f"State transition execution error: {e}")

    def get_transition_history(self) -> List[Dict[str, Any]]:
        """获取状态转换历史"""
        return list(self.transition_history)


class StatePersistence:
    """状态持久化管理器"""

    def __init__(self, state_file: str = "conductor_state.json"):
        self.state_file = state_file
        self.backup_file = f"{state_file}.backup"

    def save_state(self, playback_state: PlaybackState, user_preferences: Dict[str, Any] = None) -> bool:
        """保存状态到文件"""
        try:
            state_data = {
                'playback_state': asdict(playback_state),
                'user_preferences': user_preferences or {},
                'timestamp': time.time(),
                'version': '1.0'
            }

            # 创建备份
            if os.path.exists(self.state_file):
                os.rename(self.state_file, self.backup_file)

            # 保存新状态
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            return True

        except Exception as e:
            logging.error(f"Failed to save state: {e}")
            # 恢复备份
            if os.path.exists(self.backup_file):
                os.rename(self.backup_file, self.state_file)
            return False

    def load_state(self) -> Tuple[Optional[PlaybackState], Optional[Dict[str, Any]]]:
        """从文件加载状态"""
        try:
            if not os.path.exists(self.state_file):
                return None, None

            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # 验证数据版本
            if state_data.get('version') != '1.0':
                logging.warning("State file version mismatch, ignoring")
                return None, None

            playback_data = state_data.get('playback_state', {})
            playback_state = PlaybackState(
                current_position=playback_data.get('current_position', 0.0),
                last_pause_position=playback_data.get('last_pause_position', 0.0),
                play_start_time=playback_data.get('play_start_time', 0.0),
                total_duration=playback_data.get('total_duration', 0.0),
                volume_settings=playback_data.get('volume_settings', {}),
                gesture_mode=GestureMode(playback_data.get('gesture_mode', GestureMode.IDLE.value)),
                auto_play_enabled=playback_data.get('auto_play_enabled', True)
            )

            user_preferences = state_data.get('user_preferences', {})

            return playback_state, user_preferences

        except Exception as e:
            logging.error(f"Failed to load state: {e}")
            return None, None

    def clear_state(self) -> bool:
        """清除保存的状态"""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            if os.path.exists(self.backup_file):
                os.remove(self.backup_file)
            return True
        except Exception as e:
            logging.error(f"Failed to clear state: {e}")
            return False


class CameraManager:
    """摄像头管理器"""

    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        初始化摄像头管理器

        Args:
            camera_id: 摄像头设备ID
            resolution: 分辨率 (width, height)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame_time = time.time()

    def initialize(self) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                return False

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 测试读取一帧
            ret, frame = self.cap.read()
            if not ret:
                return False

            self.is_initialized = True
            return True

        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取视频帧"""
        if not self.is_initialized or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            # 水平翻转以获得镜像效果
            frame = cv2.flip(frame, 1)

        return ret, frame

    def get_fps(self) -> float:
        """计算实际FPS"""
        current_time = time.time()
        if current_time - self.last_frame_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time
            return fps
        return 0.0

    def cleanup(self):
        """清理摄像头资源"""
        if self.cap:
            self.cap.release()
        self.is_initialized = False


class UIRenderer:
    """用户界面渲染器"""

    def __init__(self):
        """初始化UI渲染器"""
        self.colors = {
            'primary': (0, 255, 100),      # 主色调 - 绿色
            'secondary': (100, 100, 255),   # 次色调 - 蓝色
            'warning': (0, 165, 255),       # 警告色 - 橙色
            'error': (0, 0, 255),          # 错误色 - 红色
            'text': (255, 255, 255),       # 文本色 - 白色
            'bg_overlay': (0, 0, 0),       # 背景覆盖 - 黑色
            'inactive': (128, 128, 128),   # 非激活色 - 灰色
            'active': (0, 255, 255)        # 激活色 - 黄色
        }

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def draw_status_bar(self, frame: np.ndarray, system_state: SystemState,
                       performance: PerformanceMetrics, user_presence: UserPresence,
                       playback_state=None, error_recovery=None):
        """绘制状态栏"""
        height, width = frame.shape[:2]

        # 状态栏背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), self.colors['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 系统状态
        state_color = self._get_state_color(system_state)
        cv2.putText(frame, f"Status: {system_state.value.upper()}", (10, 25),
                   self.font, self.font_scale, state_color, self.thickness)

        # 性能指标
        cv2.putText(frame, f"FPS: {performance.fps:.1f}", (10, 50),
                   self.font, 0.5, self.colors['text'], 1)

        cv2.putText(frame, f"Latency: {performance.gesture_latency:.1f}ms", (100, 50),
                   self.font, 0.5, self.colors['text'], 1)

        # 用户存在状态
        presence_text = "USER PRESENT" if user_presence.is_present else "WAITING FOR USER"
        presence_color = self.colors['primary'] if user_presence.is_present else self.colors['warning']
        cv2.putText(frame, presence_text, (width - 200, 25),
                   self.font, self.font_scale, presence_color, self.thickness)

        # 置信度
        confidence_text = f"Conf: {user_presence.confidence:.2f}"
        cv2.putText(frame, confidence_text, (width - 200, 50),
                   self.font, 0.5, self.colors['text'], 1)

    def draw_gesture_info(self, frame: np.ndarray, gesture_data: dict,
                         active_regions: Dict[str, Any], playback_state=None):
        """绘制手势信息"""
        height, width = frame.shape[:2]
        y_start = height - 120

        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (width, height), self.colors['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 手势统计
        hands_count = gesture_data.get('hands_detected', 0)
        cv2.putText(frame, f"Hands: {hands_count}", (10, y_start + 25),
                   self.font, self.font_scale, self.colors['text'], self.thickness)

        # 激活区域
        active_count = len(active_regions)
        cv2.putText(frame, f"Active Regions: {active_count}", (10, y_start + 50),
                   self.font, self.font_scale, self.colors['text'], self.thickness)

        # 中央控制状态
        central_active = gesture_data.get('central_control_active', False)
        central_text = "CENTRAL CONTROL" if central_active else "REGION CONTROL"
        central_color = self.colors['active'] if central_active else self.colors['inactive']
        cv2.putText(frame, central_text, (10, y_start + 75),
                   self.font, self.font_scale, central_color, self.thickness)

        # 活跃声部列表
        if active_regions:
            regions_text = ", ".join(list(active_regions.keys())[:3])  # 显示前3个
            if len(active_regions) > 3:
                regions_text += "..."
            cv2.putText(frame, f"Regions: {regions_text}", (200, y_start + 50),
                       self.font, 0.5, self.colors['primary'], 1)

    def draw_voice_regions(self, frame: np.ndarray, voice_regions: Dict[str, Any],
                          active_regions: Dict[str, Any], central_region: Dict[str, Any]):
        """绘制声部区域"""
        height, width = frame.shape[:2]

        # 绘制7个声部区域
        for region_name, region_data in voice_regions.items():
            bounds = region_data['bounds']
            color = region_data['color']

            # 检查是否激活
            is_active = region_name in active_regions
            line_thickness = 3 if is_active else 1

            # 转换归一化坐标到像素坐标
            x1 = int(bounds['x1'] * width)
            y1 = int(bounds['y1'] * height)
            x2 = int(bounds['x2'] * width)
            y2 = int(bounds['y2'] * height)

            # 绘制矩形边界
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            # 如果激活，添加填充效果
            if is_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # 添加区域标签
            label = f"{region_data['id']}: {region_name.split('_')[0]}"
            cv2.putText(frame, label, (x1 + 5, y1 + 20),
                       self.font, 0.4, color, 1)

            # 如果激活，显示激活强度
            if is_active and region_name in active_regions:
                activation = active_regions[region_name].get('activation_strength', 0.0)
                strength_text = f"{activation:.2f}"
                cv2.putText(frame, strength_text, (x1 + 5, y2 - 10),
                           self.font, 0.4, self.colors['active'], 1)

        # 绘制中央控制区域
        central_bounds = central_region['bounds']
        central_color = central_region['color']
        central_active = len([h for h in active_regions.values()
                            if h.get('active_region') == 'central_control']) > 0

        x1 = int(central_bounds['x1'] * width)
        y1 = int(central_bounds['y1'] * height)
        x2 = int(central_bounds['x2'] * width)
        y2 = int(central_bounds['y2'] * height)

        line_thickness = 3 if central_active else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), central_color, line_thickness)

        if central_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), central_color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        cv2.putText(frame, "Central Control", (x1 + 5, y1 + 20),
                   self.font, 0.5, central_color, 1)

    def draw_audio_levels(self, frame: np.ndarray, track_volumes: Dict[str, float]):
        """绘制音频电平"""
        height, width = frame.shape[:2]

        # 音频电平显示区域
        level_width = 20
        level_height = 100
        start_x = width - 200
        start_y = 100

        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x - 10, start_y - 10),
                     (start_x + len(track_volumes) * 25 + 10, start_y + level_height + 30),
                     self.colors['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Audio Levels", (start_x, start_y - 15),
                   self.font, 0.5, self.colors['text'], 1)

        for i, (track_name, volume) in enumerate(track_volumes.items()):
            x = start_x + i * 25

            # 背景条
            cv2.rectangle(frame, (x, start_y), (x + level_width, start_y + level_height),
                         self.colors['inactive'], 1)

            # 音量条
            if volume > 0:
                fill_height = int(level_height * volume)
                color = self._get_volume_color(volume)
                cv2.rectangle(frame, (x + 1, start_y + level_height - fill_height),
                             (x + level_width - 1, start_y + level_height - 1),
                             color, -1)

            # 声部编号
            track_id = str(i + 1)
            cv2.putText(frame, track_id, (x + 5, start_y + level_height + 15),
                       self.font, 0.4, self.colors['text'], 1)

    def draw_keyboard_help(self, frame: np.ndarray):
        """绘制键盘快捷键帮助"""
        height, width = frame.shape[:2]

        help_text = [
            "Keyboard Controls:",
            "ESC - Exit",
            "SPACE - Pause/Resume",
            "R - Reset detector",
            "D - Toggle debug",
            "F - Fullscreen",
            "H - Toggle help",
            "S - Save state",
            "C - Clear state",
            "A - Toggle auto-play",
            "U - Toggle UI mode",
            "1-9 - Direct volume",
            "0 - Mute all"
        ]

        # 背景
        text_height = len(help_text) * 20 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 180, height - text_height - 10),
                     (width - 10, height - 10), self.colors['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        for i, text in enumerate(help_text):
            y = height - text_height + i * 20 + 15
            color = self.colors['primary'] if i == 0 else self.colors['text']
            font_scale = 0.5 if i == 0 else 0.4
            cv2.putText(frame, text, (width - 175, y), self.font, font_scale, color, 1)

    def _get_state_color(self, state: SystemState) -> Tuple[int, int, int]:
        """获取状态对应的颜色"""
        state_colors = {
            SystemState.INITIALIZING: self.colors['warning'],
            SystemState.WAITING_USER: self.colors['warning'],
            SystemState.USER_DETECTED: self.colors['primary'],
            SystemState.CONDUCTING: self.colors['active'],
            SystemState.PAUSED: self.colors['secondary'],
            SystemState.ERROR: self.colors['error'],
            SystemState.SHUTDOWN: self.colors['inactive']
        }
        return state_colors.get(state, self.colors['text'])

    def _get_volume_color(self, volume: float) -> Tuple[int, int, int]:
        """根据音量获取颜色"""
        if volume < 0.3:
            return self.colors['primary']
        elif volume < 0.7:
            return self.colors['warning']
        else:
            return self.colors['error']

    def _get_gesture_mode_color(self, mode) -> Tuple[int, int, int]:
        """获取手势模式对应的颜色"""
        if hasattr(mode, 'value'):
            mode_str = mode.value.upper()
        else:
            mode_str = str(mode).upper()

        mode_colors = {
            'CENTRAL_CONTROL': self.colors['active'],
            'REGION_CONTROL': self.colors['primary'],
            'MIXED_CONTROL': self.colors['secondary'],
            'IDLE': self.colors['inactive']
        }
        return mode_colors.get(mode_str, self.colors['text'])


class ConductorControl:
    """指挥家控制主程序"""

    def __init__(self, camera_id: int = 0, audio_dir: str = None, fullscreen: bool = False):
        """
        初始化指挥家控制系统

        Args:
            camera_id: 摄像头设备ID
            audio_dir: 音频文件目录
            fullscreen: 是否全屏显示
        """
        # 设置日志
        self._setup_logging()

        # 基本配置
        self.camera_id = camera_id
        self.audio_dir = audio_dir or os.path.dirname(os.path.abspath(__file__))
        self.fullscreen = fullscreen

        # 系统状态
        self.system_state = SystemState.INITIALIZING
        self.previous_state = None
        self.is_running = False
        self.show_debug = True
        self.show_help = True

        # 组件
        self.camera_manager = None
        self.gesture_detector = None
        self.audio_controller = None
        self.ui_renderer = UIRenderer()  # 保留原有UI渲染器作为备用
        self.professional_ui = ProfessionalUIRenderer()  # 新的专业UI渲染器
        self.use_professional_ui = True  # 是否使用专业UI

        # 性能监控
        self.performance = PerformanceMetrics()
        self.last_performance_update = time.time()

        # 用户状态管理
        self.user_presence = UserPresence()
        self.playback_state = PlaybackState()
        self.error_recovery = ErrorRecoveryState()

        # 状态管理
        self.state_transition = StateTransition(self.logger)
        self.state_persistence = StatePersistence(os.path.join(self.audio_dir, "conductor_state.json"))
        self._setup_state_transitions()

        # 手势数据缓存
        self.gesture_cache = {
            'last_update': 0.0,
            'data': {},
            'active_regions': {},
            'smoothing_buffer': deque(maxlen=5)  # 平滑缓冲区
        }

        # 音量平滑控制
        self.volume_smoother = {
            'target_volumes': {},
            'current_volumes': {},
            'smoothing_factor': 0.1,  # 平滑因子
            'update_rate': 30  # 更新频率 (FPS)
        }

        # 线程控制
        self.performance_thread = None
        self.state_monitor_thread = None
        self.volume_smoother_thread = None
        self.event_queue = queue.Queue()

        # OpenCV窗口
        self.window_name = "BWV_29_in_D Conductor Control"

        # 用户偏好设置
        self.user_preferences = {
            'auto_play_enabled': True,
            'presence_timeout': 3.0,
            'gesture_sensitivity': 1.0,
            'volume_smoothing': True,
            'debug_mode': True
        }

    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('conductor_control.log')
            ]
        )
        self.logger = logging.getLogger('ConductorControl')

    def initialize(self) -> bool:
        """初始化所有系统组件"""
        try:
            self.logger.info("Initializing Conductor Control System...")

            # 加载之前保存的状态
            self._load_persistent_state()

            # 初始化摄像头
            self.logger.info("Initializing camera...")
            self.camera_manager = CameraManager(self.camera_id)
            if not self.camera_manager.initialize():
                self.logger.error("Failed to initialize camera")
                self.error_recovery.camera_errors += 1
                if not self._attempt_camera_recovery():
                    return False

            # 初始化手势检测器
            self.logger.info("Initializing gesture detector...")
            self.gesture_detector = HandGestureDetector()

            # 初始化音频控制器
            self.logger.info("Initializing audio controller...")
            self.audio_controller = AudioController(self.audio_dir)
            if not self.audio_controller.initialize():
                self.logger.error("Failed to initialize audio controller")
                self.error_recovery.audio_errors += 1
                if not self._attempt_audio_recovery():
                    return False
            else:
                # 执行音频诊断
                if not self._perform_audio_diagnostics():
                    self.logger.warning("Audio diagnostics failed, but continuing with basic functionality")
                    # 不返回False，继续运行但记录警告

            # 设置回调
            self.audio_controller.on_state_change = self._on_audio_state_change
            self.audio_controller.on_volume_change = self._on_volume_change

            # 初始化OpenCV窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            if self.fullscreen:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # 启动监控线程
            self._start_background_threads()

            # 初始化音量平滑器
            self._initialize_volume_smoother()

            # 状态转换
            self._transition_to_state(SystemState.WAITING_USER)
            self.logger.info("Conductor Control System initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            self._transition_to_state(SystemState.ERROR)
            return False

    def run(self):
        """运行主控制循环"""
        if not self.initialize():
            self.logger.error("Failed to initialize system")
            return

        self.is_running = True
        self.logger.info("Starting main control loop...")

        try:
            while self.is_running:
                loop_start = time.time()

                # 读取摄像头帧
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    self.logger.error("Failed to read camera frame")
                    break

                # 处理手势识别
                gesture_start = time.time()
                processed_frame = self._process_gestures(frame)
                self.performance.gesture_latency = (time.time() - gesture_start) * 1000

                # 更新用户存在状态
                self._update_user_presence()

                # 处理音频控制
                self._process_audio_control()

                # 渲染用户界面
                display_frame = self._render_ui(processed_frame)

                # 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard_input(key):
                    break

                # 更新性能指标
                self.performance.frame_time = (time.time() - loop_start) * 1000
                self.performance.fps = self.camera_manager.get_fps()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()

    def _process_gestures(self, frame: np.ndarray) -> np.ndarray:
        """处理手势识别"""
        try:
            # 使用手势检测器处理帧
            processed_frame = self.gesture_detector.process_frame(frame, show_regions=True)

            # 获取手势数据
            gesture_data = self.gesture_detector.gesture_data

            # 更新缓存
            self.gesture_cache['last_update'] = time.time()
            self.gesture_cache['data'] = gesture_data.copy()
            self.gesture_cache['active_regions'] = gesture_data.get('active_regions', {}).copy()

            return processed_frame

        except Exception as e:
            self.logger.error(f"Gesture processing error: {e}")
            return frame

    def _update_user_presence(self):
        """更新用户存在状态（增强版，带去抖动）"""
        try:
            gesture_data = self.gesture_cache['data']
            hands_detected = gesture_data.get('hands_detected', 0)
            current_time = time.time()

            # 计算置信度
            if hands_detected > 0:
                # 基于手的数量和检测质量计算置信度
                hand_quality = sum(hand.get('confidence', 0.0) for hand in gesture_data.get('hands', []))
                raw_confidence = min(1.0, (hands_detected + hand_quality) / 3.0)
                # 增加连续检测计数
                self.user_presence.consecutive_detections += 1
                self.user_presence.consecutive_absences = 0
            else:
                raw_confidence = 0.0
                # 增加连续缺失计数
                self.user_presence.consecutive_absences += 1
                self.user_presence.consecutive_detections = 0

            # 更新置信度历史
            self.user_presence.confidence_history.append(raw_confidence)

            # 计算加权平滑置信度（更注重最近的检测）
            if len(self.user_presence.confidence_history) > 0:
                weights = np.array([i for i in range(1, len(self.user_presence.confidence_history) + 1)])
                values = np.array(list(self.user_presence.confidence_history))
                smoothed_confidence = np.average(values, weights=weights)
            else:
                smoothed_confidence = 0.0

            self.user_presence.confidence = smoothed_confidence

            # 检查状态变化冷却期
            time_since_last_change = current_time - self.user_presence.last_state_change_time
            if time_since_last_change < self.user_presence.state_change_cooldown:
                return  # 在冷却期内，跳过状态变化

            # 检测用户出现（使用连续检测计数）
            if (smoothed_confidence > self.user_presence.detection_threshold and
                self.user_presence.consecutive_detections >= self.user_presence.required_consecutive_count):

                if not self.user_presence.is_present:
                    # 用户首次出现
                    if self.user_presence.first_seen_time == 0.0:
                        self.user_presence.first_seen_time = current_time

                    # 检查是否达到稳定存在时间
                    stable_time = current_time - self.user_presence.first_seen_time
                    if stable_time >= self.user_presence.stable_presence_time:
                        self.user_presence.is_present = True
                        self.user_presence.presence_confirmed = True
                        self.user_presence.last_seen_time = current_time
                        self.user_presence.last_state_change_time = current_time
                        self._handle_user_appeared()
                        self.logger.info(f"User detected after {stable_time:.1f}s of stable presence")

                else:
                    # 用户持续存在
                    self.user_presence.last_seen_time = current_time

            # 检测用户离开（使用连续缺失计数）
            elif (smoothed_confidence <= self.user_presence.detection_threshold and
                  self.user_presence.consecutive_absences >= self.user_presence.required_consecutive_count):

                if self.user_presence.is_present:
                    time_since_last_seen = current_time - self.user_presence.last_seen_time

                    if time_since_last_seen > self.user_presence.timeout_seconds:
                        # 用户离开
                        self.user_presence.last_state_change_time = current_time
                        self._handle_user_left()
                        self.logger.info(f"User left after {time_since_last_seen:.1f}s timeout")

                else:
                    # 重置首次检测时间
                    self.user_presence.first_seen_time = 0.0
                    self.user_presence.presence_confirmed = False

        except Exception as e:
            self.logger.error(f"User presence update error: {e}")

    def _handle_user_appeared(self):
        """处理用户出现事件"""
        try:
            self.logger.info("User appeared - initiating auto-start sequence")

            if self.system_state == SystemState.WAITING_USER:
                self._transition_to_state(SystemState.USER_DETECTED)

                # 短暂延迟后自动开始播放
                self.event_queue.put({
                    'type': 'auto_start',
                    'delay': 0.5  # 0.5秒延迟
                })

            elif self.system_state == SystemState.USER_LEFT:
                # 用户返回
                self._transition_to_state(SystemState.USER_RETURNED)

                # 恢复播放
                self.event_queue.put({
                    'type': 'resume_playback',
                    'delay': 0.2
                })

        except Exception as e:
            self.logger.error(f"Handle user appeared error: {e}")

    def _handle_user_left(self):
        """处理用户离开事件"""
        try:
            # 防止重复处理
            if self.system_state == SystemState.USER_LEFT:
                return

            self.logger.info("User left - pausing playback")

            # 保存当前播放位置
            if self.audio_controller and self.audio_controller.is_playing():
                self.playback_state.last_pause_position = self.audio_controller.get_playback_position()

            # 重置用户状态
            self.user_presence.is_present = False
            self.user_presence.presence_confirmed = False
            self.user_presence.confidence = 0.0
            self.user_presence.consecutive_detections = 0
            self.user_presence.consecutive_absences = 0
            self.user_presence.first_seen_time = 0.0

            # 状态转换
            if self.system_state in [SystemState.CONDUCTING, SystemState.USER_DETECTED, SystemState.AUTO_STARTING]:
                self._transition_to_state(SystemState.USER_LEFT)

                # 暂停播放
                self.event_queue.put({
                    'type': 'pause_playback',
                    'delay': 0.0
                })

                # 保存状态
                self._save_persistent_state()

        except Exception as e:
            self.logger.error(f"Handle user left error: {e}")

    def _process_audio_control(self):
        """处理音频控制（增强版）"""
        try:
            if not self.user_presence.presence_confirmed:
                return

            gesture_data = self.gesture_cache['data']
            active_regions = self.gesture_cache['active_regions']

            # 确定手势控制模式
            current_mode = self._determine_gesture_mode(gesture_data, active_regions)

            if current_mode != self.playback_state.gesture_mode:
                self.logger.info(f"Gesture mode changed: {self.playback_state.gesture_mode.value} -> {current_mode.value}")
                self.playback_state.gesture_mode = current_mode

            # 根据模式处理控制
            if current_mode == GestureMode.CENTRAL_CONTROL:
                self._process_central_control(gesture_data)

            elif current_mode == GestureMode.REGION_CONTROL:
                self._process_region_control(active_regions)

            elif current_mode == GestureMode.MIXED_CONTROL:
                self._process_mixed_control(gesture_data, active_regions)

            else:  # IDLE
                self._process_idle_control()

            # 更新系统状态
            if current_mode != GestureMode.IDLE and self.system_state == SystemState.USER_DETECTED:
                self._transition_to_state(SystemState.CONDUCTING)
            elif current_mode == GestureMode.IDLE and self.system_state == SystemState.CONDUCTING:
                self._transition_to_state(SystemState.USER_DETECTED)

        except Exception as e:
            self.logger.error(f"Audio control processing error: {e}")

    def _determine_gesture_mode(self, gesture_data: Dict, active_regions: Dict) -> GestureMode:
        """确定当前手势控制模式"""
        try:
            central_active = gesture_data.get('central_control_active', False)
            region_count = len(active_regions)

            if central_active and region_count > 0:
                return GestureMode.MIXED_CONTROL
            elif central_active:
                return GestureMode.CENTRAL_CONTROL
            elif region_count > 0:
                return GestureMode.REGION_CONTROL
            else:
                return GestureMode.IDLE

        except Exception:
            return GestureMode.IDLE

    def _process_central_control(self, gesture_data: Dict):
        """处理中央控制模式"""
        try:
            overall_openness = 0.0
            active_hands = 0

            for hand in gesture_data.get('hands', []):
                if hand.get('active_region') == 'central_control':
                    hand_openness = hand.get('openness', 0.0)
                    hand_confidence = hand.get('confidence', 0.0)

                    # 考虑手部检测置信度
                    weighted_openness = hand_openness * hand_confidence
                    overall_openness = max(overall_openness, weighted_openness)
                    active_hands += 1

            # 双手协调增强效果
            if active_hands >= 2:
                overall_openness = min(1.0, overall_openness * 1.2)

            # 应用音量平滑
            self._set_all_volumes_smooth(overall_openness)

        except Exception as e:
            self.logger.error(f"Central control processing error: {e}")

    def _process_region_control(self, active_regions: Dict):
        """处理区域控制模式"""
        try:
            target_volumes = {}

            # 初始化所有音轨为静音
            for track_name in self.audio_controller.tracks.keys():
                target_volumes[track_name] = 0.0

            # 处理激活区域
            for region_name, hand_data in active_regions.items():
                track_name = self._map_region_to_track(region_name)
                if track_name and track_name in self.audio_controller.tracks:
                    openness = hand_data.get('openness', 0.0)
                    activation = hand_data.get('activation_strength', 0.0)
                    confidence = hand_data.get('confidence', 0.0)

                    # 综合计算音量
                    volume = openness * activation * confidence

                    # 考虑手势敏感性设置
                    volume *= self.user_preferences.get('gesture_sensitivity', 1.0)

                    target_volumes[track_name] = min(1.0, volume)

            # 应用平滑音量变化
            self._set_volumes_smooth(target_volumes)

        except Exception as e:
            self.logger.error(f"Region control processing error: {e}")

    def _process_mixed_control(self, gesture_data: Dict, active_regions: Dict):
        """处理混合控制模式"""
        try:
            # 获取中央控制的总体音量
            central_volume = 0.0
            for hand in gesture_data.get('hands', []):
                if hand.get('active_region') == 'central_control':
                    central_volume = max(central_volume, hand.get('openness', 0.0))

            # 获取区域特定控制
            region_volumes = {}
            for region_name, hand_data in active_regions.items():
                track_name = self._map_region_to_track(region_name)
                if track_name:
                    openness = hand_data.get('openness', 0.0)
                    activation = hand_data.get('activation_strength', 0.0)
                    region_volumes[track_name] = openness * activation

            # 混合计算最终音量
            final_volumes = {}
            for track_name in self.audio_controller.tracks.keys():
                if track_name in region_volumes:
                    # 区域控制 + 中央控制的加权平均
                    region_vol = region_volumes[track_name]
                    mixed_volume = (region_vol * 0.7 + central_volume * 0.3)
                else:
                    # 只有中央控制
                    mixed_volume = central_volume * 0.5  # 降低未指定区域的音量

                final_volumes[track_name] = min(1.0, mixed_volume)

            self._set_volumes_smooth(final_volumes)

        except Exception as e:
            self.logger.error(f"Mixed control processing error: {e}")

    def _process_idle_control(self):
        """处理空闲控制模式"""
        try:
            # 在空闲模式下，保持当前音量设置
            # 可以选择逐渐降低音量或保持不变
            if self.user_preferences.get('auto_fade_in_idle', False):
                current_volumes = self.audio_controller.get_track_volumes()
                faded_volumes = {name: vol * 0.95 for name, vol in current_volumes.items()}
                self._set_volumes_smooth(faded_volumes)

        except Exception as e:
            self.logger.error(f"Idle control processing error: {e}")

    def _map_region_to_track(self, region_name: str) -> Optional[str]:
        """将手势区域名映射到音轨名"""
        mapping = {
            'Tromba_I+II+III_in_D': 'Tromba_I_in_D',  # 选择第一个铜管作为代表
            'Violins_in_D': 'Violins_in_D',
            'Viola_in_D': 'Viola_in_D',
            'Oboe_I_in_D': 'Oboe_I_in_D',
            'Continuo_in_D': 'Continuo_in_D',
            'Organo_obligato_in_D': 'Organo_obligato_in_D',
            'Timpani_in_D': 'Timpani_in_D'
        }
        return mapping.get(region_name)

    def _render_ui(self, frame: np.ndarray) -> np.ndarray:
        """渲染用户界面"""
        try:
            if self.use_professional_ui:
                # 使用专业UI渲染器
                return self._render_professional_ui(frame)
            else:
                # 使用原有UI渲染器
                return self._render_legacy_ui(frame)

        except Exception as e:
            self.logger.error(f"UI rendering error: {e}")
            # 如果专业UI失败，回退到基础UI
            if self.use_professional_ui:
                self.logger.warning("Professional UI failed, falling back to legacy UI")
                self.use_professional_ui = False
                return self._render_legacy_ui(frame)
            return frame

    def _render_professional_ui(self, frame: np.ndarray) -> np.ndarray:
        """渲染专业用户界面"""
        try:
            region_info = self.gesture_detector.get_region_info()

            # 使用专业UI渲染器绘制完整界面
            display_frame = self.professional_ui.draw_professional_interface(
                frame=frame,
                system_state=self.system_state,
                performance_metrics=self.performance,
                user_presence=self.user_presence,
                gesture_data=self.gesture_cache['data'],
                active_regions=self.gesture_cache['active_regions'],
                audio_controller=self.audio_controller,
                region_info=region_info
            )

            return display_frame

        except Exception as e:
            self.logger.error(f"Professional UI rendering error: {e}")
            raise

    def _render_legacy_ui(self, frame: np.ndarray) -> np.ndarray:
        """渲染传统用户界面（备用）"""
        try:
            display_frame = frame.copy()

            # 绘制状态栏
            self.ui_renderer.draw_status_bar(
                display_frame, self.system_state,
                self.performance, self.user_presence,
                self.playback_state, self.error_recovery
            )

            # 绘制声部区域
            region_info = self.gesture_detector.get_region_info()
            self.ui_renderer.draw_voice_regions(
                display_frame,
                region_info['voice_regions'],
                self.gesture_cache['active_regions'],
                region_info['central_control_region']
            )

            if self.show_debug:
                # 绘制手势信息
                self.ui_renderer.draw_gesture_info(
                    display_frame,
                    self.gesture_cache['data'],
                    self.gesture_cache['active_regions'],
                    self.playback_state
                )

                # 绘制音频电平
                if self.audio_controller and self.audio_controller.is_initialized:
                    track_volumes = self.audio_controller.get_track_volumes()
                    self.ui_renderer.draw_audio_levels(display_frame, track_volumes)

            if self.show_help:
                # 绘制快捷键帮助
                self.ui_renderer.draw_keyboard_help(display_frame)

            return display_frame

        except Exception as e:
            self.logger.error(f"Legacy UI rendering error: {e}")
            return frame

    def _handle_keyboard_input(self, key: int) -> bool:
        """处理键盘输入"""
        try:
            if key == 27:  # ESC
                self.logger.info("Exit requested by user")
                return False

            elif key == ord(' '):  # SPACE
                if self.audio_controller.get_playback_state().value == 'playing':
                    self.audio_controller.pause_playback()
                    self.system_state = SystemState.PAUSED
                    self.logger.info("Playback paused")
                elif self.audio_controller.get_playback_state().value == 'paused':
                    self.audio_controller.resume_playback()
                    self.system_state = SystemState.USER_DETECTED
                    self.logger.info("Playback resumed")

            elif key == ord('r') or key == ord('R'):  # Reset
                self.logger.info("Resetting gesture detector")
                self.gesture_detector = HandGestureDetector()

            elif key == ord('d') or key == ord('D'):  # Debug toggle
                self.show_debug = not self.show_debug
                self.logger.info(f"Debug display: {'ON' if self.show_debug else 'OFF'}")

            elif key == ord('f') or key == ord('F'):  # Fullscreen toggle
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            elif key == ord('h') or key == ord('H'):  # Help toggle
                self.show_help = not self.show_help

            elif key == ord('s') or key == ord('S'):  # Save state
                self._save_persistent_state()
                self.logger.info("State saved manually")

            elif key == ord('c') or key == ord('C'):  # Clear saved state
                self.state_persistence.clear_state()
                self.logger.info("Saved state cleared")

            elif key == ord('a') or key == ord('A'):  # Toggle auto-play
                self.playback_state.auto_play_enabled = not self.playback_state.auto_play_enabled
                self.user_preferences['auto_play_enabled'] = self.playback_state.auto_play_enabled
                status = "enabled" if self.playback_state.auto_play_enabled else "disabled"
                self.logger.info(f"Auto-play {status}")

            elif key == ord('u') or key == ord('U'):  # Toggle UI mode
                self.use_professional_ui = not self.use_professional_ui
                ui_mode = "Professional" if self.use_professional_ui else "Legacy"
                self.logger.info(f"UI mode switched to: {ui_mode}")

            elif key >= ord('0') and key <= ord('9'):  # Direct volume control
                volume_level = (key - ord('0')) / 9.0  # 0-9 映射到 0.0-1.0
                if key == ord('0'):
                    self._set_all_volumes_smooth(0.0)
                    self.logger.info("All tracks muted")
                else:
                    self._set_all_volumes_smooth(volume_level)
                    self.logger.info(f"All tracks volume set to {volume_level:.1f}")

            return True

        except Exception as e:
            self.logger.error(f"Keyboard input handling error: {e}")
            return True

    def _setup_state_transitions(self):
        """设置状态转换回调"""
        # 用户出现相关转换
        self.state_transition.register_callback(
            SystemState.WAITING_USER, SystemState.USER_DETECTED,
            self._on_user_detected
        )

        self.state_transition.register_callback(
            SystemState.USER_DETECTED, SystemState.AUTO_STARTING,
            self._on_auto_starting
        )

        self.state_transition.register_callback(
            SystemState.AUTO_STARTING, SystemState.CONDUCTING,
            self._on_conducting_started
        )

        # 用户离开相关转换
        self.state_transition.register_callback(
            SystemState.CONDUCTING, SystemState.USER_LEFT,
            self._on_user_left
        )

        self.state_transition.register_callback(
            SystemState.USER_LEFT, SystemState.PAUSED,
            self._on_playback_paused
        )

        # 用户返回相关转换
        self.state_transition.register_callback(
            SystemState.PAUSED, SystemState.USER_RETURNED,
            self._on_user_returned
        )

        self.state_transition.register_callback(
            SystemState.USER_RETURNED, SystemState.RESUMING,
            self._on_resuming_playback
        )

        self.state_transition.register_callback(
            SystemState.RESUMING, SystemState.CONDUCTING,
            self._on_conducting_resumed
        )

        # 错误处理转换
        self.state_transition.register_callback(
            SystemState.ERROR, SystemState.RECOVERING,
            self._on_error_recovery
        )

    def _transition_to_state(self, new_state: SystemState, context: Dict[str, Any] = None):
        """执行状态转换"""
        if self.system_state != new_state:
            old_state = self.system_state
            self.previous_state = old_state
            self.system_state = new_state
            self.state_transition.execute_transition(old_state, new_state, context)

    def _load_persistent_state(self):
        """加载持久化状态"""
        try:
            playback_state, user_prefs = self.state_persistence.load_state()

            if playback_state:
                self.playback_state = playback_state
                self.logger.info(f"Loaded playback state: position={playback_state.current_position:.1f}s")

            if user_prefs:
                self.user_preferences.update(user_prefs)
                self.logger.info("Loaded user preferences")

                # 应用用户偏好
                self.user_presence.timeout_seconds = user_prefs.get('presence_timeout', 3.0)
                self.show_debug = user_prefs.get('debug_mode', True)

        except Exception as e:
            self.logger.warning(f"Failed to load persistent state: {e}")

    def _save_persistent_state(self):
        """保存持久化状态"""
        try:
            # 更新当前播放位置
            if self.audio_controller and self.audio_controller.is_initialized:
                self.playback_state.current_position = self.audio_controller.get_playback_position()
                self.playback_state.volume_settings = self.audio_controller.get_track_volumes()

            # 保存状态
            success = self.state_persistence.save_state(self.playback_state, self.user_preferences)
            if success:
                self.logger.debug("State saved successfully")
            else:
                self.logger.warning("Failed to save state")

        except Exception as e:
            self.logger.error(f"Error saving persistent state: {e}")

    def _start_background_threads(self):
        """启动后台监控线程"""
        # 性能监控线程
        self.performance_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.performance_thread.start()

        # 状态监控线程
        self.state_monitor_thread = threading.Thread(target=self._monitor_state, daemon=True)
        self.state_monitor_thread.start()

        # 音量平滑线程
        self.volume_smoother_thread = threading.Thread(target=self._volume_smoother_loop, daemon=True)
        self.volume_smoother_thread.start()

    def _monitor_performance(self):
        """性能监控线程"""
        try:
            import psutil
            process = psutil.Process()

            while self.is_running:
                try:
                    self.performance.memory_usage = process.memory_percent()
                    self.performance.cpu_usage = process.cpu_percent()
                    time.sleep(1.0)
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                    break
        except ImportError:
            self.logger.warning("psutil not available, performance monitoring disabled")

    def _monitor_state(self):
        """状态监控线程"""
        last_save_time = time.time()
        save_interval = 30.0  # 每30秒保存一次状态

        while self.is_running:
            try:
                current_time = time.time()

                # 处理事件队列
                self._process_event_queue()

                # 定期保存状态
                if current_time - last_save_time >= save_interval:
                    if self.system_state in [SystemState.CONDUCTING, SystemState.PAUSED]:
                        self._save_persistent_state()
                    last_save_time = current_time

                # 检查错误恢复
                self._check_error_recovery()

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"State monitoring error: {e}")
                time.sleep(1.0)

    def _process_event_queue(self):
        """处理事件队列"""
        try:
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    self._handle_event(event)
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Event queue processing error: {e}")

    def _handle_event(self, event: Dict[str, Any]):
        """处理单个事件"""
        try:
            event_type = event.get('type')
            delay = event.get('delay', 0.0)

            if delay > 0:
                time.sleep(delay)

            if event_type == 'auto_start':
                self._execute_auto_start()
            elif event_type == 'pause_playback':
                self._execute_pause_playback()
            elif event_type == 'resume_playback':
                self._execute_resume_playback()
            elif event_type == 'save_state':
                self._save_persistent_state()
            else:
                self.logger.warning(f"Unknown event type: {event_type}")

        except Exception as e:
            self.logger.error(f"Event handling error: {e}")

    def _execute_auto_start(self):
        """执行自动开始播放"""
        try:
            if self.playback_state.auto_play_enabled and self.audio_controller:
                self._transition_to_state(SystemState.AUTO_STARTING)

                # 恢复之前的播放位置（如果有的话）
                if self.playback_state.last_pause_position > 0:
                    self.audio_controller.set_playback_position(self.playback_state.last_pause_position)
                    self.logger.info(f"Resuming from position: {self.playback_state.last_pause_position:.1f}s")

                # 开始播放
                self.audio_controller.start_playback()
                self.playback_state.play_start_time = time.time()

                # 转换到指挥状态
                self._transition_to_state(SystemState.CONDUCTING)

        except Exception as e:
            self.logger.error(f"Auto start execution error: {e}")
            self._transition_to_state(SystemState.ERROR)

    def _execute_pause_playback(self):
        """执行暂停播放"""
        try:
            if self.audio_controller and self.audio_controller.is_playing():
                # 保存当前位置
                self.playback_state.current_position = self.audio_controller.get_playback_position()
                self.playback_state.last_pause_position = self.playback_state.current_position

                # 暂停播放
                self.audio_controller.pause_playback()

                # 转换状态
                self._transition_to_state(SystemState.PAUSED)

                self.logger.info(f"Playback paused at position: {self.playback_state.current_position:.1f}s")

        except Exception as e:
            self.logger.error(f"Pause playback execution error: {e}")

    def _execute_resume_playback(self):
        """执行恢复播放"""
        try:
            if self.audio_controller:
                self._transition_to_state(SystemState.RESUMING)

                # 从暂停位置恢复
                if self.playback_state.last_pause_position > 0:
                    self.audio_controller.set_playback_position(self.playback_state.last_pause_position)

                # 恢复播放
                self.audio_controller.resume_playback()
                self.playback_state.play_start_time = time.time()

                # 转换到指挥状态
                self._transition_to_state(SystemState.CONDUCTING)

                self.logger.info(f"Playback resumed from position: {self.playback_state.last_pause_position:.1f}s")

        except Exception as e:
            self.logger.error(f"Resume playback execution error: {e}")
            self._transition_to_state(SystemState.ERROR)

    def _check_error_recovery(self):
        """检查错误恢复"""
        try:
            current_time = time.time()

            # 检查摄像头状态
            if self.camera_manager and not self.camera_manager.is_initialized:
                if current_time - self.error_recovery.last_error_time > self.error_recovery.recovery_cooldown:
                    self._attempt_camera_recovery()

            # 检查音频状态
            if self.audio_controller and not self.audio_controller.is_initialized:
                if current_time - self.error_recovery.last_error_time > self.error_recovery.recovery_cooldown:
                    self._attempt_audio_recovery()

        except Exception as e:
            self.logger.error(f"Error recovery check failed: {e}")

    def _attempt_camera_recovery(self) -> bool:
        """尝试摄像头恢复"""
        try:
            if self.error_recovery.recovery_attempts >= self.error_recovery.max_recovery_attempts:
                self.logger.error("Max camera recovery attempts reached")
                return False

            self.logger.info(f"Attempting camera recovery (attempt {self.error_recovery.recovery_attempts + 1})")
            self.error_recovery.recovery_attempts += 1
            self.error_recovery.last_error_time = time.time()

            # 清理现有摄像头
            if self.camera_manager:
                self.camera_manager.cleanup()

            # 重新初始化
            self.camera_manager = CameraManager(self.camera_id)
            if self.camera_manager.initialize():
                self.logger.info("Camera recovery successful")
                self.error_recovery.camera_errors = 0
                self.error_recovery.recovery_attempts = 0
                return True
            else:
                self.logger.error("Camera recovery failed")
                return False

        except Exception as e:
            self.logger.error(f"Camera recovery error: {e}")
            return False

    def _attempt_audio_recovery(self) -> bool:
        """尝试音频恢复"""
        try:
            if self.error_recovery.recovery_attempts >= self.error_recovery.max_recovery_attempts:
                self.logger.error("Max audio recovery attempts reached")
                return False

            self.logger.info(f"Attempting audio recovery (attempt {self.error_recovery.recovery_attempts + 1})")
            self.error_recovery.recovery_attempts += 1
            self.error_recovery.last_error_time = time.time()

            # 清理现有音频控制器
            if self.audio_controller:
                self.audio_controller.cleanup()

            # 重新初始化
            self.audio_controller = AudioController(self.audio_dir)
            if self.audio_controller.initialize():
                # 执行音频诊断测试
                if self._perform_audio_diagnostics():
                    self.logger.info("Audio recovery successful")
                    self.error_recovery.audio_errors = 0
                    self.error_recovery.recovery_attempts = 0

                    # 重新设置回调
                    self.audio_controller.on_state_change = self._on_audio_state_change
                    self.audio_controller.on_volume_change = self._on_volume_change

                    return True
                else:
                    self.logger.error("Audio recovery failed - diagnostics failed")
                    return False
            else:
                self.logger.error("Audio recovery failed")
                return False

        except Exception as e:
            self.logger.error(f"Audio recovery error: {e}")
            return False

    def _on_audio_state_change(self, new_state):
        """音频状态变化回调"""
        self.logger.info(f"Audio state changed to: {new_state}")

    def _on_volume_change(self, track_volumes: Dict[str, float]):
        """音量变化回调"""
        if self.show_debug:
            active_tracks = [name for name, vol in track_volumes.items() if vol > 0.1]
            if active_tracks:
                self.logger.debug(f"Active tracks: {', '.join(active_tracks)}")

    def _initialize_volume_smoother(self):
        """初始化音量平滑器"""
        try:
            if self.audio_controller and self.audio_controller.is_initialized:
                current_volumes = self.audio_controller.get_track_volumes()
                self.volume_smoother['current_volumes'] = current_volumes.copy()
                self.volume_smoother['target_volumes'] = current_volumes.copy()
                self.logger.info("Volume smoother initialized")
        except Exception as e:
            self.logger.error(f"Volume smoother initialization error: {e}")

    def _volume_smoother_loop(self):
        """音量平滑处理循环"""
        update_interval = 1.0 / self.volume_smoother['update_rate']

        while self.is_running:
            try:
                if not self.user_preferences.get('volume_smoothing', True):
                    time.sleep(update_interval)
                    continue

                current_vols = self.volume_smoother['current_volumes']
                target_vols = self.volume_smoother['target_volumes']
                smoothing_factor = self.volume_smoother['smoothing_factor']

                # 计算平滑后的音量
                smoothed_volumes = {}
                needs_update = False

                for track_name in current_vols.keys():
                    current = current_vols.get(track_name, 0.0)
                    target = target_vols.get(track_name, 0.0)

                    # 平滑插值
                    diff = target - current
                    if abs(diff) > 0.001:  # 阈值避免无限小的变化
                        new_volume = current + diff * smoothing_factor
                        smoothed_volumes[track_name] = new_volume
                        needs_update = True
                    else:
                        smoothed_volumes[track_name] = target

                # 应用平滑后的音量
                if needs_update and self.audio_controller:
                    for track_name, volume in smoothed_volumes.items():
                        self.audio_controller.set_track_volume(track_name, volume)

                    self.volume_smoother['current_volumes'] = smoothed_volumes.copy()

                time.sleep(update_interval)

            except Exception as e:
                self.logger.error(f"Volume smoother loop error: {e}")
                time.sleep(update_interval)

    def _set_all_volumes_smooth(self, volume: float):
        """设置所有音轨音量（平滑）"""
        try:
            if self.audio_controller:
                for track_name in self.audio_controller.tracks.keys():
                    self.volume_smoother['target_volumes'][track_name] = volume
        except Exception as e:
            self.logger.error(f"Set all volumes smooth error: {e}")

    def _set_volumes_smooth(self, target_volumes: Dict[str, float]):
        """设置特定音轨音量（平滑）"""
        try:
            for track_name, volume in target_volumes.items():
                if track_name in self.volume_smoother['target_volumes']:
                    self.volume_smoother['target_volumes'][track_name] = volume
        except Exception as e:
            self.logger.error(f"Set volumes smooth error: {e}")

    # 状态转换回调方法
    def _on_user_detected(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """用户检测到回调"""
        self.logger.info("User presence confirmed - preparing auto-start")

    def _on_auto_starting(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """自动开始回调"""
        self.logger.info("Auto-starting playback")

    def _on_conducting_started(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """指挥开始回调"""
        self.logger.info("Conducting session started")

    def _on_user_left(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """用户离开回调"""
        self.logger.info("User left - initiating pause sequence")

    def _on_playback_paused(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """播放暂停回调"""
        self.logger.info("Playback paused due to user absence")

    def _on_user_returned(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """用户返回回调"""
        self.logger.info("User returned - preparing to resume")

    def _on_resuming_playback(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """恢复播放回调"""
        self.logger.info("Resuming playback from saved position")

    def _on_conducting_resumed(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """指挥恢复回调"""
        self.logger.info("Conducting session resumed")

    def _on_error_recovery(self, from_state: SystemState, to_state: SystemState, context: Dict):
        """错误恢复回调"""
        self.logger.info("Attempting system recovery")

    @contextmanager
    def _error_handling_context(self, operation_name: str):
        """错误处理上下文管理器"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"{operation_name} failed: {e}")
            self.logger.error(traceback.format_exc())

            # 记录错误
            self.error_recovery.last_error_time = time.time()

            # 根据错误类型决定恢复策略
            if "camera" in operation_name.lower():
                self.error_recovery.camera_errors += 1
            elif "audio" in operation_name.lower():
                self.error_recovery.audio_errors += 1

            # 如果错误过多，转换到错误状态
            total_errors = self.error_recovery.camera_errors + self.error_recovery.audio_errors
            if total_errors >= 5:
                self._transition_to_state(SystemState.ERROR)

    def _perform_audio_diagnostics(self) -> bool:
        """执行音频系统诊断"""
        try:
            self.logger.info("Performing audio system diagnostics...")

            # 检查pygame mixer状态
            if not pygame.mixer.get_init():
                self.logger.error("Pygame mixer not initialized")
                return False

            # 获取音频设备信息
            mixer_info = pygame.mixer.get_init()
            self.logger.info(f"Mixer settings: frequency={mixer_info[0]}, format={mixer_info[1]}, channels={mixer_info[2]}")

            # 检查系统音频设备
            self._check_system_audio_devices()

            # 检查音频文件
            audio_files_found = self._check_audio_files()
            if not audio_files_found:
                self.logger.error("Critical audio files missing")
                return False

            # 执行音频播放测试
            if self.audio_controller and self.audio_controller.is_initialized:
                test_result = self.audio_controller.test_audio_output()
                if test_result:
                    self.logger.info("Audio output test successful")

                    # 额外的音量测试
                    self._test_volume_levels()
                    return True
                else:
                    self.logger.error("Audio output test failed")
                    return False
            else:
                self.logger.error("Audio controller not properly initialized")
                return False

        except Exception as e:
            self.logger.error(f"Audio diagnostics failed: {e}")
            return False

    def _check_system_audio_devices(self):
        """检查系统音频设备"""
        try:
            import platform
            system = platform.system().lower()

            if system == 'darwin':  # macOS
                self._check_macos_audio()
            elif system == 'windows':
                self._check_windows_audio()
            elif system == 'linux':
                self._check_linux_audio()

        except Exception as e:
            self.logger.warning(f"System audio device check failed: {e}")

    def _check_macos_audio(self):
        """检查macOS音频设备"""
        try:
            import subprocess

            # 检查音频输出设备
            result = subprocess.run(['system_profiler', 'SPAudioDataType', '-json'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("macOS audio devices detected")
            else:
                self.logger.warning("Could not enumerate macOS audio devices")

            # 检查音频服务
            result = subprocess.run(['launchctl', 'list', 'com.apple.audio.coreaudiod'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                self.logger.info("Core Audio daemon is running")
            else:
                self.logger.warning("Core Audio daemon status unknown")

        except Exception as e:
            self.logger.warning(f"macOS audio check failed: {e}")

    def _check_windows_audio(self):
        """检查Windows音频设备"""
        try:
            import subprocess

            # 检查Windows音频服务
            result = subprocess.run(['sc', 'query', 'AudioEndpointBuilder'],
                                  capture_output=True, text=True, timeout=5)
            if 'RUNNING' in result.stdout:
                self.logger.info("Windows Audio service is running")
            else:
                self.logger.warning("Windows Audio service status unclear")

        except Exception as e:
            self.logger.warning(f"Windows audio check failed: {e}")

    def _check_linux_audio(self):
        """检查Linux音频设备"""
        try:
            import subprocess

            # 检查ALSA设备
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'card' in result.stdout:
                self.logger.info("ALSA audio devices found")
            else:
                self.logger.warning("No ALSA audio devices found")

            # 检查PulseAudio
            result = subprocess.run(['pulseaudio', '--check'], capture_output=True, timeout=3)
            if result.returncode == 0:
                self.logger.info("PulseAudio is running")
            else:
                self.logger.info("PulseAudio not running or not available")

        except Exception as e:
            self.logger.warning(f"Linux audio check failed: {e}")

    def _check_audio_files(self) -> bool:
        """检查音频文件"""
        try:
            required_files = [
                "Violins_in_D.mp3",
                "Continuo_in_D.mp3",
                "Tromba_I_in_D.mp3"
            ]

            found_files = []
            missing_files = []

            for filename in required_files:
                file_path = os.path.join(self.audio_dir, filename)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 1024:  # 至少1KB
                        found_files.append(filename)
                        self.logger.info(f"Audio file found: {filename} ({file_size/1024/1024:.1f}MB)")
                    else:
                        missing_files.append(f"{filename} (empty)")
                else:
                    missing_files.append(filename)

            if missing_files:
                self.logger.error(f"Missing audio files: {', '.join(missing_files)}")
                self.logger.info(f"Searching in directory: {self.audio_dir}")

                # 列出目录中的所有MP3文件
                try:
                    mp3_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.mp3')]
                    if mp3_files:
                        self.logger.info(f"Available MP3 files: {', '.join(mp3_files)}")
                    else:
                        self.logger.error("No MP3 files found in audio directory")
                except Exception as e:
                    self.logger.error(f"Could not list audio directory: {e}")

            return len(found_files) >= 2  # 至少需要2个音频文件

        except Exception as e:
            self.logger.error(f"Audio file check failed: {e}")
            return False

    def _test_volume_levels(self):
        """测试音量级别"""
        try:
            if not self.audio_controller:
                return

            self.logger.info("Testing audio volume levels...")

            # 测试不同音量级别
            test_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
            test_track = "Violins_in_D"

            if test_track in self.audio_controller.tracks:
                for level in test_levels:
                    self.audio_controller.set_track_volume(test_track, level)
                    time.sleep(0.3)  # 短暂等待
                    actual_volume = self.audio_controller.get_track_volumes().get(test_track, 0)
                    self.logger.info(f"Volume test - Target: {level:.1f}, Actual: {actual_volume:.1f}")

                # 重置为中等音量
                self.audio_controller.set_track_volume(test_track, 0.4)
                self.logger.info("Volume level testing completed")
            else:
                self.logger.warning(f"Test track {test_track} not available for volume testing")

        except Exception as e:
            self.logger.error(f"Volume level testing failed: {e}")

    def cleanup(self):
        """清理系统资源"""
        self.logger.info("Cleaning up system resources...")

        self.is_running = False
        self._transition_to_state(SystemState.SHUTDOWN)

        # 保存最终状态
        self._save_persistent_state()

        # 清理摄像头
        if self.camera_manager:
            self.camera_manager.cleanup()

        # 清理音频控制器
        if self.audio_controller:
            self.audio_controller.cleanup()

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

        # 等待线程结束
        threads_to_join = [
            self.performance_thread,
            self.state_monitor_thread,
            self.volume_smoother_thread
        ]

        for thread in threads_to_join:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate gracefully")

        self.logger.info("System cleanup completed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BWV_29_in_D Conductor Control System')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--audio-dir', type=str, help='Audio files directory (default: current directory)')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    print("BWV_29_in_D Conductor Control System")
    print("=====================================")
    print(f"Camera ID: {args.camera_id}")
    print(f"Audio directory: {args.audio_dir or 'current directory'}")
    print(f"Fullscreen: {args.fullscreen}")
    print("\nStarting system...")

    try:
        # 创建并运行控制系统
        conductor = ConductorControl(
            camera_id=args.camera_id,
            audio_dir=args.audio_dir,
            fullscreen=args.fullscreen
        )

        conductor.run()

    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"\nSystem error: {e}")
        logging.error(traceback.format_exc())
    finally:
        print("System shutdown complete")


if __name__ == "__main__":
    main()