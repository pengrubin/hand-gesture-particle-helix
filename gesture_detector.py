"""
统一手势识别系统 (Unified Gesture Detection System)
合并版本 - 包含以下功能：
- 基础手势检测（来自 gesture_detector.py）
- 增强指挥手势识别（来自 hand_gesture_detector.py）
- 跨平台支持（来自 cross_platform_gesture_detector.py）
- 性能优化（来自 optimized_gesture_detector.py）

Author: Refactored from multiple sources
Date: 2025-03-26
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time
import platform
import sys
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import queue


# ============================================================================
# 辅助类 (Helper Classes)
# ============================================================================

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.last_report_time = time.time()
        self.frame_start = 0

    def start_frame(self):
        """开始帧计时"""
        self.frame_start = time.time()

    def end_frame(self):
        """结束帧计时"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)

    def record_process_time(self, process_time: float):
        """记录处理时间"""
        self.process_times.append(process_time)

    def get_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.frame_times:
            return {}

        return {
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times),
            'max_frame_time': max(self.frame_times),
            'min_frame_time': min(self.frame_times),
            'fps': 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0,
            'avg_process_time': sum(self.process_times) / len(self.process_times) if self.process_times else 0,
            'frames_processed': len(self.frame_times)
        }


class GestureCache:
    """手势结果缓存"""
    def __init__(self, max_age: float = 0.1):
        self.cache = {}
        self.timestamps = {}
        self.max_age = max_age

    def get(self, key: str) -> Optional[Any]:
        """获取缓存结果"""
        current_time = time.time()
        if key in self.cache and (current_time - self.timestamps[key]) < self.max_age:
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """设置缓存结果"""
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear_expired(self):
        """清除过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if (current_time - timestamp) >= self.max_age
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)


class GestureStabilizer:
    """手势稳定器"""
    def __init__(self, history_size: int = 5, confidence_threshold: float = 0.7):
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.gesture_history = defaultdict(lambda: deque(maxlen=history_size))
        self.stability_scores = {}

    def add_gesture(self, hand_id: str, gesture_type: str, confidence: float):
        """添加手势检测结果"""
        self.gesture_history[hand_id].append((gesture_type, confidence, time.time()))

    def get_stable_gesture(self, hand_id: str) -> Tuple[str, float]:
        """获取稳定的手势类型和置信度"""
        if hand_id not in self.gesture_history or not self.gesture_history[hand_id]:
            return 'none', 0.0

        recent_gestures = list(self.gesture_history[hand_id])
        gesture_counts = defaultdict(int)
        total_confidence = defaultdict(float)

        for gesture_type, confidence, timestamp in recent_gestures:
            gesture_counts[gesture_type] += 1
            total_confidence[gesture_type] += confidence

        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        stability_score = gesture_counts[most_common_gesture] / len(recent_gestures)
        avg_confidence = total_confidence[most_common_gesture] / gesture_counts[most_common_gesture]

        if stability_score >= 0.6 and avg_confidence >= self.confidence_threshold:
            return most_common_gesture, avg_confidence

        return 'none', 0.0


class UserStateManager:
    """用户状态管理器"""
    def __init__(self, absence_timeout: float = 2.0):
        self.absence_timeout = absence_timeout
        self.last_detection_time = 0
        self.user_present = False
        self.presence_history = deque(maxlen=10)
        self.state_persistence = {}

    def update_presence(self, hands_detected: int):
        """更新用户存在状态"""
        current_time = time.time()
        has_hands = hands_detected > 0

        if has_hands:
            self.last_detection_time = current_time

        time_since_last_detection = current_time - self.last_detection_time
        self.user_present = time_since_last_detection < self.absence_timeout
        self.presence_history.append((current_time, self.user_present))

    def get_presence_confidence(self) -> float:
        """获取用户存在的置信度"""
        if not self.presence_history:
            return 0.0
        recent_presence = [present for _, present in self.presence_history]
        return sum(recent_presence) / len(recent_presence)


class ConductorGestureAnalyzer:
    """专业指挥手势分析器"""

    @staticmethod
    def analyze_conducting_pattern(hand_positions: List[Tuple[float, float]],
                                 timestamps: List[float]) -> Dict[str, float]:
        """分析指挥动作模式"""
        if len(hand_positions) < 3:
            return {'pattern_type': 'none', 'confidence': 0.0}

        velocities = []
        for i in range(1, len(hand_positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = hand_positions[i][0] - hand_positions[i-1][0]
                dy = hand_positions[i][1] - hand_positions[i-1][1]
                velocity = math.sqrt(dx*dx + dy*dy) / dt
                velocities.append(velocity)

        if not velocities:
            return {'pattern_type': 'none', 'confidence': 0.0}

        avg_velocity = sum(velocities) / len(velocities)
        velocity_variance = sum((v - avg_velocity)**2 for v in velocities) / len(velocities)

        if avg_velocity > 0.5 and velocity_variance > 0.1:
            return {'pattern_type': 'forte_conducting', 'confidence': 0.8}
        elif avg_velocity > 0.2 and velocity_variance < 0.05:
            return {'pattern_type': 'legato_conducting', 'confidence': 0.7}
        elif len(set(v > avg_velocity for v in velocities[-5:])) <= 2:
            return {'pattern_type': 'steady_tempo', 'confidence': 0.6}
        else:
            return {'pattern_type': 'expressive_conducting', 'confidence': 0.5}

    @staticmethod
    def detect_beat_pattern(hand_center_history: deque, timestamps: deque) -> Dict[str, Any]:
        """检测拍子模式"""
        if len(hand_center_history) < 8:
            return {'beats_per_measure': 4, 'tempo_bpm': 120, 'confidence': 0.0}

        y_positions = [pos[1] for pos in hand_center_history]
        peaks = []

        for i in range(1, len(y_positions) - 1):
            if y_positions[i] < y_positions[i-1] and y_positions[i] < y_positions[i+1]:
                peaks.append((i, list(timestamps)[i]))

        if len(peaks) < 2:
            return {'beats_per_measure': 4, 'tempo_bpm': 120, 'confidence': 0.0}

        intervals = []
        for i in range(1, len(peaks)):
            interval = peaks[i][1] - peaks[i-1][1]
            intervals.append(interval)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            tempo_bpm = 60.0 / avg_interval if avg_interval > 0 else 120

            if avg_interval < 0.4:
                beats_per_measure = 2
            elif avg_interval < 0.6:
                beats_per_measure = 3
            else:
                beats_per_measure = 4

            confidence = min(1.0, len(peaks) / 8.0)

            return {
                'beats_per_measure': beats_per_measure,
                'tempo_bpm': max(60, min(200, tempo_bpm)),
                'confidence': confidence
            }

        return {'beats_per_measure': 4, 'tempo_bpm': 120, 'confidence': 0.0}


class FrameBuffer:
    """帧缓冲器 - 管理视频帧的高效处理"""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.frames: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_frame(self, frame: np.ndarray) -> bool:
        """添加新帧"""
        with self.lock:
            self.frames.append({
                'frame': frame.copy(),
                'timestamp': time.time()
            })
        return True

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """获取最新帧"""
        with self.lock:
            if self.frames:
                return self.frames[-1]
        return None

    def get_frame_for_processing(self) -> Optional[Dict[str, Any]]:
        """获取用于处理的帧（可能跳帧）"""
        with self.lock:
            if self.frames:
                if len(self.frames) >= 2:
                    return self.frames[-2]
        return None


# ============================================================================
# 主类 (Main Class)
# ============================================================================

class GestureDetector:
    """
    统一手势检测器

    功能：
    - 基础手势检测（拳头、张开、1-5指）
    - 3D手掌旋转矩阵计算
    - 跨平台支持（macOS/Windows/Linux）
    - GPU/CPU自动回退
    - 性能优化（帧跳跃、缓存）
    - 指挥手势分析
    - 7区域声部划分
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化手势检测器

        Args:
            config: 可选配置字典
        """
        # 默认配置
        self.config = {
            'frame_skip': 2,
            'max_num_hands': 2,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'cache_enabled': True,
            'stability_check': True,
            'performance_monitoring': True,
            'gesture_history_size': 10,
            'position_smoothing': 0.8,
            'confidence_threshold': 0.7
        }
        if config:
            self.config.update(config)

        # 平台检测
        self.platform_info = self._detect_platform()
        print(f"运行平台: {self.platform_info['processor_type']} ({self.platform_info['system']})")

        # 初始化MediaPipe（包含GPU/CPU回退）
        self._initialize_mediapipe()

        # 性能优化组件
        self.frame_counter = 0
        self.last_mediapipe_result = None
        self.last_mediapipe_frame = 0

        # 可选组件
        self.performance_monitor = PerformanceMonitor() if self.config['performance_monitoring'] else None
        self.gesture_cache = GestureCache() if self.config['cache_enabled'] else None
        self.gesture_stabilizer = GestureStabilizer() if self.config['stability_check'] else None
        self.user_state_manager = UserStateManager()
        self.conductor_analyzer = ConductorGestureAnalyzer()
        self.frame_buffer = FrameBuffer(max_size=5)

        # 手势历史记录
        self.hand_position_history = defaultdict(lambda: deque(maxlen=self.config['gesture_history_size']))
        self.hand_timestamp_history = defaultdict(lambda: deque(maxlen=self.config['gesture_history_size']))
        self.gesture_duration_tracker = defaultdict(float)

        # 7区域声部划分
        self.voice_regions = {
            'Tromba_I+II+III_in_D': {
                'id': 1,
                'bounds': {'x1': 0.0, 'y1': 0.0, 'x2': 0.33, 'y2': 0.4},
                'center': [0.165, 0.2],
                'color': (255, 100, 100),
                'activation_history': deque(maxlen=5)
            },
            'Violins_in_D': {
                'id': 2,
                'bounds': {'x1': 0.33, 'y1': 0.0, 'x2': 0.67, 'y2': 0.4},
                'center': [0.5, 0.2],
                'color': (100, 255, 100),
                'activation_history': deque(maxlen=5)
            },
            'Viola_in_D': {
                'id': 3,
                'bounds': {'x1': 0.67, 'y1': 0.0, 'x2': 1.0, 'y2': 0.4},
                'center': [0.835, 0.2],
                'color': (100, 100, 255),
                'activation_history': deque(maxlen=5)
            },
            'Oboe_I_in_D': {
                'id': 4,
                'bounds': {'x1': 0.0, 'y1': 0.4, 'x2': 0.33, 'y2': 0.8},
                'center': [0.165, 0.6],
                'color': (255, 255, 100),
                'activation_history': deque(maxlen=5)
            },
            'Continuo_in_D': {
                'id': 5,
                'bounds': {'x1': 0.67, 'y1': 0.4, 'x2': 1.0, 'y2': 0.8},
                'center': [0.835, 0.6],
                'color': (255, 100, 255),
                'activation_history': deque(maxlen=5)
            },
            'Organo_obligato_in_D': {
                'id': 6,
                'bounds': {'x1': 0.0, 'y1': 0.8, 'x2': 0.5, 'y2': 1.0},
                'center': [0.25, 0.9],
                'color': (100, 255, 255),
                'activation_history': deque(maxlen=5)
            },
            'Timpani_in_D': {
                'id': 7,
                'bounds': {'x1': 0.5, 'y1': 0.8, 'x2': 1.0, 'y2': 1.0},
                'center': [0.75, 0.9],
                'color': (200, 150, 100),
                'activation_history': deque(maxlen=5)
            }
        }

        # 中央控制区域
        self.central_control_region = {
            'bounds': {'x1': 0.33, 'y1': 0.4, 'x2': 0.67, 'y2': 0.8},
            'center': [0.5, 0.6],
            'color': (150, 150, 150),
            'activation_history': deque(maxlen=5)
        }

        # 手势数据
        self.gesture_data = {
            'hands_detected': 0,
            'left_hand': self._create_empty_hand_data(),
            'right_hand': self._create_empty_hand_data(),
            'hands': [],
            'active_regions': {},
            'central_control_active': False,
            'gesture_type': 'none',
            'hand_openness': 0.0,
            'hand_center': [0.5, 0.5],
            'gesture_strength': 0.0,
            'timestamp': 0.0,
            'combined_rotation': 0.0,
            'combined_rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'region_activations': {},
            'conducting_pattern': {},
            'beat_analysis': {},
            'user_presence_confidence': 0.0,
            'performance_stats': {},
            'platform_info': self.platform_info
        }

        # 相机设置
        self.camera = None
        self.is_running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self._lock = threading.Lock()

        # 调试计数器
        self._debug_counter = 0

    def _create_empty_hand_data(self) -> Dict[str, Any]:
        """创建空的手部数据结构"""
        return {
            'detected': False,
            'landmarks': [],
            'gesture': 'none',
            'openness': 0.0,
            'center': [0.5, 0.5],
            'rotation_angle': 0.0,
            'palm_direction': [0.0, 1.0],
            'rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }

    def _detect_platform(self) -> Dict[str, Any]:
        """检测运行平台和架构信息"""
        system = platform.system()
        machine = platform.machine()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # 获取macOS版本信息
        macos_version = None
        macos_version_tuple = (0, 0, 0)
        if system == "Darwin":
            try:
                macos_version = platform.mac_ver()[0]  # e.g., "10.15.7" or "14.0"
                parts = macos_version.split('.')
                macos_version_tuple = tuple(int(p) for p in parts[:3] if p.isdigit())
                # 补齐到3位
                while len(macos_version_tuple) < 3:
                    macos_version_tuple = macos_version_tuple + (0,)
                print(f"macOS 版本: {macos_version}")
            except Exception as e:
                print(f"无法获取macOS版本: {e}")

        if system == "Darwin":
            if machine == "arm64":
                processor_type = "Apple Silicon"
                has_gpu_acceleration = True
            else:
                processor_type = "Intel"
                # Intel Mac 在 macOS 10.14+ 上可能支持部分GPU加速
                # 但为了稳定性，默认禁用
                has_gpu_acceleration = False
        elif system == "Windows":
            processor_type = "x86_64" if machine == "AMD64" else machine
            has_gpu_acceleration = True
        else:
            processor_type = machine
            has_gpu_acceleration = False

        return {
            'system': system,
            'machine': machine,
            'processor_type': processor_type,
            'python_version': python_version,
            'has_gpu_acceleration': has_gpu_acceleration,
            'mediapipe_delegate': None,
            'macos_version': macos_version,
            'macos_version_tuple': macos_version_tuple
        }

    def _initialize_mediapipe(self):
        """初始化MediaPipe，包含GPU/CPU回退逻辑"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        if self.platform_info['has_gpu_acceleration']:
            try:
                print("尝试启用GPU加速...")
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.config['max_num_hands'],
                    min_detection_confidence=self.config['min_detection_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    model_complexity=1
                )
                self.platform_info['mediapipe_delegate'] = 'GPU'
                print("GPU加速启用成功")
            except Exception as e:
                print(f"GPU加速失败，切换到CPU模式: {e}")
                self._initialize_cpu_mode()
        else:
            print("使用CPU模式...")
            self._initialize_cpu_mode()

    def _initialize_cpu_mode(self):
        """初始化CPU模式"""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config['max_num_hands'],
                min_detection_confidence=0.6,
                min_tracking_confidence=0.4,
                model_complexity=0
            )
            self.platform_info['mediapipe_delegate'] = 'CPU'
            print("CPU模式启用成功")
        except Exception as e:
            print(f"CPU模式也失败了: {e}")
            raise Exception("无法初始化MediaPipe，请检查安装")

    def start_camera(self, camera_id: int = 0) -> bool:
        """启动摄像头，包含跨平台兼容性处理"""
        camera_backends = []

        if self.platform_info['system'] == 'Darwin':
            macos_ver = self.platform_info.get('macos_version_tuple', (0, 0, 0))

            # macOS 10.14 (Mojave) 之前，AVFoundation 可能不稳定
            # 先尝试 AVFoundation，再回退到 CAP_ANY
            if macos_ver >= (10, 14, 0):
                camera_backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            else:
                # 老版本 macOS，优先使用通用后端
                print(f"检测到老版本 macOS {self.platform_info.get('macos_version', 'unknown')}，使用兼容模式")
                camera_backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]

        elif self.platform_info['system'] == 'Windows':
            camera_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            camera_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        backend_names = {
            cv2.CAP_AVFOUNDATION: 'AVFoundation',
            cv2.CAP_ANY: 'Auto',
            cv2.CAP_DSHOW: 'DirectShow',
            cv2.CAP_MSMF: 'MSMF',
            cv2.CAP_V4L2: 'V4L2'
        }

        for backend in camera_backends:
            backend_name = backend_names.get(backend, str(backend))
            try:
                print(f"尝试使用 {backend_name} 后端打开摄像头...")
                self.camera = cv2.VideoCapture(camera_id, backend)

                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)

                    # 多读几帧确保摄像头稳定
                    for _ in range(3):
                        ret, test_frame = self.camera.read()
                        if ret and test_frame is not None:
                            break
                        time.sleep(0.1)

                    if ret and test_frame is not None:
                        print(f"摄像头启动成功，后端: {backend_name}")
                        self.platform_info['camera_backend'] = backend_name
                        break
                    else:
                        print(f"{backend_name} 后端无法读取帧，尝试下一个...")
                        self.camera.release()
                        self.camera = None
                else:
                    print(f"{backend_name} 后端无法打开摄像头，尝试下一个...")
                    if self.camera:
                        self.camera.release()
                        self.camera = None

            except Exception as e:
                print(f"{backend_name} 后端异常: {e}")
                if self.camera:
                    self.camera.release()
                    self.camera = None
                continue

        if not self.camera or not self.camera.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id}，请检查：\n"
                          f"  1. 摄像头是否被其他应用占用\n"
                          f"  2. 是否授予了摄像头权限\n"
                          f"  3. 运行: sudo killall VDCAssistant (macOS)")

        self.is_running = True
        self.thread = threading.Thread(target=self._camera_loop)
        self.thread.daemon = False  # 非守护线程，确保正常退出时能完成清理
        self.thread.start()

        return True

    def stop_camera(self):
        """停止摄像头"""
        print("[GestureDetector] 正在停止摄像头...")

        # 设置停止标志
        self.is_running = False

        # 等待线程结束（带超时）
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # 最多等待2秒
            if self.thread.is_alive():
                print("[GestureDetector] 警告: 摄像头线程未能在超时内结束")

        # 释放摄像头资源
        if self.camera:
            try:
                self.camera.release()
                print("[GestureDetector] 摄像头已释放")
            except Exception as e:
                print(f"[GestureDetector] 摄像头释放异常: {e}")
            finally:
                self.camera = None

    def force_release_camera(self):
        """强制释放摄像头（用于紧急清理）"""
        self.is_running = False
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None

    def _camera_loop(self):
        """摄像头循环线程"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(frame)

                with self.frame_lock:
                    self.current_frame = processed_frame

            time.sleep(1/30)

    def get_current_frame(self):
        """获取当前处理后的帧"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def _should_process_mediapipe(self) -> bool:
        """判断是否应该处理MediaPipe"""
        return (self.frame_counter % self.config['frame_skip']) == 0

    def process_frame(self, frame: np.ndarray, show_regions: bool = False) -> np.ndarray:
        """处理单帧图像"""
        if frame is None:
            return None

        with self._lock:
            if self.performance_monitor:
                self.performance_monitor.start_frame()

            process_start = time.time()
            self.frame_counter += 1

            should_process = self._should_process_mediapipe()

            if should_process:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                self.last_mediapipe_result = results
                self.last_mediapipe_frame = self.frame_counter
            else:
                results = self.last_mediapipe_result

            # 重置检测数据
            self.gesture_data['hands_detected'] = 0
            self.gesture_data['left_hand'] = self._create_empty_hand_data()
            self.gesture_data['right_hand'] = self._create_empty_hand_data()
            self.gesture_data['hands'] = []
            self.gesture_data['active_regions'] = {}
            self.gesture_data['central_control_active'] = False
            self.gesture_data['timestamp'] = time.time()

            for region_name in self.voice_regions.keys():
                self.gesture_data['region_activations'][region_name] = 0.0

            current_time = time.time()

            if results and results.multi_hand_landmarks and results.multi_handedness:
                self.gesture_data['hands_detected'] = len(results.multi_hand_landmarks)

                for i, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                    hand_label = handedness.classification[0].label.lower()
                    hand_id = f"{hand_label}_{i}"

                    gesture_type = self.detect_gesture_type(landmarks)
                    openness = self.calculate_hand_openness(landmarks)
                    center = self.calculate_hand_center(landmarks)
                    rotation_angle, palm_direction, rotation_matrix = self.calculate_palm_rotation(landmarks)

                    hand_data = {
                        'id': hand_id,
                        'label': hand_label,
                        'detected': True,
                        'landmarks': landmarks,
                        'gesture': gesture_type,
                        'gesture_type': gesture_type,
                        'openness': openness,
                        'center': center,
                        'rotation_angle': rotation_angle,
                        'palm_direction': palm_direction,
                        'rotation_matrix': rotation_matrix
                    }

                    # 区域检测
                    active_region = self.detect_hand_in_region(center)
                    if active_region:
                        activation_strength = self.calculate_region_activation_strength(center, active_region)
                        hand_data['active_region'] = active_region
                        hand_data['activation_strength'] = activation_strength

                        if active_region == 'central_control':
                            self.gesture_data['central_control_active'] = True
                        elif active_region in self.voice_regions:
                            self.gesture_data['active_regions'][active_region] = {
                                'hand_id': hand_id,
                                'activation_strength': activation_strength
                            }
                            self.gesture_data['region_activations'][active_region] = activation_strength

                    if hand_label == 'left':
                        self.gesture_data['left_hand'] = hand_data
                    else:
                        self.gesture_data['right_hand'] = hand_data

                    self.gesture_data['hands'].append(hand_data)

                    # 更新位置历史
                    self.hand_position_history[hand_id].append(tuple(center))
                    self.hand_timestamp_history[hand_id].append(current_time)

                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

            # 更新用户状态
            self.user_state_manager.update_presence(self.gesture_data['hands_detected'])
            self.gesture_data['user_presence_confidence'] = self.user_state_manager.get_presence_confidence()

            # 计算综合手势强度
            left_strength = self.gesture_data['left_hand']['openness'] if self.gesture_data['left_hand']['detected'] else 0
            right_strength = self.gesture_data['right_hand']['openness'] if self.gesture_data['right_hand']['detected'] else 0
            self.gesture_data['gesture_strength'] = max(left_strength, right_strength)

            # 计算综合旋转
            self.gesture_data['combined_rotation'], self.gesture_data['combined_rotation_matrix'] = self.calculate_combined_rotation()

            # 分析指挥模式
            self._analyze_conducting_patterns()

            # 绘制信息
            self.draw_info_on_frame(frame)

            if show_regions:
                frame = self.draw_regions_overlay(frame)

            # 记录性能
            process_time = time.time() - process_start
            if self.performance_monitor:
                self.performance_monitor.record_process_time(process_time)
                self.performance_monitor.end_frame()
                self.gesture_data['performance_stats'] = self.performance_monitor.get_stats()

            if self.gesture_cache:
                self.gesture_cache.clear_expired()

            return frame

    def detect_gesture_type(self, landmarks: List[List[float]]) -> str:
        """检测手势类型"""
        if not landmarks or len(landmarks) < 21:
            return 'none'

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        wrist = landmarks[0]
        thumb_ip = landmarks[3]

        fingers_up = []

        # 拇指检测
        thumb_tip_to_wrist = math.sqrt(
            (thumb_tip[0] - wrist[0])**2 + (thumb_tip[1] - wrist[1])**2
        )
        thumb_ip_to_wrist = math.sqrt(
            (thumb_ip[0] - wrist[0])**2 + (thumb_ip[1] - wrist[1])**2
        )
        thumb_up = thumb_tip_to_wrist > thumb_ip_to_wrist * 1.1
        fingers_up.append(thumb_up)

        # 其他四指检测
        finger_landmarks = [
            (index_tip, index_pip, landmarks[5]),
            (middle_tip, middle_pip, landmarks[9]),
            (ring_tip, ring_pip, landmarks[13]),
            (pinky_tip, pinky_pip, landmarks[17])
        ]

        for tip, pip, mcp in finger_landmarks:
            basic_up = tip[1] < pip[1]
            tip_to_wrist = math.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
            mcp_to_wrist = math.sqrt((mcp[0] - wrist[0])**2 + (mcp[1] - wrist[1])**2)
            distance_up = tip_to_wrist > mcp_to_wrist * 1.1
            finger_up = basic_up and distance_up
            fingers_up.append(finger_up)

        fingers_count = sum(fingers_up)

        gesture_map = {
            0: 'fist',
            1: 'one',
            2: 'two',
            3: 'three',
            4: 'four',
            5: 'open_hand'
        }

        return gesture_map.get(fingers_count, 'unknown')

    def calculate_hand_openness(self, landmarks: List[List[float]]) -> float:
        """计算手部张开程度"""
        if not landmarks or len(landmarks) < 21:
            return 0.0

        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        total_distance = 0
        count = 0

        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                dist = math.sqrt(
                    (finger_tips[i][0] - finger_tips[j][0])**2 +
                    (finger_tips[i][1] - finger_tips[j][1])**2
                )
                total_distance += dist
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        return min(avg_distance / 0.3, 1.0)

    def calculate_hand_center(self, landmarks: List[List[float]]) -> List[float]:
        """计算手部中心点"""
        if not landmarks or len(landmarks) < 21:
            return [0.5, 0.5]

        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return [center_x, center_y]

    def calculate_palm_rotation(self, landmarks: List[List[float]]) -> Tuple[float, List[float], List[List[float]]]:
        """计算手掌的3D旋转矩阵"""
        if not landmarks or len(landmarks) < 21:
            identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            return 0.0, [1.0, 0.0], identity_matrix

        try:
            wrist = landmarks[0]
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            pinky_mcp = landmarks[17]

            x_axis = [pinky_mcp[0] - index_mcp[0],
                      pinky_mcp[1] - index_mcp[1],
                      pinky_mcp[2] - index_mcp[2]]

            y_axis = [middle_mcp[0] - wrist[0],
                      middle_mcp[1] - wrist[1],
                      middle_mcp[2] - wrist[2]]

            def normalize(v):
                length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                return [v[0]/length, v[1]/length, v[2]/length] if length > 0.0001 else [0, 0, 1]

            x_axis = normalize(x_axis)
            y_axis = normalize(y_axis)

            z_axis = [
                x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
                x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
                x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0]
            ]
            z_axis = normalize(z_axis)

            y_axis = [
                z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
                z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
                z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0]
            ]
            y_axis = normalize(y_axis)

            rotation_matrix = [
                [x_axis[0], y_axis[0], z_axis[0]],
                [x_axis[1], y_axis[1], z_axis[1]],
                [x_axis[2], y_axis[2], z_axis[2]]
            ]

            angle = math.atan2(x_axis[1], x_axis[0])

            return angle, x_axis, rotation_matrix

        except (IndexError, ZeroDivisionError, ValueError):
            identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            return 0.0, [1.0, 0.0], identity_matrix

    def calculate_combined_rotation(self) -> Tuple[float, List[List[float]]]:
        """计算综合旋转角度和3D旋转矩阵"""
        left_detected = self.gesture_data['left_hand']['detected']
        right_detected = self.gesture_data['right_hand']['detected']

        identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        if left_detected and right_detected:
            primary_matrix = self.gesture_data['right_hand']['rotation_matrix']
            primary_angle = self.gesture_data['right_hand']['rotation_angle']
            return primary_angle, primary_matrix
        elif left_detected:
            return self.gesture_data['left_hand']['rotation_angle'], self.gesture_data['left_hand']['rotation_matrix']
        elif right_detected:
            return self.gesture_data['right_hand']['rotation_angle'], self.gesture_data['right_hand']['rotation_matrix']
        else:
            return 0.0, identity_matrix

    def detect_hand_in_region(self, hand_center: List[float]) -> Optional[str]:
        """检测手部中心点位于哪个区域"""
        x, y = hand_center

        central_bounds = self.central_control_region['bounds']
        if (central_bounds['x1'] <= x <= central_bounds['x2'] and
            central_bounds['y1'] <= y <= central_bounds['y2']):
            return 'central_control'

        for region_name, region_data in self.voice_regions.items():
            bounds = region_data['bounds']
            if (bounds['x1'] <= x <= bounds['x2'] and
                bounds['y1'] <= y <= bounds['y2']):
                return region_name

        return None

    def calculate_region_activation_strength(self, hand_center: List[float], region_name: str) -> float:
        """计算区域激活强度"""
        if region_name == 'central_control':
            region_center = self.central_control_region['center']
        elif region_name in self.voice_regions:
            region_center = self.voice_regions[region_name]['center']
        else:
            return 0.0

        distance = math.sqrt(
            (hand_center[0] - region_center[0])**2 +
            (hand_center[1] - region_center[1])**2
        )

        max_distance = 0.2
        activation = max(0.0, 1.0 - (distance / max_distance))

        return activation

    def _analyze_conducting_patterns(self):
        """分析指挥模式"""
        if not self.gesture_data['hands']:
            return

        primary_hand = None
        for hand in self.gesture_data['hands']:
            if hand['label'] == 'right' or not primary_hand:
                primary_hand = hand
                break

        if not primary_hand:
            return

        hand_id = primary_hand.get('id', 'right_0')

        if (hand_id in self.hand_position_history and
            len(self.hand_position_history[hand_id]) >= 3):

            positions = list(self.hand_position_history[hand_id])
            timestamps = list(self.hand_timestamp_history[hand_id])

            pattern_analysis = self.conductor_analyzer.analyze_conducting_pattern(positions, timestamps)
            self.gesture_data['conducting_pattern'] = pattern_analysis

            beat_analysis = self.conductor_analyzer.detect_beat_pattern(
                self.hand_position_history[hand_id],
                self.hand_timestamp_history[hand_id])
            self.gesture_data['beat_analysis'] = beat_analysis

    def draw_info_on_frame(self, frame: np.ndarray):
        """在帧上绘制信息"""
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 140), (255, 255, 255), 2)

        info_lines = [
            f"Platform: {self.platform_info['processor_type']} ({self.platform_info['mediapipe_delegate']})",
            f"Hands: {self.gesture_data['hands_detected']}",
            f"Strength: {self.gesture_data['gesture_strength']:.2f}",
        ]

        if self.gesture_data['left_hand']['detected']:
            left = self.gesture_data['left_hand']
            info_lines.append(f"Left: {left['gesture']} ({left['openness']:.2f})")

        if self.gesture_data['right_hand']['detected']:
            right = self.gesture_data['right_hand']
            info_lines.append(f"Right: {right['gesture']} ({right['openness']:.2f})")

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)

    def draw_regions_overlay(self, frame: np.ndarray) -> np.ndarray:
        """绘制区域边界和信息"""
        height, width = frame.shape[:2]

        for region_name, region_data in self.voice_regions.items():
            bounds = region_data['bounds']
            color = region_data['color']

            x1 = int(bounds['x1'] * width)
            y1 = int(bounds['y1'] * height)
            x2 = int(bounds['x2'] * width)
            y2 = int(bounds['y2'] * height)

            activation = self.gesture_data['region_activations'].get(region_name, 0.0)
            alpha = 0.3 + activation * 0.7

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(frame, 1 - alpha * 0.3, overlay, alpha * 0.3, 0)

            thickness = 3 if activation > 0.1 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{region_data['id']}-{region_name.split('_')[0]}"
            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        central_bounds = self.central_control_region['bounds']
        central_color = self.central_control_region['color']

        x1 = int(central_bounds['x1'] * width)
        y1 = int(central_bounds['y1'] * height)
        x2 = int(central_bounds['x2'] * width)
        y2 = int(central_bounds['y2'] * height)

        thickness = 3 if self.gesture_data['central_control_active'] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), central_color, thickness)

        label = "Central Control"
        if self.gesture_data['central_control_active']:
            label += " (ACTIVE)"
        cv2.putText(frame, label, (x1 + 5, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, central_color, 1)

        return frame

    def get_gesture_data(self) -> Dict[str, Any]:
        """获取当前手势数据"""
        return self.gesture_data.copy()

    def get_conductor_commands(self) -> Dict[str, Any]:
        """获取增强的指挥控制命令"""
        commands = {
            'tempo_change': 0.0,
            'volume_change': 0.0,
            'expression_level': 0.0,
            'attack_type': 'normal',
            'stop_command': False,
            'active_voices': self.get_active_voices(),
            'conducting_pattern': self.gesture_data.get('conducting_pattern', {}),
            'beat_analysis': self.gesture_data.get('beat_analysis', {}),
            'user_presence': self.gesture_data.get('user_presence_confidence', 0.0)
        }

        primary_hand = None
        for hand in self.gesture_data['hands']:
            if hand['label'] == 'right':
                primary_hand = hand
                break

        if primary_hand:
            gesture_type = primary_hand.get('gesture_type', 'none')
            openness = primary_hand.get('openness', 0.0)

            if gesture_type == 'fist':
                commands['stop_command'] = True
                commands['volume_change'] = -1.0
            elif gesture_type == 'open_hand':
                commands['volume_change'] = openness * 1.2
                commands['expression_level'] = openness
                commands['attack_type'] = 'forte'

        return commands

    def get_active_voices(self) -> List[Dict[str, Any]]:
        """获取激活的声部信息"""
        active_voices = []

        for region_name, activation_strength in self.gesture_data['region_activations'].items():
            if activation_strength > 0.1:
                voice_data = {
                    'name': region_name,
                    'id': self.voice_regions[region_name]['id'],
                    'activation_strength': activation_strength,
                    'hand_info': self.gesture_data['active_regions'].get(region_name, {})
                }
                active_voices.append(voice_data)

        return sorted(active_voices, key=lambda x: x['activation_strength'], reverse=True)

    def get_region_info(self) -> Dict[str, Any]:
        """获取区域划分信息"""
        return {
            'voice_regions': self.voice_regions,
            'central_control_region': self.central_control_region
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.performance_monitor:
            return self.performance_monitor.get_stats()
        return {}

    def update_config(self, new_config: Dict[str, Any]):
        """动态更新配置"""
        self.config.update(new_config)

        if any(key in new_config for key in ['max_num_hands', 'min_detection_confidence', 'min_tracking_confidence']):
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config['max_num_hands'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

    def cleanup(self):
        """清理资源"""
        self.stop_camera()
        if hasattr(self.hands, 'close'):
            self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ============================================================================
# 兼容性别名
# ============================================================================

# 保持与原有代码的兼容
CrossPlatformGestureDetector = GestureDetector
HandGestureDetector = GestureDetector
OptimizedHandGestureDetector = GestureDetector


# ============================================================================
# TouchDesigner 接口函数
# ============================================================================

def get_gesture_data():
    """获取当前手势数据"""
    if 'op' in dir() and hasattr(op, 'detector'):
        return op.detector.gesture_data
    return None


def get_conductor_commands():
    """获取指挥控制命令"""
    if 'op' in dir() and hasattr(op, 'detector'):
        return op.detector.get_conductor_commands()
    return None


def get_active_voices():
    """获取激活的声部信息"""
    if 'op' in dir() and hasattr(op, 'detector'):
        return op.detector.get_active_voices()
    return []


def get_region_info():
    """获取区域划分信息"""
    if 'op' in dir() and hasattr(op, 'detector'):
        return {
            'voice_regions': op.detector.voice_regions,
            'central_control_region': op.detector.central_control_region
        }
    return None


def get_performance_stats():
    """获取性能统计信息"""
    if 'op' in dir() and hasattr(op, 'detector'):
        return op.detector.get_performance_stats()
    return {}


def initialize_detector(config=None):
    """初始化手势检测器"""
    if 'op' in dir():
        if not hasattr(op, 'detector') or config:
            op.detector = GestureDetector(config)
        return True
    return False


def process_camera_frame(frame_data, show_regions=True):
    """处理摄像头帧数据"""
    if 'op' in dir():
        if not hasattr(op, 'detector'):
            initialize_detector()
        processed_frame = op.detector.process_frame(frame_data, show_regions)
        return processed_frame
    return frame_data


def set_region_bounds(region_name, x1, y1, x2, y2):
    """动态调整区域边界"""
    if 'op' in dir() and hasattr(op, 'detector') and region_name in op.detector.voice_regions:
        op.detector.voice_regions[region_name]['bounds'] = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
        op.detector.voice_regions[region_name]['center'] = [
            (x1 + x2) / 2, (y1 + y2) / 2
        ]
        return True
    return False


def update_detector_config(config_dict):
    """动态更新检测器配置"""
    if 'op' in dir() and hasattr(op, 'detector'):
        op.detector.update_config(config_dict)
        return True
    return False


def cleanup_detector():
    """清理检测器资源"""
    if 'op' in dir() and hasattr(op, 'detector'):
        op.detector.cleanup()
        delattr(op, 'detector')
        return True
    return False


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("统一手势检测器测试 (Unified Gesture Detector Test)")
    print("=" * 50)

    detector = GestureDetector()

    try:
        detector.start_camera(0)
        print("手势检测启动成功！按 'q' 退出, 'r' 显示区域")

        show_regions = False

        while True:
            frame = detector.get_current_frame()
            if frame is not None:
                if show_regions:
                    frame = detector.draw_regions_overlay(frame)

                cv2.imshow('Unified Gesture Detection', frame)

                data = detector.get_gesture_data()
                if data['hands_detected'] > 0:
                    pass  # 可打印调试信息

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                show_regions = not show_regions
                print(f"区域显示: {'开启' if show_regions else '关闭'}")
            elif key == ord('s'):
                stats = detector.get_performance_stats()
                print(f"性能统计: {stats}")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()
        print("程序已退出")
