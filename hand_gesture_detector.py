"""
TouchDesigner Conductor Gesture Recognition Script - Enhanced Version
使用MediaPipe进行实时指挥家手势识别，通过区域划分控制7个声部

优化特性：
- 性能优化：帧跳跃策略、结果缓存、内存管理
- 增强指挥手势识别：专业指挥动作、手势持续时间、稳定性过滤
- 用户状态管理：出现/离开检测、状态持久化、置信度评分
- 接口优化：简化TouchDesigner接口、性能监控、动态配置
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import threading
import weakref

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.last_report_time = time.time()

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
    def __init__(self, max_age: float = 0.1):  # 100ms缓存
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

        # 计算手势类型频率
        gesture_counts = defaultdict(int)
        total_confidence = defaultdict(float)

        for gesture_type, confidence, timestamp in recent_gestures:
            gesture_counts[gesture_type] += 1
            total_confidence[gesture_type] += confidence

        # 找到最频繁的手势类型
        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        stability_score = gesture_counts[most_common_gesture] / len(recent_gestures)
        avg_confidence = total_confidence[most_common_gesture] / gesture_counts[most_common_gesture]

        # 只有当稳定性和置信度都足够高时才返回手势
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

        # 判断用户是否存在
        time_since_last_detection = current_time - self.last_detection_time
        self.user_present = time_since_last_detection < self.absence_timeout

        self.presence_history.append((current_time, self.user_present))

    def get_presence_confidence(self) -> float:
        """获取用户存在的置信度"""
        if not self.presence_history:
            return 0.0

        recent_presence = [present for _, present in self.presence_history]
        return sum(recent_presence) / len(recent_presence)

    def save_state(self, key: str, value: Any):
        """保存状态"""
        self.state_persistence[key] = {
            'value': value,
            'timestamp': time.time()
        }

    def load_state(self, key: str, max_age: float = 60.0) -> Optional[Any]:
        """加载状态"""
        if key in self.state_persistence:
            state_data = self.state_persistence[key]
            if time.time() - state_data['timestamp'] < max_age:
                return state_data['value']
        return None

class ConductorGestureAnalyzer:
    """专业指挥手势分析器"""

    @staticmethod
    def analyze_conducting_pattern(hand_positions: List[Tuple[float, float]],
                                 timestamps: List[float]) -> Dict[str, float]:
        """分析指挥动作模式"""
        if len(hand_positions) < 3:
            return {'pattern_type': 'none', 'confidence': 0.0}

        # 计算运动轨迹
        velocities = []
        accelerations = []

        for i in range(1, len(hand_positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = hand_positions[i][0] - hand_positions[i-1][0]
                dy = hand_positions[i][1] - hand_positions[i-1][1]
                velocity = math.sqrt(dx*dx + dy*dy) / dt
                velocities.append(velocity)

        for i in range(1, len(velocities)):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                acceleration = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(acceleration)

        # 分析模式
        if not velocities:
            return {'pattern_type': 'none', 'confidence': 0.0}

        avg_velocity = sum(velocities) / len(velocities)
        velocity_variance = sum((v - avg_velocity)**2 for v in velocities) / len(velocities)

        # 检测指挥模式
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

        # 检测峰值（指挥棒的最高点）
        y_positions = [pos[1] for pos in hand_center_history]
        peaks = []

        for i in range(1, len(y_positions) - 1):
            if y_positions[i] < y_positions[i-1] and y_positions[i] < y_positions[i+1]:
                peaks.append((i, timestamps[i]))

        if len(peaks) < 2:
            return {'beats_per_measure': 4, 'tempo_bpm': 120, 'confidence': 0.0}

        # 计算拍子间隔
        intervals = []
        for i in range(1, len(peaks)):
            interval = peaks[i][1] - peaks[i-1][1]
            intervals.append(interval)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            tempo_bpm = 60.0 / avg_interval if avg_interval > 0 else 120

            # 推测拍号
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

class HandGestureDetector:
    """增强版手势检测器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 配置参数
        self.config = {
            'frame_skip': 2,  # 每2帧处理一次MediaPipe
            'max_num_hands': 4,
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

        # 平台信息
        self.platform_info = self._detect_platform()

        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config['max_num_hands'],
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 性能优化组件
        self.frame_counter = 0
        self.last_mediapipe_result = None
        self.last_mediapipe_frame = 0

        # 监控和缓存
        self.performance_monitor = PerformanceMonitor() if self.config['performance_monitoring'] else None
        self.gesture_cache = GestureCache() if self.config['cache_enabled'] else None
        self.gesture_stabilizer = GestureStabilizer() if self.config['stability_check'] else None
        self.user_state_manager = UserStateManager()
        self.conductor_analyzer = ConductorGestureAnalyzer()

        # 手势历史记录
        self.hand_position_history = defaultdict(lambda: deque(maxlen=self.config['gesture_history_size']))
        self.hand_timestamp_history = defaultdict(lambda: deque(maxlen=self.config['gesture_history_size']))
        self.gesture_duration_tracker = defaultdict(float)

        # 定义7个声部区域
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

        # 手势状态变量
        self.gesture_data = {
            'hands_detected': 0,
            'hands': [],
            'active_regions': {},
            'central_control_active': False,
            'gesture_type': 'none',
            'hand_openness': 0.0,
            'hand_center': [0.5, 0.5],
            'gesture_strength': 0.0,
            'region_activations': {},
            'conducting_pattern': {},
            'beat_analysis': {},
            'user_presence_confidence': 0.0,
            'performance_stats': {}
        }

        # 线程锁
        self._lock = threading.Lock()

        # 摄像头相关
        self.cap = None
        self.current_frame = None
        self.camera_active = False

    def _detect_platform(self) -> Dict[str, Any]:
        """检测平台信息"""
        import platform
        import sys

        system = platform.system()
        machine = platform.machine()

        # 检测处理器类型
        if system == 'Darwin':  # macOS
            if 'arm' in machine.lower() or 'apple' in platform.processor().lower():
                processor_type = 'Apple Silicon'
            else:
                processor_type = 'Intel'
        elif system == 'Windows':
            processor_type = 'Intel/AMD'
        else:
            processor_type = 'Unknown'

        # 检测GPU支持
        has_gpu = False
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            has_gpu = len(gpus) > 0
        except:
            pass

        return {
            'system': system,
            'machine': machine,
            'processor_type': processor_type,
            'python_version': sys.version,
            'python_executable': sys.executable,
            'has_gpu_acceleration': has_gpu,
            'mediapipe_delegate': 'GPU' if has_gpu else 'CPU'
        }

    def start_camera(self, camera_id: int = 0) -> bool:
        """启动摄像头"""
        try:
            if self.cap is not None:
                self.stop_camera()

            self.cap = cv2.VideoCapture(camera_id)

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {camera_id}")
                return False

            self.camera_active = True
            print(f"摄像头 {camera_id} 启动成功")
            return True

        except Exception as e:
            print(f"启动摄像头失败: {e}")
            return False

    def stop_camera(self):
        """停止摄像头"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.camera_active = False
            self.current_frame = None
            print("摄像头已停止")

        except Exception as e:
            print(f"停止摄像头失败: {e}")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        if not self.camera_active or self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                return self.current_frame
            else:
                return None
        except Exception as e:
            print(f"读取帧失败: {e}")
            return None

    def _should_process_mediapipe(self) -> bool:
        """判断是否应该处理MediaPipe"""
        return (self.frame_counter % self.config['frame_skip']) == 0

    def _get_cached_result(self, frame_hash: str) -> Optional[Any]:
        """获取缓存的处理结果"""
        if self.gesture_cache:
            return self.gesture_cache.get(frame_hash)
        return None

    def _cache_result(self, frame_hash: str, result: Any):
        """缓存处理结果"""
        if self.gesture_cache:
            self.gesture_cache.set(frame_hash, result)

    def detect_gesture_type_enhanced(self, landmarks: List[List[float]],
                                   hand_id: str) -> Tuple[str, float]:
        """增强版手势类型检测"""
        if not landmarks or len(landmarks) < 21:
            return 'none', 0.0

        # 获取关键点
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        # 检测手指伸直状态
        fingers_up = []
        confidences = []

        # 拇指检测（考虑左右手差异）
        thumb_extended = thumb_tip[0] > thumb_ip[0] if landmarks[4][0] > landmarks[3][0] else thumb_tip[0] < thumb_ip[0]
        fingers_up.append(thumb_extended)
        confidences.append(0.8)  # 拇指检测置信度

        # 其他四指检测
        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip),
                        (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            finger_extended = tip[1] < pip[1]  # Y轴向上为负
            fingers_up.append(finger_extended)

            # 计算置信度（基于关节角度）
            joint_distance = abs(tip[1] - pip[1])
            confidence = min(1.0, joint_distance * 5)  # 经验值
            confidences.append(confidence)

        fingers_count = sum(fingers_up)
        avg_confidence = sum(confidences) / len(confidences)

        # 专业指挥手势识别
        gesture_type = 'none'
        gesture_confidence = 0.0

        # 计算手腕到指尖的向量，用于检测指挥方向
        wrist_to_index = [index_tip[0] - wrist[0], index_tip[1] - wrist[1]]
        wrist_to_middle = [middle_tip[0] - wrist[0], middle_tip[1] - wrist[1]]

        # 指挥棒握持检测（食指+中指伸出，其他收拢）
        if (fingers_up[1] and fingers_up[2] and
            not fingers_up[3] and not fingers_up[4]):
            gesture_type = 'baton_grip'
            gesture_confidence = avg_confidence * 0.9

        # 开手强奏手势
        elif fingers_count >= 4:
            # 检查手指张开程度
            finger_spread = self._calculate_finger_spread(landmarks)
            if finger_spread > 0.15:
                gesture_type = 'forte_open'
                gesture_confidence = avg_confidence * finger_spread * 5
            else:
                gesture_type = 'open_hand'
                gesture_confidence = avg_confidence * 0.8

        # 握拳停止手势
        elif fingers_count <= 1:
            # 检查拳头紧密程度
            fist_tightness = self._calculate_fist_tightness(landmarks)
            if fist_tightness > 0.8:
                gesture_type = 'stop_fist'
                gesture_confidence = avg_confidence * fist_tightness
            else:
                gesture_type = 'fist'
                gesture_confidence = avg_confidence * 0.7

        # 精确指向手势
        elif fingers_up[1] and not any(fingers_up[2:]):
            # 检查指向精确度
            pointing_precision = self._calculate_pointing_precision(landmarks)
            gesture_type = 'precise_pointing'
            gesture_confidence = avg_confidence * pointing_precision

        # 表现性指挥手势
        elif 2 <= fingers_count <= 3:
            # 分析手部形状和方向
            hand_curvature = self._calculate_hand_curvature(landmarks)
            if hand_curvature > 0.5:
                gesture_type = 'expressive_conducting'
                gesture_confidence = avg_confidence * hand_curvature
            else:
                gesture_type = 'conducting'
                gesture_confidence = avg_confidence * 0.6
        else:
            gesture_type = 'neutral'
            gesture_confidence = avg_confidence * 0.5

        # 使用稳定器平滑结果
        if self.gesture_stabilizer:
            self.gesture_stabilizer.add_gesture(hand_id, gesture_type, gesture_confidence)
            stable_gesture, stable_confidence = self.gesture_stabilizer.get_stable_gesture(hand_id)
            return stable_gesture, stable_confidence

        return gesture_type, gesture_confidence

    def _calculate_finger_spread(self, landmarks: List[List[float]]) -> float:
        """计算手指张开程度"""
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

        return total_distance / count if count > 0 else 0.0

    def _calculate_fist_tightness(self, landmarks: List[List[float]]) -> float:
        """计算拳头紧密程度"""
        # 计算指尖到手掌中心的距离
        palm_center = landmarks[0]  # 使用手腕作为参考点
        finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]

        distances = []
        for tip in finger_tips:
            dist = math.sqrt(
                (tip[0] - palm_center[0])**2 +
                (tip[1] - palm_center[1])**2
            )
            distances.append(dist)

        avg_distance = sum(distances) / len(distances)
        # 距离越小，拳头越紧
        return max(0.0, 1.0 - avg_distance * 8)  # 经验值调整

    def _calculate_pointing_precision(self, landmarks: List[List[float]]) -> float:
        """计算指向精确度"""
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]

        # 计算食指的直线程度
        tip_to_pip = [index_tip[0] - index_pip[0], index_tip[1] - index_pip[1]]
        pip_to_mcp = [index_pip[0] - index_mcp[0], index_pip[1] - index_mcp[1]]

        # 计算角度差异
        if all(v != 0 for v in tip_to_pip + pip_to_mcp):
            dot_product = tip_to_pip[0] * pip_to_mcp[0] + tip_to_pip[1] * pip_to_mcp[1]
            magnitude1 = math.sqrt(tip_to_pip[0]**2 + tip_to_pip[1]**2)
            magnitude2 = math.sqrt(pip_to_mcp[0]**2 + pip_to_mcp[1]**2)

            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # 防止数值误差

            # 角度越小（越接近直线），精确度越高
            return (cos_angle + 1) / 2  # 归一化到0-1

        return 0.5

    def _calculate_hand_curvature(self, landmarks: List[List[float]]) -> float:
        """计算手部弯曲程度"""
        # 使用手指关节点计算整体弯曲
        joints = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]  # MCP关节
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]   # 指尖

        total_curvature = 0
        for joint, tip in zip(joints, tips):
            # 计算关节到指尖的弯曲程度
            joint_to_tip = math.sqrt((tip[0] - joint[0])**2 + (tip[1] - joint[1])**2)
            total_curvature += joint_to_tip

        return min(1.0, total_curvature / 0.4)  # 归一化

    def detect_hand_in_region(self, hand_center: List[float]) -> Optional[str]:
        """检测手部中心点位于哪个区域"""
        x, y = hand_center

        # 首先检查中央控制区域
        central_bounds = self.central_control_region['bounds']
        if (central_bounds['x1'] <= x <= central_bounds['x2'] and
            central_bounds['y1'] <= y <= central_bounds['y2']):
            return 'central_control'

        # 检查声部区域
        for region_name, region_data in self.voice_regions.items():
            bounds = region_data['bounds']
            if (bounds['x1'] <= x <= bounds['x2'] and
                bounds['y1'] <= y <= bounds['y2']):
                return region_name

        return None

    def calculate_region_activation_strength(self, hand_center: List[float],
                                           region_name: str) -> float:
        """计算区域激活强度"""
        if region_name == 'central_control':
            region_center = self.central_control_region['center']
        elif region_name in self.voice_regions:
            region_center = self.voice_regions[region_name]['center']
        else:
            return 0.0

        # 计算距离
        distance = math.sqrt(
            (hand_center[0] - region_center[0])**2 +
            (hand_center[1] - region_center[1])**2
        )

        # 转换为激活强度
        max_distance = 0.2
        activation = max(0.0, 1.0 - (distance / max_distance))

        # 应用历史平滑
        if region_name in self.voice_regions:
            history = self.voice_regions[region_name]['activation_history']
        else:
            history = self.central_control_region['activation_history']

        history.append(activation)

        # 平滑处理
        smoothing_factor = self.config['position_smoothing']
        if len(history) > 1:
            smoothed_activation = (smoothing_factor * history[-2] +
                                 (1 - smoothing_factor) * activation)
        else:
            smoothed_activation = activation

        return smoothed_activation

    def calculate_hand_openness(self, landmarks: List[List[float]]) -> float:
        """计算手部张开程度"""
        if not landmarks or len(landmarks) < 21:
            return 0.0

        # 使用改进的算法计算张开程度
        finger_spread = self._calculate_finger_spread(landmarks)

        # 考虑手指伸展程度
        finger_extension = 0
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        pips = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]

        for tip, pip in zip(tips, pips):
            extension = abs(tip[1] - pip[1])  # Y轴差异
            finger_extension += extension

        finger_extension /= len(tips)

        # 综合计算
        openness = (finger_spread * 0.6 + finger_extension * 0.4)
        return min(1.0, openness * 3)  # 放大并限制在0-1范围

    def calculate_hand_center(self, landmarks: List[List[float]]) -> List[float]:
        """计算手部中心点"""
        if not landmarks or len(landmarks) < 21:
            return [0.5, 0.5]

        # 使用手掌关键点计算中心（更稳定）
        palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]

        x_coords = [lm[0] for lm in palm_points]
        y_coords = [lm[1] for lm in palm_points]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return [center_x, center_y]

    def update_gesture_duration(self, hand_id: str, gesture_type: str):
        """更新手势持续时间"""
        current_time = time.time()

        if hand_id not in self.gesture_duration_tracker:
            self.gesture_duration_tracker[hand_id] = {
                'current_gesture': gesture_type,
                'start_time': current_time,
                'duration': 0.0
            }
        else:
            tracker = self.gesture_duration_tracker[hand_id]
            if tracker['current_gesture'] == gesture_type:
                tracker['duration'] = current_time - tracker['start_time']
            else:
                # 手势类型改变，重置计时
                tracker['current_gesture'] = gesture_type
                tracker['start_time'] = current_time
                tracker['duration'] = 0.0

    def process_frame(self, frame: np.ndarray, show_regions: bool = True) -> np.ndarray:
        """处理视频帧并检测指挥手势"""
        with self._lock:
            if self.performance_monitor:
                self.performance_monitor.start_frame()

            process_start = time.time()

            self.frame_counter += 1

            # 帧跳跃策略
            should_process = self._should_process_mediapipe()

            if should_process:
                # 颜色空间转换优化
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe处理
                mp_start = time.time()
                results = self.hands.process(rgb_frame)
                mp_time = time.time() - mp_start

                self.last_mediapipe_result = results
                self.last_mediapipe_frame = self.frame_counter
            else:
                # 使用缓存结果
                results = self.last_mediapipe_result
                mp_time = 0

            # 重置检测数据
            self.gesture_data.update({
                'hands_detected': 0,
                'hands': [],
                'active_regions': {},
                'central_control_active': False,
                'region_activations': {}
            })

            # 初始化区域激活强度
            for region_name in self.voice_regions.keys():
                self.gesture_data['region_activations'][region_name] = 0.0

            current_time = time.time()

            if results and results.multi_hand_landmarks and results.multi_handedness:
                self.gesture_data['hands_detected'] = len(results.multi_hand_landmarks)

                for i, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):

                    # 转换关键点
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                    # 手部标识
                    hand_label = handedness.classification[0].label.lower()
                    hand_id = f"{hand_label}_{i}"

                    # 增强手势检测
                    gesture_type, gesture_confidence = self.detect_gesture_type_enhanced(
                        landmarks, hand_id)

                    # 更新手势持续时间
                    self.update_gesture_duration(hand_id, gesture_type)

                    # 计算手部属性
                    hand_center = self.calculate_hand_center(landmarks)
                    hand_openness = self.calculate_hand_openness(landmarks)

                    # 更新位置历史
                    self.hand_position_history[hand_id].append(tuple(hand_center))
                    self.hand_timestamp_history[hand_id].append(current_time)

                    # 检测区域
                    active_region = self.detect_hand_in_region(hand_center)
                    activation_strength = 0.0

                    if active_region:
                        activation_strength = self.calculate_region_activation_strength(
                            hand_center, active_region)

                        if active_region == 'central_control':
                            self.gesture_data['central_control_active'] = True
                        elif active_region in self.voice_regions:
                            self.gesture_data['active_regions'][active_region] = {
                                'hand_id': hand_id,
                                'activation_strength': activation_strength
                            }
                            self.gesture_data['region_activations'][active_region] = activation_strength

                    # 存储手部数据
                    hand_data = {
                        'id': hand_id,
                        'label': hand_label,
                        'detected': True,
                        'landmarks': landmarks,
                        'gesture_type': gesture_type,
                        'gesture_confidence': gesture_confidence,
                        'openness': hand_openness,
                        'center': hand_center,
                        'active_region': active_region,
                        'activation_strength': activation_strength,
                        'gesture_duration': self.gesture_duration_tracker.get(hand_id, {}).get('duration', 0.0)
                    }

                    self.gesture_data['hands'].append(hand_data)

                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 绘制手势信息
                    if hand_center and len(hand_center) >= 2:
                        height, width = frame.shape[:2]
                        center_x = int(hand_center[0] * width)
                        center_y = int(hand_center[1] * height)

                        # 显示手势类型和置信度
                        gesture_text = f"{gesture_type} ({gesture_confidence:.2f})"
                        cv2.putText(frame, gesture_text, (center_x + 10, center_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # 显示区域信息
                        if active_region:
                            region_text = (active_region.split('_')[0]
                                         if active_region != 'central_control' else 'Central')
                            cv2.putText(frame, f"Region: {region_text}",
                                       (center_x + 10, center_y + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        # 显示激活强度
                        if activation_strength > 0:
                            strength_text = f"Strength: {activation_strength:.2f}"
                            cv2.putText(frame, strength_text,
                                       (center_x + 10, center_y + 35),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 更新用户状态
            self.user_state_manager.update_presence(self.gesture_data['hands_detected'])
            self.gesture_data['user_presence_confidence'] = (
                self.user_state_manager.get_presence_confidence())

            # 分析指挥模式
            self._analyze_conducting_patterns()

            # 计算综合手势强度
            total_strength = sum(hand['openness'] for hand in self.gesture_data['hands'])
            self.gesture_data['gesture_strength'] = (
                total_strength / max(1, len(self.gesture_data['hands'])))

            # 绘制区域划分
            if show_regions:
                frame = self.draw_regions_overlay(frame)

            # 绘制性能信息
            if self.performance_monitor:
                self._draw_performance_overlay(frame)

            # 记录性能数据
            process_time = time.time() - process_start

            if self.performance_monitor:
                self.performance_monitor.record_process_time(process_time)
                self.performance_monitor.end_frame()
                self.gesture_data['performance_stats'] = self.performance_monitor.get_stats()

            # 清理过期缓存
            if self.gesture_cache:
                self.gesture_cache.clear_expired()

            return frame

    def _analyze_conducting_patterns(self):
        """分析指挥模式"""
        if not self.gesture_data['hands']:
            return

        # 分析主要指挥手（通常是右手）
        primary_hand = None
        for hand in self.gesture_data['hands']:
            if hand['label'] == 'right' or not primary_hand:
                primary_hand = hand
                break

        if not primary_hand:
            return

        hand_id = primary_hand['id']

        # 获取位置历史
        if (hand_id in self.hand_position_history and
            len(self.hand_position_history[hand_id]) >= 3):

            positions = list(self.hand_position_history[hand_id])
            timestamps = list(self.hand_timestamp_history[hand_id])

            # 分析指挥模式
            pattern_analysis = self.conductor_analyzer.analyze_conducting_pattern(
                positions, timestamps)
            self.gesture_data['conducting_pattern'] = pattern_analysis

            # 分析拍子模式
            beat_analysis = self.conductor_analyzer.detect_beat_pattern(
                self.hand_position_history[hand_id],
                self.hand_timestamp_history[hand_id])
            self.gesture_data['beat_analysis'] = beat_analysis

    def draw_regions_overlay(self, frame: np.ndarray) -> np.ndarray:
        """绘制区域边界和信息"""
        height, width = frame.shape[:2]

        # 绘制声部区域
        for region_name, region_data in self.voice_regions.items():
            bounds = region_data['bounds']
            color = region_data['color']

            # 转换坐标
            x1 = int(bounds['x1'] * width)
            y1 = int(bounds['y1'] * height)
            x2 = int(bounds['x2'] * width)
            y2 = int(bounds['y2'] * height)

            # 根据激活状态调整颜色强度
            activation = self.gesture_data['region_activations'].get(region_name, 0.0)
            alpha = 0.3 + activation * 0.7

            # 绘制填充矩形
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(frame, 1 - alpha * 0.3, overlay, alpha * 0.3, 0)

            # 绘制边界
            thickness = 3 if activation > 0.1 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 添加标签
            label = f"{region_data['id']}-{region_name.split('_')[0]}"
            if activation > 0.1:
                label += f" ({activation:.2f})"

            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制中央控制区域
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

    def _draw_performance_overlay(self, frame: np.ndarray):
        """绘制性能信息覆盖层"""
        if not self.performance_monitor:
            return

        stats = self.performance_monitor.get_stats()
        if not stats:
            return

        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 1

        # FPS信息
        fps_text = f"FPS: {stats.get('fps', 0):.1f}"
        cv2.putText(frame, fps_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25

        # 处理时间
        process_time = stats.get('avg_process_time', 0) * 1000
        process_text = f"Process: {process_time:.1f}ms"
        cv2.putText(frame, process_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25

        # 用户存在置信度
        presence = self.gesture_data.get('user_presence_confidence', 0)
        presence_text = f"User: {presence:.2f}"
        cv2.putText(frame, presence_text, (10, y_offset), font, font_scale, color, thickness)

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
            'user_presence': self.gesture_data.get('user_presence_confidence', 0.0),
            'gesture_stability': {}
        }

        primary_hand = None
        secondary_hand = None

        # 识别主要和次要指挥手
        for hand in self.gesture_data['hands']:
            if hand['label'] == 'right':
                primary_hand = hand
            elif hand['label'] == 'left':
                secondary_hand = hand

        # 处理主要指挥手命令
        if primary_hand:
            gesture_type = primary_hand['gesture_type']
            openness = primary_hand['openness']
            confidence = primary_hand['gesture_confidence']
            duration = primary_hand['gesture_duration']

            commands['gesture_stability'][primary_hand['id']] = {
                'confidence': confidence,
                'duration': duration
            }

            # 根据手势类型和持续时间确定命令
            if gesture_type == 'stop_fist' and duration > 0.5:
                commands['stop_command'] = True
                commands['volume_change'] = -1.0
            elif gesture_type == 'forte_open' and confidence > 0.8:
                commands['volume_change'] = openness * 1.2
                commands['expression_level'] = openness
                commands['attack_type'] = 'forte'
            elif gesture_type == 'baton_grip':
                # 专业指挥棒手势
                commands['expression_level'] = openness * 0.8
                commands['attack_type'] = 'precise'
            elif gesture_type == 'expressive_conducting':
                commands['expression_level'] = openness * confidence
                commands['tempo_change'] = (openness - 0.5) * 0.5

        # 处理双手协调
        if primary_hand and secondary_hand:
            # 计算双手距离和相对位置
            left_center = secondary_hand['center']
            right_center = primary_hand['center']

            hand_distance = math.sqrt(
                (left_center[0] - right_center[0])**2 +
                (left_center[1] - right_center[1])**2
            )

            # 双手张开度影响音量范围
            if hand_distance > 0.3:
                commands['volume_change'] *= 1.2
            elif hand_distance < 0.1:
                commands['volume_change'] *= 0.8

        # 整合指挥模式分析
        conducting_pattern = self.gesture_data.get('conducting_pattern', {})
        if conducting_pattern.get('confidence', 0) > 0.5:
            pattern_type = conducting_pattern.get('pattern_type', 'none')

            if pattern_type == 'forte_conducting':
                commands['attack_type'] = 'forte'
                commands['expression_level'] = max(commands['expression_level'], 0.8)
            elif pattern_type == 'legato_conducting':
                commands['attack_type'] = 'legato'
                commands['expression_level'] *= 0.7

        # 整合拍子分析
        beat_analysis = self.gesture_data.get('beat_analysis', {})
        if beat_analysis.get('confidence', 0) > 0.5:
            commands['tempo_bpm'] = beat_analysis.get('tempo_bpm', 120)
            commands['beats_per_measure'] = beat_analysis.get('beats_per_measure', 4)

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

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.performance_monitor:
            return self.performance_monitor.get_stats()
        return {}

    def update_config(self, new_config: Dict[str, Any]):
        """动态更新配置"""
        self.config.update(new_config)

        # 更新MediaPipe参数
        if any(key in new_config for key in ['max_num_hands', 'min_detection_confidence', 'min_tracking_confidence']):
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config['max_num_hands'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

    def get_gesture_data(self) -> Dict[str, Any]:
        """获取当前手势数据（实例方法）"""
        return self.gesture_data.copy()

    def get_region_info(self) -> Dict[str, Any]:
        """获取区域划分信息（实例方法）"""
        return {
            'voice_regions': self.voice_regions,
            'central_control_region': self.central_control_region
        }

    def cleanup(self):
        """清理资源"""
        # 停止摄像头
        self.stop_camera()

        # 清理MediaPipe
        if hasattr(self.hands, 'close'):
            self.hands.close()

# TouchDesigner接口函数（保持兼容性）
def get_gesture_data():
    """获取当前手势数据"""
    if hasattr(op, 'detector'):
        return op.detector.gesture_data
    return None

def get_conductor_commands():
    """获取指挥控制命令"""
    if hasattr(op, 'detector'):
        return op.detector.get_conductor_commands()
    return None

def get_active_voices():
    """获取激活的声部信息"""
    if hasattr(op, 'detector'):
        return op.detector.get_active_voices()
    return []

def get_region_info():
    """获取区域划分信息"""
    if hasattr(op, 'detector'):
        return {
            'voice_regions': op.detector.voice_regions,
            'central_control_region': op.detector.central_control_region
        }
    return None

def get_performance_stats():
    """获取性能统计信息"""
    if hasattr(op, 'detector'):
        return op.detector.get_performance_stats()
    return {}

def initialize_detector(config=None):
    """初始化增强版手势检测器"""
    if not hasattr(op, 'detector') or config:
        op.detector = HandGestureDetector(config)
    return True

def process_camera_frame(frame_data, show_regions=True):
    """处理摄像头帧数据"""
    if not hasattr(op, 'detector'):
        initialize_detector()

    processed_frame = op.detector.process_frame(frame_data, show_regions)
    return processed_frame

def set_region_bounds(region_name, x1, y1, x2, y2):
    """动态调整区域边界"""
    if hasattr(op, 'detector') and region_name in op.detector.voice_regions:
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
    if hasattr(op, 'detector'):
        op.detector.update_config(config_dict)
        return True
    return False

def cleanup_detector():
    """清理检测器资源"""
    if hasattr(op, 'detector'):
        op.detector.cleanup()
        delattr(op, 'detector')
        return True
    return False

# 新增的高级接口函数
def get_gesture_confidence(hand_id=None):
    """获取手势置信度"""
    if hasattr(op, 'detector') and op.detector.gesture_data['hands']:
        if hand_id:
            for hand in op.detector.gesture_data['hands']:
                if hand['id'] == hand_id:
                    return hand.get('gesture_confidence', 0.0)
        else:
            # 返回所有手的平均置信度
            confidences = [hand.get('gesture_confidence', 0.0)
                          for hand in op.detector.gesture_data['hands']]
            return sum(confidences) / len(confidences) if confidences else 0.0
    return 0.0

def get_user_presence_confidence():
    """获取用户存在置信度"""
    if hasattr(op, 'detector'):
        return op.detector.gesture_data.get('user_presence_confidence', 0.0)
    return 0.0

def get_conducting_analysis():
    """获取指挥分析结果"""
    if hasattr(op, 'detector'):
        return {
            'conducting_pattern': op.detector.gesture_data.get('conducting_pattern', {}),
            'beat_analysis': op.detector.gesture_data.get('beat_analysis', {}),
            'tempo_stability': op.detector.gesture_data.get('tempo_stability', 0.0)
        }
    return {}