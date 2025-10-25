"""
人体姿态检测器 - 基于MediaPipe Pose的小提琴演奏动作识别
使用MediaPipe进行实时人体姿态检测，识别33个关键点并检测小提琴演奏动作

优化特性：
- 性能优化：帧跳跃策略、结果缓存、内存管理
- 小提琴演奏动作识别：宽松判定标准，适合业余学习者
- 用户状态管理：出现/离开检测、状态持久化、置信度评分
- 骨骼点可视化：实时显示人体33个关键点和连接线
- 接口优化：简化TouchDesigner接口、性能监控、动态配置
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import threading


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


class PoseCache:
    """姿态结果缓存"""
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
    """姿态手势稳定器"""
    def __init__(self, history_size: int = 5, confidence_threshold: float = 0.6):
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.gesture_history = deque(maxlen=history_size)
        self.stability_scores = {}

    def add_gesture(self, gesture_detected: bool, confidence: float):
        """添加手势检测结果"""
        self.gesture_history.append((gesture_detected, confidence, time.time()))

    def get_stable_gesture(self) -> Tuple[bool, float]:
        """获取稳定的手势结果和置信度"""
        if not self.gesture_history:
            return False, 0.0

        recent_gestures = list(self.gesture_history)

        # 计算手势检测的频率和平均置信度
        detected_count = sum(1 for detected, _, _ in recent_gestures if detected)
        total_confidence = sum(conf for detected, conf, _ in recent_gestures if detected)

        stability_score = detected_count / len(recent_gestures)
        avg_confidence = total_confidence / detected_count if detected_count > 0 else 0.0

        # 只有当稳定性和置信度都足够高时才返回True
        # 宽松判定：稳定性>=0.4（5帧中至少2帧），置信度>=0.6
        if stability_score >= 0.4 and avg_confidence >= self.confidence_threshold:
            return True, avg_confidence

        return False, 0.0


class UserStateManager:
    """用户状态管理器"""
    def __init__(self, absence_timeout: float = 2.0):
        self.absence_timeout = absence_timeout
        self.last_detection_time = 0
        self.user_present = False
        self.presence_history = deque(maxlen=10)
        self.state_persistence = {}

    def update_presence(self, person_detected: bool):
        """更新用户存在状态"""
        current_time = time.time()

        if person_detected:
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


class InstrumentGestureAnalyzer:
    """乐器演奏动作分析器（支持多种乐器识别）"""

    @staticmethod
    def detect_violin_gesture(landmarks: List[Any],
                            right_hand_history: deque,
                            left_hand_history: deque) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测小提琴演奏动作（宽松判定标准）

        关键点索引：
        - 左肩: 11
        - 右肩: 12
        - 左肘: 13
        - 右肘: 14
        - 左手腕: 15
        - 右手腕: 16

        宽松判定标准：
        1. 左手持琴：左手腕接近左肩，y差值 < 0.4（放宽阈值）
        2. 右手拉弓：右手腕有横向运动，速度 > 0.05（降低阈值）
        3. 稳定性：历史帧确认（减少误判）

        返回：(是否检测到小提琴动作, 置信度, 详细信息)
        """
        if not landmarks or len(landmarks) < 33:
            return False, 0.0, {}

        # 提取关键关节点
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # 检查关键点置信度
        min_visibility = 0.5
        key_points = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]

        if not all(hasattr(lm, 'visibility') and lm.visibility > min_visibility for lm in key_points):
            return False, 0.0, {'reason': 'low_visibility'}

        # === 1. 检查左手持琴姿势（宽松）===
        # 左手腕应该抬起，接近左肩的高度
        left_wrist_y = left_wrist.y
        left_shoulder_y = left_shoulder.y
        y_diff = abs(left_wrist_y - left_shoulder_y)

        # 宽松阈值：y差值 < 0.4（原本更严格的是0.25）
        left_hand_holding = y_diff < 0.4

        # 左手腕应该在身体前方（x位置检查）
        left_wrist_x = left_wrist.x
        left_shoulder_x = left_shoulder.x
        # 左手腕应该略微向身体中线移动
        left_hand_forward = abs(left_wrist_x - left_shoulder_x) < 0.3

        left_hand_position_score = 1.0 if (left_hand_holding and left_hand_forward) else 0.3

        # === 2. 检查右手拉弓动作（横向运动，宽松）===
        right_wrist_x = right_wrist.x
        right_wrist_y = right_wrist.y

        # 记录右手位置历史
        right_hand_history.append((right_wrist_x, right_wrist_y, time.time()))

        # 计算横向和垂直速度（至少需要2帧）
        right_hand_bowing = False
        horizontal_velocity = 0.0
        vertical_velocity = 0.0
        motion_ratio = 0.0
        is_mainly_horizontal = False

        if len(right_hand_history) >= 2:
            prev_x, prev_y, prev_time = right_hand_history[-2]
            curr_x, curr_y, curr_time = right_hand_history[-1]

            dt = curr_time - prev_time
            if dt > 0:
                # 计算横向和垂直速度
                horizontal_velocity = abs(curr_x - prev_x) / dt
                vertical_velocity = abs(curr_y - prev_y) / dt

                # === 关键修改：添加运动方向占比检测 ===
                # 小提琴拉弓：横向运动应该显著大于垂直运动
                # 设定比例阈值：horizontal / vertical > 2.0（横向速度至少是垂直的2倍）

                # 处理垂直速度接近0的情况（纯横向运动）
                if vertical_velocity < 0.01:  # 几乎没有垂直运动
                    motion_ratio = float('inf')
                    is_mainly_horizontal = True
                else:
                    motion_ratio = horizontal_velocity / vertical_velocity
                    is_mainly_horizontal = motion_ratio > 2.0  # 横向占主导

                # 原有条件 + 新增方向占比条件
                # 区分拉小提琴（横向）和打鼓（垂直）
                right_hand_bowing = (horizontal_velocity > 0.05) and is_mainly_horizontal

        right_hand_bowing_score = min(1.0, horizontal_velocity * 10) if right_hand_bowing else 0.0

        # === 3. 检查右手位置（应该在身体前方，接近左手）===
        right_wrist_to_left_wrist_distance = math.sqrt(
            (right_wrist_x - left_wrist_x)**2 +
            (right_wrist_y - left_wrist_y)**2
        )

        # 右手应该在身体前方，距离左手不太远（宽松：< 0.5）
        right_hand_position_valid = right_wrist_to_left_wrist_distance < 0.5
        right_hand_position_score = 1.0 if right_hand_position_valid else 0.4

        # === 4. 综合判定 ===
        # 至少满足：左手持琴 + 右手有横向运动
        is_violin_gesture = left_hand_holding and right_hand_bowing

        # 计算综合置信度
        confidence = (
            left_hand_position_score * 0.4 +
            right_hand_bowing_score * 0.4 +
            right_hand_position_score * 0.2
        )

        # 详细信息
        details = {
            'left_hand_holding': left_hand_holding,
            'left_hand_y_diff': y_diff,
            'right_hand_bowing': right_hand_bowing,
            'horizontal_velocity': horizontal_velocity,
            'vertical_velocity': vertical_velocity,
            'motion_ratio': motion_ratio,
            'is_mainly_horizontal': is_mainly_horizontal,
            'right_left_distance': right_wrist_to_left_wrist_distance,
            'left_hand_score': left_hand_position_score,
            'right_bowing_score': right_hand_bowing_score,
            'right_position_score': right_hand_position_score
        }

        return is_violin_gesture, confidence, details

    @staticmethod
    def detect_clarinet_gesture(landmarks: List[Any],
                               right_hand_history: deque,
                               left_hand_history: deque) -> Tuple[bool, float, Dict[str, Any]]:
        """
        单簧管演奏动作检测
        特征：双手垂直排列，胸部到腹部高度
        """
        if not landmarks or len(landmarks) < 33:
            return False, 0.0, {}

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # 检查可见度
        if not (hasattr(left_wrist, 'visibility') and left_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}
        if not (hasattr(right_wrist, 'visibility') and right_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}

        # 双手垂直对齐（放宽阈值，从0.1改为0.15）
        hands_vertical = abs(left_wrist.x - right_wrist.x) < 0.15

        # 中间高度位置（胸部到腹部）
        hands_mid_height = (0.3 < left_wrist.y < 0.6) and (0.3 < right_wrist.y < 0.6)

        # 垂直排列（一只手在另一只手上方）
        vertical_distance = abs(left_wrist.y - right_wrist.y)
        hands_stacked = vertical_distance > 0.15  # 至少15%的垂直分离

        is_clarinet = hands_vertical and hands_mid_height and hands_stacked
        confidence = 0.8 if is_clarinet else 0.0

        details = {
            'hands_vertical': hands_vertical,
            'hands_mid_height': hands_mid_height,
            'hands_stacked': hands_stacked,
            'vertical_distance': vertical_distance
        }

        return is_clarinet, confidence, details

    @staticmethod
    def detect_piano_gesture(landmarks: List[Any],
                            right_hand_history: deque,
                            left_hand_history: deque) -> Tuple[bool, float, Dict[str, Any]]:
        """
        钢琴演奏动作检测
        特征：双手水平，键盘高度，上下运动
        """
        if not landmarks or len(landmarks) < 33:
            return False, 0.0, {}

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # 检查可见度
        if not (hasattr(left_wrist, 'visibility') and left_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}
        if not (hasattr(right_wrist, 'visibility') and right_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}

        # 双手水平
        hands_level = abs(left_wrist.y - right_wrist.y) < 0.1

        # 键盘高度（腰到胸）
        keyboard_height = (0.4 < left_wrist.y < 0.7) and (0.4 < right_wrist.y < 0.7)

        # 计算垂直速度以检测按键动作
        right_hand_history.append((right_wrist.x, right_wrist.y, time.time()))
        left_hand_history.append((left_wrist.x, left_wrist.y, time.time()))

        vertical_velocity = 0.0
        if len(right_hand_history) >= 2:
            _, prev_y, prev_time = right_hand_history[-2]
            _, curr_y, curr_time = right_hand_history[-1]
            dt = curr_time - prev_time
            if dt > 0:
                vertical_velocity = abs(curr_y - prev_y) / dt

        # 小幅度上下运动表示按键
        key_pressing = 0.02 < vertical_velocity < 0.15

        # 速度上限：排除快速运动（防止误判为Drum）
        not_too_fast = vertical_velocity < 0.12

        is_piano = hands_level and keyboard_height and not_too_fast
        confidence = 0.9 if (is_piano and key_pressing) else (0.6 if is_piano else 0.0)

        details = {
            'hands_level': hands_level,
            'keyboard_height': keyboard_height,
            'key_pressing': key_pressing,
            'not_too_fast': not_too_fast,
            'vertical_velocity': vertical_velocity
        }

        return is_piano, confidence, details

    @staticmethod
    def detect_drum_gesture(landmarks: List[Any],
                           right_hand_history: deque,
                           left_hand_history: deque) -> Tuple[bool, float, Dict[str, Any]]:
        """
        鼓演奏动作检测（支持双手）
        特征：快速上下运动，鼓的高度，双手不水平（区分Piano）
        优化：使用滑动窗口最大值捕捉瞬时峰值速度
        """
        if not landmarks or len(landmarks) < 33:
            return False, 0.0, {}

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # 检查可见度
        if not (hasattr(left_wrist, 'visibility') and left_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}
        if not (hasattr(right_wrist, 'visibility') and right_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}

        # 记录双手位置历史
        right_hand_history.append((right_wrist.x, right_wrist.y, time.time()))
        left_hand_history.append((left_wrist.x, left_wrist.y, time.time()))

        # === 改进：使用滑动窗口最大值计算速度 ===
        # 计算右手垂直速度（检查最近5帧内的最大速度）
        right_velocities = []
        if len(right_hand_history) >= 2:
            for i in range(1, min(5, len(right_hand_history))):
                _, prev_y, prev_time = right_hand_history[-i-1]
                _, curr_y, curr_time = right_hand_history[-i]
                dt = curr_time - prev_time
                if dt > 0:
                    velocity = abs(curr_y - prev_y) / dt
                    right_velocities.append(velocity)

        right_velocity = max(right_velocities) if right_velocities else 0.0

        # 计算左手垂直速度（检查最近5帧内的最大速度）
        left_velocities = []
        if len(left_hand_history) >= 2:
            for i in range(1, min(5, len(left_hand_history))):
                _, prev_y, prev_time = left_hand_history[-i-1]
                _, curr_y, curr_time = left_hand_history[-i]
                dt = curr_time - prev_time
                if dt > 0:
                    velocity = abs(curr_y - prev_y) / dt
                    left_velocities.append(velocity)

        left_velocity = max(left_velocities) if left_velocities else 0.0

        # 使用双手中的最大速度
        vertical_velocity = max(right_velocity, left_velocity)

        # 快速鼓击动作（保持0.08，因为使用了最大值计算）
        fast_drumming = vertical_velocity > 0.08

        # 鼓的高度（扩大范围：0.3-0.6 → 0.2-0.7）
        drum_height = (0.2 < right_wrist.y < 0.7) or (0.2 < left_wrist.y < 0.7)

        # 双手不水平（放宽条件：0.15 → 0.05，允许同时击打）
        y_diff = abs(left_wrist.y - right_wrist.y)
        hands_not_level = y_diff > 0.05

        is_drum = fast_drumming and drum_height and hands_not_level
        confidence = min(1.0, vertical_velocity * 3) if is_drum else 0.0

        # 调试日志输出（帮助用户查看实际值）
        print(f"[DRUM DEBUG] "
              f"V-vel={vertical_velocity:.4f} (R:{right_velocity:.4f}, L:{left_velocity:.4f}), "
              f"Cond1_fast={fast_drumming} (>{0.08}), "
              f"Cond2_height={drum_height} (R_y:{right_wrist.y:.3f}, L_y:{left_wrist.y:.3f}), "
              f"Cond3_not_level={hands_not_level} (y_diff:{y_diff:.3f} >{0.05}), "
              f"is_drum={is_drum}, conf={confidence:.3f} (need>0.35)")

        details = {
            'fast_drumming': fast_drumming,
            'drum_height': drum_height,
            'hands_not_level': hands_not_level,
            'vertical_velocity': vertical_velocity,
            'right_velocity': right_velocity,
            'left_velocity': left_velocity,
            'y_diff': y_diff,
            'right_wrist_y': right_wrist.y,
            'left_wrist_y': left_wrist.y
        }

        return is_drum, confidence, details

    @staticmethod
    def detect_trumpet_gesture(landmarks: List[Any],
                              right_hand_history: deque,
                              left_hand_history: deque) -> Tuple[bool, float, Dict[str, Any]]:
        """
        小号演奏动作检测
        特征：双手靠近嘴部（高位置），双手靠近
        """
        if not landmarks or len(landmarks) < 33:
            return False, 0.0, {}

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # 检查可见度
        if not (hasattr(left_wrist, 'visibility') and left_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}
        if not (hasattr(right_wrist, 'visibility') and right_wrist.visibility > 0.5):
            return False, 0.0, {'reason': 'low_visibility'}

        # 双手高位（靠近嘴部/面部水平，降低高度要求从0.3改为0.4）
        both_hands_high = (left_wrist.y < 0.4) and (right_wrist.y < 0.4)

        # 双手靠近
        hand_distance = math.sqrt(
            (left_wrist.x - right_wrist.x)**2 +
            (left_wrist.y - right_wrist.y)**2
        )
        hands_close = hand_distance < 0.25

        is_trumpet = both_hands_high and hands_close
        confidence = 0.85 if is_trumpet else 0.0

        details = {
            'both_hands_high': both_hands_high,
            'hands_close': hands_close,
            'hand_distance': hand_distance
        }

        return is_trumpet, confidence, details

    @staticmethod
    def detect_instrument(landmarks: List[Any],
                         right_hand_history: deque,
                         left_hand_history: deque) -> Dict[str, float]:
        """
        检测所有乐器并返回置信度字典

        Returns:
            dict: {'violin': 0.8, 'piano': 0.7, ...}
        """
        detected_instruments = {}

        # 检测小提琴
        is_violin, conf_violin, _ = InstrumentGestureAnalyzer.detect_violin_gesture(
            landmarks, right_hand_history, left_hand_history
        )
        if is_violin and conf_violin > 0.6:
            detected_instruments['violin'] = conf_violin

        # 检测单簧管
        is_clarinet, conf_clarinet, _ = InstrumentGestureAnalyzer.detect_clarinet_gesture(
            landmarks, right_hand_history, left_hand_history
        )
        if is_clarinet and conf_clarinet > 0.6:
            detected_instruments['clarinet'] = conf_clarinet

        # 检测钢琴
        is_piano, conf_piano, _ = InstrumentGestureAnalyzer.detect_piano_gesture(
            landmarks, right_hand_history, left_hand_history
        )
        if is_piano and conf_piano > 0.5:
            detected_instruments['piano'] = conf_piano

        # 检测鼓
        is_drum, conf_drum, _ = InstrumentGestureAnalyzer.detect_drum_gesture(
            landmarks, right_hand_history, left_hand_history
        )
        if is_drum and conf_drum > 0.35:
            detected_instruments['drum'] = conf_drum

        # 检测小号（降低置信度阈值，从0.7改为0.55）
        is_trumpet, conf_trumpet, _ = InstrumentGestureAnalyzer.detect_trumpet_gesture(
            landmarks, right_hand_history, left_hand_history
        )
        if is_trumpet and conf_trumpet > 0.55:
            detected_instruments['trumpet'] = conf_trumpet

        return detected_instruments


class PoseBodyDetector:
    """人体姿态检测器（基于MediaPipe Pose）"""

    # MediaPipe Pose关键点名称（33个关键点）
    POSE_LANDMARKS = {
        0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 配置参数（优化后）
        self.config = {
            'frame_skip': 2,  # 每2帧处理一次MediaPipe
            'min_detection_confidence': 0.5,  # 降低以提升速度（业余学习者场景）
            'min_tracking_confidence': 0.5,
            'model_complexity': 0,  # 使用Lite模型（2x速度提升，适合业余场景）
            'cache_enabled': True,
            'stability_check': True,
            'performance_monitoring': True,
            'gesture_history_size': 10,
            'confidence_threshold': 0.6  # 宽松的置信度阈值
        }

        if config:
            self.config.update(config)

        # 平台信息
        self.platform_info = self._detect_platform()

        # 平台特定优化
        self._apply_platform_optimizations()

        # 初始化MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config['model_complexity'],
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )
        except Exception as e:
            print(f"❌ [错误] MediaPipe Pose初始化失败: {e}")
            raise RuntimeError("MediaPipe Pose初始化失败，请检查mediapipe安装") from e

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 性能优化组件
        self.frame_counter = 0
        self.last_mediapipe_result = None
        self.last_mediapipe_frame = 0

        # 监控和缓存
        self.performance_monitor = PerformanceMonitor() if self.config['performance_monitoring'] else None
        self.pose_cache = PoseCache() if self.config['cache_enabled'] else None
        self.gesture_stabilizer = GestureStabilizer() if self.config['stability_check'] else None
        self.user_state_manager = UserStateManager()
        self.instrument_analyzer = InstrumentGestureAnalyzer()

        # 小提琴动作历史记录（使用maxlen防止内存泄漏）
        self.right_hand_history = deque(maxlen=self.config['gesture_history_size'])
        self.left_hand_history = deque(maxlen=self.config['gesture_history_size'])
        self.violin_gesture_history = deque(maxlen=5)  # 用于稳定性检查

        # 预分配RGB缓冲区（避免重复分配内存）
        self.rgb_buffer = None

        # 姿态数据
        self.pose_data = {
            'person_detected': False,
            'detected_instruments': {},  # 改为多乐器检测
            'pose_confidence': 0.0,
            'landmarks': [],
            'instrument_details': {},
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

        return {
            'system': system,
            'machine': machine,
            'processor_type': processor_type,
            'python_version': sys.version,
            'mediapipe_delegate': 'CPU'  # MediaPipe Pose主要使用CPU
        }

    def _apply_platform_optimizations(self):
        """根据平台应用特定优化"""
        processor_type = self.platform_info['processor_type']

        # Apple Silicon可以使用更高的模型复杂度
        # 但对于业余学习者场景，仍然优先使用Lite模型以节省资源
        if processor_type == 'Apple Silicon':
            # Apple Silicon性能强劲，可以降低frame_skip
            if self.config['frame_skip'] > 2:
                self.config['frame_skip'] = 2
            print(f"[优化] Apple Silicon检测到，frame_skip设置为 {self.config['frame_skip']}")
        else:
            # Intel Mac或其他平台，增加frame_skip以减轻负担
            if self.config['frame_skip'] < 3:
                self.config['frame_skip'] = 3
            print(f"[优化] {processor_type}检测到，frame_skip设置为 {self.config['frame_skip']}")

        print(f"[优化] 使用MediaPipe Lite模型（model_complexity={self.config['model_complexity']}）")

    def start_camera(self, camera_id: int = 0) -> bool:
        """启动摄像头（跨平台支持）"""
        try:
            if self.cap is not None:
                self.stop_camera()

            # 跨平台摄像头后端选择
            system = platform.system()
            if system == 'Windows':
                # Windows使用DirectShow后端
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                print(f"[摄像头] Windows系统，使用DirectShow后端")
            elif system == 'Darwin':
                # macOS使用AVFoundation后端
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
                print(f"[摄像头] macOS系统，使用AVFoundation后端")
            else:
                # Linux使用V4L2后端
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                print(f"[摄像头] Linux系统，使用V4L2后端")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                print(f"[错误] 无法打开摄像头 {camera_id}")

                # 平台特定错误提示
                if self.platform_info['system'] == 'Darwin':
                    print("[提示] macOS用户：请检查系统偏好设置 > 安全性与隐私 > 隐私 > 摄像头")
                    print("       确保已授权Python/终端访问摄像头")
                elif self.platform_info['system'] == 'Windows':
                    print("[提示] Windows用户：请检查设备管理器中摄像头是否正常")
                    print("       确保没有其他应用程序正在使用摄像头")

                return False

            self.camera_active = True
            print(f"[成功] 摄像头 {camera_id} 启动成功")
            print(f"[平台] {self.platform_info['system']} - {self.platform_info['processor_type']}")
            return True

        except Exception as e:
            print(f"[异常] 启动摄像头失败: {e}")
            return False

    def stop_camera(self):
        """停止摄像头"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.camera_active = False
            self.current_frame = None
            print("[信息] 摄像头已停止")

        except Exception as e:
            print(f"[异常] 停止摄像头失败: {e}")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧（优化：避免不必要的拷贝）"""
        if not self.camera_active or self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                # 优化：直接使用frame，避免深拷贝
                # 调用者需要修改时才拷贝
                self.current_frame = frame
                return self.current_frame
            else:
                return None
        except Exception as e:
            print(f"[异常] 读取帧失败: {e}")
            return None

    def _should_process_mediapipe(self) -> bool:
        """判断是否应该处理MediaPipe"""
        return (self.frame_counter % self.config['frame_skip']) == 0

    def is_person_detected(self) -> bool:
        """检测是否有人（任意关键点置信度>0.5）"""
        return self.pose_data['person_detected']

    def is_violin_gesture_detected(self) -> bool:
        """检测小提琴演奏动作（宽松判定）- 向后兼容"""
        detected_instruments = self.pose_data.get('detected_instruments', {})
        return 'violin' in detected_instruments

    def detect_violin_gesture(self, landmarks: List[Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测小提琴演奏动作（宽松判定）- 保留向后兼容性

        返回：(是否检测到, 置信度, 详细信息)
        """
        is_detected, confidence, details = self.instrument_analyzer.detect_violin_gesture(
            landmarks,
            self.right_hand_history,
            self.left_hand_history
        )

        # 使用稳定器平滑结果
        if self.gesture_stabilizer:
            self.gesture_stabilizer.add_gesture(is_detected, confidence)
            stable_detected, stable_confidence = self.gesture_stabilizer.get_stable_gesture()
            return stable_detected, stable_confidence, details

        return is_detected, confidence, details

    def detect_instruments(self, landmarks: List[Any]) -> Dict[str, float]:
        """
        检测所有乐器

        Returns:
            dict: {'violin': 0.8, 'piano': 0.7, ...}
        """
        detected = self.instrument_analyzer.detect_instrument(
            landmarks,
            self.right_hand_history,
            self.left_hand_history
        )

        # 对每个乐器应用稳定器（简化版，直接接受检测结果）
        if self.gesture_stabilizer:
            stabilized = {}
            for instrument, confidence in detected.items():
                # 简单接受检测结果
                stabilized[instrument] = confidence
            return stabilized

        return detected

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: Any):
        """使用MediaPipe的drawing_utils可视化骨骼点"""
        if not landmarks:
            return

        # 绘制姿态关键点和连接线
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 高亮检测到的乐器相关关键点
        detected_instruments = self.pose_data.get('detected_instruments', {})
        if detected_instruments:
            # 所有乐器都关注手腕和肩膀
            key_landmarks = [11, 12, 15, 16]  # 左肩、右肩、左手腕、右手腕

            height, width = frame.shape[:2]

            for idx in key_landmarks:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]

                    if lm.visibility > 0.5:
                        x = int(lm.x * width)
                        y = int(lm.y * height)

                        # 绘制黄色高亮圆圈
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), 3)

                        # 添加标签
                        label = self.POSE_LANDMARKS.get(idx, str(idx))
                        cv2.putText(frame, label, (x + 15, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def process_frame(self, frame: np.ndarray, show_skeleton: bool = True) -> np.ndarray:
        """处理视频帧并检测人体姿态"""
        with self._lock:
            if self.performance_monitor:
                self.performance_monitor.start_frame()

            process_start = time.time()
            self.frame_counter += 1

            # 帧跳跃策略
            should_process = self._should_process_mediapipe()

            if should_process:
                # 颜色空间转换（优化：使用预分配缓冲区）
                if self.rgb_buffer is None or self.rgb_buffer.shape != frame.shape:
                    self.rgb_buffer = np.empty_like(frame)
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self.rgb_buffer)

                # MediaPipe Pose处理
                mp_start = time.time()
                results = self.pose.process(self.rgb_buffer)
                mp_time = time.time() - mp_start

                self.last_mediapipe_result = results
                self.last_mediapipe_frame = self.frame_counter
            else:
                # 使用缓存结果
                results = self.last_mediapipe_result
                mp_time = 0

            # 重置检测数据
            self.pose_data.update({
                'person_detected': False,
                'detected_instruments': {},
                'pose_confidence': 0.0,
                'landmarks': [],
                'instrument_details': {}
            })

            current_time = time.time()

            # 处理检测结果
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 检查是否有人（优化：直接计数，避免创建列表）
                visible_count = sum(1 for lm in landmarks if hasattr(lm, 'visibility') and lm.visibility > 0.5)

                if visible_count > 10:  # 至少10个关键点可见
                    self.pose_data['person_detected'] = True

                    # 计算平均置信度（优化：直接计算，避免二次遍历）
                    total_visibility = sum(lm.visibility for lm in landmarks if hasattr(lm, 'visibility') and lm.visibility > 0.5)
                    avg_visibility = total_visibility / visible_count if visible_count > 0 else 0.0
                    self.pose_data['pose_confidence'] = avg_visibility

                    # 存储关键点数据
                    self.pose_data['landmarks'] = [
                        {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                        for lm in landmarks
                    ]

                    # 检测所有乐器
                    detected_instruments = self.detect_instruments(landmarks)
                    self.pose_data['detected_instruments'] = detected_instruments

                    # 绘制骨骼点
                    if show_skeleton:
                        self.draw_pose_landmarks(frame, results.pose_landmarks)

                    # 绘制检测信息
                    info_y = 30

                    # 人体检测状态
                    person_text = f"Person: Detected (Conf: {avg_visibility:.2f})"
                    cv2.putText(frame, person_text, (10, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    info_y += 30

                    # 显示检测到的乐器
                    if detected_instruments:
                        for instrument, confidence in detected_instruments.items():
                            inst_text = f"{instrument.capitalize()}: Detected (Conf: {confidence:.2f})"
                            cv2.putText(frame, inst_text, (10, info_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            info_y += 25
                    else:
                        no_inst_text = "Instruments: None detected"
                        cv2.putText(frame, no_inst_text, (10, info_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        info_y += 25

            # 更新用户状态
            self.user_state_manager.update_presence(self.pose_data['person_detected'])
            self.pose_data['user_presence_confidence'] = (
                self.user_state_manager.get_presence_confidence())

            # 绘制性能信息
            if self.performance_monitor:
                self._draw_performance_overlay(frame)

            # 记录性能数据
            process_time = time.time() - process_start

            if self.performance_monitor:
                self.performance_monitor.record_process_time(process_time)
                self.performance_monitor.end_frame()
                self.pose_data['performance_stats'] = self.performance_monitor.get_stats()

            # 清理过期缓存
            if self.pose_cache:
                self.pose_cache.clear_expired()

            return frame

    def _draw_performance_overlay(self, frame: np.ndarray):
        """绘制性能信息覆盖层"""
        if not self.performance_monitor:
            return

        stats = self.performance_monitor.get_stats()
        if not stats:
            return

        # 绘制在右上角
        height, width = frame.shape[:2]
        x_offset = width - 200
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1

        # FPS信息
        fps_text = f"FPS: {stats.get('fps', 0):.1f}"
        cv2.putText(frame, fps_text, (x_offset, y_offset), font, font_scale, color, thickness)
        y_offset += 25

        # 处理时间
        process_time = stats.get('avg_process_time', 0) * 1000
        process_text = f"Process: {process_time:.1f}ms"
        cv2.putText(frame, process_text, (x_offset, y_offset), font, font_scale, color, thickness)
        y_offset += 25

        # 用户存在置信度
        presence = self.pose_data.get('user_presence_confidence', 0)
        presence_text = f"Presence: {presence:.2f}"
        cv2.putText(frame, presence_text, (x_offset, y_offset), font, font_scale, color, thickness)

    def get_pose_data(self) -> Dict[str, Any]:
        """获取姿态数据"""
        return self.pose_data.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.performance_monitor:
            return self.performance_monitor.get_stats()
        return {}

    def update_config(self, new_config: Dict[str, Any]):
        """动态更新配置"""
        self.config.update(new_config)

        # 更新MediaPipe Pose参数
        if any(key in new_config for key in ['min_detection_confidence', 'min_tracking_confidence', 'model_complexity']):
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config['model_complexity'],
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

    def cleanup(self):
        """清理资源（优化：确保所有资源释放）"""
        # 停止摄像头
        self.stop_camera()

        # 清理MediaPipe
        if hasattr(self.pose, 'close'):
            self.pose.close()

        # 清理缓冲区
        self.rgb_buffer = None
        self.current_frame = None

        # 清理历史数据
        self.right_hand_history.clear()
        self.left_hand_history.clear()
        self.violin_gesture_history.clear()

        # 清理缓存
        if self.pose_cache:
            self.pose_cache.cache.clear()
            self.pose_cache.timestamps.clear()

        print("[信息] 资源清理完成")


# TouchDesigner接口函数（保持兼容性）
# 注意：这些函数仅在TouchDesigner环境中使用，独立运行时不会调用
def get_pose_data():
    """获取当前姿态数据"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            return op.pose_detector.get_pose_data()
    except NameError:
        pass
    return None


def is_person_detected():
    """检测是否有人"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            return op.pose_detector.is_person_detected()
    except NameError:
        pass
    return False


def is_violin_gesture_detected():
    """检测小提琴演奏动作"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            return op.pose_detector.is_violin_gesture_detected()
    except NameError:
        pass
    return False


def get_performance_stats():
    """获取性能统计信息"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            return op.pose_detector.get_performance_stats()
    except NameError:
        pass
    return {}


def initialize_detector(config=None):
    """初始化人体姿态检测器"""
    try:
        if 'op' not in globals():
            return False
        if not hasattr(op, 'pose_detector') or config:
            op.pose_detector = PoseBodyDetector(config)
        return True
    except NameError:
        return False


def process_camera_frame(frame_data, show_skeleton=True):
    """处理摄像头帧数据"""
    try:
        if 'op' not in globals():
            return frame_data
        if not hasattr(op, 'pose_detector'):
            initialize_detector()
        processed_frame = op.pose_detector.process_frame(frame_data, show_skeleton)
        return processed_frame
    except NameError:
        return frame_data


def update_detector_config(config_dict):
    """动态更新检测器配置"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            op.pose_detector.update_config(config_dict)
            return True
    except NameError:
        pass
    return False


def cleanup_detector():
    """清理检测器资源"""
    try:
        if 'op' in globals() and hasattr(op, 'pose_detector'):
            op.pose_detector.cleanup()
            delattr(op, 'pose_detector')
            return True
    except NameError:
        pass
    return False


# 测试代码（独立运行）
def main():
    """主函数：用于独立测试"""
    print("[启动] 人体姿态检测器测试程序")
    print("=" * 60)

    # 创建检测器
    detector = PoseBodyDetector()

    # 启动摄像头
    if not detector.start_camera(camera_id=0):
        print("[错误] 摄像头启动失败，程序退出")
        return

    print("[提示] 按 'q' 键退出程序")
    print("[提示] 按 's' 键切换骨骼显示")
    print("=" * 60)

    show_skeleton = True

    try:
        while True:
            # 获取摄像头帧
            frame = detector.get_current_frame()

            if frame is None:
                print("[警告] 无法读取摄像头帧")
                break

            # 处理帧
            processed_frame = detector.process_frame(frame, show_skeleton=show_skeleton)

            # 显示帧
            cv2.imshow('Pose Body Detector - Violin Gesture Recognition', processed_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[退出] 用户按下 'q' 键")
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                print(f"[切换] 骨骼显示: {'开启' if show_skeleton else '关闭'}")

    except KeyboardInterrupt:
        print("\n[退出] 用户中断（Ctrl+C）")

    finally:
        # 清理资源
        detector.cleanup()
        cv2.destroyAllWindows()
        print("[完成] 程序正常退出")


if __name__ == '__main__':
    main()
