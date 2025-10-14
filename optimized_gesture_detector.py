#!/usr/bin/env python3
"""
Optimized Hand Gesture Detector
优化版手势检测器 - 针对实时性能优化

主要优化策略：
1. 帧跳跃处理（15FPS检测 + 30FPS显示）
2. MediaPipe参数调优
3. 多线程处理分离
4. 内存池管理
5. 缓存策略

性能目标：
- 手势检测：15FPS
- 显示渲染：30FPS
- CPU使用：<60%
- 内存使用：<500MB
- 响应延迟：<50ms

Author: Performance Engineer
Date: 2025-10-05
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import queue
from concurrent.futures import ThreadPoolExecutor
import gc


@dataclass
class OptimizedGestureData:
    """优化的手势数据结构"""
    timestamp: float
    hands_detected: int
    hands: List[Dict[str, Any]]
    active_regions: Dict[str, Any]
    central_control_active: bool
    gesture_type: str
    hand_openness: float
    hand_center: List[float]
    gesture_strength: float
    region_activations: Dict[str, float]
    processing_time_ms: float
    confidence_score: float


class FrameBuffer:
    """帧缓冲器 - 管理视频帧的高效处理"""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.frames: deque = deque(maxlen=max_size)
        self.processed_frames: deque = deque(maxlen=max_size)
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
                # 帧跳跃策略：每2帧处理一次
                if len(self.frames) >= 2:
                    frame_data = self.frames[-2]  # 使用前一帧
                    return frame_data
        return None


class GestureCache:
    """手势缓存 - 减少重复计算"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}

    def _generate_key(self, landmarks: List[List[float]]) -> str:
        """生成缓存键"""
        # 简化的哈希策略
        if not landmarks:
            return "empty"

        # 使用前5个关键点的位置生成键
        key_points = landmarks[:5] if len(landmarks) >= 5 else landmarks
        hash_input = ""
        for point in key_points:
            hash_input += f"{point[0]:.2f},{point[1]:.2f},"
        return hash_input

    def get(self, landmarks: List[List[float]]) -> Optional[Dict[str, Any]]:
        """获取缓存的手势数据"""
        key = self._generate_key(landmarks)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, landmarks: List[List[float]], gesture_data: Dict[str, Any]):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            self._cleanup_old_entries()

        key = self._generate_key(landmarks)
        self.cache[key] = gesture_data
        self.access_times[key] = time.time()

    def _cleanup_old_entries(self):
        """清理旧缓存条目"""
        current_time = time.time()
        expired_keys = [
            k for k, t in self.access_times.items()
            if current_time - t > 10.0  # 10秒过期
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)


class OptimizedHandGestureDetector:
    """优化版手势检测器"""

    def __init__(self, detection_fps: int = 15, display_fps: int = 30):
        """
        初始化优化的手势检测器

        Args:
            detection_fps: 手势检测帧率
            display_fps: 显示帧率
        """
        self.detection_fps = detection_fps
        self.display_fps = display_fps

        # MediaPipe配置优化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 减少到2只手以提升性能
            min_detection_confidence=0.5,  # 降低检测阈值以提升速度
            min_tracking_confidence=0.3,   # 降低跟踪阈值
            model_complexity=0  # 使用简化模型
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 缓存和缓冲器
        self.frame_buffer = FrameBuffer(max_size=5)
        self.gesture_cache = GestureCache(max_size=50)

        # 区域定义（与原版相同但缓存）
        self._init_regions()

        # 处理状态
        self.last_detection_time = 0.0
        self.detection_interval = 1.0 / detection_fps
        self.last_gesture_data: Optional[OptimizedGestureData] = None

        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gesture")
        self.processing_queue: queue.Queue = queue.Queue(maxsize=3)

        # 性能监控
        self.performance_stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time_ms': 0.0,
            'memory_usage_mb': 0.0
        }

        # 启动后台处理线程
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _init_regions(self):
        """初始化区域定义"""
        self.voice_regions = {
            'Tromba_I+II+III_in_D': {
                'id': 1,
                'bounds': {'x1': 0.0, 'y1': 0.0, 'x2': 0.33, 'y2': 0.4},
                'center': [0.165, 0.2],
                'color': (255, 100, 100)
            },
            'Violins_in_D': {
                'id': 2,
                'bounds': {'x1': 0.33, 'y1': 0.0, 'x2': 0.67, 'y2': 0.4},
                'center': [0.5, 0.2],
                'color': (100, 255, 100)
            },
            'Viola_in_D': {
                'id': 3,
                'bounds': {'x1': 0.67, 'y1': 0.0, 'x2': 1.0, 'y2': 0.4},
                'center': [0.835, 0.2],
                'color': (100, 100, 255)
            },
            'Oboe_I_in_D': {
                'id': 4,
                'bounds': {'x1': 0.0, 'y1': 0.4, 'x2': 0.33, 'y2': 0.8},
                'center': [0.165, 0.6],
                'color': (255, 255, 100)
            },
            'Continuo_in_D': {
                'id': 5,
                'bounds': {'x1': 0.67, 'y1': 0.4, 'x2': 1.0, 'y2': 0.8},
                'center': [0.835, 0.6],
                'color': (255, 100, 255)
            },
            'Organo_obligato_in_D': {
                'id': 6,
                'bounds': {'x1': 0.0, 'y1': 0.8, 'x2': 0.5, 'y2': 1.0},
                'center': [0.25, 0.9],
                'color': (100, 255, 255)
            },
            'Timpani_in_D': {
                'id': 7,
                'bounds': {'x1': 0.5, 'y1': 0.8, 'x2': 1.0, 'y2': 1.0},
                'center': [0.75, 0.9],
                'color': (200, 150, 100)
            }
        }

        self.central_control_region = {
            'bounds': {'x1': 0.33, 'y1': 0.4, 'x2': 0.67, 'y2': 0.8},
            'center': [0.5, 0.6],
            'color': (150, 150, 150)
        }

    def _processing_loop(self):
        """后台处理循环"""
        while self.is_running:
            try:
                # 获取处理任务
                frame_data = self.processing_queue.get(timeout=0.1)
                if frame_data is None:  # 停止信号
                    break

                # 执行手势检测
                self._process_frame_async(frame_data)
                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing loop error: {e}")

    def _process_frame_async(self, frame_data: Dict[str, Any]):
        """异步处理帧"""
        try:
            start_time = time.time()
            frame = frame_data['frame']

            # 预处理优化
            frame_small = self._preprocess_frame(frame)

            # MediaPipe处理
            rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # 提取手势数据
            gesture_data = self._extract_gesture_data(results, frame_small.shape)

            # 更新性能统计
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['avg_processing_time_ms'] = (
                self.performance_stats['avg_processing_time_ms'] * 0.9 + processing_time * 0.1
            )
            self.performance_stats['frames_processed'] += 1

            # 存储结果
            self.last_gesture_data = OptimizedGestureData(
                timestamp=frame_data['timestamp'],
                hands_detected=gesture_data['hands_detected'],
                hands=gesture_data['hands'],
                active_regions=gesture_data['active_regions'],
                central_control_active=gesture_data['central_control_active'],
                gesture_type=gesture_data['gesture_type'],
                hand_openness=gesture_data['hand_openness'],
                hand_center=gesture_data['hand_center'],
                gesture_strength=gesture_data['gesture_strength'],
                region_activations=gesture_data['region_activations'],
                processing_time_ms=processing_time,
                confidence_score=gesture_data.get('confidence_score', 0.0)
            )

        except Exception as e:
            print(f"Frame processing error: {e}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧以提升性能"""
        # 缩放到较小尺寸以提升处理速度
        height, width = frame.shape[:2]
        target_width = 320
        target_height = int(height * target_width / width)

        # 使用快速插值
        frame_small = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # 可选：应用轻度高斯模糊减少噪声
        if frame_small.shape[0] > 240:  # 只在较大图像上应用
            frame_small = cv2.GaussianBlur(frame_small, (3, 3), 0)

        return frame_small

    def _extract_gesture_data(self, results, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """提取手势数据"""
        gesture_data = {
            'hands_detected': 0,
            'hands': [],
            'active_regions': {},
            'central_control_active': False,
            'gesture_type': 'none',
            'hand_openness': 0.0,
            'hand_center': [0.5, 0.5],
            'gesture_strength': 0.0,
            'region_activations': {},
            'confidence_score': 0.0
        }

        # 初始化区域激活
        for region_name in self.voice_regions.keys():
            gesture_data['region_activations'][region_name] = 0.0

        if not results.multi_hand_landmarks:
            return gesture_data

        gesture_data['hands_detected'] = len(results.multi_hand_landmarks)
        total_confidence = 0.0

        for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # 提取关键点
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # 检查缓存
            cached_result = self.gesture_cache.get(landmarks)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                gesture_type = cached_result['gesture_type']
                hand_openness = cached_result['hand_openness']
                hand_center = cached_result['hand_center']
            else:
                self.performance_stats['cache_misses'] += 1
                # 计算手势特征
                gesture_type = self._detect_gesture_type_fast(landmarks)
                hand_openness = self._calculate_hand_openness_fast(landmarks)
                hand_center = self._calculate_hand_center_fast(landmarks)

                # 缓存结果
                cache_data = {
                    'gesture_type': gesture_type,
                    'hand_openness': hand_openness,
                    'hand_center': hand_center
                }
                self.gesture_cache.set(landmarks, cache_data)

            # 检测区域
            active_region = self._detect_hand_in_region_fast(hand_center)

            # 手部数据
            hand_data = {
                'id': i,
                'label': handedness.classification[0].label.lower(),
                'detected': True,
                'landmarks': landmarks,
                'gesture_type': gesture_type,
                'openness': hand_openness,
                'center': hand_center,
                'active_region': active_region,
                'activation_strength': 0.0
            }

            # 计算激活强度
            if active_region:
                activation_strength = self._calculate_region_activation_strength_fast(hand_center, active_region)
                hand_data['activation_strength'] = activation_strength

                if active_region == 'central_control':
                    gesture_data['central_control_active'] = True
                elif active_region in self.voice_regions:
                    gesture_data['active_regions'][active_region] = hand_data
                    gesture_data['region_activations'][active_region] = activation_strength

            gesture_data['hands'].append(hand_data)

            # 累计置信度
            confidence = handedness.classification[0].score
            total_confidence += confidence

        # 计算综合手势强度
        total_strength = sum(hand['openness'] for hand in gesture_data['hands'])
        gesture_data['gesture_strength'] = total_strength / max(1, len(gesture_data['hands']))

        # 设置主手势类型和中心
        if gesture_data['hands']:
            primary_hand = gesture_data['hands'][0]
            gesture_data['gesture_type'] = primary_hand['gesture_type']
            gesture_data['hand_center'] = primary_hand['center']
            gesture_data['hand_openness'] = primary_hand['openness']

        gesture_data['confidence_score'] = total_confidence / max(1, len(gesture_data['hands']))

        return gesture_data

    def _detect_gesture_type_fast(self, landmarks: List[List[float]]) -> str:
        """快速手势类型检测"""
        if not landmarks or len(landmarks) < 21:
            return 'none'

        # 简化的手势检测算法
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        finger_pips = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]

        # 快速计算伸直的手指数量
        fingers_up = 0

        # 拇指（水平方向）
        if abs(finger_tips[0][0] - finger_pips[0][0]) > 0.04:
            fingers_up += 1

        # 其他手指（垂直方向）
        for i in range(1, 5):
            if finger_tips[i][1] < finger_pips[i][1] - 0.02:
                fingers_up += 1

        # 简化的手势分类
        if fingers_up >= 4:
            return 'open_hand'
        elif fingers_up <= 1:
            return 'fist'
        elif fingers_up == 1:
            return 'pointing'
        else:
            return 'conducting'

    def _calculate_hand_openness_fast(self, landmarks: List[List[float]]) -> float:
        """快速计算手部张开程度"""
        if not landmarks or len(landmarks) < 21:
            return 0.0

        # 使用简化的计算方法
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        # 计算拇指和小指间的距离作为张开程度的指标
        distance = math.sqrt(
            (thumb_tip[0] - pinky_tip[0])**2 +
            (thumb_tip[1] - pinky_tip[1])**2
        )

        # 标准化到0-1范围
        return min(distance / 0.25, 1.0)

    def _calculate_hand_center_fast(self, landmarks: List[List[float]]) -> List[float]:
        """快速计算手部中心"""
        if not landmarks:
            return [0.5, 0.5]

        # 使用手腕和中指根部的中点作为手部中心
        wrist = landmarks[0]
        middle_base = landmarks[9]

        center_x = (wrist[0] + middle_base[0]) / 2
        center_y = (wrist[1] + middle_base[1]) / 2

        return [center_x, center_y]

    def _detect_hand_in_region_fast(self, hand_center: List[float]) -> Optional[str]:
        """快速检测手部区域"""
        x, y = hand_center

        # 预先计算的区域边界（避免字典查找）
        if 0.33 <= x <= 0.67 and 0.4 <= y <= 0.8:
            return 'central_control'

        # 使用更高效的区域检测
        region_checks = [
            (0.0, 0.0, 0.33, 0.4, 'Tromba_I+II+III_in_D'),
            (0.33, 0.0, 0.67, 0.4, 'Violins_in_D'),
            (0.67, 0.0, 1.0, 0.4, 'Viola_in_D'),
            (0.0, 0.4, 0.33, 0.8, 'Oboe_I_in_D'),
            (0.67, 0.4, 1.0, 0.8, 'Continuo_in_D'),
            (0.0, 0.8, 0.5, 1.0, 'Organo_obligato_in_D'),
            (0.5, 0.8, 1.0, 1.0, 'Timpani_in_D')
        ]

        for x1, y1, x2, y2, region_name in region_checks:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return region_name

        return None

    def _calculate_region_activation_strength_fast(self, hand_center: List[float], region_name: str) -> float:
        """快速计算区域激活强度"""
        if region_name == 'central_control':
            region_center = [0.5, 0.6]
        else:
            # 预计算的区域中心
            centers = {
                'Tromba_I+II+III_in_D': [0.165, 0.2],
                'Violins_in_D': [0.5, 0.2],
                'Viola_in_D': [0.835, 0.2],
                'Oboe_I_in_D': [0.165, 0.6],
                'Continuo_in_D': [0.835, 0.6],
                'Organo_obligato_in_D': [0.25, 0.9],
                'Timpani_in_D': [0.75, 0.9]
            }
            region_center = centers.get(region_name, [0.5, 0.5])

        # 快速距离计算
        dx = hand_center[0] - region_center[0]
        dy = hand_center[1] - region_center[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # 激活强度计算
        max_distance = 0.15
        activation = max(0.0, 1.0 - (distance / max_distance))
        return activation

    def process_frame(self, frame: np.ndarray, show_regions: bool = True) -> np.ndarray:
        """处理视频帧（主入口点）"""
        current_time = time.time()

        # 添加帧到缓冲器
        self.frame_buffer.add_frame(frame)

        # 决定是否进行手势检测
        should_detect = (current_time - self.last_detection_time) >= self.detection_interval

        if should_detect:
            # 获取用于处理的帧
            frame_data = self.frame_buffer.get_frame_for_processing()
            if frame_data:
                try:
                    # 异步提交处理任务
                    self.processing_queue.put_nowait(frame_data)
                    self.last_detection_time = current_time
                except queue.Full:
                    # 队列满时跳过这一帧
                    self.performance_stats['frames_skipped'] += 1

        # 在当前帧上绘制结果
        result_frame = frame.copy()

        if self.last_gesture_data:
            result_frame = self._draw_gesture_results(result_frame, show_regions)

        return result_frame

    def _draw_gesture_results(self, frame: np.ndarray, show_regions: bool) -> np.ndarray:
        """在帧上绘制手势结果"""
        if not self.last_gesture_data:
            return frame

        height, width = frame.shape[:2]
        gesture_data = self.last_gesture_data

        # 绘制区域（如果需要）
        if show_regions:
            frame = self._draw_regions_overlay(frame)

        # 绘制手部信息
        for hand in gesture_data.hands:
            if 'center' in hand and len(hand['center']) >= 2:
                center_x = int(hand['center'][0] * width)
                center_y = int(hand['center'][1] * height)

                # 手势类型
                cv2.putText(frame, f"{hand['gesture_type']}", (center_x + 10, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 区域信息
                if hand.get('active_region'):
                    region_text = hand['active_region'].split('_')[0] if hand['active_region'] != 'central_control' else 'Central'
                    cv2.putText(frame, f"Region: {region_text}", (center_x + 10, center_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示性能信息
        cv2.putText(frame, f"FPS: {self.detection_fps} | Processing: {gesture_data.processing_time_ms:.1f}ms",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_regions_overlay(self, frame: np.ndarray) -> np.ndarray:
        """绘制区域覆盖层"""
        height, width = frame.shape[:2]

        # 绘制声部区域
        for region_name, region_data in self.voice_regions.items():
            bounds = region_data['bounds']
            color = region_data['color']

            x1 = int(bounds['x1'] * width)
            y1 = int(bounds['y1'] * height)
            x2 = int(bounds['x2'] * width)
            y2 = int(bounds['y2'] * height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{region_data['id']}-{region_name.split('_')[0]}"
            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制中央控制区域
        central_bounds = self.central_control_region['bounds']
        central_color = self.central_control_region['color']

        x1 = int(central_bounds['x1'] * width)
        y1 = int(central_bounds['y1'] * height)
        x2 = int(central_bounds['x2'] * width)
        y2 = int(central_bounds['y2'] * height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), central_color, 2)
        cv2.putText(frame, "Central Control", (x1 + 5, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, central_color, 1)

        return frame

    def get_gesture_data(self) -> Optional[OptimizedGestureData]:
        """获取当前手势数据"""
        return self.last_gesture_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])
        )

        # 内存使用情况
        try:
            import psutil
            process = psutil.Process()
            stats['memory_usage_mb'] = process.memory_info().rss / (1024**2)
        except:
            stats['memory_usage_mb'] = 0.0

        return stats

    def cleanup(self):
        """清理资源"""
        print("Cleaning up optimized gesture detector...")

        self.is_running = False

        # 停止处理线程
        self.processing_queue.put(None)  # 停止信号
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        # 关闭线程池
        self.thread_pool.shutdown(wait=True)

        # 清理MediaPipe资源
        self.hands.close()

        # 清理缓存
        self.gesture_cache.cache.clear()
        self.gesture_cache.access_times.clear()

        # 强制垃圾回收
        gc.collect()

        print("Optimized gesture detector cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 工厂函数
def create_optimized_detector(detection_fps: int = 15, display_fps: int = 30) -> OptimizedHandGestureDetector:
    """创建优化的手势检测器"""
    return OptimizedHandGestureDetector(detection_fps, display_fps)


def main():
    """测试优化的手势检测器"""
    print("Optimized Hand Gesture Detector Test")
    print("=" * 40)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    with create_optimized_detector(detection_fps=15, display_fps=30) as detector:
        print("Press 'q' to quit, 's' to show performance stats")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            processed_frame = detector.process_frame(frame, show_regions=True)

            # 显示结果
            cv2.imshow('Optimized Gesture Detection', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = detector.get_performance_stats()
                print("\nPerformance Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()