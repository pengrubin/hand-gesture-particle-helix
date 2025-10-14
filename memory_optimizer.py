#!/usr/bin/env python3
"""
Memory Optimizer
内存优化管理器 - 提供对象池、numpy重用、内存监控

主要功能：
1. numpy数组对象池管理
2. OpenCV Mat对象重用
3. MediaPipe结果缓存优化
4. 自动垃圾回收调度
5. 内存泄漏检测

性能目标：
- 内存增长：<50MB/小时
- 对象分配：减少80%
- GC频率：减少50%
- 内存碎片：最小化

Author: Performance Engineer
Date: 2025-10-05
"""

import gc
import time
import threading
import weakref
import numpy as np
import cv2
import tracemalloc
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
from enum import Enum
import sys
import psutil
import os


class MemoryPoolType(Enum):
    """内存池类型"""
    NUMPY_ARRAY = "numpy_array"
    CV2_MAT = "cv2_mat"
    MEDIAPIPE_RESULT = "mediapipe_result"
    GENERAL_BUFFER = "general_buffer"


@dataclass
class MemoryUsageSnapshot:
    """内存使用快照"""
    timestamp: float
    rss_mb: float  # 常驻内存
    vms_mb: float  # 虚拟内存
    percent: float  # 内存使用百分比
    python_allocated_mb: float  # Python分配的内存
    gc_collections: Tuple[int, int, int]  # GC收集次数
    object_count: int  # 对象数量


class NumpyArrayPool:
    """numpy数组对象池"""

    def __init__(self, max_arrays: int = 200, max_size_mb: int = 100):
        """
        初始化numpy数组池

        Args:
            max_arrays: 最大数组数量
            max_size_mb: 最大总大小(MB)
        """
        self.max_arrays = max_arrays
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.pools: Dict[Tuple[tuple, np.dtype], List[np.ndarray]] = defaultdict(list)
        self.total_size_bytes = 0
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.access_times: Dict[id, float] = {}

    def get_array(self, shape: Tuple[int, ...], dtype=np.uint8) -> np.ndarray:
        """
        获取numpy数组

        Args:
            shape: 数组形状
            dtype: 数据类型

        Returns:
            numpy数组
        """
        pool_key = (shape, dtype)

        with self.lock:
            if pool_key in self.pools and self.pools[pool_key]:
                array = self.pools[pool_key].pop()
                self.hit_count += 1
                self.access_times[id(array)] = time.time()

                # 清零数组内容
                array.fill(0)
                return array

        # 缓存未命中，创建新数组
        self.miss_count += 1
        array = np.zeros(shape, dtype=dtype)
        self.access_times[id(array)] = time.time()
        return array

    def return_array(self, array: np.ndarray):
        """
        归还numpy数组到池中

        Args:
            array: 要归还的数组
        """
        if array is None:
            return

        with self.lock:
            pool_key = (array.shape, array.dtype)
            array_size = array.nbytes

            # 检查池大小限制
            if (len(self.pools[pool_key]) < self.max_arrays and
                self.total_size_bytes + array_size <= self.max_size_bytes):

                # 确保数组是连续的
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)

                self.pools[pool_key].append(array)
                self.total_size_bytes += array_size
            else:
                # 如果池满了，清理最老的数组
                self._cleanup_old_arrays()

    def _cleanup_old_arrays(self):
        """清理最老的数组"""
        current_time = time.time()
        max_age = 300  # 5分钟过期

        arrays_to_remove = []
        for pool_arrays in self.pools.values():
            for i, array in enumerate(pool_arrays):
                array_id = id(array)
                if array_id in self.access_times:
                    age = current_time - self.access_times[array_id]
                    if age > max_age:
                        arrays_to_remove.append((pool_arrays, i, array.nbytes))

        # 移除过期数组
        for pool_arrays, index, size in sorted(arrays_to_remove, key=lambda x: x[1], reverse=True):
            if index < len(pool_arrays):
                pool_arrays.pop(index)
                self.total_size_bytes -= size

    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        with self.lock:
            total_arrays = sum(len(arrays) for arrays in self.pools.values())
            hit_rate = self.hit_count / max(1, self.hit_count + self.miss_count)

            return {
                'total_arrays': total_arrays,
                'total_size_mb': self.total_size_bytes / (1024 * 1024),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'pool_types': len(self.pools)
            }

    def clear(self):
        """清空对象池"""
        with self.lock:
            self.pools.clear()
            self.total_size_bytes = 0
            self.access_times.clear()


class CV2MatPool:
    """OpenCV Mat对象池"""

    def __init__(self, max_mats: int = 50):
        """
        初始化CV2 Mat池

        Args:
            max_mats: 最大Mat数量
        """
        self.max_mats = max_mats
        self.pools: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
        self.lock = threading.RLock()

    def get_mat(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """
        获取OpenCV Mat

        Args:
            height: 高度
            width: 宽度
            channels: 通道数

        Returns:
            OpenCV Mat（作为numpy数组）
        """
        key = (height, width, channels)

        with self.lock:
            if key in self.pools and self.pools[key]:
                mat = self.pools[key].pop()
                mat.fill(0)  # 清零
                return mat

        # 创建新的Mat
        if channels == 1:
            mat = np.zeros((height, width), dtype=np.uint8)
        else:
            mat = np.zeros((height, width, channels), dtype=np.uint8)

        return mat

    def return_mat(self, mat: np.ndarray):
        """归还Mat到池中"""
        if mat is None:
            return

        with self.lock:
            if len(mat.shape) == 2:
                key = (mat.shape[0], mat.shape[1], 1)
            else:
                key = (mat.shape[0], mat.shape[1], mat.shape[2])

            if len(self.pools[key]) < self.max_mats:
                self.pools[key].append(mat)

    def clear(self):
        """清空Mat池"""
        with self.lock:
            self.pools.clear()


class MediaPipeResultCache:
    """MediaPipe结果缓存"""

    def __init__(self, max_cache_size: int = 100, max_age_seconds: float = 1.0):
        """
        初始化MediaPipe结果缓存

        Args:
            max_cache_size: 最大缓存大小
            max_age_seconds: 最大缓存时间
        """
        self.max_cache_size = max_cache_size
        self.max_age_seconds = max_age_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()

    def _generate_frame_hash(self, frame: np.ndarray, quality: str = "fast") -> str:
        """生成帧哈希值"""
        if quality == "fast":
            # 快速哈希：使用帧的一部分
            if frame.size > 1000:
                sample = frame[::10, ::10].flatten()[:100]
                return str(hash(sample.tobytes()))
            else:
                return str(hash(frame.tobytes()))
        else:
            # 精确哈希
            return str(hash(frame.tobytes()))

    def get_result(self, frame: np.ndarray, quality: str = "fast") -> Optional[Any]:
        """获取缓存的处理结果"""
        frame_hash = self._generate_frame_hash(frame, quality)

        with self.lock:
            if frame_hash in self.cache:
                result, timestamp = self.cache[frame_hash]
                current_time = time.time()

                # 检查是否过期
                if current_time - timestamp <= self.max_age_seconds:
                    # 更新访问顺序
                    if frame_hash in self.access_order:
                        self.access_order.remove(frame_hash)
                    self.access_order.append(frame_hash)
                    return result
                else:
                    # 移除过期项
                    del self.cache[frame_hash]
                    if frame_hash in self.access_order:
                        self.access_order.remove(frame_hash)

        return None

    def set_result(self, frame: np.ndarray, result: Any, quality: str = "fast"):
        """设置缓存结果"""
        frame_hash = self._generate_frame_hash(frame, quality)

        with self.lock:
            current_time = time.time()

            # 如果缓存满了，移除最久未使用的项
            if len(self.cache) >= self.max_cache_size and frame_hash not in self.cache:
                if self.access_order:
                    oldest_hash = self.access_order.popleft()
                    self.cache.pop(oldest_hash, None)

            # 添加新结果
            self.cache[frame_hash] = (result, current_time)

            # 更新访问顺序
            if frame_hash in self.access_order:
                self.access_order.remove(frame_hash)
            self.access_order.append(frame_hash)

    def clear_expired(self):
        """清理过期缓存"""
        current_time = time.time()

        with self.lock:
            expired_keys = []
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp > self.max_age_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_cache_size,
                'access_order_size': len(self.access_order)
            }

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, monitor_interval: float = 1.0, history_size: int = 300):
        """
        初始化内存监控器

        Args:
            monitor_interval: 监控间隔（秒）
            history_size: 历史记录大小
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        self.snapshots: deque = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 启用tracemalloc进行精确内存跟踪
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # 获取进程对象
        self.process = psutil.Process()

    def start_monitoring(self):
        """开始内存监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """停止内存监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 获取内存信息
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()

                # 获取Python内存分配信息
                current, peak = tracemalloc.get_traced_memory()

                # 获取GC统计
                gc_stats = gc.get_stats()
                gc_collections = (gc_stats[0]['collections'],
                                gc_stats[1]['collections'],
                                gc_stats[2]['collections'])

                # 获取对象数量
                object_count = len(gc.get_objects())

                # 创建快照
                snapshot = MemoryUsageSnapshot(
                    timestamp=time.time(),
                    rss_mb=memory_info.rss / (1024 * 1024),
                    vms_mb=memory_info.vms / (1024 * 1024),
                    percent=memory_percent,
                    python_allocated_mb=current / (1024 * 1024),
                    gc_collections=gc_collections,
                    object_count=object_count
                )

                self.snapshots.append(snapshot)

                time.sleep(self.monitor_interval)

            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.monitor_interval)

    def get_current_usage(self) -> Optional[MemoryUsageSnapshot]:
        """获取当前内存使用情况"""
        if self.snapshots:
            return self.snapshots[-1]
        return None

    def get_memory_growth_rate(self, window_minutes: int = 5) -> float:
        """
        计算内存增长率

        Args:
            window_minutes: 时间窗口（分钟）

        Returns:
            增长率（MB/小时）
        """
        if len(self.snapshots) < 2:
            return 0.0

        current_time = time.time()
        window_seconds = window_minutes * 60

        # 筛选时间窗口内的快照
        recent_snapshots = [
            s for s in self.snapshots
            if current_time - s.timestamp <= window_seconds
        ]

        if len(recent_snapshots) < 2:
            return 0.0

        # 计算增长率
        oldest = recent_snapshots[0]
        newest = recent_snapshots[-1]

        time_diff_hours = (newest.timestamp - oldest.timestamp) / 3600
        memory_diff_mb = newest.rss_mb - oldest.rss_mb

        return memory_diff_mb / max(time_diff_hours, 0.001)

    def detect_memory_leaks(self, threshold_mb_per_hour: float = 10.0) -> List[str]:
        """
        检测内存泄漏

        Args:
            threshold_mb_per_hour: 泄漏阈值（MB/小时）

        Returns:
            警告列表
        """
        warnings = []

        # 检查增长率
        growth_rate = self.get_memory_growth_rate()
        if growth_rate > threshold_mb_per_hour:
            warnings.append(f"高内存增长率: {growth_rate:.1f} MB/小时")

        # 检查对象数量增长
        if len(self.snapshots) >= 10:
            recent_snapshots = list(self.snapshots)[-10:]
            object_counts = [s.object_count for s in recent_snapshots]

            # 计算对象数量趋势
            if len(object_counts) >= 5:
                early_avg = np.mean(object_counts[:len(object_counts)//2])
                late_avg = np.mean(object_counts[len(object_counts)//2:])

                if late_avg > early_avg * 1.2:  # 增长超过20%
                    warnings.append(f"对象数量快速增长: {early_avg:.0f} -> {late_avg:.0f}")

        return warnings


class MemoryOptimizer:
    """内存优化管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化内存优化器

        Args:
            config: 配置选项
        """
        self.config = {
            'numpy_pool_max_arrays': 200,
            'numpy_pool_max_size_mb': 100,
            'cv2_pool_max_mats': 50,
            'mediapipe_cache_size': 100,
            'mediapipe_cache_age': 1.0,
            'gc_threshold_multiplier': 1.5,
            'auto_gc_interval': 30.0,
            'memory_monitor_enabled': True,
            'leak_detection_enabled': True
        }

        if config:
            self.config.update(config)

        # 初始化对象池
        self.numpy_pool = NumpyArrayPool(
            max_arrays=self.config['numpy_pool_max_arrays'],
            max_size_mb=self.config['numpy_pool_max_size_mb']
        )

        self.cv2_pool = CV2MatPool(
            max_mats=self.config['cv2_pool_max_mats']
        )

        self.mediapipe_cache = MediaPipeResultCache(
            max_cache_size=self.config['mediapipe_cache_size'],
            max_age_seconds=self.config['mediapipe_cache_age']
        )

        # 内存监控
        self.memory_monitor = MemoryMonitor() if self.config['memory_monitor_enabled'] else None

        # GC优化
        self._optimize_gc_thresholds()

        # 自动清理
        self.auto_cleanup_enabled = True
        self.cleanup_thread: Optional[threading.Thread] = None
        self._start_auto_cleanup()

    def _optimize_gc_thresholds(self):
        """优化垃圾回收阈值"""
        # 获取当前阈值
        current_thresholds = gc.get_threshold()

        # 增加阈值以减少GC频率
        multiplier = self.config['gc_threshold_multiplier']
        new_thresholds = tuple(int(t * multiplier) for t in current_thresholds)

        gc.set_threshold(*new_thresholds)

        print(f"GC thresholds: {current_thresholds} -> {new_thresholds}")

    def _start_auto_cleanup(self):
        """启动自动清理线程"""
        if self.auto_cleanup_enabled:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()

    def _cleanup_loop(self):
        """自动清理循环"""
        interval = self.config['auto_gc_interval']

        while self.auto_cleanup_enabled:
            try:
                time.sleep(interval)

                # 清理过期缓存
                self.mediapipe_cache.clear_expired()

                # 定期强制垃圾回收
                gc.collect()

                # 检测内存泄漏
                if (self.config['leak_detection_enabled'] and
                    self.memory_monitor):
                    warnings = self.memory_monitor.detect_memory_leaks()
                    if warnings:
                        print("Memory leak warnings:", warnings)

            except Exception as e:
                print(f"Cleanup loop error: {e}")

    def get_numpy_array(self, shape: Tuple[int, ...], dtype=np.uint8) -> np.ndarray:
        """获取numpy数组（推荐使用此方法）"""
        return self.numpy_pool.get_array(shape, dtype)

    def return_numpy_array(self, array: np.ndarray):
        """归还numpy数组"""
        self.numpy_pool.return_array(array)

    def get_cv2_mat(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """获取OpenCV Mat"""
        return self.cv2_pool.get_mat(height, width, channels)

    def return_cv2_mat(self, mat: np.ndarray):
        """归还OpenCV Mat"""
        self.cv2_pool.return_mat(mat)

    def cache_mediapipe_result(self, frame: np.ndarray, result: Any):
        """缓存MediaPipe结果"""
        self.mediapipe_cache.set_result(frame, result)

    def get_cached_mediapipe_result(self, frame: np.ndarray) -> Optional[Any]:
        """获取缓存的MediaPipe结果"""
        return self.mediapipe_cache.get_result(frame)

    def start_monitoring(self):
        """开始内存监控"""
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()

    def stop_monitoring(self):
        """停止内存监控"""
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()

    def force_gc(self):
        """强制垃圾回收"""
        return gc.collect()

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        stats = {
            'numpy_pool': self.numpy_pool.get_stats(),
            'mediapipe_cache': self.mediapipe_cache.get_stats(),
        }

        if self.memory_monitor:
            current_usage = self.memory_monitor.get_current_usage()
            if current_usage:
                stats['current_memory'] = {
                    'rss_mb': current_usage.rss_mb,
                    'percent': current_usage.percent,
                    'python_allocated_mb': current_usage.python_allocated_mb,
                    'object_count': current_usage.object_count
                }

                stats['memory_growth_rate'] = self.memory_monitor.get_memory_growth_rate()

        return stats

    def cleanup(self):
        """清理所有资源"""
        print("Cleaning up memory optimizer...")

        # 停止自动清理
        self.auto_cleanup_enabled = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)

        # 停止监控
        self.stop_monitoring()

        # 清空所有池
        self.numpy_pool.clear()
        self.cv2_pool.clear()
        self.mediapipe_cache.clear()

        # 强制垃圾回收
        self.force_gc()

        print("Memory optimizer cleanup completed")


# 全局优化器实例
_global_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
        _global_optimizer.start_monitoring()
    return _global_optimizer


def cleanup_memory_optimizer():
    """清理全局内存优化器"""
    global _global_optimizer
    if _global_optimizer:
        _global_optimizer.cleanup()
        _global_optimizer = None


# 便利函数
def get_optimized_array(shape: Tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    """获取优化的numpy数组"""
    return get_memory_optimizer().get_numpy_array(shape, dtype)


def return_optimized_array(array: np.ndarray):
    """归还优化的numpy数组"""
    get_memory_optimizer().return_numpy_array(array)


def get_optimized_mat(height: int, width: int, channels: int = 3) -> np.ndarray:
    """获取优化的OpenCV Mat"""
    return get_memory_optimizer().get_cv2_mat(height, width, channels)


def return_optimized_mat(mat: np.ndarray):
    """归还优化的OpenCV Mat"""
    get_memory_optimizer().return_cv2_mat(mat)


def main():
    """测试内存优化器"""
    print("Memory Optimizer Test")
    print("=" * 30)

    optimizer = MemoryOptimizer()
    optimizer.start_monitoring()

    try:
        # 测试numpy数组池
        print("\n测试numpy数组池...")
        arrays = []
        for i in range(50):
            array = optimizer.get_numpy_array((480, 640, 3))
            arrays.append(array)

        # 归还数组
        for array in arrays:
            optimizer.return_numpy_array(array)

        # 测试CV2 Mat池
        print("测试CV2 Mat池...")
        mats = []
        for i in range(20):
            mat = optimizer.get_cv2_mat(240, 320, 3)
            mats.append(mat)

        for mat in mats:
            optimizer.return_cv2_mat(mat)

        # 等待监控数据
        time.sleep(3)

        # 显示统计信息
        stats = optimizer.get_memory_stats()
        print("\n内存统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 测试内存增长
        print("\n测试内存增长检测...")
        for i in range(100):
            # 模拟内存分配
            temp_arrays = [np.random.random((100, 100)) for _ in range(10)]
            time.sleep(0.01)

        time.sleep(2)
        growth_rate = optimizer.memory_monitor.get_memory_growth_rate()
        print(f"内存增长率: {growth_rate:.2f} MB/小时")

        # 检测泄漏
        leaks = optimizer.memory_monitor.detect_memory_leaks()
        if leaks:
            print("检测到潜在内存泄漏:")
            for leak in leaks:
                print(f"  - {leak}")
        else:
            print("未检测到内存泄漏")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        optimizer.cleanup()

    print("\nMemory optimizer test completed")


if __name__ == "__main__":
    main()