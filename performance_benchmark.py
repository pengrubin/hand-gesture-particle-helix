#!/usr/bin/env python3
"""
Performance Benchmark and Analysis Tool
手势控制音频播放系统性能基准测试工具

主要功能：
- CPU/内存使用率监控
- 帧率性能测试
- 音频延迟测量
- 系统负载分析
- 平台特定优化建议

Author: Performance Engineer
Date: 2025-10-05
"""

import time
import psutil
import threading
import numpy as np
import cv2
import pygame
import gc
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
import platform
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    fps: float
    frame_time_ms: float
    audio_latency_ms: float
    gesture_processing_time_ms: float
    active_threads: int
    gpu_usage_percent: float = 0.0  # macOS Metal API监控


@dataclass
class SystemInfo:
    """系统信息"""
    platform: str = field(default_factory=lambda: platform.system())
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count())
    memory_total_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    python_version: str = field(default_factory=lambda: sys.version)
    opencv_version: str = field(default_factory=lambda: cv2.__version__)
    pygame_version: str = field(default_factory=lambda: pygame.version.ver)


class PerformanceProfiler:
    """性能分析器 - 实时监控系统性能"""

    def __init__(self, max_history: int = 1000):
        """
        初始化性能分析器

        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.system_info = SystemInfo()

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 0.1  # 100ms监控间隔

        # 性能目标
        self.target_fps = 30
        self.target_memory_mb = 500
        self.target_cpu_percent = 60
        self.target_audio_latency_ms = 50

        # 当前监控值
        self.current_fps = 0.0
        self.current_frame_time = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()

        # 定时器
        self.timers: Dict[str, float] = {}

        # 警告状态
        self.performance_warnings: List[str] = []

    def start_monitoring(self):
        """开始性能监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Performance monitoring started")

    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        print("Performance monitoring stopped")

    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024**2)
                memory_percent = memory_info.percent

                # 计算FPS
                current_time = time.time()
                if self.frame_count > 0:
                    time_diff = current_time - self.last_frame_time
                    if time_diff > 0:
                        self.current_fps = 1.0 / time_diff
                        self.current_frame_time = time_diff * 1000  # ms

                # 创建性能指标
                metrics = PerformanceMetrics(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    fps=self.current_fps,
                    frame_time_ms=self.current_frame_time,
                    audio_latency_ms=self._measure_audio_latency(),
                    gesture_processing_time_ms=self.timers.get('gesture_processing', 0),
                    active_threads=threading.active_count(),
                    gpu_usage_percent=self._get_gpu_usage()
                )

                self.metrics_history.append(metrics)
                self._check_performance_warnings(metrics)

                time.sleep(self.monitor_interval)

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitor_interval)

    def _measure_audio_latency(self) -> float:
        """测量音频延迟（简化实现）"""
        # 这里可以实现更精确的音频延迟测量
        # 目前返回pygame mixer的缓冲区延迟估算
        try:
            if pygame.mixer.get_init():
                # 根据缓冲区大小估算延迟
                frequency, size, channels = pygame.mixer.get_init()
                buffer_size = 512  # 默认缓冲区大小
                latency_ms = (buffer_size / frequency) * 1000
                return latency_ms
        except:
            pass
        return 0.0

    def _get_gpu_usage(self) -> float:
        """获取GPU使用率（macOS Metal API）"""
        # macOS特定的GPU监控
        if self.system_info.platform == "Darwin":
            try:
                # 使用系统命令获取GPU信息
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True, text=True, timeout=1
                )
                # 简化版本，实际实现需要解析JSON
                return 0.0  # 占位符
            except:
                return 0.0
        return 0.0

    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """检查性能警告"""
        warnings = []

        if metrics.fps < self.target_fps * 0.8:
            warnings.append(f"Low FPS: {metrics.fps:.1f} < {self.target_fps}")

        if metrics.memory_mb > self.target_memory_mb:
            warnings.append(f"High memory usage: {metrics.memory_mb:.1f}MB > {self.target_memory_mb}MB")

        if metrics.cpu_percent > self.target_cpu_percent:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}% > {self.target_cpu_percent}%")

        if metrics.audio_latency_ms > self.target_audio_latency_ms:
            warnings.append(f"High audio latency: {metrics.audio_latency_ms:.1f}ms > {self.target_audio_latency_ms}ms")

        self.performance_warnings = warnings

    @contextmanager
    def timer(self, name: str):
        """计时器上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = (time.time() - start_time) * 1000  # ms
            self.timers[name] = elapsed_time

    def record_frame(self):
        """记录帧计数"""
        self.frame_count += 1
        self.last_frame_time = time.time()

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_average_metrics(self, window_seconds: float = 5.0) -> Dict[str, float]:
        """获取平均性能指标"""
        if not self.metrics_history:
            return {}

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # 筛选时间窗口内的数据
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        return {
            'avg_fps': np.mean([m.fps for m in recent_metrics]),
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
            'avg_memory_mb': np.mean([m.memory_mb for m in recent_metrics]),
            'avg_frame_time_ms': np.mean([m.frame_time_ms for m in recent_metrics]),
            'avg_audio_latency_ms': np.mean([m.audio_latency_ms for m in recent_metrics]),
            'max_cpu_percent': np.max([m.cpu_percent for m in recent_metrics]),
            'max_memory_mb': np.max([m.memory_mb for m in recent_metrics]),
            'min_fps': np.min([m.fps for m in recent_metrics])
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        current = self.get_current_metrics()
        averages = self.get_average_metrics()

        report = {
            'system_info': {
                'platform': self.system_info.platform,
                'cpu_count': self.system_info.cpu_count,
                'memory_total_gb': self.system_info.memory_total_gb,
                'python_version': self.system_info.python_version.split()[0],
                'opencv_version': self.system_info.opencv_version,
                'pygame_version': self.system_info.pygame_version
            },
            'current_metrics': current.__dict__ if current else {},
            'average_metrics': averages,
            'performance_targets': {
                'target_fps': self.target_fps,
                'target_memory_mb': self.target_memory_mb,
                'target_cpu_percent': self.target_cpu_percent,
                'target_audio_latency_ms': self.target_audio_latency_ms
            },
            'warnings': self.performance_warnings,
            'optimization_recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        averages = self.get_average_metrics()

        if not averages:
            return recommendations

        # FPS优化建议
        if averages.get('avg_fps', 30) < self.target_fps * 0.8:
            recommendations.extend([
                "考虑实现帧跳跃策略：每2帧处理一次手势检测",
                "降低MediaPipe检测置信度阈值到0.5",
                "减少视频处理分辨率到320x240",
                "使用多线程分离手势检测和渲染"
            ])

        # 内存优化建议
        if averages.get('avg_memory_mb', 0) > self.target_memory_mb:
            recommendations.extend([
                "实现numpy数组对象池重用",
                "定期调用gc.collect()进行垃圾回收",
                "限制MediaPipe历史缓存大小",
                "优化OpenCV Mat对象生命周期"
            ])

        # CPU优化建议
        if averages.get('avg_cpu_percent', 0) > self.target_cpu_percent:
            recommendations.extend([
                "使用更高效的颜色空间转换",
                "减少音频更新频率到15Hz",
                "启用pygame硬件加速",
                "优化手势检测算法复杂度"
            ])

        # 音频延迟优化
        if averages.get('avg_audio_latency_ms', 0) > self.target_audio_latency_ms:
            recommendations.extend([
                "减小pygame mixer缓冲区到256字节",
                "使用专用音频线程",
                "考虑使用ASIO音频驱动",
                "优化音量渐变算法"
            ])

        # macOS特定优化
        if self.system_info.platform == "Darwin":
            recommendations.extend([
                "启用Metal Performance Shaders加速",
                "使用Core Audio低延迟模式",
                "优化线程调度器亲和性",
                "启用自动引用计数优化"
            ])

        return recommendations

    def save_report(self, filename: str = None):
        """保存性能报告到文件"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        report = self.get_performance_report()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"Performance report saved to: {filename}")
        return filename


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler

    def run_gesture_detection_benchmark(self, duration_seconds: int = 30) -> Dict[str, float]:
        """运行手势检测性能基准测试"""
        print(f"Running gesture detection benchmark for {duration_seconds} seconds...")

        # 模拟手势检测负载
        import cv2
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        frame_count = 0
        processing_times = []

        while time.time() - start_time < duration_seconds:
            with self.profiler.timer('gesture_processing'):
                # 模拟手势检测
                rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)

                # 简单的后处理
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            processing_times.append(self.profiler.timers.get('gesture_processing', 0))
            frame_count += 1
            self.profiler.record_frame()

            # 控制帧率
            time.sleep(1/60)  # 目标60FPS

        hands.close()

        return {
            'total_frames': frame_count,
            'avg_fps': frame_count / duration_seconds,
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'std_processing_time_ms': np.std(processing_times)
        }

    def run_audio_latency_benchmark(self, test_duration: int = 10) -> Dict[str, float]:
        """运行音频延迟基准测试"""
        print(f"Running audio latency benchmark for {test_duration} seconds...")

        # 初始化pygame mixer
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
        pygame.mixer.init()

        # 创建测试音频
        sample_rate = 44100
        duration = 0.1  # 100ms测试音频
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

        # 转换为pygame格式
        test_audio_int = (test_audio * 32767).astype(np.int16)
        stereo_audio = np.array([test_audio_int, test_audio_int]).T
        test_sound = pygame.sndarray.make_sound(stereo_audio)

        latencies = []
        start_time = time.time()

        while time.time() - start_time < test_duration:
            # 测量播放延迟
            play_start = time.time()
            test_sound.play()

            # 等待音频开始播放
            while pygame.mixer.get_busy():
                pass

            play_end = time.time()
            latency_ms = (play_end - play_start) * 1000
            latencies.append(latency_ms)

            time.sleep(0.1)  # 100ms间隔

        pygame.mixer.quit()

        return {
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'std_latency_ms': np.std(latencies),
            'samples_count': len(latencies)
        }

    def run_memory_stress_test(self, duration_seconds: int = 60) -> Dict[str, float]:
        """运行内存压力测试"""
        print(f"Running memory stress test for {duration_seconds} seconds...")

        initial_memory = psutil.Process().memory_info().rss / (1024**2)
        memory_samples = []

        # 模拟内存使用模式
        data_arrays = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # 模拟视频帧处理
            frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            processed_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

            # 模拟特征提取
            features = np.random.random((21, 3))

            # 周期性清理
            if len(data_arrays) > 100:
                data_arrays = data_arrays[-50:]  # 保留最近50个
                gc.collect()

            data_arrays.append((processed_frame, features))

            current_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_samples.append(current_memory)

            time.sleep(0.033)  # ~30FPS

        # 清理
        data_arrays.clear()
        gc.collect()

        final_memory = psutil.Process().memory_info().rss / (1024**2)

        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': np.max(memory_samples),
            'avg_memory_mb': np.mean(memory_samples),
            'memory_growth_mb': final_memory - initial_memory,
            'memory_variance': np.var(memory_samples)
        }


def main():
    """主测试函数"""
    print("Performance Benchmark Tool")
    print("=" * 50)

    # 创建性能分析器
    profiler = PerformanceProfiler()
    profiler.start_monitoring()

    try:
        # 运行基准测试
        runner = BenchmarkRunner(profiler)

        print("\n1. Running gesture detection benchmark...")
        gesture_results = runner.run_gesture_detection_benchmark(30)

        print("\n2. Running audio latency benchmark...")
        audio_results = runner.run_audio_latency_benchmark(10)

        print("\n3. Running memory stress test...")
        memory_results = runner.run_memory_stress_test(60)

        # 等待收集足够的监控数据
        time.sleep(2)

        # 生成报告
        print("\n4. Generating performance report...")
        report = profiler.get_performance_report()

        # 添加基准测试结果
        report['benchmark_results'] = {
            'gesture_detection': gesture_results,
            'audio_latency': audio_results,
            'memory_stress': memory_results
        }

        # 保存报告
        filename = profiler.save_report()

        # 显示摘要
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)

        if profiler.performance_warnings:
            print("⚠️  PERFORMANCE WARNINGS:")
            for warning in profiler.performance_warnings:
                print(f"   - {warning}")
        else:
            print("✅ No performance warnings detected")

        print(f"\n📊 Gesture Detection:")
        print(f"   - Average FPS: {gesture_results['avg_fps']:.1f}")
        print(f"   - Average processing time: {gesture_results['avg_processing_time_ms']:.1f}ms")

        print(f"\n🔊 Audio Latency:")
        print(f"   - Average latency: {audio_results['avg_latency_ms']:.1f}ms")

        print(f"\n💾 Memory Usage:")
        print(f"   - Peak memory: {memory_results['peak_memory_mb']:.1f}MB")
        print(f"   - Memory growth: {memory_results['memory_growth_mb']:.1f}MB")

        print(f"\n📋 Top Optimization Recommendations:")
        recommendations = report['optimization_recommendations'][:5]
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        print(f"\n📄 Full report saved to: {filename}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")

    finally:
        profiler.stop_monitoring()


if __name__ == "__main__":
    main()