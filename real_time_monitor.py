#!/usr/bin/env python3
"""
Real-time Performance Monitor
实时性能监控系统 - 提供全面的系统性能监控和可视化

主要功能：
1. 实时FPS、CPU、内存监控
2. 音频延迟和质量监控
3. 手势检测性能跟踪
4. 系统负载分析
5. 性能瓶颈检测和报警

性能指标：
- 更新频率：10Hz
- 历史数据：最近5分钟
- 报警延迟：<100ms
- 数据精度：毫秒级

Author: Performance Engineer
Date: 2025-10-05
"""

import time
import threading
import queue
import json
import psutil
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import logging
import platform
import socket
import sys
from datetime import datetime, timedelta
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: float = field(default_factory=time.time)

    # 系统指标
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0

    # 应用指标
    fps: float = 0.0
    frame_time_ms: float = 0.0

    # 手势检测指标
    gesture_detection_time_ms: float = 0.0
    gesture_confidence: float = 0.0
    hands_detected: int = 0

    # 音频指标
    audio_latency_ms: float = 0.0
    audio_buffer_underruns: int = 0
    active_audio_channels: int = 0

    # 内存指标
    numpy_arrays_pooled: int = 0
    cv2_mats_pooled: int = 0
    mediapipe_cache_hits: int = 0
    mediapipe_cache_misses: int = 0

    # 线程指标
    active_threads: int = 0
    thread_cpu_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """性能警报"""
    timestamp: float
    severity: str  # "info", "warning", "error", "critical"
    category: str  # "system", "audio", "gesture", "memory"
    message: str
    value: float
    threshold: float
    suggestion: Optional[str] = None


class PerformanceThresholds:
    """性能阈值配置"""

    def __init__(self):
        # FPS阈值
        self.fps_warning = 20.0
        self.fps_critical = 10.0

        # CPU阈值
        self.cpu_warning = 70.0
        self.cpu_critical = 90.0

        # 内存阈值
        self.memory_warning = 70.0
        self.memory_critical = 90.0

        # 音频延迟阈值
        self.audio_latency_warning = 50.0  # ms
        self.audio_latency_critical = 100.0  # ms

        # 手势检测阈值
        self.gesture_time_warning = 50.0  # ms
        self.gesture_time_critical = 100.0  # ms


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.process = psutil.Process()
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.gesture_times = deque(maxlen=30)
        self.audio_stats = {}
        self.memory_pools = {}

    def collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = self.process.cpu_percent()

            # 内存使用率
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': system_memory.percent,
                'memory_mb': memory_info.rss / (1024 * 1024),
                'active_threads': threading.active_count()
            }
        except Exception as e:
            logging.warning(f"Failed to collect system metrics: {e}")
            return {}

    def collect_fps_metrics(self) -> Dict[str, float]:
        """收集FPS指标"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_count += 1

        fps = 1.0 / frame_time if frame_time > 0 else 0.0

        return {
            'fps': fps,
            'frame_time_ms': frame_time * 1000
        }

    def record_gesture_detection_time(self, detection_time_ms: float, confidence: float, hands_count: int):
        """记录手势检测时间"""
        self.gesture_times.append(detection_time_ms)

        return {
            'gesture_detection_time_ms': detection_time_ms,
            'gesture_confidence': confidence,
            'hands_detected': hands_count
        }

    def update_audio_stats(self, latency_ms: float, buffer_underruns: int, active_channels: int):
        """更新音频统计"""
        self.audio_stats = {
            'audio_latency_ms': latency_ms,
            'audio_buffer_underruns': buffer_underruns,
            'active_audio_channels': active_channels
        }

    def update_memory_pools(self, numpy_count: int, cv2_count: int, cache_hits: int, cache_misses: int):
        """更新内存池统计"""
        self.memory_pools = {
            'numpy_arrays_pooled': numpy_count,
            'cv2_mats_pooled': cv2_count,
            'mediapipe_cache_hits': cache_hits,
            'mediapipe_cache_misses': cache_misses
        }

    def collect_all_metrics(self) -> PerformanceMetrics:
        """收集所有指标"""
        metrics = PerformanceMetrics()

        # 系统指标
        system_metrics = self.collect_system_metrics()
        for key, value in system_metrics.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        # 音频指标
        for key, value in self.audio_stats.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        # 内存池指标
        for key, value in self.memory_pools.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        return metrics


class AlertManager:
    """警报管理器"""

    def __init__(self, thresholds: PerformanceThresholds):
        self.thresholds = thresholds
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.cooldown_period = 30.0  # 30秒冷却期
        self.last_alert_times: Dict[str, float] = {}

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)

    def check_metrics(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """检查指标并生成警报"""
        alerts = []
        current_time = time.time()

        # 检查FPS
        if metrics.fps > 0:
            if metrics.fps < self.thresholds.fps_critical:
                alert = self._create_alert(
                    "critical", "system", "FPS严重偏低",
                    metrics.fps, self.thresholds.fps_critical,
                    "考虑降低分辨率或启用帧跳跃"
                )
                alerts.append(alert)
            elif metrics.fps < self.thresholds.fps_warning:
                alert = self._create_alert(
                    "warning", "system", "FPS偏低",
                    metrics.fps, self.thresholds.fps_warning,
                    "检查CPU使用率和优化设置"
                )
                alerts.append(alert)

        # 检查CPU
        if metrics.cpu_percent > self.thresholds.cpu_critical:
            alert = self._create_alert(
                "critical", "system", "CPU使用率过高",
                metrics.cpu_percent, self.thresholds.cpu_critical,
                "关闭其他应用程序或降低处理复杂度"
            )
            alerts.append(alert)
        elif metrics.cpu_percent > self.thresholds.cpu_warning:
            alert = self._create_alert(
                "warning", "system", "CPU使用率较高",
                metrics.cpu_percent, self.thresholds.cpu_warning,
                "考虑优化算法或启用多线程"
            )
            alerts.append(alert)

        # 检查内存
        if metrics.memory_percent > self.thresholds.memory_critical:
            alert = self._create_alert(
                "critical", "memory", "内存使用率过高",
                metrics.memory_percent, self.thresholds.memory_critical,
                "清理内存池或重启应用程序"
            )
            alerts.append(alert)
        elif metrics.memory_percent > self.thresholds.memory_warning:
            alert = self._create_alert(
                "warning", "memory", "内存使用率较高",
                metrics.memory_percent, self.thresholds.memory_warning,
                "启用垃圾回收或优化内存使用"
            )
            alerts.append(alert)

        # 检查音频延迟
        if metrics.audio_latency_ms > self.thresholds.audio_latency_critical:
            alert = self._create_alert(
                "critical", "audio", "音频延迟过高",
                metrics.audio_latency_ms, self.thresholds.audio_latency_critical,
                "减小音频缓冲区大小或使用专用音频线程"
            )
            alerts.append(alert)
        elif metrics.audio_latency_ms > self.thresholds.audio_latency_warning:
            alert = self._create_alert(
                "warning", "audio", "音频延迟较高",
                metrics.audio_latency_ms, self.thresholds.audio_latency_warning,
                "优化音频设置或检查驱动程序"
            )
            alerts.append(alert)

        # 检查手势检测时间
        if metrics.gesture_detection_time_ms > self.thresholds.gesture_time_critical:
            alert = self._create_alert(
                "critical", "gesture", "手势检测延迟过高",
                metrics.gesture_detection_time_ms, self.thresholds.gesture_time_critical,
                "降低MediaPipe模型复杂度或减少检测频率"
            )
            alerts.append(alert)
        elif metrics.gesture_detection_time_ms > self.thresholds.gesture_time_warning:
            alert = self._create_alert(
                "warning", "gesture", "手势检测延迟较高",
                metrics.gesture_detection_time_ms, self.thresholds.gesture_time_warning,
                "优化MediaPipe参数或启用缓存"
            )
            alerts.append(alert)

        # 过滤冷却期内的重复警报
        filtered_alerts = []
        for alert in alerts:
            alert_key = f"{alert.category}_{alert.severity}"
            last_time = self.last_alert_times.get(alert_key, 0)

            if current_time - last_time > self.cooldown_period:
                filtered_alerts.append(alert)
                self.last_alert_times[alert_key] = current_time

        # 触发回调
        for alert in filtered_alerts:
            self.alert_history.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.warning(f"Alert callback failed: {e}")

        return filtered_alerts

    def _create_alert(self, severity: str, category: str, message: str,
                     value: float, threshold: float, suggestion: str = None) -> PerformanceAlert:
        """创建警报"""
        return PerformanceAlert(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            value=value,
            threshold=threshold,
            suggestion=suggestion
        )

    def get_recent_alerts(self, minutes: int = 5) -> List[PerformanceAlert]:
        """获取最近的警报"""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class PerformanceRecorder:
    """性能记录器"""

    def __init__(self, max_history_minutes: int = 5):
        self.max_history_seconds = max_history_minutes * 60
        self.metrics_history: deque = deque()
        self.recording_enabled = True
        self.lock = threading.RLock()

    def record_metrics(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        if not self.recording_enabled:
            return

        with self.lock:
            self.metrics_history.append(metrics)
            self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """清理旧的指标数据"""
        current_time = time.time()
        cutoff_time = current_time - self.max_history_seconds

        while (self.metrics_history and
               self.metrics_history[0].timestamp < cutoff_time):
            self.metrics_history.popleft()

    def get_metrics_history(self, minutes: int = None) -> List[PerformanceMetrics]:
        """获取指标历史"""
        with self.lock:
            if minutes is None:
                return list(self.metrics_history)

            cutoff_time = time.time() - (minutes * 60)
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_average_metrics(self, minutes: int = 1) -> Optional[PerformanceMetrics]:
        """获取平均指标"""
        history = self.get_metrics_history(minutes)
        if not history:
            return None

        # 计算平均值
        avg_metrics = PerformanceMetrics()

        numeric_fields = [
            'cpu_percent', 'memory_percent', 'memory_mb', 'fps', 'frame_time_ms',
            'gesture_detection_time_ms', 'gesture_confidence', 'hands_detected',
            'audio_latency_ms', 'audio_buffer_underruns', 'active_audio_channels',
            'numpy_arrays_pooled', 'cv2_mats_pooled', 'mediapipe_cache_hits',
            'mediapipe_cache_misses', 'active_threads'
        ]

        for field in numeric_fields:
            values = [getattr(m, field) for m in history if hasattr(m, field)]
            if values:
                setattr(avg_metrics, field, np.mean(values))

        avg_metrics.timestamp = time.time()
        return avg_metrics

    def export_to_json(self, filename: str = None) -> str:
        """导出指标到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"

        with self.lock:
            data = {
                'export_timestamp': time.time(),
                'metrics_count': len(self.metrics_history),
                'metrics': [asdict(m) for m in self.metrics_history]
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return filename


class RealTimeMonitor:
    """实时性能监控器"""

    def __init__(self, update_interval: float = 0.1, config: Optional[Dict[str, Any]] = None):
        """
        初始化实时监控器

        Args:
            update_interval: 更新间隔（秒）
            config: 配置选项
        """
        self.update_interval = update_interval
        self.config = config or {}

        # 初始化组件
        self.thresholds = PerformanceThresholds()
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager(self.thresholds)
        self.recorder = PerformanceRecorder()

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 性能计时器
        self.timers: Dict[str, float] = {}
        self.timer_lock = threading.RLock()

        # 设置日志
        self._setup_logging()

        # 注册默认警报回调
        self.alert_manager.add_alert_callback(self._default_alert_handler)

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('RealTimeMonitor')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _default_alert_handler(self, alert: PerformanceAlert):
        """默认警报处理器"""
        level_map = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }

        log_func = level_map.get(alert.severity, self.logger.info)
        message = f"[{alert.category.upper()}] {alert.message} (值: {alert.value:.1f}, 阈值: {alert.threshold:.1f})"

        if alert.suggestion:
            message += f" | 建议: {alert.suggestion}"

        log_func(message)

    def start_monitoring(self):
        """开始实时监控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Real-time monitoring started")

    def stop_monitoring(self):
        """停止实时监控"""
        self.is_monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        self.logger.info("Real-time monitoring stopped")

    def _monitor_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                start_time = time.time()

                # 收集指标
                metrics = self.collector.collect_all_metrics()

                # 记录指标
                self.recorder.record_metrics(metrics)

                # 检查警报
                alerts = self.alert_manager.check_metrics(metrics)

                # 控制更新频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(self.update_interval)

    @contextmanager
    def timer(self, name: str):
        """性能计时器上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            with self.timer_lock:
                self.timers[name] = elapsed_ms

    def record_frame(self):
        """记录帧事件"""
        fps_metrics = self.collector.collect_fps_metrics()
        # 这里可以实时更新FPS指标

    def record_gesture_detection(self, detection_time_ms: float, confidence: float, hands_count: int):
        """记录手势检测结果"""
        self.collector.record_gesture_detection_time(detection_time_ms, confidence, hands_count)

    def update_audio_stats(self, latency_ms: float, buffer_underruns: int, active_channels: int):
        """更新音频统计"""
        self.collector.update_audio_stats(latency_ms, buffer_underruns, active_channels)

    def update_memory_stats(self, numpy_count: int, cv2_count: int, cache_hits: int, cache_misses: int):
        """更新内存统计"""
        self.collector.update_memory_pools(numpy_count, cv2_count, cache_hits, cache_misses)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        return self.collector.collect_all_metrics()

    def get_average_metrics(self, minutes: int = 1) -> Optional[PerformanceMetrics]:
        """获取平均性能指标"""
        return self.recorder.get_average_metrics(minutes)

    def get_recent_alerts(self, minutes: int = 5) -> List[PerformanceAlert]:
        """获取最近的警报"""
        return self.alert_manager.get_recent_alerts(minutes)

    def export_performance_data(self, filename: str = None) -> str:
        """导出性能数据"""
        return self.recorder.export_to_json(filename)

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        current_metrics = self.get_current_metrics()
        avg_metrics = self.get_average_metrics(5)  # 5分钟平均
        recent_alerts = self.get_recent_alerts(10)  # 10分钟内的警报

        report = {
            'timestamp': time.time(),
            'system_info': {
                'platform': platform.system(),
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'average_metrics_5min': asdict(avg_metrics) if avg_metrics else None,
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'alert_summary': {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in recent_alerts if a.severity == 'warning'])
            },
            'performance_timers': self.timers.copy()
        }

        return report

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """添加自定义警报回调"""
        self.alert_manager.add_alert_callback(callback)


# 全局监控器实例
_global_monitor: Optional[RealTimeMonitor] = None


def get_performance_monitor() -> RealTimeMonitor:
    """获取全局性能监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealTimeMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def cleanup_performance_monitor():
    """清理全局性能监控器"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None


def main():
    """测试实时性能监控器"""
    print("Real-time Performance Monitor Test")
    print("=" * 40)

    monitor = RealTimeMonitor(update_interval=0.5)  # 0.5秒更新间隔
    monitor.start_monitoring()

    try:
        # 模拟一些性能数据
        for i in range(20):
            # 模拟帧处理
            monitor.record_frame()

            # 模拟手势检测
            detection_time = np.random.uniform(20, 80)
            confidence = np.random.uniform(0.5, 1.0)
            hands = np.random.randint(0, 3)
            monitor.record_gesture_detection(detection_time, confidence, hands)

            # 模拟音频统计
            latency = np.random.uniform(30, 70)
            underruns = np.random.randint(0, 5)
            channels = np.random.randint(3, 9)
            monitor.update_audio_stats(latency, underruns, channels)

            # 模拟内存统计
            numpy_count = np.random.randint(50, 200)
            cv2_count = np.random.randint(10, 50)
            cache_hits = np.random.randint(100, 500)
            cache_misses = np.random.randint(10, 100)
            monitor.update_memory_stats(numpy_count, cv2_count, cache_hits, cache_misses)

            time.sleep(1)

        # 等待收集数据
        time.sleep(2)

        # 生成报告
        report = monitor.generate_performance_report()
        print("\n性能报告:")
        print(f"当前FPS: {report['current_metrics']['fps']:.1f}")
        print(f"平均CPU使用率: {report['average_metrics_5min']['cpu_percent']:.1f}%")
        print(f"平均内存使用率: {report['average_metrics_5min']['memory_percent']:.1f}%")
        print(f"警报总数: {report['alert_summary']['total_alerts']}")

        # 导出数据
        filename = monitor.export_performance_data()
        print(f"\n性能数据已导出到: {filename}")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        monitor.stop_monitoring()

    print("\nReal-time monitor test completed")


if __name__ == "__main__":
    main()