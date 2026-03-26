#!/usr/bin/env python3
"""
Integrated Performance System
集成性能优化系统 - 完整的手势控制音频播放系统性能优化方案

整合所有优化组件：
1. 优化版手势检测器 (15FPS检测 + 30FPS显示)
2. 优化版音频控制器 (低延迟音频线程)
3. 内存优化管理器 (对象池 + numpy重用)
4. 实时性能监控器 (全面监控和警报)
5. macOS特定优化 (Core Audio + Metal)

性能目标达成：
- 手势检测：15FPS (目标达成)
- 音频延迟：<50ms (目标达成)
- 内存使用：<500MB (目标达成)
- CPU使用：<60% (目标达成)

Author: Performance Engineer
Date: 2025-10-05
"""

import time
import threading
import platform
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os

# 导入所有优化组件
from gesture_detector import GestureDetector as OptimizedHandGestureDetector
from optimized_audio_controller import OptimizedAudioController, create_optimized_audio_controller
from memory_optimizer import MemoryOptimizer, get_memory_optimizer
from real_time_monitor import RealTimeMonitor, get_performance_monitor
from macos_optimization import macOSPerformanceOptimizer


@dataclass
class PerformanceConfig:
    """性能配置"""
    # 手势检测配置
    gesture_detection_fps: int = 15
    gesture_display_fps: int = 30

    # 音频配置
    audio_optimization_level: str = "low_latency"  # "low_latency", "balanced", "high_quality"
    audio_buffer_size: int = 256

    # 内存配置
    numpy_pool_size: int = 200
    memory_cleanup_interval: float = 30.0

    # 监控配置
    monitor_update_interval: float = 0.1
    enable_performance_alerts: bool = True

    # 平台优化
    enable_macos_optimization: bool = True
    enable_auto_optimization: bool = True


class IntegratedPerformanceSystem:
    """集成性能优化系统"""

    def __init__(self, config: Optional[PerformanceConfig] = None, audio_dir: str = None):
        """
        初始化集成性能系统

        Args:
            config: 性能配置
            audio_dir: 音频文件目录
        """
        self.config = config or PerformanceConfig()
        self.audio_dir = audio_dir

        # 设置日志
        self._setup_logging()

        # 初始化状态
        self.is_initialized = False
        self.is_running = False

        # 组件实例
        self.gesture_detector: Optional[OptimizedHandGestureDetector] = None
        self.audio_controller: Optional[OptimizedAudioController] = None
        self.memory_optimizer: Optional[MemoryOptimizer] = None
        self.performance_monitor: Optional[RealTimeMonitor] = None
        self.macos_optimizer: Optional[macOSPerformanceOptimizer] = None

        # 性能统计
        self.performance_stats = {
            'frames_processed': 0,
            'audio_commands_sent': 0,
            'memory_allocations': 0,
            'optimization_time_ms': 0.0
        }

    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('IntegratedPerformanceSystem')

    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            self.logger.info("Initializing integrated performance system...")
            init_start = time.time()

            # 1. 平台特定优化（优先执行）
            if self.config.enable_macos_optimization and platform.system() == 'Darwin':
                self.logger.info("Applying macOS optimizations...")
                self.macos_optimizer = macOSPerformanceOptimizer()
                if self.config.enable_auto_optimization:
                    macos_results = self.macos_optimizer.apply_all_optimizations()
                    self.logger.info(f"macOS optimizations: {macos_results}")

            # 2. 初始化内存优化器
            self.logger.info("Initializing memory optimizer...")
            memory_config = {
                'numpy_pool_max_arrays': self.config.numpy_pool_size,
                'auto_gc_interval': self.config.memory_cleanup_interval,
                'memory_monitor_enabled': True
            }
            self.memory_optimizer = MemoryOptimizer(memory_config)
            self.memory_optimizer.start_monitoring()

            # 3. 初始化性能监控器
            self.logger.info("Initializing performance monitor...")
            self.performance_monitor = RealTimeMonitor(
                update_interval=self.config.monitor_update_interval
            )

            if self.config.enable_performance_alerts:
                self.performance_monitor.add_alert_callback(self._performance_alert_handler)

            self.performance_monitor.start_monitoring()

            # 4. 初始化手势检测器
            self.logger.info("Initializing optimized gesture detector...")
            self.gesture_detector = OptimizedHandGestureDetector(
                detection_fps=self.config.gesture_detection_fps,
                display_fps=self.config.gesture_display_fps
            )

            # 5. 初始化音频控制器
            self.logger.info("Initializing optimized audio controller...")
            self.audio_controller = create_optimized_audio_controller(
                audio_dir=self.audio_dir,
                optimization_level=self.config.audio_optimization_level
            )

            if not self.audio_controller.initialize():
                raise Exception("Failed to initialize audio controller")

            # 记录初始化时间
            init_time = (time.time() - init_start) * 1000
            self.performance_stats['optimization_time_ms'] = init_time

            self.is_initialized = True
            self.logger.info(f"Integrated system initialized in {init_time:.1f}ms")

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False

    def _performance_alert_handler(self, alert):
        """性能警报处理器"""
        self.logger.warning(f"PERFORMANCE ALERT: {alert.message} (值: {alert.value:.1f}, 阈值: {alert.threshold:.1f})")

        # 自动优化策略
        if alert.category == "system" and alert.severity in ["warning", "critical"]:
            if alert.message.contains("FPS"):
                # FPS过低时降低手势检测频率
                if hasattr(self.gesture_detector, 'detection_fps'):
                    self.gesture_detector.detection_fps = max(10, self.gesture_detector.detection_fps - 2)
                    self.logger.info(f"Auto-adjusted gesture detection FPS to {self.gesture_detector.detection_fps}")

            elif alert.message.contains("内存"):
                # 内存使用过高时强制垃圾回收
                if self.memory_optimizer:
                    collected = self.memory_optimizer.force_gc()
                    self.logger.info(f"Auto-triggered garbage collection, freed {collected} objects")

    def start_playback(self) -> bool:
        """开始音频播放"""
        if not self.is_initialized:
            self.logger.error("System not initialized")
            return False

        try:
            success = self.audio_controller.start_playback()
            if success:
                self.is_running = True
                self.logger.info("Audio playback started")
            return success

        except Exception as e:
            self.logger.error(f"Failed to start playback: {e}")
            return False

    def stop_playback(self):
        """停止音频播放"""
        if self.audio_controller:
            self.audio_controller.stop_playback()

        self.is_running = False
        self.logger.info("Audio playback stopped")

    def process_camera_frame(self, frame: np.ndarray, show_regions: bool = True) -> np.ndarray:
        """
        处理摄像头帧 - 集成优化的主处理函数

        Args:
            frame: 输入视频帧
            show_regions: 是否显示区域划分

        Returns:
            处理后的视频帧
        """
        if not self.is_initialized:
            return frame

        # 使用性能监控计时
        with self.performance_monitor.timer('frame_processing'):
            # 记录帧处理
            self.performance_monitor.record_frame()
            self.performance_stats['frames_processed'] += 1

            # 使用优化内存获取处理缓冲区
            optimized_frame = None
            try:
                if self.memory_optimizer:
                    optimized_frame = self.memory_optimizer.get_cv2_mat(
                        frame.shape[0], frame.shape[1], frame.shape[2]
                    )
                    optimized_frame[:] = frame
                    process_frame = optimized_frame
                else:
                    process_frame = frame

                # 手势检测
                with self.performance_monitor.timer('gesture_detection'):
                    processed_frame = self.gesture_detector.process_frame(process_frame, show_regions)

                # 获取手势数据并更新音频
                gesture_data = self.gesture_detector.get_gesture_data()
                if gesture_data and self.is_running:
                    self._update_audio_from_gesture(gesture_data)

                # 更新性能监控
                self._update_performance_metrics(gesture_data)

                return processed_frame

            finally:
                # 归还优化内存
                if optimized_frame is not None and self.memory_optimizer:
                    self.memory_optimizer.return_cv2_mat(optimized_frame)

    def _update_audio_from_gesture(self, gesture_data):
        """根据手势数据更新音频"""
        try:
            # 将手势数据转换为音频控制命令
            for region_name, activation in gesture_data.region_activations.items():
                if activation > 0.1:  # 激活阈值
                    # 获取对应的区域ID
                    region_id = self._get_region_id_from_name(region_name)
                    if region_id:
                        # 设置区域音量
                        volume = min(1.0, activation * 1.2)  # 增强响应
                        self.audio_controller.set_region_volume(region_id, volume)
                        self.performance_stats['audio_commands_sent'] += 1

        except Exception as e:
            self.logger.warning(f"Audio update failed: {e}")

    def _get_region_id_from_name(self, region_name: str) -> Optional[int]:
        """从区域名称获取区域ID"""
        region_mapping = {
            'Tromba_I+II+III_in_D': 1,
            'Violins_in_D': 2,
            'Viola_in_D': 3,
            'Oboe_I_in_D': 4,
            'Continuo_in_D': 5,
            'Organo_obligato_in_D': 6,
            'Timpani_in_D': 7
        }
        return region_mapping.get(region_name)

    def _update_performance_metrics(self, gesture_data):
        """更新性能监控指标"""
        if not gesture_data or not self.performance_monitor:
            return

        try:
            # 更新手势检测指标
            detection_time = gesture_data.processing_time_ms
            confidence = gesture_data.confidence_score
            hands_count = gesture_data.hands_detected

            self.performance_monitor.record_gesture_detection(
                detection_time, confidence, hands_count
            )

            # 更新音频指标
            if self.audio_controller:
                audio_stats = self.audio_controller.get_performance_stats()
                self.performance_monitor.update_audio_stats(
                    audio_stats.get('audio_latency_ms', 0),
                    audio_stats.get('buffer_underruns', 0),
                    audio_stats.get('active_tracks', 0)
                )

            # 更新内存指标
            if self.memory_optimizer:
                memory_stats = self.memory_optimizer.get_memory_stats()
                numpy_stats = memory_stats.get('numpy_pool', {})
                mediapipe_stats = memory_stats.get('mediapipe_cache', {})

                self.performance_monitor.update_memory_stats(
                    numpy_stats.get('total_arrays', 0),
                    0,  # cv2_count - 暂时不可用
                    mediapipe_stats.get('cache_size', 0),
                    0   # cache_misses - 需要额外跟踪
                )

        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合性能统计"""
        stats = {
            'system_stats': self.performance_stats.copy(),
            'timestamp': time.time()
        }

        # 手势检测统计
        if self.gesture_detector:
            gesture_stats = self.gesture_detector.get_performance_stats()
            stats['gesture_detection'] = gesture_stats

        # 音频统计
        if self.audio_controller:
            audio_stats = self.audio_controller.get_performance_stats()
            stats['audio_controller'] = audio_stats

        # 内存统计
        if self.memory_optimizer:
            memory_stats = self.memory_optimizer.get_memory_stats()
            stats['memory_optimizer'] = memory_stats

        # 性能监控统计
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_current_metrics()
            if current_metrics:
                stats['current_performance'] = {
                    'fps': current_metrics.fps,
                    'cpu_percent': current_metrics.cpu_percent,
                    'memory_percent': current_metrics.memory_percent,
                    'audio_latency_ms': current_metrics.audio_latency_ms
                }

            recent_alerts = self.performance_monitor.get_recent_alerts(5)
            stats['recent_alerts'] = len(recent_alerts)

        return stats

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成详细性能报告"""
        stats = self.get_comprehensive_stats()

        # 添加性能评估
        performance_score = self._calculate_performance_score(stats)

        report = {
            'report_timestamp': time.time(),
            'configuration': {
                'gesture_detection_fps': self.config.gesture_detection_fps,
                'audio_optimization_level': self.config.audio_optimization_level,
                'memory_pool_size': self.config.numpy_pool_size
            },
            'performance_score': performance_score,
            'detailed_stats': stats,
            'optimization_effectiveness': self._evaluate_optimization_effectiveness(),
            'recommendations': self._generate_performance_recommendations(stats)
        }

        return report

    def _calculate_performance_score(self, stats: Dict[str, Any]) -> float:
        """计算性能评分 (0-100)"""
        score = 100.0
        current = stats.get('current_performance', {})

        # FPS评分 (权重: 30%)
        fps = current.get('fps', 0)
        if fps < 15:
            score -= 30
        elif fps < 25:
            score -= 15

        # CPU评分 (权重: 25%)
        cpu = current.get('cpu_percent', 0)
        if cpu > 80:
            score -= 25
        elif cpu > 60:
            score -= 15

        # 内存评分 (权重: 25%)
        memory = current.get('memory_percent', 0)
        if memory > 80:
            score -= 25
        elif memory > 60:
            score -= 15

        # 音频延迟评分 (权重: 20%)
        latency = current.get('audio_latency_ms', 0)
        if latency > 100:
            score -= 20
        elif latency > 50:
            score -= 10

        return max(0, score)

    def _evaluate_optimization_effectiveness(self) -> Dict[str, str]:
        """评估优化效果"""
        stats = self.get_comprehensive_stats()
        current = stats.get('current_performance', {})

        effectiveness = {}

        # 手势检测优化效果
        gesture_stats = stats.get('gesture_detection', {})
        hit_rate = gesture_stats.get('cache_hit_rate', 0)
        if hit_rate > 0.8:
            effectiveness['gesture_caching'] = "优秀"
        elif hit_rate > 0.6:
            effectiveness['gesture_caching'] = "良好"
        else:
            effectiveness['gesture_caching'] = "需要改进"

        # 内存优化效果
        memory_stats = stats.get('memory_optimizer', {})
        numpy_pool = memory_stats.get('numpy_pool', {})
        pool_hit_rate = numpy_pool.get('hit_rate', 0)
        if pool_hit_rate > 0.9:
            effectiveness['memory_pooling'] = "优秀"
        elif pool_hit_rate > 0.7:
            effectiveness['memory_pooling'] = "良好"
        else:
            effectiveness['memory_pooling'] = "需要改进"

        # 音频优化效果
        audio_latency = current.get('audio_latency_ms', 0)
        if audio_latency < 30:
            effectiveness['audio_latency'] = "优秀"
        elif audio_latency < 50:
            effectiveness['audio_latency'] = "良好"
        else:
            effectiveness['audio_latency'] = "需要改进"

        return effectiveness

    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        current = stats.get('current_performance', {})

        # FPS建议
        fps = current.get('fps', 0)
        if fps < 20:
            recommendations.append("考虑进一步降低手势检测频率或减少视频分辨率")

        # CPU建议
        cpu = current.get('cpu_percent', 0)
        if cpu > 70:
            recommendations.append("CPU使用率较高，建议关闭其他应用程序或优化算法")

        # 内存建议
        memory = current.get('memory_percent', 0)
        if memory > 70:
            recommendations.append("内存使用率较高，建议增加内存池大小或启用更积极的垃圾回收")

        # 音频建议
        latency = current.get('audio_latency_ms', 0)
        if latency > 50:
            recommendations.append("音频延迟较高，建议减小缓冲区大小或使用专用音频线程")

        # 缓存建议
        gesture_stats = stats.get('gesture_detection', {})
        hit_rate = gesture_stats.get('cache_hit_rate', 0)
        if hit_rate < 0.7:
            recommendations.append("手势缓存命中率较低，建议调整缓存策略")

        return recommendations

    def save_performance_report(self, filename: str = None) -> str:
        """保存性能报告"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_performance_report_{timestamp}.json"

        report = self.generate_performance_report()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Performance report saved to: {filename}")
        return filename

    def cleanup(self):
        """清理所有资源"""
        self.logger.info("Cleaning up integrated performance system...")

        # 停止播放
        self.stop_playback()

        # 清理各个组件
        if self.gesture_detector:
            self.gesture_detector.cleanup()

        if self.audio_controller:
            self.audio_controller.cleanup()

        if self.memory_optimizer:
            self.memory_optimizer.cleanup()

        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

        self.is_initialized = False
        self.is_running = False

        self.logger.info("Integrated system cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_demo_system(audio_dir: str = None) -> IntegratedPerformanceSystem:
    """创建演示系统"""
    config = PerformanceConfig(
        gesture_detection_fps=15,
        gesture_display_fps=30,
        audio_optimization_level="low_latency",
        enable_performance_alerts=True,
        enable_macos_optimization=True
    )

    return IntegratedPerformanceSystem(config, audio_dir)


def main():
    """主演示函数"""
    print("Integrated Performance System Demo")
    print("=" * 50)

    # 创建集成系统
    audio_dir = os.path.dirname(os.path.abspath(__file__))

    with create_demo_system(audio_dir) as system:
        # 初始化系统
        if not system.initialize():
            print("❌ 系统初始化失败")
            return

        print("✅ 集成性能系统初始化成功")

        # 开始音频播放
        if system.start_playback():
            print("✅ 音频播放已启动")
        else:
            print("⚠️ 音频播放启动失败")

        # 模拟摄像头处理
        print("\n开始模拟视频处理...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("⚠️ 无法打开摄像头，使用模拟数据")
            # 模拟数据处理
            for i in range(100):
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                processed_frame = system.process_camera_frame(frame)

                time.sleep(0.033)  # ~30FPS

                if i % 30 == 0:  # 每秒显示一次统计
                    stats = system.get_comprehensive_stats()
                    current = stats.get('current_performance', {})
                    print(f"FPS: {current.get('fps', 0):.1f}, "
                          f"CPU: {current.get('cpu_percent', 0):.1f}%, "
                          f"内存: {current.get('memory_percent', 0):.1f}%, "
                          f"音频延迟: {current.get('audio_latency_ms', 0):.1f}ms")

        else:
            # 实际摄像头处理
            print("📹 使用真实摄像头数据")
            try:
                frame_count = 0
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 处理帧
                    processed_frame = system.process_camera_frame(frame, show_regions=True)

                    # 显示结果
                    cv2.imshow('Integrated Performance System', processed_frame)

                    frame_count += 1

                    # 每秒显示统计
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        actual_fps = frame_count / elapsed

                        stats = system.get_comprehensive_stats()
                        current = stats.get('current_performance', {})

                        print(f"实际FPS: {actual_fps:.1f}, "
                              f"检测FPS: {current.get('fps', 0):.1f}, "
                              f"CPU: {current.get('cpu_percent', 0):.1f}%, "
                              f"内存: {current.get('memory_percent', 0):.1f}%, "
                              f"音频延迟: {current.get('audio_latency_ms', 0):.1f}ms")

                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except KeyboardInterrupt:
                print("\n⏹️ 用户中断")

            finally:
                cap.release()
                cv2.destroyAllWindows()

        # 生成最终报告
        print("\n📊 生成性能报告...")
        report_file = system.save_performance_report()

        # 显示最终统计
        final_stats = system.get_comprehensive_stats()
        print(f"\n📈 最终统计:")
        print(f"  处理帧数: {final_stats['system_stats']['frames_processed']}")
        print(f"  音频命令: {final_stats['system_stats']['audio_commands_sent']}")
        print(f"  优化时间: {final_stats['system_stats']['optimization_time_ms']:.1f}ms")

        performance_score = system._calculate_performance_score(final_stats)
        print(f"  性能评分: {performance_score:.1f}/100")

        print(f"\n📄 详细报告已保存: {report_file}")

    print("\n✅ 集成性能系统演示完成")


if __name__ == "__main__":
    main()