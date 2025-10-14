#!/usr/bin/env python3
"""
Real-time Performance Monitoring and Optimization System

Comprehensive performance monitoring with:
- Real-time metrics tracking (FPS, memory, CPU, GPU)
- Adaptive quality scaling based on system performance
- Performance profiler for bottleneck identification
- Memory leak detection and prevention
- Hardware-specific optimization recommendations
- Live performance dashboard

Optimized for maintaining 30+ FPS performance across different hardware configurations.
"""

import time
import threading
import queue
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import gc
import tracemalloc
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
import sys
import platform


try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available. GPU monitoring disabled.")

try:
    import py3nvml.py3nvml as nvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


class PerformanceLevel(Enum):
    """System performance levels for adaptive quality."""
    CRITICAL = "critical"  # < 15 FPS
    LOW = "low"          # 15-20 FPS
    MEDIUM = "medium"    # 20-30 FPS
    HIGH = "high"        # 30-60 FPS
    ULTRA = "ultra"      # > 60 FPS


class OptimizationStrategy(Enum):
    """Different optimization strategies."""
    REDUCE_POINTS = "reduce_points"
    LOWER_QUALITY = "lower_quality"
    DISABLE_EFFECTS = "disable_effects"
    INCREASE_LOD = "increase_lod"
    REDUCE_UPDATE_RATE = "reduce_update_rate"
    PARALLEL_PROCESSING = "parallel_processing"
    MEMORY_OPTIMIZATION = "memory_optimization"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    fps: float
    frame_time_ms: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_io_mb_s: float = 0.0
    network_io_mb_s: float = 0.0
    thread_count: int = 0
    points_rendered: int = 0
    points_culled: int = 0
    buffer_utilization: float = 0.0


@dataclass
class SystemCapabilities:
    """System hardware capabilities assessment."""
    cpu_cores: int
    cpu_frequency_mhz: float
    memory_total_gb: float
    gpu_name: str = "Unknown"
    gpu_memory_gb: float = 0.0
    platform_name: str = ""
    performance_score: float = 0.0  # Normalized 0-1 score
    recommended_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert/warning."""
    timestamp: float
    level: str  # 'info', 'warning', 'critical'
    category: str  # 'fps', 'memory', 'cpu', 'gpu'
    message: str
    suggested_action: str
    auto_applied: bool = False


class MemoryTracker:
    """Memory usage tracking and leak detection."""
    
    def __init__(self, track_objects: bool = True):
        """Initialize memory tracker."""
        self.track_objects = track_objects
        self.baseline_memory = 0
        self.peak_memory = 0
        self.memory_history = deque(maxlen=100)
        self.object_counts = {}
        
        if track_objects:
            tracemalloc.start()
    
    def update(self) -> Dict[str, Any]:
        """Update memory metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        current_memory = memory_info.rss / (1024 * 1024)  # MB
        self.memory_history.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if self.baseline_memory == 0:
            self.baseline_memory = current_memory
        
        # Track object counts
        if self.track_objects:
            import gc
            self.object_counts = {
                'total_objects': len(gc.get_objects()),
                'list_objects': sum(1 for obj in gc.get_objects() if isinstance(obj, list)),
                'dict_objects': sum(1 for obj in gc.get_objects() if isinstance(obj, dict)),
                'numpy_arrays': sum(1 for obj in gc.get_objects() if hasattr(obj, 'dtype'))
            }
        
        return {
            'current_mb': current_memory,
            'peak_mb': self.peak_memory,
            'growth_mb': current_memory - self.baseline_memory,
            'trend': self._calculate_memory_trend(),
            'object_counts': self.object_counts.copy()
        }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self.memory_history) < 10:
            return "insufficient_data"
        
        recent = list(self.memory_history)[-10:]
        older = list(self.memory_history)[-20:-10] if len(self.memory_history) >= 20 else recent[:5]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        growth_rate = (recent_avg - older_avg) / older_avg
        
        if growth_rate > 0.1:  # 10% growth
            return "increasing"
        elif growth_rate < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def detect_leak(self) -> Optional[str]:
        """Detect potential memory leaks."""
        if len(self.memory_history) < 30:
            return None
        
        # Check for sustained memory growth
        recent_trend = self._calculate_memory_trend()
        if recent_trend == "increasing":
            growth = self.memory_history[-1] - self.memory_history[-30]
            if growth > 100:  # 100MB growth over 30 measurements
                return f"Potential memory leak detected: {growth:.1f}MB growth"
        
        return None
    
    def cleanup_recommendations(self) -> List[str]:
        """Get memory cleanup recommendations."""
        recommendations = []
        
        if self.object_counts.get('list_objects', 0) > 10000:
            recommendations.append("Consider using numpy arrays instead of large lists")
        
        if self.object_counts.get('numpy_arrays', 0) > 1000:
            recommendations.append("Review numpy array usage and cleanup unused arrays")
        
        leak_warning = self.detect_leak()
        if leak_warning:
            recommendations.append(leak_warning)
            recommendations.append("Run garbage collection and review object lifecycles")
        
        return recommendations


class GPUMonitor:
    """GPU performance monitoring."""
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.gpu_available = GPU_AVAILABLE
        self.nvidia_available = NVIDIA_AVAILABLE
        self.gpu_info = None
        
        if self.nvidia_available:
            try:
                nvml.nvmlInit()
                self.gpu_info = "NVIDIA GPU detected"
            except:
                self.nvidia_available = False
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics."""
        metrics = {
            'gpu_percent': 0.0,
            'gpu_memory_mb': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_temperature': 0.0,
            'gpu_power_watts': 0.0
        }
        
        if not self.gpu_available:
            return metrics
        
        try:
            if self.nvidia_available:
                # Use NVIDIA ML for detailed metrics
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_percent'] = util.gpu
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['gpu_memory_mb'] = mem_info.used / (1024 * 1024)
                metrics['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                metrics['gpu_temperature'] = temp
                
                # Power usage
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics['gpu_power_watts'] = power
                except:
                    pass  # Power monitoring not available on all GPUs
                    
            else:
                # Fallback to GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_percent'] = gpu.load * 100
                    metrics['gpu_memory_mb'] = gpu.memoryUsed
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                    metrics['gpu_temperature'] = gpu.temperature
        
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        return metrics


class PerformanceProfiler:
    """Code profiler for identifying performance bottlenecks."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiler = None
        self.profiling_active = False
        self.profile_results = {}
        
    def start_profiling(self) -> None:
        """Start performance profiling."""
        if not self.profiling_active:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.profiling_active = True
            print("Performance profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        if self.profiling_active and self.profiler:
            self.profiler.disable()
            self.profiling_active = False
            
            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            profile_output = s.getvalue()
            
            # Extract top bottlenecks
            lines = profile_output.split('\n')
            bottlenecks = []
            
            for line in lines[5:15]:  # Skip header, take top 10
                if line.strip() and 'function calls' not in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        bottlenecks.append({
                            'function': ' '.join(parts[5:]),
                            'calls': parts[0],
                            'total_time': parts[3],
                            'per_call': parts[4]
                        })
            
            self.profile_results = {
                'bottlenecks': bottlenecks,
                'raw_output': profile_output
            }
            
            print("Performance profiling stopped")
            return self.profile_results
        
        return {}
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profiling."""
        if not self.profile_results:
            return ["Run profiling first to get recommendations"]
        
        recommendations = []
        bottlenecks = self.profile_results.get('bottlenecks', [])
        
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            func_name = bottleneck['function']
            
            if 'numpy' in func_name.lower():
                recommendations.append(f"Optimize numpy operations in: {func_name}")
            elif 'matplotlib' in func_name.lower():
                recommendations.append(f"Consider reducing plot complexity in: {func_name}")
            elif 'loop' in func_name.lower() or 'for' in func_name.lower():
                recommendations.append(f"Vectorize or parallelize loop: {func_name}")
            else:
                recommendations.append(f"Review performance of: {func_name}")
        
        return recommendations


class AdaptiveQualityController:
    """Adaptive quality control based on performance metrics."""
    
    def __init__(self, target_fps: float = 30.0, tolerance: float = 0.2):
        """
        Initialize adaptive quality controller.
        
        Args:
            target_fps: Target FPS to maintain
            tolerance: Tolerance for FPS variation (0.2 = 20%)
        """
        self.target_fps = target_fps
        self.tolerance = tolerance
        self.current_level = PerformanceLevel.HIGH
        self.quality_settings = self._create_quality_settings()
        self.adjustment_cooldown = 2.0  # Seconds between adjustments
        self.last_adjustment = 0.0
        self.fps_history = deque(maxlen=30)
        
    def _create_quality_settings(self) -> Dict[PerformanceLevel, Dict[str, Any]]:
        """Create quality settings for each performance level."""
        return {
            PerformanceLevel.CRITICAL: {
                'max_points': 200,
                'update_frequency_hz': 10,
                'lod_level': 'low',
                'render_mode': 'points',
                'disable_effects': True,
                'culling_distance': 5.0
            },
            PerformanceLevel.LOW: {
                'max_points': 500,
                'update_frequency_hz': 15,
                'lod_level': 'low',
                'render_mode': 'solid',
                'disable_effects': True,
                'culling_distance': 7.0
            },
            PerformanceLevel.MEDIUM: {
                'max_points': 1000,
                'update_frequency_hz': 20,
                'lod_level': 'medium',
                'render_mode': 'fade',
                'disable_effects': False,
                'culling_distance': 10.0
            },
            PerformanceLevel.HIGH: {
                'max_points': 2000,
                'update_frequency_hz': 30,
                'lod_level': 'high',
                'render_mode': 'fade',
                'disable_effects': False,
                'culling_distance': 15.0
            },
            PerformanceLevel.ULTRA: {
                'max_points': 5000,
                'update_frequency_hz': 60,
                'lod_level': 'ultra',
                'render_mode': 'particles',
                'disable_effects': False,
                'culling_distance': 20.0
            }
        }
    
    def update_performance(self, current_fps: float) -> Tuple[PerformanceLevel, Dict[str, Any], List[str]]:
        """
        Update performance level based on current FPS.
        
        Returns:
            (new_level, settings, actions_taken)
        """
        self.fps_history.append(current_fps)
        
        # Calculate average FPS over recent history
        if len(self.fps_history) < 5:
            return self.current_level, self.quality_settings[self.current_level], []
        
        avg_fps = np.mean(list(self.fps_history)[-10:])
        
        # Check if adjustment is needed and cooldown has passed
        current_time = time.time()
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return self.current_level, self.quality_settings[self.current_level], []
        
        # Determine target performance level
        target_level = self._determine_performance_level(avg_fps)
        actions_taken = []
        
        if target_level != self.current_level:
            actions_taken.append(f"Performance level changed: {self.current_level.value} -> {target_level.value}")
            self.current_level = target_level
            self.last_adjustment = current_time
            
            # Additional specific optimizations
            if target_level in [PerformanceLevel.CRITICAL, PerformanceLevel.LOW]:
                actions_taken.append("Applied emergency optimizations")
                actions_taken.append("Reduced visual effects")
                actions_taken.append("Increased LOD culling")
        
        return self.current_level, self.quality_settings[self.current_level], actions_taken
    
    def _determine_performance_level(self, avg_fps: float) -> PerformanceLevel:
        """Determine appropriate performance level based on FPS."""
        min_fps = self.target_fps * (1 - self.tolerance)
        max_fps = self.target_fps * (1 + self.tolerance)
        
        if avg_fps < 15:
            return PerformanceLevel.CRITICAL
        elif avg_fps < 20:
            return PerformanceLevel.LOW
        elif avg_fps < min_fps:
            return PerformanceLevel.MEDIUM
        elif avg_fps <= max_fps:
            return PerformanceLevel.HIGH
        else:
            return PerformanceLevel.ULTRA


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize performance dashboard."""
        self.update_interval = update_interval
        self.running = False
        
        # Create subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='black')
        self.fig.suptitle('Real-time Performance Monitor', color='white', fontsize=14)
        
        # Configure axes
        for row in self.axes:
            for ax in row:
                ax.set_facecolor('black')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
        
        # Data storage
        self.max_points = 100
        self.time_data = deque(maxlen=self.max_points)
        self.fps_data = deque(maxlen=self.max_points)
        self.cpu_data = deque(maxlen=self.max_points)
        self.memory_data = deque(maxlen=self.max_points)
        self.gpu_data = deque(maxlen=self.max_points)
        self.frame_time_data = deque(maxlen=self.max_points)
        
        # Initialize plots
        self._initialize_plots()
        
        # Performance components
        self.memory_tracker = MemoryTracker()
        self.gpu_monitor = GPUMonitor()
        self.start_time = time.time()
        
    def _initialize_plots(self) -> None:
        """Initialize dashboard plots."""
        # FPS plot
        self.axes[0, 0].set_title('FPS', color='white')
        self.axes[0, 0].set_ylabel('Frames/sec', color='white')
        self.fps_line, = self.axes[0, 0].plot([], [], 'g-', linewidth=2)
        self.fps_target_line = self.axes[0, 0].axhline(y=30, color='yellow', linestyle='--', alpha=0.7)
        
        # CPU usage
        self.axes[0, 1].set_title('CPU Usage', color='white')
        self.axes[0, 1].set_ylabel('Percent', color='white')
        self.cpu_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2)
        
        # Memory usage
        self.axes[0, 2].set_title('Memory Usage', color='white')
        self.axes[0, 2].set_ylabel('MB', color='white')
        self.memory_line, = self.axes[0, 2].plot([], [], 'b-', linewidth=2)
        
        # Frame time
        self.axes[1, 0].set_title('Frame Time', color='white')
        self.axes[1, 0].set_ylabel('Milliseconds', color='white')
        self.frame_time_line, = self.axes[1, 0].plot([], [], 'orange', linewidth=2)
        self.frame_time_target = self.axes[1, 0].axhline(y=33.33, color='yellow', linestyle='--', alpha=0.7)
        
        # GPU usage
        self.axes[1, 1].set_title('GPU Usage', color='white')
        self.axes[1, 1].set_ylabel('Percent', color='white')
        self.gpu_line, = self.axes[1, 1].plot([], [], 'm-', linewidth=2)
        
        # System info (text)
        self.axes[1, 2].set_title('System Status', color='white')
        self.axes[1, 2].axis('off')
        self.status_text = self.axes[1, 2].text(0.1, 0.9, '', transform=self.axes[1, 2].transAxes,
                                              color='white', fontsize=10, verticalalignment='top',
                                              fontfamily='monospace')
    
    def update_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update dashboard with new metrics."""
        current_time = time.time() - self.start_time
        
        # Add data points
        self.time_data.append(current_time)
        self.fps_data.append(metrics.fps)
        self.cpu_data.append(metrics.cpu_percent)
        self.memory_data.append(metrics.memory_mb)
        self.gpu_data.append(metrics.gpu_percent)
        self.frame_time_data.append(metrics.frame_time_ms)
        
        # Update plots
        time_array = list(self.time_data)
        
        # FPS
        self.fps_line.set_data(time_array, list(self.fps_data))
        self.axes[0, 0].set_xlim(max(0, current_time - 60), current_time + 2)
        if self.fps_data:
            self.axes[0, 0].set_ylim(0, max(60, max(self.fps_data) * 1.1))
        
        # CPU
        self.cpu_line.set_data(time_array, list(self.cpu_data))
        self.axes[0, 1].set_xlim(max(0, current_time - 60), current_time + 2)
        self.axes[0, 1].set_ylim(0, 100)
        
        # Memory
        self.memory_line.set_data(time_array, list(self.memory_data))
        self.axes[0, 2].set_xlim(max(0, current_time - 60), current_time + 2)
        if self.memory_data:
            self.axes[0, 2].set_ylim(0, max(self.memory_data) * 1.1)
        
        # Frame time
        self.frame_time_line.set_data(time_array, list(self.frame_time_data))
        self.axes[1, 0].set_xlim(max(0, current_time - 60), current_time + 2)
        if self.frame_time_data:
            self.axes[1, 0].set_ylim(0, max(50, max(self.frame_time_data) * 1.1))
        
        # GPU
        self.gpu_line.set_data(time_array, list(self.gpu_data))
        self.axes[1, 1].set_xlim(max(0, current_time - 60), current_time + 2)
        self.axes[1, 1].set_ylim(0, 100)
        
        # Status text
        status_text = f"""FPS: {metrics.fps:.1f}
CPU: {metrics.cpu_percent:.1f}%
Memory: {metrics.memory_mb:.0f} MB
GPU: {metrics.gpu_percent:.1f}%
Points: {metrics.points_rendered}
Culled: {metrics.points_culled}
Threads: {metrics.thread_count}"""
        
        self.status_text.set_text(status_text)
    
    def show(self) -> None:
        """Show the dashboard."""
        plt.tight_layout()
        plt.show()


class ComprehensivePerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, target_fps: float = 30.0, enable_dashboard: bool = True):
        """Initialize comprehensive performance monitor."""
        self.target_fps = target_fps
        self.enable_dashboard = enable_dashboard
        
        # Components
        self.memory_tracker = MemoryTracker()
        self.gpu_monitor = GPUMonitor()
        self.profiler = PerformanceProfiler()
        self.quality_controller = AdaptiveQualityController(target_fps)
        self.dashboard = PerformanceDashboard() if enable_dashboard else None
        
        # Metrics tracking
        self.current_metrics = PerformanceMetrics(
            timestamp=time.time(), fps=0.0, frame_time_ms=0.0,
            cpu_percent=0.0, memory_mb=0.0, memory_percent=0.0
        )
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=50)
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)  # Track last 60 frame times
        self.last_frame_time = time.time()
        
        # System capabilities
        self.system_caps = self._assess_system_capabilities()
        
    def _assess_system_capabilities(self) -> SystemCapabilities:
        """Assess system hardware capabilities."""
        caps = SystemCapabilities(
            cpu_cores=psutil.cpu_count(),
            cpu_frequency_mhz=psutil.cpu_freq().current if psutil.cpu_freq() else 2400,
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            platform_name=platform.platform()
        )
        
        # GPU assessment
        if self.gpu_monitor.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    caps.gpu_name = gpu.name
                    caps.gpu_memory_gb = gpu.memoryTotal / 1024
            except:
                pass
        
        # Calculate performance score
        score = 0.0
        score += min(caps.cpu_cores / 8, 1.0) * 0.3  # CPU cores (max 8)
        score += min(caps.cpu_frequency_mhz / 3000, 1.0) * 0.2  # CPU frequency
        score += min(caps.memory_total_gb / 16, 1.0) * 0.2  # Memory
        score += min(caps.gpu_memory_gb / 8, 1.0) * 0.3  # GPU memory
        
        caps.performance_score = score
        
        # Recommended settings based on capabilities
        if score > 0.8:
            caps.recommended_settings = {
                'max_points': 5000,
                'quality_level': 'ultra',
                'enable_effects': True,
                'target_fps': 60
            }
        elif score > 0.6:
            caps.recommended_settings = {
                'max_points': 2000,
                'quality_level': 'high',
                'enable_effects': True,
                'target_fps': 30
            }
        elif score > 0.4:
            caps.recommended_settings = {
                'max_points': 1000,
                'quality_level': 'medium',
                'enable_effects': False,
                'target_fps': 30
            }
        else:
            caps.recommended_settings = {
                'max_points': 500,
                'quality_level': 'low',
                'enable_effects': False,
                'target_fps': 20
            }
        
        return caps
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="PerformanceMonitor"
            )
            self.monitoring_thread.start()
            print("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        with self.lock:
            self.running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=2.0)
            
            print("Performance monitoring stopped")
    
    def update_frame(self, points_rendered: int = 0, points_culled: int = 0) -> None:
        """Update frame timing and metrics."""
        current_time = time.time()
        frame_time = (current_time - self.last_frame_time) * 1000  # Convert to ms
        
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            avg_frame_time = np.mean(list(self.frame_times)[-30:])  # Use last 30 frames
            fps = 1000.0 / max(avg_frame_time, 1.0)
        else:
            fps = 0.0
        
        # Update current metrics
        with self.lock:
            self.current_metrics.timestamp = current_time
            self.current_metrics.fps = fps
            self.current_metrics.frame_time_ms = frame_time
            self.current_metrics.points_rendered = points_rendered
            self.current_metrics.points_culled = points_culled
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_metrics(self.current_metrics)
                
                # Store metrics history
                self.metrics_history.append(self.current_metrics)
                
                # Sleep
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        with self.lock:
            # CPU and memory
            self.current_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.current_metrics.memory_mb = memory.used / (1024 * 1024)
            self.current_metrics.memory_percent = memory.percent
            
            # Thread count
            self.current_metrics.thread_count = threading.active_count()
            
            # GPU metrics
            gpu_metrics = self.gpu_monitor.get_gpu_metrics()
            self.current_metrics.gpu_percent = gpu_metrics['gpu_percent']
            self.current_metrics.gpu_memory_mb = gpu_metrics['gpu_memory_mb']
            self.current_metrics.gpu_memory_percent = gpu_metrics['gpu_memory_percent']
            
            # Memory tracking
            memory_info = self.memory_tracker.update()
            # Additional memory metrics could be incorporated here
    
    def _check_alerts(self) -> None:
        """Check for performance alerts."""
        alerts = []
        
        # FPS alerts
        if self.current_metrics.fps > 0:
            if self.current_metrics.fps < 15:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    level='critical',
                    category='fps',
                    message=f'Critical FPS: {self.current_metrics.fps:.1f}',
                    suggested_action='Reduce quality settings immediately'
                ))
            elif self.current_metrics.fps < self.target_fps * 0.8:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    level='warning',
                    category='fps',
                    message=f'Low FPS: {self.current_metrics.fps:.1f}',
                    suggested_action='Consider reducing visual complexity'
                ))
        
        # Memory alerts
        if self.current_metrics.memory_percent > 90:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                level='critical',
                category='memory',
                message=f'High memory usage: {self.current_metrics.memory_percent:.1f}%',
                suggested_action='Free memory or reduce buffer sizes'
            ))
        
        # CPU alerts
        if self.current_metrics.cpu_percent > 90:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                level='warning',
                category='cpu',
                message=f'High CPU usage: {self.current_metrics.cpu_percent:.1f}%',
                suggested_action='Optimize processing algorithms'
            ))
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            print(f"Performance Alert [{alert.level.upper()}]: {alert.message}")
    
    def get_adaptive_quality_settings(self) -> Tuple[Dict[str, Any], List[str]]:
        """Get adaptive quality settings based on current performance."""
        level, settings, actions = self.quality_controller.update_performance(self.current_metrics.fps)
        return settings, actions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-30:]  # Last 30 measurements
        
        return {
            'system_capabilities': asdict(self.system_caps),
            'current_performance': asdict(self.current_metrics),
            'averages': {
                'fps': np.mean([m.fps for m in recent_metrics if m.fps > 0]),
                'cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                'memory_mb': np.mean([m.memory_mb for m in recent_metrics]),
                'frame_time_ms': np.mean([m.frame_time_ms for m in recent_metrics])
            },
            'recommendations': self._get_optimization_recommendations(),
            'recent_alerts': [asdict(alert) for alert in list(self.alerts)[-10:]],
            'memory_analysis': self.memory_tracker.cleanup_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance."""
        recommendations = []
        
        if self.current_metrics.fps < self.target_fps * 0.8:
            recommendations.append("Reduce trajectory point count")
            recommendations.append("Lower rendering quality")
            recommendations.append("Enable aggressive culling")
        
        if self.current_metrics.memory_percent > 80:
            recommendations.extend(self.memory_tracker.cleanup_recommendations())
            recommendations.append("Reduce buffer sizes")
            recommendations.append("Implement more aggressive garbage collection")
        
        if self.current_metrics.cpu_percent > 80:
            recommendations.append("Enable multi-threading for processing")
            recommendations.append("Optimize algorithms for vectorization")
            recommendations.append("Reduce update frequency")
        
        if self.current_metrics.gpu_percent > 90:
            recommendations.append("Reduce shader complexity")
            recommendations.append("Lower texture resolution")
            recommendations.append("Optimize GPU memory usage")
        
        return recommendations
    
    def show_dashboard(self) -> None:
        """Show performance dashboard."""
        if self.dashboard:
            self.dashboard.show()
    
    def start_profiling(self) -> None:
        """Start performance profiling."""
        self.profiler.start_profiling()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and get results."""
        return self.profiler.stop_profiling()


# Example usage and testing
if __name__ == "__main__":
    print("Performance Monitoring System Test")
    
    # Create monitor
    monitor = ComprehensivePerformanceMonitor(target_fps=30.0, enable_dashboard=True)
    
    # Print system capabilities
    caps = monitor.system_caps
    print(f"\nSystem Assessment:")
    print(f"CPU: {caps.cpu_cores} cores @ {caps.cpu_frequency_mhz:.0f} MHz")
    print(f"Memory: {caps.memory_total_gb:.1f} GB")
    print(f"GPU: {caps.gpu_name} ({caps.gpu_memory_gb:.1f} GB)")
    print(f"Performance Score: {caps.performance_score:.2f}")
    print(f"Recommended Settings: {caps.recommended_settings}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate application loop
        print("\nSimulating application performance...")
        for i in range(100):
            # Simulate frame rendering
            points_rendered = np.random.randint(500, 2000)
            points_culled = np.random.randint(0, 500)
            
            # Update frame metrics
            monitor.update_frame(points_rendered, points_culled)
            
            # Get adaptive settings periodically
            if i % 20 == 0:
                settings, actions = monitor.get_adaptive_quality_settings()
                if actions:
                    print(f"Adaptive actions: {actions}")
            
            # Simulate variable workload
            time.sleep(0.02 + 0.03 * np.random.random())  # 20-50ms frame times
            
        # Get performance report
        report = monitor.get_performance_report()
        print(f"\nPerformance Report:")
        print(f"Average FPS: {report['averages']['fps']:.1f}")
        print(f"Average CPU: {report['averages']['cpu_percent']:.1f}%")
        print(f"Average Memory: {report['averages']['memory_mb']:.0f} MB")
        
        print(f"\nOptimization Recommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        
        # Show dashboard (commented out for automated testing)
        # monitor.show_dashboard()
        
    finally:
        monitor.stop_monitoring()
    
    print("Performance monitoring test completed!")