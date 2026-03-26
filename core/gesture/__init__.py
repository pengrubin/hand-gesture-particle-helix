"""
Hand Gesture Detection Module
手势检测核心模块

实际实现在根目录，这里提供统一的导入入口
"""

import sys
from pathlib import Path

# 确保根目录在 Python 路径中
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 从根目录模块导入
from gesture_detector import (
    GestureDetector,
    CrossPlatformGestureDetector,
    HandGestureDetector,
    OptimizedHandGestureDetector,
    PerformanceMonitor,
    GestureCache,
    GestureStabilizer,
    UserStateManager,
    ConductorGestureAnalyzer,
    FrameBuffer,
    # TouchDesigner 接口函数
    get_gesture_data,
    get_conductor_commands,
    get_active_voices,
    get_region_info,
    get_performance_stats,
    initialize_detector,
    process_camera_frame,
    set_region_bounds,
    update_detector_config,
    cleanup_detector,
)

from gesture_radius_mapper import (
    GestureToRadiusMapper,
    HandState,
)

__all__ = [
    'GestureDetector',
    'CrossPlatformGestureDetector',
    'HandGestureDetector',
    'OptimizedHandGestureDetector',
    'GestureToRadiusMapper',
    'HandState',
    'PerformanceMonitor',
    'GestureCache',
    'GestureStabilizer',
    'UserStateManager',
    'ConductorGestureAnalyzer',
    'FrameBuffer',
    'get_gesture_data',
    'get_conductor_commands',
    'get_active_voices',
    'get_region_info',
    'get_performance_stats',
    'initialize_detector',
    'process_camera_frame',
    'set_region_bounds',
    'update_detector_config',
    'cleanup_detector',
]
