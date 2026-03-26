"""
Hand Gesture Particle Helix - Core Module
手势粒子螺旋 - 核心模块

This package provides unified imports for:
- Hand gesture detection and recognition
- Audio management and analysis
- Visualization and rendering
- Gesture-to-visualization bridging
- Performance monitoring

Usage:
    from core.gesture import GestureDetector
    from core.audio import RealisticAudioManager
    from core.visualization import RenderEngine
    from core.bridge import GestureParametricBridge
    from core.performance import ComprehensivePerformanceMonitor

    # Or import directly:
    from core import GestureDetector, RealisticAudioManager
"""

__version__ = "2.0.0"
__author__ = "Hand Gesture Particle Helix Team"

# 便捷导入（延迟导入以避免循环依赖）
from .gesture import GestureDetector, GestureToRadiusMapper, HandState
from .audio import RealisticAudioManager, AudioSpectrumAnalyzer
from .visualization import ParametricEquationRenderer, RenderEngine, ParticleSystem, ParticleSphereSystem
from .bridge import GestureParametricBridge, HandAssignment
from .performance import ComprehensivePerformanceMonitor

__all__ = [
    # Gesture
    'GestureDetector',
    'GestureToRadiusMapper',
    'HandState',
    # Audio
    'RealisticAudioManager',
    'AudioSpectrumAnalyzer',
    # Visualization
    'ParametricEquationRenderer',
    'RenderEngine',
    'ParticleSystem',
    'ParticleSphereSystem',
    # Bridge
    'GestureParametricBridge',
    'HandAssignment',
    # Performance
    'ComprehensivePerformanceMonitor',
]
