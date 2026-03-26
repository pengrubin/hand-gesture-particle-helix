"""
Audio Management Module
音频管理核心模块

实际实现在根目录，这里提供统一的导入入口
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from realistic_audio_manager import (
    RealisticAudioManager,
    AudioManager,
    AdvancedAudioManager,
    initialize_audio_manager,
    update_audio_from_gesture,
    get_audio_params,
    get_audio_status,
    stop_audio,
    cleanup_audio_manager,
)

from audio_spectrum_analyzer import AudioSpectrumAnalyzer

__all__ = [
    'RealisticAudioManager',
    'AudioManager',
    'AdvancedAudioManager',
    'AudioSpectrumAnalyzer',
    'initialize_audio_manager',
    'update_audio_from_gesture',
    'get_audio_params',
    'get_audio_status',
    'stop_audio',
    'cleanup_audio_manager',
]
