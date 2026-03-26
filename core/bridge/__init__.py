"""
Bridge Module - 连接手势检测与可视化

实际实现在根目录，这里提供统一的导入入口
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from gesture_parametric_bridge import (
    GestureParametricBridge,
    HandAssignment,
)

__all__ = [
    'GestureParametricBridge',
    'HandAssignment',
]
