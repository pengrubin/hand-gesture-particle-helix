"""
Visualization Module
可视化核心模块

实际实现在根目录，这里提供统一的导入入口
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from parametric_equation_renderer import ParametricEquationRenderer
from render_engine import RenderEngine
from particle_system import (
    ParticleSystem,
    update_particle_system,
    get_particle_data_for_gpu,
    get_active_particle_count,
    generate_vertex_shader,
    generate_fragment_shader,
)
from particle_sphere_system import ParticleSphereSystem

__all__ = [
    'ParametricEquationRenderer',
    'RenderEngine',
    'ParticleSystem',
    'ParticleSphereSystem',
    'update_particle_system',
    'get_particle_data_for_gpu',
    'get_active_particle_count',
    'generate_vertex_shader',
    'generate_fragment_shader',
]
