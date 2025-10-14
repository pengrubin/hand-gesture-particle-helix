#!/usr/bin/env python3
"""
Real-time Trajectory Visualization Engine

High-performance GPU-optimized trajectory visualization with:
- Efficient point culling and LOD systems
- GPU-accelerated line drawing with OpenGL integration
- Variable trail lengths and fading effects
- Multiple rendering modes (solid, fade, glow, particle-based)
- Interactive trajectory playback controls

Optimized for real-time display of thousands of trajectory points.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PointCollection
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import queue
import warnings

# Optional OpenGL support for advanced GPU acceleration
try:
    import OpenGL.GL as gl
    import OpenGL.arrays.vbo as vbo
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL not available. Using matplotlib-based rendering.")

# Suppress matplotlib warnings for performance
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class RenderMode(Enum):
    """Different rendering modes for trajectories."""
    SOLID_LINE = "solid"
    FADE_TRAIL = "fade"
    GLOW_EFFECT = "glow"
    PARTICLE_BASED = "particles"
    POINT_CLOUD = "points"
    VECTOR_FIELD = "vectors"


class LODLevel(Enum):
    """Level of detail settings for performance scaling."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4


@dataclass
class RenderingSettings:
    """Configuration for trajectory rendering."""
    render_mode: RenderMode = RenderMode.FADE_TRAIL
    lod_level: LODLevel = LODLevel.HIGH
    max_visible_points: int = 2000
    trail_length: int = 500
    point_size: float = 2.0
    line_width: float = 1.5
    alpha_fade_factor: float = 0.95
    color_by_velocity: bool = True
    color_by_time: bool = False
    color_by_gesture: bool = False
    enable_culling: bool = True
    culling_distance: float = 10.0
    update_frequency_hz: float = 30.0


@dataclass
class TrajectoryRenderData:
    """Optimized data structure for trajectory rendering."""
    positions: np.ndarray  # Shape: (N, 3) - x, y, z coordinates
    timestamps: np.ndarray  # Shape: (N,) - time values
    colors: np.ndarray  # Shape: (N, 4) - RGBA colors
    velocities: np.ndarray  # Shape: (N, 3) - velocity vectors
    gesture_strength: np.ndarray  # Shape: (N,) - gesture intensity
    point_sizes: np.ndarray  # Shape: (N,) - variable point sizes
    visible_mask: np.ndarray  # Shape: (N,) - boolean visibility mask


class PerformanceOptimizedTrajectoryVisualizer:
    """
    High-performance trajectory visualization engine with GPU optimization.
    
    Features:
    - Efficient point culling and LOD management
    - Multiple rendering modes with hardware acceleration
    - Real-time color mapping and effects
    - Interactive camera controls
    - Performance monitoring and adaptation
    """
    
    def __init__(self, 
                 settings: Optional[RenderingSettings] = None,
                 figure_size: Tuple[int, int] = (12, 9),
                 enable_3d: bool = True):
        """
        Initialize the trajectory visualizer.
        
        Args:
            settings: Rendering configuration
            figure_size: Figure dimensions
            enable_3d: Enable 3D visualization
        """
        self.settings = settings or RenderingSettings()
        self.enable_3d = enable_3d
        
        # Matplotlib setup
        self.fig = plt.figure(figsize=figure_size, facecolor='black')
        if enable_3d:
            from mpl_toolkits.mplot3d import Axes3D
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_facecolor('black')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('black')
        
        self._setup_plot_appearance()
        
        # Data management
        self.trajectory_data: Optional[TrajectoryRenderData] = None
        self.data_lock = threading.RLock()
        
        # Rendering elements
        self.plot_elements = {}
        self.color_maps = self._create_color_maps()
        
        # Performance tracking
        self.performance_stats = {
            'render_fps': 0.0,
            'points_rendered': 0,
            'points_culled': 0,
            'frame_time_ms': 0.0,
            'last_update': time.time()
        }
        
        # Animation state
        self.animation: Optional[FuncAnimation] = None
        self.animation_running = False
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # Camera state (for 3D)
        self.camera_position = np.array([0, 0, 10]) if enable_3d else None
        self.camera_target = np.array([0, 0, 0])
        self.camera_zoom = 1.0
        
        # Initialize rendering elements
        self._initialize_plot_elements()
        
        # OpenGL acceleration (if available)
        self.use_opengl = OPENGL_AVAILABLE and self.settings.lod_level == LODLevel.ULTRA
        if self.use_opengl:
            self._initialize_opengl()
    
    def _setup_plot_appearance(self) -> None:
        """Configure plot appearance for optimal visualization."""
        if self.enable_3d:
            self.ax.set_xlabel('X', color='white')
            self.ax.set_ylabel('Y', color='white')
            self.ax.set_zlabel('Z', color='white')
            self.ax.tick_params(colors='white')
        else:
            self.ax.set_xlabel('X', color='white')
            self.ax.set_ylabel('Y', color='white')
            self.ax.tick_params(colors='white')
        
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.set_title('Real-time Trajectory Visualization', color='white', fontsize=14)
        
        # Set initial limits
        self._update_plot_limits((-5, 5), (-5, 5), (-5, 5) if self.enable_3d else None)
    
    def _create_color_maps(self) -> Dict[str, LinearSegmentedColormap]:
        """Create custom color maps for different visualization modes."""
        # Velocity-based colormap (blue -> cyan -> yellow -> red)
        velocity_colors = ['#000080', '#0080FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        velocity_cmap = LinearSegmentedColormap.from_list('velocity', velocity_colors, N=256)
        
        # Time-based colormap (dark -> bright)
        time_colors = ['#1a1a1a', '#4a4a4a', '#7a7a7a', '#aaaaaa', '#ffffff']
        time_cmap = LinearSegmentedColormap.from_list('time', time_colors, N=256)
        
        # Gesture strength colormap (purple -> magenta -> white)
        gesture_colors = ['#2D1B69', '#5E2D91', '#9A4EC7', '#D670FF', '#FFFFFF']
        gesture_cmap = LinearSegmentedColormap.from_list('gesture', gesture_colors, N=256)
        
        # Trail fade colormap (transparent -> opaque)
        trail_colors = [(1, 1, 1, 0.0), (0.0, 0.8, 1.0, 0.3), (0.0, 0.6, 1.0, 0.7), (0.0, 0.4, 1.0, 1.0)]
        trail_cmap = LinearSegmentedColormap.from_list('trail_fade', trail_colors, N=256)
        
        return {
            'velocity': velocity_cmap,
            'time': time_cmap,
            'gesture': gesture_cmap,
            'trail_fade': trail_cmap
        }
    
    def _initialize_plot_elements(self) -> None:
        """Initialize plot elements for different rendering modes."""
        # Line collection for efficient line rendering
        self.plot_elements['lines'] = LineCollection([], linewidths=self.settings.line_width, 
                                                   animated=True, capstyle='round')
        
        # Point collection for particle rendering
        self.plot_elements['points'] = self.ax.scatter([], [], [], s=[], c=[], animated=True) if self.enable_3d else \
                                     self.ax.scatter([], [], s=[], c=[], animated=True)
        
        # Trail elements for fade effect
        self.plot_elements['trail_segments'] = []
        
        # Vector field arrows (for velocity visualization)
        if self.enable_3d:
            self.plot_elements['vectors'] = self.ax.quiver([], [], [], [], [], [], 
                                                         length=0.1, normalize=True, animated=True)
        else:
            self.plot_elements['vectors'] = self.ax.quiver([], [], [], [], 
                                                         scale_units='xy', angles='xy', animated=True)
        
        # Add collections to axes
        if 'lines' in self.plot_elements:
            self.ax.add_collection3d(self.plot_elements['lines']) if self.enable_3d else \
                self.ax.add_collection(self.plot_elements['lines'])
    
    def _initialize_opengl(self) -> None:
        """Initialize OpenGL acceleration components."""
        if not self.use_opengl:
            return
        
        try:
            # Create VBOs for trajectory data
            self.gl_buffers = {
                'positions': vbo.VBO(np.array([], dtype=np.float32)),
                'colors': vbo.VBO(np.array([], dtype=np.float32)),
                'indices': vbo.VBO(np.array([], dtype=np.uint32), target=gl.GL_ELEMENT_ARRAY_BUFFER)
            }
            
            # Load shaders
            vertex_shader_src = """
            #version 330 core
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec4 color;
            
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            uniform float pointSize;
            
            out vec4 fragColor;
            
            void main() {
                gl_Position = projection * view * model * vec4(position, 1.0);
                gl_PointSize = pointSize;
                fragColor = color;
            }
            """
            
            fragment_shader_src = """
            #version 330 core
            in vec4 fragColor;
            out vec4 color;
            
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (length(coord) > 0.5) discard;
                color = fragColor;
            }
            """
            
            self.gl_shader_program = shaders.compileProgram(
                shaders.compileShader(vertex_shader_src, gl.GL_VERTEX_SHADER),
                shaders.compileShader(fragment_shader_src, gl.GL_FRAGMENT_SHADER)
            )
            
            print("OpenGL acceleration initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize OpenGL: {e}")
            self.use_opengl = False
    
    def update_trajectory_data(self, positions: np.ndarray, 
                             timestamps: Optional[np.ndarray] = None,
                             gesture_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Update trajectory data for visualization.
        
        Args:
            positions: Array of shape (N, 2) or (N, 3) with trajectory points
            timestamps: Optional timestamps for each point
            gesture_data: Optional gesture-related data (strength, type, etc.)
        """
        with self.data_lock:
            n_points = len(positions)
            
            if n_points == 0:
                self.trajectory_data = None
                return
            
            # Ensure 3D positions
            if positions.shape[1] == 2:
                positions = np.column_stack([positions, np.zeros(n_points)])
            
            # Generate timestamps if not provided
            if timestamps is None:
                timestamps = np.linspace(0, 1, n_points)
            
            # Calculate velocities
            velocities = self._calculate_velocities(positions, timestamps)
            
            # Apply LOD culling
            positions, timestamps, velocities, valid_indices = self._apply_lod_culling(
                positions, timestamps, velocities)
            
            n_points = len(positions)
            
            # Generate colors based on current settings
            colors = self._generate_colors(positions, velocities, timestamps, gesture_data, valid_indices)
            
            # Generate point sizes
            point_sizes = self._generate_point_sizes(velocities, gesture_data, valid_indices)
            
            # Create visibility mask
            visible_mask = self._create_visibility_mask(positions)
            
            # Create render data structure
            self.trajectory_data = TrajectoryRenderData(
                positions=positions,
                timestamps=timestamps,
                colors=colors,
                velocities=velocities,
                gesture_strength=gesture_data.get('strength', np.ones(n_points)) if gesture_data else np.ones(n_points),
                point_sizes=point_sizes,
                visible_mask=visible_mask
            )
    
    def _calculate_velocities(self, positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate velocity vectors from position data."""
        if len(positions) < 2:
            return np.zeros_like(positions)
        
        # Calculate time differences
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero
        
        # Calculate position differences
        dp = np.diff(positions, axis=0)
        
        # Calculate velocities
        velocities = dp / dt.reshape(-1, 1)
        
        # Extend to match position array length
        velocities = np.vstack([velocities[0:1], velocities])  # Duplicate first velocity
        
        return velocities
    
    def _apply_lod_culling(self, positions: np.ndarray, timestamps: np.ndarray, 
                          velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply level-of-detail culling to reduce point count."""
        n_points = len(positions)
        max_points = min(n_points, self.settings.max_visible_points)
        
        if n_points <= max_points:
            return positions, timestamps, velocities, np.arange(n_points)
        
        # Adaptive sampling based on LOD level
        lod_factors = {
            LODLevel.LOW: 4,
            LODLevel.MEDIUM: 2,
            LODLevel.HIGH: 1,
            LODLevel.ULTRA: 1
        }
        
        decimation_factor = lod_factors[self.settings.lod_level]
        
        if decimation_factor > 1:
            # Simple decimation
            indices = np.arange(0, n_points, decimation_factor)
        else:
            # Importance-based sampling (keep points with high curvature or velocity changes)
            if len(positions) > 3:
                # Calculate curvature
                curvatures = self._calculate_curvature(positions)
                # Calculate velocity changes
                velocity_changes = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
                velocity_changes = np.concatenate([[0, 0], velocity_changes])
                
                # Combine importance scores
                importance = curvatures + 0.5 * velocity_changes
                
                # Select most important points
                indices = np.argsort(importance)[-max_points:]
                indices = np.sort(indices)
            else:
                indices = np.arange(min(max_points, n_points))
        
        return positions[indices], timestamps[indices], velocities[indices], indices
    
    def _calculate_curvature(self, positions: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point for importance sampling."""
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        # Calculate first and second derivatives
        first_deriv = np.gradient(positions, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)
        
        # Calculate curvature magnitude
        cross_product = np.cross(first_deriv[:, :2], second_deriv[:, :2])  # Use only x,y for 2D cross product
        first_deriv_mag = np.linalg.norm(first_deriv, axis=1)
        
        curvature = np.abs(cross_product) / np.maximum(first_deriv_mag**3, 1e-6)
        
        return curvature
    
    def _generate_colors(self, positions: np.ndarray, velocities: np.ndarray, 
                        timestamps: np.ndarray, gesture_data: Optional[Dict], 
                        valid_indices: np.ndarray) -> np.ndarray:
        """Generate colors based on current color mapping settings."""
        n_points = len(positions)
        
        if self.settings.color_by_velocity:
            # Color by velocity magnitude
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            vel_norm = (vel_magnitudes - vel_magnitudes.min()) / \
                      (vel_magnitudes.max() - vel_magnitudes.min() + 1e-6)
            colors = self.color_maps['velocity'](vel_norm)
            
        elif self.settings.color_by_time:
            # Color by time (age)
            time_norm = (timestamps - timestamps.min()) / \
                       (timestamps.max() - timestamps.min() + 1e-6)
            colors = self.color_maps['time'](time_norm)
            
        elif self.settings.color_by_gesture and gesture_data:
            # Color by gesture strength
            gesture_strength = gesture_data.get('strength', np.ones(len(valid_indices)))
            gesture_norm = gesture_strength / (gesture_strength.max() + 1e-6)
            colors = self.color_maps['gesture'](gesture_norm)
            
        else:
            # Default: gradient from blue to white
            default_colors = np.linspace(0, 1, n_points)
            colors = self.color_maps['trail_fade'](default_colors)
        
        return colors
    
    def _generate_point_sizes(self, velocities: np.ndarray, gesture_data: Optional[Dict],
                            valid_indices: np.ndarray) -> np.ndarray:
        """Generate variable point sizes based on data characteristics."""
        n_points = len(velocities)
        base_size = self.settings.point_size
        
        if gesture_data and 'strength' in gesture_data:
            # Scale by gesture strength
            gesture_strength = gesture_data['strength']
            sizes = base_size * (0.5 + 1.5 * gesture_strength)
        else:
            # Scale by velocity
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            vel_norm = vel_magnitudes / (vel_magnitudes.max() + 1e-6)
            sizes = base_size * (0.3 + 1.7 * vel_norm)
        
        return sizes
    
    def _create_visibility_mask(self, positions: np.ndarray) -> np.ndarray:
        """Create visibility mask based on culling settings."""
        if not self.settings.enable_culling:
            return np.ones(len(positions), dtype=bool)
        
        # Distance culling from camera/center
        center = self.camera_target if self.camera_target is not None else np.array([0, 0, 0])
        distances = np.linalg.norm(positions - center, axis=1)
        
        return distances <= self.settings.culling_distance
    
    def _update_plot_limits(self, xlim: Tuple[float, float], ylim: Tuple[float, float], 
                          zlim: Optional[Tuple[float, float]] = None) -> None:
        """Update plot limits with margin."""
        margin = 0.1
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        self.ax.set_xlim(xlim[0] - margin * x_range, xlim[1] + margin * x_range)
        self.ax.set_ylim(ylim[0] - margin * y_range, ylim[1] + margin * y_range)
        
        if self.enable_3d and zlim:
            z_range = zlim[1] - zlim[0]
            self.ax.set_zlim(zlim[0] - margin * z_range, zlim[1] + margin * z_range)
    
    def _render_frame(self, frame_num: int) -> List[Any]:
        """Render a single frame of the trajectory visualization."""
        start_time = time.time()
        
        with self.data_lock:
            if self.trajectory_data is None:
                return []
            
            data = self.trajectory_data
            visible_indices = np.where(data.visible_mask)[0]
            
            if len(visible_indices) == 0:
                return []
            
            # Update performance stats
            self.performance_stats['points_rendered'] = len(visible_indices)
            self.performance_stats['points_culled'] = len(data.positions) - len(visible_indices)
            
            # Get visible data
            visible_positions = data.positions[visible_indices]
            visible_colors = data.colors[visible_indices]
            visible_sizes = data.point_sizes[visible_indices]
            
            updated_elements = []
            
            # Render based on current mode
            if self.settings.render_mode == RenderMode.POINT_CLOUD:
                updated_elements.extend(self._render_point_cloud(visible_positions, visible_colors, visible_sizes))
                
            elif self.settings.render_mode == RenderMode.SOLID_LINE:
                updated_elements.extend(self._render_solid_lines(visible_positions, visible_colors))
                
            elif self.settings.render_mode == RenderMode.FADE_TRAIL:
                updated_elements.extend(self._render_fade_trail(visible_positions, visible_colors, visible_indices))
                
            elif self.settings.render_mode == RenderMode.PARTICLE_BASED:
                updated_elements.extend(self._render_particles(visible_positions, visible_colors, visible_sizes))
                
            elif self.settings.render_mode == RenderMode.VECTOR_FIELD:
                visible_velocities = data.velocities[visible_indices]
                updated_elements.extend(self._render_vector_field(visible_positions, visible_velocities, visible_colors))
            
            # Update plot limits if needed
            if len(visible_positions) > 0:
                margin = 1.0
                xlim = (visible_positions[:, 0].min() - margin, visible_positions[:, 0].max() + margin)
                ylim = (visible_positions[:, 1].min() - margin, visible_positions[:, 1].max() + margin)
                zlim = (visible_positions[:, 2].min() - margin, visible_positions[:, 2].max() + margin) if self.enable_3d else None
                self._update_plot_limits(xlim, ylim, zlim)
            
        # Update performance metrics
        frame_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.performance_stats['frame_time_ms'] = frame_time
        self._update_fps_stats()
        
        return updated_elements
    
    def _render_point_cloud(self, positions: np.ndarray, colors: np.ndarray, 
                          sizes: np.ndarray) -> List[Any]:
        """Render trajectory as point cloud."""
        if self.enable_3d:
            self.plot_elements['points']._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        else:
            self.plot_elements['points'].set_offsets(positions[:, :2])
        
        self.plot_elements['points'].set_color(colors)
        self.plot_elements['points'].set_sizes(sizes)
        
        return [self.plot_elements['points']]
    
    def _render_solid_lines(self, positions: np.ndarray, colors: np.ndarray) -> List[Any]:
        """Render trajectory as solid connected lines."""
        if len(positions) < 2:
            return []
        
        # Create line segments
        segments = np.array([positions[:-1], positions[1:]]).transpose(1, 0, 2)
        
        # Update line collection
        if self.enable_3d:
            self.plot_elements['lines'] = Line3DCollection(segments, colors=colors[1:], 
                                                         linewidths=self.settings.line_width, alpha=0.8)
            self.ax.add_collection3d(self.plot_elements['lines'])
        else:
            self.plot_elements['lines'].set_segments(segments[:, :, :2])
            self.plot_elements['lines'].set_colors(colors[1:])
        
        return [self.plot_elements['lines']]
    
    def _render_fade_trail(self, positions: np.ndarray, colors: np.ndarray, indices: np.ndarray) -> List[Any]:
        """Render trajectory with fading trail effect."""
        if len(positions) < 2:
            return []
        
        trail_length = min(self.settings.trail_length, len(positions))
        
        # Create segments with varying alpha for fade effect
        segments = []
        segment_colors = []
        
        for i in range(len(positions) - 1):
            segments.append([positions[i], positions[i + 1]])
            
            # Calculate fade factor based on recency
            fade_factor = (i + 1) / len(positions)
            fade_alpha = self.settings.alpha_fade_factor ** (len(positions) - i - 1)
            
            color = colors[i + 1].copy()
            color[3] = fade_alpha  # Set alpha
            segment_colors.append(color)
        
        # Update line collection
        if segments:
            if self.enable_3d:
                # For 3D, create new collection each time
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                self.plot_elements['lines'] = Line3DCollection(segments, colors=segment_colors,
                                                             linewidths=self.settings.line_width)
                self.ax.add_collection3d(self.plot_elements['lines'])
            else:
                self.plot_elements['lines'].set_segments(segments)
                self.plot_elements['lines'].set_colors(segment_colors)
        
        return [self.plot_elements['lines']]
    
    def _render_particles(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray) -> List[Any]:
        """Render trajectory as animated particles with glow effect."""
        # Similar to point cloud but with enhanced visual effects
        elements = self._render_point_cloud(positions, colors, sizes * 2.0)  # Larger particles
        
        # Add glow effect by rendering twice with different sizes
        if len(positions) > 0:
            glow_colors = colors.copy()
            glow_colors[:, 3] *= 0.3  # Make glow more transparent
            
            # Create glow layer (larger, more transparent points)
            if self.enable_3d:
                glow_points = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                           s=sizes * 4.0, c=glow_colors, alpha=0.3, animated=True)
            else:
                glow_points = self.ax.scatter(positions[:, 0], positions[:, 1],
                                           s=sizes * 4.0, c=glow_colors, alpha=0.3, animated=True)
            
            elements.append(glow_points)
        
        return elements
    
    def _render_vector_field(self, positions: np.ndarray, velocities: np.ndarray, 
                           colors: np.ndarray) -> List[Any]:
        """Render trajectory with velocity vectors."""
        if len(positions) == 0:
            return []
        
        # Subsample for vector field (too many arrows look cluttered)
        step = max(1, len(positions) // 50)  # Show max 50 vectors
        pos_subset = positions[::step]
        vel_subset = velocities[::step]
        color_subset = colors[::step]
        
        # Normalize velocities for display
        vel_magnitudes = np.linalg.norm(vel_subset, axis=1)
        normalized_velocities = vel_subset / np.maximum(vel_magnitudes.reshape(-1, 1), 1e-6)
        
        # Scale arrows
        arrow_scale = 0.2
        
        if self.enable_3d:
            # Remove previous vectors and create new ones
            self.plot_elements['vectors'].remove()
            self.plot_elements['vectors'] = self.ax.quiver(
                pos_subset[:, 0], pos_subset[:, 1], pos_subset[:, 2],
                normalized_velocities[:, 0] * arrow_scale,
                normalized_velocities[:, 1] * arrow_scale,
                normalized_velocities[:, 2] * arrow_scale,
                colors=color_subset, arrow_length_ratio=0.3, animated=True
            )
        else:
            self.plot_elements['vectors'].set_UVC(
                normalized_velocities[:, 0] * arrow_scale,
                normalized_velocities[:, 1] * arrow_scale
            )
            self.plot_elements['vectors'].set_offsets(pos_subset[:, :2])
        
        return [self.plot_elements['vectors']]
    
    def _update_fps_stats(self) -> None:
        """Update FPS performance statistics."""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_update >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.performance_stats['render_fps'] = fps
            
            self.frame_count = 0
            self.last_fps_update = current_time
            self.performance_stats['last_update'] = current_time
    
    def start_animation(self, interval: Optional[int] = None) -> FuncAnimation:
        """Start real-time animation."""
        if interval is None:
            interval = int(1000.0 / self.settings.update_frequency_hz)
        
        self.animation = FuncAnimation(
            self.fig, self._render_frame,
            interval=interval, blit=False, cache_frame_data=False
        )
        self.animation_running = True
        
        return self.animation
    
    def stop_animation(self) -> None:
        """Stop animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation_running = False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def set_render_mode(self, mode: RenderMode) -> None:
        """Change rendering mode."""
        self.settings.render_mode = mode
        print(f"Render mode changed to: {mode.value}")
    
    def set_lod_level(self, level: LODLevel) -> None:
        """Change level of detail."""
        old_level = self.settings.lod_level
        self.settings.lod_level = level
        
        # Update max points based on LOD level
        lod_max_points = {
            LODLevel.LOW: 500,
            LODLevel.MEDIUM: 1000,
            LODLevel.HIGH: 2000,
            LODLevel.ULTRA: 5000
        }
        self.settings.max_visible_points = lod_max_points[level]
        
        print(f"LOD level changed from {old_level.name} to {level.name}")
    
    def set_color_mode(self, color_by_velocity: bool = False, color_by_time: bool = False, 
                      color_by_gesture: bool = False) -> None:
        """Set color mapping mode."""
        self.settings.color_by_velocity = color_by_velocity
        self.settings.color_by_time = color_by_time
        self.settings.color_by_gesture = color_by_gesture
        
        mode_names = []
        if color_by_velocity:
            mode_names.append("velocity")
        if color_by_time:
            mode_names.append("time")
        if color_by_gesture:
            mode_names.append("gesture")
        
        mode_str = ", ".join(mode_names) if mode_names else "default"
        print(f"Color mode set to: {mode_str}")
    
    def show(self) -> None:
        """Display the visualization."""
        plt.show()
    
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Frame saved to: {filename}")


# Example usage and test
if __name__ == "__main__":
    print("Real-time Trajectory Visualization Engine Test")
    
    # Create test trajectory data
    t = np.linspace(0, 4 * np.pi, 1000)
    x = 2.0 * np.cos(t) + 1.0 * np.cos(3*t)
    y = 2.0 * np.sin(t) + 1.0 * np.sin(3*t)
    z = 0.5 * np.sin(2*t)
    
    positions = np.column_stack([x, y, z])
    timestamps = t
    
    gesture_data = {
        'strength': 0.5 + 0.5 * np.sin(5*t),
        'type': (np.sin(t) > 0).astype(int)
    }
    
    # Create visualizer
    settings = RenderingSettings(
        render_mode=RenderMode.FADE_TRAIL,
        lod_level=LODLevel.HIGH,
        max_visible_points=1500,
        color_by_velocity=True,
        update_frequency_hz=30.0
    )
    
    visualizer = PerformanceOptimizedTrajectoryVisualizer(settings, enable_3d=True)
    
    # Update with test data
    visualizer.update_trajectory_data(positions, timestamps, gesture_data)
    
    # Start animation
    print("Starting visualization animation...")
    anim = visualizer.start_animation()
    
    # Show performance stats after a few seconds
    def print_stats():
        time.sleep(3)
        stats = visualizer.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"Render FPS: {stats['render_fps']:.1f}")
        print(f"Points rendered: {stats['points_rendered']}")
        print(f"Points culled: {stats['points_culled']}")
        print(f"Frame time: {stats['frame_time_ms']:.1f} ms")
    
    import threading
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    
    # Display
    visualizer.show()