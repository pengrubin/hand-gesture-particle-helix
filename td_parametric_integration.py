#!/usr/bin/env python3
"""
TouchDesigner Parametric Integration Module
Bridge between hand gesture detection and parametric equation visualization in TouchDesigner.

This module provides a unified interface for managing both particle and parametric rendering
systems within TouchDesigner, handling data flow between Python modules and TD operators.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum

# Import existing modules
from hand_gesture_detector import HandGestureDetector
from gesture_parametric_bridge import GestureParametricBridge, HandAssignment
from parametric_equation_renderer import ParametricEquationRenderer

# Import GPU optimization modules
from td_gpu_parametric_renderer import TouchDesignerGPUParametricRenderer, GPURenderingMode, GPULODLevel
from td_gpu_optimization import (
    TouchDesignerGPUOptimizer, PerformanceMetrics, OptimizationLevel,
    get_gpu_optimizer, record_frame_performance
)


class RenderingMode(Enum):
    """Rendering mode enumeration."""
    PARTICLE = "particle"
    PARAMETRIC = "parametric"
    HYBRID = "hybrid"


class TouchDesignerParametricIntegration:
    """
    Main integration class for TouchDesigner parametric equation system.
    
    Features:
    - Unified parameter management for both particle and parametric systems
    - Real-time gesture-to-parameter mapping
    - TouchDesigner Table DAT integration
    - Mode switching between particle/parametric/hybrid rendering
    - Performance optimization for TD frame-based execution
    """
    
    def __init__(self):
        """Initialize the TouchDesigner integration system."""
        
        # Core components
        self.gesture_bridge = GestureParametricBridge(
            r_max=2.0,
            smoothing_factor=0.85,
            hand_assignment=HandAssignment.LEFT_R1_RIGHT_R2,
            auto_pause=True
        )
        
        # Rendering state
        self.current_mode = RenderingMode.PARTICLE
        self.previous_mode = RenderingMode.PARTICLE
        self.mode_transition_time = 0.0
        self.mode_transition_duration = 1.0  # seconds
        
        # Frame management
        self.frame_count = 0
        self.last_update_time = time.time()
        self.target_fps = 30
        self.frame_skip_threshold = 0.04  # Skip if processing takes >40ms
        
        # Parameter state
        self.current_parameters = {
            'r1': 1.0, 'r2': 0.5,
            'w1': 1.0, 'w2': 2.0,
            'p1': 0.0, 'p2': 0.0,
            'theta': 0.0, 'theta_step': 0.05
        }
        
        # Trajectory management
        self.trajectory_points = []
        self.max_trajectory_points = 1000
        self.trail_length = 500
        self.rod_positions = {
            'rod1': (0.0, 0.0), 
            'rod2': (0.0, 0.0),
            'final': (0.0, 0.0)
        }
        
        # Performance monitoring
        self.performance_stats = {
            'frame_time': 0.0,
            'gesture_detection_time': 0.0,
            'parameter_update_time': 0.0,
            'table_update_time': 0.0,
            'average_fps': 30.0
        }
        
        # TouchDesigner references (will be set by TD)
        self.td_ops = {}
        
        # GPU rendering components
        self.gpu_renderer = TouchDesignerGPUParametricRenderer(max_trajectory_points=10000)
        self.gpu_optimizer = get_gpu_optimizer()
        self.gpu_enabled = True
        self.gpu_fallback_mode = False
        
        # GPU-specific performance settings
        self.gpu_performance_target = {
            'target_fps': 30.0,
            'min_fps_threshold': 20.0,
            'max_memory_mb': 128,
            'adaptive_quality': True
        }
        
    def initialize_touchdesigner_operators(self, ops_dict: Dict[str, Any]) -> None:
        """
        Initialize references to TouchDesigner operators.
        
        Args:
            ops_dict: Dictionary mapping operator names to TD operator objects
        """
        self.td_ops = ops_dict
        
        # Initialize Table DAT structures
        self._initialize_table_dats()
        
        # Initialize GPU rendering system
        if self.gpu_enabled:
            self._initialize_gpu_system()
        
    def _initialize_gpu_system(self) -> None:
        """
        Initialize GPU rendering and optimization systems.
        """
        try:
            # Initialize GPU renderer with TouchDesigner operators
            gpu_ops = self._extract_gpu_operators()
            self.gpu_renderer.initialize_gpu_operators(gpu_ops)
            
            # Configure GPU optimizer
            self.gpu_optimizer.target_fps = self.gpu_performance_target['target_fps']
            self.gpu_optimizer.min_fps_threshold = self.gpu_performance_target['min_fps_threshold']
            self.gpu_optimizer.max_gpu_memory = self.gpu_performance_target['max_memory_mb'] * 1024 * 1024
            
            # Optimize SOP operators for GPU rendering
            sop_ops = self._extract_sop_operators()
            optimization_results = self.gpu_optimizer.optimize_for_touchdesigner_sops(
                sop_ops, 
                self.max_trajectory_points,
                self.gpu_performance_target['target_fps']
            )
            
            print(f"GPU system initialized: {optimization_results}")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.gpu_enabled = False
            self.gpu_fallback_mode = True
            
    def _extract_gpu_operators(self) -> Dict[str, Any]:
        """
        Extract GPU-specific operators from TouchDesigner operators.
        """
        gpu_ops = {}
        
        # Map standard operators to GPU-optimized versions
        operator_mapping = {
            'trajectory_add_sop': ['trajectory_add', 'add_sop', 'points_add'],
            'trajectory_resample_sop': ['trajectory_resample', 'resample_sop', 'curve_resample'],
            'trajectory_instance_comp': ['trajectory_instance', 'instance_comp', 'point_instance'],
            'curve_geometry_comp': ['curve_geometry', 'geometry_comp', 'parametric_geometry'],
            'parametric_curves_glsl': ['parametric_shader', 'curve_shader', 'glsl_parametric'],
            'trail_effect_glsl': ['trail_shader', 'trail_effect', 'glsl_trail'],
            'volumetric_glsl': ['volumetric_shader', 'volume_shader', 'glsl_volume']
        }
        
        for gpu_key, possible_keys in operator_mapping.items():
            for key in possible_keys:
                if key in self.td_ops:
                    gpu_ops[gpu_key] = self.td_ops[key]
                    break
        
        return gpu_ops
        
    def _extract_sop_operators(self) -> Dict[str, Any]:
        """
        Extract SOP operators for optimization.
        """
        sop_ops = {}
        
        sop_mapping = {
            'trajectory_add': ['trajectory_add', 'add_sop', 'points_add'],
            'trajectory_resample': ['trajectory_resample', 'resample_sop', 'curve_resample'],
            'trajectory_instance': ['trajectory_instance', 'instance_comp', 'point_instance']
        }
        
        for sop_key, possible_keys in sop_mapping.items():
            for key in possible_keys:
                if key in self.td_ops:
                    sop_ops[sop_key] = self.td_ops[key]
                    break
                    
        return sop_ops
        
    def _initialize_table_dats(self) -> None:
        """Initialize all Table DAT structures with proper schemas."""
        
        # Initialize gesture_params table
        if 'gesture_params' in self.td_ops:
            table = self.td_ops['gesture_params']
            # Clear and setup headers
            table.clear()
            table.appendRow(['hand_id', 'label', 'gesture', 'pos_x', 'pos_y', 
                            'openness', 'timestamp', 'active'])
            
        # Initialize parametric_params table  
        if 'parametric_params' in self.td_ops:
            table = self.td_ops['parametric_params']
            table.clear()
            table.appendRow(['r1', 'r2', 'w1', 'w2', 'p1', 'p2', 
                            'theta', 'theta_step', 'mode', 'paused'])
            # Add initial values
            table.appendRow([1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 
                            0.0, 0.05, 'particle', False])
            
        # Initialize trajectory_data table
        if 'trajectory_data' in self.td_ops:
            table = self.td_ops['trajectory_data']
            table.clear()
            table.appendRow(['point_id', 'x', 'y', 'z', 'timestamp', 'trail_alpha',
                            'rod1_x', 'rod1_y', 'rod2_x', 'rod2_y'])
    
    def process_frame_update(self, camera_data: np.ndarray, 
                           frame_number: int) -> Dict[str, Any]:
        """
        Main frame processing function called by TouchDesigner Execute DAT.
        
        Args:
            camera_data: Camera frame data from Video In TOP
            frame_number: Current frame number from TD
            
        Returns:
            Dictionary containing processing results and statistics
        """
        start_time = time.time()
        self.frame_count = frame_number
        
        # Check if we should skip this frame for performance
        if self._should_skip_frame(start_time):
            return self._get_frame_skip_result()
        
        try:
            # Process gesture detection
            gesture_start = time.time()
            processed_frame, parameters = self.gesture_bridge.process_frame(camera_data)
            self.performance_stats['gesture_detection_time'] = time.time() - gesture_start
            
            # Update parameters
            param_start = time.time() 
            self._update_parameters(parameters)
            self.performance_stats['parameter_update_time'] = time.time() - param_start
            
            # Update TouchDesigner tables
            table_start = time.time()
            self._update_table_dats()
            self.performance_stats['table_update_time'] = time.time() - table_start
            
            # GPU rendering update
            gpu_start = time.time()
            gpu_results = self._update_gpu_rendering()
            self.performance_stats['gpu_rendering_time'] = time.time() - gpu_start
            
            # Update rendering based on current mode
            self._update_rendering_mode()
            
            # Record performance metrics for GPU optimizer
            total_time = time.time() - start_time
            self._record_gpu_performance_metrics(total_time)
            
            # Calculate performance stats
            self.performance_stats['frame_time'] = total_time
            self._update_fps_stats(total_time)
            
            result = {
                'success': True,
                'parameters': self.current_parameters.copy(),
                'gesture_info': self.gesture_bridge.get_gesture_info(),
                'performance': self.performance_stats.copy(),
                'trajectory_points': len(self.trajectory_points)
            }
            
            # Add GPU rendering results if available
            if gpu_results and gpu_results.get('success', False):
                result['gpu_performance'] = gpu_results
                result['gpu_enabled'] = True
            else:
                result['gpu_enabled'] = False
                result['gpu_fallback'] = self.gpu_fallback_mode
                
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': self.current_parameters.copy()
            }
    
    def _update_gpu_rendering(self) -> Optional[Dict[str, Any]]:
        """
        Update GPU rendering system with current trajectory data.
        """
        if not self.gpu_enabled or self.gpu_fallback_mode:
            return None
            
        try:
            # Update GPU renderer with current data
            gpu_results = self.gpu_renderer.update_gpu_rendering(
                trajectory_points=self.trajectory_points,
                rod_positions=self.rod_positions,
                parameters=self.current_parameters
            )
            
            # Handle GPU rendering results
            if gpu_results.get('success', False):
                # GPU rendering successful, update performance stats
                self.performance_stats['gpu_frame_time'] = gpu_results.get('frame_time', 0.0)
                self.performance_stats['gpu_memory_usage'] = gpu_results.get('gpu_memory_usage', 0)
                self.performance_stats['gpu_lod_level'] = gpu_results.get('lod_level', 0)
                
                return gpu_results
            else:
                # GPU rendering failed, consider fallback
                print(f"GPU rendering failed: {gpu_results.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"GPU rendering error: {e}")
            # Consider disabling GPU temporarily
            if str(e).lower() in ['out of memory', 'memory', 'gpu memory']:
                print("GPU memory issue detected, enabling fallback mode")
                self.gpu_fallback_mode = True
            return None
    
    def _record_gpu_performance_metrics(self, frame_time: float) -> None:
        """
        Record performance metrics for GPU optimization.
        """
        if not self.gpu_enabled:
            return
            
        try:
            # Calculate GPU usage approximation based on frame time and workload
            point_count = len(self.trajectory_points)
            estimated_gpu_usage = min(1.0, (frame_time / 0.033) * (point_count / 1000.0))
            
            # Calculate memory usage approximation
            estimated_memory_usage = min(1.0, point_count / self.max_trajectory_points)
            
            # Record metrics with GPU optimizer
            record_frame_performance(
                frame_time=frame_time,
                gpu_usage=estimated_gpu_usage,
                memory_usage=estimated_memory_usage,
                vertex_count=point_count * 2,  # Approximate vertex count for curves
                draw_calls=max(1, point_count // 100)  # Estimated draw calls
            )
            
        except Exception as e:
            print(f"Failed to record GPU performance metrics: {e}")
    
    def _should_skip_frame(self, current_time: float) -> bool:
        """Determine if frame should be skipped for performance."""
        time_since_last = current_time - self.last_update_time
        
        # GPU-enhanced frame skipping logic
        if self.gpu_enabled and not self.gpu_fallback_mode:
            # Use GPU optimizer recommendations
            gpu_stats = self.gpu_optimizer.get_optimization_statistics()
            current_fps = gpu_stats.get('average_fps', 30.0)
            
            # Skip frame if GPU performance is poor
            if current_fps < self.gpu_performance_target['min_fps_threshold']:
                skip_threshold = (1.0 / self.target_fps) * 1.5  # More aggressive skipping
                return time_since_last < skip_threshold
        
        # Standard frame skipping logic
        return time_since_last < (1.0 / self.target_fps) * 0.8  # Allow 20% margin
        
    def _get_frame_skip_result(self) -> Dict[str, Any]:
        """Return result for skipped frame."""
        return {
            'success': True,
            'skipped': True,
            'parameters': self.current_parameters.copy(),
            'performance': self.performance_stats.copy()
        }
    
    def _update_parameters(self, new_parameters: Dict[str, float]) -> None:
        """Update internal parameters with smoothing."""
        smoothing = 0.9  # High smoothing for TD integration
        
        for key, new_value in new_parameters.items():
            if key in self.current_parameters:
                old_value = self.current_parameters[key]
                self.current_parameters[key] = (
                    old_value * smoothing + new_value * (1.0 - smoothing)
                )
    
    def _update_table_dats(self) -> None:
        """Update all TouchDesigner Table DATs with current data."""
        
        # Update gesture parameters table
        self._update_gesture_params_table()
        
        # Update parametric parameters table
        self._update_parametric_params_table()
        
        # Update trajectory data table (only in parametric modes)
        if self.current_mode in [RenderingMode.PARAMETRIC, RenderingMode.HYBRID]:
            self._update_trajectory_data_table()
    
    def _update_gesture_params_table(self) -> None:
        """Update the gesture_params Table DAT."""
        if 'gesture_params' not in self.td_ops:
            return
            
        table = self.td_ops['gesture_params']
        gesture_info = self.gesture_bridge.get_gesture_info()
        gesture_data = self.gesture_bridge.gesture_detector.gesture_data
        
        # Clear existing data rows (keep header)
        while table.numRows > 1:
            table.deleteRow(1)
        
        # Add current hand data
        if 'hands' in gesture_data:
            for i, hand_data in enumerate(gesture_data['hands']):
                table.appendRow([
                    hand_data.get('id', i),
                    hand_data.get('label', 'unknown'),
                    hand_data.get('gesture_number', 0),
                    hand_data.get('center', [0.5, 0.5])[0],
                    hand_data.get('center', [0.5, 0.5])[1],
                    hand_data.get('openness', 0.0),
                    time.time(),
                    True
                ])
    
    def _update_parametric_params_table(self) -> None:
        """Update the parametric_params Table DAT."""
        if 'parametric_params' not in self.td_ops:
            return
            
        table = self.td_ops['parametric_params']
        
        # Update the parameter row (row 1, after header)
        if table.numRows < 2:
            table.appendRow([0] * table.numCols)
            
        table[1, 'r1'] = self.current_parameters['r1']
        table[1, 'r2'] = self.current_parameters['r2']
        table[1, 'w1'] = self.current_parameters['w1']
        table[1, 'w2'] = self.current_parameters['w2']
        table[1, 'p1'] = self.current_parameters['p1']
        table[1, 'p2'] = self.current_parameters['p2']
        table[1, 'theta'] = self.current_parameters['theta']
        table[1, 'theta_step'] = self.current_parameters['theta_step']
        table[1, 'mode'] = self.current_mode.value
        table[1, 'paused'] = self.gesture_bridge.is_paused
    
    def _update_trajectory_data_table(self) -> None:
        """Update the trajectory_data Table DAT with current trajectory points."""
        if 'trajectory_data' not in self.td_ops:
            return
            
        table = self.td_ops['trajectory_data']
        
        # Calculate current trajectory point
        current_point = self._calculate_current_parametric_point()
        self._add_trajectory_point(current_point)
        
        # Update rod positions
        self._update_rod_positions()
        
        # Clear existing data rows
        while table.numRows > 1:
            table.deleteRow(1)
            
        # Add trajectory points with alpha fade
        for i, point in enumerate(self.trajectory_points):
            # Calculate trail alpha (newer points are more opaque)
            alpha = (i + 1) / len(self.trajectory_points)
            
            table.appendRow([
                point['id'],
                point['x'],
                point['y'], 
                point.get('z', 0.0),
                point['timestamp'],
                alpha,
                self.rod_positions['rod1'][0],
                self.rod_positions['rod1'][1],
                self.rod_positions['rod2'][0],
                self.rod_positions['rod2'][1]
            ])
    
    def _calculate_current_parametric_point(self) -> Dict[str, float]:
        """Calculate current parametric equation point."""
        theta = self.current_parameters['theta']
        r1, r2 = self.current_parameters['r1'], self.current_parameters['r2']
        w1, w2 = self.current_parameters['w1'], self.current_parameters['w2']
        p1, p2 = self.current_parameters['p1'], self.current_parameters['p2']
        
        # Calculate complex components
        z1 = r1 * np.exp(1j * (w1 * theta + p1))
        z2 = r2 * np.exp(1j * (w2 * theta + p2))
        z_total = z1 + z2
        
        # Update theta for next frame
        self.current_parameters['theta'] += self.current_parameters['theta_step']
        if self.current_parameters['theta'] > 4 * np.pi:
            self.current_parameters['theta'] = 0.0
            # Clear trajectory on reset
            self.trajectory_points = []
        
        return {
            'id': len(self.trajectory_points),
            'x': float(z_total.real),
            'y': float(z_total.imag),
            'z': 0.0,
            'timestamp': time.time()
        }
    
    def _add_trajectory_point(self, point: Dict[str, float]) -> None:
        """Add point to trajectory buffer."""
        self.trajectory_points.append(point)
        
        # Maintain buffer size based on current mode
        max_points = self.trail_length
        if self.current_mode == RenderingMode.HYBRID:
            max_points = max_points // 2  # Reduce for performance
            
        if len(self.trajectory_points) > max_points:
            self.trajectory_points = self.trajectory_points[-max_points:]
    
    def _update_rod_positions(self) -> None:
        """Update rotating rod positions for visualization."""
        theta = self.current_parameters['theta']
        r1, r2 = self.current_parameters['r1'], self.current_parameters['r2']
        w1, w2 = self.current_parameters['w1'], self.current_parameters['w2']
        p1, p2 = self.current_parameters['p1'], self.current_parameters['p2']
        
        # Rod 1: from origin to r1*e^(i*(w1*theta + p1))
        z1 = r1 * np.exp(1j * (w1 * theta + p1))
        self.rod_positions['rod1'] = (float(z1.real), float(z1.imag))
        
        # Rod 2: from rod1 end to final position
        z2_relative = r2 * np.exp(1j * (w2 * theta + p2))
        z2_absolute = z1 + z2_relative
        self.rod_positions['rod2'] = (float(z2_absolute.real), float(z2_absolute.imag))
        self.rod_positions['final'] = self.rod_positions['rod2']
    
    def _update_rendering_mode(self) -> None:
        """Update rendering based on current mode."""
        # Mode transition handling
        if self.current_mode != self.previous_mode:
            self.mode_transition_time = time.time()
            self._on_mode_change()
            self.previous_mode = self.current_mode
    
    def _on_mode_change(self) -> None:
        """Handle mode change transitions."""
        if self.current_mode == RenderingMode.PARAMETRIC:
            # Reset trajectory when switching to parametric
            self.trajectory_points = []
            self.current_parameters['theta'] = 0.0
            
        elif self.current_mode == RenderingMode.PARTICLE:
            # Could trigger particle system reset here
            pass
            
        elif self.current_mode == RenderingMode.HYBRID:
            # Optimize for dual rendering
            self.trail_length = min(self.trail_length, 250)
    
    def _update_fps_stats(self, frame_time: float) -> None:
        """Update FPS statistics with exponential moving average."""
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            # Exponential moving average
            alpha = 0.1
            self.performance_stats['average_fps'] = (
                self.performance_stats['average_fps'] * (1 - alpha) + 
                current_fps * alpha
            )
    
    # Public interface methods for TouchDesigner
    
    def set_rendering_mode(self, mode: str) -> bool:
        """Set rendering mode from TouchDesigner."""
        try:
            self.current_mode = RenderingMode(mode.lower())
            return True
        except ValueError:
            return False
    
    def get_rendering_mode(self) -> str:
        """Get current rendering mode."""
        return self.current_mode.value
    
    def set_parameter(self, param_name: str, value: float) -> bool:
        """Set individual parameter value."""
        if param_name in self.current_parameters:
            self.current_parameters[param_name] = float(value)
            return True
        return False
    
    def get_parameter(self, param_name: str) -> Optional[float]:
        """Get individual parameter value."""
        return self.current_parameters.get(param_name)
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all current parameters."""
        return self.current_parameters.copy()
    
    def reset_trajectory(self) -> None:
        """Reset trajectory data."""
        self.trajectory_points = []
        self.current_parameters['theta'] = 0.0
    
    def set_trail_length(self, length: int) -> None:
        """Set trajectory trail length."""
        self.trail_length = max(10, min(2000, int(length)))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def set_hand_assignment(self, assignment: str) -> bool:
        """Set hand assignment mode."""
        try:
            assignment_enum = HandAssignment(assignment.lower())
            self.gesture_bridge.set_hand_assignment(assignment_enum)
            return True
        except ValueError:
            return False
    
    def toggle_pause(self) -> bool:
        """Toggle pause state."""
        self.gesture_bridge.is_paused = not self.gesture_bridge.is_paused
        return self.gesture_bridge.is_paused
    
    def is_paused(self) -> bool:
        """Check if system is paused."""
        return self.gesture_bridge.is_paused
    
    def get_gesture_status(self) -> Dict[str, Any]:
        """Get current gesture detection status."""
        return self.gesture_bridge.get_gesture_info()
    
    def get_trajectory_point_count(self) -> int:
        """Get current trajectory point count."""
        return len(self.trajectory_points)
    
    def set_smoothing_factor(self, factor: float) -> None:
        """Set parameter smoothing factor."""
        factor = max(0.0, min(1.0, float(factor)))
        self.gesture_bridge.set_smoothing_factor(factor)
    
    # GPU-specific interface methods
    
    def enable_gpu_rendering(self, enabled: bool = True) -> bool:
        """Enable or disable GPU rendering."""
        if enabled and not self.gpu_enabled:
            # Try to re-initialize GPU system
            try:
                self._initialize_gpu_system()
                self.gpu_enabled = True
                self.gpu_fallback_mode = False
                return True
            except Exception as e:
                print(f"Failed to enable GPU rendering: {e}")
                return False
        elif not enabled and self.gpu_enabled:
            self.gpu_enabled = False
            return True
        return self.gpu_enabled == enabled
    
    def set_gpu_rendering_mode(self, mode: str) -> bool:
        """Set GPU rendering mode (points, lines, curves, instanced, volumetric)."""
        if not self.gpu_enabled:
            return False
        return self.gpu_renderer.set_rendering_mode(mode)
    
    def get_gpu_rendering_mode(self) -> str:
        """Get current GPU rendering mode."""
        if not self.gpu_enabled:
            return "disabled"
        return self.gpu_renderer.current_mode.value
    
    def set_gpu_lod_level(self, level: int) -> bool:
        """Manually set GPU Level of Detail (0-4, where 4 is highest quality)."""
        if not self.gpu_enabled:
            return False
        return self.gpu_renderer.set_lod_level_manual(level)
    
    def enable_gpu_auto_lod(self, enabled: bool = True) -> None:
        """Enable or disable automatic GPU Level of Detail adjustment."""
        if self.gpu_enabled:
            if enabled:
                self.gpu_renderer.enable_auto_lod()
            else:
                self.gpu_renderer.auto_lod = False
    
    def set_gpu_parameter(self, param_name: str, value: float) -> bool:
        """Set GPU rendering parameter (curve_thickness, glow_intensity, etc.)."""
        if not self.gpu_enabled:
            return False
        return self.gpu_renderer.set_gpu_parameter(param_name, value)
    
    def get_gpu_parameters(self) -> Dict[str, float]:
        """Get all GPU rendering parameters."""
        if not self.gpu_enabled:
            return {}
        return self.gpu_renderer.get_gpu_parameters()
    
    def get_gpu_performance_stats(self) -> Dict[str, Any]:
        """Get detailed GPU performance statistics."""
        if not self.gpu_enabled:
            return {'gpu_enabled': False}
        
        # Combine GPU renderer and optimizer statistics
        gpu_stats = self.gpu_renderer.get_performance_stats()
        optimizer_stats = self.gpu_optimizer.get_optimization_statistics()
        
        return {
            'gpu_enabled': True,
            'gpu_fallback_mode': self.gpu_fallback_mode,
            'renderer_stats': gpu_stats,
            'optimizer_stats': optimizer_stats,
            'metal_optimizations': self.gpu_optimizer.create_metal_optimized_settings()
        }
    
    def optimize_gpu_for_point_count(self, point_count: int) -> None:
        """Optimize GPU settings for specific trajectory point count."""
        if self.gpu_enabled:
            self.gpu_renderer.optimize_for_point_count(point_count)
    
    def reset_gpu_buffers(self) -> bool:
        """Reset GPU buffers and memory pools."""
        if not self.gpu_enabled:
            return False
        try:
            self.gpu_renderer.reset_gpu_buffers()
            self.gpu_optimizer.reset_optimization_state()
            return True
        except Exception as e:
            print(f"Failed to reset GPU buffers: {e}")
            return False
    
    def enable_gpu_instancing(self, enabled: bool = True) -> bool:
        """Enable or disable GPU instancing for point/curve rendering."""
        if not self.gpu_enabled:
            return False
        self.gpu_renderer.enable_instancing(enabled)
        return True
    
    def set_gpu_memory_limit(self, limit_mb: int) -> bool:
        """Set GPU memory usage limit in megabytes."""
        if not self.gpu_enabled:
            return False
        try:
            self.gpu_renderer.set_memory_limit(limit_mb)
            self.gpu_optimizer.max_gpu_memory = limit_mb * 1024 * 1024
            self.gpu_performance_target['max_memory_mb'] = limit_mb
            return True
        except Exception as e:
            print(f"Failed to set GPU memory limit: {e}")
            return False
    
    def get_gpu_system_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU system status."""
        return {
            'gpu_enabled': self.gpu_enabled,
            'gpu_fallback_mode': self.gpu_fallback_mode,
            'gpu_performance_target': self.gpu_performance_target,
            'gpu_renderer_initialized': hasattr(self, 'gpu_renderer') and self.gpu_renderer is not None,
            'gpu_optimizer_running': self.gpu_optimizer.optimization_running if self.gpu_optimizer else False,
            'current_trajectory_points': len(self.trajectory_points),
            'max_trajectory_points': self.max_trajectory_points,
            'current_rendering_mode': self.get_rendering_mode(),
            'gpu_rendering_mode': self.get_gpu_rendering_mode()
        }


# Global instance for TouchDesigner
if not hasattr(op, 'parametric_integration'):
    op.parametric_integration = TouchDesignerParametricIntegration()


# TouchDesigner Interface Functions
def initialize_system(ops_dict=None):
    """Initialize the parametric integration system."""
    if ops_dict is None:
        # Default TouchDesigner operator references
        ops_dict = {
            'gesture_params': op('gesture_params'),
            'parametric_params': op('parametric_params'),
            'trajectory_data': op('trajectory_data'),
            'particle_params': op('particle_params')
        }
    
    op.parametric_integration.initialize_touchdesigner_operators(ops_dict)
    return True


def process_camera_frame(camera_data, frame_number):
    """Process camera frame - called by Execute DAT."""
    return op.parametric_integration.process_frame_update(camera_data, frame_number)


def set_rendering_mode(mode):
    """Set rendering mode - called from TouchDesigner UI."""
    return op.parametric_integration.set_rendering_mode(mode)


def get_current_parameters():
    """Get current parameters for display."""
    return op.parametric_integration.get_all_parameters()


def reset_system():
    """Reset the entire system."""
    op.parametric_integration.reset_trajectory()
    op.parametric_integration.gesture_bridge.reset_parameters()
    return True


def get_system_status():
    """Get complete system status."""
    integration = op.parametric_integration
    return {
        'mode': integration.get_rendering_mode(),
        'paused': integration.is_paused(),
        'parameters': integration.get_all_parameters(),
        'performance': integration.get_performance_stats(),
        'gesture_status': integration.get_gesture_status(),
        'trajectory_points': integration.get_trajectory_point_count()
    }


# Additional utility functions for TouchDesigner Execute DATs
def on_mode_change(prev_mode, new_mode):
    """Handle mode change from TouchDesigner parameter."""
    return set_rendering_mode(new_mode)


def on_parameter_change(param_name, value):
    """Handle parameter changes from TouchDesigner UI."""
    return op.parametric_integration.set_parameter(param_name, value)


def get_table_data_for_sop(table_name):
    """Get table data formatted for SOP creation."""
    integration = op.parametric_integration
    
    if table_name == 'trajectory':
        points_data = []
        for point in integration.trajectory_points:
            points_data.append([point['x'], point['y'], point.get('z', 0.0)])
        return points_data
        
    elif table_name == 'rods':
        rod_data = [
            [0.0, 0.0, 0.0],  # Origin
            [integration.rod_positions['rod1'][0], integration.rod_positions['rod1'][1], 0.0],
            [integration.rod_positions['rod2'][0], integration.rod_positions['rod2'][1], 0.0]
        ]
        return rod_data
    
    return []


# GPU-Enhanced TouchDesigner Interface Functions
def enable_gpu_rendering(enabled=True):
    """Enable or disable GPU rendering - called from TouchDesigner UI."""
    return op.parametric_integration.enable_gpu_rendering(enabled)


def set_gpu_rendering_mode(mode):
    """Set GPU rendering mode - called from TouchDesigner UI."""
    return op.parametric_integration.set_gpu_rendering_mode(mode)


def get_gpu_rendering_mode():
    """Get current GPU rendering mode."""
    return op.parametric_integration.get_gpu_rendering_mode()


def set_gpu_lod_level(level):
    """Set GPU Level of Detail manually (0-4)."""
    return op.parametric_integration.set_gpu_lod_level(int(level))


def enable_gpu_auto_lod(enabled=True):
    """Enable automatic GPU Level of Detail adjustment."""
    op.parametric_integration.enable_gpu_auto_lod(enabled)
    return enabled


def set_gpu_parameter(param_name, value):
    """Set GPU rendering parameter."""
    return op.parametric_integration.set_gpu_parameter(param_name, float(value))


def get_gpu_parameters():
    """Get all GPU rendering parameters."""
    return op.parametric_integration.get_gpu_parameters()


def get_gpu_performance_stats():
    """Get comprehensive GPU performance statistics."""
    return op.parametric_integration.get_gpu_performance_stats()


def optimize_gpu_for_point_count(point_count):
    """Optimize GPU settings for trajectory point count."""
    op.parametric_integration.optimize_gpu_for_point_count(int(point_count))
    return True


def reset_gpu_buffers():
    """Reset GPU buffers and memory pools."""
    return op.parametric_integration.reset_gpu_buffers()


def enable_gpu_instancing(enabled=True):
    """Enable GPU instancing for efficient rendering."""
    return op.parametric_integration.enable_gpu_instancing(enabled)


def set_gpu_memory_limit(limit_mb):
    """Set GPU memory usage limit in megabytes."""
    return op.parametric_integration.set_gpu_memory_limit(int(limit_mb))


def get_gpu_system_status():
    """Get comprehensive GPU system status."""
    return op.parametric_integration.get_gpu_system_status()


def initialize_gpu_system_with_ops(ops_dict=None):
    """Initialize GPU system with specific TouchDesigner operators."""
    if ops_dict is None:
        # Default GPU-optimized operator references
        ops_dict = {
            # Standard operators
            'gesture_params': op('gesture_params'),
            'parametric_params': op('parametric_params'),
            'trajectory_data': op('trajectory_data'),
            'particle_params': op('particle_params'),
            
            # GPU-optimized SOP operators
            'trajectory_add': op('trajectory_add'),
            'trajectory_resample': op('trajectory_resample'),
            'trajectory_instance': op('trajectory_instance'),
            'curve_geometry': op('curve_geometry'),
            
            # GLSL shader operators
            'parametric_shader': op('parametric_curves_glsl'),
            'trail_shader': op('trail_effect_glsl'),
            'volumetric_shader': op('volumetric_glsl')
        }
    
    op.parametric_integration.initialize_touchdesigner_operators(ops_dict)
    return True


def get_metal_optimized_settings():
    """Get macOS Metal API optimized settings."""
    integration = op.parametric_integration
    if integration.gpu_enabled:
        return integration.gpu_optimizer.create_metal_optimized_settings()
    return {}


def apply_gpu_optimization_preset(preset_name):
    """
    Apply GPU optimization preset.
    
    Presets:
    - 'maximum_quality': Best visual quality, may impact performance
    - 'balanced': Balance between quality and performance
    - 'performance': Prioritize frame rate over visual quality
    - 'battery_saver': Optimize for battery life on laptops
    """
    integration = op.parametric_integration
    if not integration.gpu_enabled:
        return False
        
    preset_settings = {
        'maximum_quality': {
            'lod_level': 4,
            'auto_lod': False,
            'instancing': True,
            'memory_limit_mb': 256,
            'gpu_parameters': {
                'curve_thickness': 0.008,
                'glow_intensity': 1.5,
                'volumetric_density': 0.8
            }
        },
        'balanced': {
            'lod_level': 3,
            'auto_lod': True,
            'instancing': True,
            'memory_limit_mb': 128,
            'gpu_parameters': {
                'curve_thickness': 0.005,
                'glow_intensity': 1.0,
                'volumetric_density': 0.5
            }
        },
        'performance': {
            'lod_level': 2,
            'auto_lod': True,
            'instancing': True,
            'memory_limit_mb': 64,
            'gpu_parameters': {
                'curve_thickness': 0.003,
                'glow_intensity': 0.7,
                'volumetric_density': 0.2
            }
        },
        'battery_saver': {
            'lod_level': 1,
            'auto_lod': True,
            'instancing': False,
            'memory_limit_mb': 32,
            'gpu_parameters': {
                'curve_thickness': 0.002,
                'glow_intensity': 0.5,
                'volumetric_density': 0.1
            }
        }
    }
    
    if preset_name not in preset_settings:
        return False
        
    settings = preset_settings[preset_name]
    
    try:
        # Apply LOD settings
        integration.set_gpu_lod_level(settings['lod_level'])
        integration.enable_gpu_auto_lod(settings['auto_lod'])
        
        # Apply instancing and memory settings
        integration.enable_gpu_instancing(settings['instancing'])
        integration.set_gpu_memory_limit(settings['memory_limit_mb'])
        
        # Apply GPU parameters
        for param_name, value in settings['gpu_parameters'].items():
            integration.set_gpu_parameter(param_name, value)
            
        return True
        
    except Exception as e:
        print(f"Failed to apply GPU optimization preset '{preset_name}': {e}")
        return False


def get_gpu_optimization_recommendations():
    """
    Get GPU optimization recommendations based on current performance.
    """
    integration = op.parametric_integration
    if not integration.gpu_enabled:
        return {'recommendations': ['Enable GPU rendering for better performance']}
        
    gpu_stats = integration.get_gpu_performance_stats()
    recommendations = []
    
    renderer_stats = gpu_stats.get('renderer_stats', {})
    current_fps = renderer_stats.get('current_fps', 30.0)
    memory_usage = renderer_stats.get('gpu_memory_usage_mb', 0)
    point_count = renderer_stats.get('point_count', 0)
    
    # Performance recommendations
    if current_fps < 20:
        recommendations.append('Consider reducing trajectory point count or enabling auto-LOD')
        recommendations.append('Try switching to "performance" optimization preset')
        
    if memory_usage > 100:
        recommendations.append('GPU memory usage is high, consider reducing memory limit')
        recommendations.append('Enable GPU instancing to reduce memory usage')
        
    if point_count > 2000:
        recommendations.append('High point count detected, enable auto-LOD for better performance')
        
    # Quality recommendations
    if current_fps > 50 and memory_usage < 50:
        recommendations.append('Performance headroom available, consider increasing visual quality')
        recommendations.append('Try switching to "maximum_quality" optimization preset')
        
    if not recommendations:
        recommendations.append('System performing well, no optimization needed')
        
    return {
        'recommendations': recommendations,
        'current_fps': current_fps,
        'memory_usage_mb': memory_usage,
        'point_count': point_count
    }