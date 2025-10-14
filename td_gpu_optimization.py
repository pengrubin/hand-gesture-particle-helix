#!/usr/bin/env python3
"""
TouchDesigner GPU Optimization Utilities
Advanced GPU optimization, memory management, and performance scaling for TouchDesigner integration.

This module provides specialized optimization tools for TouchDesigner's GPU pipeline,
including Metal API optimizations, memory pooling, and adaptive quality systems.
"""

import numpy as np
import time
import gc
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
import threading
import queue


class GPUMemoryType(Enum):
    """GPU memory type enumeration."""
    VERTEX_BUFFER = "vertex_buffer"
    INDEX_BUFFER = "index_buffer"
    TEXTURE_BUFFER = "texture_buffer"
    UNIFORM_BUFFER = "uniform_buffer"
    COMPUTE_BUFFER = "compute_buffer"


class OptimizationLevel(Enum):
    """GPU optimization level enumeration."""
    MAXIMUM_QUALITY = 0
    BALANCED = 1
    PERFORMANCE = 2
    BATTERY_SAVER = 3


@dataclass
class GPUBufferInfo:
    """GPU buffer information structure."""
    buffer_id: str
    memory_type: GPUMemoryType
    size_bytes: int
    usage_count: int
    last_access_time: float
    is_persistent: bool
    data_format: str


@dataclass
class PerformanceMetrics:
    """Performance metrics structure."""
    frame_time: float
    gpu_usage: float
    memory_usage: float
    draw_calls: int
    vertex_count: int
    triangle_count: int
    shader_switches: int


class TouchDesignerGPUOptimizer:
    """
    Advanced GPU optimization system for TouchDesigner parametric rendering.
    
    Features:
    - Metal API optimization for macOS
    - Dynamic memory pool management
    - Adaptive quality scaling based on performance
    - GPU resource usage monitoring
    - Automatic LOD adjustment
    - Shader compilation optimization
    - Batch rendering optimization
    """
    
    def __init__(self, max_memory_mb: int = 256):
        """Initialize GPU optimizer."""
        
        # Memory management
        self.max_gpu_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_gpu_memory = 0
        self.memory_pools: Dict[GPUMemoryType, List[GPUBufferInfo]] = {}
        self.active_buffers: Dict[str, GPUBufferInfo] = {}
        
        # Performance monitoring
        self.performance_history: List[PerformanceMetrics] = []
        self.target_fps = 30.0
        self.min_fps_threshold = 20.0
        self.performance_window_size = 60
        
        # Optimization state
        self.current_optimization_level = OptimizationLevel.BALANCED
        self.auto_optimization = True
        self.frame_skip_enabled = True
        self.dynamic_lod_enabled = True
        
        # GPU-specific optimizations
        self.metal_optimizations = {
            'use_vertex_amplification': True,
            'use_tile_memory': True,
            'use_memoryless_textures': True,
            'batch_similar_draws': True,
            'use_indirect_drawing': True
        }
        
        # Culling and LOD settings
        self.culling_settings = {
            'frustum_culling': True,
            'occlusion_culling': False,  # Disabled for real-time performance
            'distance_culling': True,
            'backface_culling': True,
            'small_triangle_culling': True
        }
        
        # Shader optimization
        self.shader_cache: Dict[str, Any] = {}
        self.shader_compile_queue = queue.Queue()
        self.precompile_shaders = True
        
        # Batching optimization
        self.batch_settings = {
            'max_batch_size': 1000,
            'batch_similar_materials': True,
            'sort_by_depth': True,
            'merge_small_batches': True
        }
        
        # Threading for async operations
        self.optimization_thread = None
        self.optimization_running = False
        
        # Initialize memory pools
        self._initialize_memory_pools()
        
    def _initialize_memory_pools(self) -> None:
        """Initialize GPU memory pools for different buffer types."""
        for memory_type in GPUMemoryType:
            self.memory_pools[memory_type] = []
    
    def start_optimization_thread(self) -> None:
        """Start background optimization thread."""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self.optimization_running = True
            self.optimization_thread = threading.Thread(
                target=self._optimization_worker,
                daemon=True
            )
            self.optimization_thread.start()
    
    def stop_optimization_thread(self) -> None:
        """Stop background optimization thread."""
        self.optimization_running = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=1.0)
    
    def _optimization_worker(self) -> None:
        """Background optimization worker thread."""
        while self.optimization_running:
            try:
                # Perform periodic optimizations
                self._cleanup_unused_buffers()
                self._optimize_memory_usage()
                self._precompile_pending_shaders()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(0.1)  # 10ms intervals
                
            except Exception as e:
                print(f"GPU Optimization thread error: {e}")
                time.sleep(1.0)
    
    def allocate_gpu_buffer(self, 
                          buffer_id: str,
                          memory_type: GPUMemoryType,
                          size_bytes: int,
                          data_format: str = "float32",
                          is_persistent: bool = False) -> bool:
        """
        Allocate GPU buffer with memory pool management.
        
        Args:
            buffer_id: Unique buffer identifier
            memory_type: Type of GPU memory buffer
            size_bytes: Size in bytes
            data_format: Data format (float32, uint32, etc.)
            is_persistent: Whether buffer should persist across frames
            
        Returns:
            True if allocation successful, False otherwise
        """
        
        # Check if we have enough memory
        if self.current_gpu_memory + size_bytes > self.max_gpu_memory:
            # Try to free some memory
            if not self._free_memory_for_allocation(size_bytes):
                return False
        
        # Check if we can reuse an existing buffer from pool
        reused_buffer = self._find_reusable_buffer(memory_type, size_bytes)
        
        if reused_buffer:
            # Reuse existing buffer
            reused_buffer.buffer_id = buffer_id
            reused_buffer.last_access_time = time.time()
            reused_buffer.usage_count += 1
            reused_buffer.is_persistent = is_persistent
            reused_buffer.data_format = data_format
            
            self.active_buffers[buffer_id] = reused_buffer
            
        else:
            # Create new buffer
            buffer_info = GPUBufferInfo(
                buffer_id=buffer_id,
                memory_type=memory_type,
                size_bytes=size_bytes,
                usage_count=1,
                last_access_time=time.time(),
                is_persistent=is_persistent,
                data_format=data_format
            )
            
            self.active_buffers[buffer_id] = buffer_info
            self.current_gpu_memory += size_bytes
        
        return True
    
    def deallocate_gpu_buffer(self, buffer_id: str) -> bool:
        """
        Deallocate GPU buffer and return to pool if appropriate.
        
        Args:
            buffer_id: Buffer identifier to deallocate
            
        Returns:
            True if deallocation successful
        """
        
        if buffer_id not in self.active_buffers:
            return False
        
        buffer_info = self.active_buffers[buffer_id]
        
        # Remove from active buffers
        del self.active_buffers[buffer_id]
        
        # If buffer is reusable, add to pool
        if not buffer_info.is_persistent and buffer_info.usage_count < 10:
            # Reset buffer for reuse
            buffer_info.buffer_id = ""
            buffer_info.last_access_time = time.time()
            self.memory_pools[buffer_info.memory_type].append(buffer_info)
        else:
            # Actually deallocate memory
            self.current_gpu_memory -= buffer_info.size_bytes
        
        return True
    
    def _find_reusable_buffer(self, 
                            memory_type: GPUMemoryType, 
                            size_bytes: int) -> Optional[GPUBufferInfo]:
        """Find a reusable buffer from the memory pool."""
        
        pool = self.memory_pools[memory_type]
        
        # Find buffer with similar size (within 25% tolerance)
        for i, buffer_info in enumerate(pool):
            size_ratio = buffer_info.size_bytes / size_bytes
            if 0.8 <= size_ratio <= 1.25:
                # Remove from pool and return
                return pool.pop(i)
        
        return None
    
    def _free_memory_for_allocation(self, required_bytes: int) -> bool:
        """Free memory to make space for new allocation."""
        
        freed_bytes = 0
        
        # First, try to free non-persistent buffers
        buffers_to_free = []
        for buffer_id, buffer_info in self.active_buffers.items():
            if not buffer_info.is_persistent:
                age = time.time() - buffer_info.last_access_time
                if age > 5.0:  # 5 seconds old
                    buffers_to_free.append(buffer_id)
        
        # Free old buffers
        for buffer_id in buffers_to_free:
            buffer_info = self.active_buffers[buffer_id]
            freed_bytes += buffer_info.size_bytes
            self.deallocate_gpu_buffer(buffer_id)
            
            if freed_bytes >= required_bytes:
                return True
        
        # If still not enough, clear memory pools
        if freed_bytes < required_bytes:
            for memory_type, pool in self.memory_pools.items():
                for buffer_info in pool:
                    freed_bytes += buffer_info.size_bytes
                pool.clear()
                
                if freed_bytes >= required_bytes:
                    break
        
        self.current_gpu_memory -= freed_bytes
        return freed_bytes >= required_bytes
    
    def _cleanup_unused_buffers(self) -> None:
        """Clean up unused buffers in memory pools."""
        
        current_time = time.time()
        
        for memory_type, pool in self.memory_pools.items():
            # Remove buffers older than 30 seconds
            pool[:] = [
                buffer for buffer in pool 
                if current_time - buffer.last_access_time < 30.0
            ]
    
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for optimization analysis."""
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window_size:
            self.performance_history = self.performance_history[-self.performance_window_size:]
        
        # Trigger auto-optimization if enabled
        if self.auto_optimization:
            self._analyze_performance_and_optimize()
    
    def _analyze_performance_and_optimize(self) -> None:
        """Analyze performance metrics and adjust optimization level."""
        
        if len(self.performance_history) < 10:
            return
        
        # Calculate average FPS over recent frames
        recent_metrics = self.performance_history[-10:]
        avg_frame_time = sum(m.frame_time for m in recent_metrics) / len(recent_metrics)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # Calculate GPU usage and memory pressure
        avg_gpu_usage = sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Decide optimization level
        new_optimization_level = self.current_optimization_level
        
        if current_fps < self.min_fps_threshold:
            # Performance too poor, increase optimization
            if self.current_optimization_level == OptimizationLevel.MAXIMUM_QUALITY:
                new_optimization_level = OptimizationLevel.BALANCED
            elif self.current_optimization_level == OptimizationLevel.BALANCED:
                new_optimization_level = OptimizationLevel.PERFORMANCE
            elif self.current_optimization_level == OptimizationLevel.PERFORMANCE:
                new_optimization_level = OptimizationLevel.BATTERY_SAVER
                
        elif current_fps > self.target_fps * 1.2:
            # Performance good, can reduce optimization
            if self.current_optimization_level == OptimizationLevel.BATTERY_SAVER:
                new_optimization_level = OptimizationLevel.PERFORMANCE
            elif self.current_optimization_level == OptimizationLevel.PERFORMANCE:
                new_optimization_level = OptimizationLevel.BALANCED
            elif self.current_optimization_level == OptimizationLevel.BALANCED:
                new_optimization_level = OptimizationLevel.MAXIMUM_QUALITY
        
        # Apply optimization level if changed
        if new_optimization_level != self.current_optimization_level:
            self._apply_optimization_level(new_optimization_level)
            self.current_optimization_level = new_optimization_level
    
    def _apply_optimization_level(self, level: OptimizationLevel) -> None:
        """Apply specific optimization level settings."""
        
        if level == OptimizationLevel.MAXIMUM_QUALITY:
            # Maximum quality settings
            self.culling_settings['distance_culling'] = False
            self.batch_settings['max_batch_size'] = 2000
            self.frame_skip_enabled = False
            
        elif level == OptimizationLevel.BALANCED:
            # Balanced settings
            self.culling_settings['distance_culling'] = True
            self.batch_settings['max_batch_size'] = 1000
            self.frame_skip_enabled = True
            
        elif level == OptimizationLevel.PERFORMANCE:
            # Performance-focused settings
            self.culling_settings['small_triangle_culling'] = True
            self.batch_settings['max_batch_size'] = 500
            self.batch_settings['merge_small_batches'] = True
            
        elif level == OptimizationLevel.BATTERY_SAVER:
            # Battery saving settings
            self.culling_settings['occlusion_culling'] = False  # Too expensive
            self.batch_settings['max_batch_size'] = 250
            self.metal_optimizations['use_tile_memory'] = True
    
    def optimize_for_touchdesigner_sops(self, 
                                       sop_ops: Dict[str, Any],
                                       point_count: int,
                                       target_fps: float = 30.0) -> Dict[str, Any]:
        """
        Optimize TouchDesigner SOP operators for performance.
        
        Args:
            sop_ops: Dictionary of TouchDesigner SOP operators
            point_count: Number of points being processed
            target_fps: Target frame rate
            
        Returns:
            Dictionary of optimization results
        """
        
        optimizations_applied = []
        
        # Optimize Add SOP for large point counts
        if 'trajectory_add' in sop_ops:
            add_sop = sop_ops['trajectory_add']
            if point_count > 1000:
                # Enable efficient point generation
                if hasattr(add_sop.par, 'method'):
                    add_sop.par.method = 'fast'  # Use fastest generation method
                optimizations_applied.append('add_sop_fast_mode')
        
        # Optimize Resample SOP
        if 'trajectory_resample' in sop_ops:
            resample_sop = sop_ops['trajectory_resample']
            
            # Adjust segments based on performance requirements
            if target_fps < 25:
                segments = min(8, max(4, 32 // (point_count // 100 + 1)))
            elif target_fps < 35:
                segments = min(16, max(8, 64 // (point_count // 100 + 1)))
            else:
                segments = min(32, 64 // (point_count // 100 + 1))
            
            if hasattr(resample_sop.par, 'segments'):
                resample_sop.par.segments = segments
            optimizations_applied.append(f'resample_segments_{segments}')
        
        # Optimize Instance2 COMP
        if 'trajectory_instance' in sop_ops:
            instance_comp = sop_ops['trajectory_instance']
            
            # Limit instances based on performance
            max_instances = min(point_count, int(2000 * (target_fps / 30.0)))
            if hasattr(instance_comp.par, 'instances'):
                instance_comp.par.instances = max_instances
            
            # Enable GPU instancing if available
            if hasattr(instance_comp.par, 'gpuinstancing'):
                instance_comp.par.gpuinstancing = True
                
            optimizations_applied.append(f'instance_limit_{max_instances}')
        
        return {
            'optimizations_applied': optimizations_applied,
            'estimated_performance_gain': len(optimizations_applied) * 0.1,
            'memory_usage_reduction': len(optimizations_applied) * 0.05
        }
    
    def optimize_shader_compilation(self, 
                                  shader_tops: Dict[str, Any],
                                  async_compile: bool = True) -> Dict[str, Any]:
        """
        Optimize GLSL shader compilation for TouchDesigner.
        
        Args:
            shader_tops: Dictionary of GLSL TOP operators
            async_compile: Whether to compile shaders asynchronously
            
        Returns:
            Dictionary of compilation results
        """
        
        compilation_results = {}
        
        for shader_name, shader_top in shader_tops.items():
            try:
                # Pre-warm shader compilation
                if async_compile:
                    # Add to compilation queue
                    self.shader_compile_queue.put((shader_name, shader_top))
                else:
                    # Compile immediately
                    self._compile_shader(shader_name, shader_top)
                
                compilation_results[shader_name] = 'queued' if async_compile else 'compiled'
                
            except Exception as e:
                compilation_results[shader_name] = f'error: {str(e)}'
        
        return compilation_results
    
    def _compile_shader(self, shader_name: str, shader_top: Any) -> None:
        """Compile individual shader."""
        
        # Cache compiled shader
        self.shader_cache[shader_name] = {
            'shader_top': shader_top,
            'compile_time': time.time(),
            'usage_count': 0
        }
    
    def _precompile_pending_shaders(self) -> None:
        """Precompile pending shaders from queue."""
        
        try:
            while not self.shader_compile_queue.empty():
                shader_name, shader_top = self.shader_compile_queue.get_nowait()
                self._compile_shader(shader_name, shader_top)
        except queue.Empty:
            pass
    
    def _optimize_memory_usage(self) -> None:
        """Optimize overall GPU memory usage."""
        
        # Force garbage collection if memory usage is high
        memory_usage_ratio = self.current_gpu_memory / self.max_gpu_memory
        
        if memory_usage_ratio > 0.8:
            # Aggressive cleanup
            gc.collect()
            self._cleanup_unused_buffers()
            
            # Clear old shader cache entries
            current_time = time.time()
            old_shaders = [
                name for name, info in self.shader_cache.items()
                if current_time - info['compile_time'] > 300  # 5 minutes old
                and info['usage_count'] < 5
            ]
            
            for shader_name in old_shaders:
                del self.shader_cache[shader_name]
    
    def create_metal_optimized_settings(self) -> Dict[str, Any]:
        """Create Metal API optimized settings for macOS."""
        
        return {
            # Memory settings
            'use_shared_memory': True,
            'memory_pool_size': self.max_gpu_memory // 4,  # 25% for pool
            'texture_streaming': True,
            
            # Rendering settings
            'vertex_amplification': self.metal_optimizations['use_vertex_amplification'],
            'tile_memory_optimization': self.metal_optimizations['use_tile_memory'],
            'memoryless_render_targets': self.metal_optimizations['use_memoryless_textures'],
            
            # Draw call optimization
            'batch_draws': self.metal_optimizations['batch_similar_draws'],
            'indirect_drawing': self.metal_optimizations['use_indirect_drawing'],
            'max_batch_size': self.batch_settings['max_batch_size'],
            
            # Culling settings
            'frustum_culling': self.culling_settings['frustum_culling'],
            'backface_culling': self.culling_settings['backface_culling'],
            'distance_culling': self.culling_settings['distance_culling'],
            
            # Quality settings
            'optimization_level': self.current_optimization_level.value,
            'dynamic_lod': self.dynamic_lod_enabled,
            'frame_skip': self.frame_skip_enabled
        }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        
        # Calculate average performance over recent history
        recent_metrics = self.performance_history[-10:] if self.performance_history else []
        
        if recent_metrics:
            avg_frame_time = sum(m.frame_time for m in recent_metrics) / len(recent_metrics)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            avg_gpu_usage = sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        else:
            avg_frame_time = avg_fps = avg_gpu_usage = avg_memory_usage = 0.0
        
        return {
            # Performance metrics
            'average_fps': avg_fps,
            'average_frame_time_ms': avg_frame_time * 1000,
            'average_gpu_usage_percent': avg_gpu_usage * 100,
            'average_memory_usage_percent': avg_memory_usage * 100,
            
            # Memory statistics
            'gpu_memory_used_mb': self.current_gpu_memory / (1024 * 1024),
            'gpu_memory_limit_mb': self.max_gpu_memory / (1024 * 1024),
            'memory_usage_ratio': self.current_gpu_memory / self.max_gpu_memory,
            'active_buffers': len(self.active_buffers),
            'pooled_buffers': sum(len(pool) for pool in self.memory_pools.values()),
            
            # Optimization state
            'optimization_level': self.current_optimization_level.name,
            'auto_optimization': self.auto_optimization,
            'frame_skip_enabled': self.frame_skip_enabled,
            'dynamic_lod_enabled': self.dynamic_lod_enabled,
            
            # Shader statistics
            'compiled_shaders': len(self.shader_cache),
            'pending_shader_compiles': self.shader_compile_queue.qsize(),
            
            # System health
            'optimization_thread_running': self.optimization_running,
            'performance_samples': len(self.performance_history)
        }
    
    def reset_optimization_state(self) -> None:
        """Reset optimizer to initial state."""
        
        # Clear all buffers
        self.active_buffers.clear()
        for pool in self.memory_pools.values():
            pool.clear()
        
        self.current_gpu_memory = 0
        
        # Clear performance history
        self.performance_history.clear()
        
        # Clear shader cache
        self.shader_cache.clear()
        
        # Reset optimization level
        self.current_optimization_level = OptimizationLevel.BALANCED
        
        # Clear compilation queue
        while not self.shader_compile_queue.empty():
            try:
                self.shader_compile_queue.get_nowait()
            except queue.Empty:
                break


# Global optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> TouchDesignerGPUOptimizer:
    """Get or create global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = TouchDesignerGPUOptimizer()
        _gpu_optimizer.start_optimization_thread()
    return _gpu_optimizer


# Utility functions for TouchDesigner integration

def initialize_gpu_optimizer(max_memory_mb: int = 256) -> TouchDesignerGPUOptimizer:
    """Initialize GPU optimizer with specific memory limit."""
    global _gpu_optimizer
    _gpu_optimizer = TouchDesignerGPUOptimizer(max_memory_mb)
    _gpu_optimizer.start_optimization_thread()
    return _gpu_optimizer


def optimize_touchdesigner_sops(sop_ops: Dict[str, Any], 
                              point_count: int, 
                              target_fps: float = 30.0) -> Dict[str, Any]:
    """Optimize TouchDesigner SOP operators."""
    optimizer = get_gpu_optimizer()
    return optimizer.optimize_for_touchdesigner_sops(sop_ops, point_count, target_fps)


def record_frame_performance(frame_time: float, 
                           gpu_usage: float = 0.0, 
                           memory_usage: float = 0.0,
                           draw_calls: int = 0,
                           vertex_count: int = 0) -> None:
    """Record frame performance metrics."""
    optimizer = get_gpu_optimizer()
    metrics = PerformanceMetrics(
        frame_time=frame_time,
        gpu_usage=gpu_usage,
        memory_usage=memory_usage,
        draw_calls=draw_calls,
        vertex_count=vertex_count,
        triangle_count=vertex_count // 3,
        shader_switches=0
    )
    optimizer.record_performance_metrics(metrics)


def get_metal_optimized_settings() -> Dict[str, Any]:
    """Get Metal API optimized settings for macOS."""
    optimizer = get_gpu_optimizer()
    return optimizer.create_metal_optimized_settings()


def cleanup_gpu_optimizer() -> None:
    """Clean up GPU optimizer resources."""
    global _gpu_optimizer
    if _gpu_optimizer:
        _gpu_optimizer.stop_optimization_thread()
        _gpu_optimizer.reset_optimization_state()
        _gpu_optimizer = None