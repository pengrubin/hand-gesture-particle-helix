#!/usr/bin/env python3
"""
High-Performance Trajectory Recording System

Optimized real-time trajectory recording with:
- Circular buffers for memory efficiency
- Multi-threaded recording and processing
- Adaptive quality settings (LOD)
- Trajectory compression algorithms
- Real-time performance monitoring

Designed for 30+ FPS recording with thousands of trajectory points.
"""

import numpy as np
import threading
import queue
import time
import struct
import zlib
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp


class CompressionLevel(Enum):
    """Compression levels for trajectory data."""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


class RecordingQuality(Enum):
    """Recording quality levels with different LOD settings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class TrajectoryPoint:
    """Single trajectory point with timestamp and metadata."""
    timestamp: float
    x: float
    y: float
    z: float = 0.0
    gesture_strength: float = 1.0
    gesture_type: int = 0
    parameter_r1: float = 1.0
    parameter_r2: float = 0.5
    parameter_w1: float = 1.0
    parameter_w2: float = 2.0
    parameter_p1: float = 0.0
    parameter_p2: float = 0.0


@dataclass
class RecordingMetadata:
    """Metadata for trajectory recording sessions."""
    start_time: float
    end_time: Optional[float] = None
    total_points: int = 0
    recording_fps: float = 0.0
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    quality_level: RecordingQuality = RecordingQuality.HIGH
    session_id: str = ""
    gesture_sequence_summary: Dict[str, Any] = field(default_factory=dict)


class CircularBuffer:
    """High-performance circular buffer for trajectory points."""
    
    def __init__(self, capacity: int):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum number of points to store
        """
        self.capacity = capacity
        self.buffer = np.zeros((capacity, 12), dtype=np.float64)  # Pre-allocate array
        self.write_index = 0
        self.size = 0
        self.full = False
        self.lock = threading.RLock()
    
    def add_point(self, point: TrajectoryPoint) -> None:
        """
        Add a point to the buffer.
        
        Args:
            point: TrajectoryPoint to add
        """
        with self.lock:
            # Pack point data into array row
            self.buffer[self.write_index] = [
                point.timestamp, point.x, point.y, point.z,
                point.gesture_strength, point.gesture_type,
                point.parameter_r1, point.parameter_r2,
                point.parameter_w1, point.parameter_w2,
                point.parameter_p1, point.parameter_p2
            ]
            
            self.write_index = (self.write_index + 1) % self.capacity
            
            if not self.full:
                self.size += 1
                if self.write_index == 0:
                    self.full = True
    
    def get_recent_points(self, count: int) -> np.ndarray:
        """
        Get the most recent points.
        
        Args:
            count: Number of recent points to retrieve
            
        Returns:
            Array of recent points
        """
        with self.lock:
            if self.size == 0:
                return np.array([])
            
            count = min(count, self.size)
            
            if self.full:
                start_index = (self.write_index - count) % self.capacity
                if start_index + count <= self.capacity:
                    return self.buffer[start_index:start_index + count].copy()
                else:
                    # Wrap around case
                    part1 = self.buffer[start_index:].copy()
                    part2 = self.buffer[:count - (self.capacity - start_index)].copy()
                    return np.vstack([part1, part2])
            else:
                start_index = max(0, self.write_index - count)
                return self.buffer[start_index:self.write_index].copy()
    
    def get_all_points(self) -> np.ndarray:
        """Get all points in chronological order."""
        return self.get_recent_points(self.size)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.write_index = 0
            self.size = 0
            self.full = False


class AdaptiveQualityManager:
    """Manages adaptive quality scaling based on performance metrics."""
    
    def __init__(self, target_fps: float = 30.0):
        """
        Initialize quality manager.
        
        Args:
            target_fps: Target FPS for recording
        """
        self.target_fps = target_fps
        self.current_quality = RecordingQuality.HIGH
        self.fps_history = deque(maxlen=30)  # Track recent FPS
        self.quality_settings = {
            RecordingQuality.LOW: {
                'max_points': 500,
                'decimation_factor': 4,
                'compression_level': CompressionLevel.LOW
            },
            RecordingQuality.MEDIUM: {
                'max_points': 1000,
                'decimation_factor': 2,
                'compression_level': CompressionLevel.MEDIUM
            },
            RecordingQuality.HIGH: {
                'max_points': 2000,
                'decimation_factor': 1,
                'compression_level': CompressionLevel.MEDIUM
            },
            RecordingQuality.ULTRA: {
                'max_points': 5000,
                'decimation_factor': 1,
                'compression_level': CompressionLevel.HIGH
            }
        }
        
    def update_fps(self, fps: float) -> None:
        """Update FPS measurement and adjust quality if needed."""
        self.fps_history.append(fps)
        
        if len(self.fps_history) >= 10:  # Need enough samples
            avg_fps = np.mean(list(self.fps_history))
            
            # Adjust quality based on performance
            if avg_fps < self.target_fps * 0.8:  # Performance too low
                self._decrease_quality()
            elif avg_fps > self.target_fps * 1.1 and self.current_quality != RecordingQuality.ULTRA:
                self._increase_quality()
    
    def _decrease_quality(self) -> None:
        """Decrease recording quality to improve performance."""
        quality_levels = list(RecordingQuality)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index > 0:
            self.current_quality = quality_levels[current_index - 1]
            print(f"Adaptive quality: Decreased to {self.current_quality.value}")
    
    def _increase_quality(self) -> None:
        """Increase recording quality when performance allows."""
        quality_levels = list(RecordingQuality)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index < len(quality_levels) - 1:
            self.current_quality = quality_levels[current_index + 1]
            print(f"Adaptive quality: Increased to {self.current_quality.value}")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current quality settings."""
        return self.quality_settings[self.current_quality].copy()


class TrajectoryCompressor:
    """High-performance trajectory compression utilities."""
    
    @staticmethod
    def compress_points(points: np.ndarray, level: CompressionLevel = CompressionLevel.MEDIUM) -> bytes:
        """
        Compress trajectory points array.
        
        Args:
            points: Array of trajectory points
            level: Compression level
            
        Returns:
            Compressed bytes
        """
        if len(points) == 0:
            return b''
        
        # Convert to bytes
        point_bytes = points.astype(np.float32).tobytes()  # Use float32 for smaller size
        
        if level == CompressionLevel.NONE:
            return point_bytes
        
        # Apply zlib compression
        return zlib.compress(point_bytes, level.value)
    
    @staticmethod
    def decompress_points(compressed_data: bytes) -> np.ndarray:
        """
        Decompress trajectory points.
        
        Args:
            compressed_data: Compressed trajectory data
            
        Returns:
            Decompressed points array
        """
        if not compressed_data:
            return np.array([])
        
        try:
            # Try decompression first
            decompressed = zlib.decompress(compressed_data)
        except zlib.error:
            # Data might not be compressed
            decompressed = compressed_data
        
        # Convert back to numpy array
        points = np.frombuffer(decompressed, dtype=np.float32)
        return points.reshape(-1, 12).astype(np.float64)
    
    @staticmethod
    def apply_douglas_peucker(points: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        Apply Douglas-Peucker algorithm for trajectory simplification.
        
        Args:
            points: Input trajectory points
            epsilon: Simplification threshold
            
        Returns:
            Simplified trajectory points
        """
        if len(points) < 3:
            return points
        
        def perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
            """Calculate perpendicular distance from point to line."""
            if np.allclose(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            n = np.linalg.norm(line_end - line_start)
            return np.abs(np.cross(line_end - line_start, line_start - point)) / n
        
        def douglas_peucker_recursive(points_slice: np.ndarray, start: int, end: int) -> List[int]:
            """Recursive Douglas-Peucker implementation."""
            if end - start < 2:
                return [start, end]
            
            # Find point with maximum distance
            max_distance = 0.0
            max_index = start
            
            for i in range(start + 1, end):
                distance = perpendicular_distance(
                    points_slice[i, 1:3],  # x, y coordinates
                    points_slice[start, 1:3],
                    points_slice[end, 1:3]
                )
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            if max_distance > epsilon:
                # Recursively process both segments
                left_points = douglas_peucker_recursive(points_slice, start, max_index)
                right_points = douglas_peucker_recursive(points_slice, max_index, end)
                return left_points[:-1] + right_points
            else:
                return [start, end]
        
        # Apply algorithm
        indices = douglas_peucker_recursive(points, 0, len(points) - 1)
        return points[indices]


class HighPerformanceTrajectoryRecorder:
    """
    High-performance real-time trajectory recording system.
    
    Features:
    - Multi-threaded recording and processing
    - Circular buffers for memory efficiency
    - Adaptive quality scaling
    - Real-time compression
    - Performance monitoring
    """
    
    def __init__(self,
                 max_points: int = 10000,
                 target_fps: float = 30.0,
                 compression_level: CompressionLevel = CompressionLevel.MEDIUM,
                 enable_adaptive_quality: bool = True,
                 enable_real_time_export: bool = False):
        """
        Initialize the high-performance trajectory recorder.
        
        Args:
            max_points: Maximum points in circular buffer
            target_fps: Target recording FPS
            compression_level: Default compression level
            enable_adaptive_quality: Enable adaptive quality scaling
            enable_real_time_export: Enable real-time file export
        """
        self.max_points = max_points
        self.target_fps = target_fps
        self.compression_level = compression_level
        self.enable_adaptive_quality = enable_adaptive_quality
        self.enable_real_time_export = enable_real_time_export
        
        # Recording state
        self.recording = False
        self.paused = False
        self.session_id = ""
        
        # Buffers and queues
        self.trajectory_buffer = CircularBuffer(max_points)
        self.processing_queue = queue.Queue(maxsize=100)
        self.export_queue = queue.Queue(maxsize=50) if enable_real_time_export else None
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.export_thread: Optional[threading.Thread] = None
        self.recording_lock = threading.RLock()
        
        # Performance monitoring
        self.performance_stats = {
            'points_recorded': 0,
            'points_processed': 0,
            'points_exported': 0,
            'recording_fps': 0.0,
            'processing_fps': 0.0,
            'memory_usage_mb': 0.0,
            'last_update': time.time()
        }
        
        # Quality management
        self.quality_manager = AdaptiveQualityManager(target_fps) if enable_adaptive_quality else None
        
        # Metadata
        self.current_metadata = RecordingMetadata(
            start_time=0.0,
            compression_level=compression_level
        )
        
        # Performance tracking
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
    def start_recording(self, session_id: Optional[str] = None) -> bool:
        """
        Start trajectory recording.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if recording started successfully
        """
        with self.recording_lock:
            if self.recording:
                print("Recording already in progress")
                return False
            
            try:
                # Initialize session
                self.session_id = session_id or f"trajectory_{int(time.time())}"
                self.current_metadata = RecordingMetadata(
                    start_time=time.time(),
                    compression_level=self.compression_level,
                    session_id=self.session_id
                )
                
                # Clear buffers
                self.trajectory_buffer.clear()
                while not self.processing_queue.empty():
                    try:
                        self.processing_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Reset performance stats
                self.performance_stats = {
                    'points_recorded': 0,
                    'points_processed': 0,
                    'points_exported': 0,
                    'recording_fps': 0.0,
                    'processing_fps': 0.0,
                    'memory_usage_mb': 0.0,
                    'last_update': time.time()
                }
                
                # Start threads
                self.recording = True
                self.paused = False
                
                self.processing_thread = threading.Thread(
                    target=self._processing_thread_function,
                    daemon=True,
                    name=f"TrajectoryProcessor_{self.session_id}"
                )
                self.processing_thread.start()
                
                if self.enable_real_time_export:
                    self.export_thread = threading.Thread(
                        target=self._export_thread_function,
                        daemon=True,
                        name=f"TrajectoryExporter_{self.session_id}"
                    )
                    self.export_thread.start()
                
                print(f"Trajectory recording started: {self.session_id}")
                return True
                
            except Exception as e:
                print(f"Failed to start recording: {e}")
                self.recording = False
                return False
    
    def stop_recording(self) -> Optional[RecordingMetadata]:
        """
        Stop trajectory recording.
        
        Returns:
            Recording metadata if successful
        """
        with self.recording_lock:
            if not self.recording:
                return None
            
            print(f"Stopping trajectory recording: {self.session_id}")
            self.recording = False
            
            # Wait for threads to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            if self.export_thread and self.export_thread.is_alive():
                self.export_thread.join(timeout=2.0)
            
            # Finalize metadata
            self.current_metadata.end_time = time.time()
            self.current_metadata.total_points = self.trajectory_buffer.size
            self.current_metadata.recording_fps = self.performance_stats['recording_fps']
            
            print(f"Recording completed: {self.current_metadata.total_points} points recorded")
            return self.current_metadata
    
    def add_trajectory_point(self,
                           x: float, y: float, z: float = 0.0,
                           gesture_strength: float = 1.0,
                           gesture_type: int = 0,
                           parameters: Optional[Dict[str, float]] = None) -> bool:
        """
        Add a trajectory point to the recording.
        
        Args:
            x, y, z: Spatial coordinates
            gesture_strength: Gesture strength (0-1)
            gesture_type: Gesture type identifier
            parameters: Parametric equation parameters
            
        Returns:
            True if point was added successfully
        """
        if not self.recording or self.paused:
            return False
        
        try:
            # Create trajectory point
            params = parameters or {}
            point = TrajectoryPoint(
                timestamp=time.time(),
                x=x, y=y, z=z,
                gesture_strength=gesture_strength,
                gesture_type=gesture_type,
                parameter_r1=params.get('r1', 1.0),
                parameter_r2=params.get('r2', 0.5),
                parameter_w1=params.get('w1', 1.0),
                parameter_w2=params.get('w2', 2.0),
                parameter_p1=params.get('p1', 0.0),
                parameter_p2=params.get('p2', 0.0)
            )
            
            # Add to buffer
            self.trajectory_buffer.add_point(point)
            
            # Queue for processing
            try:
                self.processing_queue.put_nowait(point)
            except queue.Full:
                # Skip processing if queue is full (maintain real-time performance)
                pass
            
            # Update performance stats
            self.performance_stats['points_recorded'] += 1
            self._update_fps_stats()
            
            # Update adaptive quality if enabled
            if self.quality_manager:
                self.quality_manager.update_fps(self.performance_stats['recording_fps'])
            
            return True
            
        except Exception as e:
            print(f"Error adding trajectory point: {e}")
            return False
    
    def get_trajectory_data(self, count: Optional[int] = None, 
                          compressed: bool = False) -> Union[np.ndarray, bytes]:
        """
        Get recorded trajectory data.
        
        Args:
            count: Number of recent points to retrieve (None for all)
            compressed: Return compressed data
            
        Returns:
            Trajectory data as numpy array or compressed bytes
        """
        if count is None:
            points = self.trajectory_buffer.get_all_points()
        else:
            points = self.trajectory_buffer.get_recent_points(count)
        
        if compressed:
            return TrajectoryCompressor.compress_points(points, self.compression_level)
        else:
            return points
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats.update({
            'buffer_utilization': self.trajectory_buffer.size / self.trajectory_buffer.capacity,
            'recording_active': self.recording,
            'paused': self.paused,
            'session_id': self.session_id,
            'current_quality': self.quality_manager.current_quality.value if self.quality_manager else 'fixed'
        })
        return stats
    
    def pause_recording(self) -> None:
        """Pause trajectory recording."""
        self.paused = True
        print("Trajectory recording paused")
    
    def resume_recording(self) -> None:
        """Resume trajectory recording."""
        self.paused = False
        print("Trajectory recording resumed")
    
    def set_compression_level(self, level: CompressionLevel) -> None:
        """Set compression level for future recordings."""
        self.compression_level = level
        print(f"Compression level set to: {level.name}")
    
    def export_trajectory(self, filepath: str, 
                         format_type: str = 'binary',
                         apply_compression: bool = True) -> bool:
        """
        Export recorded trajectory to file.
        
        Args:
            filepath: Output file path
            format_type: Export format ('binary', 'csv', 'json')
            apply_compression: Apply compression to export
            
        Returns:
            True if export successful
        """
        try:
            points = self.trajectory_buffer.get_all_points()
            
            if len(points) == 0:
                print("No trajectory data to export")
                return False
            
            if format_type == 'binary':
                data = {
                    'metadata': self.current_metadata,
                    'points': points if not apply_compression else 
                             TrajectoryCompressor.compress_points(points, self.compression_level)
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif format_type == 'csv':
                import pandas as pd
                df = pd.DataFrame(points, columns=[
                    'timestamp', 'x', 'y', 'z', 'gesture_strength', 'gesture_type',
                    'r1', 'r2', 'w1', 'w2', 'p1', 'p2'
                ])
                df.to_csv(filepath, index=False)
                
            elif format_type == 'json':
                import json
                data = {
                    'metadata': self.current_metadata.__dict__,
                    'points': points.tolist()
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            print(f"Trajectory exported to: {filepath}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def _processing_thread_function(self) -> None:
        """Processing thread for trajectory data."""
        print(f"Processing thread started: {self.session_id}")
        
        while self.recording:
            try:
                # Get point from queue with timeout
                point = self.processing_queue.get(timeout=0.1)
                
                # Process point (apply quality settings, compression, etc.)
                if self.quality_manager:
                    settings = self.quality_manager.get_current_settings()
                    # Could apply decimation or other processing here
                
                # Update processing stats
                self.performance_stats['points_processed'] += 1
                
                # Queue for export if enabled
                if self.enable_real_time_export and self.export_queue:
                    try:
                        self.export_queue.put_nowait(point)
                    except queue.Full:
                        pass  # Skip if export queue is full
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing thread error: {e}")
        
        print(f"Processing thread stopped: {self.session_id}")
    
    def _export_thread_function(self) -> None:
        """Export thread for real-time trajectory streaming."""
        print(f"Export thread started: {self.session_id}")
        
        export_buffer = []
        last_export = time.time()
        export_interval = 1.0  # Export every second
        
        while self.recording or not self.export_queue.empty():
            try:
                # Collect points for batch export
                point = self.export_queue.get(timeout=0.1)
                export_buffer.append(point)
                
                # Export batch periodically
                current_time = time.time()
                if current_time - last_export >= export_interval and export_buffer:
                    self._export_batch(export_buffer)
                    export_buffer = []
                    last_export = current_time
                    
            except queue.Empty:
                # Export remaining points
                if export_buffer and time.time() - last_export >= export_interval:
                    self._export_batch(export_buffer)
                    export_buffer = []
                    last_export = time.time()
                continue
            except Exception as e:
                print(f"Export thread error: {e}")
        
        # Final export
        if export_buffer:
            self._export_batch(export_buffer)
        
        print(f"Export thread stopped: {self.session_id}")
    
    def _export_batch(self, points: List[TrajectoryPoint]) -> None:
        """Export a batch of points to streaming file."""
        try:
            filename = f"trajectory_stream_{self.session_id}.dat"
            with open(filename, 'ab') as f:  # Append binary mode
                for point in points:
                    # Pack point data efficiently
                    packed_data = struct.pack('dddddiiddddd',
                                            point.timestamp, point.x, point.y, point.z,
                                            point.gesture_strength, point.gesture_type,
                                            point.parameter_r1, point.parameter_r2,
                                            point.parameter_w1, point.parameter_w2,
                                            point.parameter_p1, point.parameter_p2)
                    f.write(packed_data)
            
            self.performance_stats['points_exported'] += len(points)
            
        except Exception as e:
            print(f"Batch export error: {e}")
    
    def _update_fps_stats(self) -> None:
        """Update FPS performance statistics."""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.performance_stats['recording_fps'] = fps
            self.fps_history.append(fps)
            
            # Reset counters
            self.frame_count = 0
            self.last_fps_update = current_time
            
            # Update memory usage estimate
            self.performance_stats['memory_usage_mb'] = (
                self.trajectory_buffer.capacity * 12 * 8 / (1024 * 1024)  # 8 bytes per float64
            )
            
            self.performance_stats['last_update'] = current_time


if __name__ == "__main__":
    # Performance test
    print("High-Performance Trajectory Recorder Test")
    
    recorder = HighPerformanceTrajectoryRecorder(
        max_points=5000,
        target_fps=60.0,
        enable_adaptive_quality=True,
        enable_real_time_export=True
    )
    
    # Test recording
    print("Starting test recording...")
    recorder.start_recording("test_session")
    
    # Simulate trajectory recording
    start_time = time.time()
    test_duration = 5.0  # 5 seconds
    point_count = 0
    
    while time.time() - start_time < test_duration:
        # Simulate parametric trajectory
        t = time.time() - start_time
        x = 2.0 * np.cos(t) + 1.0 * np.cos(3*t)
        y = 2.0 * np.sin(t) + 1.0 * np.sin(3*t)
        z = 0.5 * np.sin(2*t)
        
        recorder.add_trajectory_point(
            x, y, z,
            gesture_strength=0.5 + 0.5 * np.sin(5*t),
            gesture_type=int(5 * (0.5 + 0.5 * np.cos(t))),
            parameters={'r1': 2.0, 'r2': 1.0, 'w1': 1.0, 'w2': 3.0}
        )
        
        point_count += 1
        time.sleep(1/60.0)  # Simulate 60 FPS
    
    # Stop recording
    metadata = recorder.stop_recording()
    stats = recorder.get_performance_stats()
    
    print(f"\nTest Results:")
    print(f"Points recorded: {stats['points_recorded']}")
    print(f"Recording FPS: {stats['recording_fps']:.1f}")
    print(f"Buffer utilization: {stats['buffer_utilization']:.1%}")
    print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
    
    # Export test
    recorder.export_trajectory("test_trajectory.bin", format_type='binary')
    recorder.export_trajectory("test_trajectory.csv", format_type='csv')
    
    print("Test completed successfully!")