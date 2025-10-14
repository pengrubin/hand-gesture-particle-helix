#!/usr/bin/env python3
"""
Live Gesture-Parametric System
Complete real-time integration of MediaPipe gesture recognition with parametric equation visualization.

This system provides:
- Real-time camera input and gesture detection
- Twelve-tone scale parameter mapping
- Live parametric equation visualization
- Interactive controls and status display
- Smooth parameter transitions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
from typing import Dict, Optional, Tuple, Any
import sys
import argparse

from gesture_parametric_bridge import GestureParametricBridge, HandAssignment
from parametric_equation_renderer import ParametricEquationRenderer
from trajectory_recorder import HighPerformanceTrajectoryRecorder, CompressionLevel, TrajectoryPoint
from trajectory_visualizer import PerformanceOptimizedTrajectoryVisualizer, RenderingSettings, RenderMode, LODLevel
from performance_monitor import ComprehensivePerformanceMonitor


class LiveGestureParametricSystem:
    """
    Complete real-time gesture-controlled parametric equation visualization system.
    
    Features:
    - Dual-threaded operation: camera processing + visualization
    - Real-time parameter updates from hand gestures
    - Interactive controls (keyboard shortcuts)
    - Status overlay and parameter display
    - Recording capabilities
    """
    
    def __init__(self,
                 camera_id: int = 0,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 r_max: float = 2.5,
                 smoothing_factor: float = 0.85,
                 visualization_fps: int = 60,
                 camera_fps: int = 30,
                 hand_assignment: HandAssignment = HandAssignment.LEFT_R1_RIGHT_R2):
        """
        Initialize the live system.
        
        Args:
            camera_id: Camera device ID
            camera_width: Camera capture width
            camera_height: Camera capture height
            r_max: Maximum radius for twelve-tone mapping
            smoothing_factor: Parameter smoothing factor
            visualization_fps: Visualization refresh rate
            camera_fps: Camera processing rate
            hand_assignment: Hand-to-parameter assignment strategy
        """
        # System configuration
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.visualization_fps = visualization_fps
        self.camera_fps = camera_fps
        
        # Initialize core components
        self.bridge = GestureParametricBridge(
            r_max=r_max,
            smoothing_factor=smoothing_factor,
            hand_assignment=hand_assignment,
            auto_pause=True
        )
        
        self.renderer = ParametricEquationRenderer(
            r1=1.0, r2=0.5,
            w1=1.0, w2=2.0,
            p1=0.0, p2=0.0,
            max_theta=8*np.pi,
            num_points=1000,
            trail_length=300
        )
        
        # Connect bridge and renderer
        self.bridge.set_renderer(self.renderer)
        
        # Threading and communication
        self.camera_thread = None
        self.parameter_queue = queue.Queue(maxsize=10)
        self.status_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # Camera state
        self.camera = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Control state
        self.show_camera = True
        self.show_status = True
        
        # High-performance trajectory recording
        self.trajectory_recorder = HighPerformanceTrajectoryRecorder(
            max_points=10000,
            target_fps=visualization_fps,
            compression_level=CompressionLevel.LZ4,
            enable_adaptive_quality=True,
            enable_real_time_export=False
        )
        
        # Advanced trajectory visualization
        viz_settings = RenderingSettings(
            render_mode=RenderMode.FADE_TRAIL,
            lod_level=LODLevel.HIGH,
            max_visible_points=2000,
            color_by_velocity=True,
            update_frequency_hz=visualization_fps
        )
        self.trajectory_visualizer = PerformanceOptimizedTrajectoryVisualizer(
            settings=viz_settings,
            enable_3d=True
        )
        
        # Performance monitoring
        self.performance_monitor = ComprehensivePerformanceMonitor(
            target_fps=visualization_fps,
            enable_dashboard=False  # Separate window
        )
        
        # Recording state
        self.recording = False
        self.record_start_time = None
        self.trajectory_points = []
        
        # Performance tracking
        self.camera_fps_actual = 0.0
        self.viz_fps_actual = 0.0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if camera initialized successfully
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            # Verify camera settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
            
    def camera_thread_function(self) -> None:
        """Camera processing thread function."""
        target_frame_time = 1.0 / self.camera_fps
        last_process_time = 0
        
        print("Camera thread started")
        
        while self.running:
            current_time = time.time()
            
            # Throttle to target FPS
            if current_time - last_process_time < target_frame_time:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Warning: Failed to capture frame")
                    continue
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process gesture detection
                processed_frame, parameters = self.bridge.process_frame(frame)
                
                # Update shared frame
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                # Send parameters to visualization thread
                try:
                    self.parameter_queue.put_nowait(parameters)
                except queue.Full:
                    # Skip if queue is full (drop frames to maintain real-time)
                    pass
                
                # Send status information
                try:
                    gesture_info = self.bridge.get_gesture_info()
                    self.status_queue.put_nowait(gesture_info)
                except queue.Full:
                    pass
                
                # Record trajectory data if recording
                if self.recording:
                    # Calculate current position from parameters
                    theta = (current_time - self.record_start_time) * 2.0  # Adjust speed as needed
                    z = self.renderer.compute_complex_point(theta)
                    
                    # Create trajectory point
                    gesture_info = self.bridge.get_gesture_info()
                    gesture_strength = np.mean([info.get('openness', 0.5) for info in gesture_info.get('hand_data', [])])
                    gesture_type = len(gesture_info.get('digit_gestures', []))
                    
                    # Add to high-performance recorder
                    self.trajectory_recorder.add_trajectory_point(
                        x=z.real, y=z.imag, z=0.0,
                        gesture_strength=gesture_strength,
                        gesture_type=gesture_type,
                        parameters=parameters
                    )
                    
                    # Store for visualization
                    point = TrajectoryPoint(
                        timestamp=current_time,
                        x=z.real, y=z.imag, z=0.0,
                        gesture_strength=gesture_strength,
                        gesture_type=gesture_type,
                        **{k: v for k, v in parameters.items() if k in ['r1', 'r2', 'w1', 'w2', 'p1', 'p2']}
                    )
                    self.trajectory_points.append(point)
                    
                    # Update trajectory visualizer
                    if len(self.trajectory_points) > 10:  # Update when we have enough points
                        positions = np.array([[p.x, p.y, p.z] for p in self.trajectory_points])
                        timestamps = np.array([p.timestamp for p in self.trajectory_points])
                        gesture_data = {
                            'strength': np.array([p.gesture_strength for p in self.trajectory_points]),
                            'type': np.array([p.gesture_type for p in self.trajectory_points])
                        }
                        self.trajectory_visualizer.update_trajectory_data(positions, timestamps, gesture_data)
                
                last_process_time = current_time
                
                # Update camera FPS counter
                self.frame_count += 1
                if current_time - self.last_fps_update >= 1.0:
                    self.camera_fps_actual = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                    
            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.1)  # Brief pause on error
                
        print("Camera thread stopped")
        
    def update_visualization(self, frame_num: int) -> list:
        """
        Visualization update function for matplotlib animation.
        
        Args:
            frame_num: Frame number from FuncAnimation
            
        Returns:
            List of updated plot elements
        """
        try:
            # Get latest parameters if available
            try:
                while True:
                    parameters = self.parameter_queue.get_nowait()
                    self.renderer.update_parameters(**parameters)
            except queue.Empty:
                pass  # Use last parameters
            
            # Update renderer animation
            renderer_elements = self.renderer.animate_frame(frame_num)
            
            # Update performance monitor
            points_rendered = len(self.trajectory_points) if self.trajectory_points else 0
            self.performance_monitor.update_frame(points_rendered=points_rendered)
            
            # Get adaptive quality settings
            if frame_num % 30 == 0:  # Check every 30 frames
                settings, actions = self.performance_monitor.get_adaptive_quality_settings()
                if actions:
                    print(f"Performance adaptation: {', '.join(actions)}")
                    # Apply settings to visualizer
                    if 'max_points' in settings:
                        self.trajectory_visualizer.settings.max_visible_points = settings['max_points']
                    if 'render_mode' in settings:
                        mode_map = {
                            'points': RenderMode.POINT_CLOUD,
                            'solid': RenderMode.SOLID_LINE,
                            'fade': RenderMode.FADE_TRAIL,
                            'particles': RenderMode.PARTICLE_BASED
                        }
                        if settings['render_mode'] in mode_map:
                            self.trajectory_visualizer.set_render_mode(mode_map[settings['render_mode']])
            
            return renderer_elements
            
        except Exception as e:
            print(f"Visualization update error: {e}")
            return []
            
    def handle_keyboard_input(self) -> None:
        """Handle keyboard input for interactive controls."""
        if not self.show_camera:
            return
            
        try:
            # Non-blocking key check for OpenCV window
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit application
                self.stop()
                
            elif key == ord('c'):
                # Toggle camera display
                self.show_camera = not self.show_camera
                if not self.show_camera:
                    cv2.destroyAllWindows()
                print(f"Camera display: {'ON' if self.show_camera else 'OFF'}")
                
            elif key == ord('s'):
                # Toggle status display
                self.show_status = not self.show_status
                print(f"Status display: {'ON' if self.show_status else 'OFF'}")
                
            elif key == ord('r'):
                # Toggle recording
                if not self.recording:
                    self.start_recording()
                else:
                    self.stop_recording()
                    
            elif key == ord('v'):
                # Toggle visualization mode
                current_mode = self.trajectory_visualizer.settings.render_mode
                modes = list(RenderMode)
                current_index = modes.index(current_mode)
                next_mode = modes[(current_index + 1) % len(modes)]
                self.trajectory_visualizer.set_render_mode(next_mode)
                print(f"Visualization mode: {next_mode.value}")
                
            elif key == ord('l'):
                # Toggle LOD level
                current_lod = self.trajectory_visualizer.settings.lod_level
                lods = list(LODLevel)
                current_index = lods.index(current_lod)
                next_lod = lods[(current_index + 1) % len(lods)]
                self.trajectory_visualizer.set_lod_level(next_lod)
                print(f"LOD level: {next_lod.name}")
                
            elif key == ord('m'):
                # Show performance dashboard
                self.performance_monitor.show_dashboard()
                print("Performance dashboard opened")
                
            elif key == ord('x'):
                # Export trajectory
                if self.trajectory_points:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"trajectory_export_{timestamp}.trj"
                    success = self.trajectory_recorder.export_trajectory(filename, 'binary', True)
                    if success:
                        print(f"Trajectory exported: {filename}")
                    else:
                        print("Export failed")
                    
            elif key == ord('p'):
                # Toggle auto-pause
                current_auto = self.bridge.auto_pause
                self.bridge.set_auto_pause(not current_auto)
                print(f"Auto-pause: {'ON' if not current_auto else 'OFF'}")
                
            elif key == ord('1'):
                # Change hand assignment to left->r1, right->r2
                self.bridge.set_hand_assignment(HandAssignment.LEFT_R1_RIGHT_R2)
                print("Hand assignment: Left→r1, Right→r2")
                
            elif key == ord('2'):
                # Change hand assignment to right->r1, left->r2
                self.bridge.set_hand_assignment(HandAssignment.RIGHT_R1_LEFT_R2)
                print("Hand assignment: Right→r1, Left→r2")
                
            elif key == ord('3'):
                # Change hand assignment to dominant primary
                self.bridge.set_hand_assignment(HandAssignment.DOMINANT_PRIMARY)
                print("Hand assignment: Dominant primary")
                
            elif key == ord('+') or key == ord('='):
                # Increase smoothing
                current = self.bridge.smoothing_factor
                new_smooth = min(0.95, current + 0.05)
                self.bridge.set_smoothing_factor(new_smooth)
                print(f"Smoothing: {new_smooth:.2f}")
                
            elif key == ord('-'):
                # Decrease smoothing
                current = self.bridge.smoothing_factor
                new_smooth = max(0.1, current - 0.05)
                self.bridge.set_smoothing_factor(new_smooth)
                print(f"Smoothing: {new_smooth:.2f}")
                
            elif key == ord(' '):
                # Reset parameters
                self.bridge.reset_parameters()
                self.renderer.reset_animation()
                print("Parameters reset")
                
        except Exception as e:
            print(f"Keyboard input error: {e}")
            
    def update_camera_display(self) -> None:
        """Update camera display window."""
        if not self.show_camera:
            return
            
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    display_frame = self.current_frame.copy()
                else:
                    return
            
            # Add status overlay if enabled
            if self.show_status:
                try:
                    status_info = self.status_queue.get_nowait()
                    self.draw_status_overlay(display_frame, status_info)
                except queue.Empty:
                    pass  # Use last status
            
            # Add control hints
            self.draw_control_hints(display_frame)
            
            cv2.imshow('Live Gesture-Parametric System', display_frame)
            
        except Exception as e:
            print(f"Camera display error: {e}")
            
    def draw_status_overlay(self, frame: np.ndarray, status_info: Dict[str, Any]) -> None:
        """Draw status information overlay on frame."""
        height, width = frame.shape[:2]
        
        # Status background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        y_offset = 30
        line_height = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        status_lines = [
            f"Hands: {status_info.get('hands_detected', 0)}",
            f"Gestures: {status_info.get('digit_gestures', [])}",
            f"Paused: {status_info.get('is_paused', False)}",
            f"Assignment: {status_info.get('hand_assignment', 'unknown')}",
            "",
            "Parameters:",
            f"r1 = {status_info.get('current_parameters', {}).get('r1', 0):.2f}",
            f"r2 = {status_info.get('current_parameters', {}).get('r2', 0):.2f}",
            f"w1 = {status_info.get('current_parameters', {}).get('w1', 0):.2f}",
            f"w2 = {status_info.get('current_parameters', {}).get('w2', 0):.2f}"
        ]
        
        for line in status_lines:
            if line:  # Skip empty lines
                cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            
    def draw_control_hints(self, frame: np.ndarray) -> None:
        """Draw control hints on frame."""
        height, width = frame.shape[:2]
        
        hints = [
            "Controls: Q-quit, C-camera, S-status, R-record, X-export",
            "V-viz mode, L-LOD, M-monitor, P-pause, +/-smoothing, SPACE-reset"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        
        y_start = height - 40
        for i, hint in enumerate(hints):
            y_pos = y_start + i * 20
            cv2.putText(frame, hint, (10, y_pos), font, font_scale, color, thickness)
            
    def start_recording(self) -> None:
        """Start high-performance trajectory recording."""
        session_id = f"gesture_parametric_{int(time.time())}"
        success = self.trajectory_recorder.start_recording(session_id)
        
        if success:
            self.recording = True
            self.record_start_time = time.time()
            self.trajectory_points = []
            print(f"High-performance recording started: {session_id}")
        else:
            print("Failed to start recording")
        
    def stop_recording(self) -> None:
        """Stop high-performance recording and save data."""
        if not self.recording:
            return
            
        self.recording = False
        
        # Stop trajectory recorder
        metadata = self.trajectory_recorder.stop_recording()
        
        if metadata:
            # Export trajectory data
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            binary_file = f"trajectory_{timestamp}.trj"
            csv_file = f"trajectory_{timestamp}.csv"
            
            # Export in multiple formats
            self.trajectory_recorder.export_trajectory(binary_file, 'binary', True)
            self.trajectory_recorder.export_trajectory(csv_file, 'csv', False)
            
            # Get performance stats
            perf_stats = self.trajectory_recorder.get_performance_stats()
            
            print(f"Recording completed:")
            print(f"  Duration: {metadata.duration_seconds:.1f}s")
            print(f"  Points: {metadata.total_points}")
            print(f"  Average FPS: {metadata.average_fps:.1f}")
            print(f"  Recording FPS: {perf_stats['recording_fps']:.1f}")
            print(f"  Memory usage: {perf_stats['memory_usage_mb']:.1f} MB")
            print(f"  Files saved: {binary_file}, {csv_file}")
        else:
            print("No recording data to save")
            
    def start(self) -> None:
        """Start the live system."""
        print("Starting Live Gesture-Parametric System...")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return False
            
        # Display twelve-tone mapping information
        print("\nTwelve-Tone Gesture Mapping:")
        scale_info = self.bridge.get_twelve_tone_info()
        print("Fingers | Radius | Frequency | Ratio | Semitones")
        print("-" * 50)
        for finger_count, info in scale_info.items():
            print(f"   {finger_count}    | {info['radius']:.3f} |  {info['frequency']:.3f}   | {info['radius_ratio']:.3f} |    {info['semitone_offset']:+d}")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start threads
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_thread_function, daemon=True)
        self.camera_thread.start()
        
        # Start matplotlib animation
        print(f"\nStarting visualization @ {self.visualization_fps}fps")
        print("Use keyboard controls in the camera window")
        print("Close the matplotlib window or press 'Q' to exit")
        
        try:
            # Setup animation
            anim_interval = 1000 // self.visualization_fps  # Convert to milliseconds
            animation = self.renderer.start_animation(interval=anim_interval)
            
            # Override the animation update function
            animation.func = self.update_visualization
            
            # Show plot and handle events
            plt.ion()  # Interactive mode
            self.renderer.show()
            
            # Main loop for camera display and input handling
            while self.running:
                self.update_camera_display()
                self.handle_keyboard_input()
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
                
                # Check if matplotlib window was closed
                if not plt.get_fignums():
                    print("Visualization window closed")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            self.stop()
            
        return True
        
    def stop(self) -> None:
        """Stop the live system."""
        print("Stopping system...")
        
        self.running = False
        
        # Stop recording if active
        if self.recording:
            self.stop_recording()
            
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
            
        # Wait for camera thread
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
            
        # Release camera
        if self.camera:
            self.camera.release()
            
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Close matplotlib
        plt.close('all')
        
        print("System stopped")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Live Gesture-Parametric System')
    
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--rmax', type=float, default=2.5, help='Maximum radius (default: 2.5)')
    parser.add_argument('--smooth', type=float, default=0.85, help='Smoothing factor (default: 0.85)')
    parser.add_argument('--viz-fps', type=int, default=60, help='Visualization FPS (default: 60)')
    parser.add_argument('--cam-fps', type=int, default=30, help='Camera FPS (default: 30)')
    parser.add_argument('--assignment', choices=['left_r1', 'right_r1', 'dominant'], 
                       default='left_r1', help='Hand assignment (default: left_r1)')
    
    args = parser.parse_args()
    
    # Map assignment argument
    assignment_map = {
        'left_r1': HandAssignment.LEFT_R1_RIGHT_R2,
        'right_r1': HandAssignment.RIGHT_R1_LEFT_R2,
        'dominant': HandAssignment.DOMINANT_PRIMARY
    }
    
    # Create and start system
    system = LiveGestureParametricSystem(
        camera_id=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        r_max=args.rmax,
        smoothing_factor=args.smooth,
        visualization_fps=args.viz_fps,
        camera_fps=args.cam_fps,
        hand_assignment=assignment_map[args.assignment]
    )
    
    success = system.start()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())