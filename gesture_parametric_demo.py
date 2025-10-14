#!/usr/bin/env python3
"""
Gesture-Parametric Demo
Demonstrates the complete gesture-controlled parametric equation system.

This demo shows:
1. How to use the gesture detection with 1-5 fingers
2. Twelve-tone parameter mapping
3. Real-time parametric visualization
4. Interactive controls

Usage:
    python gesture_parametric_demo.py

Controls:
- Show your hand(s) to the camera
- Hold up 1-5 fingers to control radius parameters
- Left hand controls r1, right hand controls r2
- Move hands to control phase parameters
- Press 'q' to quit
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time
from typing import Optional

from gesture_parametric_bridge import GestureParametricBridge, HandAssignment
from parametric_equation_renderer import ParametricEquationRenderer


class GestureParametricDemo:
    """Demo application for gesture-controlled parametric equations."""
    
    def __init__(self, camera_id: int = 0):
        """Initialize the demo."""
        self.camera_id = camera_id
        
        # Initialize components
        self.bridge = GestureParametricBridge(
            r_max=2.5,
            smoothing_factor=0.8,
            hand_assignment=HandAssignment.LEFT_R1_RIGHT_R2,
            auto_pause=False  # Keep animation running for demo
        )
        
        # Setup renderer for real-time mode
        self.renderer = ParametricEquationRenderer(
            r1=1.0, r2=0.5,
            w1=1.0, w2=2.0,
            p1=0.0, p2=0.0,
            max_theta=6*np.pi,  # Shorter for faster cycles
            num_points=800,     # Fewer points for better performance
            trail_length=200    # Shorter trail
        )
        
        self.renderer.set_real_time_mode(True)
        self.bridge.set_renderer(self.renderer)
        
        # Demo state
        self.camera = None
        self.running = False
        self.show_instructions = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
                
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def draw_instructions(self, frame: np.ndarray) -> None:
        """Draw instruction overlay on camera frame."""
        if not self.show_instructions:
            return
            
        instructions = [
            "Gesture-Parametric Demo",
            "",
            "Show 1-5 fingers to control visualization:",
            "• Left hand controls r1 (primary radius)", 
            "• Right hand controls r2 (secondary radius)",
            "• Hand position controls phase",
            "",
            "Gesture mapping (twelve-tone scale):",
            "1 finger → small radius",
            "2 fingers → medium-small radius", 
            "3 fingers → medium radius",
            "4 fingers → medium-large radius",
            "5 fingers → large radius",
            "",
            "Press 'q' to quit, 'i' to toggle instructions"
        ]
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 255)  # Yellow
        thickness = 1
        y_offset = 35
        
        for line in instructions:
            if line.startswith("•"):
                # Bullet points in green
                cv2.putText(frame, line, (30, y_offset), font, font_scale, (0, 255, 0), thickness)
            elif line.startswith("Gesture"):
                # Headers in white
                cv2.putText(frame, line, (20, y_offset), font, font_scale + 0.1, (255, 255, 255), thickness + 1)
            elif line and not line.startswith(" "):
                # Main text in yellow
                cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness)
            else:
                # Sub-items in cyan
                cv2.putText(frame, line, (30, y_offset), font, font_scale, (255, 255, 0), thickness)
            
            y_offset += 20
    
    def draw_status_info(self, frame: np.ndarray) -> None:
        """Draw status information on frame."""
        height, width = frame.shape[:2]
        gesture_info = self.bridge.get_gesture_info()
        
        # Status box in top-right
        status_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Hands: {gesture_info.get('hands_detected', 0)}",
            f"Gestures: {gesture_info.get('digit_gestures', [])}",
            "",
            "Current Parameters:",
            f"r1 = {gesture_info.get('current_parameters', {}).get('r1', 0):.2f}",
            f"r2 = {gesture_info.get('current_parameters', {}).get('r2', 0):.2f}",
        ]
        
        # Background
        box_width = 200
        box_height = len(status_lines) * 20 + 20
        cv2.rectangle(frame, (width - box_width - 10, 10), (width - 10, box_height), (0, 0, 0), -1)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 0)
        thickness = 1
        
        y_offset = 30
        for line in status_lines:
            if line:
                cv2.putText(frame, line, (width - box_width + 5, y_offset), 
                           font, font_scale, color, thickness)
            y_offset += 20
    
    def update_fps_counter(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def process_camera_frame(self) -> Optional[np.ndarray]:
        """Process one camera frame."""
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process gesture detection and update parameters
        processed_frame, parameters = self.bridge.process_frame(frame)
        
        # Draw overlays
        self.draw_instructions(processed_frame)
        self.draw_status_info(processed_frame)
        
        # Update gesture indicators in renderer
        gesture_info = self.bridge.get_gesture_info()
        hands_data = gesture_info.get('digit_gestures', [])
        
        left_gesture = None
        right_gesture = None
        
        # Extract left/right hand gestures from detected hands
        # This is a simplified approach - in practice you'd need to track hand positions
        if len(hands_data) >= 1:
            left_gesture = hands_data[0]
        if len(hands_data) >= 2:
            right_gesture = hands_data[1]
            
        self.renderer.set_gesture_indicators(
            left_hand_gesture=left_gesture,
            right_hand_gesture=right_gesture,
            hands_detected=gesture_info.get('hands_detected', 0),
            is_paused=gesture_info.get('is_paused', False)
        )
        
        self.update_fps_counter()
        return processed_frame
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input. Returns False if should quit."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('i'):
            self.show_instructions = not self.show_instructions
            print(f"Instructions: {'ON' if self.show_instructions else 'OFF'}")
        elif key == ord('r'):
            # Reset parameters
            self.bridge.reset_parameters()
            self.renderer.reset_animation()
            print("Parameters reset")
        elif key == ord('s'):
            # Change smoothing
            current = self.bridge.smoothing_factor
            new_smooth = 0.2 if current > 0.5 else 0.9
            self.bridge.set_smoothing_factor(new_smooth)
            print(f"Smoothing: {new_smooth:.1f}")
        elif key == ord('1'):
            self.bridge.set_hand_assignment(HandAssignment.LEFT_R1_RIGHT_R2)
            print("Assignment: Left→r1, Right→r2")
        elif key == ord('2'):
            self.bridge.set_hand_assignment(HandAssignment.RIGHT_R1_LEFT_R2)
            print("Assignment: Right→r1, Left→r2")
        elif key == ord('3'):
            self.bridge.set_hand_assignment(HandAssignment.DOMINANT_PRIMARY)
            print("Assignment: Dominant primary")
            
        return True
    
    def animation_update(self, frame_num: int):
        """Animation update function for matplotlib."""
        # Get latest parameters from camera processing
        # The bridge automatically updates the renderer parameters
        return self.renderer.animate_frame(frame_num)
    
    def run(self) -> bool:
        """Run the demo."""
        print("Starting Gesture-Parametric Demo...")
        
        # Initialize camera
        if not self.initialize_camera():
            return False
            
        # Show twelve-tone mapping info
        print("\nTwelve-Tone Gesture Mapping:")
        scale_info = self.bridge.get_twelve_tone_info()
        for finger_count, info in scale_info.items():
            print(f"  {finger_count} fingers: radius = {info['radius']:.3f}")
            
        # Setup visualization
        print("Setting up visualization...")
        plt.ion()  # Interactive mode
        
        # Override renderer's animation function
        anim = FuncAnimation(
            self.renderer.fig, 
            self.animation_update,
            interval=33,  # ~30 FPS
            blit=True,
            cache_frame_data=False
        )
        
        # Show the plot
        self.renderer.show()
        
        print("\nDemo started!")
        print("Controls:")
        print("  Q - Quit")
        print("  I - Toggle instructions")
        print("  R - Reset parameters")
        print("  S - Toggle smoothing")
        print("  1/2/3 - Change hand assignment")
        print("\nShow your hands to the camera with 1-5 fingers!")
        
        self.running = True
        
        try:
            # Main loop
            while self.running:
                # Process camera frame
                frame = self.process_camera_frame()
                if frame is not None:
                    cv2.imshow('Gesture-Parametric Demo', frame)
                
                # Handle keyboard input
                if not self.handle_keyboard_input():
                    break
                    
                # Check if matplotlib window is still open
                if not plt.get_fignums():
                    print("Visualization window closed")
                    break
                    
                # Update matplotlib
                plt.pause(0.001)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
            
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up...")
        
        self.running = False
        
        if self.camera:
            self.camera.release()
            
        cv2.destroyAllWindows()
        plt.close('all')
        
        print("Demo finished")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gesture-Parametric Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = GestureParametricDemo(camera_id=args.camera)
    success = demo.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())