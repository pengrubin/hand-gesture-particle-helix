#!/usr/bin/env python3
"""
Gesture-Parametric Bridge Module
Connects MediaPipe hand gesture recognition with parametric equation visualization.

This module acts as a bridge between the hand gesture detector and the parametric
equation renderer, using the twelve-tone mapping system for smooth parameter control.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
import time

from hand_gesture_detector import HandGestureDetector
from gesture_radius_mapper import GestureToRadiusMapper, HandState
from parametric_equation_renderer import ParametricEquationRenderer


class HandAssignment(Enum):
    """Hand assignment for parametric control."""
    LEFT_R1_RIGHT_R2 = "left_r1_right_r2"
    RIGHT_R1_LEFT_R2 = "right_r1_left_r2" 
    DOMINANT_PRIMARY = "dominant_primary"


class GestureParametricBridge:
    """
    Bridge between MediaPipe gesture detection and parametric equation visualization.
    
    Features:
    - Real-time gesture to parameter mapping using twelve-tone scale
    - Smooth parameter transitions
    - Hand assignment flexibility
    - Gesture state management with pause/resume functionality
    """
    
    def __init__(self, 
                 r_max: float = 2.0,
                 smoothing_factor: float = 0.85,
                 hand_assignment: HandAssignment = HandAssignment.LEFT_R1_RIGHT_R2,
                 auto_pause: bool = True):
        """
        Initialize the gesture-parametric bridge.
        
        Args:
            r_max: Maximum radius value for twelve-tone mapping
            smoothing_factor: Parameter smoothing factor (0-1)
            hand_assignment: How to assign hands to r1/r2 parameters
            auto_pause: Whether to pause animation when no hands detected
        """
        # Core components
        self.gesture_detector = HandGestureDetector()
        self.radius_mapper = GestureToRadiusMapper(r_max=r_max)
        self.renderer = None  # Will be set externally or in live system
        
        # Configuration
        self.hand_assignment = hand_assignment
        self.auto_pause = auto_pause
        self.smoothing_factor = smoothing_factor
        self.radius_mapper.set_smoothing_factor(smoothing_factor)
        
        # State tracking
        self.is_paused = False
        self.last_hands_count = 0
        self.gesture_history = []
        self.max_history_length = 10
        
        # Parameter state
        self.current_parameters = {
            'r1': 4.0, 'r2': 4.0,  # P.Georges: Equal large radii for proper flower pattern
            'w1': 1.0, 'w2': -1.96,  # Match P.Georges pattern: w2 = -2 + 1/25 for 25-fold symmetry
            'p1': 0.0, 'p2': 0.0
        }
        self.target_parameters = self.current_parameters.copy()
        
        # Hand state mapping
        self.gesture_to_hand_state = {
            0: HandState.FIST,
            1: HandState.ONE_FINGER,
            2: HandState.TWO_FINGERS, 
            3: HandState.THREE_FINGERS,
            4: HandState.FOUR_FINGERS,
            5: HandState.OPEN_HAND,
            'none': HandState.NO_HAND
        }
        
        # Timing
        self.last_update_time = time.time()
        self.fps_target = 30
        self.min_frame_time = 1.0 / self.fps_target
        
    def set_renderer(self, renderer: ParametricEquationRenderer) -> None:
        """Set the parametric equation renderer."""
        self.renderer = renderer
        
    def normalize_hand_position(self, hand_center: List[float]) -> Tuple[float, float]:
        """
        Convert MediaPipe hand center to normalized coordinates.
        
        Args:
            hand_center: [x, y] coordinates from MediaPipe (0-1 range)
            
        Returns:
            Tuple of (x, y) in normalized space (-1 to 1)
        """
        # MediaPipe gives coordinates in 0-1 range, convert to -1 to 1
        x = (hand_center[0] - 0.5) * 2.0
        y = (hand_center[1] - 0.5) * 2.0
        
        # Flip Y coordinate to match visualization coordinate system
        y = -y
        
        return (x, y)
        
    def extract_hand_states(self, gesture_data: Dict[str, Any]) -> Tuple[HandState, HandState, Tuple[float, float], Tuple[float, float]]:
        """
        Extract left and right hand states from gesture data.
        
        Args:
            gesture_data: Gesture data from HandGestureDetector
            
        Returns:
            Tuple of (left_state, right_state, left_position, right_position)
        """
        left_state = HandState.NO_HAND
        right_state = HandState.NO_HAND
        left_position = (0.0, 0.0)
        right_position = (0.0, 0.0)
        
        if not gesture_data or 'hands' not in gesture_data:
            return left_state, right_state, left_position, right_position
            
        for hand_data in gesture_data['hands']:
            gesture_number = hand_data.get('gesture_number', 'none')
            hand_label = hand_data.get('label', 'unknown').lower()
            hand_center = hand_data.get('center', [0.5, 0.5])
            
            # Convert gesture to hand state
            hand_state = self.gesture_to_hand_state.get(gesture_number, HandState.NO_HAND)
            position = self.normalize_hand_position(hand_center)
            
            # Assign to left or right based on label
            if hand_label == 'left':
                left_state = hand_state
                left_position = position
            elif hand_label == 'right':
                right_state = hand_state
                right_position = position
                
        return left_state, right_state, left_position, right_position
        
    def apply_hand_assignment(self, left_state: HandState, right_state: HandState,
                             left_pos: Tuple[float, float], right_pos: Tuple[float, float]) -> None:
        """
        Apply hand assignment strategy to update parameter mapper.
        
        Args:
            left_state: Left hand state
            right_state: Right hand state  
            left_pos: Left hand position
            right_pos: Right hand position
        """
        if self.hand_assignment == HandAssignment.LEFT_R1_RIGHT_R2:
            # Standard assignment: left hand -> r1, right hand -> r2
            self.radius_mapper.update_hand_states(
                left_hand_gesture=left_state.value,
                right_hand_gesture=right_state.value,
                left_hand_position=left_pos,
                right_hand_position=right_pos
            )
            
        elif self.hand_assignment == HandAssignment.RIGHT_R1_LEFT_R2:
            # Swapped assignment: right hand -> r1, left hand -> r2
            self.radius_mapper.update_hand_states(
                left_hand_gesture=right_state.value,
                right_hand_gesture=left_state.value,
                left_hand_position=right_pos,
                right_hand_position=left_pos
            )
            
        elif self.hand_assignment == HandAssignment.DOMINANT_PRIMARY:
            # Use most active hand as primary (r1)
            if left_state != HandState.NO_HAND and right_state == HandState.NO_HAND:
                self.radius_mapper.update_hand_states(
                    left_hand_gesture=left_state.value,
                    right_hand_gesture=HandState.NO_HAND.value,
                    left_hand_position=left_pos,
                    right_hand_position=(0.0, 0.0)
                )
            elif right_state != HandState.NO_HAND and left_state == HandState.NO_HAND:
                self.radius_mapper.update_hand_states(
                    left_hand_gesture=right_state.value,
                    right_hand_gesture=HandState.NO_HAND.value,
                    left_hand_position=right_pos,
                    right_hand_position=(0.0, 0.0)
                )
            else:
                # Both hands active - use standard assignment
                self.radius_mapper.update_hand_states(
                    left_hand_gesture=left_state.value,
                    right_hand_gesture=right_state.value,
                    left_hand_position=left_pos,
                    right_hand_position=right_pos
                )
                
    def update_pause_state(self, hands_detected: int) -> None:
        """
        Update pause state based on hand detection.
        
        Args:
            hands_detected: Number of hands currently detected
        """
        if not self.auto_pause:
            return
            
        # Pause when no hands detected for stability
        if hands_detected == 0 and not self.is_paused:
            self.is_paused = True
            if self.renderer:
                # Could add pause indication to renderer here
                pass
                
        # Resume when hands detected
        elif hands_detected > 0 and self.is_paused:
            self.is_paused = False
            if self.renderer:
                # Could resume animation here
                pass
                
        self.last_hands_count = hands_detected
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a camera frame and update parametric equation parameters.
        
        Args:
            frame: Camera frame from OpenCV
            
        Returns:
            Tuple of (processed_frame, current_parameters)
        """
        current_time = time.time()
        
        # Throttle processing to target FPS
        time_since_last = current_time - self.last_update_time
        if time_since_last < self.min_frame_time:
            return frame, self.current_parameters
            
        # Process gesture detection
        processed_frame = self.gesture_detector.process_frame(frame)
        gesture_data = self.gesture_detector.gesture_data
        
        # Extract hand states
        left_state, right_state, left_pos, right_pos = self.extract_hand_states(gesture_data)
        
        # Update gesture history
        self.gesture_history.append({
            'timestamp': current_time,
            'left_state': left_state,
            'right_state': right_state,
            'hands_detected': gesture_data.get('hands_detected', 0)
        })
        
        if len(self.gesture_history) > self.max_history_length:
            self.gesture_history = self.gesture_history[-self.max_history_length:]
        
        # Apply hand assignment and update parameters
        self.apply_hand_assignment(left_state, right_state, left_pos, right_pos)
        
        # Update pause state
        hands_detected = gesture_data.get('hands_detected', 0)
        self.update_pause_state(hands_detected)
        
        # Get smoothed parameters from mapper
        if not self.is_paused:
            self.current_parameters = self.radius_mapper.get_parameters()
            
            # Update renderer if available
            if self.renderer:
                self.renderer.update_parameters(
                    r1=self.current_parameters['r1'],
                    r2=self.current_parameters['r2'],
                    w1=self.current_parameters['w1'], 
                    w2=self.current_parameters['w2'],
                    p1=self.current_parameters['p1'],
                    p2=self.current_parameters['p2']
                )
        
        self.last_update_time = current_time
        return processed_frame, self.current_parameters
        
    def get_gesture_info(self) -> Dict[str, Any]:
        """
        Get current gesture detection information.
        
        Returns:
            Dictionary containing gesture detection details
        """
        gesture_data = self.gesture_detector.gesture_data
        
        return {
            'hands_detected': gesture_data.get('hands_detected', 0),
            'digit_gestures': gesture_data.get('digit_gestures', []),
            'is_paused': self.is_paused,
            'current_parameters': self.current_parameters.copy(),
            'smoothing_factor': self.smoothing_factor,
            'hand_assignment': self.hand_assignment.value,
            'gesture_history_length': len(self.gesture_history)
        }
        
    def get_twelve_tone_info(self) -> Dict[int, Dict[str, float]]:
        """Get twelve-tone scale mapping information."""
        return self.radius_mapper.get_twelve_tone_scale_info()
        
    def set_hand_assignment(self, assignment: HandAssignment) -> None:
        """Change hand assignment strategy."""
        self.hand_assignment = assignment
        
    def set_auto_pause(self, enabled: bool) -> None:
        """Enable or disable auto-pause functionality."""
        self.auto_pause = enabled
        if not enabled:
            self.is_paused = False
            
    def set_smoothing_factor(self, factor: float) -> None:
        """Update parameter smoothing factor."""
        self.smoothing_factor = max(0.0, min(1.0, factor))
        self.radius_mapper.set_smoothing_factor(self.smoothing_factor)
        
    def reset_parameters(self) -> None:
        """Reset all parameters to default values."""
        self.radius_mapper.reset_parameters()
        self.current_parameters = {
            'r1': 4.0, 'r2': 4.0,  # P.Georges: Equal large radii for proper flower pattern
            'w1': 1.0, 'w2': -1.96,  # Match P.Georges pattern: w2 = -2 + 1/25 for 25-fold symmetry
            'p1': 0.0, 'p2': 0.0
        }
        self.is_paused = False
        self.gesture_history = []
        
    def get_status_display_text(self) -> str:
        """Generate status text for display."""
        gesture_data = self.gesture_detector.gesture_data
        hands_detected = gesture_data.get('hands_detected', 0)
        digit_gestures = gesture_data.get('digit_gestures', [])
        
        status_lines = [
            f"Hands detected: {hands_detected}",
            f"Gestures: {digit_gestures}",
            f"Paused: {self.is_paused}",
            f"Assignment: {self.hand_assignment.value}",
            "",
            "Parameters:",
            f"r1 = {self.current_parameters['r1']:.3f}",
            f"r2 = {self.current_parameters['r2']:.3f}",
            f"w1 = {self.current_parameters['w1']:.3f}",
            f"w2 = {self.current_parameters['w2']:.3f}"
        ]
        
        return "\n".join(status_lines)


if __name__ == "__main__":
    # Test the bridge with webcam
    import sys
    
    print("Testing Gesture-Parametric Bridge")
    print("Press 'q' to quit, 's' to toggle smoothing")
    
    # Initialize bridge
    bridge = GestureParametricBridge(
        r_max=2.0,
        smoothing_factor=0.8,
        hand_assignment=HandAssignment.LEFT_R1_RIGHT_R2
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    # Display twelve-tone mapping
    print("\nTwelve-Tone Gesture Mapping:")
    scale_info = bridge.get_twelve_tone_info()
    for finger_count, info in scale_info.items():
        print(f"Fingers {finger_count}: r={info['radius']:.3f}, f={info['frequency']:.3f}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, parameters = bridge.process_frame(frame)
            
            # Display status on frame
            status_text = bridge.get_status_display_text()
            y_offset = 30
            for line in status_text.split('\n'):
                cv2.putText(processed_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow('Gesture-Parametric Bridge', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Toggle smoothing
                current_smooth = bridge.smoothing_factor
                new_smooth = 0.2 if current_smooth > 0.5 else 0.9
                bridge.set_smoothing_factor(new_smooth)
                print(f"Smoothing factor changed to {new_smooth}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Bridge test completed")