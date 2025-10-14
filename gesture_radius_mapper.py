#!/usr/bin/env python3
"""
Gesture to Radius Mapper using Twelve-Tone Scale
Maps finger counts to radius multipliers using twelve-tone equal temperament.

Formula: r = r_max * 2^((finger_count - 5) * 2/12)

This module provides gesture-based parameter control for parametric equation visualization.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum


class HandState(Enum):
    """Enumeration of possible hand states."""
    NO_HAND = "no_hand"
    FIST = "fist" 
    ONE_FINGER = "one_finger"
    TWO_FINGERS = "two_fingers"
    THREE_FINGERS = "three_fingers"
    FOUR_FINGERS = "four_fingers"
    OPEN_HAND = "open_hand"


class GestureToRadiusMapper:
    """
    Maps hand gestures to radius parameters using twelve-tone scale mathematics.
    
    The mapping uses exponential scaling based on twelve-tone equal temperament:
    r = r_max * 2^((finger_count - 5) * 2/12)
    
    This creates musically harmonious parameter relationships.
    """
    
    def __init__(self, r_max: float = 2.0, base_frequency: float = 1.0):
        """
        Initialize the gesture mapper.
        
        Args:
            r_max: Maximum radius value (at 5 fingers)
            base_frequency: Base frequency for parameter scaling
        """
        self.r_max = r_max
        self.base_frequency = base_frequency
        
        # Twelve-tone scale parameters
        self.semitone_ratio = 2**(1/12)  # Twelfth root of 2
        self.reference_finger_count = 5  # Open hand as reference
        
        # Gesture state tracking
        self.left_hand_state = HandState.NO_HAND
        self.right_hand_state = HandState.NO_HAND
        self.left_hand_position = (0.0, 0.0)  # Normalized coordinates (-1 to 1)
        self.right_hand_position = (0.0, 0.0)
        
        # Parameter smoothing
        self.smoothing_factor = 0.85
        self.current_r1 = 4.0      # P.Georges: Equal large radii
        self.current_r2 = 4.0      # P.Georges: Equal large radii  
        self.current_w1 = 1.0      # P.Georges: Inner rod velocity
        self.current_w2 = -1.96    # P.Georges: w2 = -2 + 1/25
        self.current_p1 = 0.0
        self.current_p2 = 0.0
        
        # Mapping configuration
        self.setup_gesture_mappings()
    
    def setup_gesture_mappings(self) -> None:
        """Configure gesture to parameter mappings."""
        self.hand_state_to_finger_count = {
            HandState.NO_HAND: 0,
            HandState.FIST: 0,
            HandState.ONE_FINGER: 1,
            HandState.TWO_FINGERS: 2,
            HandState.THREE_FINGERS: 3,
            HandState.FOUR_FINGERS: 4,
            HandState.OPEN_HAND: 5
        }
    
    def finger_count_to_radius(self, finger_count: int) -> float:
        """
        Convert finger count to radius using linear scaling.
        
        Formula: r = r_min + (finger_count / 5) * (r_max - r_min)
        
        Args:
            finger_count: Number of extended fingers (0-5)
            
        Returns:
            Radius value
        """
        if finger_count < 0:
            finger_count = 0
        elif finger_count > 5:
            finger_count = 5
        
        # Linear scaling from r_min to r_max
        r_min = self.r_max * 0.2  # Minimum radius is 20% of max
        radius = r_min + (finger_count / 5.0) * (self.r_max - r_min)
        
        return max(0.01, radius)  # Prevent zero or negative radius
    
    def finger_count_to_frequency(self, finger_count: int) -> float:
        """
        Convert finger count to frequency parameter using linear scaling.
        
        Args:
            finger_count: Number of extended fingers (0-5)
            
        Returns:
            Frequency multiplier
        """
        if finger_count < 0:
            finger_count = 0
        elif finger_count > 5:
            finger_count = 5
        
        # Linear frequency mapping from 0.3 to 2.5 for smooth progression
        f_min = 0.3
        f_max = 2.5
        frequency = f_min + (finger_count / 5.0) * (f_max - f_min)
        
        return max(0.1, min(10.0, frequency))  # Clamp to reasonable range
    
    def position_to_phase(self, position: Tuple[float, float]) -> float:
        """
        Convert hand position to phase parameter.
        
        Args:
            position: (x, y) coordinates in normalized space (-1 to 1)
            
        Returns:
            Phase value in radians
        """
        x, y = position
        # Convert position to angle
        angle = np.arctan2(y, x)
        return angle
    
    def update_hand_states(self, left_hand_gesture: Optional[str] = None,
                          right_hand_gesture: Optional[str] = None,
                          left_hand_position: Optional[Tuple[float, float]] = None,
                          right_hand_position: Optional[Tuple[float, float]] = None) -> None:
        """
        Update hand gesture states and positions.
        
        Args:
            left_hand_gesture: Left hand gesture string
            right_hand_gesture: Right hand gesture string  
            left_hand_position: (x, y) position of left hand
            right_hand_position: (x, y) position of right hand
        """
        # Update left hand
        if left_hand_gesture is not None:
            try:
                self.left_hand_state = HandState(left_hand_gesture.lower())
            except ValueError:
                self.left_hand_state = HandState.NO_HAND
        
        if left_hand_position is not None:
            self.left_hand_position = left_hand_position
        
        # Update right hand
        if right_hand_gesture is not None:
            try:
                self.right_hand_state = HandState(right_hand_gesture.lower())
            except ValueError:
                self.right_hand_state = HandState.NO_HAND
        
        if right_hand_position is not None:
            self.right_hand_position = right_hand_position
    
    def compute_target_parameters(self) -> Dict[str, float]:
        """
        Compute target parameters based on current hand states.
        
        Returns:
            Dictionary of target parameter values
        """
        # Get finger counts
        left_fingers = self.hand_state_to_finger_count[self.left_hand_state]
        right_fingers = self.hand_state_to_finger_count[self.right_hand_state]
        
        # Determine control scheme based on active hands
        if self.left_hand_state == HandState.NO_HAND and self.right_hand_state == HandState.NO_HAND:
            # No hands - use P.Georges default values
            target_r1 = 4.0      # P.Georges: Equal large radii
            target_r2 = 4.0      # P.Georges: Equal large radii
            target_w1 = 1.0      # P.Georges: Inner rod velocity
            target_w2 = -1.96    # P.Georges: w2 = -2 + 1/25 for flower pattern
            target_p1 = 0.0
            target_p2 = 0.0
            
        elif self.left_hand_state != HandState.NO_HAND and self.right_hand_state == HandState.NO_HAND:
            # Single hand control - left hand controls both radii
            target_r1 = self.finger_count_to_radius(left_fingers)
            target_r2 = self.finger_count_to_radius(max(1, left_fingers - 2))  # Secondary radius
            target_w1 = self.finger_count_to_frequency(left_fingers)
            target_w2 = self.finger_count_to_frequency(left_fingers) * 1.5
            target_p1 = self.position_to_phase(self.left_hand_position)
            target_p2 = self.position_to_phase(self.left_hand_position) + np.pi/4
            
        elif self.left_hand_state == HandState.NO_HAND and self.right_hand_state != HandState.NO_HAND:
            # Single hand control - right hand controls both radii
            target_r1 = self.finger_count_to_radius(right_fingers)
            target_r2 = self.finger_count_to_radius(max(1, right_fingers - 2))
            target_w1 = self.finger_count_to_frequency(right_fingers)
            target_w2 = self.finger_count_to_frequency(right_fingers) * 1.5
            target_p1 = self.position_to_phase(self.right_hand_position)
            target_p2 = self.position_to_phase(self.right_hand_position) + np.pi/4
            
        else:
            # Dual hand control
            target_r1 = self.finger_count_to_radius(left_fingers)
            target_r2 = self.finger_count_to_radius(right_fingers)
            target_w1 = self.finger_count_to_frequency(left_fingers)
            target_w2 = self.finger_count_to_frequency(right_fingers)
            target_p1 = self.position_to_phase(self.left_hand_position)
            target_p2 = self.position_to_phase(self.right_hand_position)
        
        return {
            'r1': target_r1,
            'r2': target_r2, 
            'w1': target_w1,
            'w2': target_w2,
            'p1': target_p1,
            'p2': target_p2
        }
    
    def apply_smoothing(self, target_params: Dict[str, float]) -> Dict[str, float]:
        """
        Apply temporal smoothing to parameter changes.
        
        Args:
            target_params: Target parameter values
            
        Returns:
            Smoothed parameter values
        """
        # Smooth each parameter
        self.current_r1 = (self.smoothing_factor * self.current_r1 + 
                          (1 - self.smoothing_factor) * target_params['r1'])
        
        self.current_r2 = (self.smoothing_factor * self.current_r2 + 
                          (1 - self.smoothing_factor) * target_params['r2'])
        
        self.current_w1 = (self.smoothing_factor * self.current_w1 + 
                          (1 - self.smoothing_factor) * target_params['w1'])
        
        self.current_w2 = (self.smoothing_factor * self.current_w2 + 
                          (1 - self.smoothing_factor) * target_params['w2'])
        
        # Phase parameters need circular smoothing
        self.current_p1 = self.smooth_angle(self.current_p1, target_params['p1'])
        self.current_p2 = self.smooth_angle(self.current_p2, target_params['p2'])
        
        return {
            'r1': self.current_r1,
            'r2': self.current_r2,
            'w1': self.current_w1,
            'w2': self.current_w2,
            'p1': self.current_p1,
            'p2': self.current_p2
        }
    
    def smooth_angle(self, current_angle: float, target_angle: float) -> float:
        """
        Apply circular smoothing to angle parameters.
        
        Args:
            current_angle: Current angle in radians
            target_angle: Target angle in radians
            
        Returns:
            Smoothed angle
        """
        # Handle angle wrapping
        diff = target_angle - current_angle
        
        # Wrap difference to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        # Apply smoothing
        smoothed_diff = (1 - self.smoothing_factor) * diff
        new_angle = current_angle + smoothed_diff
        
        # Wrap result to [-π, π]
        while new_angle > np.pi:
            new_angle -= 2 * np.pi
        while new_angle < -np.pi:
            new_angle += 2 * np.pi
        
        return new_angle
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current smoothed parameters.
        
        Returns:
            Dictionary of current parameter values
        """
        target_params = self.compute_target_parameters()
        smoothed_params = self.apply_smoothing(target_params)
        return smoothed_params
    
    def set_smoothing_factor(self, factor: float) -> None:
        """
        Set the smoothing factor (0 = no smoothing, 1 = no change).
        
        Args:
            factor: Smoothing factor between 0 and 1
        """
        self.smoothing_factor = max(0.0, min(1.0, factor))
    
    def reset_parameters(self) -> None:
        """Reset all parameters to P.Georges default values."""
        self.current_r1 = 4.0      # P.Georges: Equal large radii
        self.current_r2 = 4.0      # P.Georges: Equal large radii
        self.current_w1 = 1.0      # P.Georges: Inner rod velocity
        self.current_w2 = -1.96    # P.Georges: w2 = -2 + 1/25
        self.current_p1 = 0.0
        self.current_p2 = 0.0
        self.left_hand_state = HandState.NO_HAND
        self.right_hand_state = HandState.NO_HAND
        self.left_hand_position = (0.0, 0.0)
        self.right_hand_position = (0.0, 0.0)
    
    def get_linear_scale_info(self) -> Dict[int, Dict[str, float]]:
        """
        Get linear scale mapping information for all finger counts.
        
        Returns:
            Dictionary mapping finger counts to their parameter values
        """
        scale_info = {}
        for finger_count in range(6):  # 0 to 5 fingers
            scale_info[finger_count] = {
                'radius': self.finger_count_to_radius(finger_count),
                'frequency': self.finger_count_to_frequency(finger_count),
                'radius_ratio': self.finger_count_to_radius(finger_count) / self.r_max,
                'linear_scale': finger_count / 5.0
            }
        return scale_info
    
    def simulate_gesture_sequence(self, gesture_sequence: List[Tuple[str, str]], 
                                frames_per_gesture: int = 30) -> List[Dict[str, float]]:
        """
        Simulate a sequence of gestures for testing.
        
        Args:
            gesture_sequence: List of (left_gesture, right_gesture) tuples
            frames_per_gesture: Number of frames to hold each gesture
            
        Returns:
            List of parameter dictionaries for each frame
        """
        parameter_sequence = []
        
        for left_gesture, right_gesture in gesture_sequence:
            self.update_hand_states(left_gesture, right_gesture)
            
            for _ in range(frames_per_gesture):
                params = self.get_parameters()
                parameter_sequence.append(params.copy())
        
        return parameter_sequence


if __name__ == "__main__":
    # Test the gesture mapper
    mapper = GestureToRadiusMapper(r_max=2.0)
    
    print("Twelve-Tone Gesture Mapping Test")
    print("=" * 50)
    
    # Display scale information
    scale_info = mapper.get_linear_scale_info()
    print("\nLinear Scale Mapping:")
    print("Fingers | Radius | Frequency | Ratio | Linear Scale")
    print("-" * 52)
    for finger_count, info in scale_info.items():
        print(f"   {finger_count}    | {info['radius']:.3f} |  {info['frequency']:.3f}   | {info['radius_ratio']:.3f} |    {info['linear_scale']:.3f}")
    
    # Test gesture sequence
    print("\nTesting gesture sequence...")
    gestures = [
        ("no_hand", "no_hand"),
        ("fist", "no_hand"),
        ("one_finger", "no_hand"),
        ("three_fingers", "no_hand"),
        ("open_hand", "no_hand"),
        ("three_fingers", "two_fingers"),
    ]
    
    for left, right in gestures:
        mapper.update_hand_states(left, right)
        params = mapper.get_parameters()
        print(f"L:{left:12} R:{right:12} -> r1={params['r1']:.3f} r2={params['r2']:.3f} w1={params['w1']:.3f} w2={params['w2']:.3f}")