#!/usr/bin/env python3
"""
Main Prototype for Parametric Equation Visualization with Gesture Control

This prototype combines the parametric equation renderer with gesture-based
parameter control using the twelve-tone scale mapping system.

Features:
- Real-time parametric equation visualization
- Keyboard controls for simulating hand gestures
- Twelve-tone scale parameter mapping
- Smooth parameter transitions
- Interactive parameter adjustment

Controls:
- Number keys 1-5: Simulate left hand finger counts
- Shift + 1-5: Simulate right hand finger counts  
- WASD: Move left hand position
- Arrow keys: Move right hand position
- Space: Reset to no hands
- R: Reset animation
- +/-: Adjust animation speed
- ESC/Q: Quit
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import Dict, List, Optional
import time

# Import our custom modules
from parametric_equation_renderer import ParametricEquationRenderer
from gesture_radius_mapper import GestureToRadiusMapper, HandState


class InteractivePrototype:
    """
    Interactive prototype combining parametric equation visualization 
    with gesture-based control simulation.
    """
    
    def __init__(self):
        """Initialize the interactive prototype."""
        # Core components
        self.renderer = ParametricEquationRenderer(
            r1=4.0, r2=4.0,  # P.Georges: Equal large radii
            w1=1.0, w2=-1.96,  # P.Georges: w2 = -2 + 1/25
            p1=0.0, p2=0.0,
            max_theta=150 * np.pi,  # 75 full rotations for ~5 minutes
            num_points=12000,       # Much more points for ultra-smooth curves
            trail_length=5000       # Longer trail, won't disappear quickly
        )
        
        self.gesture_mapper = GestureToRadiusMapper(r_max=4.0, base_frequency=1.0)  # P.Georges: Match large radii scale
        
        # Control state
        self.left_hand_fingers = 0
        self.right_hand_fingers = 0
        self.left_hand_x = 0.0
        self.left_hand_y = 0.0
        self.right_hand_x = 0.0
        self.right_hand_y = 0.0
        self.position_step = 0.1
        
        # Animation control
        self.is_paused = False
        self.show_help = True
        
        # Setup matplotlib
        self.setup_interactive_plot()
        
        # Connect keyboard events
        self.renderer.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print("Interactive Parametric Equation Prototype")
        print("=========================================")
        print(self.get_help_text())
    
    def setup_interactive_plot(self) -> None:
        """Setup the interactive plot with additional UI elements."""
        # Adjust figure layout for additional info panels
        self.renderer.fig.set_size_inches(16, 10)
        
        # Create additional text areas for gesture info
        self.gesture_info_text = self.renderer.ax.text(
            0.02, 0.02, '', transform=self.renderer.ax.transAxes,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        )
        
        self.control_info_text = self.renderer.ax.text(
            0.98, 0.98, '', transform=self.renderer.ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
    
    def get_help_text(self) -> str:
        """Generate help text for controls."""
        return """
Controls:
---------
1-5:        Left hand finger count
Shift+1-5:  Right hand finger count  
WASD:       Move left hand position
Arrow keys: Move right hand position
Space:      Reset to no hands
R:          Reset animation
+/-:        Adjust speed
P:          Pause/unpause
H:          Toggle help
ESC/Q:      Quit

Linear Mapping:
- Each finger count maps to proportionally increasing radius/frequency
- Smooth linear progression from 1 to 5 fingers
- Formula: r = r_min + (fingers/5) * (r_max - r_min)
"""
    
    def finger_count_to_hand_state(self, finger_count: int) -> str:
        """Convert finger count to hand state string."""
        state_map = {
            0: "fist",
            1: "one_finger", 
            2: "two_fingers",
            3: "three_fingers",
            4: "four_fingers",
            5: "open_hand"
        }
        return state_map.get(finger_count, "fist")
    
    def update_gesture_states(self) -> None:
        """Update gesture mapper with current simulated hand states."""
        left_state = self.finger_count_to_hand_state(self.left_hand_fingers)
        right_state = self.finger_count_to_hand_state(self.right_hand_fingers)
        
        # Apply no-hand state if finger count is 0 and position is at origin
        if self.left_hand_fingers == 0 and abs(self.left_hand_x) < 0.01 and abs(self.left_hand_y) < 0.01:
            left_state = "no_hand"
        if self.right_hand_fingers == 0 and abs(self.right_hand_x) < 0.01 and abs(self.right_hand_y) < 0.01:
            right_state = "no_hand"
        
        self.gesture_mapper.update_hand_states(
            left_hand_gesture=left_state,
            right_hand_gesture=right_state,
            left_hand_position=(self.left_hand_x, self.left_hand_y),
            right_hand_position=(self.right_hand_x, self.right_hand_y)
        )
    
    def on_key_press(self, event) -> None:
        """Handle keyboard input for gesture simulation."""
        key = event.key.lower()
        
        # Quit commands
        if key in ['q', 'escape']:
            plt.close('all')
            sys.exit(0)
        
        # Left hand finger counts (1-5)
        elif key in ['1', '2', '3', '4', '5']:
            self.left_hand_fingers = int(key)
            print(f"Left hand: {self.left_hand_fingers} fingers")
        
        # Right hand finger counts (Shift + 1-5)
        elif key in ['!', '@', '#', '$', '%']:  # Shift + 1-5
            shift_map = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5}
            self.right_hand_fingers = shift_map[key]
            print(f"Right hand: {self.right_hand_fingers} fingers")
        
        # Left hand position (WASD)
        elif key == 'w':
            self.left_hand_y = min(1.0, self.left_hand_y + self.position_step)
            print(f"Left hand position: ({self.left_hand_x:.2f}, {self.left_hand_y:.2f})")
        elif key == 's':
            self.left_hand_y = max(-1.0, self.left_hand_y - self.position_step)
            print(f"Left hand position: ({self.left_hand_x:.2f}, {self.left_hand_y:.2f})")
        elif key == 'a':
            self.left_hand_x = max(-1.0, self.left_hand_x - self.position_step)
            print(f"Left hand position: ({self.left_hand_x:.2f}, {self.left_hand_y:.2f})")
        elif key == 'd':
            self.left_hand_x = min(1.0, self.left_hand_x + self.position_step)
            print(f"Left hand position: ({self.left_hand_x:.2f}, {self.left_hand_y:.2f})")
        
        # Right hand position (Arrow keys)
        elif key == 'up':
            self.right_hand_y = min(1.0, self.right_hand_y + self.position_step)
            print(f"Right hand position: ({self.right_hand_x:.2f}, {self.right_hand_y:.2f})")
        elif key == 'down':
            self.right_hand_y = max(-1.0, self.right_hand_y - self.position_step)
            print(f"Right hand position: ({self.right_hand_x:.2f}, {self.right_hand_y:.2f})")
        elif key == 'left':
            self.right_hand_x = max(-1.0, self.right_hand_x - self.position_step)
            print(f"Right hand position: ({self.right_hand_x:.2f}, {self.right_hand_y:.2f})")
        elif key == 'right':
            self.right_hand_x = min(1.0, self.right_hand_x + self.position_step)
            print(f"Right hand position: ({self.right_hand_x:.2f}, {self.right_hand_y:.2f})")
        
        # Reset commands
        elif key == ' ':  # Space - reset to no hands
            self.left_hand_fingers = 0
            self.right_hand_fingers = 0
            self.left_hand_x = self.left_hand_y = 0.0
            self.right_hand_x = self.right_hand_y = 0.0
            self.gesture_mapper.reset_parameters()
            print("Reset to no hands")
        
        elif key == 'r':  # Reset animation
            self.renderer.reset_animation()
            print("Animation reset")
        
        # Animation controls
        elif key == '+' or key == '=':
            current_step = self.renderer.theta_step
            self.renderer.set_theta_step(current_step * 1.2)
            print(f"Speed increased: {self.renderer.theta_step:.4f}")
        
        elif key == '-' or key == '_':
            current_step = self.renderer.theta_step
            self.renderer.set_theta_step(current_step / 1.2)
            print(f"Speed decreased: {self.renderer.theta_step:.4f}")
        
        elif key == 'p':  # Pause/unpause
            self.is_paused = not self.is_paused
            print("Paused" if self.is_paused else "Resumed")
        
        elif key == 'h':  # Toggle help
            self.show_help = not self.show_help
            print("Help toggled")
    
    def get_gesture_info_text(self) -> str:
        """Generate gesture information display text."""
        left_state = self.finger_count_to_hand_state(self.left_hand_fingers)
        right_state = self.finger_count_to_hand_state(self.right_hand_fingers)
        
        if self.left_hand_fingers == 0 and abs(self.left_hand_x) < 0.01 and abs(self.left_hand_y) < 0.01:
            left_state = "no_hand"
        if self.right_hand_fingers == 0 and abs(self.right_hand_x) < 0.01 and abs(self.right_hand_y) < 0.01:
            right_state = "no_hand"
        
        return (f"Gesture States:\n"
                f"Left:  {left_state} ({self.left_hand_fingers})\n"
                f"       pos: ({self.left_hand_x:.2f}, {self.left_hand_y:.2f})\n"
                f"Right: {right_state} ({self.right_hand_fingers})\n"
                f"       pos: ({self.right_hand_x:.2f}, {self.right_hand_y:.2f})")
    
    def get_control_info_text(self) -> str:
        """Generate control information display text."""
        if not self.show_help:
            return "Press 'H' for help"
        
        return ("Controls:\n"
                "1-5: Left fingers\n"
                "Shift+1-5: Right fingers\n"
                "WASD: Left position\n"
                "Arrows: Right position\n"
                "Space: Reset\n"
                "R: Reset animation\n"
                "+/-: Speed\n"
                "P: Pause\n"
                "H: Toggle help\n"
                "Q/Esc: Quit")
    
    def enhanced_animate_frame(self, frame: int) -> List:
        """
        Enhanced animation frame function that includes gesture control.
        
        Args:
            frame: Frame number
            
        Returns:
            List of updated plot elements
        """
        if not self.is_paused:
            # Update gesture states
            self.update_gesture_states()
            
            # Get parameters from gesture mapper
            gesture_params = self.gesture_mapper.get_parameters()
            
            # Update renderer parameters
            self.renderer.update_parameters(**gesture_params)
            
            # Update gesture info display
            self.gesture_info_text.set_text(self.get_gesture_info_text())
            self.control_info_text.set_text(self.get_control_info_text())
            
            # Call original animation frame
            animated_elements = self.renderer.animate_frame(frame)
            
            # Add our UI elements
            animated_elements.extend([self.gesture_info_text, self.control_info_text])
            
            return animated_elements
        else:
            # Just update UI when paused
            self.gesture_info_text.set_text(self.get_gesture_info_text())
            self.control_info_text.set_text(self.get_control_info_text())
            return [self.gesture_info_text, self.control_info_text]
    
    def start_interactive_demo(self) -> None:
        """Start the interactive demonstration."""
        print("\nStarting interactive demo...")
        print("Use keyboard controls to simulate hand gestures.")
        print("Watch how the twelve-tone mapping affects the parameters!")
        
        # Start animation with enhanced frame function
        self.animation = FuncAnimation(
            self.renderer.fig, 
            self.enhanced_animate_frame, 
            interval=50,
            blit=True, 
            cache_frame_data=False
        )
        
        # Show the plot
        plt.show()


def demonstrate_linear_mapping():
    """Demonstrate the linear mapping without animation."""
    print("\nLinear Mapping Demonstration")
    print("=" * 40)
    
    mapper = GestureToRadiusMapper(r_max=2.0)
    scale_info = mapper.get_linear_scale_info()
    
    print("\nFinger Count -> Radius/Frequency Mapping:")
    print("Fingers | Radius | Freq | Ratio | Linear Scale")
    print("-" * 48)
    
    for finger_count in range(6):
        info = scale_info[finger_count]
        linear_scale = info['linear_scale']
        
        print(f"   {finger_count}    | {info['radius']:.3f} | {info['frequency']:.3f} | {info['radius_ratio']:.3f} |    {linear_scale:.3f}")
    
    print("\nThis mapping ensures smooth linear progression!")
    print("Each finger count increases radius and frequency proportionally.")


def run_automated_demo():
    """Run an automated demonstration sequence."""
    print("\nRunning automated gesture sequence demo...")
    
    prototype = InteractivePrototype()
    
    # Define a sequence of gestures to demonstrate
    demo_sequence = [
        (0, 0, 0.0, 0.0, 0.0, 0.0),  # No hands
        (3, 0, 0.0, 0.0, 0.0, 0.0),  # Left hand 3 fingers
        (3, 2, 0.0, 0.0, 0.0, 0.0),  # Both hands
        (5, 1, 0.5, 0.5, -0.5, -0.5),  # Both hands with positions
        (1, 5, -0.8, 0.3, 0.8, -0.3),  # Swapped finger counts
    ]
    
    frame_count = 0
    sequence_index = 0
    frames_per_gesture = 120  # 2.4 seconds at 50ms intervals
    
    def demo_animate(frame):
        nonlocal frame_count, sequence_index
        
        # Update gesture every N frames
        if frame_count % frames_per_gesture == 0 and sequence_index < len(demo_sequence):
            left_f, right_f, left_x, left_y, right_x, right_y = demo_sequence[sequence_index]
            prototype.left_hand_fingers = left_f
            prototype.right_hand_fingers = right_f
            prototype.left_hand_x = left_x
            prototype.left_hand_y = left_y
            prototype.right_hand_x = right_x
            prototype.right_hand_y = right_y
            
            print(f"Demo step {sequence_index + 1}: L={left_f}@({left_x:.1f},{left_y:.1f}) R={right_f}@({right_x:.1f},{right_y:.1f})")
            sequence_index += 1
        
        frame_count += 1
        return prototype.enhanced_animate_frame(frame)
    
    # Start automated demo
    anim = FuncAnimation(prototype.renderer.fig, demo_animate, interval=50, 
                        blit=True, cache_frame_data=False)
    
    plt.show()


if __name__ == "__main__":
    print("Parametric Equation Visualization Prototype")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "mapping":
            demonstrate_linear_mapping()
        elif mode == "auto":
            run_automated_demo()
        elif mode == "interactive":
            prototype = InteractivePrototype()
            prototype.start_interactive_demo()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: mapping, auto, interactive")
    else:
        # Default: run interactive mode
        prototype = InteractivePrototype()
        prototype.start_interactive_demo()