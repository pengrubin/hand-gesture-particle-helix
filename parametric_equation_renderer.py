#!/usr/bin/env python3
"""
Parametric Equation Renderer for Complex Parametric Visualization
z(θ) = r1*e^(i*(w1*θ+p1)) + r2*e^(i*(w2*θ+p2))

This module provides a class to render and animate complex parametric equations
with adjustable parameters suitable for gesture-based control.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional, Dict
import time


class ParametricEquationRenderer:
    """
    Renders parametric equations of the form:
    z(θ) = r1*e^(i*(w1*θ+p1)) + r2*e^(i*(w2*θ+p2))
    
    Features:
    - Real-time parameter adjustment
    - Trajectory accumulation
    - Rotating rod visualization
    - Smooth animation
    """
    
    def __init__(self, 
                 r1: float = 4.0, r2: float = 4.0,  # P.Georges: Equal large radii for proper flower pattern
                 w1: float = 1.0, w2: float = -1.96,  # Match P.Georges pattern: w2 = -2 + 1/25 for flower pattern
                 p1: float = 0.0, p2: float = 0.0,
                 max_theta: float = 8 * np.pi,
                 num_points: int = 8000,
                 trail_length: int = 500):
        """
        Initialize the parametric equation renderer.
        
        Args:
            r1, r2: Radius parameters
            w1, w2: Frequency parameters
            p1, p2: Phase parameters
            max_theta: Maximum theta value for trajectory
            num_points: Number of points in trajectory
            trail_length: Length of accumulated trail
        """
        # Equation parameters
        self.r1 = r1
        self.r2 = r2
        self.w1 = w1
        self.w2 = w2
        self.p1 = p1
        self.p2 = p2
        
        # Rendering parameters
        self.max_theta = max_theta
        self.num_points = num_points
        self.trail_length = trail_length
        
        # Animation state
        self.current_theta = 0.0
        self.theta_step = 0.08  # Adjusted for ~5 minute complete P.Georges cycles (3 full 25-rotation cycles)
        self.trajectory_points = []
        
        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()
        
        # Plot elements
        self.trajectory_line = None
        self.current_point = None
        self.rod1_line = None
        self.rod2_line = None
        self.accumulated_trail = None
        self.parameter_text = None
        
        self.initialize_plot_elements()
    
    def setup_plot(self) -> None:
        """Configure the matplotlib plot."""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('Real Part')
        self.ax.set_ylabel('Imaginary Part')
        self.ax.set_title('Parametric Equation Visualization\nz(θ) = r₁e^(i(ω₁θ+φ₁)) + r₂e^(i(ω₂θ+φ₂))')
    
    def initialize_plot_elements(self) -> None:
        """Initialize all plot elements."""
        # Full trajectory (light blue)
        self.trajectory_line, = self.ax.plot([], [], 'lightblue', alpha=0.6, linewidth=1)
        
        # Accumulated trail (blue gradient effect will be simulated)
        self.accumulated_trail, = self.ax.plot([], [], 'blue', alpha=0.8, linewidth=2)
        
        # Current point (red)
        self.current_point, = self.ax.plot([], [], 'ro', markersize=8)
        
        # Rod 1 (green)
        self.rod1_line, = self.ax.plot([], [], 'g-', linewidth=3, alpha=0.7)
        
        # Rod 2 (orange)
        self.rod2_line, = self.ax.plot([], [], 'orange', linewidth=2, alpha=0.7)
        
        # Parameter display
        self.parameter_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                         verticalalignment='top', fontfamily='monospace',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def compute_complex_point(self, theta: float) -> complex:
        """
        Compute the complex point z(θ) for given theta.
        
        Args:
            theta: Parameter value
            
        Returns:
            Complex number representing the point
        """
        # First component: r1 * e^(i*(w1*θ + p1))
        z1 = self.r1 * np.exp(1j * (self.w1 * theta + self.p1))
        
        # Second component: r2 * e^(i*(w2*θ + p2))
        z2 = self.r2 * np.exp(1j * (self.w2 * theta + self.p2))
        
        # Combined result
        return z1 + z2
    
    def compute_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the full trajectory for current parameters.
        
        Returns:
            Tuple of (x_coords, y_coords) arrays
        """
        theta_values = np.linspace(0, self.max_theta, self.num_points)
        complex_points = [self.compute_complex_point(theta) for theta in theta_values]
        
        x_coords = np.array([z.real for z in complex_points])
        y_coords = np.array([z.imag for z in complex_points])
        
        return x_coords, y_coords
    
    def compute_rod_positions(self, theta: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Compute positions for the two rotating rods at given theta.
        
        Args:
            theta: Current parameter value
            
        Returns:
            Tuple of ((rod1_x, rod1_y), (rod2_x, rod2_y), (final_x, final_y))
        """
        # Rod 1: from origin to r1*e^(i*(w1*θ + p1))
        z1 = self.r1 * np.exp(1j * (self.w1 * theta + self.p1))
        rod1_pos = (z1.real, z1.imag)
        
        # Rod 2: from rod1 endpoint to r2*e^(i*(w2*θ + p2))
        z2_relative = self.r2 * np.exp(1j * (self.w2 * theta + self.p2))
        z2_absolute = z1 + z2_relative
        rod2_pos = (z2_absolute.real, z2_absolute.imag)
        
        # Final position
        final_pos = rod2_pos
        
        return rod1_pos, rod2_pos, final_pos
    
    def update_parameters(self, r1: Optional[float] = None, r2: Optional[float] = None,
                         w1: Optional[float] = None, w2: Optional[float] = None,
                         p1: Optional[float] = None, p2: Optional[float] = None) -> None:
        """
        Update equation parameters.
        
        Args:
            r1, r2: New radius values (if provided)
            w1, w2: New frequency values (if provided)
            p1, p2: New phase values (if provided)
        """
        if r1 is not None:
            self.r1 = max(0.01, r1)  # Prevent zero radius
        if r2 is not None:
            self.r2 = max(0.01, r2)
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2
        if p1 is not None:
            self.p1 = p1
        if p2 is not None:
            self.p2 = p2
        
        # Update plot limits based on new parameters
        max_radius = self.r1 + self.r2 + 0.5
        self.ax.set_xlim(-max_radius, max_radius)
        self.ax.set_ylim(-max_radius, max_radius)
    
    def update_trail(self, new_point: Tuple[float, float]) -> None:
        """
        Update the accumulated trail with a new point.
        
        Args:
            new_point: (x, y) coordinates of new point
        """
        self.trajectory_points.append(new_point)
        
        # Keep only the most recent points
        if len(self.trajectory_points) > self.trail_length:
            self.trajectory_points = self.trajectory_points[-self.trail_length:]
    
    def get_parameter_display_text(self) -> str:
        """Generate parameter display text."""
        return (f"r₁ = {self.r1:.3f}    r₂ = {self.r2:.3f}\n"
                f"ω₁ = {self.w1:.3f}    ω₂ = {self.w2:.3f}\n"
                f"φ₁ = {self.p1:.3f}    φ₂ = {self.p2:.3f}\n"
                f"θ = {self.current_theta:.3f}\n"
                f"Live Gesture Control Active")
    
    def animate_frame(self, frame: int) -> List:
        """
        Animation frame update function.
        
        Args:
            frame: Frame number (for FuncAnimation)
            
        Returns:
            List of updated plot elements
        """
        # Update theta
        self.current_theta += self.theta_step
        
        # Continue animation indefinitely - remove reset to let it run for 5+ minutes
        # if self.current_theta > self.max_theta:
        #     self.current_theta = 0.0
        #     self.trajectory_points = []  # Clear trail on reset
        
        # Don't display future trajectory - remove full trajectory preview
        # Only show the accumulated trail that's been walked
        self.trajectory_line.set_data([], [])  # Hide future trajectory
        
        # Compute current rod positions
        rod1_pos, rod2_pos, final_pos = self.compute_rod_positions(self.current_theta)
        
        # Update rod 1 (from origin to first endpoint)
        self.rod1_line.set_data([0, rod1_pos[0]], [0, rod1_pos[1]])
        
        # Update rod 2 (from first endpoint to final position)
        self.rod2_line.set_data([rod1_pos[0], rod2_pos[0]], [rod1_pos[1], rod2_pos[1]])
        
        # Update current point
        self.current_point.set_data([final_pos[0]], [final_pos[1]])
        
        # Update accumulated trail
        self.update_trail(final_pos)
        if len(self.trajectory_points) > 1:
            trail_x = [p[0] for p in self.trajectory_points]
            trail_y = [p[1] for p in self.trajectory_points]
            self.accumulated_trail.set_data(trail_x, trail_y)
        
        # Update parameter text
        self.parameter_text.set_text(self.get_parameter_display_text())
        
        return [self.trajectory_line, self.current_point, self.rod1_line, 
                self.rod2_line, self.accumulated_trail, self.parameter_text]
    
    def start_animation(self, interval: int = 50) -> FuncAnimation:
        """
        Start the animation.
        
        Args:
            interval: Animation interval in milliseconds
            
        Returns:
            FuncAnimation object
        """
        anim = FuncAnimation(self.fig, self.animate_frame, interval=interval, 
                           blit=True, cache_frame_data=False)
        return anim
    
    def show(self) -> None:
        """Display the plot."""
        plt.show()
    
    def reset_animation(self) -> None:
        """Reset animation to initial state."""
        self.current_theta = 0.0
        self.trajectory_points = []
    
    def set_theta_step(self, step: float) -> None:
        """Set the theta increment per frame."""
        self.theta_step = max(0.001, step)  # Prevent negative or zero steps
    
    def get_current_state(self) -> dict:
        """
        Get current renderer state.
        
        Returns:
            Dictionary containing current parameters and state
        """
        return {
            'r1': self.r1, 'r2': self.r2,
            'w1': self.w1, 'w2': self.w2,
            'p1': self.p1, 'p2': self.p2,
            'current_theta': self.current_theta,
            'theta_step': self.theta_step,
            'trail_length': len(self.trajectory_points)
        }
    
    def set_gesture_indicators(self, left_hand_gesture: Optional[int] = None,
                              right_hand_gesture: Optional[int] = None,
                              hands_detected: int = 0,
                              is_paused: bool = False) -> None:
        """
        Set gesture indicators for visual feedback.
        
        Args:
            left_hand_gesture: Left hand finger count (0-5)
            right_hand_gesture: Right hand finger count (0-5)
            hands_detected: Total number of hands detected
            is_paused: Whether the system is paused
        """
        self.gesture_state = {
            'left_hand_gesture': left_hand_gesture,
            'right_hand_gesture': right_hand_gesture,
            'hands_detected': hands_detected,
            'is_paused': is_paused
        }
        
        # Update parameter text to include gesture info
        self.update_gesture_display()
    
    def update_gesture_display(self) -> None:
        """Update parameter display with gesture information."""
        if hasattr(self, 'gesture_state'):
            gesture_info = []
            
            if self.gesture_state.get('is_paused', False):
                gesture_info.append("PAUSED")
            
            left_gesture = self.gesture_state.get('left_hand_gesture')
            right_gesture = self.gesture_state.get('right_hand_gesture')
            
            if left_gesture is not None:
                gesture_info.append(f"L:{left_gesture}")
            if right_gesture is not None:
                gesture_info.append(f"R:{right_gesture}")
                
            hands_count = self.gesture_state.get('hands_detected', 0)
            gesture_info.append(f"Hands:{hands_count}")
            
            # Update title with gesture info
            gesture_str = " | ".join(gesture_info) if gesture_info else "No gestures"
            self.ax.set_title(f'Parametric Equation Visualization\nz(θ) = r₁e^(i(ω₁θ+φ₁)) + r₂e^(i(ω₂θ+φ₂))\n{gesture_str}')
    
    def add_twelve_tone_markers(self) -> None:
        """Add twelve-tone scale radius markers to the plot."""
        # Add circular markers at twelve-tone scale radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        # Twelve-tone radii (example values)
        reference_radii = [0.5, 0.63, 0.79, 1.0, 1.26, 1.59, 2.0]  # Approximate twelve-tone scale
        colors = ['lightgray', 'silver', 'darkgray', 'gray', 'dimgray', 'black', 'red']
        
        for i, (radius, color) in enumerate(zip(reference_radii, colors)):
            if radius <= self.r1 + self.r2 + 0.5:  # Only show relevant radii
                x_circle = radius * np.cos(theta_circle)
                y_circle = radius * np.sin(theta_circle)
                self.ax.plot(x_circle, y_circle, '--', color=color, alpha=0.3, linewidth=1)
                
                # Add radius labels
                if i % 2 == 0:  # Label every other circle to avoid clutter
                    self.ax.text(radius + 0.1, 0, f'{radius:.1f}', fontsize=8, alpha=0.7)
    
    def highlight_parameter_changes(self, old_params: Dict[str, float], new_params: Dict[str, float]) -> None:
        """
        Highlight parameter changes with visual indicators.
        
        Args:
            old_params: Previous parameter values
            new_params: New parameter values
        """
        # Calculate parameter change magnitudes
        changes = {}
        for key in new_params:
            if key in old_params:
                changes[key] = abs(new_params[key] - old_params[key])
        
        # Highlight significant changes (could be implemented with color changes)
        significant_threshold = 0.1
        significant_changes = {k: v for k, v in changes.items() if v > significant_threshold}
        
        if significant_changes:
            # Could add visual feedback here (e.g., brief color change)
            pass
    
    def set_real_time_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable real-time gesture control mode.
        
        Args:
            enabled: Whether to enable real-time mode
        """
        self.real_time_mode = enabled
        
        if enabled:
            # Optimize for real-time performance
            self.theta_step = 0.08  # Adjusted for P.Georges cycles
            # Could add performance optimizations here
            
            # Add twelve-tone scale markers
            self.add_twelve_tone_markers()
        else:
            # Standard mode
            self.theta_step = 0.08  # Adjusted for P.Georges cycles
            
        # Initialize gesture state
        if not hasattr(self, 'gesture_state'):
            self.gesture_state = {
                'left_hand_gesture': None,
                'right_hand_gesture': None,
                'hands_detected': 0,
                'is_paused': False
            }


if __name__ == "__main__":
    # Simple test
    renderer = ParametricEquationRenderer(
        r1=1.5, r2=0.8,
        w1=1.0, w2=3.0,
        p1=0.0, p2=np.pi/4
    )
    
    print("Starting parametric equation visualization...")
    print("Close the window to exit.")
    
    anim = renderer.start_animation(interval=50)
    renderer.show()