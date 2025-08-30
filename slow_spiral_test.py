"""
Very Slow Euler Spiral Test
Demonstrates the effect of slow progression speed for smooth curve visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

def euler_spiral_formula(theta):
    """Euler spiral formula"""
    term1 = 11 * np.exp(1j * (11 * theta))
    term2 = 14 * np.sin(10 * theta) * np.exp(1j * theta)  
    term3 = 13 * np.exp(1j * theta)
    return term1 + term2 + term3

def create_slow_animation():
    """Create very slow animation showing smooth curve formation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Animation parameters - very slow and detailed
    theta_max = 2 * np.pi  # Smaller range for detailed view
    total_frames = 1200    # Many frames for very smooth progression
    trail_length = 600     # Long trail to see curve formation
    
    # Setup plots
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(-40, 40)
    ax1.set_title('Very Slow Euler Spiral Formation\nSpeed: 1/10 of Original', fontsize=12)
    ax1.set_xlabel('Real Part (X)')
    ax1.set_ylabel('Imaginary Part (Y)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(-40, 40) 
    ax2.set_title('Current Point with History\nRed = Recent, Blue = Older', fontsize=12)
    ax2.set_xlabel('Real Part (X)')
    ax2.set_ylabel('Imaginary Part (Y)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Line objects
    trail_line1, = ax1.plot([], [], 'b-', linewidth=1.0, alpha=0.8)
    current_point1, = ax1.plot([], [], 'ro', markersize=8)
    
    trail_line2, = ax2.plot([], [], 'b-', linewidth=0.5, alpha=0.6)
    recent_trail2, = ax2.plot([], [], 'r-', linewidth=2.0, alpha=0.9)
    current_point2, = ax2.plot([], [], 'yo', markersize=10, markeredgecolor='red', markeredgewidth=2)
    
    # Progress indicator
    progress_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Store all computed points for smooth trails
    all_points_x = []
    all_points_y = []
    
    def animate(frame):
        # Current progress through theta range
        progress = frame / total_frames
        current_theta = progress * theta_max
        
        # Calculate current point
        current_complex = euler_spiral_formula(current_theta)
        current_x = current_complex.real
        current_y = current_complex.imag
        
        # Store point
        all_points_x.append(current_x)
        all_points_y.append(current_y)
        
        # Update full trail (left plot)
        if len(all_points_x) > 1:
            trail_line1.set_data(all_points_x, all_points_y)
        current_point1.set_data([current_x], [current_y])
        
        # Update recent trail (right plot) - show last N points with emphasis
        recent_start = max(0, len(all_points_x) - 100)  # Last 100 points in red
        trail_start = max(0, len(all_points_x) - trail_length)
        
        if len(all_points_x) > trail_start:
            # Blue trail (older points)
            trail_line2.set_data(all_points_x[trail_start:recent_start], 
                               all_points_y[trail_start:recent_start])
        
        if len(all_points_x) > recent_start:
            # Red trail (recent points)
            recent_trail2.set_data(all_points_x[recent_start:], 
                                 all_points_y[recent_start:])
        
        # Current point
        current_point2.set_data([current_x], [current_y])
        
        # Update progress
        progress_text.set_text(f'Progress: {progress*100:.1f}%\n'
                             f'θ = {current_theta:.3f}\n'
                             f'Points: {len(all_points_x):,}\n'
                             f'Current: ({current_x:.2f}, {current_y:.2f})')
        
        return trail_line1, current_point1, trail_line2, recent_trail2, current_point2, progress_text
    
    # Create animation with very slow progression
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                interval=80, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Add instructions
    fig.suptitle('Euler Formula: z(θ) = 11e^(i11θ) + 14sin(10θ)e^(iθ) + 13e^(iθ)\n'
                'Very Slow Animation - Watch the Smooth Curve Formation', 
                fontsize=14, y=0.98)
    
    return ani

if __name__ == "__main__":
    print("🐌 Very Slow Euler Spiral Animation")
    print("=" * 50)
    print("Formula: z(θ) = 11e^(i11θ) + 14sin(10θ)e^(iθ) + 13e^(iθ)")
    print()
    print("This animation runs at 1/10 the original speed")
    print("Watch how the smooth curves form gradually...")
    print()
    print("Left: Full trajectory formation")
    print("Right: Current point with trail highlighting")
    print()
    print("Close the window to exit.")
    
    ani = create_slow_animation()
    plt.show()
    
    print("Animation completed!")