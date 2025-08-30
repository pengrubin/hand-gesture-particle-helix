"""
Quick test to verify the improved Euler formula visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def euler_spiral_formula(theta):
    """Euler spiral formula with academic credits"""
    term1 = 11 * np.exp(1j * (11 * theta))
    term2 = 14 * np.sin(10 * theta) * np.exp(1j * theta)  
    term3 = 13 * np.exp(1j * theta)
    return term1 + term2 + term3

def quick_visualization():
    """Generate a quick high-resolution visualization"""
    print("🧬 Euler Formula Spiral - Quick Test")
    print("Formula: z(θ) = 11e^(i11θ) + 14sin(10θ)e^(iθ) + 13e^(iθ)")
    print("Credits: Patrick Georges (U. Ottawa), Chirag Dudhat")
    print()
    
    # High resolution sampling
    theta = np.linspace(0, 6*np.pi, 8000)  # Very high sampling rate
    complex_result = euler_spiral_formula(theta)
    
    x = complex_result.real
    y = complex_result.imag
    z = theta * 0.2
    
    # Create subplots
    fig = plt.figure(figsize=(16, 8))
    
    # High-res 2D plot
    ax1 = fig.add_subplot(221)
    ax1.plot(x, y, linewidth=0.3, alpha=0.8, color='blue')
    ax1.set_title('High Resolution 2D Trajectory\n(8000 sample points)', fontsize=10)
    ax1.set_xlabel('Real Part (X)')
    ax1.set_ylabel('Imaginary Part (Y)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Color-coded trajectory
    ax2 = fig.add_subplot(222)
    scatter = ax2.scatter(x, y, c=theta, cmap='plasma', s=0.1, alpha=0.8)
    ax2.set_title('Color-Coded Time Evolution', fontsize=10)
    ax2.set_xlabel('Real Part (X)')
    ax2.set_ylabel('Imaginary Part (Y)')
    plt.colorbar(scatter, ax=ax2, label='θ (Time)')
    ax2.axis('equal')
    
    # 3D visualization
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(x, y, z, linewidth=0.3, alpha=0.9)
    ax3.set_title('3D Trajectory (Time as Z-axis)', fontsize=10)
    ax3.set_xlabel('Real Part (X)')
    ax3.set_ylabel('Imaginary Part (Y)')
    ax3.set_zlabel('Time (Z)')
    
    # Phase space analysis
    ax4 = fig.add_subplot(224)
    ax4.plot(theta, np.abs(complex_result), linewidth=0.5, alpha=0.8, label='Magnitude')
    ax4.plot(theta, np.angle(complex_result), linewidth=0.5, alpha=0.8, label='Phase')
    ax4.set_title('Magnitude and Phase Analysis', fontsize=10)
    ax4.set_xlabel('θ (Time Parameter)')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('euler_spiral_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ High-resolution plot saved as 'euler_spiral_analysis.png'")
    
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Sample points: {len(theta):,}")
    print(f"- Trajectory range: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}]")
    print(f"- Magnitude range: [{np.abs(complex_result).min():.2f}, {np.abs(complex_result).max():.2f}]")
    print(f"- Total trajectory length: {np.sum(np.abs(np.diff(complex_result))):.2f}")

if __name__ == "__main__":
    quick_visualization()