# 📚 API Reference Guide

Complete Python API documentation for the parametric equation gesture control system.

## 🏗️ Architecture Overview

### Core Module Structure

```
gesture-parametric-system/
├── hand_gesture_detector.py      # MediaPipe gesture recognition
├── gesture_radius_mapper.py      # Twelve-tone parameter mapping  
├── gesture_parametric_bridge.py  # Integration bridge
├── parametric_equation_renderer.py # Visualization renderer
├── particle_sphere_system.py     # 3D particle effects
├── render_engine.py              # OpenGL rendering engine
└── main_app.py                   # Application entry point
```

### Class Hierarchy

```python
# Primary Classes
HandGestureDetector()           # Gesture recognition
GestureToRadiusMapper()         # Parameter mapping
GestureParametricBridge()       # Integration layer
ParametricEquationRenderer()    # Visualization
ParticleSphereSystem()          # 3D effects
RenderEngine()                  # Graphics
```

## 👋 HandGestureDetector

### Class Definition

```python
class HandGestureDetector:
    """
    MediaPipe-based real-time hand gesture recognition.
    Detects hand landmarks, finger counts, and gesture classifications.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 2):
        """
        Initialize the hand gesture detector.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking  
            max_num_hands: Maximum number of hands to detect (1-2)
        """
```

### Key Methods

#### `start_camera(camera_id: int = 0) -> bool`
```python
def start_camera(self, camera_id: int = 0) -> bool:
    """
    Start camera capture for gesture detection.
    
    Args:
        camera_id: Camera device index (0 = default)
        
    Returns:
        bool: True if camera started successfully
        
    Example:
        detector = HandGestureDetector()
        if detector.start_camera(0):
            print("Camera started successfully")
    """
```

#### `get_gesture_data() -> Dict[str, Any]`
```python
def get_gesture_data(self) -> Dict[str, Any]:
    """
    Get current gesture data for both hands.
    
    Returns:
        dict: Complete gesture information
        
    Format:
        {
            'hands_detected': int,          # Number of hands (0-2)
            'gesture_strength': float,      # Overall gesture strength (0-1)
            'left_hand': {
                'detected': bool,
                'position': [x, y],         # Normalized coordinates (0-1)
                'finger_count': int,        # Extended fingers (0-5)
                'gesture': str,             # 'fist', 'one', 'two', etc.
                'openness': float,          # Hand openness (0-1)
                'rotation_angle': float     # Hand rotation in radians
            },
            'right_hand': { ... },          # Same structure as left_hand
            'both_hands_distance': float,   # Distance between hands (0-1)
            'combined_rotation': float      # Combined rotation angle
        }
        
    Example:
        data = detector.get_gesture_data()
        if data['hands_detected'] > 0:
            left = data['left_hand']
            print(f"Left hand: {left['finger_count']} fingers")
    """
```

#### `detect_finger_count(hand_landmarks) -> int`
```python
def detect_finger_count(self, hand_landmarks) -> int:
    """
    Count extended fingers from hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        int: Number of extended fingers (0-5)
        
    Implementation:
        - Thumb: Compare tip vs MCP joint
        - Fingers: Compare tip vs PIP joint
        - Account for hand orientation
    """
```

#### `classify_gesture(finger_count: int) -> str`
```python
def classify_gesture(self, finger_count: int) -> str:
    """
    Classify gesture based on finger count.
    
    Args:
        finger_count: Number of extended fingers
        
    Returns:
        str: Gesture name
        
    Mapping:
        0 → 'fist'
        1 → 'one'  
        2 → 'two'
        3 → 'three'
        4 → 'four'
        5 → 'open_hand'
    """
```

## 🎵 GestureToRadiusMapper

### Class Definition

```python
class GestureToRadiusMapper:
    """
    Maps hand gestures to radius parameters using twelve-tone scale mathematics.
    Provides smooth, musically harmonious parameter relationships.
    """
    
    def __init__(self, r_max: float = 2.0, base_frequency: float = 1.0):
        """
        Initialize the gesture mapper.
        
        Args:
            r_max: Maximum radius value (at 5 fingers)
            base_frequency: Base frequency for parameter scaling
        """
```

### Key Methods

#### `finger_count_to_radius(finger_count: int) -> float`
```python
def finger_count_to_radius(self, finger_count: int) -> float:
    """
    Convert finger count to radius using twelve-tone scale.
    
    Formula: r = r_max * 2^((finger_count - 5) * 2/12)
    
    Args:
        finger_count: Number of extended fingers (0-5)
        
    Returns:
        float: Radius value
        
    Example:
        mapper = GestureToRadiusMapper(r_max=2.0)
        radius = mapper.finger_count_to_radius(3)  # Returns ~1.78
    """
```

#### `update_from_gesture_data(gesture_data: Dict) -> Dict[str, float]`
```python
def update_from_gesture_data(self, gesture_data: Dict) -> Dict[str, float]:
    """
    Update all parameters from complete gesture data.
    
    Args:
        gesture_data: Output from HandGestureDetector.get_gesture_data()
        
    Returns:
        dict: Updated parameters
        
    Format:
        {
            'r1': float,    # Primary radius (left hand)
            'r2': float,    # Secondary radius (right hand)
            'w1': float,    # Primary frequency (left position)
            'w2': float,    # Secondary frequency (right position)  
            'p1': float,    # Primary phase (left vertical)
            'p2': float     # Secondary phase (right vertical)
        }
    """
```

#### `set_smoothing_factor(factor: float) -> None`
```python
def set_smoothing_factor(self, factor: float) -> None:
    """
    Set parameter smoothing factor.
    
    Args:
        factor: Smoothing factor (0.0-1.0)
                0.0 = no smoothing (immediate response)
                1.0 = maximum smoothing (very slow response)
                0.85 = recommended default
    """
```

## 🌉 GestureParametricBridge

### Class Definition

```python
class GestureParametricBridge:
    """
    Bridge between MediaPipe gesture detection and parametric equation visualization.
    Handles real-time parameter mapping, smoothing, and state management.
    """
    
    def __init__(self, 
                 r_max: float = 2.0,
                 smoothing_factor: float = 0.85,
                 hand_assignment: HandAssignment = HandAssignment.LEFT_R1_RIGHT_R2,
                 auto_pause: bool = True):
```

### Enums

#### `HandAssignment`
```python
class HandAssignment(Enum):
    """Hand assignment for parametric control."""
    LEFT_R1_RIGHT_R2 = "left_r1_right_r2"    # Left hand controls r1, right hand r2
    RIGHT_R1_LEFT_R2 = "right_r1_left_r2"    # Reversed assignment
    DOMINANT_PRIMARY = "dominant_primary"      # Auto-detect dominant hand
```

### Key Methods

#### `update_parameters(gesture_data: Dict) -> Dict[str, float]`
```python
def update_parameters(self, gesture_data: Dict) -> Dict[str, float]:
    """
    Main update method - converts gesture data to parametric parameters.
    
    Args:
        gesture_data: Current gesture information
        
    Returns:
        dict: Smoothed parametric parameters
        
    Process:
        1. Extract finger counts and positions
        2. Apply twelve-tone mapping
        3. Apply smoothing filter
        4. Handle hand assignment
        5. Manage pause/resume state
        
    Example:
        bridge = GestureParametricBridge()
        params = bridge.update_parameters(gesture_data)
        # Use params for visualization
    """
```

#### `set_hand_assignment(assignment: HandAssignment) -> None`
```python
def set_hand_assignment(self, assignment: HandAssignment) -> None:
    """
    Change hand assignment mode.
    
    Args:
        assignment: New hand assignment mode
        
    Example:
        bridge.set_hand_assignment(HandAssignment.RIGHT_R1_LEFT_R2)
    """
```

#### `toggle_pause() -> bool`
```python
def toggle_pause(self) -> bool:
    """
    Toggle pause state of parameter updates.
    
    Returns:
        bool: New pause state (True = paused)
        
    Use Case:
        - Freeze current parameters
        - Allow manual parameter adjustment
        - Performance optimization
    """
```

## 🎨 ParametricEquationRenderer

### Class Definition

```python
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
                 w1: float = 1.0, w2: float = -1.96,  # Flower pattern: w2 = -2 + 1/25
                 p1: float = 0.0, p2: float = 0.0,
                 max_theta: float = 8 * np.pi,
                 num_points: int = 1000,
                 trail_length: int = 500):
```

### Key Methods

#### `update_parameters(**kwargs) -> None`
```python
def update_parameters(self, **kwargs) -> None:
    """
    Update equation parameters dynamically.
    
    Args:
        **kwargs: Parameter updates (r1, r2, w1, w2, p1, p2)
        
    Example:
        renderer.update_parameters(r1=1.5, w2=4.0, p1=np.pi/4)
    """
```

#### `compute_trajectory() -> Tuple[np.ndarray, np.ndarray]`
```python
def compute_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complete trajectory points.
    
    Returns:
        tuple: (x_points, y_points) as numpy arrays
        
    Implementation:
        θ = linspace(0, max_theta, num_points)
        z₁ = r1 * e^(i*(w1*θ+p1))  
        z₂ = r2 * e^(i*(w2*θ+p2))
        z = z₁ + z₂
        return (z.real, z.imag)
    """
```

#### `animate_step(dt: float) -> Tuple[float, float]`
```python
def animate_step(self, dt: float) -> Tuple[float, float]:
    """
    Advance animation by one time step.
    
    Args:
        dt: Time delta in seconds
        
    Returns:
        tuple: Current (x, y) position
        
    Use Case:
        - Real-time animation
        - Interactive visualization
        - Trail accumulation
    """
```

#### `get_current_point() -> Tuple[float, float]`
```python
def get_current_point(self) -> Tuple[float, float]:
    """
    Get current point in animation.
    
    Returns:
        tuple: (x, y) coordinates of current point
    """
```

## 🌀 ParticleSphereSystem

### Class Definition

```python
class ParticleSphereSystem:
    """
    3D particle system with helix/spiral structures controlled by hand gestures.
    Supports 18 different spiral patterns including DNA, tornado, and galaxy spirals.
    """
    
    def __init__(self, max_particles: int = 1500):
```

### Available Spiral Patterns

```python
AVAILABLE_PATTERNS = [
    'sine_wave', 'cosine_wave', 'double_wave',      # Basic waves
    'spiral_line', 'zigzag_line', 'heart_curve',    # 2D curves  
    'infinity_curve', 'helix_3d', 'multiple_lines', # 3D patterns
    'double_helix', 'triple_helix', 'dna_structure', # Helix structures
    'twisted_ribbon', 'braided_lines', 'spiral_tower', # Complex patterns
    'coil_spring', 'tornado_helix', 'galaxy_spiral'  # Advanced spirals
]
```

### Key Methods

#### `update(dt: float, gesture_data: Dict, audio_data: Dict) -> None`
```python
def update(self, dt: float, gesture_data: Dict, audio_data: Dict = None) -> None:
    """
    Update particle system with gesture and audio data.
    
    Args:
        dt: Time delta in seconds
        gesture_data: Hand gesture information
        audio_data: Audio spectrum data (optional)
        
    Process:
        1. Update particle physics
        2. Apply gesture-based forces
        3. Handle spiral pattern changes
        4. Process audio-reactive effects
    """
```

#### `get_helix_points() -> Dict[str, np.ndarray]`
```python
def get_helix_points(self) -> Dict[str, np.ndarray]:
    """
    Get current helix/spiral structure points for rendering.
    
    Returns:
        dict: Rendering data
        
    Format:
        {
            'positions': np.ndarray,    # 3D positions (N×3)
            'colors': np.ndarray,       # RGBA colors (N×4)  
            'sizes': np.ndarray,        # Point sizes (N,)
            'pattern': str              # Current pattern name
        }
    """
```

#### `set_spiral_pattern(pattern: str) -> bool`
```python
def set_spiral_pattern(self, pattern: str) -> bool:
    """
    Change current spiral pattern.
    
    Args:
        pattern: Pattern name from AVAILABLE_PATTERNS
        
    Returns:
        bool: True if pattern changed successfully
        
    Example:
        system.set_spiral_pattern('dna_structure')
    """
```

#### `cycle_spiral_pattern() -> str`
```python
def cycle_spiral_pattern(self) -> str:
    """
    Cycle to next spiral pattern.
    
    Returns:
        str: New pattern name
        
    Use Case:
        - Keyboard controls (S key)
        - Automatic pattern cycling
        - Demo mode
    """
```

## 🖥️ RenderEngine

### Class Definition

```python
class RenderEngine:
    """
    OpenGL-based 3D rendering engine for particle systems and parametric curves.
    Handles camera control, lighting, and GPU-optimized rendering.
    """
    
    def __init__(self, width: int = 1400, height: int = 900, title: str = "Parametric Visualization"):
```

### Key Methods

#### `render_particles(positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray) -> None`
```python
def render_particles(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray) -> None:
    """
    Render particle system using OpenGL point sprites.
    
    Args:
        positions: 3D positions (N×3 array)
        colors: RGBA colors (N×4 array)  
        sizes: Point sizes (N array)
        
    Performance:
        - GPU vertex buffer objects
        - Batch rendering for efficiency
        - Automatic LOD (Level of Detail)
    """
```

#### `render_parametric_curve(x_points: np.ndarray, y_points: np.ndarray, z_points: np.ndarray = None) -> None`
```python
def render_parametric_curve(self, x_points: np.ndarray, y_points: np.ndarray, z_points: np.ndarray = None) -> None:
    """
    Render parametric curve as connected line segments.
    
    Args:
        x_points: X coordinates
        y_points: Y coordinates  
        z_points: Z coordinates (optional, defaults to 0)
        
    Features:
        - Smooth line rendering
        - Gradient coloring
        - Anti-aliasing
    """
```

#### `update_camera() -> None`
```python
def update_camera(self) -> None:
    """
    Update camera transformation matrices.
    
    Features:
        - Mouse orbit control
        - Smooth interpolation
        - Automatic bounds checking
        - Keyboard shortcuts (R to reset)
    """
```

## 🎵 Audio Integration Classes

### AudioSpectrumAnalyzer

```python
class AudioSpectrumAnalyzer:
    """Real-time audio frequency analysis for visual feedback."""
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        Get current audio analysis data.
        
        Returns:
            dict: Audio information
            
        Format:
            {
                'dominant_freq': float,      # Hz
                'pitch_class': str,          # Note name (C, D, E, etc.)
                'octave': int,               # Octave number  
                'pitch_intensity': float,    # Amplitude (0-1)
                'spectrum': np.ndarray       # Full frequency spectrum
            }
        """
```

## 🔧 Utility Functions

### Parameter Smoothing

```python
def smooth_parameter(current: float, target: float, factor: float = 0.85) -> float:
    """
    Exponential moving average parameter smoothing.
    
    Args:
        current: Current parameter value
        target: Target parameter value
        factor: Smoothing factor (0-1)
        
    Returns:
        float: Smoothed parameter value
        
    Formula:
        smoothed = current * factor + target * (1 - factor)
    """
```

### Coordinate Normalization

```python
def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to [-1, 1] range.
    
    Args:
        x, y: Pixel coordinates
        width, height: Image dimensions
        
    Returns:
        tuple: Normalized coordinates
    """
```

### Twelve-Tone Calculations

```python
def semitones_to_frequency_ratio(semitones: int) -> float:
    """
    Convert semitone interval to frequency ratio.
    
    Args:
        semitones: Number of semitones
        
    Returns:
        float: Frequency ratio
        
    Formula:
        ratio = 2^(semitones/12)
    """
```

## 🔄 Integration Example

### Complete Usage Example

```python
# Initialize components
detector = HandGestureDetector()
mapper = GestureToRadiusMapper(r_max=2.0)
bridge = GestureParametricBridge(smoothing_factor=0.85)
renderer = ParametricEquationRenderer()
particle_system = ParticleSphereSystem(max_particles=1500)
render_engine = RenderEngine(1400, 900)

# Main loop
detector.start_camera(0)
while True:
    # Get gesture data
    gesture_data = detector.get_gesture_data()
    
    # Update parameters
    parameters = bridge.update_parameters(gesture_data)
    
    # Update visualization
    renderer.update_parameters(**parameters)
    particle_system.update(dt, gesture_data)
    
    # Render frame
    trajectory_x, trajectory_y = renderer.compute_trajectory()
    helix_data = particle_system.get_helix_points()
    
    render_engine.clear_screen()
    render_engine.render_parametric_curve(trajectory_x, trajectory_y)
    render_engine.render_particles(
        helix_data['positions'],
        helix_data['colors'],
        helix_data['sizes']
    )
    render_engine.present()
```

## 🐛 Error Handling

### Common Exceptions

```python
class GestureDetectionError(Exception):
    """Raised when gesture detection fails."""
    pass

class CameraInitError(Exception):
    """Raised when camera initialization fails."""
    pass

class ParameterValidationError(Exception):  
    """Raised when parameter values are invalid."""
    pass

class RenderingError(Exception):
    """Raised when OpenGL rendering fails."""
    pass
```

### Error Handling Patterns

```python
def safe_gesture_update():
    """Safe gesture update with error handling."""
    try:
        gesture_data = detector.get_gesture_data()
        if gesture_data is None:
            return default_parameters
        
        parameters = bridge.update_parameters(gesture_data)
        return parameters
        
    except GestureDetectionError:
        # Fall back to previous parameters
        return bridge.current_parameters
    except Exception as e:
        logging.error(f"Gesture update failed: {e}")
        return default_parameters
```

---

*This API reference provides complete function signatures and usage examples. For mathematical foundations, see [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md). For setup instructions, see [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md).*