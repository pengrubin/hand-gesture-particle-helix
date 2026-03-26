# 📊 Parameter Reference Guide

Complete mathematical and technical reference for the parametric equation gesture control system.

## 🧮 Mathematical Foundation

### Core Parametric Equation

The system renders complex parametric equations of the form:

```
z(θ) = r1 × e^(i×(w1×θ+p1)) + r2 × e^(i×(w2×θ+p2))
```

Where:
- **θ (theta)**: Parameter variable (0 to 8π typically)
- **i**: Imaginary unit (√-1)
- **z(θ)**: Complex output representing (x,y) coordinates

### Parameter Breakdown

| Parameter | Symbol | Range | Function | Gesture Control |
|-----------|---------|--------|----------|-----------------|
| **Primary Radius** | r1 | 0.0 - 2.0 | Main circle size | Left hand finger count |
| **Secondary Radius** | r2 | 0.0 - 2.0 | Secondary circle size | Right hand finger count |
| **Primary Frequency** | w1 | 0.5 - 5.0 | Main rotation speed | Left hand position |
| **Secondary Frequency** | w2 | 0.5 - 5.0 | Secondary rotation speed | Right hand position |
| **Primary Phase** | p1 | 0 - 2π | Main phase offset | Left hand vertical |
| **Secondary Phase** | p2 | 0 - 2π | Secondary phase offset | Right hand vertical |

## 🎵 Twelve-Tone Gesture Mapping System

### Mathematical Formula
The system uses twelve-tone equal temperament mathematics for smooth parameter transitions:

```
r = r_max × 2^((finger_count - 5) × 2/12)
```

### Finger Count to Radius Mapping

| Fingers | Ratio | r1/r2 Value | Musical Interval | Visual Effect |
|---------|-------|-------------|------------------|---------------|
| **0 (Fist)** | 0.630 | 1.26 | Minor 7th down | Tight spiral |
| **1** | 0.707 | 1.41 | Perfect 5th down | Small loop |
| **2** | 0.794 | 1.59 | Major 3rd down | Medium arc |
| **3** | 0.891 | 1.78 | Major 2nd down | Large curve |
| **4** | 1.000 | 2.00 | Perfect unison | Reference size |
| **5 (Open)** | 1.122 | 2.24 | Major 2nd up | Extended reach |

### Hand State Recognition

```python
class HandState(Enum):
    NO_HAND = "no_hand"           # No parameters change
    FIST = "fist"                 # r = 1.26
    ONE_FINGER = "one_finger"     # r = 1.41  
    TWO_FINGERS = "two_fingers"   # r = 1.59
    THREE_FINGERS = "three_fingers" # r = 1.78
    FOUR_FINGERS = "four_fingers" # r = 2.00
    OPEN_HAND = "open_hand"       # r = 2.24
```

## 🌀 Spiral Pattern Types

### Basic Patterns (9 types)

| Pattern Name | Equation Type | Key Parameters | Visual Description |
|--------------|---------------|----------------|-------------------|
| **sine_wave** | y = A×sin(fx+t) | amplitude, frequency | Classic sine wave |
| **cosine_wave** | y = A×cos(fx+t) | amplitude, frequency | Cosine variation |
| **double_wave** | Combined sine | dual frequency | Layered waves |
| **spiral_line** | Polar spiral | radius growth | Outward spiral |
| **zigzag_line** | Triangular wave | sharp transitions | Sawtooth pattern |
| **heart_curve** | Parametric heart | scale factor | Heart shape |
| **infinity_curve** | Figure-8 | aspect ratio | Infinity symbol |
| **helix_3d** | 3D spiral | pitch, radius | Cylindrical helix |
| **multiple_lines** | Parallel waves | line count, spacing | Multi-strand |

### Advanced Helix Structures (9 types)

| Helix Type | Structure | Parameters | Mathematical Basis |
|------------|-----------|------------|-------------------|
| **double_helix** | DNA-like | radius, pitch, phase | Two intertwined spirals |
| **triple_helix** | Three-strand | 120° phase offset | Triple symmetry |
| **dna_structure** | Realistic DNA | base pair bridges | Biological accuracy |
| **twisted_ribbon** | Ribbon helix | twist rate, width | Surface deformation |
| **braided_lines** | Woven pattern | strand count | Complex weaving |
| **spiral_tower** | Layered spirals | layer spacing | Vertical progression |
| **coil_spring** | Spring coil | compression ratio | Mechanical spring |
| **tornado_helix** | Funnel shape | radius variation | Weather pattern |
| **galaxy_spiral** | Logarithmic arms | arm count, growth | Astronomical spiral |

## 🎮 Gesture Control Mapping

### Hand Assignment Modes

```python
class HandAssignment(Enum):
    LEFT_R1_RIGHT_R2 = "left_r1_right_r2"    # Standard mapping
    RIGHT_R1_LEFT_R2 = "right_r1_left_r2"    # Reversed mapping  
    DOMINANT_PRIMARY = "dominant_primary"      # Auto-detect dominant hand
```

### Position-Based Parameters

| Hand Position | Parameter | Range | Effect |
|---------------|-----------|-------|---------|
| **Left X** | w1 (frequency) | 0.5 - 5.0 | Main rotation speed |
| **Left Y** | p1 (phase) | 0 - 2π | Main phase shift |
| **Right X** | w2 (frequency) | 0.5 - 5.0 | Secondary rotation |
| **Right Y** | p2 (phase) | 0 - 2π | Secondary phase |
| **Hand Distance** | Scale multiplier | 0.5 - 2.0 | Overall size |
| **Gesture Strength** | Animation speed | 0.1 - 3.0 | Temporal scaling |

### Coordinate Normalization

```python
def normalize_hand_position(hand_center):
    """Convert MediaPipe coordinates to normalized (-1, 1) range"""
    x = (hand_center[0] - 0.5) * 2.0  # [0,1] -> [-1,1]
    y = (hand_center[1] - 0.5) * 2.0  # [0,1] -> [-1,1] 
    return (x, y)
```

## 🎨 Visual Effect Parameters

### Particle System Properties

| Property | Range | Default | Gesture Control |
|----------|--------|---------|-----------------|
| **max_particles** | 300-2000 | 1500 | Keyboard 1-5 |
| **emission_rate** | 50-200 | 100 | Gesture strength |
| **particle_life** | 2.0-8.0 | 5.0 | Pattern complexity |
| **velocity_scale** | 0.5-2.0 | 1.0 | Hand movement speed |
| **turbulence** | 0.0-3.0 | Variable | Gesture type |
| **size_scale** | 0.5-2.5 | 1.0 | Audio intensity |

### Color System

| Parameter | Control | Range | Effect |
|-----------|---------|-------|---------|
| **Hue** | Hand horizontal | 0-360° | Color wheel |
| **Saturation** | Gesture strength | 0.0-1.0 | Color intensity |
| **Brightness** | Hand vertical | 0.3-1.0 | Luminosity |
| **Alpha** | Hand openness | 0.1-1.0 | Transparency |

## 🔧 Performance Parameters

### System Optimization Settings

```python
# Performance configuration
PERFORMANCE_SETTINGS = {
    'max_particles': 1500,        # GPU memory limit
    'fps_target': 60,             # Target frame rate
    'smoothing_factor': 0.85,     # Parameter smoothing (0-1)
    'trail_length': 500,          # Trajectory history points
    'update_frequency': 30,       # Parameter update Hz
    'render_quality': 'high'      # 'low', 'medium', 'high'
}
```

### Camera Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| **Resolution** | 640×480 | Balance accuracy/speed |
| **FPS** | 30 | MediaPipe optimal |
| **Min Detection Confidence** | 0.7 | Reduce false positives |
| **Min Tracking Confidence** | 0.5 | Smooth tracking |

## 🎵 Audio Integration Parameters

### Spectrum Analysis

```python
# Audio frequency mapping
AUDIO_MAPPING = {
    'sample_rate': 22050,         # Audio sample rate
    'buffer_size': 512,           # FFT buffer size
    'frequency_bins': 256,        # Spectrum resolution
    'pitch_detection_threshold': 0.1,  # Minimum amplitude
    'note_smoothing': 0.7         # Note transition smoothing
}
```

### Track Control Mapping

| Gesture | Track | Frequency Range | Effect |
|---------|-------|----------------|---------|
| **1 Finger** | Violin | 196-1568 Hz | High harmonics |
| **2 Fingers** | Lute | 82-659 Hz | Mid-range pluck |
| **3 Fingers** | Organ | 65-523 Hz | Low fundamentals |
| **Open Hand** | All tracks | Full spectrum | Complete mix |

## 🧪 Debugging Parameters

### Diagnostic Settings

```python
DEBUG_CONFIG = {
    'show_hand_landmarks': False,    # MediaPipe landmarks
    'show_parameter_values': True,   # Real-time parameters  
    'show_fps_counter': True,        # Performance monitoring
    'log_gesture_changes': False,    # Gesture transition log
    'export_trajectory': False       # Save trajectory data
}
```

### Performance Monitoring

| Metric | Good | Acceptable | Poor | Action |
|--------|------|------------|------|---------|
| **FPS** | >45 | 30-45 | <30 | Reduce particles |
| **CPU %** | <30% | 30-60% | >60% | Optimize settings |
| **GPU %** | <70% | 70-85% | >85% | Lower quality |
| **Latency** | <50ms | 50-100ms | >100ms | Check camera |

## 🔄 Parameter Smoothing

### Smoothing Algorithm

```python
def smooth_parameter(current, target, factor=0.85):
    """Exponential moving average smoothing"""
    return current * factor + target * (1.0 - factor)
```

### Smoothing Factors by Parameter

| Parameter | Factor | Reason |
|-----------|--------|--------|
| **r1, r2** | 0.85 | Prevent jarring size changes |
| **w1, w2** | 0.75 | Allow responsive frequency control |
| **p1, p2** | 0.90 | Smooth phase transitions |
| **Color** | 0.80 | Natural color blending |
| **Position** | 0.70 | Responsive but stable |

## 📐 Advanced Mathematical Relationships

### Frequency Ratios

Common harmonic relationships for pleasing visual patterns:

```python
HARMONIC_RATIOS = {
    'octave': (w1, 2*w1),        # 1:2 ratio
    'perfect_fifth': (w1, 1.5*w1), # 2:3 ratio  
    'major_third': (w1, 1.25*w1),  # 4:5 ratio
    'golden_ratio': (w1, 1.618*w1), # φ ratio
}
```

### Lissajous Patterns

Special parameter combinations that create classic Lissajous figures:

| Pattern | w1 | w2 | p1 | p2 | Description |
|---------|----|----|----|----|-------------|
| **Circle** | 1 | 1 | 0 | π/2 | Perfect circle |
| **Line** | 1 | 1 | 0 | 0 | Diagonal line |
| **Figure-8** | 1 | 2 | 0 | 0 | Horizontal eight |
| **Rose** | 3 | 2 | 0 | 0 | Three-petal rose |
| **Star** | 5 | 4 | 0 | π/4 | Five-pointed star |

---

*This reference covers all mathematical and technical parameters. For implementation details, see [API_REFERENCE.md](API_REFERENCE.md).*