# Parametric Equation Visualization Prototype

## Overview

This prototype demonstrates a parametric equation visualization system with gesture-based control using twelve-tone scale mathematics. The system renders complex parametric equations of the form:

**z(θ) = r₁e^(i(ω₁θ+φ₁)) + r₂e^(i(ω₂θ+φ₂))**

## Files Created

1. **`parametric_equation_renderer.py`** - Core rendering engine
2. **`gesture_radius_mapper.py`** - Twelve-tone gesture mapping system  
3. **`main_prototype.py`** - Interactive demonstration and testing

## Key Features

### Mathematical Implementation
- Correct complex exponential parametric equations
- Real-time trajectory generation with configurable parameters
- Rotating rod visualization showing the geometric construction
- Smooth parameter transitions with temporal smoothing

### Twelve-Tone Scale Mapping
- Maps finger counts (0-5) to radius/frequency using musical twelve-tone equal temperament
- Formula: `r = r_max * 2^((finger_count - 5) * 2/12)`
- Creates musically harmonious parameter relationships
- Each finger count represents 2 semitones (whole tone intervals)

### Visualization Features
- Real-time animated parametric curves
- Accumulated trajectory trails
- Rotating rod display showing equation components
- Interactive parameter adjustment
- Multiple visualization modes

## Usage

### Interactive Mode (Default)
```bash
python3 main_prototype.py
```

**Keyboard Controls:**
- `1-5`: Set left hand finger count
- `Shift+1-5`: Set right hand finger count  
- `WASD`: Move left hand position
- `Arrow keys`: Move right hand position
- `Space`: Reset to no hands
- `R`: Reset animation
- `+/-`: Adjust animation speed
- `P`: Pause/unpause
- `H`: Toggle help display
- `ESC/Q`: Quit

### Demonstration Modes

**Show Twelve-Tone Mapping:**
```bash
python3 main_prototype.py mapping
```

**Automated Demo Sequence:**
```bash
python3 main_prototype.py auto
```

## Mathematical Details

### Parametric Equation Components

The equation consists of two rotating complex vectors:
- **First component**: `r₁e^(i(ω₁θ+φ₁))` - Primary rotating rod
- **Second component**: `r₂e^(i(ω₂θ+φ₂))` - Secondary rotating rod attached to first

### Twelve-Tone Scale Mapping

| Fingers | Radius | Frequency | Ratio | Semitones | Musical Note |
|---------|--------|-----------|-------|-----------|--------------|
| 0       | 1.122  | 0.749     | 0.561 | -10       | B            |
| 1       | 1.260  | 0.794     | 0.630 | -8        | C#           |
| 2       | 1.414  | 0.841     | 0.707 | -6        | D#           |
| 3       | 1.587  | 0.891     | 0.794 | -4        | F            |
| 4       | 1.782  | 0.944     | 0.891 | -2        | G            |
| 5       | 2.000  | 1.000     | 1.000 | +0        | A (reference)|

### Control Mapping Schemes

**Single Hand Control:**
- Hand finger count controls both r₁ and r₂ (with offset)
- Hand position controls phase parameters φ₁ and φ₂
- Frequency parameters derived from finger count

**Dual Hand Control:**
- Left hand: Controls r₁, ω₁, φ₁
- Right hand: Controls r₂, ω₂, φ₂
- Independent parameter control for complex interactions

## Architecture for Gesture Integration

The prototype includes placeholder methods and architecture for integrating real hand gesture detection:

### GestureToRadiusMapper Methods
```python
def update_hand_states(left_hand_gesture, right_hand_gesture, 
                      left_hand_position, right_hand_position)
def get_parameters()  # Returns smoothed parameters
def set_smoothing_factor(factor)  # Adjust responsiveness
```

### Integration Points
- Replace keyboard input with MediaPipe hand detection
- Connect to TouchDesigner via OSC or direct Python integration
- Extend to multi-user or advanced gesture recognition

## Performance Notes

- Default settings optimized for smooth real-time performance
- Adjustable trajectory resolution (default: 1000 points)
- Configurable trail length (default: 200 points)
- Smooth parameter transitions prevent jarring changes

## Dependencies

- Python 3.7+
- NumPy (mathematical computations)
- Matplotlib (visualization and animation)

## Testing

The prototype includes comprehensive testing:
```bash
python3 -c "
from parametric_equation_renderer import ParametricEquationRenderer
from gesture_radius_mapper import GestureToRadiusMapper
# Basic functionality verification
"
```

## Next Steps for TouchDesigner Integration

1. **Real Gesture Input**: Replace keyboard simulation with MediaPipe
2. **TouchDesigner Bridge**: Create data exchange mechanism (OSC/TCP)
3. **GPU Optimization**: Leverage TouchDesigner's GPU particle systems
4. **Advanced Features**: Add multi-user support, preset management
5. **Performance Tuning**: Optimize for real-time interaction rates