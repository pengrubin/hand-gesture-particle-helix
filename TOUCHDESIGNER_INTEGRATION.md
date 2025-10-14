# 🎭 TouchDesigner Integration Guide

Professional visual performance setup for real-time parametric equation visualization with gesture control.

## 🏗️ Network Architecture Overview

### Core System Design

```
[Camera Input] → [Gesture Detection] → [Parameter Bridge] → [Parametric Renderer] → [GPU Output]
      ↓               ↓                      ↓                    ↓
   [Video IN]    [Python DAT]         [Table DAT]        [Geometry COMP]
```

### Essential Operators Network

```
Main Network:
├── videoin1 (Video In TOP) - Camera input
├── python1 (DAT) - Hand gesture detector module  
├── python2 (DAT) - Gesture-parametric bridge
├── table1 (Table DAT) - Parameter storage
├── geometry1 (Geometry COMP) - Parametric curve generator
├── render1 (Render TOP) - Final output rendering
└── out1 (Out TOP) - Output to projection/display
```

## 📋 Setup Checklist

### Phase 1: Basic Network Setup (10 minutes)

✅ **Create Core Operators**
```python
# Essential operators to create:
videoin1        # Video In TOP (camera)
python1         # DAT (gesture detector)
python2         # DAT (parametric bridge)  
table1          # Table DAT (parameters)
geometry1       # Geometry COMP (renderer)
render1         # Render TOP (output)
```

✅ **Import Python Modules**
- Place `hand_gesture_detector.py` in TouchDesigner project folder
- Place `gesture_parametric_bridge.py` in project folder
- Place `parametric_equation_renderer.py` in project folder

✅ **Configure Camera Input**
```python
# videoin1 parameters:
Device Index: 0 (or your camera index)
Resolution: 640x480  
Format: YUV422
Active: On
```

### Phase 2: Python Integration (15 minutes)

✅ **Setup Gesture Detector (python1)**
```python
# python1 DAT - Gesture Detection Module
import sys
sys.path.append(project.folder)

from hand_gesture_detector import HandGestureDetector

# Initialize detector
if not hasattr(me, 'detector'):
    me.detector = HandGestureDetector()
    me.detector.start_camera(0)

def onFrameEnd():
    """Called every frame - update gesture data"""
    gesture_data = me.detector.get_gesture_data()
    
    # Store in table1 for other operators
    table = op('table1')
    if gesture_data:
        table.clear()
        table.appendRow(['left_hand_detected', gesture_data.get('left_hand', {}).get('detected', False)])
        table.appendRow(['right_hand_detected', gesture_data.get('right_hand', {}).get('detected', False)])
        table.appendRow(['left_fingers', gesture_data.get('left_hand', {}).get('finger_count', 0)])
        table.appendRow(['right_fingers', gesture_data.get('right_hand', {}).get('finger_count', 0)])
        table.appendRow(['left_x', gesture_data.get('left_hand', {}).get('position', [0,0])[0]])
        table.appendRow(['left_y', gesture_data.get('left_hand', {}).get('position', [0,0])[1]])
        table.appendRow(['right_x', gesture_data.get('right_hand', {}).get('position', [0,0])[0]])
        table.appendRow(['right_y', gesture_data.get('right_hand', {}).get('position', [0,0])[1]])
```

✅ **Setup Parameter Bridge (python2)**
```python
# python2 DAT - Parametric Bridge Module
from gesture_parametric_bridge import GestureParametricBridge
from parametric_equation_renderer import ParametricEquationRenderer

# Initialize bridge
if not hasattr(me, 'bridge'):
    me.bridge = GestureParametricBridge(r_max=2.0, smoothing_factor=0.85)

def onFrameEnd():
    """Convert gestures to parametric parameters"""
    table = op('table1')
    
    # Read gesture data from table
    gesture_data = {}
    for row in table.rows():
        if len(row) >= 2:
            key = row[0].val
            value = row[1].val
            gesture_data[key] = value
    
    # Update bridge with gesture data
    if gesture_data:
        parameters = me.bridge.update_parameters(gesture_data)
        
        # Send to geometry component
        geo = op('geometry1')
        geo.par.r1 = parameters['r1']
        geo.par.r2 = parameters['r2'] 
        geo.par.w1 = parameters['w1']
        geo.par.w2 = parameters['w2']
        geo.par.p1 = parameters['p1']
        geo.par.p2 = parameters['p2']
```

### Phase 3: Geometry Generation (10 minutes)

✅ **Configure Geometry Component (geometry1)**
```python
# Inside geometry1/table1 - SOP Network
# Create: curve1 (Curve SOP)

# curve1 parameters:
Curve Type: NURBS
Method: CV
Coordinates: 3D
```

✅ **Parametric Curve Generator (geometry1/python1)**
```python
# geometry1/python1 DAT - Curve Point Generator
import numpy as np

def onFrameEnd():
    """Generate parametric curve points"""
    # Get parameters from parent
    r1 = parent().par.r1.eval()
    r2 = parent().par.r2.eval()
    w1 = parent().par.w1.eval()
    w2 = parent().par.w2.eval()
    p1 = parent().par.p1.eval()
    p2 = parent().par.p2.eval()
    
    # Generate curve points
    points = []
    num_points = 200
    max_theta = 8 * np.pi
    
    for i in range(num_points):
        theta = i * max_theta / num_points
        
        # Parametric equation: z(θ) = r1*e^(i*(w1*θ+p1)) + r2*e^(i*(w2*θ+p2))
        z1 = r1 * np.exp(1j * (w1 * theta + p1))
        z2 = r2 * np.exp(1j * (w2 * theta + p2))
        z = z1 + z2
        
        x = z.real
        y = z.imag
        z_coord = theta / max_theta * 2 - 1  # Map to [-1, 1]
        
        points.append([x, y, z_coord])
    
    # Update curve SOP
    curve = op('curve1')
    curve.clear()
    for i, point in enumerate(points):
        curve.appendCV(point)
```

## 🖥️ macOS GPU Optimization

### Metal Renderer Settings

```python
# Preferences > Rendering
Renderer: Metal (macOS optimized)
VSync: On
Multi-threading: On
GPU Memory Pool: 1024 MB (adjust for system)
```

### Apple Silicon Optimization

```python
# For M1/M2/M3 Macs - Optimal settings:
PERFORMANCE_SETTINGS = {
    'max_particles': 1000,        # Conservative for integrated GPU
    'render_resolution': '1920x1080',  # 4K may cause slowdown
    'fps_target': 60,             # Metal handles 60fps well
    'texture_compression': True,   # Save GPU memory
    'multisampling': 4            # Good quality/performance balance
}
```

### Intel Mac Optimization

```python
# For Intel Macs - Stability focused:
PERFORMANCE_SETTINGS = {
    'max_particles': 800,         # More conservative
    'render_resolution': '1280x720',   # Lower resolution for stability  
    'fps_target': 30,             # More achievable target
    'texture_compression': True,
    'multisampling': 2            # Lower AA for performance
}
```

## 🔧 Key Operator Configurations

### Video In TOP Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Device** | 0 (auto-detect) | Camera selection |
| **Resolution** | 640x480 | Optimal for MediaPipe |
| **Frame Rate** | 30 FPS | Balance quality/performance |
| **Format** | YUV422 | Efficient color format |
| **Deinterlace** | Off | Progressive scan |

### Python DAT Best Practices

```python
# Performance optimization for Python DATs
class OptimizedGestureDetector:
    def __init__(self):
        self.frame_skip = 2  # Process every 2nd frame
        self.current_frame = 0
        
    def onFrameEnd(self):
        self.current_frame += 1
        if self.current_frame % self.frame_skip == 0:
            self.process_frame()  # Only process selected frames
```

### Geometry Component Settings

```python
# geometry1 parameters:
Render Flag: On
Display Flag: On
Bypass: Off
Viewer Active: On  # For debugging
Clone: Any non-zero value
```

## 🎨 Visual Output Configuration

### Render TOP Setup (render1)

```python
# render1 parameters:
Resolution: 1920x1080 (or target display)
Pixel Format: RGBA8
Multi-sample: 4x (quality/performance balance)
Render Mode: Real-time
Clear Color: [0.05, 0.05, 0.1, 1.0]  # Dark blue background
```

### Material and Lighting

```python
# Create material1 (Material COMP)
# Recommended settings:
Diffuse Color: Gesture-controlled (via parameters)
Specular: 0.3
Roughness: 0.4  
Emission: 0.1 (subtle glow)

# Light setup:
light1 (Light COMP):
    Type: Point
    Position: [2, 2, 2]
    Color: [1, 1, 1]
    Intensity: 1.0
```

## 🔄 Real-time Parameter Control

### Parameter Binding System

```python
# Create parameter1 (Parameter COMP) for real-time control
# Bind to external control surfaces or MIDI

# Example bindings:
r1 → MIDI CC 1 (left hand radius)
r2 → MIDI CC 2 (right hand radius)  
w1 → MIDI CC 3 (left frequency)
w2 → MIDI CC 4 (right frequency)
p1 → MIDI CC 5 (left phase)
p2 → MIDI CC 6 (right phase)
```

### Custom Parameters Window

```python
# In geometry1 - add custom parameters:
# Right-click > Customize Component > Parameters

# Add:
Page: Parametric
    r1: Float, 0-2, default 1.0
    r2: Float, 0-2, default 0.5
    w1: Float, 0.5-5, default 1.0  
    w2: Float, 0.5-5, default 2.0
    p1: Float, 0-6.28, default 0.0
    p2: Float, 0-6.28, default 0.0
```

## 🚀 Performance Optimization

### Frame Rate Optimization

```python
# Performance monitoring script (execute1 DAT)
def onFrameEnd():
    fps = app.fps
    if fps < 30:  # Automatic quality adjustment
        # Reduce particle count
        geo = op('geometry1')
        current_points = geo.par.numpoints.eval()
        geo.par.numpoints = max(100, current_points * 0.9)
    elif fps > 50:  # Can increase quality
        current_points = geo.par.numpoints.eval() 
        geo.par.numpoints = min(500, current_points * 1.1)
```

### Memory Management

```python
# Memory optimization techniques:
class MemoryOptimizer:
    def __init__(self):
        self.cleanup_frequency = 300  # frames
        self.frame_count = 0
    
    def onFrameEnd(self):
        self.frame_count += 1
        if self.frame_count % self.cleanup_frequency == 0:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear unused textures
            for op in ops('*'):
                if hasattr(op, 'clearCache'):
                    op.clearCache()
```

## 🎭 Advanced Integration Features

### Multi-screen Output

```python
# For multiple displays/projectors:
# Create multiple render1, render2, render3 TOPs
# Each with different view angles

# render1 (center screen):
camera.translate = [0, 0, 0]
camera.rotate = [0, 0, 0]

# render2 (left screen):  
camera.translate = [-2, 0, 0]
camera.rotate = [0, 30, 0]

# render3 (right screen):
camera.translate = [2, 0, 0] 
camera.rotate = [0, -30, 0]
```

### OSC Integration

```python
# For external control via OSC
# Create oscin1 (OSC In DAT)

def onReceiveOSC(address, args):
    """Handle OSC messages for parameter control"""
    if address == '/r1':
        op('geometry1').par.r1 = args[0]
    elif address == '/r2':
        op('geometry1').par.r2 = args[0]
    # Add more parameter mappings...
```

## 🐛 Common Troubleshooting

### Issue: Low Frame Rate

**Symptoms**: FPS < 20, stuttering animation
**Solutions**:
1. Reduce geometry resolution (fewer curve points)
2. Lower render resolution
3. Disable unnecessary visual effects
4. Check Python DAT optimization

### Issue: Hand Detection Not Working

**Symptoms**: No gesture data in table1
**Solutions**:
1. Check camera permissions in macOS System Preferences
2. Verify camera index in videoin1 
3. Ensure adequate lighting
4. Check MediaPipe installation

### Issue: Parameter Jitter

**Symptoms**: Unstable, jerky parameter changes
**Solutions**:
```python
# Increase smoothing factor in gesture_parametric_bridge
bridge.smoothing_factor = 0.95  # Higher = more stable

# Add additional smoothing in TouchDesigner
def smooth_parameter(current, target, factor=0.85):
    return current * factor + target * (1.0 - factor)
```

### Issue: GPU Memory Errors

**Symptoms**: Rendering artifacts, crashes
**Solutions**:
1. Reduce max_particles setting
2. Lower render resolution
3. Enable texture compression
4. Restart TouchDesigner periodically

## 🔧 Development Workflow

### Recommended Development Process

1. **Start Simple**: Basic curve with fixed parameters
2. **Add Gesture Input**: Connect gesture detector
3. **Test Parameter Mapping**: Verify smooth control
4. **Optimize Performance**: Adjust for target frame rate
5. **Add Visual Polish**: Materials, lighting, effects
6. **Production Testing**: Extended runtime stability

### Version Control Tips

```bash
# Save TouchDesigner files with meaningful names:
parametric_gesture_v1.0.toe     # Basic functionality
parametric_gesture_v1.1.toe     # Added smoothing  
parametric_gesture_v1.2.toe     # Performance optimized
parametric_gesture_production.toe # Final version
```

### Backup Strategy

- **Daily**: Save .toe file with date
- **Weekly**: Export Python DAT code to .py files  
- **Monthly**: Full project archive with media files

---

*For mathematical details and parameter explanations, see [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md).*  
*For Python API details, see [API_REFERENCE.md](API_REFERENCE.md).*