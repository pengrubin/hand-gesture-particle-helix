# TouchDesigner Integration Guide

## Overview

This comprehensive guide provides step-by-step instructions for integrating the Hand Gesture Parametric Control System with TouchDesigner. The integration supports both parametric equation visualization and the existing particle system, with advanced GPU optimization for macOS and Windows.

## Table of Contents

1. [Prerequisites and Setup](#prerequisites-and-setup)
2. [TouchDesigner Network Architecture](#touchdesigner-network-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [GPU Optimization Configuration](#gpu-optimization-configuration)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Features](#advanced-features)

---

## Prerequisites and Setup

### System Requirements

**TouchDesigner Version**: 2022.25370 or newer  
**Python Version**: 3.9+ (TouchDesigner embedded)  
**Operating System**: macOS 12+ (Metal) or Windows 10+ (DirectX 11)

### Required Python Modules

Install these modules in TouchDesigner's Python environment:

```python
# In TouchDesigner's Textport, run:
import subprocess
import sys

# Install required packages
packages = [
    'opencv-python==4.8.1.78',
    'mediapipe==0.10.8',
    'numpy>=1.21.0'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
```

### Project Files Setup

1. **Create Project Directory**:
   ```
   HandGestureParametric/
   ├── python/
   │   ├── hand_gesture_detector.py
   │   ├── gesture_radius_mapper.py
   │   ├── parametric_equation_renderer.py
   │   ├── gesture_parametric_bridge.py
   │   └── td_gpu_optimization.py
   ├── touchdesigner/
   │   └── HandGestureParametric.toe
   └── docs/
   ```

2. **TouchDesigner Path Configuration**:
   - In TouchDesigner: **Edit → Preferences → Python**
   - Add project `python/` directory to Python Path

---

## TouchDesigner Network Architecture

### Network Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Camera      │───▶│ Gesture     │───▶│ Parameter   │───▶│ Dual        │
│ Input       │    │ Detection   │    │ Processing  │    │ Rendering   │
│             │    │             │    │             │    │             │
│ videoin1    │    │ Python DATs │    │ Table DATs  │    │ SOP/COMP    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Core Node Structure

#### 1. Input Layer
- **videoin1** (Video In TOP) - Camera input
- **camera_settings** (Parameter COMP) - Camera configuration

#### 2. Processing Layer
- **hand_detector** (Text DAT) - Hand gesture detection
- **parameter_mapper** (Text DAT) - Twelve-tone parameter mapping
- **bridge_controller** (Text DAT) - System coordination
- **frame_processor** (Execute DAT) - Main processing callback

#### 3. Data Layer
- **gesture_data** (Table DAT) - Raw gesture information
- **parametric_params** (Table DAT) - Parametric equation parameters
- **particle_params** (Table DAT) - Particle system parameters
- **trajectory_points** (Table DAT) - Real-time curve points

#### 4. Rendering Layer
- **parametric_geometry** (Geometry COMP) - Parametric curve rendering
- **particle_system** (Instance2 COMP) - Particle effects
- **final_composite** (Composite TOP) - Combined output

---

## Step-by-Step Implementation

### Step 1: Create Base Network

1. **Start New TouchDesigner Project**
   - File → New
   - Save as `HandGestureParametric.toe`

2. **Add Camera Input**
   ```python
   # Create Video In TOP
   video_in = root.create(videodeviceinTOP, 'videoin1')
   video_in.par.device = 0  # Default camera
   video_in.par.resolution = '640x480'
   video_in.par.format = 'RGB888'
   ```

3. **Configure Camera Settings**
   ```python
   # Optimal settings for gesture detection
   video_in.par.brightness = 0.0
   video_in.par.contrast = 1.0
   video_in.par.saturation = 1.0
   video_in.par.exposure = -5  # Auto-exposure
   ```

### Step 2: Python Module Integration

1. **Create Hand Detector Text DAT**
   ```python
   # Create Text DAT
   hand_detector = root.create(textDAT, 'hand_detector')
   hand_detector.par.file = 'python/hand_gesture_detector.py'
   
   # Initialize detector in onStart callback
   def onStart():
       if not hasattr(me.parent(), 'detector'):
           me.parent().detector = me.module.HandGestureDetector()
   ```

2. **Create Parameter Mapper Text DAT**
   ```python
   parameter_mapper = root.create(textDAT, 'parameter_mapper')
   parameter_mapper.par.file = 'python/gesture_radius_mapper.py'
   
   # Initialize mapper
   def onStart():
       if not hasattr(me.parent(), 'mapper'):
           me.parent().mapper = me.module.GestureToRadiusMapper(r_max=2.0)
   ```

3. **Create Bridge Controller Text DAT**
   ```python
   bridge_controller = root.create(textDAT, 'bridge_controller')
   bridge_controller.par.file = 'python/gesture_parametric_bridge.py'
   
   # Initialize bridge
   def onStart():
       if not hasattr(me.parent(), 'bridge'):
           me.parent().bridge = me.module.GestureParametricBridge()
   ```

### Step 3: Data Storage Tables

1. **Create Gesture Data Table**
   ```python
   gesture_data = root.create(tableDAT, 'gesture_data')
   
   # Initialize table structure
   def initializeGestureTable():
       table = op('gesture_data')
       table.clear()
       
       # Header row
       table.appendRow(['frame', 'hands_detected', 'hand_id', 
                       'label', 'gesture_number', 'openness', 
                       'center_x', 'center_y', 'timestamp'])
   ```

2. **Create Parametric Parameters Table**
   ```python
   parametric_params = root.create(tableDAT, 'parametric_params')
   
   # Initialize parametric table
   def initializeParametricTable():
       table = op('parametric_params')
       table.clear()
       
       # Header and initial values
       table.appendRow(['parameter', 'value', 'target', 'smoothing'])
       table.appendRow(['r1', '1.0', '1.0', '0.85'])
       table.appendRow(['r2', '0.5', '0.5', '0.85'])
       table.appendRow(['w1', '1.0', '1.0', '0.80'])
       table.appendRow(['w2', '2.0', '2.0', '0.80'])
       table.appendRow(['p1', '0.0', '0.0', '0.90'])
       table.appendRow(['p2', '0.0', '0.0', '0.90'])
       table.appendRow(['theta', '0.0', '0.0', '1.00'])
   ```

3. **Create Trajectory Points Table**
   ```python
   trajectory_points = root.create(tableDAT, 'trajectory_points')
   
   # Initialize trajectory table
   def initializeTrajectoryTable():
       table = op('trajectory_points')
       table.clear()
       
       # Header row
       table.appendRow(['point_id', 'x', 'y', 'z', 'timestamp', 'alpha'])
   ```

### Step 4: Main Processing Execute DAT

1. **Create Frame Processor Execute DAT**
   ```python
   frame_processor = root.create(executeDAT, 'frame_processor')
   
   # Set callback to onFrameEnd
   frame_processor.par.frameendcallback = True
   ```

2. **Implement Core Processing Logic**
   ```python
   # In frame_processor Execute DAT
   
   def onFrameEnd(frame):
       try:
           # Get camera frame data
           camera_frame = op('videoin1').numpyArray()
           
           # Process gestures
           bridge = parent().bridge
           processed_frame, parameters = bridge.process_frame(camera_frame)
           
           # Update parameter table
           updateParametricTable(parameters)
           
           # Update trajectory points
           updateTrajectoryPoints(parameters)
           
           # Update performance metrics
           updatePerformanceMetrics(frame)
           
       except Exception as e:
           print(f"Frame processing error: {e}")
   
   def updateParametricTable(params):
       table = op('parametric_params')
       
       # Update parameter values
       for i, (param_name, value) in enumerate(params.items(), 1):
           if i < table.numRows:
               table[i, 'value'] = str(value)
   
   def updateTrajectoryPoints(params):
       table = op('trajectory_points')
       
       # Calculate current point
       r1, r2 = params['r1'], params['r2']
       w1, w2 = params['w1'], params['w2']
       p1, p2 = params['p1'], params['p2']
       theta = params.get('theta', absTime.frame * 0.05)
       
       # Complex exponential calculation
       import numpy as np
       z1 = r1 * np.exp(1j * (w1 * theta + p1))
       z2 = r2 * np.exp(1j * (w2 * theta + p2))
       z = z1 + z2
       
       # Add current point
       current_time = absTime.seconds
       table.appendRow([
           str(absTime.frame),
           str(z.real),
           str(z.imag),
           '0.0',
           str(current_time),
           '1.0'
       ])
       
       # Maintain trail length
       max_trail_length = 500
       if table.numRows > max_trail_length + 1:  # +1 for header
           table.deleteRow(1)  # Remove oldest point
   
   def updatePerformanceMetrics(frame):
       # Record performance data for optimization
       current_time = absTime.seconds
       frame_time = current_time - getattr(me, 'last_frame_time', current_time)
       me.last_frame_time = current_time
       
       # Update GPU optimizer if available
       if hasattr(parent(), 'gpu_optimizer'):
           parent().gpu_optimizer.record_frame_performance(
               frame_time=frame_time,
               vertex_count=op('trajectory_points').numRows
           )
   ```

### Step 5: Parametric Geometry Rendering

1. **Create Trajectory Generator Add SOP**
   ```python
   trajectory_add = root.create(addSOP, 'trajectory_add')
   
   # Connect to trajectory points table
   trajectory_add.par.points = 'on'
   trajectory_add.par.pointsop = 'trajectory_points'
   ```

2. **Configure Add SOP for Efficiency**
   ```python
   # Optimization settings
   trajectory_add.par.pointattribs = 'P'  # Position only
   trajectory_add.par.addattrib1 = 'alpha float 1.0'  # Alpha channel
   trajectory_add.par.method = 'by_index'
   ```

3. **Create Line SOP for Curve Connection**
   ```python
   trajectory_line = root.create(lineSOP, 'trajectory_line')
   trajectory_line.setInput(0, trajectory_add)
   
   # Line settings
   trajectory_line.par.closed = False
   trajectory_line.par.reverse = False
   ```

4. **Create Geometry COMP for Rendering**
   ```python
   parametric_geometry = root.create(geometryCOMP, 'parametric_geometry')
   parametric_geometry.setInput(0, trajectory_line)
   
   # Material and rendering settings
   mat = parametric_geometry.create(pbrMAT, 'parametric_material')
   mat.par.basecolor = [0.2, 0.6, 1.0]
   mat.par.metallic = 0.0
   mat.par.rough = 0.1
   mat.par.ior = 1.4
   ```

### Step 6: GPU Optimization Integration

1. **Create GPU Optimizer Text DAT**
   ```python
   gpu_optimizer = root.create(textDAT, 'gpu_optimizer')
   gpu_optimizer.par.file = 'python/td_gpu_optimization.py'
   
   # Initialize optimizer in onStart
   def onStart():
       if not hasattr(me.parent(), 'gpu_optimizer'):
           me.parent().gpu_optimizer = me.module.get_gpu_optimizer()
           me.parent().gpu_optimizer.start_optimization_thread()
   ```

2. **Apply SOP Optimizations**
   ```python
   # In frame_processor Execute DAT, add optimization call
   def onFrameEnd(frame):
       # ... existing code ...
       
       # Apply SOP optimizations
       if hasattr(parent(), 'gpu_optimizer'):
           sop_ops = {
               'trajectory_add': op('trajectory_add'),
               'trajectory_line': op('trajectory_line')
           }
           
           point_count = op('trajectory_points').numRows
           optimization_results = parent().gpu_optimizer.optimize_for_touchdesigner_sops(
               sop_ops, point_count, target_fps=30.0
           )
   ```

### Step 7: Performance Monitoring Display

1. **Create Info DAT for Performance Display**
   ```python
   performance_info = root.create(infoDATCHOP, 'performance_info')
   
   # Custom performance info script
   def updatePerformanceInfo():
       info_dat = op('performance_info')
       
       if hasattr(parent(), 'gpu_optimizer'):
           stats = parent().gpu_optimizer.get_optimization_statistics()
           
           # Format performance data
           performance_text = f"""
   Performance Statistics:
   FPS: {stats['average_fps']:.1f}
   Frame Time: {stats['average_frame_time_ms']:.1f}ms
   GPU Memory: {stats['gpu_memory_used_mb']:.1f}MB / {stats['gpu_memory_limit_mb']:.1f}MB
   Active Buffers: {stats['active_buffers']}
   Optimization Level: {stats['optimization_level']}
   
   Gesture Status:
   Hands Detected: {getattr(parent(), 'bridge', {}).get_gesture_info().get('hands_detected', 0)}
   System Paused: {getattr(parent(), 'bridge', {}).get_gesture_info().get('is_paused', False)}
   """
           
           info_dat.text = performance_text
   ```

---

## GPU Optimization Configuration

### Metal API Optimization (macOS)

1. **Enable Metal Compute in TouchDesigner**
   ```python
   # In Preferences → Rendering
   # Set Graphics API to Metal
   # Enable GPU Compute
   ```

2. **Configure Metal Settings**
   ```python
   # In gpu_optimizer Text DAT
   def configureMetal():
       optimizer = parent().gpu_optimizer
       metal_settings = optimizer.create_metal_optimized_settings()
       
       # Apply Metal-specific optimizations
       if hasattr(ui.preferences, 'graphics'):
           ui.preferences.graphics.gpucompute = True
           ui.preferences.graphics.metalcompute = True
   ```

### DirectX Optimization (Windows)

1. **DirectX 11 Configuration**
   ```python
   # In Preferences → Rendering
   # Set Graphics API to DirectX 11
   # Enable Hardware Acceleration
   ```

2. **GPU Memory Management**
   ```python
   def configureDirectX():
       optimizer = parent().gpu_optimizer
       
       # Set DirectX-specific memory limits
       optimizer.max_gpu_memory = 512 * 1024 * 1024  # 512MB
       optimizer.optimization_level = OptimizationLevel.BALANCED
   ```

### Adaptive Quality System

1. **Performance-Based Quality Adjustment**
   ```python
   # In frame_processor Execute DAT
   def adaptQualityBasedOnPerformance():
       if hasattr(parent(), 'gpu_optimizer'):
           optimizer = parent().gpu_optimizer
           stats = optimizer.get_optimization_statistics()
           
           current_fps = stats['average_fps']
           
           # Adjust trajectory resolution based on performance
           if current_fps < 25:
               # Reduce quality for better performance
               op('trajectory_points').par.maxrows = 300
           elif current_fps > 35:
               # Increase quality when performance allows
               op('trajectory_points').par.maxrows = 800
   ```

---

## Performance Tuning

### Optimization Levels

#### Level 1: Maximum Quality
```python
def setMaximumQuality():
    # Trajectory settings
    op('trajectory_points').par.maxrows = 1000
    
    # Material settings
    mat = op('parametric_geometry/parametric_material')
    mat.par.roughsamples = 64
    mat.par.metalsamples = 64
    
    # Rendering settings
    render = op('final_composite')
    render.par.samples = 4
```

#### Level 2: Balanced Performance
```python
def setBalancedPerformance():
    # Trajectory settings
    op('trajectory_points').par.maxrows = 500
    
    # Material settings
    mat = op('parametric_geometry/parametric_material')
    mat.par.roughsamples = 32
    mat.par.metalsamples = 32
    
    # Rendering settings
    render = op('final_composite')
    render.par.samples = 2
```

#### Level 3: High Performance
```python
def setHighPerformance():
    # Trajectory settings
    op('trajectory_points').par.maxrows = 200
    
    # Simplified materials
    mat = op('parametric_geometry/parametric_material')
    mat.par.roughsamples = 16
    mat.par.metalsamples = 16
    
    # Rendering settings
    render = op('final_composite')
    render.par.samples = 1
```

### Memory Optimization

1. **Table DAT Memory Management**
   ```python
   def optimizeTableMemory():
       # Set reasonable limits for all tables
       op('gesture_data').par.maxrows = 100
       op('trajectory_points').par.maxrows = 500
       
       # Enable automatic cleanup
       op('gesture_data').par.trimrows = True
       op('trajectory_points').par.trimrows = True
   ```

2. **Buffer Pool Configuration**
   ```python
   def configureBufferPools():
       if hasattr(parent(), 'gpu_optimizer'):
           optimizer = parent().gpu_optimizer
           
           # Configure memory pools for different buffer types
           optimizer.max_gpu_memory = 256 * 1024 * 1024  # 256MB
           
           # Enable aggressive cleanup
           optimizer.auto_optimization = True
           optimizer.frame_skip_enabled = True
   ```

### Frame Rate Optimization

1. **Dynamic Frame Rate Control**
   ```python
   def adjustFrameRate():
       # Monitor performance and adjust accordingly
       if hasattr(parent(), 'gpu_optimizer'):
           stats = parent().gpu_optimizer.get_optimization_statistics()
           current_fps = stats['average_fps']
           
           if current_fps < 20:
               # Enable frame skipping
               me.par.frameendcallback = False
               run("me.par.frameendcallback = True", delayFrames=2)
   ```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Camera Not Detected
**Symptoms**: Video In TOP shows black screen  
**Solutions**:
```python
# Check camera availability
def diagnoseCameraIssues():
    video_in = op('videoin1')
    
    # Try different device indices
    for device_idx in range(5):
        video_in.par.device = device_idx
        if video_in.width > 0 and video_in.height > 0:
            print(f"Camera found at device index {device_idx}")
            break
    
    # Check camera permissions (macOS)
    import subprocess
    try:
        result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                              capture_output=True, text=True)
        print("Camera hardware info:", result.stdout)
    except:
        print("Could not get camera information")
```

#### Issue 2: Hand Detection Not Working
**Symptoms**: No hands detected despite visible hands in frame  
**Solutions**:
```python
def diagnoseHandDetection():
    # Check if detector is properly initialized
    if not hasattr(parent(), 'detector'):
        op('hand_detector').module.onStart()
    
    # Test detection with static image
    camera_frame = op('videoin1').numpyArray()
    if camera_frame is not None:
        print(f"Camera frame shape: {camera_frame.shape}")
        
        # Process test frame
        detector = parent().detector
        processed = detector.process_frame(camera_frame)
        gesture_data = detector.gesture_data
        
        print(f"Detection result: {gesture_data}")
    else:
        print("No camera frame available")
```

#### Issue 3: Poor Performance
**Symptoms**: Low FPS, stuttering animation  
**Solutions**:
```python
def diagnosePerformance():
    # Check system resources
    import psutil
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_percent}%")
    
    # Check TouchDesigner performance
    print(f"TouchDesigner FPS: {absTime.rate}")
    print(f"Cook Time: {op('/').cookTime}")
    
    # Apply automatic optimizations
    if cpu_percent > 80 or memory_percent > 80:
        setHighPerformance()
        print("Applied high-performance settings")
```

#### Issue 4: Parameter Values Not Updating
**Symptoms**: Gestures detected but parameters don't change  
**Solutions**:
```python
def diagnoseParameterUpdates():
    # Check bridge initialization
    if not hasattr(parent(), 'bridge'):
        print("Bridge not initialized")
        op('bridge_controller').module.onStart()
        return
    
    # Check parameter mapper
    if not hasattr(parent(), 'mapper'):
        print("Mapper not initialized")
        op('parameter_mapper').module.onStart()
        return
    
    # Test parameter calculation
    bridge = parent().bridge
    info = bridge.get_gesture_info()
    print(f"Bridge info: {info}")
    
    # Check table updates
    param_table = op('parametric_params')
    print(f"Parameter table rows: {param_table.numRows}")
    for i in range(1, param_table.numRows):
        param_name = param_table[i, 'parameter'].val
        param_value = param_table[i, 'value'].val
        print(f"{param_name}: {param_value}")
```

### Debug Mode Setup

1. **Enable Debug Logging**
   ```python
   # Create debug_logger Execute DAT
   debug_logger = root.create(executeDAT, 'debug_logger')
   
   def onFrameEnd(frame):
       # Log detailed information every second
       if frame % 30 == 0:  # 30 FPS = 1 second
           logSystemState()
   
   def logSystemState():
       print("=== System State Debug ===")
       
       # Camera state
       video_in = op('videoin1')
       print(f"Camera: {video_in.width}x{video_in.height}")
       
       # Gesture detection
       if hasattr(parent(), 'bridge'):
           info = parent().bridge.get_gesture_info()
           print(f"Hands: {info['hands_detected']}")
           print(f"Gestures: {info['digit_gestures']}")
           print(f"Parameters: {info['current_parameters']}")
       
       # Performance
       if hasattr(parent(), 'gpu_optimizer'):
           stats = parent().gpu_optimizer.get_optimization_statistics()
           print(f"FPS: {stats['average_fps']:.1f}")
           print(f"Memory: {stats['gpu_memory_used_mb']:.1f}MB")
       
       print("========================")
   ```

---

## Advanced Features

### Dual Mode Operation

1. **Particle/Parametric Mode Switching**
   ```python
   # Create mode switcher
   def switchToParametricMode():
       # Hide particle system
       op('particle_system').par.display = False
       
       # Show parametric system
       op('parametric_geometry').par.display = True
       
       # Update processing
       parent().current_mode = 'parametric'
   
   def switchToParticleMode():
       # Show particle system
       op('particle_system').par.display = True
       
       # Hide parametric system
       op('parametric_geometry').par.display = False
       
       # Update processing
       parent().current_mode = 'particle'
   
   def switchToHybridMode():
       # Show both systems
       op('particle_system').par.display = True
       op('parametric_geometry').par.display = True
       
       # Balance performance
       setBalancedPerformance()
       parent().current_mode = 'hybrid'
   ```

### MIDI Integration

1. **MIDI Controller Parameter Control**
   ```python
   # Create MIDI In CHOP
   midi_in = root.create(midiinCHOP, 'midi_in')
   
   # Map MIDI controls to parameters
   def processMIDIInput():
       midi_data = op('midi_in')
       
       # Map CC controllers to parameters
       if midi_data['cc1'] is not None:
           r1_value = float(midi_data['cc1']) / 127.0 * 2.0  # Scale to 0-2
           op('parametric_params')[1, 'target'] = str(r1_value)
       
       if midi_data['cc2'] is not None:
           r2_value = float(midi_data['cc2']) / 127.0 * 2.0
           op('parametric_params')[2, 'target'] = str(r2_value)
   ```

### Audio Reactive Features

1. **Audio Analysis Integration**
   ```python
   # Create Audio File In CHOP
   audio_in = root.create(audiofileinCHOP, 'audio_in')
   
   # Create Audio Analysis
   audio_analysis = root.create(audiospectralCHOP, 'audio_analysis')
   audio_analysis.setInput(0, audio_in)
   
   # Map audio features to parameters
   def processAudioReactive():
       analysis = op('audio_analysis')
       
       # Map bass energy to r1
       bass_energy = analysis['bass_rms']
       if bass_energy:
           r1_audio = float(bass_energy) * 3.0
           # Combine with gesture control
           r1_gesture = float(op('parametric_params')[1, 'value'])
           r1_combined = (r1_gesture + r1_audio) / 2.0
           op('parametric_params')[1, 'target'] = str(r1_combined)
   ```

### Network Synchronization

1. **OSC Communication Setup**
   ```python
   # Create OSC In DAT
   osc_in = root.create(oscInDAT, 'osc_in')
   osc_in.par.port = 8000
   
   # Create OSC Out DAT
   osc_out = root.create(oscOutDAT, 'osc_out')
   osc_out.par.address = '127.0.0.1'
   osc_out.par.port = 8001
   
   # Handle incoming OSC messages
   def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
       if address == '/parameters/r1':
           op('parametric_params')[1, 'target'] = str(args[0])
       elif address == '/parameters/r2':
           op('parametric_params')[2, 'target'] = str(args[0])
   
   # Send parameter updates via OSC
   def sendParameterUpdates():
       params = op('parametric_params')
       for i in range(1, params.numRows):
           param_name = params[i, 'parameter'].val
           param_value = params[i, 'value'].val
           
           osc_address = f'/parameters/{param_name}'
           op('osc_out').sendOSC(osc_address, [float(param_value)])
   ```

---

## Complete Network Template

### Final Network Structure
```
HandGestureParametric/
├── videoin1 (Video In TOP)
├── Python_Scripts/
│   ├── hand_detector (Text DAT)
│   ├── parameter_mapper (Text DAT)
│   ├── bridge_controller (Text DAT)
│   └── gpu_optimizer (Text DAT)
├── Processing/
│   ├── frame_processor (Execute DAT)
│   └── performance_monitor (Execute DAT)
├── Data_Storage/
│   ├── gesture_data (Table DAT)
│   ├── parametric_params (Table DAT)
│   └── trajectory_points (Table DAT)
├── Parametric_Rendering/
│   ├── trajectory_add (Add SOP)
│   ├── trajectory_line (Line SOP)
│   └── parametric_geometry (Geometry COMP)
├── Particle_System/
│   ├── particle_add (Add SOP)
│   └── particle_instances (Instance2 COMP)
└── Output/
    └── final_composite (Composite TOP)
```

### Save Project Template
1. **File → Save As...**
2. **Select "Save External References"**
3. **Include Python scripts in project**
4. **Create `.tox` file for reuse**

---

*This integration guide provides comprehensive instructions for implementing the Hand Gesture Parametric Control System in TouchDesigner. For additional customization and advanced features, refer to the API Reference and Advanced Usage documentation.*