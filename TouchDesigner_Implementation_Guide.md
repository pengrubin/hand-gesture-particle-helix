# TouchDesigner Implementation Guide
## Step-by-Step Setup for Parametric Integration

This guide provides detailed instructions for implementing the parametric equation integration in TouchDesigner, including specific operator configurations, Execute DAT scripts, and testing procedures.

## 1. TouchDesigner Network Setup

### 1.1 Core Network Structure

Create the following TouchDesigner network hierarchy:

```
/project
├── input/
│   ├── videoin1 (Video In TOP)
│   └── cam1 (Camera COMP) 
├── processing/
│   ├── hand_gesture_detector (Text DAT)
│   ├── parametric_integration (Text DAT)
│   ├── frame_processor (Execute DAT)
│   └── mode_controller (Execute DAT)
├── data/
│   ├── gesture_params (Table DAT)
│   ├── parametric_params (Table DAT)
│   ├── trajectory_data (Table DAT)
│   └── particle_params (Table DAT)
├── rendering/
│   ├── particle_system/
│   │   ├── particle_generator (Add SOP)
│   │   ├── particle_instances (Instance2 COMP)
│   │   └── particle_material (Phong MAT)
│   └── parametric_system/
│       ├── trajectory_generator (Add SOP)
│       ├── trajectory_lines (Line SOP)
│       ├── rod1_generator (Add SOP)
│       ├── rod2_generator (Add SOP)
│       ├── rod_tubes (Tube SOP)
│       └── parametric_material (Phong MAT)
└── output/
    ├── render1 (Render TOP)
    └── out1 (Out TOP)
```

### 1.2 Operator Configuration Details

#### Video In TOP (`videoin1`)
```python
# Parameters
Device Index: 0
Resolution: Custom
Custom Width: 640
Custom Height: 480
Pixel Format: RGB24
FPS: 30
```

#### Text DAT (`hand_gesture_detector`)
```python
# Import existing hand gesture detector
# File > Import > Select hand_gesture_detector.py
```

#### Text DAT (`parametric_integration`)
```python
# Import the new integration module
# File > Import > Select td_parametric_integration.py
```

## 2. Execute DAT Scripts

### 2.1 Main Frame Processor (`frame_processor`)

```python
"""
Main frame processing Execute DAT
Handles camera input, gesture detection, and parameter updates
"""

def onFrameEnd(frame):
    """Called every frame to process camera input and update systems."""
    
    try:
        # Get camera data
        video_in = op('videoin1')
        if video_in.width == 0 or video_in.height == 0:
            return  # No valid input
        
        # Convert TouchDesigner texture to numpy array
        camera_data = video_in.numpyArray(delayed=True)
        
        if camera_data is None:
            return
        
        # Process through integration system
        integration = op('parametric_integration').module
        
        # Ensure system is initialized
        if not hasattr(integration, 'td_ops') or not integration.td_ops:
            initialize_integration_system()
        
        # Process frame
        result = integration.process_frame_update(camera_data, frame)
        
        # Handle results
        if result['success']:
            # Update performance display
            update_performance_display(result['performance'])
            
            # Update gesture status display
            update_gesture_display(result['gesture_info'])
            
        else:
            # Handle processing error
            print(f"Frame processing error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Frame processor exception: {e}")


def initialize_integration_system():
    """Initialize the integration system with TouchDesigner operators."""
    
    ops_dict = {
        'gesture_params': op('gesture_params'),
        'parametric_params': op('parametric_params'),
        'trajectory_data': op('trajectory_data'),
        'particle_params': op('particle_params')
    }
    
    # Initialize the system
    op('parametric_integration').module.initialize_touchdesigner_operators(ops_dict)
    
    print("Integration system initialized")


def update_performance_display(perf_stats):
    """Update performance monitoring display."""
    
    if 'performance_display' in op.keys():
        perf_text = f"""Performance Stats:
FPS: {perf_stats.get('average_fps', 0):.1f}
Frame Time: {perf_stats.get('frame_time', 0)*1000:.1f}ms  
Gesture Detection: {perf_stats.get('gesture_detection_time', 0)*1000:.1f}ms
Parameter Update: {perf_stats.get('parameter_update_time', 0)*1000:.1f}ms
Table Update: {perf_stats.get('table_update_time', 0)*1000:.1f}ms"""
        
        op('performance_display').par.text = perf_text


def update_gesture_display(gesture_info):
    """Update gesture status display."""
    
    if 'gesture_display' in op.keys():
        gesture_text = f"""Gesture Status:
Hands Detected: {gesture_info.get('hands_detected', 0)}
Active Gestures: {gesture_info.get('digit_gestures', [])}
Paused: {gesture_info.get('is_paused', False)}
Hand Assignment: {gesture_info.get('hand_assignment', 'unknown')}"""
        
        op('gesture_display').par.text = gesture_text
```

### 2.2 Mode Controller (`mode_controller`)

```python
"""
Mode Controller Execute DAT
Handles rendering mode switching and system state management
"""

def onValueChange(prev, current):
    """Handle mode changes from UI controls."""
    
    new_mode = current.val
    integration = op('parametric_integration').module
    
    print(f"Mode change requested: {prev.val} -> {new_mode}")
    
    # Attempt mode switch
    success = integration.set_rendering_mode(new_mode)
    
    if success:
        # Update system state
        update_rendering_state(new_mode)
        
        # Update UI feedback
        if 'mode_status' in op.keys():
            op('mode_status').par.text = f"Mode: {new_mode.upper()}"
        
        # Trigger visual transition
        trigger_mode_transition(prev.val, new_mode)
        
        print(f"Successfully switched to {new_mode} mode")
        
    else:
        print(f"Failed to switch to {new_mode} mode")
        # Revert UI
        current.val = prev.val


def update_rendering_state(mode):
    """Update TouchDesigner operators based on rendering mode."""
    
    if mode == 'particle':
        # Enable particle system
        op('particle_generator').par.display = True
        op('particle_instances').par.display = True
        op('particle_material').par.active = True
        
        # Disable parametric system
        op('trajectory_generator').par.display = False
        op('trajectory_lines').par.display = False
        op('rod1_generator').par.display = False
        op('rod2_generator').par.display = False
        op('parametric_material').par.active = False
        
    elif mode == 'parametric':
        # Disable particle system
        op('particle_generator').par.display = False
        op('particle_instances').par.display = False
        op('particle_material').par.active = False
        
        # Enable parametric system
        op('trajectory_generator').par.display = True
        op('trajectory_lines').par.display = True
        op('rod1_generator').par.display = True
        op('rod2_generator').par.display = True
        op('parametric_material').par.active = True
        
    elif mode == 'hybrid':
        # Enable both systems
        op('particle_generator').par.display = True
        op('particle_instances').par.display = True
        op('particle_material').par.active = True
        op('trajectory_generator').par.display = True
        op('trajectory_lines').par.display = True
        op('rod1_generator').par.display = True
        op('rod2_generator').par.display = True
        op('parametric_material').par.active = True


def trigger_mode_transition(old_mode, new_mode):
    """Trigger smooth visual transition between modes."""
    
    # Start transition animation
    if 'transition_opacity' in op.keys():
        transition_op = op('transition_opacity')
        transition_op.par.start.pulse()
        
    # Could add more sophisticated transition effects here
    # such as crossfade, scaling, rotation, etc.


def onPulse(info):
    """Handle pulse events for manual controls."""
    
    if info['name'] == 'reset_system':
        # Reset entire system
        integration = op('parametric_integration').module
        integration.reset_trajectory()
        integration.gesture_bridge.reset_parameters()
        print("System reset completed")
        
    elif info['name'] == 'toggle_pause':
        # Toggle pause state
        integration = op('parametric_integration').module
        paused = integration.toggle_pause()
        print(f"System {'paused' if paused else 'resumed'}")
```

### 2.3 Parameter Monitor (`parameter_monitor`)

```python
"""
Parameter Monitor Execute DAT
Monitors and logs parameter changes for debugging
"""

def onValueChange(prev, current):
    """Monitor parameter table changes."""
    
    param_name = current.path.split('/')[-1]
    
    print(f"Parameter changed: {param_name} = {prev.val} -> {current.val}")
    
    # Log significant changes
    if abs(float(current.val) - float(prev.val)) > 0.1:
        log_parameter_change(param_name, prev.val, current.val)


def log_parameter_change(param_name, old_val, new_val):
    """Log parameter changes for analysis."""
    
    timestamp = absTime.seconds
    log_entry = f"{timestamp:.3f}: {param_name} changed from {old_val} to {new_val}"
    
    # Write to log file or table
    if 'parameter_log' in op.keys():
        log_table = op('parameter_log')
        log_table.appendRow([timestamp, param_name, old_val, new_val])
        
        # Keep only recent entries
        if log_table.numRows > 1000:
            log_table.deleteRow(1)  # Remove oldest entry


def onFrameStart(frame):
    """Monitor system health every frame."""
    
    # Check performance
    current_fps = op('performchamp1')['fps'] if 'performchamp1' in op.keys() else 30
    
    if current_fps < 15:
        print(f"Performance warning: FPS dropped to {current_fps:.1f}")
        
        # Trigger automatic quality reduction
        integration = op('parametric_integration').module
        if hasattr(integration, 'resource_manager'):
            integration.resource_manager.monitor_and_adjust()
```

## 3. SOP Configuration Details

### 3.1 Trajectory Generator (`trajectory_generator`)

```python
# Add SOP Configuration
Point Count: len(op('trajectory_data')) - 1 if op('trajectory_data').numRows > 1 else 0

# Per-point position expressions
# Point X:
row = me.inputIndex + 1
op('trajectory_data')[row, 'x'].val if op('trajectory_data').numRows > row else 0

# Point Y: 
row = me.inputIndex + 1
op('trajectory_data')[row, 'y'].val if op('trajectory_data').numRows > row else 0

# Point Z:
row = me.inputIndex + 1  
op('trajectory_data')[row, 'z'].val if op('trajectory_data').numRows > row else 0
```

### 3.2 Rod Generators (`rod1_generator`, `rod2_generator`)

```python
# Rod 1 Generator (Add SOP)
Point Count: 2

# Point 0 (Origin)
X: 0
Y: 0
Z: 0

# Point 1 (Rod 1 End)
X: op('parametric_params')[1, 'r1'] * cos(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
Y: op('parametric_params')[1, 'r1'] * sin(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
Z: 0

# Rod 2 Generator (Add SOP)  
Point Count: 2

# Point 0 (Rod 1 End - same as Rod 1 Point 1)
X: op('parametric_params')[1, 'r1'] * cos(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
Y: op('parametric_params')[1, 'r1'] * sin(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
Z: 0

# Point 1 (Final Position - Rod 1 + Rod 2)
r1_x = op('parametric_params')[1, 'r1'] * cos(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
r1_y = op('parametric_params')[1, 'r1'] * sin(op('parametric_params')[1, 'w1'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p1'])
r2_x = op('parametric_params')[1, 'r2'] * cos(op('parametric_params')[1, 'w2'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p2'])
r2_y = op('parametric_params')[1, 'r2'] * sin(op('parametric_params')[1, 'w2'] * op('parametric_params')[1, 'theta'] + op('parametric_params')[1, 'p2'])

X: r1_x + r2_x  
Y: r1_y + r2_y
Z: 0
```

### 3.3 Line and Tube SOPs

```python
# Line SOP Configuration (trajectory_lines)
Input: trajectory_generator
Method: By Distance
Close Distance: 1.0
Smooth: On
Smooth Passes: 2

# Tube SOP Configuration (rod_tubes)
Input: merge1 (rod1_generator + rod2_generator merged)
Radius: 0.02
Divisions: 8
Caps: On
End Caps: On
```

## 4. Material Configuration

### 4.1 Parametric Material (`parametric_material`)

```python
# Phong MAT Parameters
Diffuse Color: [0.2, 0.6, 1.0]
Ambient Color: [0.1, 0.3, 0.5]
Specular Color: [1.0, 1.0, 1.0]
Specular Rolloff: 20.0
Emit Color: [0.05, 0.15, 0.3]
Alpha: 0.9
Two Sided Lighting: On
```

### 4.2 Particle Material (Enhanced)

```python
# Phong MAT Parameters for particles
Diffuse Color: [1.0, 0.4, 0.1]  # Orange
Ambient Color: [0.5, 0.2, 0.05]
Specular Color: [1.0, 0.8, 0.6]
Specular Rolloff: 15.0
Emit Color: [0.3, 0.12, 0.03]
Alpha: 0.8
```

## 5. Testing and Validation

### 5.1 Initial System Test

```python
def test_system_initialization():
    """Test basic system setup and initialization."""
    
    # Check all required operators exist
    required_ops = [
        'videoin1', 'hand_gesture_detector', 'parametric_integration',
        'gesture_params', 'parametric_params', 'trajectory_data',
        'trajectory_generator', 'rod1_generator', 'rod2_generator'
    ]
    
    missing_ops = []
    for op_name in required_ops:
        if op_name not in op.keys():
            missing_ops.append(op_name)
    
    if missing_ops:
        print(f"Missing operators: {missing_ops}")
        return False
    
    # Test integration module
    try:
        integration = op('parametric_integration').module
        status = integration.get_system_status()
        print(f"System status: {status}")
        return True
    except Exception as e:
        print(f"Integration module error: {e}")
        return False


def test_mode_switching():
    """Test all rendering modes."""
    
    integration = op('parametric_integration').module
    modes = ['particle', 'parametric', 'hybrid']
    
    for mode in modes:
        print(f"Testing {mode} mode...")
        success = integration.set_rendering_mode(mode)
        
        if success:
            # Wait for mode to activate
            run("args[0].set_rendering_mode(args[1])", integration, mode, delayFrames=30)
            current_mode = integration.get_rendering_mode()
            
            if current_mode == mode:
                print(f"✓ {mode} mode activated successfully")
            else:
                print(f"✗ {mode} mode activation failed")
                return False
        else:
            print(f"✗ Failed to switch to {mode} mode")
            return False
    
    return True


def test_gesture_response():
    """Test gesture detection and parameter response."""
    
    integration = op('parametric_integration').module
    
    # Create test gesture data
    import numpy as np
    test_frame = np.random.rand(480, 640, 3).astype(np.uint8) * 255
    
    # Process test frame
    result = integration.process_frame_update(test_frame, 0)
    
    if result['success']:
        print("✓ Gesture processing successful")
        print(f"Parameters: {result['parameters']}")
        return True
    else:
        print(f"✗ Gesture processing failed: {result.get('error', 'Unknown')}")
        return False
```

### 5.2 Performance Validation

```python
def run_performance_test(duration=30):
    """Run performance test for specified duration."""
    
    print(f"Running {duration}s performance test...")
    
    integration = op('parametric_integration').module
    start_time = absTime.seconds
    frame_count = 0
    
    # Test all modes
    modes = ['particle', 'parametric', 'hybrid']
    test_duration_per_mode = duration / len(modes)
    
    for mode in modes:
        print(f"Testing {mode} mode performance...")
        integration.set_rendering_mode(mode)
        
        mode_start_time = absTime.seconds
        mode_frame_count = 0
        
        while absTime.seconds - mode_start_time < test_duration_per_mode:
            frame_count += 1
            mode_frame_count += 1
            
            # Check if system is still responsive
            status = integration.get_system_status()
            if not status:
                print(f"✗ System became unresponsive in {mode} mode")
                return False
        
        mode_fps = mode_frame_count / test_duration_per_mode
        print(f"{mode} mode average FPS: {mode_fps:.2f}")
        
        if mode_fps < 15:
            print(f"⚠ Performance warning: {mode} mode FPS below threshold")
    
    total_fps = frame_count / duration
    print(f"Overall average FPS: {total_fps:.2f}")
    
    return total_fps >= 20  # Minimum acceptable performance
```

## 6. Troubleshooting Guide

### 6.1 Common Issues

#### Camera Not Working
```python
# Check camera access
if op('videoin1').width == 0:
    print("Camera not accessible - check permissions")
    # Try different device index
    op('videoin1').par.deviceindex = 1
```

#### Module Import Errors
```python
# Refresh Python modules
op('hand_gesture_detector').par.loadonstartup.pulse()
op('parametric_integration').par.loadonstartup.pulse()
```

#### Performance Issues
```python
# Reduce quality for better performance
integration = op('parametric_integration').module
integration.set_trail_length(200)  # Reduce from default 500
```

### 6.2 Debug Information

```python
def print_debug_info():
    """Print comprehensive debug information."""
    
    print("=== TouchDesigner Parametric Integration Debug Info ===")
    
    # System status
    integration = op('parametric_integration').module
    status = integration.get_system_status()
    
    for key, value in status.items():
        print(f"{key}: {value}")
    
    # Table data
    print(f"\nTable Rows:")
    print(f"gesture_params: {op('gesture_params').numRows}")
    print(f"parametric_params: {op('parametric_params').numRows}")
    print(f"trajectory_data: {op('trajectory_data').numRows}")
    
    # Performance
    if 'performchamp1' in op.keys():
        print(f"\nPerformance:")
        print(f"FPS: {op('performchamp1')['fps']}")
        print(f"Memory: {op('performchamp1')['ram_usage_mb']}MB")
```

This implementation guide provides all the necessary details to successfully set up and run the parametric integration system in TouchDesigner, with comprehensive testing and debugging support.