# TouchDesigner Parametric Integration Architecture Specification

## Overview

This document specifies the complete TouchDesigner node architecture for integrating the parametric equation system (`parametric_equation_renderer.py`, `gesture_parametric_bridge.py`) with the existing hand gesture particle system. The architecture preserves existing functionality while adding seamless parametric curve visualization capabilities.

## 1. TouchDesigner Network Architecture

### 1.1 Core Network Flow

```
Camera Input → Gesture Processing → Parameter Mapping → Dual Rendering Pipeline
     ↓              ↓                    ↓                    ↓
Video In TOP → Python Scripts → Table DATs → Particle/Parametric Renderers
```

### 1.2 Node Structure Specification

#### Input Layer
- **Video In TOP** (`videoin1`)
  - Resolution: 640x480 (optimized for gesture detection)
  - Format: RGB24 
  - Connection to system camera

#### Processing Layer
- **Text DAT** (`hand_gesture_detector`)
  - Contains: `hand_gesture_detector.py` 
  - Functions: MediaPipe hand tracking and gesture classification

- **Text DAT** (`parametric_integration`)
  - Contains: `td_parametric_integration.py` (new bridge module)
  - Functions: Unified parameter management and mode switching

- **Execute DAT** (`frame_processor`)
  - Script: Camera frame processing and parameter updates
  - Callback: `onFrameEnd` at 30 FPS

#### Data Storage Layer
- **Table DAT** (`gesture_params`)
  - Schema: Hand gesture data and raw parameters
  - Columns: `hand_id`, `gesture_type`, `position_x`, `position_y`, `openness`, `timestamp`

- **Table DAT** (`parametric_params`)  
  - Schema: Parametric equation parameters
  - Columns: `r1`, `r2`, `w1`, `w2`, `p1`, `p2`, `theta`, `theta_step`, `mode`

- **Table DAT** (`particle_params`)
  - Schema: Particle system parameters (existing)
  - Columns: `emission_rate`, `radius`, `velocity`, `size`, `turbulence`, `deformation`

- **Table DAT** (`trajectory_data`)
  - Schema: Real-time trajectory points for parametric curves
  - Columns: `point_id`, `x`, `y`, `z`, `timestamp`, `trail_alpha`

#### Rendering Layer

##### Particle System Path (Existing)
- **Add SOP** (`particle_generator`)
  - macOS-compatible particle generation
  - Points created from `particle_params` table

- **Instance2 COMP** (`particle_instances`) 
  - Instancing for performance
  - Connected to particle geometry

##### Parametric System Path (New)
- **Add SOP** (`trajectory_generator`)
  - Point generation from `trajectory_data` table
  - Real-time curve point creation

- **Line SOP** (`trajectory_lines`)
  - Connect trajectory points into smooth curves
  - Trail effect with alpha falloff

- **Geometry COMP** (`parametric_renderer`)
  - Combined trajectory and rod visualization
  - Dynamic material properties

#### Output Layer
- **Render TOP** (`final_output`)
  - Composite renderer combining both systems
  - Switch between particle/parametric modes

### 1.3 Execute DAT Callback Architecture

#### Primary Frame Processor (`frame_processor`)
```python
def onFrameEnd(frame):
    # Process camera input
    camera_data = op('videoin1').numpyArray()
    
    # Get gesture data from hand detector
    gesture_data = op('hand_gesture_detector').module.get_gesture_data()
    
    # Update unified parameter system
    op('parametric_integration').module.process_frame_update(
        camera_data, gesture_data, frame
    )
    
    # Update rendering mode
    current_mode = op('parametric_params')['mode', 1].val
    op('parametric_integration').module.update_rendering_mode(current_mode)
```

#### Mode Switch Controller (`mode_controller`)
```python
def onValueChange(prev, current):
    mode = current.val
    integration_module = op('parametric_integration').module
    
    if mode == 'particle':
        integration_module.activate_particle_mode()
    elif mode == 'parametric':  
        integration_module.activate_parametric_mode()
    elif mode == 'hybrid':
        integration_module.activate_hybrid_mode()
```

## 2. Table DAT Schema Specifications

### 2.1 Gesture Parameters Table (`gesture_params`)
```
| Column    | Type  | Description                    |
|-----------|-------|--------------------------------|
| hand_id   | int   | Hand identifier (0-2)         |
| label     | str   | 'left', 'right', 'unknown'    |
| gesture   | int   | Gesture type (0-5)            |
| pos_x     | float | Normalized X position          |
| pos_y     | float | Normalized Y position          |
| openness  | float | Hand openness (0-1)           |
| timestamp | float | Frame timestamp               |
| active    | bool  | Whether hand is detected       |
```

### 2.2 Parametric Parameters Table (`parametric_params`)
```
| Column     | Type  | Description                    |
|------------|-------|--------------------------------|
| r1         | float | Radius 1 parameter             |
| r2         | float | Radius 2 parameter             |
| w1         | float | Frequency 1 parameter          |
| w2         | float | Frequency 2 parameter          |
| p1         | float | Phase 1 parameter              |
| p2         | float | Phase 2 parameter              |
| theta      | float | Current theta value            |
| theta_step | float | Animation increment            |
| mode       | str   | Rendering mode                 |
| paused     | bool  | Animation pause state          |
```

### 2.3 Trajectory Data Table (`trajectory_data`)
```
| Column      | Type  | Description                    |
|-------------|-------|--------------------------------|
| point_id    | int   | Sequential point ID            |
| x           | float | World X coordinate             |
| y           | float | World Y coordinate             |
| z           | float | World Z coordinate (0 for 2D)  |
| timestamp   | float | Point creation time            |
| trail_alpha | float | Alpha for trail effect         |
| rod1_x      | float | Rod 1 endpoint X               |
| rod1_y      | float | Rod 1 endpoint Y               |
| rod2_x      | float | Rod 2 endpoint X               |
| rod2_y      | float | Rod 2 endpoint Y               |
```

## 3. Data Flow Pipeline

### 3.1 Input Processing Flow
```
Camera Frame → MediaPipe Detection → Gesture Classification → Parameter Mapping
     ↓               ↓                      ↓                     ↓
Video In TOP → hand_gesture_detector → gesture_params → parametric_integration
```

### 3.2 Parameter Update Flow
```
Gesture Data → Twelve-Tone Mapping → Smoothing → Parameter Tables → Renderers
     ↓              ↓                    ↓            ↓              ↓
gesture_params → radius_mapper → smoothing_filter → parametric_params → SOPs
```

### 3.3 Rendering Pipeline Flow
```
Parameters → Point Generation → Geometry Creation → Material Application → Output
     ↓             ↓                 ↓                   ↓               ↓
Table DATs → Add/Line SOPs → Geometry COMPs → Material/Render → Render TOP
```

## 4. Performance Architecture

### 4.1 Frame Rate Optimization
- **Target Frame Rate**: 30 FPS sustained
- **Processing Throttling**: Skip frames if processing takes >33ms
- **Table Update Batching**: Batch parameter updates per frame
- **GPU Memory Management**: Limit trajectory history to 1000 points

### 4.2 Memory Management
- **Trajectory Buffer**: Circular buffer with configurable size
- **Particle Limits**: 2000 particles max (existing system)
- **Texture Caching**: Cache parameter textures between frames
- **Garbage Collection**: Clean up old trajectory points automatically

### 4.3 macOS Compatibility
- **Particle System**: Use Add SOP + Instance2 instead of Particle GPU TOP
- **GLSL Shaders**: Optional enhancement, not required for core functionality  
- **Camera Access**: Handle macOS camera permissions gracefully
- **Python Dependencies**: MediaPipe, NumPy, OpenCV compatible versions

## 5. Mode Switching Architecture

### 5.1 Rendering Modes

#### Particle Mode (Default)
- Activates existing particle system
- Disables parametric trajectory rendering
- Uses gesture data for particle parameters
- Maintains compatibility with existing TouchDesigner setup

#### Parametric Mode
- Activates parametric equation visualization
- Shows trajectory, rods, and trail
- Uses gesture data for r1, r2, w1, w2 parameters
- Real-time twelve-tone scale mapping

#### Hybrid Mode
- Both systems active simultaneously
- Particle system uses reduced particle count (1000)
- Parametric system uses reduced trail length (250 points)
- Shared gesture input with different parameter mappings

### 5.2 Transition Management
- **Smooth Transitions**: Parameter interpolation during mode switches
- **State Preservation**: Save/restore parameters when switching modes
- **Resource Management**: Cleanup unused resources in inactive modes

## 6. Integration Points

### 6.1 Existing System Integration
- **Preserve Interfaces**: Maintain existing TouchDesigner node connections
- **Backward Compatibility**: Existing particle system functions unchanged
- **Parameter Mapping**: Extend existing gesture-to-particle mapping
- **Performance Profile**: No degradation to existing particle system

### 6.2 New System Integration  
- **Unified Parameter Interface**: Single bridge module manages both systems
- **Gesture Mapping**: Twelve-tone scale mapping for parametric parameters
- **Real-time Updates**: Frame-synchronized parameter updates
- **Visual Feedback**: Display current gesture state and parameter values

## 7. Error Handling and Fallbacks

### 7.1 Camera Failures
- Graceful degradation to manual parameter control
- Error display in TouchDesigner network
- Automatic camera reconnection attempts

### 7.2 Gesture Detection Failures
- Fallback to last known good parameters
- Timeout-based parameter reset
- Visual indicators for detection status

### 7.3 Performance Degradation
- Automatic quality reduction (fewer particles/points)
- Frame rate monitoring and adjustment
- Emergency fallback to particle-only mode

## 8. Development and Testing Strategy

### 8.1 Incremental Implementation
1. Create basic parametric integration module
2. Implement Table DAT schemas
3. Add parametric rendering pipeline
4. Integrate mode switching
5. Optimize performance
6. Add error handling

### 8.2 Testing Approach
- **Unit Testing**: Individual module functionality
- **Integration Testing**: Cross-module parameter flow
- **Performance Testing**: Frame rate under load
- **User Testing**: Real-time gesture control responsiveness

This architecture specification provides the foundation for seamlessly integrating parametric equation visualization with the existing TouchDesigner hand gesture particle system while maintaining performance, compatibility, and extensibility.