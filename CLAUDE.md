# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a TouchDesigner hand gesture recognition project that creates real-time particle sphere effects controlled by camera-based hand tracking. The system uses MediaPipe for gesture detection and TouchDesigner for real-time 3D rendering.

## Project Architecture

### Core Components

1. **Hand Gesture Detection** (`hand_gesture_detector.py`)
   - Uses MediaPipe to detect hand landmarks and classify gestures
   - Outputs gesture data including hand positions, openness, and gesture types
   - Functions: `get_gesture_data()`, `initialize_detector()`, `process_camera_frame()`

2. **Data Processing** (`td_data_processor.py`)
   - Converts gesture data into TouchDesigner-compatible parameters
   - Maps hand gestures to particle and sphere control parameters
   - Functions: `process_gesture_update()`, `get_particle_table_data()`, `get_sphere_table_data()`

3. **Particle System** (`particle_system.py`)
   - Manages 3D particle effects with sphere-based emission
   - Controls particle life, velocity, color, and turbulence based on gestures
   - Functions: `update_particle_system()`, `get_particle_data_for_gpu()`

4. **Sphere Renderer** (`sphere_renderer.py`)
   - Handles 3D sphere geometry generation and deformation
   - Provides PBR materials and dynamic animations
   - Functions: `update_sphere_renderer()`, `get_sphere_geometry()`, `get_shader_uniforms()`

## TouchDesigner Integration

### Required Operators
- **Video In TOP**: Camera input
- **Text DATs**: Contains Python script modules
- **Execute DATs**: Frame callbacks and main control logic
- **Table DATs**: Parameter storage (particle_params, sphere_params)
- **Add SOP**: Particle point generation (macOS alternative to Particle GPU TOP)
- **Instance2 COMP**: Particle instancing for macOS
- **Geometry COMP**: Dynamic sphere mesh
- **GLSL TOP**: Custom shaders (optional)

### Key Parameters Mapping

| Gesture Input | Particle Effect | Sphere Effect |
|---------------|-----------------|---------------|
| Hand openness | Particle size & spread | Surface noise strength |
| Gesture strength | Velocity & emission rate | Radius & pulsation |
| Gesture type (fist/open) | Turbulence level | Rotation speed |
| Two-hand distance | Spread range | Deformation amount |
| Hand position | Color hue/saturation | Material properties |

## Development Commands

### Dependencies Installation
```bash
pip install opencv-python mediapipe numpy
```

### Common TouchDesigner Operations
- Initialize gesture detector: `op('gesture_detector').module.initialize_detector()`
- Update particle system: `op('particle_system').module.update_particle_system(params, dt)`
- Get current gesture data: `op('gesture_detector').module.get_gesture_data()`

## File Structure

- `hand_gesture_detector.py` - MediaPipe hand tracking and gesture classification
- `td_data_processor.py` - Gesture-to-parameter mapping and smoothing
- `particle_system.py` - 3D particle system with sphere emission pattern
- `sphere_renderer.py` - Dynamic sphere geometry and PBR rendering
- `touchdesigner_setup_guide.md` - Detailed TouchDesigner network setup
- `README.md` - Project overview and usage instructions

## Key Classes and Functions

### HandGestureDetector
- `detect_gesture_type()`: Classify hand gestures (fist, open, one-five fingers)
- `calculate_hand_openness()`: Measure hand spread (0-1 range)
- `process_frame()`: Main gesture detection pipeline

### TouchDesignerDataProcessor  
- `update_particle_params()`: Map gestures to particle controls
- `update_sphere_params()`: Map gestures to sphere deformation
- `apply_smoothing()`: Smooth parameter changes for stable animation

### ParticleSystem
- `create_particle()`: Generate particles on sphere surface
- `update_particles()`: Physics simulation with turbulence and attraction
- `get_particle_data_for_gpu()`: Format data for GPU particle rendering

### SphereRenderer
- `generate_base_sphere()`: Create base sphere mesh
- `apply_deformation()`: Real-time vertex deformation
- `generate_shader_uniforms()`: Shader parameter generation

## Performance Considerations

- Default max particles: 2000 (adjust based on hardware)
- Sphere resolution: 64 subdivisions (reduce for better performance)
- Camera resolution: 640x480 (balance between accuracy and speed)
- Smoothing factor: 0.8 (higher = more stable, lower = more responsive)

## Common Issues

- MediaPipe requires good lighting for accurate hand detection
- Large particle counts may cause GPU memory issues
- Binary `.toe` files are not suitable for version control
- Python scripts should be kept in separate `.py` files for proper versioning