# TouchDesigner GPU Rendering Pipeline Design
## Parametric Trajectory Visualization

This document specifies the GPU-optimized rendering pipeline for real-time parametric equation visualization in TouchDesigner, with full macOS compatibility and performance optimization.

## 1. Rendering Pipeline Overview

### 1.1 Pipeline Architecture
```
Table DAT Data → SOP Generation → Geometry Processing → Material Application → GPU Rendering
     ↓                ↓                   ↓                    ↓                ↓
trajectory_data → Add/Line SOPs → Transform/Normal → PBR Material → Render TOP
```

### 1.2 Parallel Processing Streams

#### Stream 1: Trajectory Curve
- **Input**: `trajectory_data` Table DAT
- **Processing**: Line SOP with smooth interpolation
- **Output**: Dynamic curve geometry

#### Stream 2: Rotating Rods
- **Input**: Rod position data from integration module
- **Processing**: Line SOP with cylindrical geometry
- **Output**: Animated rod visualization

#### Stream 3: Current Point Marker
- **Input**: Latest trajectory point
- **Processing**: Sphere SOP with dynamic scaling
- **Output**: Moving point indicator

## 2. SOP-Based Geometry Pipeline

### 2.1 Trajectory Generation (macOS Compatible)

#### Add SOP Configuration (`trajectory_generator`)
```python
# Add SOP Python expression for point generation
def generate_trajectory_points():
    """Generate points from trajectory_data Table DAT."""
    table = op('trajectory_data')
    points = []
    
    for row_idx in range(1, table.numRows):  # Skip header
        x = table[row_idx, 'x'].val
        y = table[row_idx, 'y'].val 
        z = table[row_idx, 'z'].val
        alpha = table[row_idx, 'trail_alpha'].val
        
        # Create point with attributes
        point = {
            'position': [x, y, z],
            'Cd': [0.2, 0.6, 1.0],  # Blue color
            'Alpha': alpha,
            'pscale': 0.02
        }
        points.append(point)
    
    return points

# Expression in Add SOP's Point Count parameter
len(op('parametric_integration').module.trajectory_points)

# Expression for point positions (called per point)
table = op('trajectory_data')
point_idx = me.inputIndex
if table.numRows > point_idx + 1:
    me.x = table[point_idx + 1, 'x'].val
    me.y = table[point_idx + 1, 'y'].val
    me.z = table[point_idx + 1, 'z'].val
```

#### Line SOP Configuration (`trajectory_lines`)
```python
# Connect trajectory points into smooth curves
Input: trajectory_generator (Add SOP)
Method: By Distance
Close Distance: 0.5  # Connect nearby points
Smooth: On
Smooth Passes: 2
```

### 2.2 Rod Visualization Pipeline

#### Rod 1 Generator (`rod1_generator`)
```python
# Add SOP for Rod 1 (origin to first rotation point)
def create_rod1_points():
    integration = op('parametric_integration').module
    rod1_pos = integration.rod_positions['rod1']
    
    return [
        [0.0, 0.0, 0.0],  # Origin
        [rod1_pos[0], rod1_pos[1], 0.0]  # Rod 1 endpoint
    ]

# Point Count: 2
# Position expressions:
# Point 0: [0, 0, 0]
# Point 1: [op('parametric_params')[1, 'r1'] * cos(op('parametric_params')[1, 'theta'] * op('parametric_params')[1, 'w1']), 
#           op('parametric_params')[1, 'r1'] * sin(op('parametric_params')[1, 'theta'] * op('parametric_params')[1, 'w1']), 0]
```

#### Rod 2 Generator (`rod2_generator`)
```python
# Add SOP for Rod 2 (rod1 end to final position)
def create_rod2_points():
    integration = op('parametric_integration').module
    rod1_pos = integration.rod_positions['rod1']
    rod2_pos = integration.rod_positions['rod2']
    
    return [
        [rod1_pos[0], rod1_pos[1], 0.0],  # Rod 1 endpoint
        [rod2_pos[0], rod2_pos[1], 0.0]   # Final position
    ]
```

#### Tube SOP (`rod_tubes`)
```python
# Convert lines to cylindrical tubes
Input: rod1_generator + rod2_generator (Merge SOP)
Radius: 0.02
Divisions: 8  # Low poly for performance
Caps: On
```

### 2.3 Current Point Indicator

#### Sphere SOP (`current_point_marker`)
```python
# Dynamic sphere at current trajectory position
Radius: 0.05 * (1 + 0.3 * sin(absTime.frame * 0.2))  # Pulsating
Frequency: 12  # Low poly sphere
Center X: op('trajectory_data')[op('trajectory_data').numRows-1, 'x'] if op('trajectory_data').numRows > 1 else 0
Center Y: op('trajectory_data')[op('trajectory_data').numRows-1, 'y'] if op('trajectory_data').numRows > 1 else 0
Center Z: 0
```

### 2.4 Trail Effect System

#### Trail Alpha Calculation
```python
# Attribute Create SOP for alpha values
Name: Alpha
Type: Float
Default Value: 1.0

# Expression for alpha based on point age
point_idx = me.inputIndex
total_points = me.inputGeometry.numPoints
alpha = point_idx / total_points if total_points > 0 else 1.0
me.Alpha = alpha * 0.8 + 0.2  # Ensure minimum visibility
```

#### Color Gradient System
```python
# Attribute Create SOP for color gradient
Name: Cd
Type: Vector
Default Value: [0.2, 0.6, 1.0]

# HSV color shift for trail
import colorsys
point_idx = me.inputIndex
total_points = me.inputGeometry.numPoints

if total_points > 0:
    # Hue shift based on position in trail
    hue = (240 + (point_idx / total_points) * 60) / 360  # Blue to cyan
    saturation = 0.8
    value = 0.6 + (point_idx / total_points) * 0.4  # Fade to brighter
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    me.Cd = list(rgb)
```

## 3. Material and Shading Pipeline

### 3.1 PBR Material Setup (`parametric_material`)

#### Base Material Properties
```glsl
// Phong MAT parameters for parametric visualization
Diffuse Color: point('Cd')  // Use point color attribute
Alpha: point('Alpha')       // Use alpha attribute
Specular: 0.3
Shininess: 20.0
Two Sided Lighting: On
```

#### Advanced Material (Optional GLSL Enhancement)
```glsl
// Custom GLSL material for enhanced visuals
// Vertex Shader
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in float alpha;
layout (location = 3) in float pscale;

uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix; 
uniform mat4 uModelMatrix;
uniform float uTime;

out vec3 vColor;
out float vAlpha;
out float vScale;

void main() {
    vColor = color;
    vAlpha = alpha;
    vScale = pscale;
    
    vec3 pos = position;
    
    // Add subtle motion for dynamic effect
    pos += 0.01 * sin(uTime * 2.0 + position.x * 10.0) * normalize(position);
    
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(pos, 1.0);
}

// Fragment Shader
#version 330 core

in vec3 vColor;
in float vAlpha;
in float vScale;

uniform float uTime;
uniform vec3 uLightDirection;

out vec4 FragColor;

void main() {
    // Simple lighting calculation
    vec3 normal = normalize(vec3(0.0, 0.0, 1.0));  // Simplified for 2D
    float lighting = max(dot(normal, normalize(uLightDirection)), 0.3);
    
    vec3 finalColor = vColor * lighting;
    
    // Add glow effect for trail
    float glow = 1.0 + 0.3 * sin(uTime * 3.0);
    finalColor *= glow;
    
    FragColor = vec4(finalColor, vAlpha);
}
```

### 3.2 Multi-Pass Rendering

#### Pass 1: Trail Lines (Low Priority)
- Render trajectory lines with alpha blending
- Low tessellation for performance
- Depth write disabled for transparency

#### Pass 2: Rod Geometry (Medium Priority)
- Render rotating rods with solid materials
- Medium tessellation for smooth cylinders
- Full depth testing enabled

#### Pass 3: Current Point (High Priority)
- Render current point marker with emission
- High tessellation for smooth sphere
- Depth bias for prominence

## 4. Performance Optimization Strategies

### 4.1 LOD (Level of Detail) System

#### Distance-Based LOD
```python
# Calculate camera distance for LOD adjustment
camera_pos = op('cam1').transform.translation
trajectory_center = [0, 0, 0]  # Parametric curve center

distance = math.sqrt(sum((a - b)**2 for a, b in zip(camera_pos, trajectory_center)))

# Adjust detail based on distance
if distance > 10.0:
    trajectory_detail = 0.25  # Low detail
    rod_segments = 4
elif distance > 5.0:
    trajectory_detail = 0.5   # Medium detail  
    rod_segments = 6
else:
    trajectory_detail = 1.0   # High detail
    rod_segments = 8
```

#### Frame Rate Adaptive LOD
```python
# Adjust quality based on performance
current_fps = op('performchamp1')['fps']

if current_fps < 20:
    # Emergency performance mode
    max_trajectory_points = 100
    rod_detail = 'low'
    effects_enabled = False
elif current_fps < 25:
    # Reduced quality mode
    max_trajectory_points = 250
    rod_detail = 'medium'
    effects_enabled = True
else:
    # Full quality mode
    max_trajectory_points = 500
    rod_detail = 'high'
    effects_enabled = True
```

### 4.2 Geometry Caching and Instancing

#### Trajectory Point Instancing
```python
# Use Instance2 COMP for efficient trail rendering
# Instead of individual geometry for each point
template_sphere = op('trail_point_template')  # Single low-poly sphere
instance_table = op('trajectory_data')       # Position/rotation data

# Instance2 COMP setup:
# Template: trail_point_template
# Instance Source: trajectory_data
# Translate: tx, ty, tz columns
# Scale: uniform scale based on trail_alpha
```

#### Rod Geometry Caching
```python
# Cache rod geometry when parameters change slowly
class RodGeometryCache:
    def __init__(self):
        self.cached_geometry = {}
        self.cache_threshold = 0.001  # Minimum change to trigger update
        
    def get_rod_geometry(self, r1, r2, theta):
        cache_key = (round(r1, 3), round(r2, 3), round(theta, 3))
        
        if cache_key not in self.cached_geometry:
            # Generate new geometry
            self.cached_geometry[cache_key] = self._generate_rod_geometry(r1, r2, theta)
            
        return self.cached_geometry[cache_key]
```

### 4.3 Culling and Clipping

#### Frustum Culling
```python
# Automatically cull trajectory points outside view
def is_point_in_frustum(point, camera_matrix, projection_matrix):
    # Transform point to clip space
    clip_pos = projection_matrix * camera_matrix * vec4(point, 1.0)
    
    # Check if point is within clip bounds
    return (-clip_pos.w <= clip_pos.x <= clip_pos.w and
            -clip_pos.w <= clip_pos.y <= clip_pos.w and
            -clip_pos.w <= clip_pos.z <= clip_pos.w)

# Apply in Add SOP expression
if is_point_in_frustum(me.position, op('cam1').matrix, op('cam1').projection):
    me.display = 1
else:
    me.display = 0  # Cull point
```

#### Temporal Culling
```python
# Remove old trajectory points beyond trail length
current_time = absTime.seconds
max_age = 10.0  # seconds

# In trajectory_data Table DAT update
for row in range(table.numRows - 1, 0, -1):  # Reverse iteration
    point_time = table[row, 'timestamp'].val
    if current_time - point_time > max_age:
        table.deleteRow(row)
```

## 5. macOS Compatibility Considerations

### 5.1 Metal Rendering Backend
- Use TouchDesigner's Metal renderer when available
- Fallback to OpenGL with compatibility profile
- Avoid advanced GPU features not supported on older Macs

### 5.2 Memory Management
```python
# Conservative memory usage for macOS
max_trajectory_points_macos = 750  # Reduced from 1000
max_geometry_cache_size = 100      # Limit cached geometry
texture_resolution = 1024          # Reasonable texture size
```

### 5.3 Performance Monitoring
```python
# Built-in performance monitoring
def monitor_performance():
    perf_data = {
        'fps': op('performchamp1')['fps'],
        'memory': op('performchamp1')['ram_usage_mb'],
        'gpu_utilization': op('performchamp1')['gpu_percent'],
        'geometry_count': op('trajectory_generator').outputGeometry.numPoints
    }
    
    # Auto-adjust quality if performance drops
    if perf_data['fps'] < 20 or perf_data['memory'] > 2000:  # 2GB limit
        activate_performance_mode()
    
    return perf_data
```

## 6. Integration with Hybrid Mode

### 6.1 Resource Allocation
```python
# Hybrid mode resource management
if rendering_mode == 'hybrid':
    # Particle system gets 60% of resources
    max_particles = int(default_max_particles * 0.6)
    
    # Parametric system gets 40% of resources  
    max_trajectory_points = int(default_trajectory_points * 0.4)
    rod_detail_level = 'medium'
    
    # Shared resources
    total_geometry_budget = 3000  # Total points/vertices
```

### 6.2 Synchronized Animation
```python
# Keep both systems synchronized
def sync_animation_systems():
    # Use same time base for both particle and parametric systems
    unified_time = absTime.seconds
    unified_frame = absTime.frame
    
    # Update parametric theta
    op('parametric_integration').module.current_parameters['theta'] = unified_time * 0.5
    
    # Update particle system timing
    op('particle_system').module.current_frame = unified_frame
```

This GPU rendering pipeline design ensures optimal performance while maintaining visual quality and full compatibility with TouchDesigner's execution model and macOS requirements.