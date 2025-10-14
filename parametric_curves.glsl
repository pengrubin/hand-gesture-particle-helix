//
// TouchDesigner GLSL Shaders for Parametric Curve Rendering
// Optimized for macOS Metal API compatibility and high-performance real-time visualization
//

//
// ===================== VERTEX SHADER =====================
//

#ifdef VERTEX_SHADER

// Vertex shader for parametric curve rendering
in vec3 P;          // Position
in vec3 N;          // Normal
in vec2 uv;         // UV coordinates
in vec4 Cd;         // Color

// Uniforms for parametric parameters
uniform float uCurveThickness;      // uniformf0: Curve thickness
uniform float uGlowIntensity;       // uniformf1: Glow effect intensity
uniform float uTime;                // uniformf2: Current time for animation
uniform float uTheta;               // uniformf3: Current theta parameter
uniform float uR1;                  // uniformf4: First radius
uniform float uR2;                  // uniformf5: Second radius
uniform float uW1;                  // uniformf6: First frequency
uniform float uW2;                  // uniformf7: Second frequency
uniform float uP1;                  // uniformf8: First phase
uniform float uP2;                  // uniformf9: Second phase
uniform float uColorFrequency;      // uniformf10: Color variation frequency
uniform float uAnimationSpeed;      // uniformf11: Animation speed multiplier
uniform float uTrailDecay;          // uniformf12: Trail fade factor
uniform float uVolumetricDensity;   // uniformf13: Volumetric effect density

// Camera and transformation matrices
uniform mat4 uTDMat;
uniform mat4 uWorldCamMat;
uniform mat4 uProjMat;

// Output to fragment shader
out vec3 worldPos;
out vec3 viewPos;
out vec3 normal;
out vec2 texCoord;
out vec4 color;
out float pointAlpha;
out float distanceFromCamera;
out vec3 velocity;

void main()
{
    // Calculate parametric curve position
    vec3 localPos = P;
    
    // Add curve deformation based on parametric equations
    float t = uTime * uAnimationSpeed + localPos.x * uColorFrequency;
    
    // Complex number calculation: z1 = r1 * e^(i*(w1*t + p1))
    vec2 z1 = vec2(
        uR1 * cos(uW1 * t + uP1),
        uR1 * sin(uW1 * t + uP1)
    );
    
    // Complex number calculation: z2 = r2 * e^(i*(w2*t + p2))
    vec2 z2 = vec2(
        uR2 * cos(uW2 * t + uP2),
        uR2 * sin(uW2 * t + uP2)
    );
    
    // Total complex position
    vec2 zTotal = z1 + z2;
    
    // Apply parametric displacement
    localPos.xy += zTotal * 0.1; // Scale factor for visual effect
    
    // Calculate velocity for motion blur and trails
    float dt = 0.016; // Approximate frame time
    vec2 z1_next = vec2(
        uR1 * cos(uW1 * (t + dt) + uP1),
        uR1 * sin(uW1 * (t + dt) + uP1)
    );
    vec2 z2_next = vec2(
        uR2 * cos(uW2 * (t + dt) + uP2),
        uR2 * sin(uW2 * (t + dt) + uP2)
    );
    vec2 zTotal_next = z1_next + z2_next;
    
    velocity = vec3((zTotal_next - zTotal) / dt, 0.0);
    
    // Transform to world space
    worldPos = (uTDMat * vec4(localPos, 1.0)).xyz;
    
    // Transform to view space
    viewPos = (uWorldCamMat * vec4(worldPos, 1.0)).xyz;
    
    // Calculate distance from camera for LOD and alpha
    distanceFromCamera = length(viewPos);
    
    // Transform normal
    normal = normalize((uTDMat * vec4(N, 0.0)).xyz);
    
    // Pass through texture coordinates
    texCoord = uv;
    
    // Calculate color based on position and parametric values
    float hue = (zTotal.x + zTotal.y) * uColorFrequency + uTime * 0.1;
    hue = fract(hue);
    
    // HSV to RGB conversion for dynamic colors
    vec3 hsv = vec3(hue, 0.8, 0.9);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
    vec3 rgb = hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
    
    // Add glow intensity
    rgb *= uGlowIntensity;
    
    // Calculate alpha based on trail position and decay
    pointAlpha = Cd.a * uTrailDecay;
    
    // Distance-based alpha for LOD
    float distanceAlpha = 1.0 - smoothstep(5.0, 20.0, distanceFromCamera);
    pointAlpha *= distanceAlpha;
    
    color = vec4(rgb, pointAlpha);
    
    // Final vertex position
    gl_Position = uProjMat * vec4(viewPos, 1.0);
    
    // Point size for point rendering mode
    gl_PointSize = uCurveThickness * 100.0 / distanceFromCamera;
}

#endif

//
// ===================== FRAGMENT SHADER =====================
//

#ifdef FRAGMENT_SHADER

// Fragment shader for parametric curve rendering with effects
in vec3 worldPos;
in vec3 viewPos;
in vec3 normal;
in vec2 texCoord;
in vec4 color;
in float pointAlpha;
in float distanceFromCamera;
in vec3 velocity;

// Uniforms (same as vertex shader)
uniform float uCurveThickness;
uniform float uGlowIntensity;
uniform float uTime;
uniform float uTheta;
uniform float uVolumetricDensity;
uniform float uTrailDecay;

// Output color
out vec4 fragColor;

// Noise function for procedural effects
float noise(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

// Smooth noise function
float smoothNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(mix(noise(i + vec3(0,0,0)), noise(i + vec3(1,0,0)), f.x),
            mix(noise(i + vec3(0,1,0)), noise(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(noise(i + vec3(0,0,1)), noise(i + vec3(1,0,1)), f.x),
            mix(noise(i + vec3(0,1,1)), noise(i + vec3(1,1,1)), f.x), f.y), f.z);
}

// Fractal noise for complex effects
float fractalNoise(vec3 p) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    
    for (int i = 0; i < 4; i++) {
        value += amplitude * smoothNoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// Glow effect calculation
float calculateGlow(vec2 coord, float radius) {
    float dist = length(coord - 0.5);
    return 1.0 - smoothstep(0.0, radius, dist);
}

// Motion blur effect
vec3 calculateMotionBlur(vec3 baseColor, vec3 velocity) {
    float speed = length(velocity);
    vec3 blurColor = baseColor;
    
    if (speed > 0.1) {
        // Add motion streaks
        float streakIntensity = min(speed * 2.0, 1.0);
        vec3 streakColor = baseColor * streakIntensity * 0.5;
        blurColor = mix(baseColor, streakColor, 0.3);
    }
    
    return blurColor;
}

// Volumetric scattering effect
vec3 calculateVolumetricScattering(vec3 worldPos, vec3 color) {
    // Simple volumetric scattering based on position and time
    float density = uVolumetricDensity;
    vec3 scatterPos = worldPos + vec3(uTime * 0.1);
    
    float scatter = fractalNoise(scatterPos * 2.0) * density;
    scatter = pow(scatter, 2.0); // Increase contrast
    
    return color + vec3(scatter * 0.2);
}

void main()
{
    vec3 finalColor = color.rgb;
    float finalAlpha = color.a * pointAlpha;
    
    // Add noise-based texture for curve surface
    vec3 noisePos = worldPos * 5.0 + vec3(uTime * 0.5);
    float surfaceNoise = fractalNoise(noisePos) * 0.2;
    finalColor += vec3(surfaceNoise);
    
    // Calculate glow effect
    float glow = calculateGlow(texCoord, uCurveThickness);
    finalColor *= (1.0 + glow * uGlowIntensity);
    
    // Add motion blur effect
    finalColor = calculateMotionBlur(finalColor, velocity);
    
    // Add volumetric scattering
    if (uVolumetricDensity > 0.0) {
        finalColor = calculateVolumetricScattering(worldPos, finalColor);
    }
    
    // Distance-based effects
    float distanceFade = 1.0 - smoothstep(10.0, 50.0, distanceFromCamera);
    finalAlpha *= distanceFade;
    
    // Time-based pulsing effect
    float pulse = 0.5 + 0.5 * sin(uTime * 2.0 + worldPos.x * 0.5);
    finalColor *= (0.8 + 0.2 * pulse);
    
    // Edge enhancement for curve definition
    vec3 viewNormal = normalize(normal);
    float edgeFactor = 1.0 - abs(dot(viewNormal, normalize(-viewPos)));
    finalColor *= (0.7 + 0.3 * edgeFactor);
    
    // Final color output with proper alpha blending
    fragColor = vec4(finalColor, finalAlpha);
}

#endif

//
// ===================== GEOMETRY SHADER (Optional) =====================
//

#ifdef GEOMETRY_SHADER

// Geometry shader for line/curve expansion
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

// Input from vertex shader
in vec3 worldPos[];
in vec3 viewPos[];
in vec3 normal[];
in vec2 texCoord[];
in vec4 color[];
in float pointAlpha[];
in float distanceFromCamera[];
in vec3 velocity[];

// Output to fragment shader
out vec3 worldPos_g;
out vec3 viewPos_g;
out vec3 normal_g;
out vec2 texCoord_g;
out vec4 color_g;
out float pointAlpha_g;
out float distanceFromCamera_g;
out vec3 velocity_g;

uniform float uCurveThickness;
uniform mat4 uProjMat;

void main()
{
    // Calculate line direction and perpendicular
    vec3 lineDir = normalize(viewPos[1] - viewPos[0]);
    vec3 perpendicular = normalize(cross(lineDir, vec3(0, 0, 1)));
    
    // Calculate thickness based on distance
    float thickness = uCurveThickness / max(distanceFromCamera[0], 1.0);
    
    // Create quad for line segment
    vec3 offset = perpendicular * thickness * 0.5;
    
    // Vertex 0 (bottom-left)
    worldPos_g = worldPos[0];
    viewPos_g = viewPos[0] - offset;
    normal_g = normal[0];
    texCoord_g = vec2(0.0, 0.0);
    color_g = color[0];
    pointAlpha_g = pointAlpha[0];
    distanceFromCamera_g = distanceFromCamera[0];
    velocity_g = velocity[0];
    gl_Position = uProjMat * vec4(viewPos_g, 1.0);
    EmitVertex();
    
    // Vertex 1 (top-left)
    worldPos_g = worldPos[0];
    viewPos_g = viewPos[0] + offset;
    normal_g = normal[0];
    texCoord_g = vec2(0.0, 1.0);
    color_g = color[0];
    pointAlpha_g = pointAlpha[0];
    distanceFromCamera_g = distanceFromCamera[0];
    velocity_g = velocity[0];
    gl_Position = uProjMat * vec4(viewPos_g, 1.0);
    EmitVertex();
    
    // Vertex 2 (bottom-right)
    worldPos_g = worldPos[1];
    viewPos_g = viewPos[1] - offset;
    normal_g = normal[1];
    texCoord_g = vec2(1.0, 0.0);
    color_g = color[1];
    pointAlpha_g = pointAlpha[1];
    distanceFromCamera_g = distanceFromCamera[1];
    velocity_g = velocity[1];
    gl_Position = uProjMat * vec4(viewPos_g, 1.0);
    EmitVertex();
    
    // Vertex 3 (top-right)
    worldPos_g = worldPos[1];
    viewPos_g = viewPos[1] + offset;
    normal_g = normal[1];
    texCoord_g = vec2(1.0, 1.0);
    color_g = color[1];
    pointAlpha_g = pointAlpha[1];
    distanceFromCamera_g = distanceFromCamera[1];
    velocity_g = velocity[1];
    gl_Position = uProjMat * vec4(viewPos_g, 1.0);
    EmitVertex();
    
    EndPrimitive();
}

#endif

//
// ===================== COMPUTE SHADER (For GPU-based curve generation) =====================
//

#ifdef COMPUTE_SHADER

// Compute shader for GPU-based parametric curve generation
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Output buffers
layout(std430, binding = 0) writeonly buffer PositionBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) writeonly buffer VelocityBuffer {
    vec3 velocities[];
};

layout(std430, binding = 2) writeonly buffer ColorBuffer {
    vec4 colors[];
};

// Parametric parameters
uniform float uR1;
uniform float uR2;
uniform float uW1;
uniform float uW2;
uniform float uP1;
uniform float uP2;
uniform float uTime;
uniform float uThetaStep;
uniform int uPointCount;
uniform float uColorFrequency;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    
    if (index >= uPointCount) return;
    
    // Calculate theta for this point
    float theta = float(index) * uThetaStep + uTime;
    
    // Calculate parametric position
    vec2 z1 = vec2(
        uR1 * cos(uW1 * theta + uP1),
        uR1 * sin(uW1 * theta + uP1)
    );
    
    vec2 z2 = vec2(
        uR2 * cos(uW2 * theta + uP2),
        uR2 * sin(uW2 * theta + uP2)
    );
    
    vec2 zTotal = z1 + z2;
    
    // Store position
    positions[index] = vec4(zTotal.x, zTotal.y, 0.0, 1.0);
    
    // Calculate velocity for next frame
    float nextTheta = theta + uThetaStep;
    vec2 z1_next = vec2(
        uR1 * cos(uW1 * nextTheta + uP1),
        uR1 * sin(uW1 * nextTheta + uP1)
    );
    vec2 z2_next = vec2(
        uR2 * cos(uW2 * nextTheta + uP2),
        uR2 * sin(uW2 * nextTheta + uP2)
    );
    vec2 zTotal_next = z1_next + z2_next;
    
    velocities[index] = vec3((zTotal_next - zTotal) / uThetaStep, 0.0);
    
    // Calculate color
    float hue = (zTotal.x + zTotal.y) * uColorFrequency + uTime * 0.1;
    hue = fract(hue);
    
    // HSV to RGB
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(hue + K.xyz) * 6.0 - K.www);
    vec3 rgb = mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), 0.8);
    
    // Alpha based on position in trail
    float alpha = 1.0 - float(index) / float(uPointCount - 1);
    alpha = pow(alpha, 0.5); // Smooth falloff
    
    colors[index] = vec4(rgb, alpha);
}

#endif

//
// ===================== TRAIL EFFECT SHADER =====================
//

// Specialized fragment shader for trail effects
#ifdef TRAIL_FRAGMENT_SHADER

in vec2 texCoord;
in vec4 color;
in float pointAlpha;

uniform float uTrailDecay;
uniform float uTime;
uniform sampler2D uTrailTexture;

out vec4 fragColor;

void main()
{
    // Sample current trail texture
    vec4 currentTrail = texture(uTrailTexture, texCoord);
    
    // Apply decay to existing trail
    currentTrail.rgb *= uTrailDecay;
    currentTrail.a *= uTrailDecay;
    
    // Add new point contribution
    vec3 newContribution = color.rgb * pointAlpha;
    
    // Combine with existing trail
    vec3 finalColor = max(currentTrail.rgb, newContribution);
    float finalAlpha = max(currentTrail.a, pointAlpha);
    
    fragColor = vec4(finalColor, finalAlpha);
}

#endif

//
// ===================== VOLUMETRIC EFFECT SHADER =====================
//

// Fragment shader for volumetric rendering effects
#ifdef VOLUMETRIC_FRAGMENT_SHADER

in vec3 worldPos;
in vec3 viewPos;
in vec2 texCoord;
in vec4 color;

uniform float uVolumetricDensity;
uniform float uTime;
uniform vec3 uCameraPos;

out vec4 fragColor;

// Simple 3D noise for volumetric effects
float noise3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

void main()
{
    // Calculate ray direction
    vec3 rayDir = normalize(worldPos - uCameraPos);
    
    // Sample volumetric noise along ray
    vec3 samplePos = worldPos + vec3(uTime * 0.2);
    float density = noise3D(samplePos) * uVolumetricDensity;
    
    // Add multiple octaves for complexity
    density += noise3D(samplePos * 2.0) * 0.5 * uVolumetricDensity;
    density += noise3D(samplePos * 4.0) * 0.25 * uVolumetricDensity;
    
    // Calculate scattering color
    vec3 scatterColor = color.rgb * density;
    
    // Distance-based attenuation
    float distance = length(viewPos);
    float attenuation = 1.0 / (1.0 + distance * 0.1);
    
    scatterColor *= attenuation;
    
    // Output with alpha blending
    fragColor = vec4(scatterColor, density * color.a);
}

#endif