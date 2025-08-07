"""
TouchDesigner球形渲染和动画脚本
控制3D球形的渲染、变形、旋转和视觉效果

使用方法：
1. 在Geometry COMP中使用球形网格生成函数
2. 在Material或GLSL TOP中使用着色器代码
3. 通过Transform参数控制动画
"""

import math
import random

class SphereRenderer:
    def __init__(self):
        self.sphere_resolution = 64  # 球面细分度
        self.current_frame = 0
        self.animation_time = 0.0
        
        # 渲染参数
        self.render_params = {
            'base_radius': 2.0,
            'deformation_strength': 0.0,
            'rotation_speed': [1.0, 0.5, 0.3],  # X, Y, Z轴旋转速度
            'pulsation_amplitude': 0.2,
            'pulsation_frequency': 1.0,
            'surface_noise_scale': 1.0,
            'surface_noise_strength': 0.0,
            'wireframe_mode': False,
            'material_properties': {
                'metallic': 0.3,
                'roughness': 0.4,
                'emission_strength': 0.2,
                'transparency': 0.8
            }
        }
        
        # 动画状态
        self.animation_state = {
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            'position': [0, 0, 0],
            'current_radius': 2.0
        }
        
        # 几何数据缓存
        self.geometry_cache = {
            'vertices': [],
            'normals': [],
            'uvs': [],
            'indices': []
        }
        
        self.generate_base_sphere()
    
    def generate_base_sphere(self):
        """生成基础球形网格"""
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        # 生成球面顶点
        for i in range(self.sphere_resolution + 1):
            theta = i * math.pi / self.sphere_resolution
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            for j in range(self.sphere_resolution + 1):
                phi = j * 2 * math.pi / self.sphere_resolution
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                
                # 顶点位置
                x = sin_theta * cos_phi
                y = cos_theta
                z = sin_theta * sin_phi
                
                vertices.extend([x, y, z])
                normals.extend([x, y, z])  # 球面法线就是归一化位置
                
                # UV坐标
                u = j / self.sphere_resolution
                v = i / self.sphere_resolution
                uvs.extend([u, v])
        
        # 生成三角形索引
        for i in range(self.sphere_resolution):
            for j in range(self.sphere_resolution):
                # 第一个三角形
                first = i * (self.sphere_resolution + 1) + j
                second = first + self.sphere_resolution + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        self.geometry_cache['vertices'] = vertices
        self.geometry_cache['normals'] = normals
        self.geometry_cache['uvs'] = uvs
        self.geometry_cache['indices'] = indices
    
    def apply_deformation(self, vertices, deformation_params):
        """应用球面变形"""
        deformed_vertices = []
        time = self.animation_time
        
        for i in range(0, len(vertices), 3):
            x, y, z = vertices[i], vertices[i+1], vertices[i+2]
            
            # 基础半径
            radius = math.sqrt(x*x + y*y + z*z)
            if radius == 0:
                deformed_vertices.extend([x, y, z])
                continue
            
            # 归一化方向
            nx, ny, nz = x/radius, y/radius, z/radius
            
            # 应用各种变形
            new_radius = self.render_params['base_radius']
            
            # 脉动效果
            pulse_amp = self.render_params['pulsation_amplitude']
            pulse_freq = self.render_params['pulsation_frequency']
            pulse_factor = 1 + pulse_amp * math.sin(time * pulse_freq * 2 * math.pi)
            new_radius *= pulse_factor
            
            # 表面噪声
            noise_scale = self.render_params['surface_noise_scale']
            noise_strength = self.render_params['surface_noise_strength']
            if noise_strength > 0:
                noise_x = math.sin(nx * noise_scale * 5 + time * 0.5)
                noise_y = math.sin(ny * noise_scale * 7 + time * 0.3)
                noise_z = math.sin(nz * noise_scale * 6 + time * 0.7)
                noise_factor = 1 + noise_strength * (noise_x + noise_y + noise_z) / 3
                new_radius *= noise_factor
            
            # 手势变形
            deform_strength = self.render_params['deformation_strength']
            if deform_strength > 0:
                # 基于坐标的非均匀缩放
                deform_x = 1 + deform_strength * math.sin(nx * math.pi * 2)
                deform_y = 1 + deform_strength * math.cos(ny * math.pi * 3)
                deform_z = 1 + deform_strength * math.sin(nz * math.pi * 1.5)
                
                new_x = nx * new_radius * deform_x
                new_y = ny * new_radius * deform_y
                new_z = nz * new_radius * deform_z
            else:
                new_x = nx * new_radius
                new_y = ny * new_radius
                new_z = nz * new_radius
            
            deformed_vertices.extend([new_x, new_y, new_z])
        
        return deformed_vertices
    
    def update_animation(self, dt, gesture_params=None):
        """更新动画状态"""
        self.current_frame += 1
        self.animation_time += dt
        
        # 更新渲染参数
        if gesture_params:
            self.render_params.update({
                'base_radius': gesture_params.get('radius', 2.0),
                'deformation_strength': gesture_params.get('deformation', 0.0),
                'pulsation_frequency': gesture_params.get('pulsation', 1.0),
                'surface_noise_strength': gesture_params.get('surface_noise', 0.0)
            })
            
            # 更新旋转速度
            rotation_speed = gesture_params.get('rotation_speed', 1.0)
            self.render_params['rotation_speed'] = [
                rotation_speed * 1.0,
                rotation_speed * 0.7,
                rotation_speed * 0.5
            ]
        
        # 更新旋转
        rot_speed = self.render_params['rotation_speed']
        self.animation_state['rotation'][0] += rot_speed[0] * dt * 60
        self.animation_state['rotation'][1] += rot_speed[1] * dt * 60
        self.animation_state['rotation'][2] += rot_speed[2] * dt * 60
        
        # 保持旋转角度在0-360范围内
        for i in range(3):
            self.animation_state['rotation'][i] %= 360
    
    def get_transformed_geometry(self):
        """获取变形后的几何数据"""
        base_vertices = self.geometry_cache['vertices']
        deformed_vertices = self.apply_deformation(base_vertices, self.render_params)
        
        return {
            'vertices': deformed_vertices,
            'normals': self.calculate_normals(deformed_vertices),
            'uvs': self.geometry_cache['uvs'],
            'indices': self.geometry_cache['indices']
        }
    
    def calculate_normals(self, vertices):
        """重新计算法线"""
        normals = []
        indices = self.geometry_cache['indices']
        
        # 初始化法线数组
        vertex_count = len(vertices) // 3
        normal_accumulator = [[0, 0, 0] for _ in range(vertex_count)]
        
        # 计算面法线并累加到顶点
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
            
            # 获取三角形顶点
            v0 = vertices[i0*3:i0*3+3]
            v1 = vertices[i1*3:i1*3+3]
            v2 = vertices[i2*3:i2*3+3]
            
            # 计算边向量
            edge1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]]
            edge2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]]
            
            # 叉积得到面法线
            normal = [
                edge1[1]*edge2[2] - edge1[2]*edge2[1],
                edge1[2]*edge2[0] - edge1[0]*edge2[2],
                edge1[0]*edge2[1] - edge1[1]*edge2[0]
            ]
            
            # 累加到顶点法线
            for idx in [i0, i1, i2]:
                normal_accumulator[idx][0] += normal[0]
                normal_accumulator[idx][1] += normal[1]
                normal_accumulator[idx][2] += normal[2]
        
        # 归一化顶点法线
        for normal in normal_accumulator:
            length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            if length > 0:
                normal[0] /= length
                normal[1] /= length
                normal[2] /= length
            normals.extend(normal)
        
        return normals
    
    def get_material_params(self):
        """获取材质参数"""
        time = self.animation_time
        base_props = self.render_params['material_properties']
        
        # 动态调整发光强度
        emission_base = base_props['emission_strength']
        emission_variation = 0.1 * math.sin(time * 2 * math.pi)
        current_emission = emission_base + emission_variation
        
        # 动态调整透明度
        transparency_base = base_props['transparency']
        transparency_variation = 0.1 * math.sin(time * 1.5 * math.pi)
        current_transparency = max(0.1, transparency_base + transparency_variation)
        
        return {
            'metallic': base_props['metallic'],
            'roughness': base_props['roughness'],
            'emission_strength': current_emission,
            'transparency': current_transparency,
            'base_color': self.get_dynamic_color()
        }
    
    def get_dynamic_color(self):
        """获取动态颜色"""
        time = self.animation_time
        
        # 基于时间的彩虹色变化
        hue = (time * 0.1) % 1.0
        saturation = 0.8
        brightness = 0.9
        
        return self.hsv_to_rgb(hue, saturation, brightness)
    
    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB"""
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0:
            return [v, t, p]
        elif i == 1:
            return [q, v, p]
        elif i == 2:
            return [p, v, t]
        elif i == 3:
            return [p, q, v]
        elif i == 4:
            return [t, p, v]
        else:
            return [v, p, q]
    
    def generate_shader_uniforms(self):
        """生成着色器uniform参数"""
        return {
            'uTime': self.animation_time,
            'uBaseRadius': self.render_params['base_radius'],
            'uDeformationStrength': self.render_params['deformation_strength'],
            'uPulsationAmplitude': self.render_params['pulsation_amplitude'],
            'uPulsationFrequency': self.render_params['pulsation_frequency'],
            'uNoiseScale': self.render_params['surface_noise_scale'],
            'uNoiseStrength': self.render_params['surface_noise_strength'],
            'uRotation': self.animation_state['rotation'],
            'uMetallic': self.render_params['material_properties']['metallic'],
            'uRoughness': self.render_params['material_properties']['roughness'],
            'uEmissionStrength': self.render_params['material_properties']['emission_strength'],
            'uTransparency': self.render_params['material_properties']['transparency']
        }

# 全局渲染器实例
if not hasattr(op, 'sphere_renderer'):
    op.sphere_renderer = SphereRenderer()

# TouchDesigner接口函数
def update_sphere_renderer(gesture_params, dt=0.016):
    """更新球形渲染器"""
    renderer = op.sphere_renderer
    renderer.update_animation(dt, gesture_params)

def get_sphere_geometry():
    """获取球形几何数据"""
    renderer = op.sphere_renderer
    return renderer.get_transformed_geometry()

def get_sphere_material_params():
    """获取球形材质参数"""
    renderer = op.sphere_renderer
    return renderer.get_material_params()

def get_shader_uniforms():
    """获取着色器uniform参数"""
    renderer = op.sphere_renderer
    return renderer.generate_shader_uniforms()

def generate_sphere_vertex_shader():
    """生成球形顶点着色器"""
    return '''
    #version 330 core
    
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;
    
    uniform mat4 uModelMatrix;
    uniform mat4 uViewMatrix;
    uniform mat4 uProjectionMatrix;
    uniform float uTime;
    uniform float uDeformationStrength;
    uniform float uPulsationAmplitude;
    uniform float uPulsationFrequency;
    uniform float uNoiseScale;
    uniform float uNoiseStrength;
    
    out vec3 vWorldPos;
    out vec3 vNormal;
    out vec2 vTexCoord;
    out float vDeformation;
    
    // 简单噪声函数
    float noise(vec3 p) {
        return sin(p.x * uNoiseScale) * cos(p.y * uNoiseScale) * sin(p.z * uNoiseScale);
    }
    
    void main() {
        vec3 pos = aPos;
        vec3 normal = aNormal;
        
        // 脉动效果
        float pulsation = 1.0 + uPulsationAmplitude * sin(uTime * uPulsationFrequency * 6.28318);
        pos *= pulsation;
        
        // 表面噪声
        if (uNoiseStrength > 0.0) {
            float noiseValue = noise(pos + uTime * 0.5);
            pos += normal * noiseValue * uNoiseStrength;
        }
        
        // 变形效果
        if (uDeformationStrength > 0.0) {
            float deformX = 1.0 + uDeformationStrength * sin(pos.x * 3.14159 * 2.0);
            float deformY = 1.0 + uDeformationStrength * cos(pos.y * 3.14159 * 3.0);
            float deformZ = 1.0 + uDeformationStrength * sin(pos.z * 3.14159 * 1.5);
            
            pos.x *= deformX;
            pos.y *= deformY;
            pos.z *= deformZ;
        }
        
        vWorldPos = (uModelMatrix * vec4(pos, 1.0)).xyz;
        vNormal = mat3(uModelMatrix) * normal;
        vTexCoord = aTexCoord;
        vDeformation = length(pos - aPos);
        
        gl_Position = uProjectionMatrix * uViewMatrix * vec4(vWorldPos, 1.0);
    }
    '''

def generate_sphere_fragment_shader():
    """生成球形片段着色器"""
    return '''
    #version 330 core
    
    in vec3 vWorldPos;
    in vec3 vNormal;
    in vec2 vTexCoord;
    in float vDeformation;
    
    uniform float uTime;
    uniform float uMetallic;
    uniform float uRoughness;
    uniform float uEmissionStrength;
    uniform float uTransparency;
    uniform vec3 uCameraPos;
    uniform vec3 uLightPos;
    
    out vec4 FragColor;
    
    // 简化的PBR着色
    vec3 calculatePBR(vec3 albedo, vec3 normal, vec3 viewDir, vec3 lightDir) {
        float NdotL = max(dot(normal, lightDir), 0.0);
        float NdotV = max(dot(normal, viewDir), 0.0);
        
        // 漫反射
        vec3 diffuse = albedo * NdotL;
        
        // 镜面反射
        vec3 halfDir = normalize(lightDir + viewDir);
        float NdotH = max(dot(normal, halfDir), 0.0);
        float spec = pow(NdotH, (1.0 - uRoughness) * 128.0);
        vec3 specular = vec3(spec) * uMetallic;
        
        return diffuse + specular;
    }
    
    void main() {
        vec3 normal = normalize(vNormal);
        vec3 viewDir = normalize(uCameraPos - vWorldPos);
        vec3 lightDir = normalize(uLightPos - vWorldPos);
        
        // 动态基础颜色
        float hue = uTime * 0.1 + vTexCoord.x * 2.0;
        vec3 baseColor = vec3(
            0.5 + 0.5 * sin(hue * 6.28318),
            0.5 + 0.5 * sin(hue * 6.28318 + 2.094),
            0.5 + 0.5 * sin(hue * 6.28318 + 4.188)
        );
        
        // 基于变形的颜色调制
        baseColor = mix(baseColor, vec3(1.0, 0.3, 0.1), vDeformation * 0.5);
        
        // PBR着色
        vec3 color = calculatePBR(baseColor, normal, viewDir, lightDir);
        
        // 发光效果
        vec3 emission = baseColor * uEmissionStrength;
        color += emission;
        
        // 边缘光效果
        float fresnel = 1.0 - dot(viewDir, normal);
        color += vec3(0.2, 0.5, 1.0) * fresnel * 0.3;
        
        FragColor = vec4(color, uTransparency);
    }
    '''