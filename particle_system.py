"""
TouchDesigner粒子系统控制脚本
基于手势数据控制3D球形粒子效果

使用方法：
1. 在Particle GPU TOP或GLSL TOP中使用此脚本的函数
2. 通过Text DAT连接参数数据
3. 在Geometry COMP中使用生成的粒子数据
"""

import math
import random

class ParticleSystem:
    def __init__(self):
        self.particles = []
        self.max_particles = 2000
        self.sphere_center = [0, 0, 0]
        self.current_frame = 0
        
        # 粒子属性
        self.particle_properties = {
            'position': [],
            'velocity': [],
            'life': [],
            'size': [],
            'color': [],
            'rotation': []
        }
        
        # 系统参数
        self.system_params = {
            'emission_rate': 100,
            'particle_life': 5.0,
            'sphere_radius': 2.0,
            'velocity_scale': 1.0,
            'size_scale': 1.0,
            'color_palette': 'rainbow',
            'turbulence_strength': 0.5,
            'gravity': [0, -0.1, 0],
            'sphere_deformation': 0.0,
            'pulsation_frequency': 1.0,
            'rotation_speed': 1.0
        }
        
        self.initialize_particles()
    
    def initialize_particles(self):
        """初始化粒子系统"""
        self.particles = []
        for i in range(self.max_particles):
            particle = self.create_particle()
            particle['life'] = random.uniform(0, self.system_params['particle_life'])
            self.particles.append(particle)
    
    def create_particle(self):
        """创建新粒子"""
        # 在球面上生成随机位置
        phi = random.uniform(0, 2 * math.pi)
        theta = random.uniform(0, math.pi)
        radius = self.system_params['sphere_radius']
        
        # 球坐标转笛卡尔坐标
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        # 添加变形
        deformation = self.system_params['sphere_deformation']
        if deformation > 0:
            # 椭球变形
            x *= (1 + deformation * math.sin(phi * 2))
            y *= (1 + deformation * math.cos(theta * 3))
            z *= (1 + deformation * math.sin(phi + theta))
        
        # 初始速度（向外扩散）
        speed = random.uniform(0.5, 1.5) * self.system_params['velocity_scale']
        vel_x = (x / radius) * speed * random.uniform(0.5, 1.0)
        vel_y = (y / radius) * speed * random.uniform(0.5, 1.0)
        vel_z = (z / radius) * speed * random.uniform(0.5, 1.0)
        
        return {
            'position': [x, y, z],
            'velocity': [vel_x, vel_y, vel_z],
            'life': self.system_params['particle_life'],
            'max_life': self.system_params['particle_life'],
            'size': random.uniform(0.5, 2.0) * self.system_params['size_scale'],
            'color_offset': random.uniform(0, 1),
            'rotation': random.uniform(0, 360),
            'rotation_speed': random.uniform(-2, 2) * self.system_params['rotation_speed']
        }
    
    def update_particles(self, dt):
        """更新粒子状态"""
        self.current_frame += 1
        time = self.current_frame * dt
        
        for particle in self.particles:
            # 更新生命周期
            particle['life'] -= dt
            
            if particle['life'] <= 0:
                # 重新生成粒子
                new_particle = self.create_particle()
                particle.update(new_particle)
                continue
            
            # 更新位置
            pos = particle['position']
            vel = particle['velocity']
            
            pos[0] += vel[0] * dt
            pos[1] += vel[1] * dt
            pos[2] += vel[2] * dt
            
            # 应用重力
            gravity = self.system_params['gravity']
            vel[0] += gravity[0] * dt
            vel[1] += gravity[1] * dt
            vel[2] += gravity[2] * dt
            
            # 应用湍流
            turbulence = self.system_params['turbulence_strength']
            if turbulence > 0:
                noise_x = math.sin(time * 0.5 + pos[0] * 0.1) * turbulence
                noise_y = math.cos(time * 0.7 + pos[1] * 0.1) * turbulence
                noise_z = math.sin(time * 0.3 + pos[2] * 0.1) * turbulence
                
                vel[0] += noise_x * dt
                vel[1] += noise_y * dt
                vel[2] += noise_z * dt
            
            # 球形约束力（吸引回球面）
            center_dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            if center_dist > self.system_params['sphere_radius'] * 2:
                attract_strength = 2.0
                attract_x = -pos[0] / center_dist * attract_strength
                attract_y = -pos[1] / center_dist * attract_strength
                attract_z = -pos[2] / center_dist * attract_strength
                
                vel[0] += attract_x * dt
                vel[1] += attract_y * dt
                vel[2] += attract_z * dt
            
            # 应用阻尼
            damping = 0.98
            vel[0] *= damping
            vel[1] *= damping
            vel[2] *= damping
            
            # 更新旋转
            particle['rotation'] += particle['rotation_speed'] * dt * 60
            
            # 脉动效果
            pulsation = self.system_params['pulsation_frequency']
            if pulsation > 0:
                pulse_factor = 1 + 0.3 * math.sin(time * pulsation * 2 * math.pi)
                particle['current_size'] = particle['size'] * pulse_factor
            else:
                particle['current_size'] = particle['size']
    
    def get_particle_positions(self):
        """获取所有粒子位置数据"""
        positions = []
        for particle in self.particles:
            if particle['life'] > 0:
                positions.extend(particle['position'])
        return positions
    
    def get_particle_colors(self):
        """获取粒子颜色数据"""
        colors = []
        time = self.current_frame * 0.016  # 假设60fps
        
        for particle in self.particles:
            if particle['life'] > 0:
                # 基于生命周期和位置的颜色
                life_ratio = particle['life'] / particle['max_life']
                color_offset = particle['color_offset']
                
                if self.system_params['color_palette'] == 'rainbow':
                    hue = (time * 0.1 + color_offset) % 1.0
                    r, g, b = self.hsv_to_rgb(hue, 0.8, life_ratio)
                elif self.system_params['color_palette'] == 'fire':
                    # 火焰色彩
                    r = 1.0
                    g = life_ratio * 0.8
                    b = life_ratio * 0.3
                else:
                    # 默认蓝色调
                    r = 0.3
                    g = 0.6
                    b = life_ratio
                
                colors.extend([r, g, b, life_ratio])  # RGBA
        
        return colors
    
    def get_particle_sizes(self):
        """获取粒子大小数据"""
        sizes = []
        for particle in self.particles:
            if particle['life'] > 0:
                sizes.append(particle.get('current_size', particle['size']))
        return sizes
    
    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB"""
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q
    
    def update_system_params(self, new_params):
        """更新系统参数"""
        self.system_params.update(new_params)
    
    def get_geometry_data(self):
        """获取几何数据用于Geometry COMP"""
        geometry_data = {
            'points': [],
            'normals': [],
            'uvs': [],
            'colors': []
        }
        
        for particle in self.particles:
            if particle['life'] > 0:
                pos = particle['position']
                geometry_data['points'].extend(pos)
                
                # 法线（指向中心）
                length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                if length > 0:
                    normal = [pos[0]/length, pos[1]/length, pos[2]/length]
                else:
                    normal = [0, 1, 0]
                geometry_data['normals'].extend(normal)
                
                # UV坐标（基于球面映射）
                phi = math.atan2(pos[1], pos[0])
                theta = math.acos(pos[2] / length) if length > 0 else 0
                u = (phi + math.pi) / (2 * math.pi)
                v = theta / math.pi
                geometry_data['uvs'].extend([u, v])
        
        return geometry_data

# 全局粒子系统实例
if not hasattr(op, 'particle_system'):
    op.particle_system = ParticleSystem()

# TouchDesigner接口函数
def update_particle_system(gesture_params, dt=0.016):
    """更新粒子系统"""
    system = op.particle_system
    
    # 更新系统参数
    if gesture_params:
        system.update_system_params({
            'emission_rate': gesture_params.get('emission_rate', 100),
            'sphere_radius': gesture_params.get('radius', 2.0),
            'velocity_scale': gesture_params.get('velocity', 1.0),
            'size_scale': gesture_params.get('size', 1.0),
            'turbulence_strength': gesture_params.get('turbulence', 0.5),
            'sphere_deformation': gesture_params.get('deformation', 0.0),
            'pulsation_frequency': gesture_params.get('pulsation', 1.0)
        })
    
    # 更新粒子状态
    system.update_particles(dt)

def get_particle_data_for_gpu():
    """获取GPU粒子数据"""
    system = op.particle_system
    return {
        'positions': system.get_particle_positions(),
        'colors': system.get_particle_colors(),
        'sizes': system.get_particle_sizes()
    }

def get_active_particle_count():
    """获取活跃粒子数量"""
    system = op.particle_system
    return sum(1 for p in system.particles if p['life'] > 0)

# GLSL着色器辅助函数（在GLSL TOP中使用）
def generate_vertex_shader():
    """生成顶点着色器代码"""
    return '''
    #version 330 core
    
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec4 aColor;
    layout (location = 2) in float aSize;
    
    uniform mat4 uProjectionMatrix;
    uniform mat4 uViewMatrix;
    uniform float uTime;
    
    out vec4 vColor;
    out float vSize;
    
    void main() {
        vColor = aColor;
        vSize = aSize;
        
        vec3 pos = aPos;
        
        // 添加轻微的噪声扰动
        pos.x += sin(uTime + aPos.y * 0.1) * 0.1;
        pos.y += cos(uTime + aPos.z * 0.1) * 0.1;
        pos.z += sin(uTime + aPos.x * 0.1) * 0.1;
        
        gl_Position = uProjectionMatrix * uViewMatrix * vec4(pos, 1.0);
        gl_PointSize = vSize * 10.0;
    }
    '''

def generate_fragment_shader():
    """生成片段着色器代码"""
    return '''
    #version 330 core
    
    in vec4 vColor;
    in float vSize;
    
    out vec4 FragColor;
    
    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        
        // 创建圆形粒子
        if (dist > 0.5) {
            discard;
        }
        
        // 软边效果
        float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
        
        FragColor = vec4(vColor.rgb, vColor.a * alpha);
    }
    '''