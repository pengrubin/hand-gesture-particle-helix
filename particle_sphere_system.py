"""
粒子球形系统
基于手势数据控制的3D粒子和球形效果系统
"""

import numpy as np
import math
import random
import time
from typing import List, Dict, Tuple

class ParticleSystem:
    def __init__(self, max_particles=2000):
        self.max_particles = max_particles
        self.particles = []
        self.current_time = 0.0
        self.dt = 1.0 / 60.0  # 60 FPS
        
        # 系统参数
        self.params = {
            'emission_rate': 100,
            'particle_life': 5.0,
            'base_radius': 2.0,
            'velocity_scale': 1.0,
            'size_scale': 1.0,
            'turbulence_strength': 0.0,
            'color_hue': 0.5,
            'color_saturation': 0.8,
            'gravity': [0, -0.5, 0],
            'attraction_strength': 1.0,
            'shape_mode': 'sine_wave',  # 新增：形状模式
            'wave_amplitude': 2.0,      # 波浪幅度
            'wave_frequency': 1.0,      # 波浪频率
            'line_length': 6.0,         # 线条长度
            'wave_speed': 1.0           # 波浪动画速度
        }
        
        # 可用的形状模式
        self.available_shapes = [
            'sine_wave',        # 正弦波
            'cosine_wave',      # 余弦波
            'double_wave',      # 双重波浪
            'spiral_line',      # 螺旋线
            'zigzag_line',      # 锯齿波
            'heart_curve',      # 心形曲线
            'infinity_curve',   # 无穷符号
            'helix_3d',         # 3D螺旋
            'multiple_lines',   # 多条平行线
            'double_helix',     # 双螺旋结构 ⭐ 新增
            'triple_helix',     # 三重螺旋 ⭐ 新增
            'dna_structure',    # DNA双螺旋 ⭐ 新增
            'twisted_ribbon',   # 扭转带状结构 ⭐ 新增
            'braided_lines',    # 编织线条 ⭐ 新增
            'spiral_tower',     # 螺旋塔 ⭐ 新增
            'coil_spring',      # 弹簧线圈 ⭐ 新增
            'tornado_helix',    # 龙卷风螺旋 ⭐ 新增
            'galaxy_spiral'     # 银河螺旋 ⭐ 新增
        ]
        self.current_shape_index = 0
        
        self.initialize_particles()
    
    def initialize_particles(self):
        """初始化粒子"""
        self.particles = []
        for i in range(self.max_particles):
            particle = self.create_particle()
            # 随机初始生命周期，避免所有粒子同时重生
            particle['life'] = random.uniform(0, self.params['particle_life'])
            self.particles.append(particle)
    
    def create_particle(self):
        """创建新粒子"""
        # 根据形状模式获取位置
        x, y, z = self.get_shape_position()
        
        # 添加随机偏移
        offset = 0.3
        x += random.uniform(-offset, offset)
        y += random.uniform(-offset, offset) 
        z += random.uniform(-offset, offset)
        
        # 初始速度（沿形状切线方向或随机方向）
        speed = random.uniform(0.3, 1.0) * self.params['velocity_scale']
        
        # 根据形状计算初始速度方向
        vel_x, vel_y, vel_z = self.get_initial_velocity(x, y, z, speed)
        
        return {
            'position': [x, y, z],
            'velocity': [vel_x, vel_y, vel_z],
            'life': self.params['particle_life'],
            'max_life': self.params['particle_life'],
            'size': random.uniform(0.5, 2.0) * self.params['size_scale'],
            'color_offset': random.uniform(0, 1),
            'birth_time': self.current_time
        }
    
    def get_shape_position(self):
        """根据当前形状模式获取粒子初始位置"""
        shape_mode = self.params['shape_mode']
        t = random.uniform(0, 1)  # 参数 t 在 [0, 1] 范围内
        time_offset = self.current_time * self.params['wave_speed']
        
        if shape_mode == 'sine_wave':
            # 正弦波
            x = (t - 0.5) * self.params['line_length']
            y = self.params['wave_amplitude'] * math.sin(t * 2 * math.pi * self.params['wave_frequency'] + time_offset)
            z = 0
            
        elif shape_mode == 'cosine_wave':
            # 余弦波  
            x = (t - 0.5) * self.params['line_length']
            y = self.params['wave_amplitude'] * math.cos(t * 2 * math.pi * self.params['wave_frequency'] + time_offset)
            z = 0
            
        elif shape_mode == 'double_wave':
            # 双重波浪
            x = (t - 0.5) * self.params['line_length']
            y = self.params['wave_amplitude'] * (
                math.sin(t * 2 * math.pi * self.params['wave_frequency'] + time_offset) +
                0.5 * math.sin(t * 4 * math.pi * self.params['wave_frequency'] + time_offset * 1.5)
            )
            z = 0
            
        elif shape_mode == 'spiral_line':
            # 螺旋线
            angle = t * 4 * math.pi + time_offset
            radius = t * 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = (t - 0.5) * self.params['line_length'] * 0.5
            
        elif shape_mode == 'zigzag_line':
            # 锯齿波
            x = (t - 0.5) * self.params['line_length']
            wave_t = (t * self.params['wave_frequency'] + time_offset * 0.1) % 1.0
            y = self.params['wave_amplitude'] * (2 * abs(2 * wave_t - 1) - 1)
            z = 0
            
        elif shape_mode == 'heart_curve':
            # 心形曲线
            angle = t * 2 * math.pi + time_offset * 0.5
            scale = self.params['wave_amplitude'] * 0.5
            x = scale * (16 * math.sin(angle)**3) * 0.1
            y = scale * (13 * math.cos(angle) - 5 * math.cos(2*angle) - 2 * math.cos(3*angle) - math.cos(4*angle)) * 0.1
            z = 0
            
        elif shape_mode == 'infinity_curve':
            # 无穷符号 (∞)
            angle = t * 2 * math.pi + time_offset * 0.5
            scale = self.params['wave_amplitude']
            x = scale * math.sin(angle) / (1 + math.cos(angle)**2)
            y = scale * math.sin(angle) * math.cos(angle) / (1 + math.cos(angle)**2)
            z = 0
            
        elif shape_mode == 'helix_3d':
            # 3D螺旋
            angle = t * 6 * math.pi + time_offset
            radius = self.params['wave_amplitude'] * 0.8
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = (t - 0.5) * self.params['line_length']
            
        elif shape_mode == 'multiple_lines':
            # 多条平行线
            line_count = 5
            line_index = random.randint(0, line_count - 1)
            offset = (line_index - line_count // 2) * 0.8
            
            x = (t - 0.5) * self.params['line_length']
            y = offset + self.params['wave_amplitude'] * 0.3 * math.sin(t * 3 * math.pi + time_offset + line_index)
            z = 0
            
        elif shape_mode == 'double_helix':
            # DNA双螺旋结构
            angle = t * 6 * math.pi + time_offset
            radius = self.params['wave_amplitude'] * 0.8
            height = (t - 0.5) * self.params['line_length']
            
            # 选择第一条或第二条螺旋线
            helix_choice = random.choice([0, 1])
            if helix_choice == 0:
                # 第一条螺旋
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = height
            else:
                # 第二条螺旋 (相位差π)
                x = radius * math.cos(angle + math.pi)
                y = radius * math.sin(angle + math.pi)
                z = height
                
        elif shape_mode == 'triple_helix':
            # 三重螺旋结构
            angle = t * 8 * math.pi + time_offset
            radius = self.params['wave_amplitude'] * 0.9
            height = (t - 0.5) * self.params['line_length']
            
            # 选择三条螺旋线中的一条
            helix_choice = random.choice([0, 1, 2])
            phase_offset = helix_choice * 2 * math.pi / 3
            
            x = radius * math.cos(angle + phase_offset)
            y = radius * math.sin(angle + phase_offset)
            z = height
            
        elif shape_mode == 'dna_structure':
            # 真实DNA双螺旋结构（带连接桥）
            angle = t * 4 * math.pi + time_offset
            radius = self.params['wave_amplitude'] * 0.7
            height = (t - 0.5) * self.params['line_length']
            
            # 70%概率生成螺旋骨架，30%概率生成连接桥
            structure_type = random.choices([0, 1, 2], weights=[35, 35, 30])[0]
            
            if structure_type == 0:
                # 第一条螺旋骨架
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = height
            elif structure_type == 1:
                # 第二条螺旋骨架
                x = radius * math.cos(angle + math.pi)
                y = radius * math.sin(angle + math.pi)
                z = height
            else:
                # 连接桥（碱基对）
                bridge_t = random.uniform(0.2, 0.8)
                x1 = radius * math.cos(angle)
                y1 = radius * math.sin(angle)
                x2 = radius * math.cos(angle + math.pi)
                y2 = radius * math.sin(angle + math.pi)
                
                x = x1 + bridge_t * (x2 - x1)
                y = y1 + bridge_t * (y2 - y1)
                z = height
                
        elif shape_mode == 'twisted_ribbon':
            # 扭转带状结构
            angle = t * 4 * math.pi + time_offset
            width = self.params['wave_amplitude'] * 0.4
            height = (t - 0.5) * self.params['line_length']
            
            # 带状结构的宽度分布
            ribbon_pos = random.uniform(-1, 1)
            local_radius = width * abs(ribbon_pos)
            
            x = local_radius * math.cos(angle + ribbon_pos * math.pi)
            y = local_radius * math.sin(angle + ribbon_pos * math.pi)
            z = height
            
        elif shape_mode == 'braided_lines':
            # 编织线条（3条线编织）
            angle = t * 6 * math.pi + time_offset
            braid_radius = self.params['wave_amplitude'] * 0.6
            height = (t - 0.5) * self.params['line_length']
            
            # 选择3条编织线中的一条
            braid_choice = random.choice([0, 1, 2])
            phase_offset = braid_choice * 2 * math.pi / 3
            
            # 编织效果：半径随高度变化
            radius_mod = braid_radius * (1 + 0.3 * math.sin(t * 12 * math.pi + phase_offset))
            
            x = radius_mod * math.cos(angle + phase_offset)
            y = radius_mod * math.sin(angle + phase_offset)
            z = height
            
        elif shape_mode == 'spiral_tower':
            # 螺旋塔（多层螺旋）
            layers = 4
            layer = random.randint(0, layers - 1)
            layer_height = self.params['line_length'] / layers
            
            # 每层的参数
            layer_t = (t * layers - layer) % 1.0
            if layer_t < 0:
                layer_t += 1.0
            
            angle = layer_t * 8 * math.pi + time_offset + layer * math.pi / 2
            radius = self.params['wave_amplitude'] * (1 - layer * 0.15)  # 向上递减
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = (layer + layer_t - layers/2) * layer_height
            
        elif shape_mode == 'coil_spring':
            # 弹簧线圈
            angle = t * 12 * math.pi + time_offset
            radius = self.params['wave_amplitude'] * 0.8
            height = (t - 0.5) * self.params['line_length']
            
            # 弹簧压缩效果
            compression = 1 + 0.5 * math.sin(time_offset * 2)
            compressed_height = height / compression
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = compressed_height
            
        elif shape_mode == 'tornado_helix':
            # 龙卷风螺旋（半径随高度变化）
            angle = t * 8 * math.pi + time_offset
            height = (t - 0.5) * self.params['line_length']
            
            # 龙卷风形状：下大上小
            height_ratio = abs(height) / (self.params['line_length'] / 2)
            radius = self.params['wave_amplitude'] * (1.5 - height_ratio)
            radius = max(radius, 0.1)  # 保证最小半径
            
            # 添加扰动
            radius += 0.2 * math.sin(t * 20 * math.pi + time_offset)
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height
            
        elif shape_mode == 'galaxy_spiral':
            # 银河螺旋（对数螺旋）
            # 使用对数螺旋公式: r = a * e^(b*θ)
            max_angle = 6 * math.pi
            angle = t * max_angle + time_offset * 0.5
            
            a = 0.3
            b = 0.2
            radius = a * math.exp(b * angle) * self.params['wave_amplitude']
            
            # 多个螺旋臂
            arm_count = 3
            arm_choice = random.randint(0, arm_count - 1)
            arm_offset = arm_choice * 2 * math.pi / arm_count
            
            x = radius * math.cos(angle + arm_offset)
            y = radius * math.sin(angle + arm_offset)
            z = (t - 0.5) * self.params['line_length'] * 0.3  # 较扁的结构
            
        else:
            # 默认：简单直线
            x = (t - 0.5) * self.params['line_length']
            y = 0
            z = 0
            
        return x, y, z
    
    def get_initial_velocity(self, x, y, z, speed):
        """获取粒子初始速度"""
        shape_mode = self.params['shape_mode']
        
        if shape_mode in ['sine_wave', 'cosine_wave', 'double_wave', 'zigzag_line']:
            # 线性波浪：主要沿X轴方向移动
            vel_x = speed * random.uniform(0.3, 1.0)
            vel_y = speed * random.uniform(-0.5, 0.5)
            vel_z = speed * random.uniform(-0.3, 0.3)
            
        elif shape_mode in ['spiral_line', 'helix_3d']:
            # 螺旋：切线方向
            vel_x = speed * random.uniform(-0.8, 0.8)
            vel_y = speed * random.uniform(-0.8, 0.8) 
            vel_z = speed * random.uniform(-0.5, 0.5)
            
        elif shape_mode in ['heart_curve', 'infinity_curve']:
            # 闭合曲线：切线方向
            vel_x = speed * random.uniform(-1.0, 1.0)
            vel_y = speed * random.uniform(-1.0, 1.0)
            vel_z = speed * random.uniform(-0.2, 0.2)
            
        else:
            # 默认：随机方向
            vel_x = speed * random.uniform(-0.8, 0.8)
            vel_y = speed * random.uniform(-0.8, 0.8)
            vel_z = speed * random.uniform(-0.8, 0.8)
        
        return vel_x, vel_y, vel_z
    
    def cycle_shape_mode(self):
        """切换到下一个形状模式"""
        self.current_shape_index = (self.current_shape_index + 1) % len(self.available_shapes)
        new_shape = self.available_shapes[self.current_shape_index]
        self.params['shape_mode'] = new_shape
        print(f"切换到形状: {new_shape}")
        return new_shape
    
    def update(self, dt):
        """更新粒子系统"""
        self.current_time += dt
        
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
            gravity = self.params['gravity']
            vel[0] += gravity[0] * dt
            vel[1] += gravity[1] * dt
            vel[2] += gravity[2] * dt
            
            # 应用湍流
            turbulence = self.params['turbulence_strength']
            if turbulence > 0:
                noise_x = math.sin(self.current_time * 0.5 + pos[0] * 0.1) * turbulence
                noise_y = math.cos(self.current_time * 0.7 + pos[1] * 0.1) * turbulence
                noise_z = math.sin(self.current_time * 0.3 + pos[2] * 0.1) * turbulence
                
                vel[0] += noise_x * dt
                vel[1] += noise_y * dt
                vel[2] += noise_z * dt
            
            # 球形吸引力
            center_dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            max_distance = self.params['base_radius'] * 3
            
            if center_dist > max_distance:
                # 强制吸引回球面
                attract_strength = self.params['attraction_strength'] * 2
                attract_x = -pos[0] / center_dist * attract_strength
                attract_y = -pos[1] / center_dist * attract_strength
                attract_z = -pos[2] / center_dist * attract_strength
                
                vel[0] += attract_x * dt
                vel[1] += attract_y * dt
                vel[2] += attract_z * dt
            
            # 阻尼
            damping = 0.99
            vel[0] *= damping
            vel[1] *= damping
            vel[2] *= damping
    
    def get_positions(self):
        """获取所有粒子位置"""
        positions = []
        for particle in self.particles:
            if particle['life'] > 0:
                positions.extend(particle['position'])
        return positions
    
    def get_colors(self):
        """获取所有粒子颜色"""
        colors = []
        for particle in self.particles:
            if particle['life'] > 0:
                # 基于生命周期和颜色偏移的动态颜色
                life_ratio = particle['life'] / particle['max_life']
                color_offset = particle['color_offset']
                
                # 基础色调
                base_hue = self.params['color_hue']
                hue = (base_hue + color_offset * 0.3 + self.current_time * 0.1) % 1.0
                saturation = self.params['color_saturation']
                brightness = life_ratio * 0.9
                
                r, g, b = self.hsv_to_rgb(hue, saturation, brightness)
                alpha = life_ratio * 0.8
                
                colors.extend([r, g, b, alpha])
        
        return colors
    
    def get_sizes(self):
        """获取所有粒子大小"""
        sizes = []
        for particle in self.particles:
            if particle['life'] > 0:
                life_ratio = particle['life'] / particle['max_life']
                # 粒子在生命周期中的大小变化
                size_factor = math.sin(life_ratio * math.pi)  # 中间最大，两端最小
                current_size = particle['size'] * size_factor
                sizes.append(current_size)
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
    
    def update_params(self, gesture_data):
        """根据手势数据更新参数"""
        if not gesture_data:
            return
        
        hands_count = gesture_data.get('hands_detected', 0)
        left_hand = gesture_data.get('left_hand', {})
        right_hand = gesture_data.get('right_hand', {})
        gesture_strength = gesture_data.get('gesture_strength', 0.0)
        
        # 发射速率：基于检测到的手数
        base_rate = 80
        if hands_count == 2:
            self.params['emission_rate'] = base_rate * 3
        elif hands_count == 1:
            self.params['emission_rate'] = base_rate * 2
        else:
            self.params['emission_rate'] = base_rate * 0.5
        
        # 粒子速度：基于手势强度
        self.params['velocity_scale'] = 0.3 + gesture_strength * 1.5
        
        # 根据手势类型切换形状
        main_gesture = 'none'
        if left_hand.get('detected', False):
            main_gesture = left_hand.get('gesture', 'none')
        elif right_hand.get('detected', False):
            main_gesture = right_hand.get('gesture', 'none')
        
        # 手势到形状的映射 (优先展示螺旋结构)
        gesture_to_shape = {
            'fist': 'tornado_helix',      # 握拳 → 龙卷风螺旋
            'one': 'double_helix',        # 1指 → 双螺旋
            'two': 'triple_helix',        # 2指 → 三重螺旋  
            'three': 'dna_structure',     # 3指 → DNA结构
            'four': 'braided_lines',      # 4指 → 编织线条
            'open_hand': 'galaxy_spiral'  # 张开 → 银河螺旋
        }
        
        if main_gesture in gesture_to_shape:
            new_shape = gesture_to_shape[main_gesture]
            if new_shape != self.params['shape_mode']:
                self.params['shape_mode'] = new_shape
                print(f"手势 {main_gesture} -> 形状: {new_shape}")
        
        # 波浪参数：基于手势强度和手部位置
        self.params['wave_amplitude'] = 1.0 + gesture_strength * 2.5
        self.params['wave_frequency'] = 0.5 + gesture_strength * 2.0
        self.params['wave_speed'] = 0.5 + gesture_strength * 2.0
        self.params['line_length'] = 4.0 + gesture_strength * 4.0
        
        # 粒子大小：基于左手张开程度
        if left_hand.get('detected', False):
            openness = left_hand.get('openness', 0)
            self.params['size_scale'] = max(0.2, openness * 2.5)
        
        # 双手控制特殊效果
        if hands_count == 2 and left_hand.get('detected') and right_hand.get('detected'):
            left_center = left_hand.get('center', [0, 0.5])
            right_center = right_hand.get('center', [1, 0.5])
            
            # 计算双手距离
            distance = math.sqrt(
                (left_center[0] - right_center[0])**2 + 
                (left_center[1] - right_center[1])**2
            )
            
            # 距离控制波浪复杂度
            self.params['wave_frequency'] = 0.5 + distance * 3.0
            
            # 双手时启用多线模式
            if distance > 0.3:
                self.params['shape_mode'] = 'multiple_lines'
        
        # 湍流强度：基于手势类型
        turbulence_level = 0.0
        for hand_data in [left_hand, right_hand]:
            if hand_data.get('detected', False):
                gesture_type = hand_data.get('gesture', 'none')
                if gesture_type == 'fist':
                    turbulence_level = max(turbulence_level, 2.5)
                elif gesture_type in ['one', 'two']:
                    turbulence_level = max(turbulence_level, 1.2)
                elif gesture_type == 'open_hand':
                    turbulence_level = max(turbulence_level, 0.4)
        
        self.params['turbulence_strength'] = turbulence_level
        
        # 颜色：基于手部位置
        if left_hand.get('detected', False):
            center = left_hand.get('center', [0.5, 0.5])
            self.params['color_hue'] = center[0]
            self.params['color_saturation'] = max(0.4, center[1] * 1.3)
        
        # 右手控制颜色变化速度
        if right_hand.get('detected', False):
            right_center = right_hand.get('center', [0.5, 0.5])
            color_speed = right_center[1]  # Y位置控制颜色变化速度
            self.params['color_hue'] = (self.params['color_hue'] + color_speed * 0.01) % 1.0


class HelixRenderer:
    """螺旋结构渲染器 - 替代原来的球形渲染器"""
    def __init__(self):
        self.current_time = 0.0
        
        # 螺旋参数
        self.params = {
            'base_radius': 1.5,
            'helix_height': 4.0,          # 螺旋总高度
            'helix_count': 2,             # 螺旋线条数量
            'twist_rate': 3.0,            # 扭转速度
            'rotation_speed': [0.5, 1.0, 0.3],
            'pulsation_amplitude': 0.1,
            'pulsation_frequency': 0.5,
            'helix_type': 'double_helix',  # 螺旋类型
            'color': [0.4, 0.7, 1.0],
            'transparency': 0.4,
            'wireframe_enabled': True,
            'wireframe_color': [0.6, 1.0, 0.7],
            'connecting_bridges': True,    # 是否显示连接桥
            'bridge_frequency': 8         # 连接桥频率
        }
        
        # 动画状态
        self.rotation = [0, 0, 0]
        
    def update(self, dt):
        """更新螺旋状态"""
        self.current_time += dt
        
        # 更新旋转
        self.rotation[0] += self.params['rotation_speed'][0] * dt * 60
        self.rotation[1] += self.params['rotation_speed'][1] * dt * 60
        self.rotation[2] += self.params['rotation_speed'][2] * dt * 60
        
        # 保持角度在0-360范围
        for i in range(3):
            self.rotation[i] = self.rotation[i] % 360
    
    def generate_helix_points(self):
        """生成螺旋结构的点集"""
        points = []
        colors = []
        
        # 螺旋参数
        radius = self.params['base_radius']
        height = self.params['helix_height']
        helix_count = self.params['helix_count']
        twist_rate = self.params['twist_rate']
        
        # 脉动效果
        pulsation = 1.0 + self.params['pulsation_amplitude'] * math.sin(
            self.current_time * self.params['pulsation_frequency'] * 2 * math.pi
        )
        current_radius = radius * pulsation
        
        # 生成每条螺旋线
        points_per_helix = 100
        for helix_id in range(helix_count):
            phase_offset = helix_id * 2 * math.pi / helix_count
            
            for i in range(points_per_helix):
                t = i / (points_per_helix - 1)  # 0 to 1
                
                # 高度
                z = (t - 0.5) * height
                
                # 角度
                angle = t * twist_rate * 2 * math.pi + phase_offset + self.current_time * 0.5
                
                # 位置
                x = current_radius * math.cos(angle)
                y = current_radius * math.sin(angle)
                
                points.extend([x, y, z])
                
                # 颜色：基于螺旋ID和高度
                hue = (helix_id / helix_count + t * 0.3 + self.current_time * 0.1) % 1.0
                r, g, b = self.hsv_to_rgb(hue, 0.8, 0.9)
                colors.extend([r, g, b, 0.8])
        
        # 生成连接桥
        if self.params['connecting_bridges'] and helix_count >= 2:
            bridge_points = self.generate_connecting_bridges()
            points.extend(bridge_points['positions'])
            colors.extend(bridge_points['colors'])
        
        return {
            'positions': points,
            'colors': colors
        }
    
    def generate_connecting_bridges(self):
        """生成螺旋间的连接桥"""
        positions = []
        colors = []
        
        radius = self.params['base_radius']
        height = self.params['helix_height']
        twist_rate = self.params['twist_rate']
        bridge_count = self.params['bridge_frequency']
        
        for i in range(bridge_count):
            t = i / bridge_count
            z = (t - 0.5) * height
            angle = t * twist_rate * 2 * math.pi + self.current_time * 0.5
            
            # 第一条螺旋的点
            x1 = radius * math.cos(angle)
            y1 = radius * math.sin(angle)
            
            # 第二条螺旋的点
            x2 = radius * math.cos(angle + math.pi)
            y2 = radius * math.sin(angle + math.pi)
            
            # 连接桥上的点
            bridge_points = 5
            for j in range(bridge_points):
                bridge_t = j / (bridge_points - 1)
                
                x = x1 + bridge_t * (x2 - x1)
                y = y1 + bridge_t * (y2 - y1)
                
                positions.extend([x, y, z])
                
                # 连接桥颜色
                hue = (0.3 + t * 0.2 + self.current_time * 0.05) % 1.0
                r, g, b = self.hsv_to_rgb(hue, 0.6, 0.7)
                colors.extend([r, g, b, 0.6])
        
        return {
            'positions': positions,
            'colors': colors
        }
    
    def get_helix_params(self):
        """获取当前螺旋参数"""
        pulsation = 1.0 + self.params['pulsation_amplitude'] * math.sin(
            self.current_time * self.params['pulsation_frequency'] * 2 * math.pi
        )
        
        return {
            'radius': self.params['base_radius'] * pulsation,
            'height': self.params['helix_height'],
            'rotation': self.rotation,
            'color': self.params['color'],
            'transparency': self.params['transparency']
        }
    
    def get_wireframe_params(self):
        """获取线框参数"""
        if not self.params['wireframe_enabled']:
            return None
        
        return {
            'radius': self.params['base_radius'] * 1.2,
            'rotation': [r * 0.3 for r in self.rotation],
            'color': self.params['wireframe_color'],
            'line_width': 1.5
        }
    
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

    def update_params(self, gesture_data):
        """根据手势数据更新螺旋参数"""
        if not gesture_data:
            return
        
        hands_count = gesture_data.get('hands_detected', 0)
        left_hand = gesture_data.get('left_hand', {})
        right_hand = gesture_data.get('right_hand', {})
        gesture_strength = gesture_data.get('gesture_strength', 0.0)
        
        # 基础半径：手势强度控制
        self.params['base_radius'] = 1.2 + gesture_strength * 1.8
        
        # 螺旋高度：双手距离控制
        if hands_count == 2 and left_hand.get('detected') and right_hand.get('detected'):
            left_center = left_hand.get('center', [0, 0.5])
            right_center = right_hand.get('center', [1, 0.5])
            
            distance = math.sqrt(
                (left_center[0] - right_center[0])**2 + 
                (left_center[1] - right_center[1])**2
            )
            self.params['helix_height'] = 3.0 + distance * 4.0
            self.params['helix_count'] = min(int(2 + distance * 3), 5)  # 2-5条螺旋
        else:
            self.params['helix_height'] = 3.5 + gesture_strength * 2.0
        
        # 扭转速度：基于主手势
        main_gesture = 'none'
        if left_hand.get('detected'):
            main_gesture = left_hand.get('gesture', 'none')
        elif right_hand.get('detected'):
            main_gesture = right_hand.get('gesture', 'none')
        
        twist_map = {
            'fist': 5.0,      # 握拳：快速扭转
            'one': 2.0,       # 1指：慢速扭转
            'two': 3.0,       # 2指：中速扭转
            'three': 4.0,     # 3指：快速扭转
            'four': 2.5,      # 4指：中等扭转
            'open_hand': 1.5, # 张开：慢速扭转
            'none': 2.0
        }
        
        self.params['twist_rate'] = twist_map.get(main_gesture, 2.0)
        
        # 旋转速度：基于手势强度
        base_speed = 0.3 + gesture_strength * 1.2
        self.params['rotation_speed'] = [
            base_speed * 0.8,
            base_speed * 1.0,
            base_speed * 0.6
        ]
        
        # 脉动：基于手势强度
        self.params['pulsation_amplitude'] = gesture_strength * 0.3
        self.params['pulsation_frequency'] = 0.3 + gesture_strength * 1.0
        
        # 连接桥：基于手势类型
        bridge_enabled = True
        bridge_frequency = 8
        
        if main_gesture == 'fist':
            bridge_frequency = 12  # 握拳：更多连接桥
        elif main_gesture in ['three', 'four']:
            bridge_frequency = 6   # 3-4指：较少连接桥
        elif main_gesture == 'open_hand':
            bridge_enabled = False # 张开：无连接桥
            
        self.params['connecting_bridges'] = bridge_enabled
        self.params['bridge_frequency'] = bridge_frequency
        
        # 颜色：基于手部位置
        if right_hand.get('detected', False):
            center = right_hand.get('center', [0.5, 0.5])
            hue = center[0]  # X位置控制色调
            saturation = max(0.4, center[1])  # Y位置控制饱和度
            
            # HSV转RGB
            r, g, b = self.hsv_to_rgb(hue, saturation, 0.8)
            self.params['color'] = [r, g, b]
        
        # 左手控制透明度
        if left_hand.get('detected', False):
            openness = left_hand.get('openness', 0.5)
            self.params['transparency'] = 0.2 + openness * 0.5


class ParticleSphereSystem:
    """整合的粒子螺旋系统"""
    
    def __init__(self, max_particles=1500):
        self.particle_system = ParticleSystem(max_particles)
        self.helix_renderer = HelixRenderer()  # 替换为螺旋渲染器
        self.current_time = 0.0
    
    def update(self, dt, gesture_data=None):
        """更新整个系统"""
        self.current_time += dt
        
        # 更新参数
        if gesture_data:
            self.particle_system.update_params(gesture_data)
            self.helix_renderer.update_params(gesture_data)
        
        # 更新系统
        self.particle_system.update(dt)
        self.helix_renderer.update(dt)
    
    def get_particle_data(self):
        """获取粒子渲染数据"""
        return {
            'positions': self.particle_system.get_positions(),
            'colors': self.particle_system.get_colors(),
            'sizes': self.particle_system.get_sizes()
        }
    
    def get_helix_data(self):
        """获取螺旋渲染数据"""
        return self.helix_renderer.get_helix_params()
    
    def get_helix_points(self):
        """获取螺旋点集数据"""
        return self.helix_renderer.generate_helix_points()
    
    def get_wireframe_data(self):
        """获取线框数据"""
        return self.helix_renderer.get_wireframe_params()
    
    # 保持向后兼容的接口
    def get_sphere_data(self):
        """获取球形渲染数据（向后兼容）"""
        return self.get_helix_data()

if __name__ == "__main__":
    # 测试粒子球形系统
    import time
    
    system = ParticleSphereSystem(max_particles=500)
    
    # 模拟手势数据
    test_gesture = {
        'hands_detected': 1,
        'left_hand': {
            'detected': True,
            'gesture': 'open_hand',
            'openness': 0.8,
            'center': [0.3, 0.6]
        },
        'right_hand': {'detected': False},
        'gesture_strength': 0.8
    }
    
    print("粒子球形系统测试...")
    start_time = time.time()
    
    for i in range(100):  # 模拟100帧
        current_time = time.time() - start_time
        dt = 1.0 / 60.0
        
        system.update(dt, test_gesture if i > 10 else None)
        
        if i % 20 == 0:
            particle_data = system.get_particle_data()
            sphere_data = system.get_sphere_data()
            print(f"Frame {i}: Particles={len(particle_data['positions'])//3}, "
                  f"Sphere radius={sphere_data['radius']:.2f}")
        
        time.sleep(dt)
    
    print("测试完成！")