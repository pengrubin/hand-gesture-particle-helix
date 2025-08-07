"""
TouchDesigner数据处理脚本
处理手势数据并转换为TouchDesigner可用格式，用于控制粒子系统和视觉效果

使用方法：
1. 将此脚本放入Execute DAT中
2. 在frameStart或frameEnd回调中调用相关函数
3. 通过Table DAT输出处理后的数据
"""

import math
import random

class TouchDesignerDataProcessor:
    def __init__(self):
        self.current_data = {}
        self.smoothed_data = {}
        self.previous_data = {}
        self.smoothing_factor = 0.8  # 数据平滑系数
        
        # 粒子控制参数
        self.particle_params = {
            'count': 1000,
            'velocity': 1.0,
            'size': 1.0,
            'spread': 1.0,
            'color_hue': 0.5,
            'color_saturation': 1.0,
            'emission_rate': 100,
            'turbulence': 0.0
        }
        
        # 球形变形参数
        self.sphere_params = {
            'radius': 1.0,
            'deformation': 0.0,
            'rotation_speed': 1.0,
            'pulsation': 0.0,
            'surface_noise': 0.0
        }
        
    def smooth_value(self, current, previous, factor):
        """数值平滑处理"""
        if previous is None:
            return current
        return previous * factor + current * (1 - factor)
    
    def process_gesture_data(self, gesture_data):
        """处理手势数据并更新控制参数"""
        if not gesture_data:
            return
            
        self.current_data = gesture_data.copy()
        
        # 获取主要控制数据
        hands_count = gesture_data.get('hands_detected', 0)
        left_hand = gesture_data.get('left_hand', {})
        right_hand = gesture_data.get('right_hand', {})
        gesture_strength = gesture_data.get('gesture_strength', 0.0)
        
        # 更新粒子参数
        self.update_particle_params(hands_count, left_hand, right_hand, gesture_strength)
        
        # 更新球形参数
        self.update_sphere_params(hands_count, left_hand, right_hand, gesture_strength)
        
        # 平滑处理
        self.apply_smoothing()
        
        # 保存上一帧数据
        self.previous_data = self.current_data.copy()
    
    def update_particle_params(self, hands_count, left_hand, right_hand, gesture_strength):
        """根据手势更新粒子参数"""
        
        # 粒子数量：双手检测时增加
        base_count = 500
        if hands_count == 2:
            self.particle_params['count'] = base_count * 2
        elif hands_count == 1:
            self.particle_params['count'] = base_count * 1.5
        else:
            self.particle_params['count'] = base_count * 0.3
        
        # 粒子速度：基于手势强度
        self.particle_params['velocity'] = 0.5 + gesture_strength * 2.0
        
        # 粒子大小：基于手部张开程度
        if left_hand.get('detected', False):
            left_openness = left_hand.get('openness', 0)
            self.particle_params['size'] = max(0.5, left_openness * 3.0)
        
        # 发射速率：双手时更高
        self.particle_params['emission_rate'] = 50 + hands_count * 100 + gesture_strength * 200
        
        # 扩散范围：右手控制
        if right_hand.get('detected', False):
            right_openness = right_hand.get('openness', 0)
            self.particle_params['spread'] = 0.5 + right_openness * 2.0
        
        # 湍流：基于手势类型
        left_gesture = left_hand.get('gesture_type', 'none')
        right_gesture = right_hand.get('gesture_type', 'none')
        
        turbulence_level = 0.0
        if left_gesture == 'fist' or right_gesture == 'fist':
            turbulence_level = 2.0
        elif left_gesture in ['one', 'two'] or right_gesture in ['one', 'two']:
            turbulence_level = 1.0
        elif left_gesture == 'open_hand' or right_gesture == 'open_hand':
            turbulence_level = 0.2
            
        self.particle_params['turbulence'] = turbulence_level
        
        # 颜色：基于手部位置
        if left_hand.get('detected', False):
            center = left_hand.get('center', [0.5, 0.5])
            self.particle_params['color_hue'] = center[0]  # X位置控制色调
            self.particle_params['color_saturation'] = max(0.3, center[1])  # Y位置控制饱和度
    
    def update_sphere_params(self, hands_count, left_hand, right_hand, gesture_strength):
        """根据手势更新球形参数"""
        
        # 基础半径：手势强度控制
        self.sphere_params['radius'] = 0.8 + gesture_strength * 1.5
        
        # 变形程度：双手距离控制
        if hands_count == 2 and left_hand.get('detected') and right_hand.get('detected'):
            left_center = left_hand.get('center', [0, 0.5])
            right_center = right_hand.get('center', [1, 0.5])
            
            # 计算双手距离
            distance = math.sqrt(
                (left_center[0] - right_center[0])**2 + 
                (left_center[1] - right_center[1])**2
            )
            self.sphere_params['deformation'] = min(distance * 3.0, 2.0)
        else:
            self.sphere_params['deformation'] = gesture_strength * 0.5
        
        # 旋转速度：基于主手势类型
        main_gesture = 'none'
        if left_hand.get('detected'):
            main_gesture = left_hand.get('gesture_type', 'none')
        elif right_hand.get('detected'):
            main_gesture = right_hand.get('gesture_type', 'none')
        
        speed_map = {
            'fist': 3.0,
            'one': 1.5,
            'two': 2.0,
            'three': 2.5,
            'four': 1.0,
            'open_hand': 0.5,
            'none': 0.1
        }
        self.sphere_params['rotation_speed'] = speed_map.get(main_gesture, 1.0)
        
        # 脉动：基于手势强度变化
        self.sphere_params['pulsation'] = gesture_strength * 0.8
        
        # 表面噪声：基于湍流参数
        self.sphere_params['surface_noise'] = self.particle_params['turbulence'] * 0.3
    
    def apply_smoothing(self):
        """应用数据平滑"""
        for key in self.particle_params:
            if key in self.smoothed_data.get('particle_params', {}):
                self.particle_params[key] = self.smooth_value(
                    self.particle_params[key],
                    self.smoothed_data['particle_params'][key],
                    self.smoothing_factor
                )
        
        for key in self.sphere_params:
            if key in self.smoothed_data.get('sphere_params', {}):
                self.sphere_params[key] = self.smooth_value(
                    self.sphere_params[key],
                    self.smoothed_data['sphere_params'][key],
                    self.smoothing_factor
                )
        
        # 更新平滑数据
        self.smoothed_data = {
            'particle_params': self.particle_params.copy(),
            'sphere_params': self.sphere_params.copy()
        }
    
    def get_particle_table_data(self):
        """获取粒子参数表格数据，用于Table DAT"""
        return [
            ['parameter', 'value'],
            ['count', self.particle_params['count']],
            ['velocity', self.particle_params['velocity']],
            ['size', self.particle_params['size']],
            ['spread', self.particle_params['spread']],
            ['color_hue', self.particle_params['color_hue']],
            ['color_saturation', self.particle_params['color_saturation']],
            ['emission_rate', self.particle_params['emission_rate']],
            ['turbulence', self.particle_params['turbulence']]
        ]
    
    def get_sphere_table_data(self):
        """获取球形参数表格数据，用于Table DAT"""
        return [
            ['parameter', 'value'],
            ['radius', self.sphere_params['radius']],
            ['deformation', self.sphere_params['deformation']],
            ['rotation_speed', self.sphere_params['rotation_speed']],
            ['pulsation', self.sphere_params['pulsation']],
            ['surface_noise', self.sphere_params['surface_noise']]
        ]

# 全局处理器实例
if not hasattr(op, 'data_processor'):
    op.data_processor = TouchDesignerDataProcessor()

# TouchDesigner回调函数
def onFrameStart(frame):
    """每帧开始时调用"""
    # 这里应该从手势检测模块获取数据
    # gesture_data = parent().fetch('gesture_detector_data')  # 示例
    pass

def onFrameEnd(frame):
    """每帧结束时调用，更新输出数据"""
    # 更新Table DAT数据
    update_output_tables()

def update_output_tables():
    """更新输出数据表"""
    processor = op.data_processor
    
    # 更新粒子参数表
    particle_table = op('particle_params')  # 假设有名为particle_params的Table DAT
    if particle_table:
        particle_table.clear()
        for row in processor.get_particle_table_data():
            particle_table.appendRow(row)
    
    # 更新球形参数表
    sphere_table = op('sphere_params')  # 假设有名为sphere_params的Table DAT
    if sphere_table:
        sphere_table.clear()
        for row in processor.get_sphere_table_data():
            sphere_table.appendRow(row)

def process_gesture_update(gesture_data):
    """处理手势数据更新"""
    op.data_processor.process_gesture_data(gesture_data)
    update_output_tables()

# 工具函数
def map_range(value, from_min, from_max, to_min, to_max):
    """数值范围映射"""
    return to_min + (value - from_min) * (to_max - to_min) / (from_max - from_min)

def ease_in_out(t):
    """缓动函数"""
    return t * t * (3.0 - 2.0 * t)