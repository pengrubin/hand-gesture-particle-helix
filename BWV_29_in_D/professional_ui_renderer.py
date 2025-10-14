#!/usr/bin/env python3
"""
BWV_29_in_D 专业指挥家控制界面渲染器
为指挥家手势控制系统设计的专业级实时状态显示界面

核心功能：
- 专业音乐制作风格的界面设计
- 实时7声部可视化边界和激活状态
- 动态音频电平表和频谱显示
- 手势轨迹追踪和指挥分析
- 性能监控和调试信息面板
- 响应式布局和动画效果

Author: Claude Code
Date: 2025-10-05
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import colorsys


class ProfessionalUIRenderer:
    """专业指挥家控制界面渲染器"""

    def __init__(self):
        """初始化专业UI渲染器"""
        # 专业音乐制作风格颜色方案
        self.colors = {
            # 主要界面颜色
            'bg_dark': (26, 26, 26),        # 深色背景 #1a1a1a
            'bg_medium': (45, 45, 45),      # 中等背景 #2d2d2d
            'bg_light': (64, 64, 64),       # 浅色背景 #404040
            'bg_panel': (35, 35, 35),       # 面板背景 #232323

            # 文本颜色
            'text_primary': (255, 255, 255),    # 主要文本 - 白色
            'text_secondary': (200, 200, 200),  # 次要文本 - 浅灰
            'text_muted': (140, 140, 140),      # 静音文本 - 灰色
            'text_accent': (255, 215, 0),       # 强调文本 - 金色

            # 状态颜色
            'active_green': (76, 255, 76),      # 激活绿色 #4cff4c
            'active_yellow': (255, 255, 76),    # 激活黄色 #ffff4c
            'warning_orange': (255, 165, 0),    # 警告橙色 #ffa500
            'error_red': (255, 76, 76),         # 错误红色 #ff4c4c
            'paused_blue': (76, 153, 255),      # 暂停蓝色 #4c99ff
            'conducting_gold': (255, 215, 0),   # 指挥金色 #ffd700

            # 7个声部专业调色板
            'voice_colors': [
                (255, 99, 71),    # 1. Tromba - 番茄红 (铜管)
                (138, 43, 226),   # 2. Violins - 蓝紫色 (弦乐)
                (255, 140, 0),    # 3. Viola - 橙色 (中音弦乐)
                (50, 205, 50),    # 4. Oboe - 酸橙绿 (木管)
                (30, 144, 255),   # 5. Continuo - 道奇蓝 (数字低音)
                (255, 20, 147),   # 6. Organo - 深粉色 (管风琴)
                (255, 215, 0)     # 7. Timpani - 金色 (打击乐)
            ],

            # 音频电平颜色 (专业音频设备风格)
            'level_low': (76, 255, 76),      # 低电平 - 绿色
            'level_mid': (255, 255, 76),     # 中电平 - 黄色
            'level_high': (255, 165, 0),     # 高电平 - 橙色
            'level_peak': (255, 76, 76),     # 峰值 - 红色
            'level_bg': (20, 20, 20),        # 电平背景 - 深灰

            # 特殊效果
            'glow_effect': (255, 255, 255),  # 发光效果
            'shadow': (0, 0, 0),             # 阴影
            'grid_line': (60, 60, 60),       # 网格线
            'border_accent': (100, 100, 100), # 边框强调
        }

        # 字体设置
        self.fonts = {
            'title': cv2.FONT_HERSHEY_SIMPLEX,
            'subtitle': cv2.FONT_HERSHEY_SIMPLEX,
            'body': cv2.FONT_HERSHEY_SIMPLEX,
            'mono': cv2.FONT_HERSHEY_DUPLEX,
            'small': cv2.FONT_HERSHEY_SIMPLEX
        }

        self.font_scales = {
            'title': 0.9,
            'subtitle': 0.7,
            'body': 0.5,
            'small': 0.4,
            'tiny': 0.3
        }

        self.thickness = {
            'thin': 1,
            'normal': 2,
            'thick': 3,
            'bold': 4
        }

        # 布局参数 (响应式设计)
        self.layout = {
            'header_height': 120,
            'footer_height': 180,
            'sidebar_width': 260,
            'panel_margin': 12,
            'text_margin': 8,
            'line_spacing': 22,
            'corner_radius': 8,
            'border_width': 2
        }

        # 动画和效果参数
        self.animation = {
            'pulse_speed': 0.05,
            'fade_speed': 0.02,
            'glow_intensity': 0.4,
            'bounce_factor': 1.2,
            'transition_speed': 0.1
        }

        # 历史数据缓存 (用于动态效果)
        self.history = {
            'volume_levels': {},
            'gesture_strength': deque(maxlen=60),  # 1秒历史 (60FPS)
            'fps_history': deque(maxlen=30),       # 0.5秒历史
            'conducting_analysis': deque(maxlen=100), # 指挥分析历史
            'hand_positions': deque(maxlen=30),    # 手部位置历史
            'max_history_length': 120
        }

        # 时间跟踪
        self.last_update_time = time.time()
        self.animation_time = 0.0
        self.frame_count = 0

        # 特效缓存
        self.effects_cache = {
            'glow_masks': {},
            'gradient_cache': {},
            'particle_systems': []
        }

        # 指挥分析状态
        self.conducting_analysis = {
            'tempo_detection': 0.0,
            'rhythm_pattern': [],
            'gesture_intensity': 0.0,
            'coordination_score': 0.0,
            'musical_expression': 0.0
        }

    def update_animation_time(self):
        """更新动画时间"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.animation_time += dt
        self.last_update_time = current_time
        self.frame_count += 1

    def draw_professional_interface(self, frame: np.ndarray, system_state, performance_metrics,
                                  user_presence, gesture_data: Dict, active_regions: Dict,
                                  audio_controller=None, region_info: Dict = None) -> np.ndarray:
        """绘制完整的专业界面"""
        try:
            # 更新动画时间
            self.update_animation_time()

            display_frame = frame.copy()
            height, width = frame.shape[:2]

            # 1. 绘制主背景和网格
            self._draw_background_and_grid(display_frame)

            # 2. 绘制专业头部状态栏
            self._draw_professional_header(display_frame, system_state, performance_metrics,
                                         user_presence, audio_controller)

            # 3. 绘制7声部专业区域界面
            if region_info:
                self._draw_voice_regions_professional(display_frame, region_info['voice_regions'],
                                                    active_regions, region_info['central_control_region'])

            # 4. 绘制专业音频电平表和频谱
            if audio_controller and audio_controller.is_initialized:
                track_volumes = audio_controller.get_track_volumes()
                self._draw_professional_audio_meters(display_frame, track_volumes, audio_controller)

            # 5. 绘制手势分析和指挥分析面板
            self._draw_conducting_analysis_panel(display_frame, gesture_data, active_regions)

            # 6. 绘制性能监控面板
            self._draw_performance_panel(display_frame, performance_metrics)

            # 7. 绘制手势轨迹和效果
            self._draw_gesture_trails_and_effects(display_frame, gesture_data, active_regions)

            # 8. 绘制系统控制面板
            self._draw_system_control_panel(display_frame)

            return display_frame

        except Exception as e:
            logging.error(f"Professional UI rendering error: {e}")
            return frame

    def _draw_background_and_grid(self, frame: np.ndarray):
        """绘制专业背景和网格"""
        height, width = frame.shape[:2]

        # 渐变背景
        for y in range(height):
            alpha = 0.1 + 0.05 * np.sin(y / height * np.pi)
            color = tuple(int(c * alpha + self.colors['bg_dark'][i] * (1 - alpha))
                         for i, c in enumerate(self.colors['bg_medium']))
            cv2.line(frame, (0, y), (width, y), color, 1)

        # 网格线 (音乐制作风格)
        grid_spacing = 50
        for x in range(0, width, grid_spacing):
            cv2.line(frame, (x, 0), (x, height), self.colors['grid_line'], 1)
        for y in range(0, height, grid_spacing):
            cv2.line(frame, (0, y), (width, y), self.colors['grid_line'], 1)

    def _draw_professional_header(self, frame: np.ndarray, system_state, performance_metrics,
                                user_presence, audio_controller=None):
        """绘制专业头部状态栏"""
        height, width = frame.shape[:2]
        header_height = self.layout['header_height']

        # 头部背景渐变
        overlay = frame.copy()
        for y in range(header_height):
            alpha = 0.9 - (y / header_height) * 0.2
            color = tuple(int(c * alpha) for c in self.colors['bg_panel'])
            cv2.line(overlay, (0, y), (width, y), color, 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 顶部强调线
        cv2.line(frame, (0, 0), (width, 0), self.colors['conducting_gold'], 3)

        # 主标题区域
        title_area_height = 45
        title_bg = frame.copy()
        cv2.rectangle(title_bg, (0, 0), (width, title_area_height), self.colors['bg_dark'], -1)
        cv2.addWeighted(title_bg, 0.9, frame, 0.1, 0, frame)

        # 主标题 (带音乐符号)
        title = "♫ BWV 29 in D - Professional Conductor Control System ♫"
        title_size = cv2.getTextSize(title, self.fonts['title'], self.font_scales['title'],
                                   self.thickness['normal'])[0]
        title_x = (width - title_size[0]) // 2

        # 标题发光效果
        self._draw_text_with_glow(frame, title, (title_x, 28), self.fonts['title'],
                                self.font_scales['title'], self.colors['text_accent'],
                                self.thickness['normal'])

        # 系统状态指示器 (左侧)
        self._draw_system_status_indicator(frame, 20, 60, system_state)

        # 用户存在状态 (中央)
        self._draw_user_presence_status(frame, width // 2 - 150, 60, user_presence)

        # 实时时钟和播放信息 (右侧)
        self._draw_playback_info(frame, width - 250, 60, audio_controller)

        # 分隔线
        cv2.line(frame, (0, header_height - 2), (width, header_height - 2),
                self.colors['border_accent'], 2)

    def _draw_system_status_indicator(self, frame: np.ndarray, x: int, y: int, system_state):
        """绘制系统状态指示器"""
        # 状态颜色映射
        state_colors = {
            'INITIALIZING': self.colors['warning_orange'],
            'WAITING_USER': self.colors['paused_blue'],
            'USER_DETECTED': self.colors['active_green'],
            'CONDUCTING': self.colors['conducting_gold'],
            'PAUSED': self.colors['paused_blue'],
            'ERROR': self.colors['error_red'],
        }

        state_name = system_state.value.upper() if hasattr(system_state, 'value') else str(system_state).upper()
        state_color = state_colors.get(state_name, self.colors['text_muted'])

        # 状态指示灯 (带脉冲效果)
        pulse_intensity = 0.7 + 0.3 * abs(np.sin(self.animation_time * 3))
        pulse_color = tuple(int(c * pulse_intensity) for c in state_color)

        cv2.circle(frame, (x, y), 12, pulse_color, -1)
        cv2.circle(frame, (x, y), 14, state_color, 2)

        # 状态文字
        status_text = f"System: {state_name}"
        cv2.putText(frame, status_text, (x + 25, y + 5),
                   self.fonts['body'], self.font_scales['body'], state_color, self.thickness['normal'])

    def _draw_user_presence_status(self, frame: np.ndarray, x: int, y: int, user_presence):
        """绘制用户存在状态"""
        presence_text = "🎭 CONDUCTOR ACTIVE" if user_presence.is_present else "⏳ AWAITING CONDUCTOR"
        presence_color = self.colors['active_green'] if user_presence.is_present else self.colors['warning_orange']

        # 状态背景
        text_size = cv2.getTextSize(presence_text, self.fonts['subtitle'],
                                  self.font_scales['subtitle'], self.thickness['normal'])[0]

        # 圆角矩形背景
        self._draw_rounded_rectangle(frame, (x - 10, y - 20), (x + text_size[0] + 10, y + 10),
                                   self.colors['bg_light'], presence_color)

        cv2.putText(frame, presence_text, (x, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'], presence_color, self.thickness['normal'])

        # 置信度条
        if user_presence.is_present:
            conf_bar_y = y + 15
            conf_bar_width = 200
            conf_bar_height = 6

            # 背景
            cv2.rectangle(frame, (x, conf_bar_y), (x + conf_bar_width, conf_bar_y + conf_bar_height),
                         self.colors['bg_light'], -1)

            # 置信度填充
            conf_fill_width = int(conf_bar_width * user_presence.confidence)
            conf_color = self._interpolate_color(self.colors['error_red'], self.colors['active_green'],
                                               user_presence.confidence)
            cv2.rectangle(frame, (x, conf_bar_y), (x + conf_fill_width, conf_bar_y + conf_bar_height),
                         conf_color, -1)

            # 置信度数值
            conf_text = f"Confidence: {user_presence.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, conf_bar_y + conf_bar_height + 15),
                       self.fonts['small'], self.font_scales['small'], self.colors['text_secondary'], 1)

    def _draw_playback_info(self, frame: np.ndarray, x: int, y: int, audio_controller=None):
        """绘制播放信息"""
        if audio_controller:
            try:
                # 播放时间
                current_time = time.strftime("%H:%M:%S")
                time_text = f"Time: {current_time}"
                cv2.putText(frame, time_text, (x, y),
                           self.fonts['mono'], self.font_scales['body'], self.colors['text_primary'],
                           self.thickness['normal'])

                # 播放位置 (如果可用)
                if hasattr(audio_controller, 'get_playback_position'):
                    position = audio_controller.get_playback_position()
                    position_text = f"Position: {position:.1f}s"
                    cv2.putText(frame, position_text, (x, y + 20),
                               self.fonts['mono'], self.font_scales['small'], self.colors['text_secondary'], 1)

                # 播放状态指示
                if hasattr(audio_controller, 'is_playing') and audio_controller.is_playing():
                    play_indicator = "▶ PLAYING"
                    play_color = self.colors['active_green']
                else:
                    play_indicator = "⏸ PAUSED"
                    play_color = self.colors['paused_blue']

                cv2.putText(frame, play_indicator, (x, y + 40),
                           self.fonts['body'], self.font_scales['small'], play_color, self.thickness['normal'])

            except Exception as e:
                logging.error(f"Error drawing playback info: {e}")

    def _draw_voice_regions_professional(self, frame: np.ndarray, voice_regions: Dict[str, Any],
                                       active_regions: Dict[str, Any], central_region: Dict[str, Any]):
        """绘制专业7声部区域界面"""
        height, width = frame.shape[:2]
        header_height = self.layout['header_height']
        footer_height = self.layout['footer_height']
        sidebar_width = self.layout['sidebar_width']

        # 可用区域 (排除头部、底部和右侧边栏)
        available_width = width - sidebar_width
        available_height = height - header_height - footer_height
        start_x = 0
        start_y = header_height

        # 7个声部区域的专业布局
        for i, (region_name, region_data) in enumerate(voice_regions.items()):
            bounds = region_data['bounds']
            voice_color = self.colors['voice_colors'][i % len(self.colors['voice_colors'])]

            # 检查激活状态
            is_active = region_name in active_regions
            activation_strength = 0.0
            if is_active:
                activation_strength = active_regions[region_name].get('activation_strength', 0.0)

            # 转换坐标到可用区域
            x1 = int(start_x + bounds['x1'] * available_width)
            y1 = int(start_y + bounds['y1'] * available_height)
            x2 = int(start_x + bounds['x2'] * available_width)
            y2 = int(start_y + bounds['y2'] * available_height)

            # 绘制声部区域
            self._draw_voice_region_enhanced(frame, x1, y1, x2, y2, voice_color,
                                           region_name, i + 1, is_active, activation_strength)

        # 绘制中央控制区域 (指挥台)
        self._draw_central_podium(frame, central_region, active_regions, start_x, start_y,
                                available_width, available_height)

        # 绘制声部连接线
        self._draw_voice_connections(frame, voice_regions, active_regions, start_x, start_y,
                                   available_width, available_height)

    def _draw_voice_region_enhanced(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                                  voice_color: Tuple[int, int, int], region_name: str,
                                  voice_number: int, is_active: bool, activation_strength: float):
        """绘制增强的声部区域"""

        # 动态边框效果
        if is_active:
            # 脉冲效果
            pulse = 0.7 + 0.3 * abs(np.sin(self.animation_time * 4))
            line_color = tuple(int(c * pulse) for c in voice_color)
            line_thickness = 4

            # 发光效果
            for offset in range(8, 0, -1):
                alpha = 0.1 * (9 - offset)
                glow_color = tuple(int(c * alpha) for c in voice_color)
                cv2.rectangle(frame, (x1 - offset, y1 - offset),
                            (x2 + offset, y2 + offset), glow_color, 1)

            # 激活填充
            overlay = frame.copy()
            fill_alpha = 0.15 + 0.1 * activation_strength
            cv2.rectangle(overlay, (x1, y1), (x2, y2), voice_color, -1)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        else:
            line_color = tuple(int(c * 0.6) for c in voice_color)
            line_thickness = 2

        # 主边框 (圆角)
        self._draw_rounded_rectangle(frame, (x1, y1), (x2, y2), None, line_color, line_thickness)

        # 声部信息面板
        self._draw_voice_info_panel(frame, x1, y1, x2, y2, voice_color, region_name,
                                  voice_number, is_active, activation_strength)

    def _draw_voice_info_panel(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                             voice_color: Tuple[int, int, int], region_name: str,
                             voice_number: int, is_active: bool, activation_strength: float):
        """绘制声部信息面板"""

        panel_height = 80
        panel_y = y1 + 8

        # 声部编号徽章 (专业设计)
        badge_size = 35
        badge_x = x1 + 12
        badge_y = panel_y + badge_size // 2

        # 徽章背景 (渐变)
        cv2.circle(frame, (badge_x + badge_size // 2, badge_y), badge_size // 2, voice_color, -1)
        cv2.circle(frame, (badge_x + badge_size // 2, badge_y), badge_size // 2 + 2,
                  self.colors['text_primary'], 2)

        # 声部编号
        badge_text = str(voice_number)
        text_size = cv2.getTextSize(badge_text, self.fonts['title'], self.font_scales['subtitle'],
                                  self.thickness['bold'])[0]
        text_x = badge_x + (badge_size - text_size[0]) // 2
        text_y = badge_y + text_size[1] // 2
        cv2.putText(frame, badge_text, (text_x, text_y),
                   self.fonts['title'], self.font_scales['subtitle'],
                   self.colors['bg_dark'], self.thickness['bold'])

        # 声部名称 (处理长名称)
        voice_name = region_name.split('_')[0].replace('_', ' ')
        if len(voice_name) > 12:
            voice_name = voice_name[:12] + "..."

        name_x = badge_x + badge_size + 12
        cv2.putText(frame, voice_name, (name_x, panel_y + 20),
                   self.fonts['body'], self.font_scales['body'], voice_color, self.thickness['normal'])

        # 乐器类型
        instrument_parts = region_name.split('_')
        if len(instrument_parts) > 1:
            instrument = instrument_parts[1]
            cv2.putText(frame, instrument, (name_x, panel_y + 40),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)

        # 调性信息
        if 'in_D' in region_name:
            cv2.putText(frame, "Key: D Major", (name_x, panel_y + 55),
                       self.fonts['small'], self.font_scales['tiny'],
                       self.colors['text_muted'], 1)

        # 激活状态指示
        if is_active:
            # 激活强度条
            strength_bar_x = x2 - 100
            strength_bar_y = panel_y + 8
            strength_bar_width = 80
            strength_bar_height = 12

            # 背景
            cv2.rectangle(frame, (strength_bar_x, strength_bar_y),
                         (strength_bar_x + strength_bar_width, strength_bar_y + strength_bar_height),
                         self.colors['bg_light'], -1)

            # 强度填充 (渐变色)
            fill_width = int(strength_bar_width * activation_strength)
            if fill_width > 0:
                # 创建渐变效果
                for i in range(fill_width):
                    progress = i / strength_bar_width
                    color = self._interpolate_color(voice_color, self.colors['active_yellow'], progress * 0.3)
                    cv2.line(frame, (strength_bar_x + i, strength_bar_y),
                            (strength_bar_x + i, strength_bar_y + strength_bar_height), color, 1)

            # 强度数值
            strength_text = f"{activation_strength:.2f}"
            cv2.putText(frame, strength_text, (strength_bar_x, strength_bar_y + strength_bar_height + 15),
                       self.fonts['small'], self.font_scales['tiny'], voice_color, 1)

            # 活动指示器 (音符符号)
            indicator_x = x2 - 20
            indicator_y = y1 + 20
            cv2.putText(frame, "♪", (indicator_x, indicator_y),
                       self.fonts['body'], self.font_scales['body'], self.colors['active_green'],
                       self.thickness['normal'])

    def _draw_central_podium(self, frame: np.ndarray, central_region: Dict[str, Any],
                           active_regions: Dict[str, Any], start_x: int, start_y: int,
                           available_width: int, available_height: int):
        """绘制中央指挥台"""
        central_bounds = central_region['bounds']
        central_color = self.colors['conducting_gold']

        # 检查中央控制是否激活
        central_active = any(h.get('active_region') == 'central_control' for h in active_regions.values())

        # 坐标转换
        x1 = int(start_x + central_bounds['x1'] * available_width)
        y1 = int(start_y + central_bounds['y1'] * available_height)
        x2 = int(start_x + central_bounds['x2'] * available_width)
        y2 = int(start_y + central_bounds['y2'] * available_height)

        # 指挥台特效
        if central_active:
            # 强烈发光效果
            for radius in range(15, 0, -1):
                alpha = 0.08 * (16 - radius)
                glow_color = tuple(int(c * alpha) for c in central_color)
                cv2.rectangle(frame, (x1 - radius, y1 - radius),
                            (x2 + radius, y2 + radius), glow_color, 1)

            # 脉冲边框
            pulse = 0.6 + 0.4 * abs(np.sin(self.animation_time * 2))
            border_color = tuple(int(c * pulse) for c in central_color)
            line_thickness = 6

            # 激活填充 (金色渐变)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), central_color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        else:
            border_color = tuple(int(c * 0.7) for c in central_color)
            line_thickness = 3

        # 主边框 (圆角)
        self._draw_rounded_rectangle(frame, (x1, y1), (x2, y2), None, border_color, line_thickness)

        # 指挥台图标和文字
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 指挥棒图标 (动态)
        if central_active:
            # 动态指挥棒
            baton_angle = np.sin(self.animation_time * 3) * 0.5
            baton_length = 40
            baton_end_x = center_x + int(baton_length * np.cos(baton_angle))
            baton_end_y = center_y - int(baton_length * np.sin(baton_angle))

            cv2.line(frame, (center_x, center_y), (baton_end_x, baton_end_y),
                    self.colors['text_primary'], 4)
            cv2.circle(frame, (center_x, center_y), 6, central_color, -1)
            cv2.circle(frame, (baton_end_x, baton_end_y), 3, self.colors['text_primary'], -1)

        # 指挥台标签
        label = "🎼 CONDUCTOR PODIUM 🎼" if central_active else "🎼 Conductor Podium"
        label_size = cv2.getTextSize(label, self.fonts['subtitle'], self.font_scales['subtitle'],
                                   self.thickness['normal'])[0]
        label_x = center_x - label_size[0] // 2
        label_y = y2 - 20

        # 标签背景
        self._draw_rounded_rectangle(frame, (label_x - 8, label_y - 25),
                                   (label_x + label_size[0] + 8, label_y + 5),
                                   self.colors['bg_dark'], central_color)

        cv2.putText(frame, label, (label_x, label_y),
                   self.fonts['subtitle'], self.font_scales['subtitle'], central_color,
                   self.thickness['normal'])

    def _draw_professional_audio_meters(self, frame: np.ndarray, track_volumes: Dict[str, float],
                                      audio_controller=None):
        """绘制专业音频电平表"""
        height, width = frame.shape[:2]
        sidebar_width = self.layout['sidebar_width']
        header_height = self.layout['header_height']
        footer_height = self.layout['footer_height']

        # 音频面板位置 (右侧边栏)
        panel_x = width - sidebar_width
        panel_y = header_height + 20
        panel_width = sidebar_width - 20
        panel_height = height - header_height - footer_height - 40

        # 面板背景 (专业音频设备风格)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['bg_panel'], -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

        # 面板边框
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['border_accent'], 2)

        # 标题
        title = "🎚️ AUDIO MIXING CONSOLE"
        title_y = panel_y + 25
        cv2.putText(frame, title, (panel_x + 10, title_y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_accent'], self.thickness['normal'])

        # 主输出电平
        master_y = title_y + 30
        if audio_controller:
            try:
                master_volume = max(track_volumes.values()) if track_volumes else 0.0
                self._draw_master_level_meter(frame, panel_x + 10, master_y,
                                            panel_width - 20, master_volume)
            except:
                pass

        # 个别声部电平表
        meters_start_y = master_y + 80
        meter_height = 35
        meter_spacing = 40

        # 更新历史数据
        self._update_volume_history(track_volumes)

        # 绘制每个声部的专业电平表
        for i, (track_name, volume) in enumerate(track_volumes.items()):
            meter_y = meters_start_y + i * meter_spacing

            if meter_y + meter_height > panel_y + panel_height - 60:
                break  # 超出面板范围

            self._draw_professional_voice_meter(frame, panel_x + 10, meter_y,
                                               panel_width - 20, meter_height,
                                               track_name, volume, i)

        # 音频统计信息
        self._draw_audio_statistics(frame, panel_x + 10, panel_y + panel_height - 50,
                                   panel_width - 20, track_volumes)

    def _draw_master_level_meter(self, frame: np.ndarray, x: int, y: int, width: int, volume: float):
        """绘制主输出电平表"""
        meter_height = 50

        # 背景
        cv2.rectangle(frame, (x, y), (x + width, y + meter_height), self.colors['level_bg'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + meter_height), self.colors['border_accent'], 1)

        # 专业分段电平显示
        segments = 24
        segment_width = (width - 8) / segments
        segment_height = meter_height - 8

        for i in range(segments):
            segment_x = x + 4 + int(i * segment_width)
            segment_level = (i + 1) / segments

            if volume >= segment_level:
                # 根据电平段选择颜色
                if segment_level < 0.6:
                    color = self.colors['level_low']
                elif segment_level < 0.8:
                    color = self.colors['level_mid']
                elif segment_level < 0.9:
                    color = self.colors['level_high']
                else:
                    color = self.colors['level_peak']

                cv2.rectangle(frame, (segment_x, y + 4),
                             (segment_x + int(segment_width) - 1, y + 4 + segment_height),
                             color, -1)

        # 数值显示
        volume_text = f"MASTER: {volume:.3f}"
        cv2.putText(frame, volume_text, (x, y + meter_height + 18),
                   self.fonts['mono'], self.font_scales['small'],
                   self.colors['text_accent'], self.thickness['normal'])

    def _draw_professional_voice_meter(self, frame: np.ndarray, x: int, y: int, width: int,
                                     height: int, track_name: str, volume: float, index: int):
        """绘制专业声部电平表"""
        # 声部颜色
        voice_color = self.colors['voice_colors'][index % len(self.colors['voice_colors'])]

        # 背景
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['level_bg'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), voice_color, 1)

        # 电平条
        bar_width = width - 80
        bar_x = x + 70
        bar_height = height - 8

        # 背景条
        cv2.rectangle(frame, (bar_x, y + 4), (bar_x + bar_width, y + 4 + bar_height),
                     self.colors['bg_dark'], -1)

        # 音量填充 (带渐变)
        if volume > 0:
            fill_width = int(bar_width * volume)

            # 创建渐变填充
            for i in range(fill_width):
                progress = i / bar_width
                if progress < 0.7:
                    color = voice_color
                elif progress < 0.9:
                    color = self.colors['level_high']
                else:
                    color = self.colors['level_peak']

                cv2.line(frame, (bar_x + i, y + 4), (bar_x + i, y + 4 + bar_height), color, 1)

            # 峰值指示
            if volume > 0.9:
                peak_x = bar_x + fill_width - 3
                cv2.rectangle(frame, (peak_x, y + 4), (peak_x + 3, y + 4 + bar_height),
                             self.colors['level_peak'], -1)

        # 声部编号和名称
        voice_num = str(index + 1)
        cv2.putText(frame, voice_num, (x + 8, y + height - 8),
                   self.fonts['mono'], self.font_scales['body'], voice_color, self.thickness['bold'])

        # 简化的轨道名
        short_name = track_name.split('_')[0][:8]
        cv2.putText(frame, short_name, (x + 25, y + height - 8),
                   self.fonts['small'], self.font_scales['tiny'], self.colors['text_secondary'], 1)

        # 音量数值
        volume_text = f"{volume:.2f}"
        cv2.putText(frame, volume_text, (bar_x + bar_width + 5, y + height - 8),
                   self.fonts['mono'], self.font_scales['tiny'], self.colors['text_primary'], 1)

        # 峰值保持指示器
        if volume > 0.8:
            cv2.circle(frame, (x + 50, y + height // 2), 3, self.colors['level_peak'], -1)

    def _update_volume_history(self, track_volumes: Dict[str, float]):
        """更新音量历史数据"""
        current_time = time.time()
        for track_name, volume in track_volumes.items():
            if track_name not in self.history['volume_levels']:
                self.history['volume_levels'][track_name] = deque(maxlen=60)  # 1秒历史

            self.history['volume_levels'][track_name].append((current_time, volume))

    def _draw_audio_statistics(self, frame: np.ndarray, x: int, y: int, width: int,
                             track_volumes: Dict[str, float]):
        """绘制音频统计信息"""
        active_tracks = sum(1 for v in track_volumes.values() if v > 0.1)
        avg_volume = sum(track_volumes.values()) / len(track_volumes) if track_volumes else 0.0
        peak_volume = max(track_volumes.values()) if track_volumes else 0.0

        stats = [
            f"Active: {active_tracks}/{len(track_volumes)}",
            f"Average: {avg_volume:.2f}",
            f"Peak: {peak_volume:.2f}"
        ]

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (x, y + i * 15),
                       self.fonts['small'], self.font_scales['tiny'],
                       self.colors['text_secondary'], 1)

    def _draw_conducting_analysis_panel(self, frame: np.ndarray, gesture_data: Dict,
                                      active_regions: Dict[str, Any]):
        """绘制指挥分析面板"""
        height, width = frame.shape[:2]
        footer_height = self.layout['footer_height']
        panel_y = height - footer_height

        # 面板背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), self.colors['bg_panel'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # 顶部分隔线
        cv2.line(frame, (0, panel_y), (width, panel_y), self.colors['border_accent'], 2)

        # 三个分析区域
        section_width = width // 3

        # 1. 手势统计分析 (左侧)
        self._draw_gesture_statistics_advanced(frame, 10, panel_y + 10,
                                             section_width - 20, footer_height - 20,
                                             gesture_data, active_regions)

        # 2. 指挥技巧分析 (中央)
        self._draw_conducting_technique_analysis(frame, section_width + 10, panel_y + 10,
                                               section_width - 20, footer_height - 20,
                                               gesture_data, active_regions)

        # 3. 系统控制面板 (右侧)
        self._draw_advanced_system_controls(frame, 2 * section_width + 10, panel_y + 10,
                                          section_width - 20, footer_height - 20)

    def _draw_gesture_statistics_advanced(self, frame: np.ndarray, x: int, y: int,
                                        width: int, height: int, gesture_data: Dict,
                                        active_regions: Dict[str, Any]):
        """绘制高级手势统计"""
        # 标题
        cv2.putText(frame, "🤲 GESTURE ANALYTICS", (x, y + 20),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_accent'], self.thickness['normal'])

        stats_y = y + 45
        line_height = 20

        # 统计数据
        hands_count = gesture_data.get('hands_detected', 0)
        active_count = len(active_regions)

        stats = [
            f"Hands Detected: {hands_count}",
            f"Active Regions: {active_count}/7",
            f"Detection Rate: {hands_count * 100 / 2:.0f}%",  # 最多2只手
        ]

        # 手势强度历史
        if gesture_data.get('hands', []):
            avg_openness = sum(hand.get('openness', 0) for hand in gesture_data['hands']) / len(gesture_data['hands'])
            self.history['gesture_strength'].append(avg_openness)
            stats.append(f"Gesture Intensity: {avg_openness:.2f}")

        # 绘制统计文本
        for i, stat in enumerate(stats):
            color = self.colors['active_green'] if i == 0 and hands_count > 0 else self.colors['text_secondary']
            cv2.putText(frame, stat, (x, stats_y + i * line_height),
                       self.fonts['small'], self.font_scales['small'], color, 1)

        # 手势强度图表 (迷你图表)
        if len(self.history['gesture_strength']) > 1:
            self._draw_mini_chart(frame, x, stats_y + len(stats) * line_height + 10,
                                width - 20, 30, list(self.history['gesture_strength']),
                                self.colors['active_green'])

    def _draw_conducting_technique_analysis(self, frame: np.ndarray, x: int, y: int,
                                          width: int, height: int, gesture_data: Dict,
                                          active_regions: Dict[str, Any]):
        """绘制指挥技巧分析"""
        # 标题
        cv2.putText(frame, "🎼 CONDUCTING ANALYSIS", (x, y + 20),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_accent'], self.thickness['normal'])

        analysis_y = y + 45
        line_height = 18

        # 分析指挥技巧
        conducting_detected = len(active_regions) > 0
        technique_score = self._calculate_technique_score(gesture_data, active_regions)

        analysis = [
            f"Conducting Mode: {'ACTIVE' if conducting_detected else 'STANDBY'}",
            f"Technique Score: {technique_score:.1f}/10",
            f"Coordination: {'EXCELLENT' if technique_score > 8 else 'GOOD' if technique_score > 6 else 'DEVELOPING'}",
        ]

        # 音乐表达分析
        if active_regions:
            dynamic_range = self._calculate_dynamic_range(active_regions)
            analysis.append(f"Dynamic Range: {dynamic_range:.2f}")

            balance_score = self._calculate_balance_score(active_regions)
            analysis.append(f"Balance Score: {balance_score:.1f}/10")

        # 绘制分析文本
        for i, text in enumerate(analysis):
            color = self.colors['conducting_gold'] if conducting_detected and i == 0 else self.colors['text_secondary']
            cv2.putText(frame, text, (x, analysis_y + i * line_height),
                       self.fonts['small'], self.font_scales['small'], color, 1)

        # 技巧评分可视化
        score_bar_y = analysis_y + len(analysis) * line_height + 15
        self._draw_score_bar(frame, x, score_bar_y, width - 20, 12, technique_score / 10,
                           self.colors['conducting_gold'])

    def _draw_advanced_system_controls(self, frame: np.ndarray, x: int, y: int,
                                     width: int, height: int):
        """绘制高级系统控制面板"""
        # 标题
        cv2.putText(frame, "⚙️ SYSTEM CONTROLS", (x, y + 20),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_accent'], self.thickness['normal'])

        controls_y = y + 45
        line_height = 16

        # 控制提示 (分类显示)
        controls = [
            "PLAYBACK:",
            "  SPACE - Play/Pause",
            "  R - Reset System",
            "",
            "DISPLAY:",
            "  D - Debug Toggle",
            "  F - Fullscreen",
            "  H - Help Toggle",
            "",
            "VOLUME:",
            "  1-9 - Direct Control",
            "  0 - Mute All"
        ]

        for i, control in enumerate(controls):
            if control.startswith("  "):
                color = self.colors['text_secondary']
                font_scale = self.font_scales['tiny']
            elif control.endswith(":"):
                color = self.colors['text_accent']
                font_scale = self.font_scales['small']
            elif control == "":
                continue
            else:
                color = self.colors['text_muted']
                font_scale = self.font_scales['tiny']

            if i * line_height < height - 60:
                cv2.putText(frame, control, (x, controls_y + i * line_height),
                           self.fonts['small'], font_scale, color, 1)

    def _draw_performance_panel(self, frame: np.ndarray, performance_metrics):
        """绘制性能监控面板"""
        # 在右上角绘制小型性能面板
        height, width = frame.shape[:2]
        panel_x = width - 200
        panel_y = 130  # 在header下方

        # 简洁的性能显示
        perf_data = [
            f"FPS: {performance_metrics.fps:.1f}",
            f"Latency: {performance_metrics.gesture_latency:.1f}ms",
            f"Memory: {performance_metrics.memory_usage:.1f}%"
        ]

        # 更新FPS历史
        self.history['fps_history'].append(performance_metrics.fps)

        for i, data in enumerate(perf_data):
            color = self._get_performance_color(performance_metrics.fps if i == 0 else
                                              (50 - performance_metrics.gesture_latency) if i == 1 else
                                              (100 - performance_metrics.memory_usage),
                                              30 if i == 0 else 0 if i == 1 else 0,
                                              60 if i == 0 else 50 if i == 1 else 80)

            cv2.putText(frame, data, (panel_x, panel_y + i * 18),
                       self.fonts['mono'], self.font_scales['tiny'], color, 1)

    def _draw_gesture_trails_and_effects(self, frame: np.ndarray, gesture_data: Dict,
                                       active_regions: Dict[str, Any]):
        """绘制手势轨迹和特效"""
        # 更新手部位置历史
        if gesture_data.get('hands', []):
            for hand in gesture_data['hands']:
                hand_pos = hand.get('position', (0, 0))
                self.history['hand_positions'].append(hand_pos)

        # 绘制手势轨迹 (如果有历史位置)
        if len(self.history['hand_positions']) > 1:
            positions = list(self.history['hand_positions'])
            height, width = frame.shape[:2]

            for i in range(1, len(positions)):
                if i < len(positions) - 10:  # 只显示最近10个点
                    continue

                alpha = (i - (len(positions) - 10)) / 10  # 渐变透明度

                pos1 = (int(positions[i-1][0] * width), int(positions[i-1][1] * height))
                pos2 = (int(positions[i][0] * width), int(positions[i][1] * height))

                trail_color = tuple(int(c * alpha) for c in self.colors['active_green'])
                cv2.line(frame, pos1, pos2, trail_color, 2)

    def _draw_system_control_panel(self, frame: np.ndarray):
        """绘制系统控制面板"""
        # 这个方法可以用来绘制额外的系统控制元素
        # 如快捷键提示、状态指示器等
        pass

    # 辅助方法
    def _draw_rounded_rectangle(self, frame: np.ndarray, pt1: Tuple[int, int],
                              pt2: Tuple[int, int], fill_color: Optional[Tuple[int, int, int]] = None,
                              border_color: Optional[Tuple[int, int, int]] = None,
                              thickness: int = 1, radius: int = 8):
        """绘制圆角矩形"""
        x1, y1 = pt1
        x2, y2 = pt2

        if fill_color:
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), fill_color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), fill_color, -1)
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, fill_color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, fill_color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, fill_color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, fill_color, -1)

        if border_color:
            # 绘制圆角边框 (简化版)
            cv2.rectangle(frame, pt1, pt2, border_color, thickness)

    def _draw_text_with_glow(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                           font, font_scale: float, color: Tuple[int, int, int], thickness: int):
        """绘制带发光效果的文字"""
        x, y = position

        # 发光效果 (多层阴影)
        glow_offsets = [(2, 2), (1, 1), (-1, -1), (-2, -2), (2, -2), (-2, 2)]
        glow_color = tuple(int(c * 0.3) for c in color)

        for offset in glow_offsets:
            cv2.putText(frame, text, (x + offset[0], y + offset[1]),
                       font, font_scale, glow_color, thickness)

        # 主文字
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def _interpolate_color(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int],
                          factor: float) -> Tuple[int, int, int]:
        """在两个颜色之间插值"""
        factor = max(0.0, min(1.0, factor))
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

    def _get_performance_color(self, value: float, min_good: float, max_good: float) -> Tuple[int, int, int]:
        """根据性能值获取颜色"""
        if min_good <= value <= max_good:
            return self.colors['active_green']
        elif value >= min_good * 0.7:
            return self.colors['warning_orange']
        else:
            return self.colors['error_red']

    def _draw_mini_chart(self, frame: np.ndarray, x: int, y: int, width: int, height: int,
                        data: List[float], color: Tuple[int, int, int]):
        """绘制迷你图表"""
        if len(data) < 2:
            return

        # 标准化数据
        max_val = max(data) if max(data) > 0 else 1.0
        min_val = min(data)
        range_val = max_val - min_val if max_val > min_val else 1.0

        # 绘制背景
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['bg_dark'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)

        # 绘制数据线
        step = width / (len(data) - 1)
        for i in range(1, len(data)):
            x1 = int(x + (i - 1) * step)
            y1 = int(y + height - ((data[i-1] - min_val) / range_val * height))
            x2 = int(x + i * step)
            y2 = int(y + height - ((data[i] - min_val) / range_val * height))

            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _draw_score_bar(self, frame: np.ndarray, x: int, y: int, width: int, height: int,
                       score: float, color: Tuple[int, int, int]):
        """绘制评分条"""
        # 背景
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['bg_dark'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['border_accent'], 1)

        # 评分填充
        fill_width = int(width * score)
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    def _calculate_technique_score(self, gesture_data: Dict, active_regions: Dict) -> float:
        """计算指挥技巧评分"""
        score = 5.0  # 基础分

        # 手势检测质量
        hands_count = gesture_data.get('hands_detected', 0)
        if hands_count > 0:
            score += 2.0

        # 区域激活数量
        active_count = len(active_regions)
        if active_count > 0:
            score += min(3.0, active_count * 0.5)

        return min(10.0, score)

    def _calculate_dynamic_range(self, active_regions: Dict) -> float:
        """计算动态范围"""
        if not active_regions:
            return 0.0

        strengths = [data.get('activation_strength', 0.0) for data in active_regions.values()]
        return max(strengths) - min(strengths) if len(strengths) > 1 else 0.0

    def _calculate_balance_score(self, active_regions: Dict) -> float:
        """计算平衡评分"""
        if not active_regions:
            return 0.0

        # 简化的平衡评分算法
        strengths = [data.get('activation_strength', 0.0) for data in active_regions.values()]
        avg_strength = sum(strengths) / len(strengths)
        variance = sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)

        # 方差越小，平衡性越好
        balance_score = max(0.0, 10.0 - variance * 10)
        return min(10.0, balance_score)

    def _draw_voice_connections(self, frame: np.ndarray, voice_regions: Dict[str, Any],
                              active_regions: Dict[str, Any], start_x: int, start_y: int,
                              available_width: int, available_height: int):
        """绘制声部间连接线 (和声关系可视化)"""
        # 只在有活动声部时显示连接
        active_voice_positions = []

        for region_name, region_data in voice_regions.items():
            if region_name in active_regions:
                bounds = region_data['bounds']
                center_x = int(start_x + (bounds['x1'] + bounds['x2']) * available_width / 2)
                center_y = int(start_y + (bounds['y1'] + bounds['y2']) * available_height / 2)
                active_voice_positions.append((center_x, center_y))

        # 绘制连接线
        if len(active_voice_positions) > 1:
            for i, pos1 in enumerate(active_voice_positions):
                for j, pos2 in enumerate(active_voice_positions[i+1:], i+1):
                    # 半透明连接线，带动画效果
                    alpha = 0.3 + 0.2 * abs(np.sin(self.animation_time + i + j))
                    line_color = tuple(int(c * alpha) for c in self.colors['active_green'])
                    cv2.line(frame, pos1, pos2, line_color, 1)