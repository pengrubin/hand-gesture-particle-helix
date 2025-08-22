#!/usr/bin/env python3
"""
音频可视化器
实时显示MP3文件的音调高低和频谱分析
"""

import pygame
import numpy as np
import threading
import time
import math
import os
from typing import Dict, List, Tuple

class AudioVisualizer:
    def __init__(self):
        print("初始化音频可视化器...")
        
        # 初始化pygame
        pygame.init()
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 显示设置
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("音频可视化器 - 音调高低分析")
        
        # 字体
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # 音频文件
        self.audio_files = {
            "violin": "Fugue in G Trio violin-Violin.mp3",
            "lute": "Fugue in G Trio-Tenor_Lute.mp3",
            "organ": "Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 音频对象
        self.audio_sounds = {}
        self.audio_channels = {}
        
        # 颜色定义
        self.colors = {
            "violin": (255, 100, 100),    # 红色 - 小提琴
            "lute": (100, 255, 100),      # 绿色 - 鲁特琴
            "organ": (100, 100, 255),     # 蓝色 - 管风琴
            "background": (20, 20, 30),
            "grid": (60, 60, 60),
            "text": (255, 255, 255)
        }
        
        # 频谱分析参数
        self.sample_rate = 22050
        self.fft_size = 1024
        self.freq_bands = 64  # 显示的频率段数
        
        # 音调映射 (Hz)
        self.pitch_ranges = {
            "ultra_low": (20, 80),      # 极低音
            "low": (80, 250),           # 低音
            "mid_low": (250, 500),      # 中低音
            "mid": (500, 1000),         # 中音
            "mid_high": (1000, 2000),   # 中高音
            "high": (2000, 4000),       # 高音
            "ultra_high": (4000, 8000)  # 极高音
        }
        
        # 控制状态
        self.playing = {"violin": False, "lute": False, "organ": False}
        self.volumes = {"violin": 0.7, "lute": 0.7, "organ": 0.7}
        self.solo_mode = None  # None 或 "violin"/"lute"/"organ"
        
        # 频谱数据
        self.spectrum_data = {"violin": [], "lute": [], "organ": []}
        self.pitch_levels = {"violin": {}, "lute": {}, "organ": {}}
        
        # 运行状态
        self.running = True
        self.clock = pygame.time.Clock()
        
        print("音频可视化器初始化完成")
    
    def initialize_audio(self):
        """初始化音频文件"""
        print("加载音频文件...")
        
        for name, file_path in self.audio_files.items():
            if os.path.exists(file_path):
                try:
                    sound = pygame.mixer.Sound(file_path)
                    self.audio_sounds[name] = sound
                    self.audio_channels[name] = pygame.mixer.Channel(["violin", "lute", "organ"].index(name))
                    print(f"✅ 加载成功: {name}")
                except Exception as e:
                    print(f"❌ 加载失败 {name}: {e}")
            else:
                print(f"❌ 文件不存在: {file_path}")
        
        return len(self.audio_sounds) > 0
    
    def generate_mock_spectrum(self, instrument: str, time_offset: float) -> List[float]:
        """生成模拟的频谱数据（基于乐器特性）"""
        spectrum = [0.0] * self.freq_bands
        
        # 不同乐器的频率特征
        if instrument == "violin":
            # 小提琴：高频较强
            for i in range(self.freq_bands):
                freq = (i / self.freq_bands) * (self.sample_rate / 2)
                if 200 <= freq <= 3000:  # 小提琴主要频率范围
                    intensity = math.sin(time_offset * 2 + i * 0.1) * 0.5 + 0.5
                    if freq > 1000:  # 高频增强
                        intensity *= 1.5
                    spectrum[i] = max(0, min(1, intensity))
        
        elif instrument == "lute":
            # 鲁特琴：中频为主
            for i in range(self.freq_bands):
                freq = (i / self.freq_bands) * (self.sample_rate / 2)
                if 100 <= freq <= 2000:  # 鲁特琴主要频率范围
                    intensity = math.sin(time_offset * 1.5 + i * 0.15) * 0.4 + 0.4
                    if 300 <= freq <= 1200:  # 中频增强
                        intensity *= 1.3
                    spectrum[i] = max(0, min(1, intensity))
        
        elif instrument == "organ":
            # 管风琴：低频强，全频谱
            for i in range(self.freq_bands):
                freq = (i / self.freq_bands) * (self.sample_rate / 2)
                if 50 <= freq <= 4000:  # 管风琴宽频率范围
                    intensity = math.sin(time_offset * 1.0 + i * 0.08) * 0.6 + 0.3
                    if freq < 500:  # 低频增强
                        intensity *= 1.8
                    spectrum[i] = max(0, min(1, intensity))
        
        return spectrum
    
    def analyze_pitch_levels(self, spectrum: List[float], instrument: str):
        """分析频谱数据中的音调层次"""
        pitch_levels = {}
        
        for range_name, (low_freq, high_freq) in self.pitch_ranges.items():
            # 计算频率范围对应的频谱索引
            low_idx = int((low_freq / (self.sample_rate / 2)) * self.freq_bands)
            high_idx = int((high_freq / (self.sample_rate / 2)) * self.freq_bands)
            
            # 计算该频率范围的平均强度
            if low_idx < len(spectrum) and high_idx <= len(spectrum):
                range_spectrum = spectrum[low_idx:high_idx]
                if range_spectrum:
                    avg_intensity = sum(range_spectrum) / len(range_spectrum)
                    pitch_levels[range_name] = avg_intensity
                else:
                    pitch_levels[range_name] = 0.0
            else:
                pitch_levels[range_name] = 0.0
        
        self.pitch_levels[instrument] = pitch_levels
    
    def update_audio_analysis(self):
        """更新音频分析数据"""
        current_time = time.time()
        
        for instrument in ["violin", "lute", "organ"]:
            if self.playing[instrument] or True:  # 始终显示模拟数据
                # 生成模拟频谱数据
                spectrum = self.generate_mock_spectrum(instrument, current_time)
                self.spectrum_data[instrument] = spectrum
                
                # 分析音调层次
                self.analyze_pitch_levels(spectrum, instrument)
    
    def draw_spectrum_bars(self):
        """绘制频谱柱状图"""
        bar_width = self.width // (self.freq_bands * 3)  # 三个乐器并排
        max_height = 200
        start_y = self.height - max_height - 50
        
        # 绘制每个乐器的频谱
        for idx, (instrument, spectrum) in enumerate(self.spectrum_data.items()):
            if not spectrum:
                continue
            
            color = self.colors[instrument]
            x_offset = idx * (self.width // 3)
            
            # 绘制频谱柱
            for i, amplitude in enumerate(spectrum):
                if amplitude > 0.01:  # 只绘制有信号的频率
                    bar_height = int(amplitude * max_height)
                    x = x_offset + i * bar_width
                    y = start_y + max_height - bar_height
                    
                    # 根据频率调整颜色亮度
                    freq = (i / self.freq_bands) * (self.sample_rate / 2)
                    brightness = 0.3 + amplitude * 0.7
                    draw_color = tuple(int(c * brightness) for c in color)
                    
                    pygame.draw.rect(self.screen, draw_color, 
                                   (x, y, bar_width-1, bar_height))
            
            # 绘制乐器标签
            label = self.font_medium.render(instrument.upper(), True, color)
            self.screen.blit(label, (x_offset + 10, start_y - 30))
    
    def draw_pitch_analysis(self):
        """绘制音调高低分析"""
        panel_width = 350
        panel_height = 500
        start_x = self.width - panel_width - 20
        start_y = 50
        
        # 绘制背景
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (start_x, start_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (start_x, start_y, panel_width, panel_height), 2)
        
        # 标题
        title = self.font_large.render("音调分析", True, self.colors["text"])
        self.screen.blit(title, (start_x + 10, start_y + 10))
        
        # 音调范围标签
        pitch_labels = {
            "ultra_high": "极高音 (4K-8K Hz)",
            "high": "高音 (2K-4K Hz)",
            "mid_high": "中高音 (1K-2K Hz)",
            "mid": "中音 (500-1K Hz)",
            "mid_low": "中低音 (250-500 Hz)",
            "low": "低音 (80-250 Hz)",
            "ultra_low": "极低音 (20-80 Hz)"
        }
        
        y_offset = 60
        bar_height = 25
        bar_spacing = 35
        
        for pitch_range, label in pitch_labels.items():
            y = start_y + y_offset
            
            # 绘制标签
            text = self.font_small.render(label, True, self.colors["text"])
            self.screen.blit(text, (start_x + 10, y))
            
            # 绘制三个乐器的强度条
            for idx, instrument in enumerate(["violin", "lute", "organ"]):
                intensity = self.pitch_levels.get(instrument, {}).get(pitch_range, 0)
                
                bar_x = start_x + 180 + idx * 50
                bar_y = y + 3
                bar_w = 40
                bar_h = 15
                
                # 背景
                pygame.draw.rect(self.screen, (60, 60, 60), 
                               (bar_x, bar_y, bar_w, bar_h))
                
                # 强度条
                if intensity > 0.01:
                    fill_width = int(intensity * bar_w)
                    color = self.colors[instrument]
                    pygame.draw.rect(self.screen, color, 
                                   (bar_x, bar_y, fill_width, bar_h))
                
                # 强度数值
                if intensity > 0.1:
                    value_text = self.font_small.render(f"{intensity:.1f}", True, self.colors["text"])
                    self.screen.blit(value_text, (bar_x, bar_y + bar_h + 2))
            
            y_offset += bar_spacing
    
    def draw_controls(self):
        """绘制控制面板"""
        panel_width = 300
        panel_height = 200
        start_x = 20
        start_y = 20
        
        # 绘制背景
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (start_x, start_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (start_x, start_y, panel_width, panel_height), 2)
        
        # 标题
        title = self.font_large.render("控制面板", True, self.colors["text"])
        self.screen.blit(title, (start_x + 10, start_y + 10))
        
        # 控制说明
        controls = [
            "1 - 小提琴 (红色)",
            "2 - 鲁特琴 (绿色)", 
            "3 - 管风琴 (蓝色)",
            "A - 播放所有",
            "S - 停止所有",
            "ESC - 退出"
        ]
        
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, self.colors["text"])
            self.screen.blit(text, (start_x + 10, start_y + 50 + i * 20))
        
        # 播放状态
        status_y = start_y + 170
        for idx, (instrument, playing) in enumerate(self.playing.items()):
            color = self.colors[instrument]
            status = "播放中" if playing else "已停止"
            text = self.font_small.render(f"{instrument}: {status}", True, color)
            self.screen.blit(text, (start_x + 10, status_y + idx * 15))
    
    def toggle_instrument(self, instrument: str):
        """切换乐器播放状态"""
        if instrument in self.audio_sounds:
            channel = self.audio_channels[instrument]
            
            if self.playing[instrument]:
                # 停止播放
                channel.stop()
                self.playing[instrument] = False
                print(f"停止 {instrument}")
            else:
                # 开始播放
                sound = self.audio_sounds[instrument]
                sound.set_volume(self.volumes[instrument])
                channel.play(sound, loops=-1)  # 循环播放
                self.playing[instrument] = True
                print(f"播放 {instrument}")
    
    def play_all(self):
        """播放所有乐器"""
        for instrument in ["violin", "lute", "organ"]:
            if not self.playing[instrument]:
                self.toggle_instrument(instrument)
    
    def stop_all(self):
        """停止所有乐器"""
        for instrument in ["violin", "lute", "organ"]:
            if self.playing[instrument]:
                self.toggle_instrument(instrument)
    
    def handle_events(self):
        """处理用户输入"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_1:
                    self.toggle_instrument("violin")
                elif event.key == pygame.K_2:
                    self.toggle_instrument("lute")
                elif event.key == pygame.K_3:
                    self.toggle_instrument("organ")
                elif event.key == pygame.K_a:
                    self.play_all()
                elif event.key == pygame.K_s:
                    self.stop_all()
    
    def run(self):
        """主运行循环"""
        print("\n🎵 音频可视化器启动")
        print("="*50)
        print("控制说明:")
        print("1 - 切换小提琴")
        print("2 - 切换鲁特琴") 
        print("3 - 切换管风琴")
        print("A - 播放所有")
        print("S - 停止所有")
        print("ESC - 退出")
        print()
        
        # 初始化音频
        if not self.initialize_audio():
            print("❌ 音频初始化失败")
            return
        
        try:
            while self.running:
                # 处理事件
                self.handle_events()
                
                # 更新音频分析
                self.update_audio_analysis()
                
                # 绘制界面
                self.screen.fill(self.colors["background"])
                
                self.draw_controls()
                self.draw_spectrum_bars()
                self.draw_pitch_analysis()
                
                # 更新显示
                pygame.display.flip()
                self.clock.tick(30)  # 30 FPS
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        self.running = False
        
        # 停止所有音频
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except:
            pass
        
        try:
            pygame.quit()
        except:
            pass
        
        print("清理完成")


def main():
    try:
        visualizer = AudioVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()