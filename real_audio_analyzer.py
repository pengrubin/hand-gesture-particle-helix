#!/usr/bin/env python3
"""
真实音频分析器
使用librosa分析MP3文件的真实频谱数据
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import threading
from pathlib import Path

# 尝试导入音频分析库
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
    print("✅ librosa 可用 - 将进行真实音频分析")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa 不可用 - 将使用模拟分析")
    print("安装命令: pip install librosa")

class RealAudioAnalyzer:
    def __init__(self):
        print("初始化真实音频分析器...")
        
        # 初始化pygame
        pygame.init()
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 显示设置
        self.width = 1400
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("真实音频分析器 - MP3音调高低实时分析")
        
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
        
        # 颜色定义
        self.colors = {
            "violin": (255, 120, 120),    # 红色
            "lute": (120, 255, 120),      # 绿色
            "organ": (120, 120, 255),     # 蓝色
            "background": (25, 25, 35),
            "panel": (40, 40, 50),
            "grid": (70, 70, 70),
            "text": (255, 255, 255),
            "accent": (255, 255, 100)
        }
        
        # 音频数据
        self.audio_data = {}
        self.sample_rates = {}
        self.durations = {}
        
        # 播放控制
        self.audio_sounds = {}
        self.audio_channels = {}
        self.playing = {"violin": False, "lute": False, "organ": False}
        self.current_time = 0.0
        self.start_time = None
        
        # 分析参数
        self.fft_size = 2048
        self.hop_length = 512
        self.freq_bins = 128
        
        # 音调范围定义 (Hz)
        self.pitch_ranges = {
            "Sub Bass": (20, 60),        # 超低音
            "Bass": (60, 250),           # 低音  
            "Low Mid": (250, 500),       # 中低音
            "Mid": (500, 2000),          # 中音
            "High Mid": (2000, 4000),    # 中高音
            "Presence": (4000, 8000),    # 临场感
            "Brilliance": (8000, 20000) # 明亮度
        }
        
        # 实时分析数据
        self.current_spectrum = {"violin": None, "lute": None, "organ": None}
        self.pitch_levels = {"violin": {}, "lute": {}, "organ": {}}
        self.peak_frequencies = {"violin": [], "lute": [], "organ": []}
        
        # 控制状态
        self.running = True
        self.clock = pygame.time.Clock()
        self.analysis_enabled = True
        
        print("真实音频分析器初始化完成")
    
    def load_audio_files(self):
        """加载并分析音频文件"""
        print("加载音频文件...")
        
        for name, file_path in self.audio_files.items():
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            try:
                # 加载pygame声音
                sound = pygame.mixer.Sound(file_path)
                self.audio_sounds[name] = sound
                self.audio_channels[name] = pygame.mixer.Channel(["violin", "lute", "organ"].index(name))
                
                if LIBROSA_AVAILABLE:
                    # 使用librosa加载音频数据
                    y, sr = librosa.load(file_path, sr=22050)
                    self.audio_data[name] = y
                    self.sample_rates[name] = sr
                    self.durations[name] = len(y) / sr
                    
                    print(f"✅ {name}: {self.durations[name]:.1f}秒, {sr}Hz")
                else:
                    # 估算音频长度
                    self.durations[name] = 180.0  # 假设3分钟
                    print(f"✅ {name}: 已加载 (模拟分析)")
                
            except Exception as e:
                print(f"❌ 加载失败 {name}: {e}")
        
        return len(self.audio_sounds) > 0
    
    def analyze_audio_at_time(self, instrument: str, time_pos: float):
        """分析指定时间点的音频频谱"""
        if not LIBROSA_AVAILABLE or instrument not in self.audio_data:
            return self.generate_mock_analysis(instrument, time_pos)
        
        try:
            y = self.audio_data[instrument]
            sr = self.sample_rates[instrument]
            
            # 计算时间对应的样本位置
            sample_pos = int(time_pos * sr)
            
            # 确保不超出音频长度
            if sample_pos >= len(y):
                sample_pos = sample_pos % len(y)  # 循环
            
            # 提取一小段音频进行分析
            window_size = self.fft_size
            start = max(0, sample_pos - window_size // 2)
            end = min(len(y), start + window_size)
            
            if end - start < window_size:
                # 如果音频段太短，用零填充
                audio_segment = np.zeros(window_size)
                audio_segment[:end-start] = y[start:end]
            else:
                audio_segment = y[start:end]
            
            # 计算频谱
            fft = np.fft.fft(audio_segment)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # 分析各频率范围的能量
            pitch_analysis = {}
            for range_name, (low_freq, high_freq) in self.pitch_ranges.items():
                # 找到频率范围对应的索引
                freq_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
                if np.any(freq_mask):
                    energy = np.mean(positive_magnitude[freq_mask])
                    # 归一化到0-1范围
                    pitch_analysis[range_name] = min(1.0, energy / (np.max(positive_magnitude) + 1e-10))
                else:
                    pitch_analysis[range_name] = 0.0
            
            # 找到峰值频率
            peak_indices = np.argsort(positive_magnitude)[-5:]  # 前5个峰值
            peak_freqs = [positive_freqs[i] for i in peak_indices if positive_magnitude[i] > np.max(positive_magnitude) * 0.1]
            
            return pitch_analysis, peak_freqs, positive_freqs, positive_magnitude
            
        except Exception as e:
            print(f"音频分析错误 {instrument}: {e}")
            return self.generate_mock_analysis(instrument, time_pos)
    
    def generate_mock_analysis(self, instrument: str, time_pos: float):
        """生成模拟的音频分析数据"""
        pitch_analysis = {}
        
        # 不同乐器的特征频率模拟
        if instrument == "violin":
            # 小提琴：中高频较强
            ranges_strength = {
                "Sub Bass": 0.1, "Bass": 0.2, "Low Mid": 0.4, 
                "Mid": 0.8, "High Mid": 0.9, "Presence": 0.7, "Brilliance": 0.3
            }
        elif instrument == "lute":
            # 鲁特琴：中频为主
            ranges_strength = {
                "Sub Bass": 0.1, "Bass": 0.3, "Low Mid": 0.7, 
                "Mid": 0.9, "High Mid": 0.6, "Presence": 0.3, "Brilliance": 0.1
            }
        else:  # organ
            # 管风琴：低频强，全频谱
            ranges_strength = {
                "Sub Bass": 0.8, "Bass": 0.9, "Low Mid": 0.7, 
                "Mid": 0.6, "High Mid": 0.4, "Presence": 0.3, "Brilliance": 0.2
            }
        
        # 添加时间变化
        for range_name, base_strength in ranges_strength.items():
            variation = np.sin(time_pos * 2 + hash(range_name) % 10) * 0.3
            pitch_analysis[range_name] = max(0, min(1, base_strength + variation))
        
        # 模拟峰值频率
        peak_freqs = [196, 440, 880, 1760]  # 一些音乐频率
        
        return pitch_analysis, peak_freqs, [], []
    
    def update_analysis(self):
        """更新音频分析"""
        if not self.analysis_enabled:
            return
        
        # 计算当前播放时间
        if any(self.playing.values()):
            if self.start_time is None:
                self.start_time = time.time()
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += 1/30  # 模拟时间推进
        
        # 分析每个活跃的乐器
        for instrument in ["violin", "lute", "organ"]:
            if self.playing[instrument] or True:  # 始终分析以便演示
                analysis = self.analyze_audio_at_time(instrument, self.current_time)
                if len(analysis) == 4:
                    pitch_analysis, peak_freqs, freqs, magnitude = analysis
                    self.pitch_levels[instrument] = pitch_analysis
                    self.peak_frequencies[instrument] = peak_freqs
                    
                    # 保存频谱数据用于显示
                    if len(freqs) > 0:
                        # 重新采样到固定数量的频率点
                        if len(magnitude) > self.freq_bins:
                            indices = np.linspace(0, len(magnitude)-1, self.freq_bins, dtype=int)
                            self.current_spectrum[instrument] = magnitude[indices]
                        else:
                            self.current_spectrum[instrument] = magnitude
                else:
                    pitch_analysis, peak_freqs = analysis
                    self.pitch_levels[instrument] = pitch_analysis
                    self.peak_frequencies[instrument] = peak_freqs
    
    def draw_spectrum_display(self):
        """绘制频谱显示"""
        panel_width = 400
        panel_height = 250
        start_x = 50
        start_y = 300
        
        for idx, (instrument, spectrum) in enumerate(self.current_spectrum.items()):
            if spectrum is None or len(spectrum) == 0:
                continue
            
            x = start_x + idx * (panel_width + 20)
            y = start_y
            
            # 绘制背景
            pygame.draw.rect(self.screen, self.colors["panel"], (x, y, panel_width, panel_height))
            pygame.draw.rect(self.screen, self.colors[instrument], (x, y, panel_width, panel_height), 2)
            
            # 绘制标题
            title = self.font_medium.render(f"{instrument.upper()} 频谱", True, self.colors[instrument])
            self.screen.blit(title, (x + 10, y + 10))
            
            # 绘制频谱柱状图
            if len(spectrum) > 0:
                bar_width = (panel_width - 20) // len(spectrum)
                max_height = panel_height - 60
                
                max_val = np.max(spectrum) if np.max(spectrum) > 0 else 1
                
                for i, magnitude in enumerate(spectrum):
                    bar_height = int((magnitude / max_val) * max_height)
                    bar_x = x + 10 + i * bar_width
                    bar_y = y + panel_height - 10 - bar_height
                    
                    # 频率着色
                    freq_ratio = i / len(spectrum)
                    color_intensity = 0.3 + (magnitude / max_val) * 0.7
                    
                    if freq_ratio < 0.3:  # 低频 - 红色调
                        color = (int(255 * color_intensity), int(100 * color_intensity), int(100 * color_intensity))
                    elif freq_ratio < 0.7:  # 中频 - 绿色调
                        color = (int(100 * color_intensity), int(255 * color_intensity), int(100 * color_intensity))
                    else:  # 高频 - 蓝色调
                        color = (int(100 * color_intensity), int(100 * color_intensity), int(255 * color_intensity))
                    
                    pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width-1, bar_height))
    
    def draw_pitch_analysis_panel(self):
        """绘制音调分析面板"""
        panel_width = 450
        panel_height = 600
        start_x = self.width - panel_width - 20
        start_y = 50
        
        # 绘制背景
        pygame.draw.rect(self.screen, self.colors["panel"], (start_x, start_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors["accent"], (start_x, start_y, panel_width, panel_height), 3)
        
        # 标题
        title = self.font_large.render("实时音调分析", True, self.colors["accent"])
        self.screen.blit(title, (start_x + 15, start_y + 15))
        
        # 时间显示
        time_text = self.font_medium.render(f"时间: {self.current_time:.1f}秒", True, self.colors["text"])
        self.screen.blit(time_text, (start_x + 15, start_y + 50))
        
        # 分析每个频率范围
        y_offset = 90
        range_height = 60
        
        for range_name, (low_freq, high_freq) in self.pitch_ranges.items():
            y = start_y + y_offset
            
            # 频率范围标签
            range_label = f"{range_name} ({low_freq}-{high_freq}Hz)"
            label_surface = self.font_small.render(range_label, True, self.colors["text"])
            self.screen.blit(label_surface, (start_x + 15, y))
            
            # 绘制三个乐器的强度条
            for idx, instrument in enumerate(["violin", "lute", "organ"]):
                intensity = self.pitch_levels.get(instrument, {}).get(range_name, 0)
                
                bar_x = start_x + 20 + idx * 130
                bar_y = y + 20
                bar_width = 120
                bar_height = 20
                
                # 背景条
                pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
                
                # 强度条
                if intensity > 0.01:
                    fill_width = int(intensity * bar_width)
                    color = self.colors[instrument]
                    pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
                
                # 数值标签
                value_text = f"{intensity:.2f}"
                value_surface = self.font_small.render(value_text, True, self.colors["text"])
                self.screen.blit(value_surface, (bar_x, bar_y + bar_height + 2))
                
                # 乐器标签
                if y_offset == 90:  # 只在第一行显示
                    instr_surface = self.font_small.render(instrument[:4], True, self.colors[instrument])
                    self.screen.blit(instr_surface, (bar_x + 40, start_y + 75))
            
            y_offset += range_height
    
    def draw_controls_panel(self):
        """绘制控制面板"""
        panel_width = 350
        panel_height = 200
        start_x = 50
        start_y = 50
        
        # 绘制背景
        pygame.draw.rect(self.screen, self.colors["panel"], (start_x, start_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors["text"], (start_x, start_y, panel_width, panel_height), 2)
        
        # 标题
        title = self.font_large.render("音频控制", True, self.colors["text"])
        self.screen.blit(title, (start_x + 15, start_y + 15))
        
        # 控制说明
        controls = [
            "1 - 切换小提琴 (红色)",
            "2 - 切换鲁特琴 (绿色)",
            "3 - 切换管风琴 (蓝色)",
            "A - 播放全部",
            "S - 停止全部",
            "SPACE - 分析开关",
            "ESC - 退出"
        ]
        
        for i, control in enumerate(controls):
            color = self.colors["text"]
            if "小提琴" in control:
                color = self.colors["violin"]
            elif "鲁特琴" in control:
                color = self.colors["lute"]
            elif "管风琴" in control:
                color = self.colors["organ"]
            
            text = self.font_small.render(control, True, color)
            self.screen.blit(text, (start_x + 15, start_y + 55 + i * 18))
        
        # 播放状态
        status_text = "播放状态:"
        status_surface = self.font_small.render(status_text, True, self.colors["accent"])
        self.screen.blit(status_surface, (start_x + 200, start_y + 55))
        
        for idx, (instrument, playing) in enumerate(self.playing.items()):
            status = "●" if playing else "○"
            color = self.colors[instrument] if playing else (100, 100, 100)
            status_surface = self.font_medium.render(f"{status} {instrument}", True, color)
            self.screen.blit(status_surface, (start_x + 200, start_y + 80 + idx * 25))
    
    def toggle_instrument(self, instrument: str):
        """切换乐器播放"""
        if instrument in self.audio_sounds:
            channel = self.audio_channels[instrument]
            
            if self.playing[instrument]:
                channel.stop()
                self.playing[instrument] = False
                print(f"停止播放 {instrument}")
            else:
                sound = self.audio_sounds[instrument]
                channel.play(sound, loops=-1)
                self.playing[instrument] = True
                print(f"开始播放 {instrument}")
                
                # 重置时间
                if self.start_time is None:
                    self.start_time = time.time()
    
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
                    for instrument in ["violin", "lute", "organ"]:
                        if not self.playing[instrument]:
                            self.toggle_instrument(instrument)
                elif event.key == pygame.K_s:
                    for instrument in ["violin", "lute", "organ"]:
                        if self.playing[instrument]:
                            self.toggle_instrument(instrument)
                elif event.key == pygame.K_SPACE:
                    self.analysis_enabled = not self.analysis_enabled
                    print(f"分析 {'开启' if self.analysis_enabled else '关闭'}")
    
    def run(self):
        """主运行循环"""
        print("\n🎵 真实音频分析器")
        print("="*60)
        print("此工具可以实时分析MP3文件的音调高低")
        print("显示每个乐器在不同频率范围的强度")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("控制说明:")
        print("1/2/3 - 切换乐器播放")
        print("A - 播放全部")
        print("S - 停止全部")
        print("SPACE - 切换分析开关")
        print("ESC - 退出")
        print()
        
        try:
            while self.running:
                # 处理事件
                self.handle_events()
                
                # 更新分析
                self.update_analysis()
                
                # 绘制界面
                self.screen.fill(self.colors["background"])
                
                self.draw_controls_panel()
                self.draw_spectrum_display()
                self.draw_pitch_analysis_panel()
                
                # 更新显示
                pygame.display.flip()
                self.clock.tick(30)
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
            pygame.quit()
        except:
            pass
        
        print("清理完成")


def main():
    try:
        analyzer = RealAudioAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()