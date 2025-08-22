#!/usr/bin/env python3
"""
实时音频分析折线图
同时播放三个MP3文件并显示音频高低的实时折线图
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import os
import threading
from collections import deque
from datetime import datetime

# 尝试导入音频分析库
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ librosa 可用 - 将进行真实音频分析")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa 不可用 - 将使用模拟分析")

class RealtimeAudioChart:
    def __init__(self):
        print("初始化实时音频分析折线图...")
        
        # 初始化pygame
        pygame.init()
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 显示设置
        self.width = 1600
        self.height = 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("实时音频分析折线图 - 三个MP3同时播放")
        
        # 字体
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # 音频文件
        self.audio_files = {
            "violin": "../Fugue in G Trio violin-Violin.mp3",
            "lute": "../Fugue in G Trio-Tenor_Lute.mp3", 
            "organ": "../Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 颜色定义
        self.colors = {
            "violin": (255, 80, 80),      # 红色
            "lute": (80, 255, 80),        # 绿色
            "organ": (80, 80, 255),       # 蓝色
            "background": (25, 25, 35),
            "panel": (40, 40, 50),
            "text": (255, 255, 255),
            "accent": (255, 255, 100)
        }
        
        # 音频数据
        self.audio_data = {}
        self.sample_rates = {}
        self.durations = {}
        
        # pygame音频对象
        self.audio_sounds = {}
        self.audio_channels = {}
        self.playing = {"violin": False, "lute": False, "organ": False}
        
        # 时间控制
        self.current_time = 0.0
        self.start_time = None
        self.running = True
        self.clock = pygame.time.Clock()
        
        # 分析参数
        self.fft_size = 2048
        self.update_interval = 0.1  # 100ms更新一次
        
        # 音调分析数据存储 (用于折线图)
        self.max_history_length = 300  # 保存30秒的数据 (300 * 0.1s)
        self.time_history = deque(maxlen=self.max_history_length)
        self.pitch_history = {
            "violin": {"high": deque(maxlen=self.max_history_length),
                      "mid": deque(maxlen=self.max_history_length),
                      "low": deque(maxlen=self.max_history_length)},
            "lute": {"high": deque(maxlen=self.max_history_length),
                    "mid": deque(maxlen=self.max_history_length),
                    "low": deque(maxlen=self.max_history_length)},
            "organ": {"high": deque(maxlen=self.max_history_length),
                     "mid": deque(maxlen=self.max_history_length),
                     "low": deque(maxlen=self.max_history_length)}
        }
        
        # 频率范围定义
        self.pitch_ranges = {
            "low": (20, 500),      # 低音
            "mid": (500, 2000),    # 中音
            "high": (2000, 8000)   # 高音
        }
        
        # matplotlib图表设置
        self.setup_matplotlib()
        
        print("实时音频分析折线图初始化完成")
    
    def setup_matplotlib(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8), facecolor='black')
        self.fig.suptitle('Real-time Audio Pitch Analysis', color='white', fontsize=16)
        
        # 为每个乐器设置子图
        instrument_names = ['Violin', 'Lute', 'Organ']
        for i, (ax, name) in enumerate(zip(self.axes, instrument_names)):
            ax.set_title(name, color='white', fontsize=12)
            ax.set_xlabel('Time (seconds)', color='white')
            ax.set_ylabel('Intensity', color='white')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
        
        # 创建空的线条对象
        self.lines = {}
        instruments = ['violin', 'lute', 'organ']
        line_colors = ['red', 'green', 'blue']
        
        for i, instrument in enumerate(instruments):
            ax = self.axes[i]
            self.lines[instrument] = {
                'high': ax.plot([], [], color='orange', label='High', linewidth=2)[0],
                'mid': ax.plot([], [], color='yellow', label='Mid', linewidth=2)[0],
                'low': ax.plot([], [], color='cyan', label='Low', linewidth=2)[0]
            }
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 将matplotlib图表转换为pygame表面
        self.canvas = FigureCanvasAgg(self.fig)
    
    def load_audio_files(self):
        """加载音频文件"""
        print("加载音频文件...")
        
        for name, file_path in self.audio_files.items():
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            try:
                # 加载pygame声音
                sound = pygame.mixer.Sound(file_path)
                self.audio_sounds[name] = sound
                channel_id = ["violin", "lute", "organ"].index(name)
                self.audio_channels[name] = pygame.mixer.Channel(channel_id)
                
                if LIBROSA_AVAILABLE:
                    # 使用librosa加载音频数据用于分析
                    y, sr = librosa.load(file_path, sr=22050)
                    self.audio_data[name] = y
                    self.sample_rates[name] = sr
                    self.durations[name] = len(y) / sr
                    print(f"✅ {name}: {self.durations[name]:.1f}秒, {sr}Hz")
                else:
                    # 估算音频长度
                    self.durations[name] = 180.0
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
            if sample_pos >= len(y):
                sample_pos = sample_pos % len(y)  # 循环播放
            
            # 提取音频段进行分析
            window_size = self.fft_size
            start = max(0, sample_pos - window_size // 2)
            end = min(len(y), start + window_size)
            
            if end - start < window_size:
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
                freq_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
                if np.any(freq_mask):
                    energy = np.mean(positive_magnitude[freq_mask])
                    # 归一化到0-1范围
                    max_energy = np.max(positive_magnitude) + 1e-10
                    pitch_analysis[range_name] = min(1.0, energy / max_energy)
                else:
                    pitch_analysis[range_name] = 0.0
            
            return pitch_analysis
            
        except Exception as e:
            print(f"音频分析错误 {instrument}: {e}")
            return self.generate_mock_analysis(instrument, time_pos)
    
    def generate_mock_analysis(self, instrument: str, time_pos: float):
        """生成模拟的音频分析数据"""
        pitch_analysis = {}
        
        # 不同乐器的特征频率模拟
        if instrument == "violin":
            # 小提琴：高音较强
            base_strengths = {"low": 0.3, "mid": 0.7, "high": 0.9}
        elif instrument == "lute":
            # 鲁特琴：中音为主
            base_strengths = {"low": 0.4, "mid": 0.8, "high": 0.5}
        else:  # organ
            # 管风琴：低音强
            base_strengths = {"low": 0.9, "mid": 0.6, "high": 0.3}
        
        # 添加时间变化和音乐性
        for range_name, base_strength in base_strengths.items():
            # 主旋律变化
            melody_var = np.sin(time_pos * 0.8 + hash(instrument + range_name) % 10) * 0.2
            # 节拍变化
            rhythm_var = np.sin(time_pos * 2.5) * 0.15
            # 随机波动
            random_var = (np.random.random() - 0.5) * 0.1
            
            final_strength = base_strength + melody_var + rhythm_var + random_var
            pitch_analysis[range_name] = max(0.0, min(1.0, final_strength))
        
        return pitch_analysis
    
    def update_analysis(self):
        """更新音频分析数据"""
        # 更新当前时间
        if any(self.playing.values()):
            if self.start_time is None:
                self.start_time = time.time()
            self.current_time = time.time() - self.start_time
        else:
            # 没有播放时也继续分析（演示用）
            self.current_time += self.update_interval
        
        # 分析每个乐器的音频
        current_data = {}
        for instrument in ["violin", "lute", "organ"]:
            pitch_data = self.analyze_audio_at_time(instrument, self.current_time)
            current_data[instrument] = pitch_data
            
            # 存储到历史数据
            for freq_type in ["high", "mid", "low"]:
                self.pitch_history[instrument][freq_type].append(pitch_data[freq_type])
        
        # 添加时间点
        self.time_history.append(self.current_time)
    
    def update_matplotlib_chart(self):
        """更新matplotlib图表"""
        # 如果没有足够的数据，不更新
        if len(self.time_history) < 2:
            return
        
        # 获取时间轴数据
        times = list(self.time_history)
        
        # 更新每个乐器的线条
        instruments = ["violin", "lute", "organ"]
        for i, instrument in enumerate(instruments):
            ax = self.axes[i]
            
            for freq_type in ["high", "mid", "low"]:
                values = list(self.pitch_history[instrument][freq_type])
                if len(values) == len(times):
                    line = self.lines[instrument][freq_type]
                    line.set_data(times, values)
            
            # 更新x轴范围
            if len(times) > 0:
                ax.set_xlim(max(0, times[-1] - 30), times[-1] + 1)  # 显示最近30秒
        
        # 重新绘制
        self.canvas.draw()
    
    def pygame_draw_matplotlib(self):
        """将matplotlib图表绘制到pygame表面"""
        # 获取matplotlib渲染的原始数据
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        
        # 获取图表尺寸
        size = self.canvas.get_width_height()
        
        # 创建pygame表面
        chart_surface = pygame.image.frombuffer(raw_data, size, 'RGBA')
        
        # 缩放到合适的尺寸
        chart_width = self.width - 100
        chart_height = self.height - 200
        chart_surface = pygame.transform.scale(chart_surface, (chart_width, chart_height))
        
        return chart_surface
    
    def toggle_instrument(self, instrument: str):
        """切换乐器播放状态"""
        if instrument in self.audio_sounds:
            channel = self.audio_channels[instrument]
            
            if self.playing[instrument]:
                channel.stop()
                self.playing[instrument] = False
                print(f"停止播放 {instrument}")
            else:
                sound = self.audio_sounds[instrument]
                channel.play(sound, loops=-1)  # 循环播放
                self.playing[instrument] = True
                print(f"开始播放 {instrument}")
                
                # 如果是第一次播放，重置时间
                if self.start_time is None:
                    self.start_time = time.time()
                    self.current_time = 0.0
    
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
    
    def draw_controls(self):
        """绘制控制面板"""
        panel_width = 350
        panel_height = 150
        panel_x = 20
        panel_y = 20
        
        # 绘制控制面板背景
        pygame.draw.rect(self.screen, self.colors["panel"], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors["text"], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # 标题
        title = self.font_large.render("音频控制", True, self.colors["text"])
        self.screen.blit(title, (panel_x + 15, panel_y + 15))
        
        # 控制说明
        controls = [
            "1/2/3 - 切换单个乐器",
            "A - 播放全部  S - 停止全部",
            "SPACE - 重置时间  ESC - 退出"
        ]
        
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, self.colors["text"])
            self.screen.blit(text, (panel_x + 15, panel_y + 50 + i * 20))
        
        # 播放状态
        status_y = panel_y + 120
        for i, (instrument, playing) in enumerate(self.playing.items()):
            status = "●" if playing else "○"
            color = self.colors[instrument] if playing else (100, 100, 100)
            status_text = f"{status} {instrument}"
            text = self.font_small.render(status_text, True, color)
            self.screen.blit(text, (panel_x + 15 + i * 100, status_y))
    
    def draw_time_info(self):
        """绘制时间信息"""
        time_text = f"播放时间: {self.current_time:.1f}秒"
        text = self.font_medium.render(time_text, True, self.colors["accent"])
        self.screen.blit(text, (400, 30))
        
        # 数据点数量
        data_count = len(self.time_history)
        data_text = f"数据点: {data_count}/{self.max_history_length}"
        text = self.font_medium.render(data_text, True, self.colors["accent"])
        self.screen.blit(text, (400, 60))
        
        # 当前音频分析值
        y_offset = 90
        for instrument in ["violin", "lute", "organ"]:
            if len(self.pitch_history[instrument]["high"]) > 0:
                high = self.pitch_history[instrument]["high"][-1]
                mid = self.pitch_history[instrument]["mid"][-1] 
                low = self.pitch_history[instrument]["low"][-1]
                
                analysis_text = f"{instrument}: H:{high:.2f} M:{mid:.2f} L:{low:.2f}"
                color = self.colors[instrument]
                text = self.font_small.render(analysis_text, True, color)
                self.screen.blit(text, (400, y_offset))
                y_offset += 20
    
    def handle_events(self):
        """处理用户输入事件"""
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
                elif event.key == pygame.K_SPACE:
                    # 重置时间
                    self.start_time = time.time()
                    self.current_time = 0.0
                    # 清空历史数据
                    self.time_history.clear()
                    for instrument in ["violin", "lute", "organ"]:
                        for freq_type in ["high", "mid", "low"]:
                            self.pitch_history[instrument][freq_type].clear()
                    print("时间和数据已重置")
    
    def run(self):
        """主运行循环"""
        print("\n🎵 实时音频分析折线图")
        print("=" * 60)
        print("功能：同时播放三个MP3文件并显示音频高低的实时折线图")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("控制说明:")
        print("1/2/3 - 切换单个乐器播放")
        print("A - 播放全部乐器")
        print("S - 停止全部乐器")
        print("SPACE - 重置时间和数据")
        print("ESC - 退出程序")
        print()
        
        last_update = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # 处理事件
                self.handle_events()
                
                # 定期更新分析数据
                if current_time - last_update >= self.update_interval:
                    self.update_analysis()
                    self.update_matplotlib_chart()
                    last_update = current_time
                
                # 绘制界面
                self.screen.fill(self.colors["background"])
                
                # 绘制控制面板和信息
                self.draw_controls()
                self.draw_time_info()
                
                # 绘制matplotlib图表
                chart_surface = self.pygame_draw_matplotlib()
                chart_x = 50
                chart_y = 180
                self.screen.blit(chart_surface, (chart_x, chart_y))
                
                # 更新显示
                pygame.display.flip()
                self.clock.tick(60)  # 60 FPS
        
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
            plt.close('all')
        except:
            pass
        
        print("清理完成")


def main():
    """主函数"""
    try:
        analyzer = RealtimeAudioChart()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()