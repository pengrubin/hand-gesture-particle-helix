#!/usr/bin/env python3
"""
音高强度滑动窗口图
像汽车窗外风景一样显示音高强度的实时变化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import time
import os
import math
from collections import deque

# 尝试导入音频分析库
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ librosa 可用 - 将进行真实音频分析")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa 不可用 - 将使用模拟分析")

class PitchWindowChart:
    def __init__(self):
        print("初始化音高强度滑动窗口图...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件
        self.audio_files = {
            "violin": "../Fugue in G Trio violin-Violin.mp3",
            "lute": "../Fugue in G Trio-Tenor_Lute.mp3", 
            "organ": "../Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 音频数据
        self.audio_data = {}
        self.sample_rates = {}
        
        # pygame音频对象
        self.audio_sounds = {}
        self.audio_channels = {}
        
        # 时间控制
        self.current_time = 0.0
        self.start_time = None
        self.running = True
        
        # 分析参数
        self.fft_size = 2048
        self.update_interval = 0.05  # 50ms更新一次，更平滑
        self.time_window = 8.0  # 8秒滑动窗口
        
        # 历史数据存储 (固定时间窗口)
        self.max_points = int(self.time_window / self.update_interval)  # 160个数据点
        self.time_history = deque(maxlen=self.max_points)
        self.pitch_intensity_history = deque(maxlen=self.max_points)
        
        # 频率范围 (重点关注音乐频率)
        self.freq_range = (80, 2000)  # 80Hz - 2KHz，主要音乐频率范围
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"音高强度滑动窗口图初始化完成 (窗口: {self.time_window}秒)")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建单个大图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8))
        self.fig.suptitle('Real-time Pitch Intensity - Sliding Window View', 
                         fontsize=18, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title(f'Pitch Intensity Flow ({self.time_window}s window)', 
                         color='cyan', fontsize=16, pad=20)
        self.ax.set_xlabel('Time (seconds)', color='white', fontsize=14)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=14)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 创建渐变填充效果的线条
        self.pitch_line = self.ax.plot([], [], color='#00ff88', linewidth=3, 
                                      label='Pitch Intensity', alpha=0.9)[0]
        
        # 添加填充区域
        self.pitch_fill = None
        
        # 设置图例
        self.ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
                      fontsize=12, framealpha=0.8)
        
        # 美化图表
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white') 
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        plt.tight_layout()
    
    def load_audio_files(self):
        """加载音频文件"""
        print("加载音频文件...")
        
        success_count = 0
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
                    print(f"✅ {name}: {len(y)/sr:.1f}秒, {sr}Hz")
                else:
                    print(f"✅ {name}: 已加载 (模拟分析)")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 加载失败 {name}: {e}")
        
        return success_count > 0
    
    def calculate_pitch_intensity(self, time_pos: float):
        """计算指定时间点的音高强度"""
        if not LIBROSA_AVAILABLE:
            return self.generate_mock_intensity(time_pos)
        
        # 合并所有乐器的音频数据
        combined_audio = np.zeros(self.fft_size)
        valid_instruments = 0
        
        for instrument in ["violin", "lute", "organ"]:
            if instrument not in self.audio_data:
                continue
            
            y = self.audio_data[instrument]
            sr = self.sample_rates[instrument]
            
            # 计算样本位置
            sample_pos = int(time_pos * sr)
            if sample_pos >= len(y):
                sample_pos = sample_pos % len(y)  # 循环播放
            
            # 提取音频段
            start = max(0, sample_pos - self.fft_size // 2)
            end = min(len(y), start + self.fft_size)
            
            if end - start > self.fft_size // 2:
                audio_segment = np.zeros(self.fft_size)
                actual_length = min(self.fft_size, end - start)
                audio_segment[:actual_length] = y[start:start + actual_length]
                combined_audio += audio_segment
                valid_instruments += 1
        
        if valid_instruments == 0:
            return self.generate_mock_intensity(time_pos)
        
        # 归一化合并的音频
        combined_audio /= valid_instruments
        
        try:
            # 执行FFT分析
            fft = np.fft.fft(combined_audio)
            freqs = np.fft.fftfreq(len(fft), 1/22050)
            magnitude = np.abs(fft)
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # 提取指定频率范围的能量
            freq_mask = (positive_freqs >= self.freq_range[0]) & (positive_freqs <= self.freq_range[1])
            
            if np.any(freq_mask):
                # 计算频率范围内的平均能量
                target_magnitudes = positive_magnitude[freq_mask]
                target_freqs = positive_freqs[freq_mask]
                
                # 加权计算：高频率的音符给予更高权重（更突出旋律）
                weights = np.sqrt(target_freqs / self.freq_range[1])
                weighted_energy = np.average(target_magnitudes, weights=weights)
                
                # 归一化到0-1范围
                max_possible = np.max(positive_magnitude) + 1e-10
                intensity = min(1.0, weighted_energy / max_possible)
                
                # 应用音乐性曲线增强
                intensity = self.enhance_musicality(intensity)
                
                return intensity
            else:
                return 0.0
            
        except Exception as e:
            print(f"音频分析错误: {e}")
            return self.generate_mock_intensity(time_pos)
    
    def enhance_musicality(self, raw_intensity):
        """增强音乐性的强度曲线"""
        # 使用平方根函数让小的变化更明显
        enhanced = math.sqrt(raw_intensity)
        
        # 添加动态范围压缩，让变化更平滑
        if enhanced > 0.8:
            enhanced = 0.8 + (enhanced - 0.8) * 0.5
        
        return min(1.0, enhanced)
    
    def generate_mock_intensity(self, time_pos: float):
        """生成模拟的音高强度数据"""
        # 创建音乐性的强度变化
        
        # 主旋律线 (慢变化)
        main_melody = 0.4 + 0.3 * math.sin(time_pos * 0.6)
        
        # 节奏变化 (中等变化)
        rhythm = 0.2 * math.sin(time_pos * 2.5)
        
        # 细节装饰 (快变化)
        ornaments = 0.1 * math.sin(time_pos * 8.0) * math.sin(time_pos * 0.3)
        
        # 随机波动
        noise = 0.05 * (np.random.random() - 0.5)
        
        # 组合所有成分
        intensity = main_melody + rhythm + ornaments + noise
        
        # 添加音乐性的突发峰值
        if int(time_pos * 4) % 16 == 0:  # 每4秒一个峰值
            peak_factor = 1.0 + 0.4 * math.exp(-((time_pos * 4) % 1) * 5)
            intensity *= peak_factor
        
        return max(0.0, min(1.0, intensity))
    
    def update_data(self, frame):
        """更新数据的回调函数"""
        # 更新时间
        if self.start_time:
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += self.update_interval
        
        # 计算当前时间点的音高强度
        intensity = self.calculate_pitch_intensity(self.current_time)
        
        # 存储历史数据
        self.time_history.append(self.current_time)
        self.pitch_intensity_history.append(intensity)
        
        # 更新图表
        if len(self.time_history) > 1:
            times = list(self.time_history)
            intensities = list(self.pitch_intensity_history)
            
            # 更新主线条
            self.pitch_line.set_data(times, intensities)
            
            # 更新填充区域
            if self.pitch_fill:
                self.pitch_fill.remove()
            
            # 创建渐变填充效果
            self.pitch_fill = self.ax.fill_between(times, 0, intensities, 
                                                  color='#00ff88', alpha=0.3)
            
            # 设置固定的x轴范围（滑动窗口效果）
            current_end = times[-1]
            window_start = current_end - self.time_window
            
            self.ax.set_xlim(window_start, current_end)
            
            # 动态更新标题显示当前强度
            current_intensity_percent = intensity * 100
            self.ax.set_title(
                f'Pitch Intensity Flow ({self.time_window}s window) | '
                f'Current: {current_intensity_percent:.1f}%', 
                color='cyan', fontsize=16, pad=20
            )
        
        return [self.pitch_line]
    
    def play_all(self):
        """播放所有音频文件"""
        print("🎵 开始播放所有音频文件...")
        self.start_time = time.time()
        
        for name, sound in self.audio_sounds.items():
            try:
                channel = self.audio_channels[name]
                channel.play(sound, loops=-1)  # 循环播放
                print(f"✅ 播放 {name}")
            except Exception as e:
                print(f"❌ 播放失败 {name}: {e}")
    
    def stop_all(self):
        """停止所有音频"""
        print("停止所有音频...")
        pygame.mixer.stop()
        self.start_time = None
    
    def run(self):
        """运行主程序"""
        print("\n🚗 音高强度滑动窗口 - 汽车风景式显示")
        print("=" * 60)
        print("功能：显示音高强度的实时变化，像汽车窗外的风景一样流动")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示说明:")
        print(f"- 绿色线条：音高强度变化")
        print(f"- 时间窗口：{self.time_window}秒 (固定)")
        print(f"- 更新频率：{1000*self.update_interval:.0f}ms")
        print(f"- 分析频率范围：{self.freq_range[0]}-{self.freq_range[1]}Hz")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎼 开始音高强度流动显示...")
            interval_ms = int(self.update_interval * 1000)
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_data, 
                interval=interval_ms,
                blit=False,
                cache_frame_data=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
            plt.close('all')
        except:
            pass
        print("清理完成")


def main():
    """主函数"""
    try:
        analyzer = PitchWindowChart()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()