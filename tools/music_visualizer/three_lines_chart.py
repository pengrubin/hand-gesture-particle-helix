#!/usr/bin/env python3
"""
三线音高强度滑动窗口图
分别显示三个MP3文件的音高强度，像汽车窗外风景一样流动
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

class ThreeLinesChart:
    def __init__(self):
        print("初始化三线音高强度滑动窗口图...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件定义
        self.instruments = {
            "violin": {
                "file": "../Fugue in G Trio violin-Violin.mp3",
                "name": "Violin",
                "color": "#ff4444",  # 红色
                "channel": 0
            },
            "lute": {
                "file": "../Fugue in G Trio-Tenor_Lute.mp3", 
                "name": "Lute",
                "color": "#44ff44",  # 绿色
                "channel": 1
            },
            "organ": {
                "file": "../Fugue in G Trio Organ-Organ.mp3",
                "name": "Organ", 
                "color": "#4444ff",  # 蓝色
                "channel": 2
            }
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
        self.update_interval = 0.05  # 50ms更新一次
        self.time_window = 8.0  # 8秒滑动窗口
        
        # 频率范围
        self.freq_range = (80, 2000)  # 主要音乐频率范围
        
        # 历史数据存储 - 为每个乐器分别存储
        self.max_points = int(self.time_window / self.update_interval)
        self.time_history = deque(maxlen=self.max_points)
        
        # 每个乐器的音高强度历史
        self.intensity_history = {}
        for instrument in self.instruments.keys():
            self.intensity_history[instrument] = deque(maxlen=self.max_points)
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"三线音高强度滑动窗口图初始化完成 (窗口: {self.time_window}秒)")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建单个大图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 10))
        self.fig.suptitle('Three-Track Pitch Intensity - Sliding Window View', 
                         fontsize=20, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title(f'Individual Instrument Pitch Intensity ({self.time_window}s window)', 
                         color='cyan', fontsize=16, pad=20)
        self.ax.set_xlabel('Time (seconds)', color='white', fontsize=14)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=14)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 为每个乐器创建线条
        self.lines = {}
        self.fills = {}
        
        for instrument_id, info in self.instruments.items():
            # 创建主线条
            line = self.ax.plot([], [], color=info["color"], linewidth=3, 
                              label=info["name"], alpha=0.9)[0]
            self.lines[instrument_id] = line
            
            # 初始化填充区域
            self.fills[instrument_id] = None
        
        # 设置图例
        self.ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
                      fontsize=14, framealpha=0.8)
        
        # 美化图表
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
    
    def load_audio_files(self):
        """加载音频文件"""
        print("加载音频文件...")
        
        success_count = 0
        for instrument_id, info in self.instruments.items():
            file_path = info["file"]
            
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            try:
                # 加载pygame声音
                sound = pygame.mixer.Sound(file_path)
                self.audio_sounds[instrument_id] = sound
                self.audio_channels[instrument_id] = pygame.mixer.Channel(info["channel"])
                
                if LIBROSA_AVAILABLE:
                    # 使用librosa加载音频数据用于分析
                    y, sr = librosa.load(file_path, sr=22050)
                    self.audio_data[instrument_id] = y
                    self.sample_rates[instrument_id] = sr
                    print(f"✅ {info['name']}: {len(y)/sr:.1f}秒, {sr}Hz")
                else:
                    print(f"✅ {info['name']}: 已加载 (模拟分析)")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 加载失败 {info['name']}: {e}")
        
        return success_count > 0
    
    def calculate_pitch_intensity(self, instrument_id: str, time_pos: float):
        """计算指定乐器在指定时间点的音高强度"""
        if not LIBROSA_AVAILABLE or instrument_id not in self.audio_data:
            return self.generate_mock_intensity(instrument_id, time_pos)
        
        try:
            y = self.audio_data[instrument_id]
            sr = self.sample_rates[instrument_id]
            
            # 计算样本位置
            sample_pos = int(time_pos * sr)
            if sample_pos >= len(y):
                sample_pos = sample_pos % len(y)  # 循环播放
            
            # 提取音频段
            start = max(0, sample_pos - self.fft_size // 2)
            end = min(len(y), start + self.fft_size)
            
            if end - start < self.fft_size // 2:
                return 0.0
            
            # 准备音频段
            audio_segment = np.zeros(self.fft_size)
            actual_length = min(self.fft_size, end - start)
            audio_segment[:actual_length] = y[start:start + actual_length]
            
            # 执行FFT分析
            fft = np.fft.fft(audio_segment)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # 提取指定频率范围的能量
            freq_mask = (positive_freqs >= self.freq_range[0]) & (positive_freqs <= self.freq_range[1])
            
            if np.any(freq_mask):
                # 计算频率范围内的加权能量
                target_magnitudes = positive_magnitude[freq_mask]
                target_freqs = positive_freqs[freq_mask]
                
                # 不同乐器的频率权重
                if instrument_id == "violin":
                    # 小提琴：偏重高频
                    weights = np.power(target_freqs / self.freq_range[1], 0.8)
                elif instrument_id == "lute":
                    # 鲁特琴：中频均衡
                    weights = np.ones(len(target_freqs))
                else:  # organ
                    # 管风琴：偏重低频
                    weights = np.power(self.freq_range[1] / target_freqs, 0.5)
                
                # 加权平均
                if np.sum(weights) > 0:
                    weighted_energy = np.average(target_magnitudes, weights=weights)
                else:
                    weighted_energy = np.mean(target_magnitudes)
                
                # 归一化到0-1范围
                max_possible = np.max(positive_magnitude) + 1e-10
                intensity = min(1.0, weighted_energy / max_possible)
                
                # 应用乐器特定的增强
                intensity = self.enhance_instrument_characteristics(instrument_id, intensity)
                
                return intensity
            else:
                return 0.0
            
        except Exception as e:
            print(f"{instrument_id} 音频分析错误: {e}")
            return self.generate_mock_intensity(instrument_id, time_pos)
    
    def enhance_instrument_characteristics(self, instrument_id: str, raw_intensity):
        """根据乐器特性增强音高强度"""
        if instrument_id == "violin":
            # 小提琴：强调高音变化，更敏感
            enhanced = math.pow(raw_intensity, 0.7)
        elif instrument_id == "lute":
            # 鲁特琴：保持自然变化
            enhanced = math.sqrt(raw_intensity)
        else:  # organ
            # 管风琴：平滑低音，减少突变
            enhanced = math.pow(raw_intensity, 1.2)
        
        return min(1.0, enhanced)
    
    def generate_mock_intensity(self, instrument_id: str, time_pos: float):
        """生成模拟的音高强度数据"""
        # 为不同乐器创建不同的特征模式
        
        if instrument_id == "violin":
            # 小提琴：活跃的高音旋律
            main = 0.5 + 0.3 * math.sin(time_pos * 0.8)
            ornaments = 0.2 * math.sin(time_pos * 6.0) * math.cos(time_pos * 0.4)
            rhythm = 0.15 * math.sin(time_pos * 3.2)
            
        elif instrument_id == "lute":
            # 鲁特琴：中音和弦与拨弦
            main = 0.4 + 0.25 * math.sin(time_pos * 0.6 + 1)  # 相位偏移
            ornaments = 0.15 * math.sin(time_pos * 4.0) * math.sin(time_pos * 0.3)
            rhythm = 0.2 * math.sin(time_pos * 2.8 + 2)
            
        else:  # organ
            # 管风琴：稳定的低音基础
            main = 0.6 + 0.2 * math.sin(time_pos * 0.4 + 2)  # 更慢的变化
            ornaments = 0.1 * math.sin(time_pos * 2.0) * math.cos(time_pos * 0.2)
            rhythm = 0.1 * math.sin(time_pos * 1.5 + 3)
        
        # 添加随机性但保持乐器特征
        noise = 0.03 * (np.random.random() - 0.5)
        
        # 组合所有成分
        intensity = main + ornaments + rhythm + noise
        
        # 添加乐器特定的峰值模式
        peak_timing = {
            "violin": 0.25,   # 4秒一个峰值
            "lute": 0.2,      # 5秒一个峰值  
            "organ": 0.125    # 8秒一个峰值
        }
        
        if int(time_pos / (1/peak_timing[instrument_id])) % 4 == 0:
            peak_phase = (time_pos / (1/peak_timing[instrument_id])) % 1
            peak_factor = 1.0 + 0.3 * math.exp(-peak_phase * 6)
            intensity *= peak_factor
        
        return max(0.0, min(1.0, intensity))
    
    def update_data(self, frame):
        """更新数据的回调函数"""
        # 更新时间
        if self.start_time:
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += self.update_interval
        
        # 计算每个乐器的音高强度
        current_intensities = {}
        for instrument_id in self.instruments.keys():
            intensity = self.calculate_pitch_intensity(instrument_id, self.current_time)
            current_intensities[instrument_id] = intensity
            self.intensity_history[instrument_id].append(intensity)
        
        # 存储时间
        self.time_history.append(self.current_time)
        
        # 更新图表
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # 更新每条线
            for instrument_id, info in self.instruments.items():
                intensities = list(self.intensity_history[instrument_id])
                
                if len(intensities) == len(times):
                    # 更新主线条
                    line = self.lines[instrument_id]
                    line.set_data(times, intensities)
                    
                    # 更新填充区域（透明度较低避免重叠）
                    if self.fills[instrument_id]:
                        self.fills[instrument_id].remove()
                    
                    self.fills[instrument_id] = self.ax.fill_between(
                        times, 0, intensities, 
                        color=info["color"], alpha=0.15
                    )
            
            # 设置固定的x轴范围（滑动窗口效果）
            current_end = times[-1]
            window_start = current_end - self.time_window
            self.ax.set_xlim(window_start, current_end)
            
            # 动态更新标题显示当前强度
            intensity_info = " | ".join([
                f"{info['name']}: {current_intensities[inst_id]*100:.1f}%" 
                for inst_id, info in self.instruments.items()
            ])
            
            self.ax.set_title(
                f'Individual Instrument Pitch Intensity ({self.time_window}s window)\n{intensity_info}', 
                color='cyan', fontsize=14, pad=20
            )
        
        return list(self.lines.values())
    
    def play_all(self):
        """播放所有音频文件"""
        print("🎵 开始播放所有音频文件...")
        self.start_time = time.time()
        
        for instrument_id, info in self.instruments.items():
            try:
                sound = self.audio_sounds[instrument_id]
                channel = self.audio_channels[instrument_id]
                channel.play(sound, loops=-1)  # 循环播放
                print(f"✅ 播放 {info['name']}")
            except Exception as e:
                print(f"❌ 播放失败 {info['name']}: {e}")
    
    def stop_all(self):
        """停止所有音频"""
        print("停止所有音频...")
        pygame.mixer.stop()
        self.start_time = None
    
    def run(self):
        """运行主程序"""
        print("\n🎼 三线音高强度滑动窗口 - 分别显示三个乐器")
        print("=" * 70)
        print("功能：分别显示每个乐器的音高强度，像汽车窗外的风景一样流动")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示说明:")
        for instrument_id, info in self.instruments.items():
            print(f"- {info['color']} {info['name']} 线：{info['file'].split('/')[-1]}")
        print(f"- 时间窗口：{self.time_window}秒 (固定)")
        print(f"- 更新频率：{1000*self.update_interval:.0f}ms")
        print(f"- 分析频率范围：{self.freq_range[0]}-{self.freq_range[1]}Hz")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎶 开始三线音高强度流动显示...")
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
        analyzer = ThreeLinesChart()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()