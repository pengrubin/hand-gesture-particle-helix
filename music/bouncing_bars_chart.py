#!/usr/bin/env python3
"""
跳动柱状图音高显示
针对MIDI生成的音频，用柱状图显示三个乐器的音高强度跳动
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

class BouncingBarsChart:
    def __init__(self):
        print("初始化跳动柱状图音高显示...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件定义
        self.instruments = {
            "violin": {
                "file": "../Fugue in G Trio violin-Violin.mp3",
                "name": "Violin",
                "color": "#ff3366",  # 鲜红色
                "position": 0,
                "freq_focus": (200, 2000)  # 小提琴频率范围
            },
            "lute": {
                "file": "../Fugue in G Trio-Tenor_Lute.mp3", 
                "name": "Lute",
                "color": "#33ff66",  # 鲜绿色
                "position": 1,
                "freq_focus": (100, 800)   # 鲁特琴频率范围
            },
            "organ": {
                "file": "../Fugue in G Trio Organ-Organ.mp3",
                "name": "Organ", 
                "color": "#3366ff",  # 鲜蓝色
                "position": 2,
                "freq_focus": (50, 500)    # 管风琴频率范围
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
        self.fft_size = 4096  # 更大的FFT for better frequency resolution
        self.update_interval = 0.1  # 100ms更新一次，适合柱状图
        self.time_window = 10.0  # 10秒滑动窗口
        self.bar_width = 0.08   # 柱状图宽度（时间轴上的宽度）
        
        # 历史数据存储
        self.max_points = int(self.time_window / self.update_interval)
        self.time_history = deque(maxlen=self.max_points)
        
        # 每个乐器的强度历史
        self.intensity_history = {}
        for instrument in self.instruments.keys():
            self.intensity_history[instrument] = deque(maxlen=self.max_points)
        
        # 动画效果参数
        self.bounce_factor = 1.2  # 跳动放大系数
        self.decay_speed = 0.85   # 下降速度
        self.current_heights = {inst: 0.0 for inst in self.instruments.keys()}
        self.target_heights = {inst: 0.0 for inst in self.instruments.keys()}
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"跳动柱状图音高显示初始化完成 (窗口: {self.time_window}秒)")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建主图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 10))
        self.fig.suptitle('Bouncing Bars - MIDI Audio Pitch Visualization', 
                         fontsize=20, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title(f'Musical Instrument Intensity Bars ({self.time_window}s window)', 
                         color='cyan', fontsize=16, pad=20)
        self.ax.set_xlabel('Time (seconds)', color='white', fontsize=14)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=14)
        self.ax.set_ylim(0, 1.2)  # 稍微高一点给跳动留空间
        
        # 设置背景网格
        self.ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 为每个乐器创建初始的空柱状图容器
        self.bar_containers = {}
        
        # 美化图表
        for spine in self.ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        
        # 添加乐器标签
        legend_elements = []
        for inst_id, info in self.instruments.items():
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=info['color'], 
                                               label=info['name'], alpha=0.8))
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      facecolor='black', edgecolor='white', 
                      fontsize=14, framealpha=0.9)
        
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
                self.audio_channels[instrument_id] = pygame.mixer.Channel(info["position"])
                
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
            
            # 应用窗函数
            window = np.hanning(len(audio_segment))
            audio_segment *= window
            
            # 执行FFT分析
            fft = np.fft.fft(audio_segment)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # 获取乐器特定的频率范围
            freq_range = self.instruments[instrument_id]["freq_focus"]
            freq_mask = (positive_freqs >= freq_range[0]) & (positive_freqs <= freq_range[1])
            
            if np.any(freq_mask):
                # 提取目标频率的能量
                target_magnitudes = positive_magnitude[freq_mask]
                target_freqs = positive_freqs[freq_mask]
                
                # 寻找峰值（MIDI特征：离散的音符）
                if len(target_magnitudes) > 10:
                    # 找到前几个峰值
                    peak_indices = np.argsort(target_magnitudes)[-5:]
                    peak_energies = target_magnitudes[peak_indices]
                    
                    # 计算峰值能量的总和
                    peak_energy = np.sum(peak_energies)
                else:
                    peak_energy = np.sum(target_magnitudes)
                
                # 归一化
                max_possible = np.max(positive_magnitude) + 1e-10
                intensity = min(1.0, peak_energy / max_possible)
                
                # MIDI特性增强：强调音符的跳跃性
                intensity = self.enhance_midi_characteristics(intensity)
                
                return intensity
            else:
                return 0.0
            
        except Exception as e:
            print(f"{instrument_id} 音频分析错误: {e}")
            return self.generate_mock_intensity(instrument_id, time_pos)
    
    def enhance_midi_characteristics(self, raw_intensity):
        """增强MIDI音频的特性：更明显的音符跳跃"""
        # 使用非线性变换强调峰值
        if raw_intensity > 0.1:
            # 对于有效的音符，放大变化
            enhanced = math.pow(raw_intensity, 0.6)  # 让变化更明显
            # 添加阈值效应
            if enhanced > 0.3:
                enhanced = 0.3 + (enhanced - 0.3) * 1.5
        else:
            # 对于微弱信号，进一步压制
            enhanced = raw_intensity * 0.5
        
        return min(1.0, enhanced)
    
    def generate_mock_intensity(self, instrument_id: str, time_pos: float):
        """生成模拟的MIDI风格音高强度数据"""
        # 模拟MIDI的离散音符特征
        
        if instrument_id == "violin":
            # 小提琴：快速的旋律音符
            note_timing = time_pos * 4  # 每0.25秒一个音符
            note_phase = note_timing % 1
            
            # 模拟音符的攻击和衰减
            if note_phase < 0.1:  # 攻击阶段
                base_intensity = note_phase * 10  # 快速上升
            elif note_phase < 0.7:  # 持续阶段
                base_intensity = 1.0 - (note_phase - 0.1) * 0.5  # 缓慢下降
            else:  # 衰减阶段
                base_intensity = 0.7 - (note_phase - 0.7) * 2  # 快速下降
            
            # 添加旋律变化
            melody_factor = 0.7 + 0.3 * math.sin(time_pos * 0.8)
            intensity = base_intensity * melody_factor
            
        elif instrument_id == "lute":
            # 鲁特琴：和弦拨弦
            chord_timing = time_pos * 2  # 每0.5秒一个和弦
            chord_phase = chord_timing % 1
            
            # 拨弦效果：快速攻击，中等衰减
            if chord_phase < 0.05:
                base_intensity = chord_phase * 20  # 非常快的攻击
            else:
                base_intensity = 1.0 * math.exp(-(chord_phase - 0.05) * 3)  # 指数衰减
            
            # 和弦变化
            harmony_factor = 0.6 + 0.4 * math.sin(time_pos * 0.5 + 1)
            intensity = base_intensity * harmony_factor
            
        else:  # organ
            # 管风琴：持续的低音
            bass_timing = time_pos * 1  # 每1秒变化
            bass_phase = bass_timing % 1
            
            # 管风琴的渐进变化
            base_intensity = 0.8 + 0.2 * math.sin(bass_phase * math.pi * 2)
            
            # 低音变化
            bass_factor = 0.7 + 0.3 * math.sin(time_pos * 0.3 + 2)
            intensity = base_intensity * bass_factor
        
        # 添加随机性但保持MIDI的离散特征
        noise = 0.02 * (np.random.random() - 0.5)
        intensity += noise
        
        return max(0.0, min(1.0, intensity))
    
    def update_bounce_animation(self):
        """更新跳动动画效果"""
        for instrument_id in self.instruments.keys():
            current = self.current_heights[instrument_id]
            target = self.target_heights[instrument_id]
            
            # 如果目标值比当前值高，快速跳上去（跳动效果）
            if target > current:
                self.current_heights[instrument_id] = min(target * self.bounce_factor, 1.2)
            else:
                # 否则缓慢下降（重力效果）
                self.current_heights[instrument_id] = current * self.decay_speed
                if self.current_heights[instrument_id] < 0.01:
                    self.current_heights[instrument_id] = 0.0
    
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
            
            # 更新目标高度
            self.target_heights[instrument_id] = intensity
        
        # 更新跳动动画
        self.update_bounce_animation()
        
        # 存储时间
        self.time_history.append(self.current_time)
        
        # 清除之前的柱状图
        self.ax.clear()
        
        # 重新设置图表属性
        self.ax.set_ylim(0, 1.2)
        self.ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 绘制柱状图
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # 绘制历史的小柱子
            for i, t in enumerate(times[:-10]):  # 不绘制最近的10个，避免重叠
                for j, (instrument_id, info) in enumerate(self.instruments.items()):
                    if i < len(self.intensity_history[instrument_id]):
                        intensity = list(self.intensity_history[instrument_id])[i]
                        
                        # 历史柱子：透明度递减
                        alpha = 0.3 * (i / len(times))
                        bar_pos = t + j * self.bar_width * 0.3  # 稍微错开位置
                        
                        self.ax.bar(bar_pos, intensity, width=self.bar_width * 0.25,
                                  color=info['color'], alpha=alpha, edgecolor='none')
            
            # 绘制当前的大柱子（跳动效果）
            current_time = times[-1]
            for j, (instrument_id, info) in enumerate(self.instruments.items()):
                # 当前柱子位置
                bar_pos = current_time + j * self.bar_width * 1.2
                current_height = self.current_heights[instrument_id]
                
                # 绘制主柱子
                bar = self.ax.bar(bar_pos, current_height, width=self.bar_width,
                                color=info['color'], alpha=0.9, edgecolor='white', linewidth=1)
                
                # 添加柱子顶部的亮点效果
                if current_height > 0.1:
                    self.ax.scatter(bar_pos, current_height + 0.05, 
                                  color=info['color'], s=50, alpha=0.8)
            
            # 设置固定的x轴范围（滑动窗口效果）
            current_end = times[-1]
            window_start = current_end - self.time_window
            self.ax.set_xlim(window_start, current_end + 0.5)
            
            # 动态更新标题
            intensity_info = " | ".join([
                f"{info['name']}: {current_intensities[inst_id]*100:.0f}%" 
                for inst_id, info in self.instruments.items()
            ])
            
            self.ax.set_title(
                f'Musical Bouncing Bars ({self.time_window}s window)\n{intensity_info}', 
                color='cyan', fontsize=14, pad=20
            )
            self.ax.set_xlabel('Time (seconds)', color='white', fontsize=14)
            self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=14)
        
        return []
    
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
        print("\n🎯 跳动柱状图音高显示 - MIDI风格可视化")
        print("=" * 70)
        print("功能：用跳动的柱状图显示MIDI生成音频的音符变化")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎼 显示说明:")
        for instrument_id, info in self.instruments.items():
            freq_range = info['freq_focus']
            print(f"- {info['color']} {info['name']}: 专注 {freq_range[0]}-{freq_range[1]}Hz")
        print(f"- 柱状图跳动效果：音符攻击时快速上升，然后缓慢下降")
        print(f"- 时间窗口：{self.time_window}秒 (滑动)")
        print(f"- 更新频率：{1000*self.update_interval:.0f}ms")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎶 开始跳动柱状图显示...")
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
        analyzer = BouncingBarsChart()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()