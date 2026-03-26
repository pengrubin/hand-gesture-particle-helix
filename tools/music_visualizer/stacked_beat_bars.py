#!/usr/bin/env python3
"""
纵向重叠节拍推进柱状图
三个乐器的柱子在同一位置纵向重叠显示
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

class StackedBeatBars:
    def __init__(self):
        print("初始化纵向重叠节拍推进柱状图...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件定义 - 调整透明度和层次
        self.instruments = {
            "organ": {  # 底层 - 最厚重
                "file": "../Fugue in G Trio Organ-Organ.mp3",
                "name": "Organ",
                "color": "#4444ff",  # 蓝色
                "alpha": 0.7,       # 较透明
                "layer": 1,         # 底层
                "freq_focus": (50, 500),
                "width_factor": 1.0  # 最宽
            },
            "lute": {   # 中层 - 中等
                "file": "../Fugue in G Trio-Tenor_Lute.mp3", 
                "name": "Lute",
                "color": "#33ff66",  # 绿色
                "alpha": 0.8,       # 中等透明
                "layer": 2,         # 中层
                "freq_focus": (100, 800),
                "width_factor": 0.8  # 中等宽度
            },
            "violin": { # 顶层 - 最突出
                "file": "../Fugue in G Trio violin-Violin.mp3",
                "name": "Violin",
                "color": "#ff3366",  # 红色
                "alpha": 0.9,       # 最不透明
                "layer": 3,         # 顶层
                "freq_focus": (200, 2000),
                "width_factor": 0.6  # 最窄，突出显示
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
        self.fft_size = 4096
        self.update_interval = 0.15  # 150ms一个节拍
        self.num_bars = 16  # 显示16个柱子位置
        self.bar_width = 0.6  # 基础柱子宽度
        self.bar_spacing = 1.0  # 柱子间距
        
        # 柱子数据存储
        self.bar_data = []
        for i in range(self.num_bars):
            self.bar_data.append({"violin": 0.0, "lute": 0.0, "organ": 0.0, "time": 0.0})
        
        # 动画效果参数
        self.bounce_height = 1.3  # 跳动最高点
        self.bounce_frames = 4    # 跳动持续帧数
        self.current_bounce_frame = 0
        self.is_bouncing = False
        
        # 推进动画参数
        self.push_frames = 3      # 推进动画帧数
        self.current_push_frame = 0
        self.is_pushing = False
        self.push_offset = 0.0
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"纵向重叠节拍推进柱状图初始化完成 (显示{self.num_bars}个位置)")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建主图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 12))
        self.fig.suptitle('Stacked Beat Bars - Layered Instrument Visualization', 
                         fontsize=22, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title('Vertically Stacked Musical Beats - Each Beat Pushes Forward', 
                         color='cyan', fontsize=18, pad=25)
        self.ax.set_xlabel('Beat Position (newest on right)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=16)
        self.ax.set_ylim(0, 1.4)  # 给跳动留空间
        
        # 设置x轴范围
        total_width = self.num_bars * self.bar_spacing
        self.ax.set_xlim(-0.5, total_width - 0.5)
        
        # 设置背景网格
        self.ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
        self.ax.tick_params(colors='white', labelsize=14)
        
        # 设置x轴刻度
        tick_positions = [i * self.bar_spacing for i in range(self.num_bars)]
        tick_labels = [f'{i+1}' for i in range(self.num_bars)]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)
        
        # 美化图表
        for spine in self.ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(2)
        
        # 添加图例 - 按层次顺序
        legend_elements = []
        sorted_instruments = sorted(self.instruments.items(), key=lambda x: x[1]['layer'])
        for inst_id, info in sorted_instruments:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=info['color'], 
                                               alpha=info['alpha'], label=f"{info['name']} (Layer {info['layer']})",
                                               edgecolor='white', linewidth=1))
        
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      facecolor='black', edgecolor='white', 
                      fontsize=16, framealpha=0.9, title='Instrument Layers',
                      title_fontsize=14)
        
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
                self.audio_channels[instrument_id] = pygame.mixer.Channel(info["layer"] - 1)
                
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
                sample_pos = sample_pos % len(y)
            
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
                target_magnitudes = positive_magnitude[freq_mask]
                
                # 寻找峰值
                if len(target_magnitudes) > 10:
                    peak_indices = np.argsort(target_magnitudes)[-5:]
                    peak_energy = np.sum(target_magnitudes[peak_indices])
                else:
                    peak_energy = np.sum(target_magnitudes)
                
                # 归一化
                max_possible = np.max(positive_magnitude) + 1e-10
                intensity = min(1.0, peak_energy / max_possible)
                
                # MIDI特性增强
                intensity = self.enhance_midi_characteristics(intensity)
                
                return intensity
            else:
                return 0.0
            
        except Exception as e:
            print(f"{instrument_id} 音频分析错误: {e}")
            return self.generate_mock_intensity(instrument_id, time_pos)
    
    def enhance_midi_characteristics(self, raw_intensity):
        """增强MIDI音频的特性"""
        if raw_intensity > 0.1:
            enhanced = math.pow(raw_intensity, 0.6)
            if enhanced > 0.3:
                enhanced = 0.3 + (enhanced - 0.3) * 1.5
        else:
            enhanced = raw_intensity * 0.5
        
        return min(1.0, enhanced)
    
    def generate_mock_intensity(self, instrument_id: str, time_pos: float):
        """生成模拟的MIDI风格音高强度数据"""
        if instrument_id == "violin":
            # 小提琴：活跃的高音旋律
            beat = time_pos / self.update_interval
            note_pattern = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1]
            base = note_pattern[int(beat) % len(note_pattern)]
            variation = 0.2 * math.sin(time_pos * 0.9)
            
        elif instrument_id == "lute":
            # 鲁特琴：中音和弦
            beat = time_pos / self.update_interval
            chord_pattern = [0.6, 0.2, 0.7, 0.2, 0.5, 0.3, 0.8, 0.1]
            base = chord_pattern[int(beat) % len(chord_pattern)]
            variation = 0.15 * math.sin(time_pos * 0.6)
            
        else:  # organ
            # 管风琴：稳定的低音基础
            beat = time_pos / self.update_interval
            bass_pattern = [0.8, 0.8, 0.7, 0.7, 0.9, 0.9, 0.6, 0.6]
            base = bass_pattern[int(beat) % len(bass_pattern)]
            variation = 0.1 * math.sin(time_pos * 0.4)
        
        intensity = base + variation
        return max(0.0, min(1.0, intensity))
    
    def push_bars_forward(self):
        """将所有柱子向前推进一格"""
        # 移除最左边的柱子，所有柱子向左移动
        self.bar_data.pop(0)
        
        # 在最右边添加新柱子
        self.bar_data.append({"violin": 0.0, "lute": 0.0, "organ": 0.0, "time": self.current_time})
        
        # 开始推进动画
        self.is_pushing = True
        self.current_push_frame = 0
    
    def add_new_bar(self, intensities):
        """添加新的柱子数据并触发跳动效果"""
        # 设置最右边柱子的数据
        self.bar_data[-1] = {
            "violin": intensities["violin"],
            "lute": intensities["lute"], 
            "organ": intensities["organ"],
            "time": self.current_time
        }
        
        # 开始跳动动画
        self.is_bouncing = True
        self.current_bounce_frame = 0
    
    def update_animations(self):
        """更新推进和跳动动画"""
        # 更新推进动画
        if self.is_pushing:
            self.current_push_frame += 1
            # 推进偏移：平滑过渡
            progress = self.current_push_frame / self.push_frames
            self.push_offset = -self.bar_spacing * (1 - math.cos(progress * math.pi)) * 0.5
            
            if self.current_push_frame >= self.push_frames:
                self.is_pushing = False
                self.push_offset = 0.0
        
        # 更新跳动动画
        if self.is_bouncing:
            self.current_bounce_frame += 1
            if self.current_bounce_frame >= self.bounce_frames:
                self.is_bouncing = False
    
    def get_bar_height(self, bar_index, instrument_id):
        """获取柱子的当前高度（考虑跳动效果）"""
        base_height = self.bar_data[bar_index][instrument_id]
        
        # 只有最右边的柱子会跳动
        if bar_index == len(self.bar_data) - 1 and self.is_bouncing:
            progress = self.current_bounce_frame / self.bounce_frames
            if progress < 0.5:
                # 上升阶段
                bounce_factor = 1.0 + (self.bounce_height - 1.0) * (progress * 2)
            else:
                # 下降阶段
                bounce_factor = self.bounce_height - (self.bounce_height - 1.0) * ((progress - 0.5) * 2)
            
            return base_height * bounce_factor
        
        return base_height
    
    def get_bar_position(self, bar_index):
        """获取柱子的当前x位置（考虑推进效果）"""
        base_position = bar_index * self.bar_spacing
        return base_position + self.push_offset
    
    def update_data(self, frame):
        """更新数据的回调函数"""
        # 更新时间
        if self.start_time:
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += self.update_interval
        
        # 更新动画
        self.update_animations()
        
        # 每隔一定帧数添加新柱子
        if frame % int(60 * self.update_interval) == 0:  # 假设60fps
            # 计算当前时间点的音高强度
            current_intensities = {}
            for instrument_id in self.instruments.keys():
                intensity = self.calculate_pitch_intensity(instrument_id, self.current_time)
                current_intensities[instrument_id] = intensity
            
            # 推进柱子并添加新柱子
            self.push_bars_forward()
            self.add_new_bar(current_intensities)
        
        # 清除图表并重绘
        self.ax.clear()
        
        # 重新设置图表属性
        self.ax.set_ylim(0, 1.4)
        total_width = self.num_bars * self.bar_spacing
        self.ax.set_xlim(-0.5, total_width - 0.5)
        self.ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
        self.ax.tick_params(colors='white', labelsize=14)
        
        # 按层次顺序绘制所有柱子（从底层到顶层）
        sorted_instruments = sorted(self.instruments.items(), key=lambda x: x[1]['layer'])
        
        for bar_index in range(len(self.bar_data)):
            x_pos = self.get_bar_position(bar_index)
            
            # 按层次顺序绘制（底层先绘制，顶层后绘制）
            for instrument_id, info in sorted_instruments:
                bar_height = self.get_bar_height(bar_index, instrument_id)
                
                if bar_height > 0.01:
                    # 计算柱子宽度（不同层次有不同宽度）
                    current_width = self.bar_width * info['width_factor']
                    
                    # 计算透明度：历史柱子渐隐
                    base_alpha = info['alpha']
                    if bar_index < len(self.bar_data) - 1:
                        # 历史柱子：基于位置的透明度衰减
                        history_factor = 0.3 + 0.7 * (bar_index / (len(self.bar_data) - 1))
                        alpha = base_alpha * history_factor
                    else:
                        # 最新柱子：完全不透明
                        alpha = base_alpha
                    
                    # 边框设置
                    if bar_index == len(self.bar_data) - 1:
                        # 最新柱子：白色边框
                        edge_color = 'white'
                        edge_width = 2
                    else:
                        # 历史柱子：同色边框
                        edge_color = info['color']
                        edge_width = 1
                    
                    # 绘制柱子
                    self.ax.bar(x_pos, bar_height, width=current_width,
                              color=info['color'], alpha=alpha, 
                              edgecolor=edge_color, linewidth=edge_width,
                              zorder=info['layer'])  # 使用zorder控制层次
                    
                    # 最新柱子的特殊效果
                    if bar_index == len(self.bar_data) - 1 and bar_height > 0.1:
                        # 顶部光点
                        self.ax.scatter(x_pos, bar_height + 0.05, 
                                      color=info['color'], s=100, alpha=alpha, 
                                      zorder=info['layer'] + 10, edgecolor='white', linewidth=1)
                        
                        # 顶层乐器额外的光环效果
                        if instrument_id == "violin":
                            self.ax.scatter(x_pos, bar_height + 0.1, 
                                          color=info['color'], s=150, alpha=0.3, 
                                          zorder=info['layer'] + 5)
        
        # 设置标题和标签
        newest_data = self.bar_data[-1]
        intensity_info = f"Violin: {newest_data['violin']*100:.0f}% | Lute: {newest_data['lute']*100:.0f}% | Organ: {newest_data['organ']*100:.0f}%"
        
        self.ax.set_title(f'Stacked Beat Bars - Layered Musical Visualization\n{intensity_info}', 
                         color='cyan', fontsize=16, pad=25)
        self.ax.set_xlabel('Beat Position (newest on right)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=16)
        
        # 设置x轴刻度
        tick_positions = [i * self.bar_spacing for i in range(self.num_bars)]
        tick_labels = [f'{i+1}' for i in range(self.num_bars)]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)
        
        return []
    
    def play_all(self):
        """播放所有音频文件"""
        print("🎵 开始播放所有音频文件...")
        self.start_time = time.time()
        
        for instrument_id, info in self.instruments.items():
            try:
                sound = self.audio_sounds[instrument_id]
                channel = self.audio_channels[instrument_id]
                channel.play(sound, loops=-1)
                print(f"✅ 播放 {info['name']} (Layer {info['layer']})")
            except Exception as e:
                print(f"❌ 播放失败 {info['name']}: {e}")
    
    def stop_all(self):
        """停止所有音频"""
        print("停止所有音频...")
        pygame.mixer.stop()
        self.start_time = None
    
    def run(self):
        """运行主程序"""
        print("\n🎼 纵向重叠节拍推进柱状图")
        print("=" * 70)
        print("功能：三个乐器的柱子在同一位置纵向重叠，每节拍推进一格")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示说明:")
        print("📊 层次结构 (从底到顶):")
        print("   Layer 1: 管风琴 (蓝色，最宽，底层基础)")
        print("   Layer 2: 鲁特琴 (绿色，中等，中层和弦)")  
        print("   Layer 3: 小提琴 (红色，最窄，顶层旋律)")
        print(f"- 显示 {self.num_bars} 个节拍位置")
        print(f"- 每 {self.update_interval*1000:.0f}ms 推进一次")
        print("- 纵向重叠显示三个乐器强度")
        print("- 历史透明度渐变，最新柱子最亮")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎶 开始纵向重叠节拍显示...")
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_data, 
                interval=50,  # 20fps for smooth animation
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
        analyzer = StackedBeatBars()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()