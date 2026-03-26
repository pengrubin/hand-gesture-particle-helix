#!/usr/bin/env python3
"""
音高位置柱状图
根据音调频率决定柱子的垂直位置，实现半透明叠加
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

class PitchHeightBars:
    def __init__(self):
        print("初始化音高位置柱状图...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件定义
        self.instruments = {
            "organ": {  # 低音 - 底部
                "file": "../Fugue in G Trio Organ-Organ.mp3",
                "name": "Organ",
                "color": "#4466ff",  # 蓝色
                "alpha": 0.6,      
                "freq_focus": (50, 500),
                "pitch_range": (0.0, 0.4)  # 垂直位置范围：底部
            },
            "lute": {   # 中音 - 中部
                "file": "../Fugue in G Trio-Tenor_Lute.mp3", 
                "name": "Lute",
                "color": "#44ff66",  # 绿色
                "alpha": 0.6,       
                "freq_focus": (100, 800),
                "pitch_range": (0.3, 0.7)  # 垂直位置范围：中部
            },
            "violin": { # 高音 - 顶部
                "file": "../Fugue in G Trio violin-Violin.mp3",
                "name": "Violin",
                "color": "#ff4466",  # 红色
                "alpha": 0.6,       
                "freq_focus": (200, 2000),
                "pitch_range": (0.6, 1.0)  # 垂直位置范围：顶部
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
        self.update_interval = 0.12   # 120ms一次盖章
        self.stamp_interval = 0.02    # 20ms一次画布移动
        
        # 画布参数
        self.canvas_width = 30        # 画布显示30个柱子位置
        self.stamp_position = 25      # 印章固定在第25个位置
        self.bar_width = 0.95         # 柱子宽度
        self.bar_height = 0.15        # 柱子高度（垂直方向的厚度）
        self.move_speed = 1.0         # 画布移动速度
        
        # 画布数据存储
        self.canvas_buffer_size = 100
        self.canvas_data = []
        for i in range(self.canvas_buffer_size):
            self.canvas_data.append({
                "violin": {"pitch": 0.8, "intensity": 0.0},
                "lute": {"pitch": 0.5, "intensity": 0.0},
                "organ": {"pitch": 0.2, "intensity": 0.0},
                "time": 0.0
            })
        
        # 画布偏移量
        self.canvas_offset = 0.0
        self.target_canvas_offset = 0.0
        
        # 印章动画参数
        self.stamp_bounce_height = 1.15
        self.stamp_frames = 4
        self.current_stamp_frame = 0
        self.is_stamping = False
        
        # 数据更新计数器
        self.frame_counter = 0
        self.stamp_counter = 0
        self.frames_per_stamp = int(60 * self.stamp_interval)
        self.frames_per_update = int(60 * self.update_interval)
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"音高位置柱状图初始化完成 (印章位置: {self.stamp_position})")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建主图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 12))
        self.fig.suptitle('Pitch-Based Height Visualization - Each Instrument at Different Pitch Level', 
                         fontsize=24, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title('Musical Pitch Heights - Low/Mid/High Frequency Separation', 
                         color='cyan', fontsize=18, pad=25)
        self.ax.set_xlabel('Canvas Position (stamp at fixed position)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Height (Frequency)', color='white', fontsize=16)
        self.ax.set_ylim(0, 1.2)
        
        # 设置x轴范围
        self.ax.set_xlim(0, self.canvas_width)
        
        # 设置背景网格
        self.ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 添加音高区域标签
        self.ax.axhspan(0.0, 0.4, alpha=0.05, color='blue', label='Low (Organ)')
        self.ax.axhspan(0.3, 0.7, alpha=0.05, color='green', label='Mid (Lute)')
        self.ax.axhspan(0.6, 1.0, alpha=0.05, color='red', label='High (Violin)')
        
        # 添加印章位置指示线
        self.stamp_line = self.ax.axvline(x=self.stamp_position, color='yellow', 
                                         linestyle='--', linewidth=2, alpha=0.8, 
                                         label='Stamp Position')
        
        # 美化图表
        for spine in self.ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(2)
        
        # 添加图例
        legend_elements = []
        for inst_id, info in self.instruments.items():
            pitch_range = info['pitch_range']
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=info['color'], 
                                               alpha=info['alpha'], 
                                               label=f"{info['name']} ({pitch_range[0]:.1f}-{pitch_range[1]:.1f})",
                                               edgecolor='white', linewidth=1))
        
        legend_elements.append(plt.Line2D([0], [0], color='yellow', linestyle='--', 
                                        linewidth=2, label='Fixed Stamp Position'))
        
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      facecolor='black', edgecolor='white', 
                      fontsize=14, framealpha=0.9, title='Pitch Ranges',
                      title_fontsize=12)
        
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
                channel_id = {"organ": 0, "lute": 1, "violin": 2}[instrument_id]
                self.audio_channels[instrument_id] = pygame.mixer.Channel(channel_id)
                
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
    
    def calculate_pitch_and_intensity(self, instrument_id: str, time_pos: float):
        """计算指定乐器的音高（频率）和强度"""
        if not LIBROSA_AVAILABLE or instrument_id not in self.audio_data:
            return self.generate_mock_pitch_intensity(instrument_id, time_pos)
        
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
                return {"pitch": 0.5, "intensity": 0.0}
            
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
                target_freqs = positive_freqs[freq_mask]
                
                # 找到主频率（最强的频率）
                if len(target_magnitudes) > 0:
                    max_idx = np.argmax(target_magnitudes)
                    dominant_freq = target_freqs[max_idx]
                    
                    # 计算音高位置（归一化到乐器的音高范围）
                    pitch_range = self.instruments[instrument_id]["pitch_range"]
                    freq_normalized = (dominant_freq - freq_range[0]) / (freq_range[1] - freq_range[0])
                    pitch_position = pitch_range[0] + freq_normalized * (pitch_range[1] - pitch_range[0])
                    pitch_position = max(pitch_range[0], min(pitch_range[1], pitch_position))
                    
                    # 计算强度
                    peak_indices = np.argsort(target_magnitudes)[-5:]
                    peak_energy = np.sum(target_magnitudes[peak_indices])
                    max_possible = np.max(positive_magnitude) + 1e-10
                    intensity = min(1.0, peak_energy / max_possible)
                    
                    # MIDI特性增强
                    intensity = self.enhance_midi_characteristics(intensity)
                    
                    return {"pitch": pitch_position, "intensity": intensity}
            
            return {"pitch": 0.5, "intensity": 0.0}
            
        except Exception as e:
            print(f"{instrument_id} 音频分析错误: {e}")
            return self.generate_mock_pitch_intensity(instrument_id, time_pos)
    
    def enhance_midi_characteristics(self, raw_intensity):
        """增强MIDI音频的特性"""
        if raw_intensity > 0.1:
            enhanced = math.pow(raw_intensity, 0.6)
            if enhanced > 0.3:
                enhanced = 0.3 + (enhanced - 0.3) * 1.5
        else:
            enhanced = raw_intensity * 0.5
        
        return min(1.0, enhanced)
    
    def generate_mock_pitch_intensity(self, instrument_id: str, time_pos: float):
        """生成模拟的音高和强度数据"""
        # 使用模运算确保时间值在合理范围内，避免数值问题
        normalized_time = time_pos % 60.0  # 每60秒循环一次
        beat = normalized_time / self.update_interval
        
        if instrument_id == "violin":
            # 小提琴：高音区变化
            pitch_range = self.instruments[instrument_id]["pitch_range"]
            note_pattern = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.85, 0.1]
            intensity = note_pattern[int(beat) % len(note_pattern)]
            
            # 音高在高音区变化
            pitch_variation = 0.2 * math.sin(normalized_time * 1.2)
            pitch = pitch_range[0] + (pitch_range[1] - pitch_range[0]) * (0.7 + pitch_variation)
            
        elif instrument_id == "lute":
            # 鲁特琴：中音区变化
            pitch_range = self.instruments[instrument_id]["pitch_range"]
            chord_pattern = [0.65, 0.25, 0.7, 0.2, 0.6, 0.3, 0.75, 0.15]
            intensity = chord_pattern[int(beat) % len(chord_pattern)]
            
            # 音高在中音区变化
            pitch_variation = 0.15 * math.sin(normalized_time * 0.8)
            pitch = pitch_range[0] + (pitch_range[1] - pitch_range[0]) * (0.5 + pitch_variation)
            
        else:  # organ
            # 管风琴：低音区稳定
            pitch_range = self.instruments[instrument_id]["pitch_range"]
            bass_pattern = [0.8, 0.75, 0.7, 0.72, 0.85, 0.8, 0.65, 0.7]
            intensity = bass_pattern[int(beat) % len(bass_pattern)]
            
            # 音高在低音区稳定
            pitch_variation = 0.1 * math.sin(normalized_time * 0.5)
            pitch = pitch_range[0] + (pitch_range[1] - pitch_range[0]) * (0.3 + pitch_variation)
        
        return {"pitch": pitch, "intensity": intensity}
    
    def stamp_new_data(self):
        """在固定位置盖新章（更新数据）"""
        # 计算当前时间点的音高和强度
        current_data = {}
        for instrument_id in self.instruments.keys():
            data = self.calculate_pitch_and_intensity(instrument_id, self.current_time)
            current_data[instrument_id] = data
        
        # 计算画布上的实际位置
        canvas_pos = int(self.canvas_offset + self.stamp_position)
        if canvas_pos < len(self.canvas_data):
            # 在画布上盖章
            self.canvas_data[canvas_pos] = {
                "violin": current_data["violin"],
                "lute": current_data["lute"], 
                "organ": current_data["organ"],
                "time": self.current_time
            }
        
        # 开始印章动画
        self.is_stamping = True
        self.current_stamp_frame = 0
        
        # 设置画布移动目标
        self.target_canvas_offset += self.move_speed
    
    def move_canvas(self):
        """平滑移动画布"""
        if abs(self.canvas_offset - self.target_canvas_offset) > 0.01:
            diff = self.target_canvas_offset - self.canvas_offset
            self.canvas_offset += diff * 0.15
        else:
            self.canvas_offset = self.target_canvas_offset
    
    def update_stamp_animation(self):
        """更新印章动画"""
        if self.is_stamping:
            self.current_stamp_frame += 1
            if self.current_stamp_frame >= self.stamp_frames:
                self.is_stamping = False
    
    def get_stamp_height_factor(self, bar_index):
        """获取印章位置的高度因子"""
        display_pos = bar_index - self.canvas_offset
        
        if abs(display_pos - self.stamp_position) < 0.5 and self.is_stamping:
            progress = self.current_stamp_frame / self.stamp_frames
            if progress < 0.3:
                return 1.0 + (self.stamp_bounce_height - 1.0) * (progress / 0.3)
            else:
                return self.stamp_bounce_height - (self.stamp_bounce_height - 1.0) * ((progress - 0.3) / 0.7)
        
        return 1.0
    
    def update_data(self, frame):
        """更新数据的回调函数"""
        self.frame_counter += 1
        self.stamp_counter += 1
        
        # 更新时间
        if self.start_time:
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += self.stamp_interval
        
        # 更新印章动画
        self.update_stamp_animation()
        
        # 定期移动画布
        if self.stamp_counter % self.frames_per_stamp == 0:
            self.move_canvas()
            self.stamp_counter = 0  # 重置计数器防止溢出
        
        # 定期盖章 - 使用更稳定的时间基准
        if self.frame_counter % self.frames_per_update == 0:
            self.stamp_new_data()
            self.frame_counter = 0  # 重置计数器防止溢出
        
        # 清除图表并重绘
        self.ax.clear()
        
        # 重新设置图表属性
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xlim(0, self.canvas_width)
        self.ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 重新添加音高区域背景
        self.ax.axhspan(0.0, 0.4, alpha=0.03, color='blue')
        self.ax.axhspan(0.3, 0.7, alpha=0.03, color='green')
        self.ax.axhspan(0.6, 1.0, alpha=0.03, color='red')
        
        # 重新绘制印章位置线
        self.ax.axvline(x=self.stamp_position, color='yellow', 
                       linestyle='--', linewidth=3, alpha=0.9)
        
        # 添加印章位置标签
        self.ax.text(self.stamp_position + 0.5, 1.1, 'STAMP', 
                    color='yellow', fontsize=12, fontweight='bold')
        
        # 计算可见范围
        start_index = max(0, int(self.canvas_offset))
        end_index = min(len(self.canvas_data), int(self.canvas_offset + self.canvas_width + 2))
        
        # 绘制所有乐器的柱子（不按层次，各自独立）
        for canvas_index in range(start_index, end_index):
            display_x = canvas_index - self.canvas_offset
            
            if 0 <= display_x <= self.canvas_width:
                # 绘制每个乐器（独立位置，不重叠）
                for instrument_id, info in self.instruments.items():
                    data = self.canvas_data[canvas_index][instrument_id]
                    pitch_height = data["pitch"]
                    intensity = data["intensity"]
                    
                    if intensity > 0.01:
                        # 应用印章动画效果
                        height_factor = self.get_stamp_height_factor(canvas_index)
                        
                        # 计算透明度：固定半透明，不随距离变化
                        alpha = info['alpha']
                        
                        # 边框设置
                        distance_from_stamp = abs(display_x - self.stamp_position)
                        if distance_from_stamp < 2:
                            edge_color = 'white'
                            edge_width = 1.5
                        else:
                            edge_color = info['color']
                            edge_width = 0.5
                        
                        # 根据音高位置绘制柱子
                        bar_y = pitch_height  # 使用计算出的音高位置
                        bar_height = self.bar_height * intensity * height_factor
                        
                        # 绘制柱子（水平条形，垂直位置由音高决定）
                        self.ax.bar(display_x, bar_height, width=self.bar_width,
                                  bottom=bar_y - bar_height/2,  # 居中在音高位置
                                  color=info['color'], alpha=alpha, 
                                  edgecolor=edge_color, linewidth=edge_width)
                        
                        # 印章位置的特殊效果
                        if distance_from_stamp < 0.5 and intensity > 0.1:
                            self.ax.scatter(display_x, bar_y, 
                                          color='yellow', s=100, alpha=0.8, 
                                          zorder=10, marker='*')
        
        # 设置标题和标签
        canvas_pos = int(self.canvas_offset + self.stamp_position)
        if canvas_pos < len(self.canvas_data):
            current_data = self.canvas_data[canvas_pos]
            pitch_info = []
            for inst_id, info in self.instruments.items():
                pitch = current_data[inst_id]["pitch"]
                intensity = current_data[inst_id]["intensity"]
                pitch_info.append(f"{info['name']}: P={pitch:.2f} I={intensity*100:.0f}%")
            intensity_info = " | ".join(pitch_info)
        else:
            intensity_info = "Initializing..."
        
        self.ax.set_title(f'Pitch-Based Height Visualization\\n{intensity_info}', 
                         color='cyan', fontsize=16, pad=25)
        self.ax.set_xlabel('Canvas Position (moves left continuously)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Height (Low → Mid → High)', color='white', fontsize=16)
        
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
                pitch_range = info['pitch_range']
                print(f"✅ 播放 {info['name']} (音高范围: {pitch_range[0]:.1f}-{pitch_range[1]:.1f})")
            except Exception as e:
                print(f"❌ 播放失败 {info['name']}: {e}")
    
    def stop_all(self):
        """停止所有音频"""
        print("停止所有音频...")
        pygame.mixer.stop()
        self.start_time = None
    
    def run(self):
        """运行主程序"""
        print("\n🎼 音高位置柱状图")
        print("=" * 70)
        print("功能：根据音调频率决定柱子的垂直位置")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示说明:")
        print("📍 音高分层:")
        print("   - 管风琴：低音区 (0.0-0.4) 蓝色")
        print("   - 鲁特琴：中音区 (0.3-0.7) 绿色")
        print("   - 小提琴：高音区 (0.6-1.0) 红色")
        print("📊 可视化特点:")
        print("   - 垂直位置由音调频率决定")
        print("   - 柱子高度由音量强度决定")
        print("   - 半透明独立显示，不混色")
        print("   - 无渐变变暗效果")
        print(f"- 印章位置：第 {self.stamp_position} 个位置")
        print(f"- 每 {self.update_interval*1000:.0f}ms 盖一次章")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎶 开始音高位置可视化...")
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_data, 
                interval=17,  # ~60fps
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
        analyzer = PitchHeightBars()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()