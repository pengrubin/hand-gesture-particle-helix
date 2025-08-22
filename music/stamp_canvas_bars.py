#!/usr/bin/env python3
"""
印章画布柱状图
固定位置盖章，画布匀速左移的音频可视化
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

class StampCanvasBars:
    def __init__(self):
        print("初始化印章画布柱状图...")
        
        # 初始化pygame音频
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 音频文件定义
        self.instruments = {
            "organ": {  # 底层
                "file": "../Fugue in G Trio Organ-Organ.mp3",
                "name": "Organ",
                "color": "#4466ff",  # 蓝色
                "alpha": 0.6,      
                "layer": 1,         
                "freq_focus": (50, 500)
            },
            "lute": {   # 中层
                "file": "../Fugue in G Trio-Tenor_Lute.mp3", 
                "name": "Lute",
                "color": "#44ff66",  # 绿色
                "alpha": 0.7,       
                "layer": 2,         
                "freq_focus": (100, 800)
            },
            "violin": { # 顶层
                "file": "../Fugue in G Trio violin-Violin.mp3",
                "name": "Violin",
                "color": "#ff4466",  # 红色
                "alpha": 0.8,       
                "layer": 3,         
                "freq_focus": (200, 2000)
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
        self.stamp_interval = 0.02    # 20ms一次画布移动（更细腻的移动）
        
        # 画布参数
        self.canvas_width = 30        # 画布显示30个柱子位置
        self.stamp_position = 25      # 印章固定在第25个位置（靠右）
        self.bar_width = 0.95         # 柱子宽度
        self.move_speed = 1.0         # 画布移动速度（每次盖章移动1个单位）
        
        # 画布数据存储 - 使用更大的缓冲区
        self.canvas_buffer_size = 100  # 画布缓冲区大小
        self.canvas_data = []
        for i in range(self.canvas_buffer_size):
            self.canvas_data.append({"violin": 0.0, "lute": 0.0, "organ": 0.0, "time": 0.0})
        
        # 画布偏移量（控制显示窗口）
        self.canvas_offset = 0.0      # 画布当前偏移
        self.target_canvas_offset = 0.0
        
        # 印章动画参数
        self.stamp_bounce_height = 1.15  # 印章跳动高度
        self.stamp_frames = 4           # 印章动画帧数
        self.current_stamp_frame = 0
        self.is_stamping = False
        
        # 数据更新计数器
        self.frame_counter = 0
        self.stamp_counter = 0
        self.frames_per_stamp = int(60 * self.stamp_interval)      # 画布移动频率
        self.frames_per_update = int(60 * self.update_interval)    # 数据更新频率
        
        # 设置matplotlib
        self.setup_plot()
        
        print(f"印章画布柱状图初始化完成 (印章位置: {self.stamp_position})")
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        
        # 创建主图
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 12))
        self.fig.suptitle('Stamp Canvas - Fixed Position Stamping with Moving Canvas', 
                         fontsize=24, color='white', y=0.95)
        
        # 设置主图
        self.ax.set_title('Musical Stamp Effect - Canvas Moves Left, Stamp Stays Fixed', 
                         color='cyan', fontsize=18, pad=25)
        self.ax.set_xlabel('Canvas Position (stamp at fixed position)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=16)
        self.ax.set_ylim(0, 1.2)
        
        # 设置x轴范围（固定的显示窗口）
        self.ax.set_xlim(0, self.canvas_width)
        
        # 设置背景网格
        self.ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
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
        sorted_instruments = sorted(self.instruments.items(), key=lambda x: x[1]['layer'])
        for inst_id, info in sorted_instruments:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=info['color'], 
                                               alpha=info['alpha'], label=f"{info['name']} (Layer {info['layer']})",
                                               edgecolor='white', linewidth=1))
        
        # 添加印章位置到图例
        legend_elements.append(plt.Line2D([0], [0], color='yellow', linestyle='--', 
                                        linewidth=2, label='Fixed Stamp Position'))
        
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      facecolor='black', edgecolor='white', 
                      fontsize=14, framealpha=0.9, title='Layers & Stamp',
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
            note_pattern = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.85, 0.1]
            base = note_pattern[int(beat) % len(note_pattern)]
            variation = 0.2 * math.sin(time_pos * 0.9)
            
        elif instrument_id == "lute":
            # 鲁特琴：中音和弦
            beat = time_pos / self.update_interval
            chord_pattern = [0.65, 0.25, 0.7, 0.2, 0.6, 0.3, 0.75, 0.15]
            base = chord_pattern[int(beat) % len(chord_pattern)]
            variation = 0.15 * math.sin(time_pos * 0.6)
            
        else:  # organ
            # 管风琴：稳定的低音基础
            beat = time_pos / self.update_interval
            bass_pattern = [0.8, 0.75, 0.7, 0.72, 0.85, 0.8, 0.65, 0.7]
            base = bass_pattern[int(beat) % len(bass_pattern)]
            variation = 0.1 * math.sin(time_pos * 0.4)
        
        intensity = base + variation
        return max(0.0, min(1.0, intensity))
    
    def stamp_new_data(self):
        """在固定位置盖新章（更新数据）"""
        # 计算当前时间点的音高强度
        current_intensities = {}
        for instrument_id in self.instruments.keys():
            intensity = self.calculate_pitch_intensity(instrument_id, self.current_time)
            current_intensities[instrument_id] = intensity
        
        # 计算画布上的实际位置
        canvas_pos = int(self.canvas_offset + self.stamp_position)
        if canvas_pos < len(self.canvas_data):
            # 在画布上盖章
            self.canvas_data[canvas_pos] = {
                "violin": current_intensities["violin"],
                "lute": current_intensities["lute"], 
                "organ": current_intensities["organ"],
                "time": self.current_time
            }
        
        # 开始印章动画
        self.is_stamping = True
        self.current_stamp_frame = 0
        
        # 设置画布移动目标
        self.target_canvas_offset += self.move_speed
    
    def move_canvas(self):
        """平滑移动画布"""
        # 平滑移动到目标位置
        if abs(self.canvas_offset - self.target_canvas_offset) > 0.01:
            # 使用缓动函数
            diff = self.target_canvas_offset - self.canvas_offset
            self.canvas_offset += diff * 0.15  # 平滑移动
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
        # 检查是否是印章位置
        display_pos = bar_index - self.canvas_offset
        
        if abs(display_pos - self.stamp_position) < 0.5 and self.is_stamping:
            # 印章动画：快速下压然后回弹
            progress = self.current_stamp_frame / self.stamp_frames
            if progress < 0.3:
                # 下压阶段
                return 1.0 + (self.stamp_bounce_height - 1.0) * (progress / 0.3)
            else:
                # 回弹阶段
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
        
        # 定期盖章（更新数据）
        if self.frame_counter % self.frames_per_update == 0:
            self.stamp_new_data()
        
        # 清除图表并重绘
        self.ax.clear()
        
        # 重新设置图表属性
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xlim(0, self.canvas_width)
        self.ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        self.ax.tick_params(colors='white', labelsize=12)
        
        # 重新绘制印章位置线
        self.ax.axvline(x=self.stamp_position, color='yellow', 
                       linestyle='--', linewidth=3, alpha=0.9)
        
        # 添加印章位置标签
        self.ax.text(self.stamp_position + 0.5, 1.1, 'STAMP', 
                    color='yellow', fontsize=12, fontweight='bold')
        
        # 按层次顺序绘制可见范围内的柱子
        sorted_instruments = sorted(self.instruments.items(), key=lambda x: x[1]['layer'])
        
        # 计算可见范围
        start_index = max(0, int(self.canvas_offset))
        end_index = min(len(self.canvas_data), int(self.canvas_offset + self.canvas_width + 2))
        
        for canvas_index in range(start_index, end_index):
            # 计算显示位置
            display_x = canvas_index - self.canvas_offset
            
            # 只绘制在可见范围内的柱子
            if 0 <= display_x <= self.canvas_width:
                # 按层次顺序绘制
                for instrument_id, info in sorted_instruments:
                    bar_height = self.canvas_data[canvas_index][instrument_id]
                    
                    if bar_height > 0.01:
                        # 应用印章动画效果
                        height_factor = self.get_stamp_height_factor(canvas_index)
                        animated_height = bar_height * height_factor
                        
                        # 计算透明度：离印章位置越远越透明
                        distance_from_stamp = abs(display_x - self.stamp_position)
                        fade_factor = max(0.3, 1.0 - distance_from_stamp * 0.04)
                        alpha = info['alpha'] * fade_factor
                        
                        # 边框设置
                        if distance_from_stamp < 2:  # 印章附近的柱子
                            edge_color = 'white'
                            edge_width = 1.5
                        else:
                            edge_color = info['color']
                            edge_width = 0.5
                        
                        # 绘制柱子
                        self.ax.bar(display_x, animated_height, width=self.bar_width,
                                  color=info['color'], alpha=alpha, 
                                  edgecolor=edge_color, linewidth=edge_width,
                                  zorder=info['layer'])
                        
                        # 印章位置的特殊效果
                        if distance_from_stamp < 0.5 and animated_height > 0.1:
                            # 印章光效
                            self.ax.scatter(display_x, animated_height + 0.05, 
                                          color='yellow', s=100, alpha=0.8, 
                                          zorder=10, marker='*')
        
        # 设置标题和标签
        canvas_pos = int(self.canvas_offset + self.stamp_position)
        if canvas_pos < len(self.canvas_data):
            current_data = self.canvas_data[canvas_pos]
            intensity_info = f"Violin: {current_data['violin']*100:.0f}% | Lute: {current_data['lute']*100:.0f}% | Organ: {current_data['organ']*100:.0f}%"
        else:
            intensity_info = "Initializing..."
        
        self.ax.set_title(f'Stamp Canvas Effect - Fixed Position Stamping\n{intensity_info}', 
                         color='cyan', fontsize=16, pad=25)
        self.ax.set_xlabel('Canvas Position (moves left continuously)', color='white', fontsize=16)
        self.ax.set_ylabel('Pitch Intensity', color='white', fontsize=16)
        
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
        print("\n📏 印章画布柱状图")
        print("=" * 70)
        print("功能：固定位置盖章，画布匀速左移的音频可视化")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示说明:")
        print("📍 印章效果:")
        print("   - 黄色虚线：固定的印章位置")
        print("   - 印章始终在同一位置盖章")
        print("   - 画布向左连续移动")
        print("   - 每次盖章都有下压回弹动画")
        print("📊 层次结构:")
        print("   Layer 1: 管风琴 (蓝色，底层基础)")
        print("   Layer 2: 鲁特琴 (绿色，中层和弦)")  
        print("   Layer 3: 小提琴 (红色，顶层旋律)")
        print(f"- 印章位置：第 {self.stamp_position} 个位置")
        print(f"- 每 {self.update_interval*1000:.0f}ms 盖一次章")
        print("- 画布连续向左移动")
        print("- 离印章越远透明度越低")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎶 开始印章画布效果...")
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_data, 
                interval=17,  # ~60fps for smooth canvas movement
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
        analyzer = StampCanvasBars()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()