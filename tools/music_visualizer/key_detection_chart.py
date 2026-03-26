#!/usr/bin/env python3
"""
音调识别折线图
分析MP3文件的音高，识别调性(key)，显示固定时间窗口的折线图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import time
import os
import math
from collections import deque, defaultdict

# 尝试导入音频分析库
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ librosa 可用 - 将进行真实音频分析")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa 不可用 - 将使用模拟分析")

class KeyDetectionChart:
    def __init__(self):
        print("初始化音调识别折线图...")
        
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
        self.fft_size = 4096  # 更大的FFT提高频率分辨率
        self.update_interval = 0.1  # 100ms更新一次
        self.time_window = 60.0  # 固定显示60秒窗口
        
        # 音符定义 (基于A4=440Hz)
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.note_frequencies = self._generate_note_frequencies()
        
        # 常见调性定义
        self.key_signatures = {
            'C major': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'G major': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'D major': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
            'A major': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
            'E major': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
            'B major': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
            'F# major': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
            'F major': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
            'Bb major': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
            'Eb major': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
            'Ab major': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
            'Db major': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
            'A minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'E minor': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
            'B minor': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
            'F# minor': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
            'C# minor': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
            'G# minor': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
            'D minor': ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
            'G minor': ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
            'C minor': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
            'F minor': ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'Eb']
        }
        
        # 历史数据存储 (固定时间窗口)
        self.max_points = int(self.time_window / self.update_interval)  # 600个数据点
        self.time_history = deque(maxlen=self.max_points)
        
        # 只存储关键信息的历史
        self.dominant_note_history = deque(maxlen=self.max_points)  # 主导音符
        self.key_confidence_history = deque(maxlen=self.max_points)  # 调性置信度
        self.detected_key_history = deque(maxlen=self.max_points)   # 检测到的调性
        self.pitch_strength_history = deque(maxlen=self.max_points) # 音高强度
        
        # 当前分析结果
        self.current_analysis = {
            'dominant_note': 'C',
            'detected_key': 'C major',
            'key_confidence': 0.0,
            'pitch_strength': 0.0,
            'note_confidences': {note: 0.0 for note in self.note_names}
        }
        
        # 设置matplotlib
        self.setup_plot()
        
        print("音调识别折线图初始化完成")
    
    def _generate_note_frequencies(self):
        """生成音符频率表 (多个八度)"""
        frequencies = {}
        A4_freq = 440.0
        
        # 生成C1到C8的所有音符频率
        for octave in range(1, 9):
            for i, note in enumerate(self.note_names):
                # 计算相对于A4的半音数
                semitones_from_A4 = (octave - 4) * 12 + (i - 9)  # A是第9个音符(索引9)
                frequency = A4_freq * (2 ** (semitones_from_A4 / 12))
                frequencies[f"{note}{octave}"] = frequency
        
        return frequencies
    
    def setup_plot(self):
        """设置matplotlib图表"""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 1, figsize=(15, 10))
        self.fig.suptitle('Real-time Musical Key Detection', fontsize=16, color='white')
        
        # 第一个子图：主导音符和调性置信度
        ax1 = self.axes[0]
        ax1.set_title('Detected Musical Key & Confidence', color='cyan', fontsize=14)
        ax1.set_xlabel('Time (seconds)', color='white')
        ax1.set_ylabel('Confidence / Strength', color='white')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        # 创建线条
        self.key_confidence_line = ax1.plot([], [], color='orange', label='Key Confidence', linewidth=3)[0]
        self.pitch_strength_line = ax1.plot([], [], color='yellow', label='Pitch Strength', linewidth=2)[0]
        ax1.legend(loc='upper right', facecolor='black', edgecolor='white')
        
        # 第二个子图：音符检测强度
        ax2 = self.axes[1]
        ax2.set_title('Note Detection Over Time', color='lightgreen', fontsize=14)
        ax2.set_xlabel('Time (seconds)', color='white')
        ax2.set_ylabel('Note (C=0, C#=1, D=2, ...)', color='white')
        ax2.set_ylim(-0.5, 11.5)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        
        # 音符散点图
        self.note_scatter = ax2.scatter([], [], c=[], s=[], cmap='viridis', alpha=0.7)
        
        # 设置y轴标签为音符名称
        ax2.set_yticks(range(12))
        ax2.set_yticklabels(self.note_names)
        
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
    
    def frequency_to_note(self, frequency):
        """将频率转换为最接近的音符"""
        if frequency < 20:
            return 'C', 0.0
        
        # 计算相对于A4的半音数
        A4_freq = 440.0
        semitones_from_A4 = 12 * math.log2(frequency / A4_freq)
        
        # 四舍五入到最近的半音
        nearest_semitone = round(semitones_from_A4)
        
        # 计算置信度
        standard_freq = A4_freq * (2 ** (nearest_semitone / 12))
        confidence = max(0, 1 - abs(frequency - standard_freq) / standard_freq)
        
        # 转换为音符名称
        note_index = (nearest_semitone + 9) % 12  # +9因为A是索引9
        note_name = self.note_names[note_index]
        
        return note_name, confidence
    
    def analyze_audio_at_time(self, time_pos: float):
        """分析指定时间点的音频，返回合并的分析结果"""
        if not LIBROSA_AVAILABLE:
            return self.generate_mock_analysis(time_pos)
        
        # 合并所有乐器的音频数据进行分析
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
                sample_pos = sample_pos % len(y)
            
            # 提取音频段
            start = max(0, sample_pos - self.fft_size // 2)
            end = min(len(y), start + self.fft_size)
            
            if end - start > self.fft_size // 2:  # 确保有足够的数据
                audio_segment = np.zeros(self.fft_size)
                audio_segment[:end-start] = y[start:end]
                combined_audio += audio_segment
                valid_instruments += 1
        
        if valid_instruments == 0:
            return self.generate_mock_analysis(time_pos)
        
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
            
            # 找到峰值频率
            peak_indices = np.argsort(positive_magnitude)[-10:]  # 前10个峰值
            peak_frequencies = positive_freqs[peak_indices]
            peak_magnitudes = positive_magnitude[peak_indices]
            
            # 分析音符强度
            note_confidences = {note: 0.0 for note in self.note_names}
            total_strength = 0.0
            
            for freq, mag in zip(peak_frequencies, peak_magnitudes):
                if 80 <= freq <= 2000:  # 主要音符范围
                    note, confidence = self.frequency_to_note(freq)
                    weight = mag / np.max(positive_magnitude)
                    note_confidences[note] += confidence * weight
                    total_strength += weight
            
            # 归一化音符置信度
            if total_strength > 0:
                for note in note_confidences:
                    note_confidences[note] /= total_strength
            
            # 找到主导音符
            dominant_note = max(note_confidences, key=note_confidences.get)
            
            # 检测调性
            detected_key, key_confidence = self.detect_key(note_confidences)
            
            return {
                'dominant_note': dominant_note,
                'detected_key': detected_key,
                'key_confidence': key_confidence,
                'pitch_strength': total_strength,
                'note_confidences': note_confidences
            }
            
        except Exception as e:
            print(f"音频分析错误: {e}")
            return self.generate_mock_analysis(time_pos)
    
    def detect_key(self, note_confidences):
        """根据音符强度检测调性"""
        key_scores = {}
        
        for key_name, key_notes in self.key_signatures.items():
            score = 0.0
            for note in key_notes:
                # 处理升降号
                if note in note_confidences:
                    score += note_confidences[note]
                elif note == 'Bb' and 'A#' in note_confidences:
                    score += note_confidences['A#']
                elif note == 'Db' and 'C#' in note_confidences:
                    score += note_confidences['C#']
                elif note == 'Eb' and 'D#' in note_confidences:
                    score += note_confidences['D#']
                elif note == 'Gb' and 'F#' in note_confidences:
                    score += note_confidences['F#']
                elif note == 'Ab' and 'G#' in note_confidences:
                    score += note_confidences['G#']
            
            key_scores[key_name] = score / len(key_notes)  # 平均分数
        
        # 找到最高分的调性
        best_key = max(key_scores, key=key_scores.get)
        confidence = key_scores[best_key]
        
        return best_key, confidence
    
    def generate_mock_analysis(self, time_pos: float):
        """生成模拟的音调分析数据"""
        # 模拟一个在G major调性中的音乐
        base_notes = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
        
        # 根据时间创建变化的音符模式
        note_index = int(time_pos * 0.5) % len(base_notes)
        dominant_note = base_notes[note_index]
        
        # 创建音符置信度
        note_confidences = {note: 0.0 for note in self.note_names}
        for note in base_notes:
            if note == dominant_note:
                note_confidences[note] = 0.8 + 0.2 * math.sin(time_pos * 2)
            else:
                note_confidences[note] = 0.1 + 0.1 * math.sin(time_pos + hash(note) % 10)
        
        return {
            'dominant_note': dominant_note,
            'detected_key': 'G major',
            'key_confidence': 0.7 + 0.2 * math.sin(time_pos * 0.3),
            'pitch_strength': 0.6 + 0.3 * math.sin(time_pos * 1.5),
            'note_confidences': note_confidences
        }
    
    def update_data(self, frame):
        """更新数据的回调函数"""
        # 更新时间
        if self.start_time:
            self.current_time = time.time() - self.start_time
        else:
            self.current_time += self.update_interval
        
        # 分析当前时间点的音频
        analysis = self.analyze_audio_at_time(self.current_time)
        self.current_analysis = analysis
        
        # 存储历史数据
        self.time_history.append(self.current_time)
        self.dominant_note_history.append(analysis['dominant_note'])
        self.key_confidence_history.append(analysis['key_confidence'])
        self.detected_key_history.append(analysis['detected_key'])
        self.pitch_strength_history.append(analysis['pitch_strength'])
        
        # 更新图表
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # 更新第一个子图（调性置信度和音高强度）
            key_conf = list(self.key_confidence_history)
            pitch_str = list(self.pitch_strength_history)
            
            self.key_confidence_line.set_data(times, key_conf)
            self.pitch_strength_line.set_data(times, pitch_str)
            
            # 更新第二个子图（音符检测散点图）
            note_indices = []
            note_times = []
            note_strengths = []
            
            for i, note_name in enumerate(self.dominant_note_history):
                if note_name in self.note_names:
                    note_index = self.note_names.index(note_name)
                    note_indices.append(note_index)
                    note_times.append(times[i])
                    note_strengths.append(self.pitch_strength_history[i] * 100)  # 调整点大小
            
            # 更新散点图
            if note_times:
                self.note_scatter.set_offsets(np.column_stack([note_times, note_indices]))
                self.note_scatter.set_sizes(note_strengths)
                colors = plt.cm.viridis(np.array(note_strengths) / 100)
                self.note_scatter.set_color(colors)
            
            # 设置固定的x轴范围（时间窗口）
            current_end = times[-1]
            window_start = max(0, current_end - self.time_window)
            
            for ax in self.axes:
                ax.set_xlim(window_start, current_end)
        
        # 更新图表标题显示当前信息
        current_info = f"Current: {analysis['detected_key']} (Confidence: {analysis['key_confidence']:.2f}) | Note: {analysis['dominant_note']}"
        self.axes[0].set_title(f'Detected Musical Key & Confidence\n{current_info}', color='cyan', fontsize=12)
        
        return [self.key_confidence_line, self.pitch_strength_line, self.note_scatter]
    
    def play_all(self):
        """播放所有音频文件"""
        print("开始播放所有音频文件...")
        self.start_time = time.time()
        
        for name, sound in self.audio_sounds.items():
            try:
                channel = self.audio_channels[name]
                channel.play(sound, loops=-1)
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
        print("\n🎼 音调识别折线图")
        print("=" * 60)
        print("功能：分析MP3音频，识别调性(key)，显示固定时间窗口的变化")
        print()
        
        # 加载音频文件
        if not self.load_audio_files():
            print("❌ 无法加载音频文件")
            return
        
        print("🎯 显示信息:")
        print("- 上图：调性置信度(橙色) 和 音高强度(黄色)")
        print("- 下图：检测到的音符随时间变化(散点图)")
        print(f"- 固定时间窗口：{self.time_window}秒")
        print("- 关闭窗口退出程序")
        print()
        
        # 自动开始播放
        self.play_all()
        
        try:
            print("🎵 开始音调识别分析...")
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
        analyzer = KeyDetectionChart()
        analyzer.run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()