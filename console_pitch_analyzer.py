#!/usr/bin/env python3
"""
控制台音频音调分析器
在终端显示MP3文件的音调高低分析，无需GUI
"""

import os
import time
import math
import numpy as np

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

class ConsolePitchAnalyzer:
    def __init__(self):
        self.audio_files = {
            "violin": "Fugue in G Trio violin-Violin.mp3",
            "lute": "Fugue in G Trio-Tenor_Lute.mp3", 
            "organ": "Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 音调范围定义 (Hz)
        self.pitch_ranges = {
            "Sub Bass": (20, 60),        # 超低音
            "Bass": (60, 250),           # 低音  
            "Low Mid": (250, 500),       # 中低音
            "Mid": (500, 2000),          # 中音
            "High Mid": (2000, 4000),    # 中高音
            "Presence": (4000, 8000),    # 临场感
            "Brilliance": (8000, 20000)  # 明亮度
        }
        
        # 音频数据存储
        self.audio_data = {}
        self.sample_rates = {}
        self.durations = {}
        
    def check_files(self):
        """检查音频文件"""
        print("🎵 检查MP3文件:")
        all_exist = True
        
        for name, file_path in self.audio_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"✅ {name}: {file_path} ({size:.1f}MB)")
            else:
                print(f"❌ 缺失: {file_path}")
                all_exist = False
                
        return all_exist
    
    def load_audio_files(self):
        """加载音频文件"""
        if not LIBROSA_AVAILABLE:
            print("⚠️ 无法进行真实音频分析，将使用模拟数据演示")
            return True
            
        print("\n🔧 加载音频文件进行分析...")
        
        for name, file_path in self.audio_files.items():
            if not os.path.exists(file_path):
                continue
                
            try:
                print(f"正在加载 {name}...")
                y, sr = librosa.load(file_path, sr=22050)
                self.audio_data[name] = y
                self.sample_rates[name] = sr
                self.durations[name] = len(y) / sr
                print(f"✅ {name}: {self.durations[name]:.1f}秒, {sr}Hz")
                
            except Exception as e:
                print(f"❌ {name} 加载失败: {e}")
                
        return len(self.audio_data) > 0 or not LIBROSA_AVAILABLE
    
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
                sample_pos = sample_pos % len(y)  # 循环
            
            # 提取音频段进行分析
            window_size = 2048
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
                    pitch_analysis[range_name] = min(1.0, energy / (np.max(positive_magnitude) + 1e-10))
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
            variation = math.sin(time_pos * 2 + hash(range_name) % 10) * 0.3
            pitch_analysis[range_name] = max(0, min(1, base_strength + variation))
        
        return pitch_analysis
    
    def print_bar_chart(self, value, max_width=20):
        """打印文本条形图"""
        bar_length = int(value * max_width)
        bar = "█" * bar_length + "░" * (max_width - bar_length)
        return f"{bar} {value:.2f}"
    
    def analyze_and_display(self, duration=30):
        """分析并显示音调信息"""
        print(f"\n🎼 开始音调分析 (持续 {duration}秒)")
        print("=" * 80)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🎵 音调高低分析 - 时间: {current_time:.1f}秒")
            print("=" * 80)
            
            # 分析每个乐器
            for instrument in ["violin", "lute", "organ"]:
                pitch_data = self.analyze_audio_at_time(instrument, current_time)
                
                print(f"\n🎻 {instrument.upper()}:")
                print("-" * 60)
                
                for range_name, intensity in pitch_data.items():
                    freq_range = self.pitch_ranges[range_name]
                    bar = self.print_bar_chart(intensity)
                    print(f"{range_name:>12} ({freq_range[0]:>5}-{freq_range[1]:>5}Hz): {bar}")
            
            # 总结
            print("\n📊 音调高低总结:")
            print("-" * 60)
            
            for instrument in ["violin", "lute", "organ"]:
                pitch_data = self.analyze_audio_at_time(instrument, current_time)
                
                # 找出最强的音调范围
                max_range = max(pitch_data.items(), key=lambda x: x[1])
                
                # 分类高低音
                high_freq_total = sum(pitch_data[r] for r in ["High Mid", "Presence", "Brilliance"])
                low_freq_total = sum(pitch_data[r] for r in ["Sub Bass", "Bass", "Low Mid"])
                mid_freq_total = pitch_data["Mid"]
                
                if high_freq_total > low_freq_total and high_freq_total > mid_freq_total:
                    tone_type = "高音为主"
                elif low_freq_total > high_freq_total and low_freq_total > mid_freq_total:
                    tone_type = "低音为主"
                else:
                    tone_type = "中音为主"
                
                print(f"{instrument:>8}: {tone_type} | 最强: {max_range[0]} ({max_range[1]:.2f})")
            
            print(f"\n⏰ 分析时间: {current_time:.1f}/{duration}秒 | ESC退出")
            
            time.sleep(0.5)  # 更新频率
    
    def run(self):
        """主运行函数"""
        print("🎵 控制台音频音调分析器")
        print("=" * 60)
        
        # 检查文件
        if not self.check_files():
            print("\n❌ 缺少必要的MP3文件")
            return
        
        # 加载音频
        if not self.load_audio_files():
            print("\n❌ 音频文件加载失败")
            return
        
        print(f"\n🎯 功能说明:")
        print("此工具可以分析三个MP3文件的音调高低")
        print("显示每个乐器在不同频率范围的强度")
        print("帮助识别哪些部分是高音、低音")
        
        # 开始分析
        try:
            self.analyze_and_display(60)  # 分析60秒
        except KeyboardInterrupt:
            print("\n\n👋 用户停止分析")
        except Exception as e:
            print(f"\n❌ 分析错误: {e}")

def main():
    analyzer = ConsolePitchAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()