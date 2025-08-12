#!/usr/bin/env python3
"""
音频频谱分析器
实时分析音频频谱，提取音高和强度信息用于视觉效果控制
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioSpectrumAnalyzer:
    def __init__(self):
        """初始化音频频谱分析器"""
        
        # 音频参数
        self.sample_rate = 44100  # 采样率
        self.chunk_size = 4096    # 缓冲区大小
        self.channels = 2         # 立体声
        
        # 分析参数
        self.fft_size = 2048      # FFT大小
        self.hop_size = 512       # 跳跃大小
        self.overlap = self.fft_size - self.hop_size
        
        # 频率范围设置
        self.min_freq = 80.0      # 最低频率 (Hz)
        self.max_freq = 2000.0    # 最高频率 (Hz)
        self.num_bands = 32       # 频段数量
        
        # 分析结果存储
        self.spectrum_data = {
            'frequencies': np.array([]),
            'magnitudes': np.array([]),
            'dominant_freq': 0.0,     # 主导频率
            'pitch_strength': 0.0,    # 音高强度
            'overall_volume': 0.0,    # 总体音量
            'frequency_bands': np.zeros(self.num_bands),  # 分段频谱
            'pitch_class': 'C',       # 音高类别 (C, D, E, F, G, A, B)
            'octave': 4,              # 八度
            'note_confidence': 0.0,   # 音符识别置信度
            'timestamp': 0.0
        }
        
        # 音高到音符的映射
        self.note_frequencies = {
            'C':  [65.41, 130.81, 261.63, 523.25, 1046.50],
            'C#': [69.30, 138.59, 277.18, 554.37, 1108.73],
            'D':  [73.42, 146.83, 293.66, 587.33, 1174.66],
            'D#': [77.78, 155.56, 311.13, 622.25, 1244.51],
            'E':  [82.41, 164.81, 329.63, 659.25, 1318.51],
            'F':  [87.31, 174.61, 349.23, 698.46, 1396.91],
            'F#': [92.50, 185.00, 369.99, 739.99, 1479.98],
            'G':  [98.00, 196.00, 392.00, 783.99, 1567.98],
            'G#': [103.83, 207.65, 415.30, 830.61, 1661.22],
            'A':  [110.00, 220.00, 440.00, 880.00, 1760.00],
            'A#': [116.54, 233.08, 466.16, 932.33, 1864.66],
            'B':  [123.47, 246.94, 493.88, 987.77, 1975.53]
        }
        
        # 运行状态
        self.is_running = False
        self.analysis_thread = None
        
        # 音轨分析模式
        self.track_analysis_mode = True  # 是否分析音轨而不是麦克风
        self.audio_manager = None        # 音频管理器引用
        self.track_files = {
            1: "Fugue in G Trio violin-Violin.mp3",
            2: "Fugue in G Trio-Tenor_Lute.mp3", 
            3: "Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 音频输入
        self.audio_stream = None
        self.audio_buffer = np.zeros(self.chunk_size * 2)  # 双缓冲
        self.buffer_lock = threading.Lock()
        
        # 平滑处理
        self.smoothing_factor = 0.7
        self.prev_spectrum = np.zeros(self.num_bands)
        self.prev_pitch_strength = 0.0
        
        print("🎼 音频频谱分析器初始化完成")
    
    def set_audio_manager(self, audio_manager):
        """设置音频管理器引用"""
        self.audio_manager = audio_manager
        self.track_analysis_mode = True
        print("✅ 音频分析器切换到音轨分析模式")
    
    def initialize(self) -> bool:
        """初始化音频输入"""
        try:
            if not PYAUDIO_AVAILABLE:
                print("⚠️ PyAudio不可用，使用模拟分析器")
                return self._initialize_mock_analyzer()
            
            import pyaudio
            
            # 初始化PyAudio
            self.p = pyaudio.PyAudio()
            
            # 打开音频流
            self.audio_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            print("✅ 音频输入流已打开")
            return True
            
        except Exception as e:
            print(f"❌ 音频初始化失败: {e}")
            return self._initialize_mock_analyzer()
    
    def _initialize_mock_analyzer(self) -> bool:
        """初始化模拟分析器（用于测试）"""
        print("🔄 启用模拟音频分析器")
        self.mock_mode = True
        self.mock_time_start = time.time()
        return True
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        try:
            # 转换音频数据
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # 如果是立体声，取平均值转为单声道
            if self.channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # 更新缓冲区
            with self.buffer_lock:
                self.audio_buffer[:-len(audio_data)] = self.audio_buffer[len(audio_data):]
                self.audio_buffer[-len(audio_data):] = audio_data
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            print(f"❌ 音频回调错误: {e}")
            return (None, pyaudio.paComplete)
    
    def start_analysis(self):
        """开始频谱分析"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动音频流
        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.start_stream()
        
        # 启动分析线程
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        print("🎵 频谱分析已启动")
    
    def _analysis_loop(self):
        """分析循环"""
        window = np.hanning(self.fft_size)  # 汉宁窗
        
        while self.is_running:
            try:
                if hasattr(self, 'mock_mode') and self.mock_mode:
                    # 模拟模式：生成测试数据
                    self._generate_mock_spectrum()
                else:
                    # 真实模式：分析音频数据
                    self._analyze_real_audio(window)
                
                time.sleep(1/60)  # 60 FPS更新频率
                
            except Exception as e:
                print(f"❌ 分析循环错误: {e}")
                time.sleep(0.1)
    
    def _generate_mock_spectrum(self):
        """生成模拟频谱数据（基于正在播放的音轨）"""
        current_time = time.time() - self.mock_time_start
        
        # 检查哪些音轨正在播放
        active_tracks = []
        if self.audio_manager and hasattr(self.audio_manager, 'target_volumes'):
            for track_id, volume in self.audio_manager.target_volumes.items():
                if volume > 0.1:  # 音量大于0.1认为是激活的
                    active_tracks.append(track_id)
        
        if not active_tracks:
            # 没有激活音轨时使用静默
            freq1, freq2, freq3 = 220, 440, 880
            volume = 0.05
        else:
            # 根据激活的音轨生成相应的频谱特征
            freq1, freq2, freq3, volume = self._get_track_frequencies(active_tracks, current_time)
        
        # 生成频谱数据
        freqs = np.linspace(self.min_freq, self.max_freq, self.num_bands)
        spectrum = np.zeros(self.num_bands)
        
        # 在主要频率附近添加能量
        for freq in [freq1, freq2, freq3]:
            if self.min_freq <= freq <= self.max_freq:
                idx = int((freq - self.min_freq) / (self.max_freq - self.min_freq) * self.num_bands)
                if 0 <= idx < self.num_bands:
                    spectrum[idx] += volume
        
        # 添加一些噪音
        spectrum += np.random.normal(0, 0.01, self.num_bands)
        spectrum = np.maximum(0, spectrum)
        
        # 平滑处理
        spectrum = self.prev_spectrum * self.smoothing_factor + spectrum * (1 - self.smoothing_factor)
        self.prev_spectrum = spectrum
        
        # 更新分析结果
        dominant_freq = freq1  # 使用第一个频率作为主导频率
        pitch_strength = volume
        pitch_class, octave, confidence = self._frequency_to_note(dominant_freq)
        
        self.spectrum_data.update({
            'frequencies': freqs,
            'magnitudes': spectrum,
            'dominant_freq': dominant_freq,
            'pitch_strength': pitch_strength,
            'overall_volume': np.sum(spectrum),
            'frequency_bands': spectrum,
            'pitch_class': pitch_class,
            'octave': octave,
            'note_confidence': confidence,
            'timestamp': time.time()
        })
    
    def _analyze_real_audio(self, window):
        """分析真实音频数据"""
        with self.buffer_lock:
            audio_chunk = self.audio_buffer[-self.fft_size:].copy()
        
        # 应用窗函数
        windowed_chunk = audio_chunk * window
        
        # 执行FFT
        fft_result = np.fft.rfft(windowed_chunk)
        magnitude_spectrum = np.abs(fft_result)
        
        # 计算频率轴
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        
        # 提取指定频率范围
        freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        valid_freqs = freqs[freq_mask]
        valid_magnitudes = magnitude_spectrum[freq_mask]
        
        # 分段分析
        frequency_bands = self._compute_frequency_bands(valid_freqs, valid_magnitudes)
        
        # 寻找主导频率
        if len(valid_magnitudes) > 0:
            peak_idx = np.argmax(valid_magnitudes)
            dominant_freq = valid_freqs[peak_idx]
            pitch_strength = valid_magnitudes[peak_idx] / np.max(valid_magnitudes)
        else:
            dominant_freq = 0.0
            pitch_strength = 0.0
        
        # 平滑处理
        pitch_strength = self.prev_pitch_strength * self.smoothing_factor + pitch_strength * (1 - self.smoothing_factor)
        self.prev_pitch_strength = pitch_strength
        
        frequency_bands = self.prev_spectrum * self.smoothing_factor + frequency_bands * (1 - self.smoothing_factor)
        self.prev_spectrum = frequency_bands
        
        # 音符识别
        pitch_class, octave, confidence = self._frequency_to_note(dominant_freq)
        
        # 更新结果
        self.spectrum_data.update({
            'frequencies': valid_freqs,
            'magnitudes': valid_magnitudes,
            'dominant_freq': dominant_freq,
            'pitch_strength': pitch_strength,
            'overall_volume': np.sum(valid_magnitudes),
            'frequency_bands': frequency_bands,
            'pitch_class': pitch_class,
            'octave': octave,
            'note_confidence': confidence,
            'timestamp': time.time()
        })
    
    def _compute_frequency_bands(self, freqs: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """计算频段能量"""
        bands = np.zeros(self.num_bands)
        
        if len(freqs) == 0:
            return bands
        
        # 对数分布的频段边界
        freq_min = self.min_freq
        freq_max = self.max_freq
        
        for i in range(self.num_bands):
            # 计算每个频段的边界
            band_start = freq_min * (freq_max / freq_min) ** (i / self.num_bands)
            band_end = freq_min * (freq_max / freq_min) ** ((i + 1) / self.num_bands)
            
            # 找到频段内的频率
            band_mask = (freqs >= band_start) & (freqs < band_end)
            if np.any(band_mask):
                bands[i] = np.mean(magnitudes[band_mask])
        
        return bands
    
    def _frequency_to_note(self, frequency: float) -> Tuple[str, int, float]:
        """将频率转换为音符"""
        if frequency < 20:  # 太低的频率
            return 'C', 4, 0.0
        
        # 计算与A4 (440Hz)的半音数差异
        A4_freq = 440.0
        semitones_from_A4 = 12 * math.log2(frequency / A4_freq)
        
        # 四舍五入到最近的半音
        nearest_semitone = round(semitones_from_A4)
        
        # 计算置信度（频率与标准频率的接近程度）
        standard_freq = A4_freq * (2 ** (nearest_semitone / 12))
        confidence = max(0, 1 - abs(frequency - standard_freq) / standard_freq)
        
        # 转换为音符名称和八度
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        note_index = nearest_semitone % 12
        note_name = note_names[note_index]
        
        # 计算八度
        octave = 4 + (nearest_semitone + 9) // 12  # +9 因为从A开始计算
        
        return note_name, octave, confidence
    
    def _get_track_frequencies(self, active_tracks, current_time):
        """根据激活的音轨生成相应的频率特征"""
        # 不同音轨的特征频率范围
        track_characteristics = {
            1: {  # 小提琴 - 高音域
                'base_freq': 440,  # A4
                'freq_range': 150,
                'harmonics': [1, 2, 3, 5],  # 丰富的泛音
                'vibrato_rate': 0.8,
                'volume_factor': 0.6
            },
            2: {  # 鲁特琴 - 中音域
                'base_freq': 330,  # E4
                'freq_range': 80, 
                'harmonics': [1, 1.5, 2, 3],  # 弦乐器特征泛音
                'vibrato_rate': 0.5,
                'volume_factor': 0.5
            },
            3: {  # 管风琴 - 低音域
                'base_freq': 220,  # A3
                'freq_range': 60,
                'harmonics': [1, 2, 4, 8],  # 管风琴特征泛音
                'vibrato_rate': 0.3,
                'volume_factor': 0.8
            }
        }
        
        # 合成激活音轨的频率特征
        freq1, freq2, freq3 = 0, 0, 0
        total_volume = 0
        
        for track_id in active_tracks:
            if track_id in track_characteristics:
                char = track_characteristics[track_id]
                track_volume = self.audio_manager.target_volumes.get(track_id, 0.0)
                
                # 为每个音轨生成主频率
                base_freq = char['base_freq']
                vibrato = char['freq_range'] * 0.3 * math.sin(current_time * char['vibrato_rate'])
                track_freq = base_freq + vibrato
                
                # 加权合成到总频率
                weight = track_volume * char['volume_factor']
                freq1 += track_freq * weight
                freq2 += track_freq * char['harmonics'][1] * weight * 0.7
                freq3 += track_freq * char['harmonics'][2] * weight * 0.5
                
                total_volume += track_volume * char['volume_factor']
        
        # 归一化
        if total_volume > 0:
            freq1 /= len(active_tracks)
            freq2 /= len(active_tracks) 
            freq3 /= len(active_tracks)
        
        # 添加音乐性的时间变化
        freq1 += 20 * math.sin(current_time * 0.4)
        freq2 += 15 * math.cos(current_time * 0.6)
        freq3 += 10 * math.sin(current_time * 0.8)
        
        # 音量变化
        volume = 0.2 + total_volume * 0.6 + 0.1 * math.sin(current_time * 1.5)
        
        return freq1, freq2, freq3, min(volume, 1.0)
    
    def get_pitch_intensity(self) -> float:
        """获取音高强度（用于控制粒子大小）"""
        # 结合主导频率强度和总体音量
        pitch_factor = self.spectrum_data['pitch_strength']
        volume_factor = self.spectrum_data['overall_volume'] / 100.0  # 归一化
        
        # 加权组合
        intensity = pitch_factor * 0.7 + volume_factor * 0.3
        
        # 限制在合理范围内
        return max(0.0, min(2.0, intensity))
    
    def get_frequency_bands_normalized(self) -> np.ndarray:
        """获取归一化的频段数据"""
        bands = self.spectrum_data['frequency_bands']
        if np.max(bands) > 0:
            return bands / np.max(bands)
        return bands
    
    def get_status_info(self) -> dict:
        """获取分析器状态信息"""
        return {
            'is_running': self.is_running,
            'dominant_freq': self.spectrum_data['dominant_freq'],
            'pitch_strength': self.spectrum_data['pitch_strength'],
            'overall_volume': self.spectrum_data['overall_volume'],
            'pitch_class': self.spectrum_data['pitch_class'],
            'octave': self.spectrum_data['octave'],
            'note_confidence': self.spectrum_data['note_confidence'],
            'pitch_intensity': self.get_pitch_intensity(),
            'timestamp': self.spectrum_data['timestamp']
        }
    
    def stop_analysis(self):
        """停止频谱分析"""
        self.is_running = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        
        print("🎵 频谱分析已停止")
    
    def cleanup(self):
        """清理资源"""
        self.stop_analysis()
        print("✅ 音频分析器已清理")