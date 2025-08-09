#!/usr/bin/env python3
"""
音频位置管理器 - 实现真正的断点续播
使用音频切片和同步播放来模拟位置控制
"""

import pygame
import threading
import time
import os
import tempfile
from typing import Dict, Set, Optional
from pydub import AudioSegment
from pydub.playback import play
import io

class AudioPositionManager:
    def __init__(self):
        """初始化音频位置管理器"""
        self.audio_files = {
            1: "Fugue in G Trio violin-Violin.mp3",      # 小提琴声部
            2: "Fugue in G Trio-Tenor_Lute.mp3",        # 鲁特琴声部  
            3: "Fugue in G Trio Organ-Organ.mp3"        # 管风琴声部
        }
        
        # 音频数据 - 使用pydub加载完整音频
        self.audio_segments: Dict[int, AudioSegment] = {}
        self.audio_lengths: Dict[int, float] = {}
        
        # pygame音频对象（用于播放切片）
        self.current_sounds: Dict[int, pygame.mixer.Sound] = {}
        self.audio_channels: Dict[int, pygame.mixer.Channel] = {}
        
        # 播放控制
        self.audio_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.target_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.playing_tracks: Set[int] = set()
        
        # 位置控制
        self.master_playing = False
        self.current_position = 0.0  # 当前播放位置（秒）
        self.play_start_time: Optional[float] = None
        self.pause_position = 0.0
        
        # 音量渐变控制
        self.volume_fade_speed = 0.15
        self.fade_thread_running = False
        
        # 切片播放控制
        self.slice_duration = 5.0  # 每个切片5秒
        self.playback_thread = None
        self.playback_thread_running = False
        
        self.enabled = False
        
        print("🎵 音频位置管理器初始化...")
    
    def initialize(self) -> bool:
        """初始化音频系统"""
        try:
            # 检查pydub是否可用
            try:
                from pydub import AudioSegment
                print("✅ Pydub可用")
            except ImportError:
                print("❌ 需要安装pydub: pip install pydub")
                # 降级到基础版本
                return self._init_basic_version()
            
            # 检查文件存在性
            missing_files = []
            for track_id, filename in self.audio_files.items():
                if not os.path.exists(filename):
                    missing_files.append(filename)
            
            if missing_files:
                print("⚠️ 部分音频文件缺失:")
                for file in missing_files:
                    print(f"   - {file}")
                return False
            
            # 加载音频文件为AudioSegment
            for track_id, filename in self.audio_files.items():
                try:
                    print(f"🔄 加载音轨{track_id}: {filename}")
                    segment = AudioSegment.from_mp3(filename)
                    
                    # 转换为pygame兼容的格式
                    segment = segment.set_channels(2).set_frame_rate(22050)
                    
                    length_seconds = len(segment) / 1000.0
                    print(f"  📏 音频长度: {length_seconds:.2f}秒")
                    
                    self.audio_segments[track_id] = segment
                    self.audio_lengths[track_id] = length_seconds
                    self.audio_channels[track_id] = pygame.mixer.Channel(track_id - 1)
                    
                    print(f"✅ 音轨{track_id}加载成功")
                except Exception as e:
                    print(f"❌ 音轨{track_id}加载失败: {e}")
                    continue
            
            if not self.audio_segments:
                print("❌ 没有任何音频文件加载成功")
                return False
            
            self.enabled = True
            self.start_fade_thread()
            self.start_playback_thread()
            
            print(f"✅ 音频位置管理器初始化完成，加载了 {len(self.audio_segments)} 个音轨")
            return True
            
        except Exception as e:
            print(f"❌ 音频位置管理器初始化失败: {e}")
            return self._init_basic_version()
    
    def _init_basic_version(self):
        """降级到基础版本（不支持精确位置控制）"""
        print("🔄 降级到基础音频管理器...")
        from advanced_audio_manager import AdvancedAudioManager
        self.basic_manager = AdvancedAudioManager()
        return self.basic_manager.initialize()
    
    def start_fade_thread(self):
        """启动音量渐变线程"""
        if self.fade_thread_running:
            return
        
        self.fade_thread_running = True
        fade_thread = threading.Thread(target=self._fade_loop, daemon=True)
        fade_thread.start()
    
    def start_playback_thread(self):
        """启动切片播放线程"""
        if self.playback_thread_running:
            return
        
        self.playback_thread_running = True
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        print("🔊 切片播放线程已启动")
    
    def _fade_loop(self):
        """音量渐变循环"""
        while self.fade_thread_running:
            try:
                for track_id in self.audio_segments.keys():
                    current_vol = self.audio_volumes[track_id]
                    target_vol = self.target_volumes[track_id]
                    
                    if abs(current_vol - target_vol) > 0.01:
                        volume_diff = target_vol - current_vol
                        new_vol = current_vol + volume_diff * self.volume_fade_speed
                        
                        self.audio_volumes[track_id] = new_vol
                        
                        # 更新当前播放声音的音量
                        if track_id in self.current_sounds:
                            self.current_sounds[track_id].set_volume(new_vol)
                
                time.sleep(1/60)
            except Exception as e:
                print(f"❌ 音量渐变线程错误: {e}")
                time.sleep(0.1)
    
    def _playback_loop(self):
        """切片播放循环 - 核心位置控制逻辑"""
        while self.playback_thread_running:
            try:
                if self.master_playing:
                    # 更新当前位置
                    if self.play_start_time:
                        elapsed = time.time() - self.play_start_time
                        self.current_position = self.pause_position + elapsed
                    
                    # 检查是否需要加载新的切片
                    self._update_audio_slices()
                
                time.sleep(0.1)  # 10fps检查频率，减少CPU占用
            except Exception as e:
                print(f"❌ 播放循环错误: {e}")
                time.sleep(1)
    
    def _update_audio_slices(self):
        """更新音频切片播放"""
        if not self.master_playing:
            return
        
        # 计算当前应该播放的切片
        current_slice_start = int(self.current_position // self.slice_duration) * self.slice_duration
        next_slice_needed = current_slice_start + self.slice_duration
        
        # 检查是否需要准备新的切片
        remaining_time = next_slice_needed - self.current_position
        
        if remaining_time < 1.0:  # 提前1秒准备下一个切片
            self._prepare_next_slice(next_slice_needed)
    
    def _prepare_next_slice(self, start_time: float):
        """准备下一个音频切片"""
        try:
            for track_id in self.audio_segments.keys():
                if track_id not in self.playing_tracks:
                    continue
                
                segment = self.audio_segments[track_id]
                
                # 提取切片（以毫秒为单位）
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + self.slice_duration) * 1000)
                
                # 检查是否超出音频长度
                if start_ms >= len(segment):
                    # 循环播放
                    start_ms = start_ms % len(segment)
                    end_ms = start_ms + int(self.slice_duration * 1000)
                
                slice_segment = segment[start_ms:min(end_ms, len(segment))]
                
                # 转换为pygame Sound对象
                # 导出为wav数据
                wav_io = io.BytesIO()
                slice_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                # 创建pygame Sound
                sound = pygame.mixer.Sound(wav_io)
                sound.set_volume(self.audio_volumes[track_id])
                
                # 停止旧的声音
                if track_id in self.current_sounds:
                    self.audio_channels[track_id].stop()
                
                # 播放新切片
                self.current_sounds[track_id] = sound
                self.audio_channels[track_id].play(sound)
                
        except Exception as e:
            print(f"❌ 准备切片失败: {e}")
    
    def start_playback_from_position(self, position: float = None):
        """从指定位置开始播放"""
        if not self.enabled:
            return
        
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.start_playback_from_position(position)
        
        if position is not None:
            self.pause_position = position
            self.current_position = position
        
        if not self.master_playing:
            print(f"🎼 从位置 {self.pause_position:.2f}秒 开始播放")
            
            # 准备第一个切片
            self._prepare_first_slice()
            
            self.master_playing = True
            self.play_start_time = time.time()
            
            print(f"✅ 位置播放已启动")
    
    def _prepare_first_slice(self):
        """准备第一个播放切片"""
        start_time = self.pause_position
        
        for track_id in self.audio_segments.keys():
            try:
                segment = self.audio_segments[track_id]
                
                # 提取从指定位置开始的切片
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + self.slice_duration) * 1000)
                
                if start_ms >= len(segment):
                    start_ms = start_ms % len(segment)
                    end_ms = start_ms + int(self.slice_duration * 1000)
                
                slice_segment = segment[start_ms:min(end_ms, len(segment))]
                
                # 转换为pygame Sound
                wav_io = io.BytesIO()
                slice_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                sound = pygame.mixer.Sound(wav_io)
                sound.set_volume(self.audio_volumes[track_id])
                
                self.current_sounds[track_id] = sound
                self.playing_tracks.add(track_id)
                
            except Exception as e:
                print(f"❌ 准备首个切片失败 (音轨{track_id}): {e}")
    
    def pause_playback(self):
        """暂停播放"""
        if not self.master_playing:
            return
        
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.pause_playback()
        
        # 记录当前位置
        if self.play_start_time:
            elapsed = time.time() - self.play_start_time
            self.pause_position = self.pause_position + elapsed
        
        print(f"⏸️ 暂停播放，位置: {self.pause_position:.2f}秒")
        
        # 停止所有播放
        for track_id in list(self.playing_tracks):
            if track_id in self.current_sounds:
                self.audio_channels[track_id].stop()
        
        self.playing_tracks.clear()
        self.current_sounds.clear()
        self.master_playing = False
        self.play_start_time = None
        
        print(f"✅ 已暂停在位置: {self.pause_position:.2f}秒")
    
    def update_from_gestures(self, digit_gestures: list):
        """根据手势更新播放"""
        if not self.enabled:
            return
        
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.update_from_gestures(digit_gestures)
        
        active_gestures = set(digit_gestures)
        has_any_gesture = len(active_gestures) > 0
        
        # 更新目标音量
        for track_id in self.audio_segments.keys():
            should_be_audible = track_id in active_gestures
            self.target_volumes[track_id] = 1.0 if should_be_audible else 0.0
        
        if has_any_gesture and not self.master_playing:
            self.start_playback_from_position()
        elif not has_any_gesture and self.master_playing:
            self.pause_playback()
    
    def get_status_info(self) -> dict:
        """获取状态信息"""
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.get_status_info()
        
        # 更新当前位置
        if self.master_playing and self.play_start_time:
            elapsed = time.time() - self.play_start_time
            current_pos = self.pause_position + elapsed
        else:
            current_pos = self.pause_position
        
        return {
            'enabled': self.enabled,
            'master_playing': self.master_playing,
            'current_position': current_pos,
            'pause_position': self.pause_position,
            'volumes': self.audio_volumes.copy(),
            'target_volumes': self.target_volumes.copy(),
            'playing_tracks': list(self.playing_tracks),
            'audio_lengths': self.audio_lengths.copy()
        }
    
    def reset_position(self):
        """重置播放位置"""
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.reset_position()
        
        was_playing = self.master_playing
        
        if was_playing:
            self.pause_playback()
            time.sleep(0.1)
        
        self.pause_position = 0.0
        self.current_position = 0.0
        print("🔄 播放位置已重置到开头")
        
        if was_playing:
            current_gestures = []
            for track_id, vol in self.target_volumes.items():
                if vol > 0.5:
                    current_gestures.append(track_id)
            
            if current_gestures:
                self.update_from_gestures(current_gestures)
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理音频位置管理器...")
        
        if hasattr(self, 'basic_manager'):
            return self.basic_manager.cleanup()
        
        self.fade_thread_running = False
        self.playback_thread_running = False
        
        if self.master_playing:
            self.pause_playback()
        
        self.enabled = False
        print("✅ 音频位置管理器已清理")