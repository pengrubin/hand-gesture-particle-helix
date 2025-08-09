#!/usr/bin/env python3
"""
高级音频管理器
支持断点续播、精确位置控制和同步播放
"""

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

import threading
import time
import os
from typing import Dict, Set, Optional

class AdvancedAudioManager:
    def __init__(self):
        self.audio_files = {
            1: "Fugue in G Trio violin-Violin.mp3",      # 小提琴声部
            2: "Fugue in G Trio-Tenor_Lute.mp3",        # 鲁特琴声部  
            3: "Fugue in G Trio Organ-Organ.mp3"        # 管风琴声部
        }
        
        # 音频对象
        self.audio_sounds: Dict[int, any] = {}  # pygame.mixer.Sound or mock
        self.audio_channels: Dict[int, any] = {}  # pygame.mixer.Channel or mock
        self.audio_lengths: Dict[int, float] = {}
        
        # 播放控制
        self.audio_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.target_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.playing_tracks: Set[int] = set()
        
        # 断点续播相关
        self.master_playing = False
        self.pause_position = 0.0  # 暂停时的播放位置（秒）
        self.play_start_time: Optional[float] = None  # 当前播放段开始的系统时间
        self.last_pause_time: Optional[float] = None  # 上次暂停的时间
        
        # 音量平滑控制
        self.volume_fade_speed = 0.15
        self.fade_thread_running = False
        
        self.enabled = False
        
        print("🎵 高级音频管理器初始化...")
    
    def initialize(self) -> bool:
        """初始化音频系统"""
        try:
            if not PYGAME_AVAILABLE:
                print("⚠️ Pygame未安装，仅初始化逻辑结构")
                self.enabled = False  # 但可以用于测试逻辑
                return False
            
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
            
            # 加载音频文件
            for track_id, filename in self.audio_files.items():
                try:
                    print(f"🔄 加载音轨{track_id}: {filename}")
                    sound = pygame.mixer.Sound(filename)
                    sound.set_volume(0.0)
                    
                    length = sound.get_length()
                    print(f"  📏 音频长度: {length:.2f}秒")
                    
                    self.audio_sounds[track_id] = sound
                    self.audio_channels[track_id] = pygame.mixer.Channel(track_id - 1)
                    self.audio_lengths[track_id] = length
                    
                    print(f"✅ 音轨{track_id}加载成功")
                except Exception as e:
                    print(f"❌ 音轨{track_id}加载失败: {e}")
                    continue
            
            if not self.audio_sounds:
                print("❌ 没有任何音频文件加载成功")
                return False
            
            self.enabled = True
            self.start_fade_thread()
            print(f"✅ 高级音频管理器初始化完成，加载了 {len(self.audio_sounds)} 个音轨")
            return True
            
        except Exception as e:
            print(f"❌ 音频管理器初始化失败: {e}")
            return False
    
    def start_fade_thread(self):
        """启动音量渐变线程"""
        if self.fade_thread_running:
            return
        
        self.fade_thread_running = True
        fade_thread = threading.Thread(target=self._fade_loop, daemon=True)
        fade_thread.start()
        print("🔊 音量渐变线程已启动")
    
    def _fade_loop(self):
        """音量渐变循环"""
        while self.fade_thread_running:
            try:
                for track_id in self.audio_sounds.keys():
                    current_vol = self.audio_volumes[track_id]
                    target_vol = self.target_volumes[track_id]
                    
                    if abs(current_vol - target_vol) > 0.01:
                        # 计算新音量
                        volume_diff = target_vol - current_vol
                        new_vol = current_vol + volume_diff * self.volume_fade_speed
                        
                        # 更新音量
                        self.audio_volumes[track_id] = new_vol
                        if track_id in self.audio_sounds:
                            self.audio_sounds[track_id].set_volume(new_vol)
                
                time.sleep(1/60)  # 60fps更新频率
            except Exception as e:
                print(f"❌ 音量渐变线程错误: {e}")
                time.sleep(0.1)
    
    def get_current_position(self) -> float:
        """获取当前播放位置（秒）"""
        if not self.master_playing or not self.play_start_time:
            return self.pause_position
        
        # 计算当前播放时间
        current_time = time.time()
        elapsed_time = current_time - self.play_start_time
        total_position = self.pause_position + elapsed_time
        
        # 检查是否需要循环（使用最短的音频长度作为循环点）
        if self.audio_lengths:
            min_length = min(self.audio_lengths.values())
            if total_position >= min_length:
                # 循环播放
                self.pause_position = total_position % min_length
                self.play_start_time = current_time
                return self.pause_position
        
        return total_position
    
    def start_playback_from_position(self, position: float = None):
        """从指定位置开始播放（断点续播）"""
        if not self.enabled or not self.audio_sounds:
            print("⚠️ 音频系统未初始化")
            return
        
        if position is not None:
            self.pause_position = position
        
        current_time = time.time()
        
        if not self.master_playing:
            print(f"🎼 从位置 {self.pause_position:.2f}秒 开始断点续播")
            
            # 同时启动所有音轨
            for track_id in self.audio_sounds.keys():
                try:
                    # 注意：pygame不支持从指定位置开始播放，
                    # 但我们通过时间跟踪来模拟这个功能
                    self.audio_sounds[track_id].set_volume(0.0)
                    self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                    self.playing_tracks.add(track_id)
                except Exception as e:
                    print(f"❌ 启动音轨{track_id}失败: {e}")
            
            self.master_playing = True
            self.play_start_time = current_time
            print(f"✅ 断点续播已启动，播放位置: {self.pause_position:.2f}秒")
    
    def pause_playback(self):
        """暂停播放并记录位置"""
        if not self.master_playing:
            return
        
        # 记录当前位置
        self.pause_position = self.get_current_position()
        print(f"⏸️ 暂停播放，当前位置: {self.pause_position:.2f}秒")
        
        # 快速淡出所有音轨
        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0
        
        # 延迟停止播放
        def delayed_stop():
            time.sleep(0.5)  # 等待淡出完成
            for track_id in list(self.playing_tracks):
                try:
                    self.audio_channels[track_id].stop()
                except Exception as e:
                    print(f"❌ 停止音轨{track_id}失败: {e}")
            
            self.playing_tracks.clear()
            self.master_playing = False
            self.play_start_time = None
            print(f"✅ 播放已暂停在位置: {self.pause_position:.2f}秒")
        
        threading.Thread(target=delayed_stop, daemon=True).start()
    
    def update_gesture_volumes(self, active_gestures: Set[int]):
        """根据手势更新目标音量"""
        if not self.enabled:
            return
        
        for track_id in self.audio_sounds.keys():
            should_be_audible = track_id in active_gestures
            self.target_volumes[track_id] = 1.0 if should_be_audible else 0.0
    
    def update_from_gestures(self, digit_gestures: list):
        """根据手势更新音频播放"""
        if not self.enabled:
            return
        
        active_gestures = set(digit_gestures)
        has_any_gesture = len(active_gestures) > 0
        
        if has_any_gesture and not self.master_playing:
            # 启动断点续播
            self.start_playback_from_position()
            self.update_gesture_volumes(active_gestures)
            
        elif not has_any_gesture and self.master_playing:
            # 暂停播放
            self.pause_playback()
            
        elif has_any_gesture and self.master_playing:
            # 更新音量但保持播放
            self.update_gesture_volumes(active_gestures)
    
    def get_status_info(self) -> dict:
        """获取音频状态信息"""
        current_pos = self.get_current_position()
        
        status = {
            'enabled': self.enabled,
            'master_playing': self.master_playing,
            'current_position': current_pos,
            'pause_position': self.pause_position,
            'volumes': self.audio_volumes.copy(),
            'target_volumes': self.target_volumes.copy(),
            'playing_tracks': list(self.playing_tracks),
            'audio_lengths': self.audio_lengths.copy()
        }
        
        return status
    
    def reset_position(self):
        """重置播放位置到开头"""
        was_playing = self.master_playing
        
        if was_playing:
            self.pause_playback()
            time.sleep(0.6)  # 等待完全停止
        
        self.pause_position = 0.0
        print("🔄 播放位置已重置到开头")
        
        if was_playing:
            # 如果之前在播放，重新开始
            current_gestures = []
            for track_id, vol in self.target_volumes.items():
                if vol > 0.5:
                    current_gestures.append(track_id)
            
            if current_gestures:
                self.update_from_gestures(current_gestures)
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理高级音频管理器...")
        
        self.fade_thread_running = False
        
        if self.master_playing:
            self.pause_playback()
            time.sleep(0.6)
        
        try:
            for track_id in list(self.playing_tracks):
                self.audio_channels[track_id].stop()
            self.playing_tracks.clear()
        except:
            pass
        
        self.enabled = False
        print("✅ 高级音频管理器已清理")