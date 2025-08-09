#!/usr/bin/env python3
"""
现实主义音频管理器
承认pygame的限制，但提供最佳可能的用户体验
"""

import pygame
import threading
import time
import os
from typing import Dict, Set, Optional

class RealisticAudioManager:
    def __init__(self):
        """初始化现实主义音频管理器"""
        self.audio_files = {
            1: "Fugue in G Trio violin-Violin.mp3",      # 小提琴声部
            2: "Fugue in G Trio-Tenor_Lute.mp3",        # 鲁特琴声部  
            3: "Fugue in G Trio Organ-Organ.mp3"        # 管风琴声部
        }
        
        # 音频对象
        self.audio_sounds: Dict[int, pygame.mixer.Sound] = {}
        self.audio_channels: Dict[int, pygame.mixer.Channel] = {}
        self.audio_lengths: Dict[int, float] = {}
        
        # 播放控制
        self.audio_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.target_volumes: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.playing_tracks: Set[int] = set()
        
        # 现实的位置跟踪
        self.master_playing = False
        self.session_start_time: Optional[float] = None  # 整个播放会话的开始时间
        self.total_pause_duration = 0.0  # 总暂停时间
        self.current_pause_start: Optional[float] = None  # 当前暂停开始时间
        
        # 用户体验设置
        self.continuous_playback_mode = True  # 连续播放模式
        self.restart_on_long_pause = True    # 长暂停时重启
        self.long_pause_threshold = 15.0     # 长暂停阈值（秒）
        
        # 音量控制
        self.volume_fade_speed = 0.25
        self.fade_thread_running = False
        
        # 状态跟踪
        self.playback_state = "stopped"  # stopped, playing, paused
        self.last_gesture_time = 0.0
        self.gesture_timeout = 0.5  # 手势超时时间
        
        self.enabled = False
        
        print("🎵 现实主义音频管理器初始化...")
    
    def initialize(self) -> bool:
        """初始化音频系统"""
        try:
            # 检查文件存在性
            missing_files = []
            for track_id, filename in self.audio_files.items():
                if not os.path.exists(filename):
                    missing_files.append(filename)
            
            if missing_files:
                print("⚠️ 缺失音频文件:")
                for file in missing_files:
                    print(f"   - {file}")
                return False
            
            # 加载音频文件
            for track_id, filename in self.audio_files.items():
                try:
                    print(f"加载音轨 {track_id}: {filename}")
                    sound = pygame.mixer.Sound(filename)
                    sound.set_volume(0.0)
                    
                    length = sound.get_length()
                    print(f"  时长: {length:.1f}秒")
                    
                    self.audio_sounds[track_id] = sound
                    self.audio_channels[track_id] = pygame.mixer.Channel(track_id - 1)
                    self.audio_lengths[track_id] = length
                    
                    print(f"✅ 音轨 {track_id} 加载成功")
                except Exception as e:
                    print(f"❌ 音轨 {track_id} 加载失败: {e}")
                    continue
            
            if not self.audio_sounds:
                print("❌ 没有音频文件加载成功")
                return False
            
            self.enabled = True
            self.start_fade_thread()
            print(f"✅ 音频管理器就绪，已加载 {len(self.audio_sounds)} 个音轨")
            
            # 如果启用连续播放，立即开始播放
            if self.continuous_playback_mode:
                self._start_continuous_playback()
            
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def _start_continuous_playback(self):
        """开始连续播放模式"""
        print("🔄 启动连续播放模式")
        self.session_start_time = time.time()
        self.master_playing = True
        self.playback_state = "playing"
        
        # 启动所有音轨（静音状态）
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ 启动音轨 {track_id} 失败: {e}")
        
        print("✅ 连续播放已开始，通过手势控制音量")
    
    def start_fade_thread(self):
        """启动音量渐变线程"""
        if self.fade_thread_running:
            return
        
        self.fade_thread_running = True
        fade_thread = threading.Thread(target=self._fade_loop, daemon=True)
        fade_thread.start()
    
    def _fade_loop(self):
        """音量渐变循环"""
        while self.fade_thread_running:
            try:
                for track_id in self.audio_sounds.keys():
                    current_vol = self.audio_volumes[track_id]
                    target_vol = self.target_volumes[track_id]
                    
                    if abs(current_vol - target_vol) > 0.01:
                        volume_diff = target_vol - current_vol
                        new_vol = current_vol + volume_diff * self.volume_fade_speed
                        
                        self.audio_volumes[track_id] = new_vol
                        if track_id in self.audio_sounds:
                            self.audio_sounds[track_id].set_volume(new_vol)
                
                time.sleep(1/30)
            except Exception as e:
                time.sleep(0.1)
    
    def get_current_position(self) -> float:
        """获取当前播放位置（考虑暂停时间）"""
        if not self.session_start_time:
            return 0.0
        
        current_time = time.time()
        
        # 计算总的实际播放时间
        elapsed_since_session = current_time - self.session_start_time
        actual_play_time = elapsed_since_session - self.total_pause_duration
        
        # 如果当前正在暂停，还要减去当前暂停的时间
        if self.playback_state == "paused" and self.current_pause_start:
            current_pause_time = current_time - self.current_pause_start
            actual_play_time -= current_pause_time
        
        # 循环播放检查
        if self.audio_lengths:
            min_length = min(self.audio_lengths.values())
            if actual_play_time >= min_length:
                actual_play_time = actual_play_time % min_length
        
        return max(0.0, actual_play_time)
    
    def _handle_gesture_change(self, active_gestures: Set[int]):
        """处理手势变化"""
        current_time = time.time()
        self.last_gesture_time = current_time
        
        has_gestures = len(active_gestures) > 0
        
        if has_gestures and self.playback_state == "paused":
            # 从暂停恢复
            self._resume_from_pause()
        elif not has_gestures and self.playback_state == "playing":
            # 进入暂停
            self._enter_pause_state()
        
        # 更新音量目标
        for track_id in self.audio_sounds.keys():
            should_be_audible = track_id in active_gestures
            self.target_volumes[track_id] = 1.0 if should_be_audible else 0.0
    
    def _resume_from_pause(self):
        """从暂停状态恢复"""
        if self.playback_state != "paused" or not self.current_pause_start:
            return
        
        current_time = time.time()
        pause_duration = current_time - self.current_pause_start
        
        # 累计暂停时间
        self.total_pause_duration += pause_duration
        self.current_pause_start = None
        self.playback_state = "playing"
        
        print(f"▶️ 从暂停恢复 (暂停了 {pause_duration:.1f}秒)")
        
        # 检查是否需要重启播放
        if pause_duration > self.long_pause_threshold and self.restart_on_long_pause:
            self._restart_playback()
        else:
            # 简单恢复，音轨应该还在播放
            if not self.master_playing:
                # 如果音轨已停止，重新启动
                self._restart_audio_tracks()
    
    def _enter_pause_state(self):
        """进入暂停状态"""
        if self.playback_state == "paused":
            return
        
        current_time = time.time()
        self.current_pause_start = current_time
        self.playback_state = "paused"
        
        current_pos = self.get_current_position()
        print(f"⏸️ 进入暂停状态，当前位置: {current_pos:.1f}秒")
        
        # 音量渐变到0，但不停止播放
        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0
    
    def _restart_playback(self):
        """重启播放"""
        print("🔄 长时间暂停，重启播放")
        
        # 重置时间跟踪
        self.session_start_time = time.time()
        self.total_pause_duration = 0.0
        self.current_pause_start = None
        
        # 重新启动所有音轨
        self._restart_audio_tracks()
    
    def _restart_audio_tracks(self):
        """重新启动音频轨道"""
        # 停止所有当前播放
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass
        
        self.playing_tracks.clear()
        
        # 重新启动
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ 重启音轨 {track_id} 失败: {e}")
        
        self.master_playing = True
        print("✅ 音轨重启完成")
    
    def update_from_gestures(self, digit_gestures: list):
        """根据手势更新音频"""
        if not self.enabled:
            return
        
        active_gestures = set(digit_gestures)
        
        # 初始化播放（如果还未开始）
        if not self.session_start_time:
            if active_gestures:  # 只有在有手势时才开始
                self._start_continuous_playback()
        
        # 处理手势变化
        self._handle_gesture_change(active_gestures)
        
        # 定期输出状态（降低频率）
        current_time = time.time()
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0
        
        if current_time - self._last_status_time > 2.0:  # 每2秒输出一次
            pos = self.get_current_position()
            state = self.playback_state.upper()
            print(f"🎵 音频状态: {state}, 位置: {pos:.1f}秒, 手势: {digit_gestures}")
            self._last_status_time = current_time
    
    def manual_pause_resume(self):
        """手动暂停/恢复"""
        if self.playback_state == "playing":
            self._enter_pause_state()
            print("⏸️ 手动暂停")
        elif self.playback_state == "paused":
            self._resume_from_pause()
            print("▶️ 手动恢复")
        else:
            # 如果停止状态，开始播放
            self._start_continuous_playback()
            print("▶️ 开始播放")
    
    def reset_position(self):
        """重置播放位置"""
        print("🔄 重置播放位置")
        self._restart_playback()
        
        # 清除所有音量，等待手势
        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0
    
    def get_status_info(self) -> dict:
        """获取状态信息"""
        current_pos = self.get_current_position()
        
        return {
            'enabled': self.enabled,
            'master_playing': self.master_playing,
            'current_position': current_pos,
            'pause_position': current_pos,  # 兼容性
            'volumes': self.audio_volumes.copy(),
            'target_volumes': self.target_volumes.copy(),
            'playing_tracks': list(self.playing_tracks),
            'audio_lengths': self.audio_lengths.copy(),
            'playback_state': self.playback_state,
            'total_pause_duration': self.total_pause_duration
        }
    
    def set_continuous_mode(self, enabled: bool):
        """设置连续播放模式"""
        self.continuous_playback_mode = enabled
        mode = "连续播放" if enabled else "手势启动"
        print(f"播放模式: {mode}")
    
    def set_restart_on_long_pause(self, enabled: bool):
        """设置长暂停时是否重启"""
        self.restart_on_long_pause = enabled
        mode = "重启" if enabled else "继续"
        print(f"长暂停策略: {mode}")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理音频管理器...")
        
        self.fade_thread_running = False
        
        # 停止所有播放
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass
        
        self.playing_tracks.clear()
        self.master_playing = False
        self.enabled = False
        
        print("✅ 音频管理器已清理")