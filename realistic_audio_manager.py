#!/usr/bin/env python3
"""
统一音频管理器 (Unified Audio Manager)
合并版本 - 包含以下功能：
- 现实主义播放控制（来自 realistic_audio_manager.py）
- TouchDesigner 接口（来自 audio_manager.py）
- 高级断点续播（来自 advanced_audio_manager.py）

Author: Refactored from multiple sources
Date: 2025-03-26
"""

import pygame
import threading
import time
import os
from pathlib import Path
from typing import Dict, Set, Optional, Any, List


class RealisticAudioManager:
    """
    统一音频管理器

    功能：
    - 多轨音频同时播放控制
    - 手势驱动的音量控制
    - 连续播放模式
    - 断点续播支持
    - TouchDesigner 参数输出
    """

    def __init__(self, audio_directory: Optional[str] = None):
        """
        初始化统一音频管理器

        Args:
            audio_directory: 音频文件所在目录，默认为当前脚本目录
        """
        # 设置音频文件路径
        if audio_directory is None:
            self.audio_directory = Path(__file__).parent
        else:
            self.audio_directory = Path(audio_directory)

        # 定义音频文件映射
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
        self.active_tracks: Set[int] = set()

        # 位置跟踪
        self.master_playing = False
        self.session_start_time: Optional[float] = None
        self.total_pause_duration = 0.0
        self.current_pause_start: Optional[float] = None
        self.pause_position = 0.0
        self.play_start_time: Optional[float] = None

        # 用户体验设置
        self.continuous_playback_mode = True
        self.restart_on_long_pause = True
        self.long_pause_threshold = 15.0

        # 音量控制
        self.volume_fade_speed = 0.25
        self.fade_thread_running = False
        self.fade_duration = 0.5

        # 状态跟踪
        self.playback_state = "stopped"
        self.last_gesture_time = 0.0
        self.gesture_timeout = 0.5

        # TouchDesigner 参数
        self.td_audio_params = {
            'track1_volume': 0.0,
            'track2_volume': 0.0,
            'track3_volume': 0.0,
            'track1_file': str(self.audio_directory / self.audio_files.get(1, "")),
            'track2_file': str(self.audio_directory / self.audio_files.get(2, "")),
            'track3_file': str(self.audio_directory / self.audio_files.get(3, "")),
            'master_play': False,
            'last_update_time': time.time()
        }

        self.enabled = False

        print("🎵 统一音频管理器初始化...")

    def initialize(self) -> bool:
        """初始化音频系统"""
        try:
            # 检查文件存在性
            missing_files = []
            for track_id, filename in self.audio_files.items():
                file_path = self.audio_directory / filename
                if not file_path.exists():
                    # 尝试直接路径
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
                    # 尝试完整路径
                    file_path = self.audio_directory / filename
                    if not file_path.exists():
                        file_path = Path(filename)

                    print(f"加载音轨 {track_id}: {filename}")
                    sound = pygame.mixer.Sound(str(file_path))
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
        self.play_start_time = time.time()
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

        self.td_audio_params['master_play'] = True
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

                        # 更新 TouchDesigner 参数
                        self.td_audio_params[f'track{track_id}_volume'] = new_vol

                self.td_audio_params['last_update_time'] = time.time()
                time.sleep(1/30)
            except Exception as e:
                time.sleep(0.1)

    def get_current_position(self) -> float:
        """获取当前播放位置（考虑暂停时间）"""
        if not self.session_start_time:
            return self.pause_position

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
            self._resume_from_pause()
        elif not has_gestures and self.playback_state == "playing":
            self._enter_pause_state()

        # 更新音量目标
        for track_id in self.audio_sounds.keys():
            should_be_audible = track_id in active_gestures
            self.target_volumes[track_id] = 1.0 if should_be_audible else 0.0

        self.active_tracks = active_gestures

    def _resume_from_pause(self):
        """从暂停状态恢复"""
        if self.playback_state != "paused" or not self.current_pause_start:
            return

        current_time = time.time()
        pause_duration = current_time - self.current_pause_start

        self.total_pause_duration += pause_duration
        self.current_pause_start = None
        self.playback_state = "playing"

        print(f"▶️ 从暂停恢复 (暂停了 {pause_duration:.1f}秒)")

        if pause_duration > self.long_pause_threshold and self.restart_on_long_pause:
            self._restart_playback()
        else:
            if not self.master_playing:
                self._restart_audio_tracks()

    def _enter_pause_state(self):
        """进入暂停状态"""
        if self.playback_state == "paused":
            return

        current_time = time.time()
        self.current_pause_start = current_time
        self.playback_state = "paused"

        # 记录当前位置
        self.pause_position = self.get_current_position()

        current_pos = self.pause_position
        print(f"⏸️ 进入暂停状态，当前位置: {current_pos:.1f}秒")

        # 音量渐变到0，但不停止播放
        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0

    def _restart_playback(self):
        """重启播放"""
        print("🔄 长时间暂停，重启播放")

        self.session_start_time = time.time()
        self.play_start_time = time.time()
        self.total_pause_duration = 0.0
        self.current_pause_start = None
        self.pause_position = 0.0

        self._restart_audio_tracks()

    def _restart_audio_tracks(self):
        """重新启动音频轨道"""
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()

        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ 重启音轨 {track_id} 失败: {e}")

        self.master_playing = True
        self.td_audio_params['master_play'] = True
        print("✅ 音轨重启完成")

    def start_playback_from_position(self, position: float = None):
        """从指定位置开始播放（断点续播）"""
        if not self.enabled or not self.audio_sounds:
            print("⚠️ 音频系统未初始化")
            return

        if position is not None:
            self.pause_position = position

        if not self.master_playing:
            print(f"🎼 从位置 {self.pause_position:.2f}秒 开始断点续播")

            for track_id in self.audio_sounds.keys():
                try:
                    self.audio_sounds[track_id].set_volume(0.0)
                    self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                    self.playing_tracks.add(track_id)
                except Exception as e:
                    print(f"❌ 启动音轨{track_id}失败: {e}")

            self.master_playing = True
            self.play_start_time = time.time()
            self.session_start_time = self.play_start_time
            self.playback_state = "playing"
            self.td_audio_params['master_play'] = True
            print(f"✅ 断点续播已启动")

    def pause_playback(self):
        """暂停播放并记录位置"""
        if not self.master_playing:
            return

        self.pause_position = self.get_current_position()
        print(f"⏸️ 暂停播放，当前位置: {self.pause_position:.2f}秒")

        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0

        def delayed_stop():
            time.sleep(0.5)
            for track_id in list(self.playing_tracks):
                try:
                    self.audio_channels[track_id].stop()
                except Exception as e:
                    pass

            self.playing_tracks.clear()
            self.master_playing = False
            self.play_start_time = None
            self.playback_state = "paused"
            self.td_audio_params['master_play'] = False

        threading.Thread(target=delayed_stop, daemon=True).start()

    def update_from_gestures(self, digit_gestures: List[int]):
        """根据手势更新音频"""
        if not self.enabled:
            return

        active_gestures = set(digit_gestures)

        # 初始化播放（如果还未开始）
        if not self.session_start_time:
            if active_gestures:
                self._start_continuous_playback()

        # 处理手势变化
        self._handle_gesture_change(active_gestures)

        # 定期输出状态
        current_time = time.time()
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0

        if current_time - self._last_status_time > 2.0:
            pos = self.get_current_position()
            state = self.playback_state.upper()
            print(f"🎵 音频状态: {state}, 位置: {pos:.1f}秒, 手势: {digit_gestures}")
            self._last_status_time = current_time

    def update_gesture_input(self, gesture_data: Dict[str, Any]):
        """
        根据手势数据更新音频控制（TouchDesigner 兼容接口）

        Args:
            gesture_data: 从手势检测器获得的数据
        """
        if not self.enabled:
            return

        active_gestures = set()
        if 'digit_gestures' in gesture_data:
            active_gestures = set(gesture_data['digit_gestures'])

        # 更新目标音量
        for track_id in [1, 2, 3]:
            if track_id in self.audio_sounds:
                should_play = track_id in active_gestures
                self.target_volumes[track_id] = 1.0 if should_play else 0.0

        self.active_tracks = active_gestures

        # 更新播放状态
        should_play = len(active_gestures) > 0 or self.master_playing
        if should_play != self.master_playing and len(active_gestures) > 0:
            if not self.master_playing:
                self.start_playback_from_position()

        self.td_audio_params['master_play'] = self.master_playing

    def manual_pause_resume(self):
        """手动暂停/恢复"""
        if self.playback_state == "playing":
            self._enter_pause_state()
            print("⏸️ 手动暂停")
        elif self.playback_state == "paused":
            self._resume_from_pause()
            print("▶️ 手动恢复")
        else:
            self._start_continuous_playback()
            print("▶️ 开始播放")

    def reset_position(self):
        """重置播放位置"""
        print("🔄 重置播放位置")
        self._restart_playback()

        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0

    def get_touchdesigner_params(self) -> Dict[str, Any]:
        """获取 TouchDesigner 所需的音频参数"""
        return self.td_audio_params.copy()

    def get_audio_state(self) -> Dict[str, Any]:
        """获取当前音频状态"""
        return {
            'initialized': self.enabled,
            'playing': self.master_playing,
            'volumes': self.audio_volumes.copy(),
            'active_tracks': list(self.active_tracks),
            'audio_files': self.audio_files.copy()
        }

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        current_pos = self.get_current_position()

        return {
            'enabled': self.enabled,
            'master_playing': self.master_playing,
            'current_position': current_pos,
            'pause_position': self.pause_position,
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

    def set_master_volume(self, volume: float):
        """设置主音量 (0.0-1.0)"""
        volume = max(0.0, min(1.0, volume))
        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = self.target_volumes.get(track_id, 0.0) * volume

    def stop_all_audio(self):
        """停止所有音频播放"""
        self.master_playing = False
        self.active_tracks = set()

        for track_id in self.audio_sounds.keys():
            self.target_volumes[track_id] = 0.0

        self.td_audio_params['master_play'] = False
        self.playback_state = "stopped"

        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()

    def cleanup(self):
        """清理资源"""
        print("🧹 清理音频管理器...")

        self.fade_thread_running = False

        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()
        self.master_playing = False
        self.enabled = False

        print("✅ 音频管理器已清理")


# ============================================================================
# 兼容性别名
# ============================================================================

# 保持与原有代码的兼容
AudioManager = RealisticAudioManager
AdvancedAudioManager = RealisticAudioManager


# ============================================================================
# TouchDesigner 接口函数
# ============================================================================

def initialize_audio_manager(audio_directory=None):
    """初始化音频管理器"""
    if 'op' in dir():
        if not hasattr(op, 'audio_manager'):
            op.audio_manager = RealisticAudioManager(audio_directory)
        return op.audio_manager.enabled
    return False


def update_audio_from_gesture(gesture_data):
    """根据手势数据更新音频"""
    if 'op' in dir():
        if not hasattr(op, 'audio_manager'):
            initialize_audio_manager()

        op.audio_manager.update_gesture_input(gesture_data)
        return op.audio_manager.get_touchdesigner_params()
    return None


def get_audio_params():
    """获取 TouchDesigner 音频参数"""
    if 'op' in dir() and hasattr(op, 'audio_manager'):
        return op.audio_manager.get_touchdesigner_params()
    return None


def get_audio_status():
    """获取音频状态"""
    if 'op' in dir() and hasattr(op, 'audio_manager'):
        return op.audio_manager.get_audio_state()
    return None


def stop_audio():
    """停止音频播放"""
    if 'op' in dir() and hasattr(op, 'audio_manager'):
        op.audio_manager.stop_all_audio()
        return True
    return False


def cleanup_audio_manager():
    """清理音频管理器资源"""
    if 'op' in dir() and hasattr(op, 'audio_manager'):
        op.audio_manager.cleanup()
        delattr(op, 'audio_manager')
        return True
    return False


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("统一音频管理器测试")
    print("=" * 50)

    # 初始化 pygame
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    manager = RealisticAudioManager()

    if manager.initialize():
        print("\n音频管理器初始化成功！")
        print("测试手势控制...")

        # 模拟手势序列
        test_gestures = [
            [1],        # 只有音轨1
            [1, 2],     # 音轨1和2
            [1, 2, 3],  # 所有音轨
            [2, 3],     # 音轨2和3
            [],         # 暂停
            [1],        # 恢复音轨1
        ]

        for i, gestures in enumerate(test_gestures):
            print(f"\n测试 {i+1}: 手势 {gestures}")
            manager.update_from_gestures(gestures)
            time.sleep(2)

            status = manager.get_status_info()
            print(f"  状态: {status['playback_state']}")
            print(f"  位置: {status['current_position']:.1f}秒")

        manager.cleanup()
    else:
        print("初始化失败，请检查音频文件是否存在")

    pygame.quit()
