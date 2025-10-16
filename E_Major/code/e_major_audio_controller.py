#!/usr/bin/env python3
"""
E Major 音频控制器
基于人体检测和小提琴动作的音频控制系统

管理11个音轨：
- 非小提琴音轨（1-8）：Oboe, Organ, Timpani, Trumpet, Violas
- 小提琴音轨（9-11）：Violin, Violins_1, Violins_2

状态机：
1. NO_PERSON: 无人检测 → 所有音轨暂停
2. PERSON_NO_VIOLIN: 检测到人但无小提琴动作 → 非小提琴音轨播放，小提琴音轨静音
3. PERSON_WITH_VIOLIN: 检测到人和小提琴动作 → 所有音轨播放
"""

import pygame
import threading
import time
import os
from typing import Dict, Set, Optional
from enum import Enum


class PlaybackState(Enum):
    """播放状态枚举"""
    NO_PERSON = "no_person"                     # 状态1：无人检测
    PERSON_DETECTED = "person_detected"         # 状态2：检测到人


class EMajorAudioController:
    """E Major 音频控制器"""

    def __init__(self):
        """初始化 E Major 音频控制器"""

        # 主旋律（始终100%）
        self.MAIN_MELODY = {
            9: "violin_in_E.mp3"
        }

        # 小提琴组（由小提琴手势激活）
        self.VIOLIN_GROUP = {
            8: "Violas_in_E.mp3",
            10: "Violins_1_in_E.mp3",
            11: "Violins_2_in_E.mp3"
        }

        # 单簧管组（由单簧管手势激活）
        self.CLARINET_GROUP = {
            1: "Oboe_1_in_E.mp3",
            2: "Oboe_2_in_E.mp3"
        }

        # 钢琴组（由钢琴手势激活）
        self.PIANO_GROUP = {
            3: "Organ_in_E.mp3"
        }

        # 鼓组（由鼓手势激活）
        self.DRUM_GROUP = {
            4: "Timpani_in_E.mp3"
        }

        # 小号组（由小号手势激活）
        self.TRUMPET_GROUP = {
            5: "Trumpet_in_C_1_in_E.mp3",
            6: "Trumpet_in_C_2_in_E.mp3",
            7: "Trumpet_in_C_3_in_E.mp3"
        }

        # 合并所有音轨
        self.audio_files = {
            **self.MAIN_MELODY,
            **self.VIOLIN_GROUP,
            **self.CLARINET_GROUP,
            **self.PIANO_GROUP,
            **self.DRUM_GROUP,
            **self.TRUMPET_GROUP
        }

        # 音频路径基准目录（使用相对路径）
        # 当前文件在 E_Major/code/ 下，音频在 E_Major/ 下
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 音频对象
        self.audio_sounds: Dict[int, pygame.mixer.Sound] = {}
        self.audio_channels: Dict[int, pygame.mixer.Channel] = {}
        self.audio_lengths: Dict[int, float] = {}

        # 播放控制
        self.audio_volumes: Dict[int, float] = {i: 0.0 for i in range(1, 12)}
        self.target_volumes: Dict[int, float] = {i: 0.0 for i in range(1, 12)}
        self.playing_tracks: Set[int] = set()

        # 激活组跟踪（记忆哪些组已被激活）
        self.activated_groups: Set[str] = set()  # {'violin', 'clarinet', 'piano', 'drum', 'trumpet'}

        # 断点续播：位置跟踪
        self.master_playing = False
        self.session_start_time: Optional[float] = None  # 播放会话开始时间
        self.total_pause_duration = 0.0                  # 总暂停时间
        self.current_pause_start: Optional[float] = None # 当前暂停开始时间

        # 状态机
        self.current_state = PlaybackState.NO_PERSON
        self.previous_state = PlaybackState.NO_PERSON

        # 音量渐变控制
        self.volume_fade_speed = 0.25  # 音量渐变速度（0-1之间，值越大渐变越快）
        self.fade_thread_running = False

        # 状态稳定性控制（避免状态抖动）
        self.state_change_threshold = 0.3  # 状态切换阈值（秒）
        self.last_state_change_time = 0.0

        # 启用标志
        self.enabled = False

        print("🎵 E Major 音频控制器初始化...")
        print(f"   主旋律: {len(self.MAIN_MELODY)} 个")
        print(f"   小提琴组: {len(self.VIOLIN_GROUP)} 个")
        print(f"   单簧管组: {len(self.CLARINET_GROUP)} 个")
        print(f"   钢琴组: {len(self.PIANO_GROUP)} 个")
        print(f"   鼓组: {len(self.DRUM_GROUP)} 个")
        print(f"   小号组: {len(self.TRUMPET_GROUP)} 个")
        print(f"   总计: {len(self.audio_files)} 个音轨")

    def initialize(self) -> bool:
        """
        初始化音频系统

        Returns:
            bool: 初始化成功返回 True，失败返回 False
        """
        try:
            # 确保有足够的混音通道（11个音轨需要至少11个通道）
            required_channels = len(self.audio_files)
            current_channels = pygame.mixer.get_num_channels()
            if current_channels < required_channels:
                pygame.mixer.set_num_channels(required_channels + 1)  # +1 作为安全缓冲
                print(f"✅ 设置混音通道数: {current_channels} → {required_channels + 1}")

            # 检查文件存在性
            missing_files = []
            for track_id, filename in self.audio_files.items():
                filepath = os.path.join(self.base_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filepath)

            if missing_files:
                print("⚠️ 缺失音频文件:")
                for file in missing_files:
                    print(f"   - {file}")
                return False

            # 加载音频文件
            for track_id, filename in self.audio_files.items():
                filepath = os.path.join(self.base_dir, filename)
                try:
                    print(f"加载音轨 {track_id}: {filename}")
                    sound = pygame.mixer.Sound(filepath)
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

            # 启用控制器
            self.enabled = True

            # 启动音量渐变线程
            self.start_fade_thread()

            # 🆕 自动启动播放会话（确保音轨立即可用）
            self._start_playback_session()

            print(f"✅ E Major 音频控制器就绪，已加载 {len(self.audio_sounds)} 个音轨")

            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False

    def start_fade_thread(self):
        """启动音量渐变线程"""
        if self.fade_thread_running:
            return

        self.fade_thread_running = True
        fade_thread = threading.Thread(target=self._fade_loop, daemon=True)
        fade_thread.start()
        print("✅ 音量渐变线程已启动")

    def _fade_loop(self):
        """
        音量渐变循环（优化版）
        在独立线程中运行，平滑过渡音量变化
        优化：降低更新频率到20 FPS，早退出未变化音轨
        """
        while self.fade_thread_running:
            try:
                has_changes = False

                for track_id in self.audio_sounds.keys():
                    current_vol = self.audio_volumes[track_id]
                    target_vol = self.target_volumes[track_id]

                    # 如果当前音量与目标音量差异超过阈值，进行渐变
                    if abs(current_vol - target_vol) > 0.01:
                        has_changes = True
                        volume_diff = target_vol - current_vol
                        new_vol = current_vol + volume_diff * self.volume_fade_speed

                        # 更新音量
                        self.audio_volumes[track_id] = new_vol
                        if track_id in self.audio_sounds:
                            self.audio_sounds[track_id].set_volume(new_vol)

                # 优化：20 FPS渐变更新频率（从30降低，节省CPU）
                # 人耳无法区分20 FPS vs 30 FPS的音量变化
                time.sleep(1/20)

            except KeyError as e:
                print(f"⚠️ 音量渐变线程键错误: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ 音量渐变线程错误: {e}")
                time.sleep(0.1)

    def get_current_position(self) -> float:
        """
        获取当前播放位置（考虑暂停时间）
        实现断点续播机制

        Returns:
            float: 当前播放位置（秒）
        """
        if not self.session_start_time:
            return 0.0

        current_time = time.time()

        # 计算总的实际播放时间
        elapsed_since_session = current_time - self.session_start_time
        actual_play_time = elapsed_since_session - self.total_pause_duration

        # 如果当前正在暂停，还要减去当前暂停的时间
        if self.current_state == PlaybackState.NO_PERSON and self.current_pause_start:
            current_pause_time = current_time - self.current_pause_start
            actual_play_time -= current_pause_time

        # 循环播放检查
        if self.audio_lengths:
            min_length = min(self.audio_lengths.values())
            if actual_play_time >= min_length:
                actual_play_time = actual_play_time % min_length

        return max(0.0, actual_play_time)

    def update_from_instruments(self, person_detected: bool,
                               detected_instruments: Dict[str, float]):
        """
        根据检测到的乐器更新音频

        Args:
            person_detected: 是否检测到人
            detected_instruments: 检测到的乐器字典 {'violin': 0.8, ...}
        """
        if not self.enabled:
            return

        # 确定新状态
        new_state = (PlaybackState.PERSON_DETECTED if person_detected
                    else PlaybackState.NO_PERSON)

        # 状态转换
        current_time = time.time()
        if new_state != self.current_state:
            time_since_last_change = current_time - self.last_state_change_time
            if time_since_last_change >= self.state_change_threshold:
                self._transition_to_state(new_state)
                self.last_state_change_time = current_time

        # 如果有人，处理乐器激活
        if new_state == PlaybackState.PERSON_DETECTED:
            self._update_instrument_volumes(detected_instruments)

        # 定期输出状态
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0

        if current_time - self._last_status_time > 2.0:
            pos = self.get_current_position()
            state_name = self.current_state.value.upper()
            activated = ', '.join(self.activated_groups) if self.activated_groups else 'None'
            print(f"🎵 音频: {state_name}, 位置: {pos:.1f}秒, "
                  f"检测到: {list(detected_instruments.keys())}, "
                  f"已激活: {activated}")
            self._last_status_time = current_time

    def _update_instrument_volumes(self, detected_instruments: Dict[str, float]):
        """根据检测结果更新乐器组音量"""
        # 将新检测到的乐器加入激活组
        if 'violin' in detected_instruments:
            self.activated_groups.add('violin')

        if 'clarinet' in detected_instruments:
            self.activated_groups.add('clarinet')

        if 'piano' in detected_instruments:
            self.activated_groups.add('piano')

        if 'drum' in detected_instruments:
            self.activated_groups.add('drum')

        if 'trumpet' in detected_instruments:
            self.activated_groups.add('trumpet')

        # 应用基于激活组的音量
        self._apply_activated_volumes()

    def _apply_activated_volumes(self):
        """基于激活组应用音量"""
        # 主旋律始终100%
        for track_id in self.MAIN_MELODY.keys():
            self.target_volumes[track_id] = 1.0

        # 小提琴组
        for track_id in self.VIOLIN_GROUP.keys():
            self.target_volumes[track_id] = 1.0 if 'violin' in self.activated_groups else 0.0

        # 单簧管组
        for track_id in self.CLARINET_GROUP.keys():
            self.target_volumes[track_id] = 1.0 if 'clarinet' in self.activated_groups else 0.0

        # 钢琴组
        for track_id in self.PIANO_GROUP.keys():
            self.target_volumes[track_id] = 1.0 if 'piano' in self.activated_groups else 0.0

        # 鼓组
        for track_id in self.DRUM_GROUP.keys():
            self.target_volumes[track_id] = 1.0 if 'drum' in self.activated_groups else 0.0

        # 小号组
        for track_id in self.TRUMPET_GROUP.keys():
            self.target_volumes[track_id] = 1.0 if 'trumpet' in self.activated_groups else 0.0

    def _transition_to_state(self, new_state: PlaybackState):
        """状态转换处理"""
        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state

        print(f"🔄 状态: {old_state.value} → {new_state.value}")

        if new_state == PlaybackState.NO_PERSON:
            # 人消失 - 清除所有激活组
            self.activated_groups.clear()
            self._pause_all_tracks()

        elif new_state == PlaybackState.PERSON_DETECTED:
            # 人出现 - 恢复并播放主旋律
            self._resume_if_paused()
            self._play_main_melody()

    def _pause_all_tracks(self):
        """
        暂停所有音轨
        State 1: NO_PERSON
        """
        print("⏸️ 暂停所有音轨")

        # 记录暂停开始时间
        if self.current_pause_start is None:
            self.current_pause_start = time.time()

        # 音量渐变到0（不立即停止播放，保持位置）
        for track_id in range(1, 12):
            self.target_volumes[track_id] = 0.0

    def _resume_if_paused(self):
        """如果当前处于暂停状态，则恢复播放"""
        if self.previous_state == PlaybackState.NO_PERSON and self.current_pause_start:
            # 累计暂停时间
            current_time = time.time()
            pause_duration = current_time - self.current_pause_start
            self.total_pause_duration += pause_duration
            self.current_pause_start = None

            print(f"▶️ 从暂停恢复 (暂停了 {pause_duration:.1f}秒)")

            # 如果会话尚未开始，现在开始
            if not self.session_start_time:
                self._start_playback_session()

            # 确保所有音轨在播放中（即使音量为0）
            self._ensure_tracks_playing()

    def _start_playback_session(self):
        """开始播放会话"""
        print("🔄 启动播放会话")
        self.session_start_time = time.time()
        self.master_playing = True

        # 启动所有音轨（静音状态）
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ 启动音轨 {track_id} 失败: {e}")

        print("✅ 播放会话已开始")

    def _ensure_tracks_playing(self):
        """确保所有音轨在播放中"""
        for track_id in self.audio_sounds.keys():
            if track_id not in self.playing_tracks:
                try:
                    self.audio_sounds[track_id].set_volume(0.0)
                    self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                    self.playing_tracks.add(track_id)
                except Exception as e:
                    print(f"❌ 启动音轨 {track_id} 失败: {e}")

    def _play_main_melody(self):
        """播放主旋律（violin_in_E 始终100%）"""
        for track_id in self.MAIN_MELODY.keys():
            self.target_volumes[track_id] = 1.0

    def manual_pause_resume(self):
        """手动暂停/恢复（用于调试或手动控制）"""
        if self.current_state != PlaybackState.NO_PERSON:
            # 手动进入暂停状态
            self._transition_to_state(PlaybackState.NO_PERSON)
            print("⏸️ 手动暂停")
        else:
            # 手动恢复到有人状态
            self._transition_to_state(PlaybackState.PERSON_DETECTED)
            print("▶️ 手动恢复")

    def pause_all(self):
        """暂停所有音轨（外部调用接口）"""
        self._transition_to_state(PlaybackState.NO_PERSON)
        print("⏸️ 手动暂停所有音轨")

    def resume_all(self):
        """恢复所有音轨（外部调用接口）"""
        # 恢复到有人状态
        self._transition_to_state(PlaybackState.PERSON_DETECTED)
        print("▶️ 手动恢复所有音轨")

    def reset_position(self):
        """重置播放位置"""
        print("🔄 重置播放位置")

        # 重置时间跟踪
        self.session_start_time = time.time()
        self.total_pause_duration = 0.0
        self.current_pause_start = None

        # 停止所有当前播放
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()

        # 重新启动所有音轨
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ 重启音轨 {track_id} 失败: {e}")

        # 清除所有音量，等待姿态检测
        for track_id in range(1, 12):
            self.target_volumes[track_id] = 0.0

        print("✅ 播放位置已重置")

    def get_status_info(self) -> dict:
        """
        获取当前状态信息

        Returns:
            dict: 包含所有状态信息的字典
        """
        current_pos = self.get_current_position()

        # 获取当前正在播放的音轨
        playing_tracks_list = [
            track_id for track_id, vol in self.target_volumes.items()
            if vol > 0.01
        ]

        return {
            'enabled': self.enabled,
            'current_state': self.current_state.value,
            'activated_groups': list(self.activated_groups),
            'master_playing': self.master_playing,
            'playing_tracks': playing_tracks_list,
            'volumes': self.audio_volumes.copy(),
            'target_volumes': self.target_volumes.copy(),
            'playback_position': current_pos,
            'current_position': current_pos,  # 兼容性
            'audio_lengths': self.audio_lengths.copy(),
            'total_pause_duration': self.total_pause_duration,
            'session_start_time': self.session_start_time
        }

    def cleanup(self):
        """清理资源"""
        print("🧹 清理 E Major 音频控制器...")

        # 停止渐变线程
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

        print("✅ E Major 音频控制器已清理")


# 使用示例
if __name__ == "__main__":
    # 初始化 pygame.mixer
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    # 创建控制器
    controller = EMajorAudioController()

    # 初始化
    if controller.initialize():
        print("\n" + "="*60)
        print("E Major 音频控制器测试")
        print("="*60)

        try:
            # 模拟状态转换
            print("\n1. 模拟检测到人（无乐器）")
            controller.update_from_instruments(person_detected=True, detected_instruments={})
            time.sleep(3)

            print("\n2. 模拟检测到小提琴")
            controller.update_from_instruments(person_detected=True, detected_instruments={'violin': 0.8})
            time.sleep(3)

            print("\n3. 模拟检测到钢琴")
            controller.update_from_instruments(person_detected=True, detected_instruments={'piano': 0.7})
            time.sleep(3)

            print("\n4. 模拟检测到小号和鼓")
            controller.update_from_instruments(person_detected=True, detected_instruments={'trumpet': 0.85, 'drum': 0.9})
            time.sleep(3)

            print("\n5. 模拟人消失")
            controller.update_from_instruments(person_detected=False, detected_instruments={})
            time.sleep(2)

            print("\n6. 获取状态信息")
            status = controller.get_status_info()
            print(f"当前状态: {status['current_state']}")
            print(f"激活组: {status['activated_groups']}")
            print(f"播放位置: {status['playback_position']:.2f}秒")
            print(f"正在播放的音轨: {status['playing_tracks']}")

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            controller.cleanup()
    else:
        print("❌ 控制器初始化失败")
