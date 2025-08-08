"""
TouchDesigner Audio Manager
处理基于手势的多声部音频控制系统
支持同时播放三个音频文件，根据手势数字控制各声部的音量
"""

import os
import threading
import time
from pathlib import Path

class AudioManager:
    def __init__(self, audio_directory=None):
        """
        初始化音频管理器
        
        Args:
            audio_directory: 音频文件所在目录，默认为当前脚本目录
        """
        # 设置音频文件路径
        if audio_directory is None:
            self.audio_directory = Path(__file__).parent
        else:
            self.audio_directory = Path(audio_directory)
        
        # 定义三个音频文件
        self.audio_files = {
            1: "Fugue in G Trio violin-Violin.mp3",
            2: "Fugue in G Trio-Tenor_Lute.mp3", 
            3: "Fugue in G Trio Organ-Organ.mp3"
        }
        
        # 音频控制状态
        self.audio_state = {
            'is_initialized': False,
            'is_playing': False,
            'current_volumes': {1: 0.0, 2: 0.0, 3: 0.0},  # 当前音量 0.0-1.0
            'target_volumes': {1: 0.0, 2: 0.0, 3: 0.0},   # 目标音量 0.0-1.0
            'active_tracks': set(),  # 当前活跃的音轨
            'fade_duration': 0.5,    # 音量淡入淡出时间(秒)
        }
        
        # TouchDesigner音频参数（用于传递给TouchDesigner的Audio Device Out CHOP）
        self.td_audio_params = {
            'track1_volume': 0.0,
            'track2_volume': 0.0, 
            'track3_volume': 0.0,
            'track1_file': str(self.audio_directory / self.audio_files[1]),
            'track2_file': str(self.audio_directory / self.audio_files[2]),
            'track3_file': str(self.audio_directory / self.audio_files[3]),
            'master_play': False,
            'last_update_time': time.time()
        }
        
        # 音量渐变线程控制
        self.fade_thread = None
        self.fade_running = False
        
        # 验证音频文件存在性
        self._validate_audio_files()
    
    def _validate_audio_files(self):
        """验证音频文件是否存在"""
        missing_files = []
        for track_id, filename in self.audio_files.items():
            file_path = self.audio_directory / filename
            if not file_path.exists():
                missing_files.append(f"Track {track_id}: {filename}")
        
        if missing_files:
            print(f"Warning: Missing audio files:")
            for file in missing_files:
                print(f"  - {file}")
        else:
            print("All audio files found successfully")
            self.audio_state['is_initialized'] = True
    
    def update_gesture_input(self, gesture_data):
        """
        根据手势数据更新音频控制
        
        Args:
            gesture_data: 从手势检测器获得的数据
        """
        if not self.audio_state['is_initialized']:
            return
        
        # 获取当前检测到的数字手势
        active_gestures = set()
        if 'digit_gestures' in gesture_data:
            active_gestures = set(gesture_data['digit_gestures'])
        
        # 更新目标音量
        new_target_volumes = {1: 0.0, 2: 0.0, 3: 0.0}
        for gesture_num in active_gestures:
            if gesture_num in [1, 2, 3]:
                new_target_volumes[gesture_num] = 1.0
        
        # 检查是否需要更新音量
        volume_changed = False
        for track_id in [1, 2, 3]:
            if self.audio_state['target_volumes'][track_id] != new_target_volumes[track_id]:
                self.audio_state['target_volumes'][track_id] = new_target_volumes[track_id]
                volume_changed = True
        
        # 如果音量发生变化，启动渐变
        if volume_changed:
            self._start_volume_fade()
        
        # 更新活跃音轨集合
        self.audio_state['active_tracks'] = active_gestures
        
        # 如果有任何手势，开始播放；如果没有手势，继续播放但音量为0
        should_play = len(active_gestures) > 0 or self.audio_state['is_playing']
        if should_play != self.audio_state['is_playing']:
            self.audio_state['is_playing'] = should_play
            self.td_audio_params['master_play'] = should_play
    
    def _start_volume_fade(self):
        """启动音量渐变线程"""
        # 停止之前的渐变线程
        if self.fade_thread and self.fade_thread.is_alive():
            self.fade_running = False
            self.fade_thread.join(timeout=0.1)
        
        # 启动新的渐变线程
        self.fade_running = True
        self.fade_thread = threading.Thread(target=self._volume_fade_worker)
        self.fade_thread.daemon = True
        self.fade_thread.start()
    
    def _volume_fade_worker(self):
        """音量渐变工作线程"""
        fade_steps = 20  # 渐变步数
        step_duration = self.audio_state['fade_duration'] / fade_steps
        
        # 记录渐变开始时的音量
        start_volumes = self.audio_state['current_volumes'].copy()
        target_volumes = self.audio_state['target_volumes'].copy()
        
        for step in range(fade_steps + 1):
            if not self.fade_running:
                break
            
            # 计算当前步骤的插值比例
            progress = step / fade_steps
            
            # 更新各声部音量
            for track_id in [1, 2, 3]:
                start_vol = start_volumes[track_id]
                target_vol = target_volumes[track_id]
                current_vol = start_vol + (target_vol - start_vol) * progress
                
                self.audio_state['current_volumes'][track_id] = current_vol
                self.td_audio_params[f'track{track_id}_volume'] = current_vol
            
            # 更新时间戳
            self.td_audio_params['last_update_time'] = time.time()
            
            time.sleep(step_duration)
        
        self.fade_running = False
    
    def get_touchdesigner_params(self):
        """获取TouchDesigner所需的音频参数"""
        return self.td_audio_params.copy()
    
    def get_audio_state(self):
        """获取当前音频状态"""
        return {
            'initialized': self.audio_state['is_initialized'],
            'playing': self.audio_state['is_playing'],
            'volumes': self.audio_state['current_volumes'].copy(),
            'active_tracks': list(self.audio_state['active_tracks']),
            'audio_files': self.audio_files.copy()
        }
    
    def set_master_volume(self, volume):
        """设置主音量 (0.0-1.0)"""
        volume = max(0.0, min(1.0, volume))
        # 这个功能可以后续扩展
        pass
    
    def stop_all_audio(self):
        """停止所有音频播放"""
        self.audio_state['is_playing'] = False
        self.audio_state['active_tracks'] = set()
        
        # 将所有音量设为0
        for track_id in [1, 2, 3]:
            self.audio_state['target_volumes'][track_id] = 0.0
        
        self.td_audio_params['master_play'] = False
        self._start_volume_fade()
    
    def cleanup(self):
        """清理资源"""
        self.fade_running = False
        if self.fade_thread and self.fade_thread.is_alive():
            self.fade_thread.join(timeout=1.0)

# TouchDesigner接口函数
def initialize_audio_manager():
    """初始化音频管理器"""
    if not hasattr(op, 'audio_manager'):
        op.audio_manager = AudioManager()
    return op.audio_manager.audio_state['is_initialized']

def update_audio_from_gesture(gesture_data):
    """根据手势数据更新音频"""
    if not hasattr(op, 'audio_manager'):
        initialize_audio_manager()
    
    op.audio_manager.update_gesture_input(gesture_data)
    return op.audio_manager.get_touchdesigner_params()

def get_audio_params():
    """获取TouchDesigner音频参数"""
    if hasattr(op, 'audio_manager'):
        return op.audio_manager.get_touchdesigner_params()
    return None

def get_audio_status():
    """获取音频状态"""
    if hasattr(op, 'audio_manager'):
        return op.audio_manager.get_audio_state()
    return None

def stop_audio():
    """停止音频播放"""
    if hasattr(op, 'audio_manager'):
        op.audio_manager.stop_all_audio()
        return True
    return False