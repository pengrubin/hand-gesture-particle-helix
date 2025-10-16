#!/usr/bin/env python3
"""
BWV 29 in D Audio Controller
Based on body detection and violin gesture audio control system

Manages 9 tracks:
- Non-violin tracks (1-8): Continuo, Oboe, Organ, Timpani, Trumpets, Viola
- Violin track (9): Violins

State machine:
1. NO_PERSON: No person detected -> All tracks paused
2. PERSON_NO_VIOLIN: Person detected but no violin gesture -> Non-violin tracks play, violin track muted
3. PERSON_WITH_VIOLIN: Person detected and violin gesture -> All tracks play
"""

import pygame
import threading
import time
import os
from typing import Dict, Set, Optional
from enum import Enum


class PlaybackState(Enum):
    """Playback state enum"""
    NO_PERSON = "no_person"                     # State 1: No person detected
    PERSON_NO_VIOLIN = "person_no_violin"       # State 2: Person detected but no violin gesture
    PERSON_WITH_VIOLIN = "person_with_violin"   # State 3: Person detected and violin gesture


class EMajorAudioController:
    """BWV 29 in D Audio Controller"""

    def __init__(self):
        """Initialize BWV 29 in D Audio Controller"""

        # Audio file definitions (relative to BWV_29_in_D directory)
        self.NON_VIOLIN_TRACKS = {
            1: "Continuo_in_D.mp3",
            2: "Oboe_I_in_D.mp3",
            3: "Organo_obligato_in_D.mp3",
            4: "Timpani_in_D.mp3",
            5: "Tromba_I_in_D.mp3",
            6: "Tromba_II_in_D.mp3",
            7: "Tromba_III_in_D.mp3",
            8: "Viola_in_D.mp3"
        }

        self.VIOLIN_TRACKS = {
            9: "Violins_in_D.mp3"
        }

        # Merge all tracks
        self.audio_files = {**self.NON_VIOLIN_TRACKS, **self.VIOLIN_TRACKS}

        # Audio path base directory (using relative path)
        # Current file is in BWV_29_in_D/code/, audio files in BWV_29_in_D/
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Audio objects
        self.audio_sounds: Dict[int, pygame.mixer.Sound] = {}
        self.audio_channels: Dict[int, pygame.mixer.Channel] = {}
        self.audio_lengths: Dict[int, float] = {}

        # Playback control
        self.audio_volumes: Dict[int, float] = {i: 0.0 for i in range(1, 10)}
        self.target_volumes: Dict[int, float] = {i: 0.0 for i in range(1, 10)}
        self.playing_tracks: Set[int] = set()

        # Resume from pause: position tracking
        self.master_playing = False
        self.session_start_time: Optional[float] = None  # Playback session start time
        self.total_pause_duration = 0.0                  # Total pause duration
        self.current_pause_start: Optional[float] = None # Current pause start time

        # State machine
        self.current_state = PlaybackState.NO_PERSON
        self.previous_state = PlaybackState.NO_PERSON

        # Volume fade control
        self.volume_fade_speed = 0.25  # Volume fade speed (0-1, higher = faster fade)
        self.fade_thread_running = False

        # State stability control (avoid state jitter)
        self.state_change_threshold = 0.3  # State change threshold (seconds)
        self.last_state_change_time = 0.0

        # Enable flag
        self.enabled = False

        print("🎵 BWV 29 in D Audio Controller initializing...")
        print(f"   Non-violin tracks: {len(self.NON_VIOLIN_TRACKS)} tracks")
        print(f"   Violin tracks: {len(self.VIOLIN_TRACKS)} track")
        print(f"   Total: {len(self.audio_files)} tracks")

    def initialize(self) -> bool:
        """
        Initialize audio system

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Ensure enough mixer channels (9 tracks need at least 9 channels)
            required_channels = len(self.audio_files)
            current_channels = pygame.mixer.get_num_channels()
            if current_channels < required_channels:
                pygame.mixer.set_num_channels(required_channels + 1)  # +1 as safety buffer
                print(f"✅ Set mixer channels: {current_channels} → {required_channels + 1}")

            # Check file existence
            missing_files = []
            for track_id, filename in self.audio_files.items():
                filepath = os.path.join(self.base_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filepath)

            if missing_files:
                print("⚠️ Missing audio files:")
                for file in missing_files:
                    print(f"   - {file}")
                return False

            # Load audio files
            for track_id, filename in self.audio_files.items():
                filepath = os.path.join(self.base_dir, filename)
                try:
                    print(f"Loading track {track_id}: {filename}")
                    sound = pygame.mixer.Sound(filepath)
                    sound.set_volume(0.0)

                    length = sound.get_length()
                    print(f"  Duration: {length:.1f}s")

                    self.audio_sounds[track_id] = sound
                    self.audio_channels[track_id] = pygame.mixer.Channel(track_id - 1)
                    self.audio_lengths[track_id] = length

                    print(f"✅ Track {track_id} loaded successfully")
                except Exception as e:
                    print(f"❌ Track {track_id} loading failed: {e}")
                    continue

            if not self.audio_sounds:
                print("❌ No audio files loaded successfully")
                return False

            # Enable controller
            self.enabled = True

            # Start volume fade thread
            self.start_fade_thread()

            # Auto-start playback session (ensure tracks are immediately available)
            self._start_playback_session()

            print(f"✅ BWV 29 in D Audio Controller ready, loaded {len(self.audio_sounds)} tracks")

            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def start_fade_thread(self):
        """Start volume fade thread"""
        if self.fade_thread_running:
            return

        self.fade_thread_running = True
        fade_thread = threading.Thread(target=self._fade_loop, daemon=True)
        fade_thread.start()
        print("✅ Volume fade thread started")

    def _fade_loop(self):
        """
        Volume fade loop (optimized version)
        Runs in separate thread, smoothly transitions volume changes
        Optimization: Lower update frequency to 20 FPS, early exit for unchanged tracks
        """
        while self.fade_thread_running:
            try:
                has_changes = False

                for track_id in self.audio_sounds.keys():
                    current_vol = self.audio_volumes[track_id]
                    target_vol = self.target_volumes[track_id]

                    # If current volume differs from target volume by more than threshold, fade
                    if abs(current_vol - target_vol) > 0.01:
                        has_changes = True
                        volume_diff = target_vol - current_vol
                        new_vol = current_vol + volume_diff * self.volume_fade_speed

                        # Update volume
                        self.audio_volumes[track_id] = new_vol
                        if track_id in self.audio_sounds:
                            self.audio_sounds[track_id].set_volume(new_vol)

                # Optimization: 20 FPS fade update frequency (reduced from 30, saves CPU)
                # Human ear cannot distinguish 20 FPS vs 30 FPS volume changes
                time.sleep(1/20)

            except KeyError as e:
                print(f"⚠️ Volume fade thread key error: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Volume fade thread error: {e}")
                time.sleep(0.1)

    def get_current_position(self) -> float:
        """
        Get current playback position (considering pause time)
        Implements resume from pause mechanism

        Returns:
            float: Current playback position (seconds)
        """
        if not self.session_start_time:
            return 0.0

        current_time = time.time()

        # Calculate total actual play time
        elapsed_since_session = current_time - self.session_start_time
        actual_play_time = elapsed_since_session - self.total_pause_duration

        # If currently paused, also subtract current pause time
        if self.current_state == PlaybackState.NO_PERSON and self.current_pause_start:
            current_pause_time = current_time - self.current_pause_start
            actual_play_time -= current_pause_time

        # Loop playback check
        if self.audio_lengths:
            min_length = min(self.audio_lengths.values())
            if actual_play_time >= min_length:
                actual_play_time = actual_play_time % min_length

        return max(0.0, actual_play_time)

    def update_from_pose(self, person_detected: bool, violin_gesture_detected: bool):
        """
        Update audio based on pose detection results

        State transition logic:
        - NO_PERSON → PERSON_NO_VIOLIN: Person detected
        - PERSON_NO_VIOLIN → PERSON_WITH_VIOLIN: Violin gesture detected
        - PERSON_WITH_VIOLIN → PERSON_NO_VIOLIN: Violin gesture stopped
        - Any state → NO_PERSON: Person disappeared

        Args:
            person_detected: Whether person is detected
            violin_gesture_detected: Whether violin gesture is detected
        """
        if not self.enabled:
            return

        # Determine new state
        if not person_detected:
            new_state = PlaybackState.NO_PERSON
        elif person_detected and not violin_gesture_detected:
            new_state = PlaybackState.PERSON_NO_VIOLIN
        elif person_detected and violin_gesture_detected:
            new_state = PlaybackState.PERSON_WITH_VIOLIN
        else:
            # Should not reach here
            new_state = self.current_state

        # State change detection (with debounce)
        current_time = time.time()
        if new_state != self.current_state:
            # Check if state change time threshold is met (avoid rapid jitter)
            time_since_last_change = current_time - self.last_state_change_time
            if time_since_last_change >= self.state_change_threshold:
                self._transition_to_state(new_state)
                self.last_state_change_time = current_time

        # Periodic status output (lower frequency)
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0

        if current_time - self._last_status_time > 2.0:  # Output every 2 seconds
            pos = self.get_current_position()
            state_name = self.current_state.value.upper()
            print(f"🎵 Audio status: {state_name}, Position: {pos:.1f}s, "
                  f"Person: {person_detected}, Violin: {violin_gesture_detected}")
            self._last_status_time = current_time

    def _transition_to_state(self, new_state: PlaybackState):
        """
        State transition handling

        Args:
            new_state: New state
        """
        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state

        print(f"🔄 State transition: {old_state.value} → {new_state.value}")

        # Execute operations based on new state
        if new_state == PlaybackState.NO_PERSON:
            self._pause_all_tracks()
        elif new_state == PlaybackState.PERSON_NO_VIOLIN:
            self._resume_if_paused()
            self._play_non_violin_tracks()
            self._mute_violin_tracks()
        elif new_state == PlaybackState.PERSON_WITH_VIOLIN:
            self._resume_if_paused()
            self._play_all_tracks()

    def _pause_all_tracks(self):
        """
        Pause all tracks
        State 1: NO_PERSON
        """
        print("⏸️ Pausing all tracks")

        # Record pause start time
        if self.current_pause_start is None:
            self.current_pause_start = time.time()

        # Fade volume to 0 (don't stop playback immediately, maintain position)
        for track_id in range(1, 10):
            self.target_volumes[track_id] = 0.0

    def _resume_if_paused(self):
        """If currently paused, resume playback"""
        if self.previous_state == PlaybackState.NO_PERSON and self.current_pause_start:
            # Accumulate pause time
            current_time = time.time()
            pause_duration = current_time - self.current_pause_start
            self.total_pause_duration += pause_duration
            self.current_pause_start = None

            print(f"▶️ Resuming from pause (paused for {pause_duration:.1f}s)")

            # If session hasn't started, start now
            if not self.session_start_time:
                self._start_playback_session()

            # Ensure all tracks are playing (even if volume is 0)
            self._ensure_tracks_playing()

    def _start_playback_session(self):
        """Start playback session"""
        print("🔄 Starting playback session")
        self.session_start_time = time.time()
        self.master_playing = True

        # Start all tracks (muted)
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ Starting track {track_id} failed: {e}")

        print("✅ Playback session started")

    def _ensure_tracks_playing(self):
        """Ensure all tracks are playing"""
        for track_id in self.audio_sounds.keys():
            if track_id not in self.playing_tracks:
                try:
                    self.audio_sounds[track_id].set_volume(0.0)
                    self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                    self.playing_tracks.add(track_id)
                except Exception as e:
                    print(f"❌ Starting track {track_id} failed: {e}")

    def _play_non_violin_tracks(self):
        """
        Play non-violin tracks, mute violin tracks
        State 2: PERSON_NO_VIOLIN
        """
        # Non-violin tracks volume 100%
        for track_id in self.NON_VIOLIN_TRACKS.keys():
            self.target_volumes[track_id] = 1.0

        # Violin tracks volume 0% (handled in _mute_violin_tracks)

    def _mute_violin_tracks(self):
        """Mute violin tracks"""
        for track_id in self.VIOLIN_TRACKS.keys():
            self.target_volumes[track_id] = 0.0

    def _play_all_tracks(self):
        """
        Play all tracks
        State 3: PERSON_WITH_VIOLIN
        """
        # All tracks volume 100%
        for track_id in range(1, 10):
            self.target_volumes[track_id] = 1.0

    def manual_pause_resume(self):
        """Manual pause/resume (for debugging or manual control)"""
        if self.current_state != PlaybackState.NO_PERSON:
            # Manually enter pause state
            self._transition_to_state(PlaybackState.NO_PERSON)
            print("⏸️ Manual pause")
        else:
            # Manually resume to person no violin state
            self._transition_to_state(PlaybackState.PERSON_NO_VIOLIN)
            print("▶️ Manual resume")

    def pause_all(self):
        """Pause all tracks (external call interface)"""
        self._transition_to_state(PlaybackState.NO_PERSON)
        print("⏸️ Manually paused all tracks")

    def resume_all(self):
        """Resume all tracks (external call interface)"""
        # Resume to person no violin state (conservative resume)
        self._transition_to_state(PlaybackState.PERSON_NO_VIOLIN)
        print("▶️ Manually resumed all tracks")

    def reset_position(self):
        """Reset playback position"""
        print("🔄 Resetting playback position")

        # Reset time tracking
        self.session_start_time = time.time()
        self.total_pause_duration = 0.0
        self.current_pause_start = None

        # Stop all current playback
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()

        # Restart all tracks
        for track_id in self.audio_sounds.keys():
            try:
                self.audio_sounds[track_id].set_volume(0.0)
                self.audio_channels[track_id].play(self.audio_sounds[track_id], loops=-1)
                self.playing_tracks.add(track_id)
            except Exception as e:
                print(f"❌ Restarting track {track_id} failed: {e}")

        # Clear all volumes, wait for pose detection
        for track_id in range(1, 10):
            self.target_volumes[track_id] = 0.0

        print("✅ Playback position reset")

    def get_status_info(self) -> dict:
        """
        Get current status information

        Returns:
            dict: Dictionary containing all status information
        """
        current_pos = self.get_current_position()

        # Get currently playing tracks
        playing_tracks_list = [
            track_id for track_id, vol in self.target_volumes.items()
            if vol > 0.01
        ]

        return {
            'enabled': self.enabled,
            'current_state': self.current_state.value,
            'master_playing': self.master_playing,
            'playing_tracks': playing_tracks_list,
            'volumes': self.audio_volumes.copy(),
            'target_volumes': self.target_volumes.copy(),
            'playback_position': current_pos,
            'current_position': current_pos,  # Compatibility
            'audio_lengths': self.audio_lengths.copy(),
            'total_pause_duration': self.total_pause_duration,
            'session_start_time': self.session_start_time
        }

    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up BWV 29 in D Audio Controller...")

        # Stop fade thread
        self.fade_thread_running = False

        # Stop all playback
        for track_id in list(self.playing_tracks):
            try:
                self.audio_channels[track_id].stop()
            except:
                pass

        self.playing_tracks.clear()
        self.master_playing = False
        self.enabled = False

        print("✅ BWV 29 in D Audio Controller cleaned up")


# Usage example
if __name__ == "__main__":
    # Initialize pygame.mixer
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    # Create controller
    controller = EMajorAudioController()

    # Initialize
    if controller.initialize():
        print("\n" + "="*60)
        print("BWV 29 in D Audio Controller Test")
        print("="*60)

        try:
            # Simulate state transitions
            print("\n1. Simulating person detected (no violin gesture)")
            controller.update_from_pose(person_detected=True, violin_gesture_detected=False)
            time.sleep(3)

            print("\n2. Simulating violin gesture detected")
            controller.update_from_pose(person_detected=True, violin_gesture_detected=True)
            time.sleep(3)

            print("\n3. Simulating violin gesture stopped")
            controller.update_from_pose(person_detected=True, violin_gesture_detected=False)
            time.sleep(3)

            print("\n4. Simulating person disappeared")
            controller.update_from_pose(person_detected=False, violin_gesture_detected=False)
            time.sleep(2)

            print("\n5. Getting status information")
            status = controller.get_status_info()
            print(f"Current state: {status['current_state']}")
            print(f"Playback position: {status['playback_position']:.2f}s")
            print(f"Playing tracks: {status['playing_tracks']}")

        except KeyboardInterrupt:
            print("\nUser interrupted")
        finally:
            controller.cleanup()
    else:
        print("❌ Controller initialization failed")
