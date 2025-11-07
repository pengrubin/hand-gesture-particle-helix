"""
E Major Audio Controller - Multi-track audio playback with independent volume control.

This module manages synchronized playback of multiple audio tracks with
individual volume control using pydub and simpleaudio.
"""

import os
import time
import threading
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    from pydub import AudioSegment
    from pydub.playback import _play_with_simpleaudio
    import simpleaudio as sa
except ImportError:
    raise ImportError(
        "pydub and simpleaudio are required. Install with: "
        "pip install pydub simpleaudio"
    )

import config

logger = logging.getLogger(__name__)


@dataclass
class AudioTrack:
    """Data structure for an audio track."""
    name: str
    file_path: str
    audio_segment: Optional[AudioSegment] = None
    current_volume: float = 0.0  # 0.0 to 1.0
    target_volume: float = 0.0  # 0.0 to 1.0
    play_obj: Optional[sa.PlayObject] = None


class EMajorAudioController:
    """
    Controls multi-track audio playback with independent volume control.

    Features:
    - Synchronized playback of multiple tracks
    - Independent volume control per track
    - Smooth volume transitions (fade in/out)
    - Global play/pause control
    """

    def __init__(
        self,
        audio_directory: str,
        volume_transition_speed: float = 0.5,  # Volume change per second
        update_rate: float = 0.05  # Volume update interval in seconds
    ):
        """
        Initialize the audio controller.

        Args:
            audio_directory: Directory containing audio files
            volume_transition_speed: Speed of volume changes (0.0-1.0 per second)
            update_rate: How often to update volumes (seconds)
        """
        self.audio_directory = Path(audio_directory)
        self.volume_transition_speed = volume_transition_speed
        self.update_rate = update_rate

        # Track management
        self.tracks: Dict[str, AudioTrack] = {}
        self.is_playing = False
        self.is_paused = False

        # Threading
        self._volume_control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playback_lock = threading.Lock()

        # Playback position tracking
        self._start_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._total_pause_duration: float = 0.0

    def load_track(self, track_name: str, file_path: str) -> bool:
        """
        Load an audio track from file.

        Args:
            track_name: Unique identifier for the track
            file_path: Path to the audio file

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            full_path = self.audio_directory / file_path
            if not full_path.exists():
                logger.error(f"Audio file not found: {full_path}")
                return False

            audio_segment = AudioSegment.from_file(str(full_path))

            self.tracks[track_name] = AudioTrack(
                name=track_name,
                file_path=str(full_path),
                audio_segment=audio_segment,
                current_volume=0.0,
                target_volume=0.0
            )

            logger.info(f"Loaded track: {track_name} ({file_path})")
            return True

        except Exception as e:
            logger.error(f"Error loading track {track_name}: {e}")
            return False

    def load_multiple_tracks(self, track_mapping: Dict[str, List[str]]) -> int:
        """
        Load multiple audio tracks from a mapping.

        Args:
            track_mapping: Dictionary mapping track names to file paths

        Returns:
            Number of successfully loaded tracks
        """
        loaded_count = 0
        for track_name, file_path in track_mapping.items():
            if self.load_track(track_name, file_path):
                loaded_count += 1
        return loaded_count

    def set_target_volume(self, track_name: str, volume: float):
        """
        Set the target volume for a track (smooth transition).

        Args:
            track_name: Name of the track
            volume: Target volume (0.0 to 1.0)
        """
        if track_name not in self.tracks:
            logger.warning(f"Track not found: {track_name}")
            return

        volume = max(0.0, min(1.0, volume))  # Clamp to valid range
        self.tracks[track_name].target_volume = volume
        logger.debug(f"Set target volume for {track_name}: {volume:.2f}")

    def set_immediate_volume(self, track_name: str, volume: float):
        """
        Set the volume immediately (no transition).

        Args:
            track_name: Name of the track
            volume: Volume level (0.0 to 1.0)
        """
        if track_name not in self.tracks:
            logger.warning(f"Track not found: {track_name}")
            return

        volume = max(0.0, min(1.0, volume))
        self.tracks[track_name].current_volume = volume
        self.tracks[track_name].target_volume = volume

    def play_all_tracks(self):
        """Start synchronized playback of all tracks."""
        if self.is_playing and not self.is_paused:
            logger.warning("Tracks are already playing")
            return

        with self._playback_lock:
            if self.is_paused:
                # Resume from pause
                self._resume_playback()
            else:
                # Start fresh playback
                self._start_playback()

    def pause_all_tracks(self):
        """Pause all tracks."""
        if not self.is_playing or self.is_paused:
            logger.warning("No tracks are playing")
            return

        with self._playback_lock:
            self.is_paused = True
            self._pause_time = time.time()

            # Stop all play objects
            for track in self.tracks.values():
                if track.play_obj and track.play_obj.is_playing():
                    track.play_obj.stop()
                    track.play_obj = None

            logger.info("Paused all tracks")

    def stop_all_tracks(self):
        """Stop all tracks and reset playback."""
        with self._playback_lock:
            self.is_playing = False
            self.is_paused = False

            # Stop all play objects
            for track in self.tracks.values():
                if track.play_obj and track.play_obj.is_playing():
                    track.play_obj.stop()
                    track.play_obj = None

            # Reset timing
            self._start_time = None
            self._pause_time = None
            self._total_pause_duration = 0.0

            logger.info("Stopped all tracks")

    def _start_playback(self):
        """Start fresh playback of all tracks."""
        self.is_playing = True
        self.is_paused = False
        self._start_time = time.time()
        self._total_pause_duration = 0.0

        # Start volume control thread
        self._stop_event.clear()
        self._volume_control_thread = threading.Thread(
            target=self._volume_control_loop,
            daemon=True
        )
        self._volume_control_thread.start()

        # Start playback for each track
        for track in self.tracks.values():
            self._start_track_playback(track)

        logger.info("Started playback of all tracks")

    def _resume_playback(self):
        """Resume playback from pause."""
        if self._pause_time:
            self._total_pause_duration += time.time() - self._pause_time
            self._pause_time = None

        self.is_paused = False

        # Calculate current position
        elapsed_time = self._get_elapsed_playback_time()

        # Resume each track from current position
        for track in self.tracks.values():
            self._start_track_playback(track, start_position_ms=elapsed_time)

        logger.info(f"Resumed playback from {elapsed_time:.2f}ms")

    def _start_track_playback(self, track: AudioTrack, start_position_ms: float = 0.0):
        """
        Start playback for a single track.

        Args:
            track: AudioTrack object
            start_position_ms: Starting position in milliseconds
        """
        if track.audio_segment is None:
            return

        # Apply current volume
        volume_db = self._volume_to_db(track.current_volume)

        # Apply track-specific volume boost if configured
        boost_db = config.TRACK_VOLUME_BOOST.get(track.name, 0.0)
        total_volume_db = volume_db + boost_db

        adjusted_audio = track.audio_segment + total_volume_db

        # Seek to start position if needed
        if start_position_ms > 0:
            adjusted_audio = adjusted_audio[int(start_position_ms):]

        # Start playback in background thread
        def play_track():
            try:
                track.play_obj = _play_with_simpleaudio(adjusted_audio)
            except Exception as e:
                logger.error(f"Error playing track {track.name}: {e}")

        play_thread = threading.Thread(target=play_track, daemon=True)
        play_thread.start()

    def _volume_control_loop(self):
        """Background thread that smoothly adjusts volumes."""
        while not self._stop_event.is_set():
            if self.is_playing and not self.is_paused:
                self._update_all_volumes()
            time.sleep(self.update_rate)

    def _update_all_volumes(self):
        """Update volumes for all tracks (smooth transition)."""
        volume_change = self.volume_transition_speed * self.update_rate

        for track in self.tracks.values():
            if track.current_volume != track.target_volume:
                # Move towards target volume
                if track.current_volume < track.target_volume:
                    track.current_volume = min(
                        track.target_volume,
                        track.current_volume + volume_change
                    )
                else:
                    track.current_volume = max(
                        track.target_volume,
                        track.current_volume - volume_change
                    )

                # Restart track with new volume if significant change
                if abs(track.current_volume - track.target_volume) > 0.01:
                    self._restart_track_with_volume(track)

    def _restart_track_with_volume(self, track: AudioTrack):
        """
        Restart a track with updated volume.

        Args:
            track: AudioTrack to restart
        """
        # Stop current playback
        if track.play_obj and track.play_obj.is_playing():
            track.play_obj.stop()

        # Get current playback position
        elapsed_time = self._get_elapsed_playback_time()

        # Restart with new volume
        self._start_track_playback(track, start_position_ms=elapsed_time)

    def _get_elapsed_playback_time(self) -> float:
        """
        Get elapsed playback time in milliseconds.

        Returns:
            Elapsed time in milliseconds
        """
        if not self._start_time:
            return 0.0

        if self.is_paused and self._pause_time:
            elapsed = (self._pause_time - self._start_time - self._total_pause_duration) * 1000
        else:
            elapsed = (time.time() - self._start_time - self._total_pause_duration) * 1000

        return max(0.0, elapsed)

    @staticmethod
    def _volume_to_db(volume: float) -> float:
        """
        Convert linear volume (0.0-1.0) to decibels.

        Args:
            volume: Linear volume (0.0-1.0)

        Returns:
            Volume in decibels (dB)
        """
        if volume <= 0.0:
            return -60.0  # Effectively silent
        elif volume >= 1.0:
            return 0.0  # No attenuation

        # Logarithmic scaling for perceptual volume
        return 20 * (volume - 1) * 3  # ~-60dB to 0dB range

    def get_track_info(self, track_name: str) -> Optional[Dict]:
        """
        Get information about a track.

        Args:
            track_name: Name of the track

        Returns:
            Dictionary with track information, or None if not found
        """
        if track_name not in self.tracks:
            return None

        track = self.tracks[track_name]
        return {
            "name": track.name,
            "file_path": track.file_path,
            "current_volume": track.current_volume,
            "target_volume": track.target_volume,
            "is_playing": track.play_obj is not None and track.play_obj.is_playing() if track.play_obj else False
        }

    def close(self):
        """Release resources and stop playback."""
        self._stop_event.set()
        self.stop_all_tracks()

        if self._volume_control_thread and self._volume_control_thread.is_alive():
            self._volume_control_thread.join(timeout=1.0)

        logger.info("Audio controller closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
