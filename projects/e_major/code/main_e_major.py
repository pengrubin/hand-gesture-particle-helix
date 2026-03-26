"""
E Major Virtual Orchestra Conductor - Main Program

Interactive virtual orchestra conductor using hand gestures to control
multi-track audio playback through 9-zone spatial mapping.
"""

import cv2
import logging
import sys
import time
import atexit
import signal
from typing import Dict, Set
from pathlib import Path

# Import project modules
from hand_gesture_detector import HandGestureDetector, GestureType, HandData
from grid_zone_detector import GridZoneDetector
from e_major_audio_controller import EMajorAudioController
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# 全局实例用于紧急清理
_conductor_instance = None
_cleanup_done = False

def _emergency_cleanup():
    """紧急清理函数 - 确保摄像头被释放"""
    global _conductor_instance, _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    logger.info("[紧急清理] 正在释放资源...")

    if _conductor_instance is not None:
        try:
            _conductor_instance._cleanup()
        except Exception as e:
            logger.error(f"[紧急清理] 异常: {e}")

    try:
        cv2.destroyAllWindows()
    except:
        pass

    logger.info("[紧急清理] 完成")

def _signal_handler(signum, frame):
    """信号处理器"""
    signal_names = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}
    sig_name = signal_names.get(signum, str(signum))
    logger.info(f"收到信号 {sig_name}，正在安全退出...")

    global _conductor_instance
    if _conductor_instance is not None:
        _conductor_instance.is_running = False

    _emergency_cleanup()
    sys.exit(0)

# 注册信号处理器和atexit
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_emergency_cleanup)


class VirtualOrchestraConductor:
    """
    Main application class for the virtual orchestra conductor.

    Integrates hand gesture detection, zone mapping, and audio control
    to create an interactive conducting experience.
    """

    def __init__(self):
        """Initialize the virtual orchestra conductor."""
        global _conductor_instance
        _conductor_instance = self  # 注册全局实例用于紧急清理

        logger.info("Initializing Virtual Orchestra Conductor...")

        # Validate audio files
        self._validate_audio_files()

        # Initialize camera
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")

        # Get actual frame dimensions
        ret, test_frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")

        frame_height, frame_width = test_frame.shape[:2]
        logger.info(f"Camera resolution: {frame_width}x{frame_height}")

        # Initialize components
        self.gesture_detector = HandGestureDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            max_num_hands=config.MAX_NUM_HANDS,
            fist_threshold=config.FIST_OPENNESS_THRESHOLD,
            fist_hold_duration=config.FIST_HOLD_DURATION
        )

        self.zone_detector = GridZoneDetector(
            frame_width=frame_width,
            frame_height=frame_height
        )

        self.audio_controller = EMajorAudioController(
            audio_directory=str(config.AUDIO_DIR),
            volume_transition_speed=config.VOLUME_TRANSITION_SPEED,
            update_rate=config.VOLUME_UPDATE_RATE
        )

        # Load all audio tracks
        self._load_audio_tracks()

        # State tracking
        self.active_zones: Set[int] = set()
        self.previous_gesture_states: Dict[str, tuple] = {}  # hand_id -> (zone, gesture)
        self.fist_triggered_zones: Set[tuple] = set()  # (hand_id, zone) pairs that triggered fist action

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        logger.info("Initialization complete")

    def _validate_audio_files(self):
        """Validate that all required audio files exist."""
        logger.info("Validating audio files...")
        status = config.validate_audio_files()

        missing_files = [name for name, exists in status.items() if not exists]

        if missing_files:
            logger.error(f"Missing audio files: {', '.join(missing_files)}")
            logger.error(f"Please ensure all files are in: {config.AUDIO_DIR}")
            raise FileNotFoundError(f"Missing {len(missing_files)} audio file(s)")

        logger.info(f"All {len(status)} audio files found")

    def _load_audio_tracks(self):
        """Load all audio tracks into the controller."""
        logger.info("Loading audio tracks...")
        loaded_count = self.audio_controller.load_multiple_tracks(config.AUDIO_FILES)

        if loaded_count != len(config.AUDIO_FILES):
            logger.warning(f"Only loaded {loaded_count}/{len(config.AUDIO_FILES)} tracks")
        else:
            logger.info(f"Successfully loaded all {loaded_count} tracks")

        # Set initial volumes to 0
        for track_name in config.get_all_tracks():
            self.audio_controller.set_immediate_volume(track_name, config.VOLUME_INITIAL)

    def run(self):
        """Main application loop."""
        logger.info("Starting main loop. Press 'q' to quit.")

        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Mirror frame for intuitive interaction
                frame = cv2.flip(frame, 1)

                # Process frame
                self._process_frame(frame)

                # Display frame
                cv2.imshow(config.WINDOW_NAME, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit command received")
                    break
                elif key == ord('p'):
                    # Manual play/pause toggle
                    self._toggle_playback()
                elif key == ord('s'):
                    # Stop all
                    self._stop_all()

                # Update frame counter
                self.frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self._cleanup()

    def _process_frame(self, frame):
        """
        Process a single frame for gesture detection and audio control.

        Args:
            frame: OpenCV frame (numpy array)
        """
        # Detect hands and gestures
        hand_data_list, annotated_frame = self.gesture_detector.process_frame(frame)

        # Draw grid overlay
        if config.SHOW_GRID:
            self.zone_detector.draw_grid_on_frame(annotated_frame)

        # Clear active zones for this frame
        current_active_zones = set()

        # Process each detected hand
        for hand_data in hand_data_list:
            self._process_hand(hand_data, annotated_frame, current_active_zones)

        # Update active zones
        self.active_zones = current_active_zones

        # Draw volume bars
        if config.SHOW_VOLUME_BARS:
            self._draw_volume_bars(annotated_frame)

        # Draw FPS
        self._draw_fps(annotated_frame)

        # Copy annotated frame back to original
        frame[:] = annotated_frame

    def _process_hand(self, hand_data: HandData, frame, active_zones: Set[int]):
        """
        Process a single detected hand for zone and gesture control.

        Args:
            hand_data: HandData object with gesture information
            frame: OpenCV frame for visualization
            active_zones: Set to track active zones in this frame
        """
        # Determine which zone the hand is in
        zone = self.zone_detector.get_zone_from_position(
            hand_data.center_x,
            hand_data.center_y
        )

        if zone == 0:
            return  # Hand outside frame

        # Add to active zones
        active_zones.add(zone)

        # Highlight active zone
        self._highlight_zone(frame, zone)

        # Hand identifier for state tracking
        hand_id = hand_data.handedness

        # Get previous state
        prev_state = self.previous_gesture_states.get(hand_id, (0, GestureType.UNKNOWN))
        prev_zone, prev_gesture = prev_state

        # Check for gesture changes
        if config.is_global_control_zone(zone):
            # Zone 5: Global control
            self._handle_global_control(hand_data)
        else:
            # Individual zone control
            self._handle_zone_control(hand_data, zone, hand_id)

        # Update previous state
        self.previous_gesture_states[hand_id] = (zone, hand_data.gesture)

    def _handle_global_control(self, hand_data: HandData):
        """
        Handle global play/pause control in zone 5.

        Args:
            hand_data: HandData object
        """
        if hand_data.gesture == GestureType.OPEN_PALM:
            # Set all tracks to maximum volume for synchronized volume bars
            all_tracks = config.get_all_tracks()
            for track_name in all_tracks:
                self.audio_controller.set_target_volume(track_name, config.VOLUME_MAX)

            # Then start/resume playback if needed
            if not self.audio_controller.is_playing or self.audio_controller.is_paused:
                logger.info("Global play triggered - all volumes set to maximum")
                self.audio_controller.play_all_tracks()

        elif hand_data.gesture == GestureType.CLOSED_FIST:
            # Check if fist has been held for required duration
            if self.gesture_detector.is_sustained_fist(hand_data):
                hand_id = hand_data.handedness
                trigger_key = (hand_id, 5)

                # Only trigger once per sustained fist
                if trigger_key not in self.fist_triggered_zones:
                    logger.info("Global pause triggered (fist held for 1 second)")
                    self.audio_controller.pause_all_tracks()
                    self.fist_triggered_zones.add(trigger_key)
        else:
            # Reset trigger tracking when gesture changes
            hand_id = hand_data.handedness
            trigger_key = (hand_id, 5)
            if trigger_key in self.fist_triggered_zones:
                self.fist_triggered_zones.remove(trigger_key)

    def _handle_zone_control(self, hand_data: HandData, zone: int, hand_id: str):
        """
        Handle individual zone volume control.

        Args:
            hand_data: HandData object
            zone: Zone number (1-9)
            hand_id: Hand identifier ("Left" or "Right")
        """
        # Get tracks for this zone
        tracks = config.get_tracks_for_zone(zone)
        if not tracks:
            return

        if hand_data.gesture == GestureType.OPEN_PALM:
            # Increase volume to max
            for track_name in tracks:
                self.audio_controller.set_target_volume(track_name, config.VOLUME_MAX)

            # Start playback if not already playing
            if not self.audio_controller.is_playing:
                logger.info("Starting playback (triggered by zone gesture)")
                self.audio_controller.play_all_tracks()

        elif hand_data.gesture == GestureType.CLOSED_FIST:
            # Check if fist has been held for required duration
            if self.gesture_detector.is_sustained_fist(hand_data):
                trigger_key = (hand_id, zone)

                # Only trigger once per sustained fist
                if trigger_key not in self.fist_triggered_zones:
                    logger.info(f"Decreasing volume for zone {zone} tracks (fist held)")
                    for track_name in tracks:
                        self.audio_controller.set_target_volume(track_name, config.VOLUME_MIN)
                    self.fist_triggered_zones.add(trigger_key)
        else:
            # Reset trigger tracking when gesture changes
            trigger_key = (hand_id, zone)
            if trigger_key in self.fist_triggered_zones:
                self.fist_triggered_zones.remove(trigger_key)

    def _highlight_zone(self, frame, zone: int):
        """
        Highlight an active zone on the frame.

        Args:
            frame: OpenCV frame
            zone: Zone number to highlight
        """
        boundaries = self.zone_detector.get_zone_boundaries(zone)
        if boundaries:
            x_min, y_min, x_max, y_max = boundaries

            # Draw colored overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), config.COLOR_ACTIVE_ZONE, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Draw zone description
            zone_desc = config.get_zone_description(zone)
            center = self.zone_detector.get_zone_center(zone)
            if center:
                cv2.putText(
                    frame,
                    zone_desc,
                    (center[0] - 50, center[1] + 30),
                    config.FONT_FACE,
                    1.0,
                    config.COLOR_ACTIVE_ZONE,
                    2
                )

    def _draw_volume_bars(self, frame):
        """
        Draw volume level bars for all zones.

        Args:
            frame: OpenCV frame
        """
        bar_width = 20
        bar_height = 100
        margin = 10

        for zone in range(1, 10):
            tracks = config.get_tracks_for_zone(zone)
            if not tracks:
                continue

            # Calculate average volume for zone
            total_volume = 0.0
            for track_name in tracks:
                track_info = self.audio_controller.get_track_info(track_name)
                if track_info:
                    total_volume += track_info['current_volume']

            avg_volume = total_volume / len(tracks) if tracks else 0.0

            # Get zone center
            center = self.zone_detector.get_zone_center(zone)
            if not center:
                continue

            # Draw volume bar
            bar_x = center[0] - bar_width // 2
            bar_y = center[1] - bar_height - 50

            # Background
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )

            # Volume level
            filled_height = int(bar_height * avg_volume)
            if filled_height > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y + bar_height - filled_height),
                    (bar_x + bar_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1
                )

            # Border
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2
            )

    def _draw_fps(self, frame):
        """
        Draw FPS counter on frame.

        Args:
            frame: OpenCV frame
        """
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Draw status text
        status_text = f"FPS: {fps:.1f} | Playing: {self.audio_controller.is_playing} | Paused: {self.audio_controller.is_paused}"

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            config.FONT_FACE,
            0.7,
            (0, 255, 0),
            2
        )

    def _toggle_playback(self):
        """Toggle between play and pause."""
        if self.audio_controller.is_playing and not self.audio_controller.is_paused:
            logger.info("Manual pause")
            self.audio_controller.pause_all_tracks()
        else:
            logger.info("Manual play")
            self.audio_controller.play_all_tracks()

    def _stop_all(self):
        """Stop all playback and reset volumes."""
        logger.info("Stopping all tracks")
        self.audio_controller.stop_all_tracks()

        # Reset all volumes to 0
        for track_name in config.get_all_tracks():
            self.audio_controller.set_immediate_volume(track_name, config.VOLUME_MIN)

    def _cleanup(self):
        """Clean up resources."""
        global _cleanup_done

        # 防止重复清理
        if _cleanup_done:
            return
        _cleanup_done = True

        logger.info("Cleaning up...")

        # 最重要：首先释放摄像头
        try:
            if hasattr(self, 'camera') and self.camera and self.camera.isOpened():
                self.camera.release()
                logger.info("Camera released")
        except Exception as e:
            logger.error(f"Camera release error: {e}")

        # Close audio controller
        try:
            if hasattr(self, 'audio_controller') and self.audio_controller:
                self.audio_controller.close()
                logger.info("Audio controller closed")
        except Exception as e:
            logger.error(f"Audio controller close error: {e}")

        # Close gesture detector
        try:
            if hasattr(self, 'gesture_detector') and self.gesture_detector:
                self.gesture_detector.close()
                logger.info("Gesture detector closed")
        except Exception as e:
            logger.error(f"Gesture detector close error: {e}")

        # Close all windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.error(f"OpenCV close error: {e}")

        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    conductor = None
    try:
        conductor = VirtualOrchestraConductor()
        conductor.run()
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        # 确保清理被调用
        if conductor is not None:
            conductor._cleanup()


if __name__ == "__main__":
    main()
