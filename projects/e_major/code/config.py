"""
Configuration file for E Major Virtual Orchestra Conductor.

Contains all constants, zone mappings, and audio file mappings.
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directory (parent of code directory)
BASE_DIR = Path(__file__).parent.parent

# Audio files directory
AUDIO_DIR = BASE_DIR

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ============================================================================
# HAND GESTURE DETECTION SETTINGS
# ============================================================================

# MediaPipe Hands configuration
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Gesture thresholds
FIST_OPENNESS_THRESHOLD = 0.3  # Below this = fist
FIST_HOLD_DURATION = 1.0  # Seconds to hold fist for trigger action

# ============================================================================
# ZONE GRID SETTINGS
# ============================================================================

GRID_ROWS = 3
GRID_COLS = 3

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

# Volume control
VOLUME_TRANSITION_SPEED = 2.0  # Volume units per second (0-1 range)
VOLUME_UPDATE_RATE = 0.05  # Update interval in seconds (20 Hz)

# Target volumes
VOLUME_MAX = 1.0  # Maximum volume for open palm gesture
VOLUME_MIN = 0.0  # Minimum volume for closed fist gesture
VOLUME_INITIAL = 0.0  # Starting volume for all tracks

# ============================================================================
# AUDIO FILE MAPPINGS
# ============================================================================

# Individual audio files
AUDIO_FILES = {
    "Oboe_1": "Oboe_1_in_E.mp3",
    "Oboe_2": "Oboe_2_in_E.mp3",
    "Timpani": "Timpani_in_E.mp3",
    "Trumpet_1": "Trumpet_in_C_1_in_E.mp3",
    "Trumpet_2": "Trumpet_in_C_2_in_E.mp3",
    "Trumpet_3": "Trumpet_in_C_3_in_E.mp3",
    "Violas": "Violas_in_E.mp3",
    "Organ": "Organ_in_E.mp3",
    "Violin": "violin_in_E.mp3",
    "Violins_1": "Violins_1_in_E.mp3",
    "Violins_2": "Violins_2_in_E.mp3",
}

# Volume boost for specific tracks (in dB)
# Positive values increase volume, negative values decrease
# Note: +3.5 dB ≈ 150% volume, +6 dB = 200% volume
TRACK_VOLUME_BOOST: Dict[str, float] = {
    "Violin": 3.5,  # 150% volume boost for violin_in_E.mp3
}

# ============================================================================
# ZONE TO AUDIO TRACK MAPPING
# ============================================================================

# Zone layout (like phone keypad):
#   1  2  3
#   4  5  6
#   7  8  9

ZONE_TRACK_MAPPING: Dict[int, List[str]] = {
    1: ["Oboe_1", "Oboe_2"],
    2: ["Timpani"],
    3: ["Trumpet_1", "Trumpet_2", "Trumpet_3"],
    4: ["Violas"],
    5: [],  # GLOBAL CONTROL ZONE (no individual tracks)
    6: ["Organ"],
    7: ["Violin", "Violins_1", "Violins_2"],
    8: [],  # Reserved
    9: [],  # Reserved
}

# Global control zone
GLOBAL_CONTROL_ZONE = 5

# ============================================================================
# UI DISPLAY SETTINGS
# ============================================================================

# Colors (BGR format for OpenCV)
COLOR_GRID_LINES = (0, 255, 0)  # Green
COLOR_ZONE_NUMBERS = (0, 255, 0)  # Green
COLOR_HAND_CENTER = (255, 0, 255)  # Magenta
COLOR_ACTIVE_ZONE = (0, 255, 255)  # Yellow
COLOR_TEXT_BACKGROUND = (0, 0, 0)  # Black
COLOR_TEXT = (255, 255, 255)  # White

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
ZONE_NUMBER_SCALE = 2.0
ZONE_NUMBER_THICKNESS = 3

# Display options
SHOW_GRID = True
SHOW_ZONE_NUMBERS = True
SHOW_HAND_LANDMARKS = True
SHOW_GESTURE_INFO = True
SHOW_VOLUME_BARS = True

# Window name
WINDOW_NAME = "E Major Virtual Orchestra Conductor"

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Frame processing
PROCESS_EVERY_N_FRAMES = 1  # Process every frame (set to 2 to skip every other frame)

# Volume update throttling
MIN_VOLUME_CHANGE = 0.01  # Minimum volume change to trigger track restart

# ============================================================================
# ZONE DESCRIPTIONS (for UI display)
# ============================================================================

ZONE_DESCRIPTIONS = {
    1: "Oboes",
    2: "Timpani",
    3: "Trumpets",
    4: "Violas",
    5: "GLOBAL",
    6: "Organ",
    7: "Violins",
    8: "Reserved",
    9: "Reserved",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_tracks() -> List[str]:
    """Get list of all track names."""
    return list(AUDIO_FILES.keys())

def get_tracks_for_zone(zone: int) -> List[str]:
    """
    Get list of track names for a specific zone.

    Args:
        zone: Zone number (1-9)

    Returns:
        List of track names
    """
    return ZONE_TRACK_MAPPING.get(zone, [])

def is_global_control_zone(zone: int) -> bool:
    """
    Check if a zone is the global control zone.

    Args:
        zone: Zone number (1-9)

    Returns:
        True if zone is global control zone
    """
    return zone == GLOBAL_CONTROL_ZONE

def get_zone_description(zone: int) -> str:
    """
    Get description text for a zone.

    Args:
        zone: Zone number (1-9)

    Returns:
        Description string
    """
    return ZONE_DESCRIPTIONS.get(zone, f"Zone {zone}")

def validate_audio_files() -> Dict[str, bool]:
    """
    Check if all audio files exist.

    Returns:
        Dictionary mapping file names to existence status
    """
    status = {}
    for track_name, filename in AUDIO_FILES.items():
        file_path = AUDIO_DIR / filename
        status[track_name] = file_path.exists()
    return status
