"""
Grid Zone Detector - Maps hand positions to 9-zone grid system.

This module divides the camera frame into 9 zones (like a phone keypad)
and determines which zone a hand is positioned in.

Zone layout:
    1  2  3
    4  5  6
    7  8  9
"""

from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ZoneConfig:
    """Configuration for zone grid system."""
    rows: int = 3
    cols: int = 3


class GridZoneDetector:
    """
    Detects which zone (1-9) a hand position falls into.

    Zones are numbered like a phone keypad:
    1 2 3 (top row)
    4 5 6 (middle row)
    7 8 9 (bottom row)
    """

    def __init__(self, frame_width: int, frame_height: int, config: Optional[ZoneConfig] = None):
        """
        Initialize the grid zone detector.

        Args:
            frame_width: Width of the camera frame in pixels
            frame_height: Height of the camera frame in pixels
            config: Optional zone configuration (default: 3x3 grid)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config or ZoneConfig()

        # Calculate zone boundaries
        self.zone_width = frame_width / self.config.cols
        self.zone_height = frame_height / self.config.rows

    def get_zone_from_position(self, x: float, y: float) -> int:
        """
        Determine which zone a position falls into.

        Args:
            x: X coordinate (0.0 to 1.0, normalized from MediaPipe)
            y: Y coordinate (0.0 to 1.0, normalized from MediaPipe)

        Returns:
            Zone number (1-9), or 0 if position is invalid
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_x = x * self.frame_width
        pixel_y = y * self.frame_height

        # Check if coordinates are within frame bounds
        if pixel_x < 0 or pixel_x >= self.frame_width or pixel_y < 0 or pixel_y >= self.frame_height:
            return 0

        # Calculate column (0-2) and row (0-2)
        col = int(pixel_x // self.zone_width)
        row = int(pixel_y // self.zone_height)

        # Clamp values to valid range
        col = max(0, min(col, self.config.cols - 1))
        row = max(0, min(row, self.config.rows - 1))

        # Convert to zone number (1-9)
        zone = row * self.config.cols + col + 1

        return zone

    def get_zone_boundaries(self, zone: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the pixel boundaries of a specific zone.

        Args:
            zone: Zone number (1-9)

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixels, or None if invalid zone
        """
        if zone < 1 or zone > 9:
            return None

        # Convert zone number to row and column (0-indexed)
        zone_index = zone - 1
        row = zone_index // self.config.cols
        col = zone_index % self.config.cols

        # Calculate boundaries
        x_min = int(col * self.zone_width)
        y_min = int(row * self.zone_height)
        x_max = int((col + 1) * self.zone_width)
        y_max = int((row + 1) * self.zone_height)

        return (x_min, y_min, x_max, y_max)

    def get_zone_center(self, zone: int) -> Optional[Tuple[int, int]]:
        """
        Get the center point of a specific zone in pixels.

        Args:
            zone: Zone number (1-9)

        Returns:
            Tuple of (x, y) coordinates in pixels, or None if invalid zone
        """
        boundaries = self.get_zone_boundaries(zone)
        if boundaries is None:
            return None

        x_min, y_min, x_max, y_max = boundaries
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        return (center_x, center_y)

    def draw_grid_on_frame(self, frame):
        """
        Draw the zone grid on a frame for visualization.

        Args:
            frame: OpenCV frame (numpy array)

        Returns:
            Frame with grid lines and zone numbers drawn
        """
        import cv2

        # Draw vertical lines
        for i in range(1, self.config.cols):
            x = int(i * self.zone_width)
            cv2.line(frame, (x, 0), (x, self.frame_height), (0, 255, 0), 2)

        # Draw horizontal lines
        for i in range(1, self.config.rows):
            y = int(i * self.zone_height)
            cv2.line(frame, (0, y), (self.frame_width, y), (0, 255, 0), 2)

        # Draw zone numbers
        for zone in range(1, 10):
            center = self.get_zone_center(zone)
            if center:
                cv2.putText(
                    frame,
                    str(zone),
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    3
                )

        return frame
