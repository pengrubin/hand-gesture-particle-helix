"""
Hand Gesture Detector - MediaPipe Hands-based gesture recognition.

This module detects hand gestures (open palm vs closed fist) and tracks
hand positions using MediaPipe Hands solution.
"""

import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is required. Install with: pip install mediapipe")


class GestureType(Enum):
    """Types of hand gestures recognized by the system."""
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    UNKNOWN = "unknown"


@dataclass
class HandData:
    """Data structure for detected hand information."""
    center_x: float  # Normalized X coordinate (0.0-1.0)
    center_y: float  # Normalized Y coordinate (0.0-1.0)
    gesture: GestureType
    handedness: str  # "Left" or "Right"
    openness: float  # 0.0 (closed) to 1.0 (open)
    fist_duration: float  # Duration in seconds that fist has been held


class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe Hands.

    Recognizes:
    - Open palm (fingers spread)
    - Closed fist (with duration tracking for sustained gestures)
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        fist_threshold: float = 0.3,  # Openness threshold for fist detection
        fist_hold_duration: float = 1.0  # Seconds to hold fist for trigger
    ):
        """
        Initialize the hand gesture detector.

        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect (1 or 2)
            fist_threshold: Openness value below which hand is considered a fist
            fist_hold_duration: Duration in seconds a fist must be held to trigger action
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.fist_threshold = fist_threshold
        self.fist_hold_duration = fist_hold_duration

        # Track fist gesture timing for each hand
        self._fist_start_times = {}  # hand_id -> start_time

    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandData], np.ndarray]:
        """
        Process a camera frame to detect hands and gestures.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of (list of HandData objects, annotated frame)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process the frame
        results = self.hands.process(rgb_frame)

        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        hand_data_list = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract hand data
                hand_data = self._analyze_hand(hand_landmarks, handedness)
                hand_data_list.append(hand_data)

                # Draw gesture info on frame
                self._draw_gesture_info(annotated_frame, hand_data, frame.shape)

        return hand_data_list, annotated_frame

    def _analyze_hand(self, hand_landmarks, handedness) -> HandData:
        """
        Analyze hand landmarks to extract gesture data.

        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: MediaPipe handedness information

        Returns:
            HandData object with gesture information
        """
        # Calculate hand center (average of all landmarks)
        center_x = np.mean([lm.x for lm in hand_landmarks.landmark])
        center_y = np.mean([lm.y for lm in hand_landmarks.landmark])

        # Get handedness label
        hand_label = handedness.classification[0].label

        # Calculate hand openness
        openness = self._calculate_hand_openness(hand_landmarks)

        # Determine gesture type
        gesture = self._classify_gesture(openness)

        # Track fist duration
        fist_duration = self._update_fist_timing(hand_label, gesture)

        return HandData(
            center_x=center_x,
            center_y=center_y,
            gesture=gesture,
            handedness=hand_label,
            openness=openness,
            fist_duration=fist_duration
        )

    def _calculate_hand_openness(self, hand_landmarks) -> float:
        """
        Calculate how open the hand is (0.0 = closed, 1.0 = fully open).

        Method: Measure average 3D distance from fingertips to palm center.
        Uses 3D coordinates (x, y, z) for rotation-invariant recognition.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Openness value between 0.0 and 1.0
        """
        landmarks = hand_landmarks.landmark

        # Get palm center (average of wrist and base of middle finger) in 3D
        wrist = landmarks[self.WRIST]
        middle_base = landmarks[9]  # Base of middle finger
        palm_center_x = (wrist.x + middle_base.x) / 2
        palm_center_y = (wrist.y + middle_base.y) / 2
        palm_center_z = (wrist.z + middle_base.z) / 2

        # Calculate 3D distances from fingertips to palm center
        fingertip_indices = [
            self.THUMB_TIP,
            self.INDEX_TIP,
            self.MIDDLE_TIP,
            self.RING_TIP,
            self.PINKY_TIP
        ]

        distances = []
        for tip_idx in fingertip_indices:
            tip = landmarks[tip_idx]
            # Use 3D Euclidean distance for rotation invariance
            distance = np.sqrt(
                (tip.x - palm_center_x) ** 2 +
                (tip.y - palm_center_y) ** 2 +
                (tip.z - palm_center_z) ** 2
            )
            distances.append(distance)

        # Average distance
        avg_distance = np.mean(distances)

        # Normalize to 0-1 range (adjusted for 3D distances)
        # 3D distances are typically larger than 2D projections
        # Typical closed fist: ~0.06-0.12
        # Typical open hand: ~0.25-0.35
        openness = np.clip((avg_distance - 0.06) / 0.29, 0.0, 1.0)

        return float(openness)

    def _classify_gesture(self, openness: float) -> GestureType:
        """
        Classify gesture based on hand openness.

        Args:
            openness: Hand openness value (0.0-1.0)

        Returns:
            GestureType enum value
        """
        if openness < self.fist_threshold:
            return GestureType.CLOSED_FIST
        elif openness > self.fist_threshold + 0.1:  # Add hysteresis
            return GestureType.OPEN_PALM
        else:
            return GestureType.UNKNOWN

    def _update_fist_timing(self, hand_label: str, gesture: GestureType) -> float:
        """
        Track how long a fist gesture has been held.

        Args:
            hand_label: "Left" or "Right"
            gesture: Current gesture type

        Returns:
            Duration in seconds that fist has been held (0 if not a fist)
        """
        current_time = time.time()

        if gesture == GestureType.CLOSED_FIST:
            if hand_label not in self._fist_start_times:
                # Start tracking fist
                self._fist_start_times[hand_label] = current_time
                return 0.0
            else:
                # Continue tracking fist
                duration = current_time - self._fist_start_times[hand_label]
                return duration
        else:
            # Reset fist timer
            if hand_label in self._fist_start_times:
                del self._fist_start_times[hand_label]
            return 0.0

    def is_sustained_fist(self, hand_data: HandData) -> bool:
        """
        Check if a fist has been held for the required duration.

        Args:
            hand_data: HandData object

        Returns:
            True if fist has been held for fist_hold_duration seconds
        """
        return (
            hand_data.gesture == GestureType.CLOSED_FIST and
            hand_data.fist_duration >= self.fist_hold_duration
        )

    def _draw_gesture_info(self, frame: np.ndarray, hand_data: HandData, frame_shape: Tuple):
        """
        Draw gesture information on the frame.

        Args:
            frame: OpenCV frame to draw on
            hand_data: HandData object with gesture information
            frame_shape: Shape of the frame (height, width, channels)
        """
        height, width, _ = frame_shape

        # Convert normalized coordinates to pixels
        pixel_x = int(hand_data.center_x * width)
        pixel_y = int(hand_data.center_y * height)

        # Draw hand center
        cv2.circle(frame, (pixel_x, pixel_y), 10, (255, 0, 255), -1)

        # Prepare text
        gesture_text = hand_data.gesture.value.replace('_', ' ').title()
        info_text = f"{hand_data.handedness}: {gesture_text}"

        # Show fist timer if applicable
        if hand_data.gesture == GestureType.CLOSED_FIST:
            timer_text = f"Hold: {hand_data.fist_duration:.1f}s"
            info_text += f" - {timer_text}"

        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (pixel_x - 5, pixel_y - text_height - 15),
            (pixel_x + text_width + 5, pixel_y - 5),
            (0, 0, 0),
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            info_text,
            (pixel_x, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    def close(self):
        """Release resources."""
        self.hands.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
