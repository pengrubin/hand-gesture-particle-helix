# E Major Virtual Orchestra Conductor

An interactive virtual orchestra conductor system that uses hand gestures to control multi-track audio playback through spatial zone mapping. Conduct an 11-piece orchestra using MediaPipe hand tracking and intuitive gesture controls.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands for accurate hand detection and gesture classification
- **9-Zone Spatial Control**: Camera view divided into 9 zones (like a phone keypad) for intuitive control
- **Multi-track Audio Playback**: Synchronous playback of 11 orchestral tracks with independent volume control
- **Gesture-based Control**:
  - **Open Palm**: Increase volume or play
  - **Closed Fist (1 second)**: Decrease volume or pause
- **Smooth Volume Transitions**: Gradual fade in/out for professional audio control
- **Global Controls**: Zone 5 provides play/pause control for all tracks simultaneously

## Zone Layout and Instrument Mapping

```
┌─────────┬─────────┬─────────┐
│    1    │    2    │    3    │
│  Oboes  │ Timpani │Trumpets │
├─────────┼─────────┼─────────┤
│    4    │    5    │    6    │
│ Violas  │ GLOBAL  │  Organ  │
├─────────┼─────────┼─────────┤
│    7    │    8    │    9    │
│ Violins │Reserved │Reserved │
└─────────┴─────────┴─────────┘
```

### Zone-to-Instrument Mapping

| Zone | Instruments | Audio Files |
|------|-------------|-------------|
| 1 | Oboes | Oboe_1_in_E.mp3, Oboe_2_in_E.mp3 |
| 2 | Timpani | Timpani_in_E.mp3 |
| 3 | Trumpets | Trumpet_in_C_1_in_E.mp3, Trumpet_in_C_2_in_E.mp3, Trumpet_in_C_3_in_E.mp3 |
| 4 | Violas | Violas_in_E.mp3 |
| 5 | **GLOBAL** | Global play/pause control |
| 6 | Organ | Organ_in_E.mp3 |
| 7 | Violins | violin_in_E.mp3, Violins_1_in_E.mp3, Violins_2_in_E.mp3 |
| 8 | Reserved | (not used) |
| 9 | Reserved | (not used) |

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- macOS, Linux, or Windows

### Dependencies

Install required packages:

```bash
pip install opencv-python mediapipe pydub simpleaudio numpy
```

### Audio Files Setup

Ensure all 11 audio files are in the `E_Major/` directory:

- Oboe_1_in_E.mp3
- Oboe_2_in_E.mp3
- Timpani_in_E.mp3
- Trumpet_in_C_1_in_E.mp3
- Trumpet_in_C_2_in_E.mp3
- Trumpet_in_C_3_in_E.mp3
- Violas_in_E.mp3
- Organ_in_E.mp3
- violin_in_E.mp3
- Violins_1_in_E.mp3
- Violins_2_in_E.mp3

## Usage

### Running the Application

```bash
cd /Users/hongweipeng/hand-gesture-particle-helix/E_Major/code
python main_e_major.py
```

### Controls

#### Hand Gestures

1. **Open Palm (fingers spread)**
   - In zones 1-4, 6-7: Increase volume of zone instruments to maximum
   - In zone 5: Play or resume all tracks

2. **Closed Fist (held for 1 second)**
   - In zones 1-4, 6-7: Decrease volume of zone instruments to minimum
   - In zone 5: Pause all tracks

#### Keyboard Shortcuts

- `q`: Quit application
- `p`: Manual play/pause toggle
- `s`: Stop all tracks and reset volumes

### How to Conduct

1. **Start the application** - Camera view will open with 9-zone grid overlay
2. **Position your hand(s)** in front of the camera
3. **Open palm** in a zone to bring in those instruments
4. **Move between zones** to control different instrument sections
5. **Hold a fist for 1 second** in a zone to fade out those instruments
6. **Use zone 5 (center)** for global play/pause control
7. **Conduct expressively** - the system tracks both left and right hands

### Tips for Best Performance

- Ensure good lighting for accurate hand detection
- Keep hands visible and within camera frame
- Use deliberate gestures for consistent recognition
- Allow 1 second for fist gestures to trigger actions
- Start with zone 5 open palm to begin playback
- Gradually bring in instruments by moving through zones

## Architecture

### Project Structure

```
E_Major/
├── code/
│   ├── main_e_major.py              # Main application
│   ├── hand_gesture_detector.py     # MediaPipe hand tracking
│   ├── grid_zone_detector.py        # 9-zone spatial mapping
│   ├── e_major_audio_controller.py  # Multi-track audio playback
│   ├── config.py                    # Configuration constants
│   └── README.md                    # This file
├── Oboe_1_in_E.mp3
├── Oboe_2_in_E.mp3
└── ... (other audio files)
```

### Component Overview

#### `hand_gesture_detector.py`
- Uses MediaPipe Hands for landmark detection
- Classifies gestures (open palm vs closed fist)
- Tracks fist duration for sustained gesture detection
- Calculates hand openness based on fingertip-to-palm distances

#### `grid_zone_detector.py`
- Divides camera frame into 3x3 grid (9 zones)
- Maps hand position to zone numbers (1-9)
- Provides zone boundaries and visualization

#### `e_major_audio_controller.py`
- Manages synchronized multi-track audio playback
- Independent volume control per track using pydub
- Smooth volume transitions via background thread
- Global play/pause functionality

#### `config.py`
- Centralized configuration for all system parameters
- Zone-to-instrument mappings
- Audio file paths and settings
- UI display options

#### `main_e_major.py`
- Integrates all components
- Main event loop for camera processing
- Gesture-to-audio control logic
- UI rendering and visualization

## Configuration

Edit `config.py` to customize:

- Camera resolution and FPS
- Gesture detection thresholds
- Volume transition speed
- Fist hold duration
- UI colors and display options
- Zone mappings

### Key Configuration Parameters

```python
# Gesture detection
FIST_OPENNESS_THRESHOLD = 0.3  # Lower = more sensitive fist detection
FIST_HOLD_DURATION = 1.0       # Seconds to hold fist for trigger

# Audio control
VOLUME_TRANSITION_SPEED = 2.0  # Volume change speed (0-1 per second)
VOLUME_UPDATE_RATE = 0.05      # Volume update interval (20 Hz)

# Camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
```

## Troubleshooting

### Camera Issues

- **Camera not detected**: Check camera index in `config.py` (try 0, 1, or 2)
- **Low frame rate**: Reduce camera resolution or disable volume bars

### Hand Detection Issues

- **Hands not detected**: Improve lighting, ensure hands are clearly visible
- **False detections**: Increase `MIN_DETECTION_CONFIDENCE` in config
- **Jittery tracking**: Increase `MIN_TRACKING_CONFIDENCE` in config

### Audio Issues

- **No audio playback**: Verify all audio files are in correct directory
- **Choppy audio**: Increase `VOLUME_UPDATE_RATE` or reduce `VOLUME_TRANSITION_SPEED`
- **Volume not changing**: Check track names match config mappings

### Performance Issues

- **Lag or slowdown**: Reduce camera resolution, disable UI elements
- **High CPU usage**: Increase `VOLUME_UPDATE_RATE` to reduce update frequency

## Technical Details

### Gesture Recognition Algorithm

1. **Hand Detection**: MediaPipe Hands detects 21 landmarks per hand
2. **Hand Center**: Calculated as average of all landmark positions
3. **Openness Calculation**: Average distance from fingertips to palm center
4. **Gesture Classification**:
   - Openness < 0.3 → Closed Fist
   - Openness > 0.4 → Open Palm
   - Between → Unknown (hysteresis for stability)
5. **Fist Duration Tracking**: Timer starts when fist detected, resets on gesture change

### Audio Synchronization

- All tracks start simultaneously using threading
- Playback position tracked using timestamps
- Volume changes applied via audio segment modification
- Tracks restarted with new volume at current playback position
- Smooth transitions achieved through incremental volume adjustments

### Performance Optimizations

- Background thread for volume control (decoupled from video processing)
- Frame mirroring for intuitive left-right mapping
- Efficient zone calculation using grid division
- Minimal audio processing overhead using pydub

## Development

### Code Quality

- Type hints for all function parameters
- Comprehensive docstrings
- PEP 8 compliant code formatting
- Modular design with clear separation of concerns
- Error handling and logging throughout

### Testing Recommendations

1. Test with various hand sizes and positions
2. Verify gesture recognition under different lighting
3. Test sustained fist detection timing accuracy
4. Verify audio synchronization across all tracks
5. Test with both single-hand and two-hand control

## License

This project is part of the hand-gesture-particle-helix repository.

## Credits

- **MediaPipe**: Google's MediaPipe Hands for hand tracking
- **pydub**: Audio processing and playback
- **OpenCV**: Video capture and visualization

## Future Enhancements

Potential improvements:

- [ ] Add more gestures (e.g., pinch for fine volume control)
- [ ] Tempo control via hand movement speed
- [ ] Recording and playback of conducting sessions
- [ ] Support for custom audio track assignments
- [ ] Web-based interface for remote conducting
- [ ] Multi-camera support for larger conducting spaces
- [ ] Visual feedback for beat tracking
- [ ] MIDI output for DAW integration

---

**Enjoy conducting your virtual orchestra!**
