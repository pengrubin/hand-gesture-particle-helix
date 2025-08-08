# 🧬 Hand Gesture Controlled Particle Helix System

**Real-time hand gesture recognition with stunning 3D particle helix visualizations**

A pure Python implementation featuring 18 different wave shapes and 9 spiral structures, including DNA double helix, tornado spirals, and galaxy spirals - all controlled by your hand gestures!

![Demo](https://via.placeholder.com/800x400/1a1a1a/00ff00?text=Demo+Video+Coming+Soon)
<!-- Replace with actual demo GIF/video -->

## ✨ Features

### 🎵 NEW: Synchronized Multi-Track Audio Control
- **Real-time gesture-controlled audio mixing**
- **3 synchronized audio tracks** playing simultaneously
- **Seamless track switching** with volume control
- **No interruption** - maintain perfect timing when switching between tracks
- **Multi-gesture support** - combine tracks for rich musical compositions

| Audio Gesture | Track | Effect |
|---------------|-------|---------|
| ☝️ **1 Finger** → 🎻 **Violin** | Classical string melody |
| ✌️ **2 Fingers** → 🎸 **Lute** | Renaissance plucked strings |
| 🤟 **3 Fingers** → 🎹 **Organ** | Rich harmonic foundation |
| 🤘 **Multiple gestures** → 🎼 **Full Orchestra** | Layer multiple tracks |

### 🌀 9 Spiral Structures
- **DNA Double Helix** - Classic biological structure with connecting bridges
- **Triple Helix** - Three intertwined spirals
- **Tornado Helix** - Dynamic funnel-shaped spiral
- **Braided Lines** - Complex 3-strand braiding effect
- **Galaxy Spiral** - Logarithmic spiral arms
- **Spiral Tower** - Multi-layer decreasing spirals
- **And 3 more unique structures**

### 🌊 18 Wave Patterns
- Sine waves, cosine waves, double waves
- Heart curves, infinity curves
- Zigzag patterns, multiple parallel lines
- **Plus 9 new spiral patterns**

### 🎮 Dual-Layer Hand Controls

**Visual Effects (Particle Spirals):**
| Gesture | Visual Effect |
|---------|---------------|
| 👊 **Fist** → Tornado Spiral | Fast twisting, high turbulence |
| ☝️ **1 Finger** → DNA Double Helix | Classic biological structure |
| ✌️ **2 Fingers** → Triple Helix | Three-strand spiral |
| 🤟 **3 Fingers** → DNA with Bridges | Realistic DNA with base pairs |
| 🖖 **4 Fingers** → Braided Lines | Complex weaving pattern |
| ✋ **Open Hand** → Galaxy Spiral | Cosmic spiral arms |
| 🙌 **Both Hands** → Multi-Helix Tower | Multiple spirals, controlled by distance |

**Audio Control (Simultaneous):**
| Gesture | Audio Track | Description |
|---------|-------------|-------------|
| ☝️ **1 Finger** | 🎻 Violin | Classical string melody |
| ✌️ **2 Fingers** | 🎸 Lute | Renaissance plucked strings |
| 🤟 **3 Fingers** | 🎹 Organ | Rich harmonic foundation |
| 🤘 **Multiple** | 🎼 Mixed | Layer multiple instruments |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Webcam
- OpenGL-compatible graphics card
- Audio output device (speakers/headphones)
- MP3 audio files (for full audio experience)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hand-gesture-particle-helix.git
cd hand-gesture-particle-helix
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python run.py
```

### Test Mode (No Camera Required)
```bash
python run.py test
```

### 🎵 Audio Setup (Optional)
For the full audio experience, place MP3 files in the project directory:
- `Fugue in G Trio violin-Violin.mp3` (Track 1 - Violin)
- `Fugue in G Trio-Tenor_Lute.mp3` (Track 2 - Lute)  
- `Fugue in G Trio Organ-Organ.mp3` (Track 3 - Organ)

**Note**: The application will work without audio files, just without sound.

## 🎯 Usage

### Basic Controls
- **Mouse**: Drag to rotate 3D view
- **R**: Reset camera view
- **S**: Manually cycle through shapes
- **C**: Toggle camera window
- **M**: Toggle audio control on/off
- **I**: Toggle info display
- **W**: Toggle wireframe display
- **1-5**: Adjust particle count
- **ESC**: Exit

### Hand Gesture Controls
- **Gesture Strength**: Controls spiral radius and height
- **Hand Position**: Controls colors and twist speed
- **Both Hands Distance**: Controls helix count and connecting bridges
- **Left Hand Openness**: Controls transparency
- **Right Hand Position**: Controls color and twist parameters

## 🏗️ Project Structure

```
├── main_app.py                 # Main application entry point
├── gesture_detector.py         # MediaPipe hand tracking
├── render_engine.py            # OpenGL 3D rendering
├── particle_sphere_system.py   # Particle and helix systems
├── run.py                      # Smart launcher with dependency checks
├── test_shapes.py              # Shape testing utilities
├── test_gesture.py             # Gesture accuracy testing
├── requirements.txt            # Python dependencies
└── docs/                       # Documentation
    ├── PYTHON_README.md        # Detailed technical docs
    ├── CHANGELOG.md            # Version history
    └── BUGFIX_SUMMARY.md       # Recent fixes
```

## 🧪 Testing & Debugging

### Test Gesture Recognition
```bash
python test_gesture.py
```

### Test All Shapes
```bash
python run.py test
```

### Performance Monitoring
- Real-time FPS counter
- Particle count display
- Current shape indicator
- Hand detection status

## 🔧 Technical Details

### Core Technologies
- **MediaPipe**: Hand landmark detection
- **OpenGL**: 3D rendering engine
- **NumPy**: Mathematical computations
- **OpenCV**: Computer vision processing
- **Pygame**: Window management and input

### Key Algorithms
- **DNA Structure Simulation**: Spiral backbone with base pair bridges
- **Logarithmic Spirals**: Mathematical galaxy arm generation
- **Particle Physics**: Gravity, turbulence, and attraction forces
- **Real-time Deformation**: Dynamic helix parameter adjustment

### Performance Optimizations
- GPU-accelerated particle rendering
- Adaptive particle count (300-2000 particles)
- Efficient helix point generation
- 60fps smooth animation

## 🎨 Customization

### Adding New Shapes
1. Add shape function to `particle_sphere_system.py`
2. Update gesture mapping
3. Test with `test_shapes.py`

### Adjusting Parameters
Edit parameters in `particle_sphere_system.py`:
```python
self.params = {
    'wave_amplitude': 2.0,      # Helix size
    'wave_frequency': 1.0,      # Twist rate
    'twist_rate': 3.0,          # Spiral tightness
    'helix_count': 2,           # Number of helices
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_gesture.py
python test_shapes.py

# Check code style
python -m flake8 *.py
```

## 📊 System Requirements

### Minimum
- Python 3.7+
- 4GB RAM
- Integrated graphics
- 640x480 webcam

### Recommended
- Python 3.9+
- 8GB RAM
- Dedicated GPU
- 1080p webcam
- Good lighting conditions

## 🐛 Known Issues & Solutions

- **Hand detection inaccuracy**: Ensure good lighting and full hand visibility
- **Performance issues**: Reduce particle count with number keys 1-3
- **Camera permission**: Check system camera permissions
- **OpenGL errors**: Update graphics drivers

See [BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md) for recent fixes.

## 📜 Changelog

### v3.0.0 - Spiral Structure Upgrade 🧬
- Complete replacement of sphere rendering with helix structures
- 9 new spiral shapes including DNA, tornado, galaxy spirals
- Enhanced gesture mapping system
- Connection bridge system for DNA structures

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** team for excellent hand tracking
- **OpenGL** community for 3D graphics resources
- **TouchDesigner** for inspiration
- Mathematical spiral equations from various academic sources

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/hand-gesture-particle-helix/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hand-gesture-particle-helix/discussions)

---

**⭐ Star this repository if you found it interesting!**

*Experience the beauty of mathematical spirals controlled by your hands* 🌀✨