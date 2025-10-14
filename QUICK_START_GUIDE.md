# 🚀 Quick Start Guide

Get your parametric equation gesture control system running in 10 minutes!

## ⚡ Fast Track Installation

### System Requirements
- **Python 3.7+** (3.9+ recommended)
- **Webcam** (any USB/built-in camera)
- **OpenGL-compatible graphics card**
- **4GB RAM minimum** (8GB recommended)

### Platform-Specific Notes
- **macOS**: Requires camera permissions in System Preferences
- **Windows**: Ensure camera privacy settings allow Python access
- **Linux**: May need additional OpenGL packages

## 🔧 Installation (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/hand-gesture-particle-helix.git
cd hand-gesture-particle-helix

# 2. Install dependencies (auto-detects missing packages)
python run.py

# The launcher will automatically:
# - Check dependencies
# - Install missing packages
# - Test camera access
# - Start the application
```

### Manual Installation (if needed)
```bash
pip install opencv-python mediapipe numpy pygame PyOpenGL
```

## 🎮 First Launch

### Quick Start (No Camera)
```bash
python run.py test  # Test all spiral patterns without camera
```

### Full Application
```bash
python run.py  # Launch complete gesture control system
```

## 📋 Setup Verification Checklist

✅ **Camera Access**: Hand detection window appears  
✅ **3D Rendering**: Particle spirals visible in main window  
✅ **Performance**: 30+ FPS (shown in info panel)  
✅ **Gesture Response**: Spirals change with hand gestures  

### Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| "Camera not found" | Check camera permissions, try different camera IDs |
| Low FPS (<15) | Press keys 1-3 to reduce particles, close other apps |
| No hand detection | Ensure good lighting, full hand in view |
| Import errors | Run `pip install -r requirements.txt` |

## 🎯 Basic Controls (30 seconds to learn)

### Essential Keyboard Shortcuts
| Key | Action |
|-----|--------|
| **C** | Toggle camera window |
| **R** | Reset view and audio |
| **ESC** | Exit application |
| **1-5** | Adjust particle count (20%-100%) |
| **I** | Toggle information display |

### Core Hand Gestures
| Gesture | Visual Result | Audio Effect |
|---------|---------------|--------------|
| **👊 Fist** | Tornado Spiral | Audio pause |
| **☝️ 1 Finger** | DNA Double Helix | Violin track |
| **✌️ 2 Fingers** | Triple Helix | Lute track |
| **🤟 3 Fingers** | DNA with Bridges | Organ track |
| **✋ Open Hand** | Galaxy Spiral | Full Orchestra |

## 🎵 Audio Setup (Optional, 2 minutes)

For the complete audio experience, add these MP3 files to your project directory:

```
project-folder/
├── Fugue in G Trio violin-Violin.mp3
├── Fugue in G Trio-Tenor_Lute.mp3
└── Fugue in G Trio Organ-Organ.mp3
```

**Audio Controls:**
- **P**: Manual pause/resume
- **T**: Toggle restart strategy
- **M**: Disable/enable audio control

> **Note**: The system works perfectly without audio files - you'll just see "Audio: DISABLED" in the info panel.

## ⚡ Performance Optimization

### Immediate Performance Boost
```bash
# Reduce particles for better performance
python run.py
# Then press: 1 (20%), 2 (40%), or 3 (60%) particles
```

### System Optimization Tips
1. **Close unnecessary applications**
2. **Ensure good lighting** for accurate hand detection
3. **Use recommended camera resolution** (640x480)
4. **Update graphics drivers** for OpenGL performance

## 🧬 Understanding the System (2 minutes)

### What You're Seeing
- **Spiral Patterns**: Mathematical helix structures (DNA, tornado, galaxy)
- **Real-time Control**: Your hand gestures directly control spiral parameters
- **Audio Visualization**: Music frequency analysis affects particle size
- **3D Interaction**: Mouse controls viewing angle

### Parameter Control Mapping
- **Hand Gestures** → Spiral shape/structure
- **Hand Position** → Color and twist speed  
- **Gesture Strength** → Spiral radius and height
- **Both Hands Distance** → Multiple helix count

## 🎯 Next Steps

### Explore Advanced Features
1. **Try all gesture combinations** - each creates unique patterns
2. **Experiment with audio control** - layer different instruments
3. **Test performance settings** - find your optimal particle count
4. **Read the Parameter Reference** - understand the mathematics

### Advanced Usage
- **Parameter Reference**: Complete mathematical documentation
- **TouchDesigner Integration**: Professional visual performance setup
- **API Reference**: Extend the system with custom code

## 🆘 Need Help?

### Quick Diagnostic Commands
```bash
python run.py --help     # Show all options
python test_gesture.py   # Test gesture recognition only
python debug_camera.py   # Camera troubleshooting
```

### Common Solutions
- **Gesture not detected**: Improve lighting, clear background
- **Audio not working**: Check file names and format (MP3 only)
- **Low performance**: Reduce particle count, update drivers
- **Camera permission**: Grant camera access in system settings

### Getting Support
- Check the **README.md** for comprehensive information
- Review **BUGFIX_SUMMARY.md** for known issues
- Create GitHub issue with error details and system info

---

**🎉 Congratulations! You're now running a real-time hand gesture controlled parametric visualization system!**

*Next: Read the [Parameter Reference](PARAMETER_REFERENCE.md) to understand the mathematical foundation.*