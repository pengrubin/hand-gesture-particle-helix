# Quick Start Guide - E Major Virtual Orchestra Conductor

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd /Users/hongweipeng/hand-gesture-particle-helix/E_Major/code
pip install -r requirements.txt
```

## Step 2: Verify Audio Files

All 11 audio files should be in `/Users/hongweipeng/hand-gesture-particle-helix/E_Major/`:

✓ Oboe_1_in_E.mp3
✓ Oboe_2_in_E.mp3
✓ Timpani_in_E.mp3
✓ Trumpet_in_C_1_in_E.mp3
✓ Trumpet_in_C_2_in_E.mp3
✓ Trumpet_in_C_3_in_E.mp3
✓ Violas_in_E.mp3
✓ Organ_in_E.mp3
✓ violin_in_E.mp3
✓ Violins_1_in_E.mp3
✓ Violins_2_in_E.mp3

## Step 3: Run the Application

```bash
python main_e_major.py
```

## Step 4: Start Conducting!

### Basic Workflow

1. **Camera window opens** with 9-zone grid overlay
2. **Show your hand** to the camera
3. **Open palm in Zone 5 (center)** → Starts playback
4. **Move to other zones with open palm** → Bring in instruments
5. **Make a fist and hold for 1 second** → Fade out instruments
6. **Press 'q'** → Quit

### Zone Map (as seen on screen)

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

### Simple Example

1. **Open palm in Zone 5** → Orchestra starts (all at volume 0)
2. **Open palm in Zone 7** → Violins fade in
3. **Open palm in Zone 3** → Trumpets fade in
4. **Open palm in Zone 2** → Timpani fades in
5. **Close fist in Zone 7 for 1 second** → Violins fade out
6. **Close fist in Zone 5 for 1 second** → Global pause

## Keyboard Shortcuts

- `q` - Quit application
- `p` - Manual play/pause toggle
- `s` - Stop all and reset

## Troubleshooting

### "Camera not found"
- Try changing `CAMERA_INDEX` in `config.py` (try 0, 1, or 2)

### "Hand not detected"
- Improve lighting
- Move closer to camera
- Ensure hand is clearly visible

### "No audio"
- Check audio file paths in `config.py`
- Verify all 11 MP3 files are present

### "Slow performance"
- Lower camera resolution in `config.py`:
  ```python
  CAMERA_WIDTH = 640
  CAMERA_HEIGHT = 480
  ```

## Tips for Best Experience

1. **Good lighting is essential** for accurate hand tracking
2. **Use deliberate gestures** - slow, clear movements work best
3. **Hold fist for full 1 second** to trigger volume decrease
4. **Start with Zone 5** to begin playback
5. **Both hands work!** Control two zones simultaneously

## What to Expect

- **Green grid overlay** showing 9 zones
- **Hand landmarks** drawn in real-time
- **Gesture labels** showing "Open Palm" or "Closed Fist"
- **Volume bars** above each zone showing current levels
- **FPS counter** in top-left corner
- **Yellow highlight** on active zones

## Need Help?

Check the full README.md for:
- Detailed architecture documentation
- Configuration options
- Advanced troubleshooting
- Development information

---

**Ready to conduct? Run `python main_e_major.py` and wave your hands!**
