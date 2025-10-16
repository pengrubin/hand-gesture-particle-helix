"""
BWV_29_in_D Body Pose Audio Control System Main Program
Simplified version - Focus on audio control, remove particle system

Core features:
- Use MediaPipe Pose to detect body pose
- Recognize violin playing gestures
- Control 9 audio tracks playback and volume based on detection results
- Real-time camera window display and skeleton visualization
"""

import cv2
import pygame
import time
import os
import sys
import platform
import numpy as np

# Add parent directory to path (for importing parent directory modules)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root)

# Import local modules (need to create these modules)
try:
    from pose_body_detector import PoseBodyDetector
    from e_major_audio_controller import EMajorAudioController
    print("✓ Successfully imported core modules")
except ImportError as e:
    print(f"✗ Module import failed: {e}")
    print("Please ensure pose_body_detector.py and e_major_audio_controller.py are in the same directory")
    sys.exit(1)


class EMajorApp:
    """BWV_29_in_D Body Pose Audio Control Application Main Class"""

    def __init__(self):
        """Initialize BWV_29_in_D application"""
        print("\n" + "="*60)
        print("=== BWV_29_in_D Body Pose Audio Control System ===")
        print("="*60)
        print("\nSystem features:")
        print("• Detect person -> Play orchestra (violin muted)")
        print("• Detect violin gesture -> Enhance violin section")
        print("• No person detected -> Auto pause all tracks\n")

        # Display platform information
        system = platform.system()
        machine = platform.machine()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        if system == "Darwin":
            processor_type = "Apple Silicon" if machine == "arm64" else "Intel"
        else:
            processor_type = machine

        print(f"Running platform: {system} {processor_type}")
        print(f"Python version: {python_version}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Pygame version: {pygame.version.ver}\n")

        # Initialize pygame (for window management and audio)
        print("Initializing pygame system...")
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        print("✓ Pygame system initialized\n")

        # Initialize pose detector
        print("Initializing pose detector...")
        try:
            self.pose_detector = PoseBodyDetector()
            print("✓ Pose detector initialized")
        except Exception as e:
            print(f"✗ Pose detector initialization failed: {e}")
            raise

        # Initialize audio controller
        print("\nInitializing audio controller...")
        try:
            self.audio_controller = EMajorAudioController()
            print("✓ Audio controller initialized")
        except Exception as e:
            print(f"✗ Audio controller initialization failed: {e}")
            raise

        # Running status flags
        self.is_running = True
        self.show_camera = True  # Whether to show camera window
        self.show_info = True    # Whether to show info overlay
        self.paused = False      # Manual pause flag

        # Performance monitoring
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        # Camera window name
        self.window_name = 'BWV_29_in_D - Body Pose Audio Control'

        print("\n" + "="*60)
        print("Initialization complete! Ready to start application...")
        print("="*60 + "\n")

    def start(self):
        """Start application"""
        print("\nStarting application...")

        try:
            # Start camera
            print("\n1. Starting camera...")
            try:
                if not self.pose_detector.start_camera(0):
                    print("✗ Camera startup failed")

                    # Provide platform-specific solutions
                    system = platform.system()
                    if system == "Darwin":  # macOS
                        print("\n🔧 macOS Camera Permission Solution:")
                        print("  1. System Preferences > Security & Privacy > Privacy > Camera")
                        print("  2. Ensure Terminal or your Python IDE has camera permission")
                        print("  3. Restart terminal or IDE")
                        print("  4. Ensure no other application is using the camera")
                    elif system == "Windows":
                        print("\n🔧 Windows Camera Permission Solution:")
                        print("  1. Settings > Privacy > Camera")
                        print("  2. Ensure application has camera permission")
                        print("  3. Check camera status in Device Manager")

                    return

                print("✓ Camera started successfully")
            except Exception as camera_error:
                print(f"✗ Camera startup exception: {camera_error}")
                import traceback
                traceback.print_exc()
                return

            # Initialize audio system
            print("\n2. Initializing audio system...")
            try:
                if not self.audio_controller.initialize():
                    print("✗ Audio initialization failed, please check audio file paths")
                    return
                print("✓ Audio system ready (9 tracks loaded)")
            except Exception as audio_error:
                print(f"✗ Audio initialization exception: {audio_error}")
                import traceback
                traceback.print_exc()
                return

            # Wait for system stabilization
            print("\n3. System initializing...")
            time.sleep(1.0)

            # Display control instructions
            self.print_control_instructions()

            # Enter main loop
            print("\n" + "="*60)
            print("Application started successfully! Running main loop...")
            print("="*60 + "\n")

            self.run_main_loop()

        except KeyboardInterrupt:
            print("\n\nUser pressed Ctrl+C, preparing to exit...")
        except Exception as e:
            print(f"\n✗ Startup error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def print_control_instructions(self):
        """Display control instructions"""
        print("\n" + "="*60)
        print("=== Control Instructions ===")
        print("="*60)

        print("\n【Keyboard Controls】")
        print("  C key - Toggle camera display on/off")
        print("  I key - Toggle info display on/off")
        print("  P key - Manual pause/resume audio")
        print("  R key - Reset audio to start position")
        print("  ESC key - Exit application")

        print("\n【Audio Control Logic】")
        print("  • No person detected -> Pause all tracks")
        print("  • Person detected (no violin gesture) -> Play orchestra (violin muted)")
        print("  • Person detected + violin gesture -> Play all (violin volume 100%)")

        print("\n【Violin Gesture Recognition】")
        print("  • Left hand raised (above shoulder)")
        print("  • Right hand raised (above shoulder)")
        print("  • Both arms in violin playing posture")

        print("\n【9 Track List】")
        track_names = [
            "1: Continuo", "2: Oboe I", "3: Organo obligato", "4: Timpani",
            "5: Tromba I", "6: Tromba II", "7: Tromba III",
            "8: Viola", "9: Violins (solo)"
        ]
        for name in track_names:
            print(f"  • Track {name}")

        print("\n" + "="*60 + "\n")

    def run_main_loop(self):
        """Main loop"""
        last_time = time.time()
        frame_count = 0

        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            frame_count += 1

            # Handle keyboard events
            self.handle_events()

            # Get camera frame
            frame = self.pose_detector.get_current_frame()

            if frame is not None:
                # Process pose detection (with skeleton visualization)
                processed_frame = self.pose_detector.process_frame(frame, show_skeleton=True)

                # Get pose data
                pose_data = self.pose_detector.get_pose_data()

                # Update audio control (unless manually paused)
                if not self.paused:
                    self.audio_controller.update_from_pose(
                        person_detected=pose_data['person_detected'],
                        violin_gesture_detected=pose_data['violin_gesture_detected']
                    )

                # Display camera window
                if self.show_camera:
                    self.show_camera_window(processed_frame, pose_data)

            # Update FPS
            self.update_fps()

            # Limit framerate to 30fps (reduce CPU usage)
            target_fps = 30.0
            frame_time = 1.0 / target_fps
            if dt < frame_time:
                time.sleep(frame_time - dt)

            # Display performance info every 100 frames (optional)
            if frame_count % 100 == 0:
                print(f"[Performance] FPS: {self.current_fps:.1f} | Running time: {current_time - self.fps_timer:.1f}s")

    def handle_events(self):
        """Handle keyboard events"""
        if not self.show_camera:
            return

        # OpenCV window events (only process when window is displayed)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # No key pressed
            return

        # ESC key - Exit
        if key == 27:
            print("\n[User action] ESC key pressed, preparing to exit...")
            self.is_running = False

        # C key - Toggle camera display
        elif key == ord('c') or key == ord('C'):
            self.show_camera = not self.show_camera
            if not self.show_camera:
                cv2.destroyAllWindows()
            status = "On" if self.show_camera else "Off"
            print(f"[User action] Camera display: {status}")

        # I key - Toggle info display
        elif key == ord('i') or key == ord('I'):
            self.show_info = not self.show_info
            status = "On" if self.show_info else "Off"
            print(f"[User action] Info display: {status}")

        # P key - Manual pause/resume audio
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            if self.paused:
                self.audio_controller.pause_all()
                print("[User action] Audio manually paused")
            else:
                self.audio_controller.resume_all()
                print("[User action] Audio manually resumed")

        # R key - Reset audio position
        elif key == ord('r') or key == ord('R'):
            self.audio_controller.reset_position()
            print("[User action] Audio position reset to start")

    def show_camera_window(self, frame, pose_data):
        """Display camera window"""
        if frame is None:
            return

        # Add info overlay
        if self.show_info:
            self.add_info_overlay(frame, pose_data)

        # Display window
        cv2.imshow(self.window_name, frame)

    def add_info_overlay(self, frame, pose_data):
        """Add info overlay to camera view"""
        h, w = frame.shape[:2]

        # Info panel position and size
        panel_width = 360
        panel_x = w - panel_width - 10
        panel_y = 10

        # Dynamically calculate panel height
        base_height = 380
        panel_height = base_height

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (w - 10, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (w - 10, panel_y + panel_height), (255, 255, 255), 2)

        # Prepare info text
        info_lines = []

        # System performance
        info_lines.append(f"FPS: {self.current_fps:.1f}")
        info_lines.append("")

        # Detection status
        info_lines.append("=== Detection Status ===")
        person_status = "✓ Yes" if pose_data['person_detected'] else "✗ No"
        violin_status = "✓ Yes" if pose_data['violin_gesture_detected'] else "✗ No"
        info_lines.append(f"Person: {person_status}")
        info_lines.append(f"Violin Gesture: {violin_status}")
        info_lines.append(f"Confidence: {pose_data['pose_confidence']:.2f}")
        info_lines.append("")

        # Audio status
        info_lines.append("=== Audio Status ===")
        audio_status = self.audio_controller.get_status_info()

        current_state = audio_status.get('current_state', 'Unknown')
        info_lines.append(f"State: {current_state}")

        playback_pos = audio_status.get('playback_position', 0.0)
        info_lines.append(f"Position: {playback_pos:.1f}s")

        if self.paused:
            info_lines.append("Mode: Manual Pause")
        else:
            info_lines.append("Mode: Auto Control")

        info_lines.append("")

        # Track volumes (with visualization progress bars)
        info_lines.append("=== Track Volumes ===")

        track_names = {
            1: "Continuo", 2: "Oboe_I", 3: "Organo", 4: "Timpani",
            5: "Tromba_I", 6: "Tromba_II", 7: "Tromba_III", 8: "Viola",
            9: "Violins"
        }

        volumes = audio_status.get('volumes', {})

        for track_id in sorted(track_names.keys()):
            name = track_names[track_id]
            vol = volumes.get(track_id, 0.0)
            vol_percent = int(vol * 100)

            # Create progress bar
            bar_length = 10
            filled = int(bar_length * vol)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Special marker for violin solo
            marker = " *" if track_id == 9 else ""
            info_lines.append(f"{name:10s}: {bar} {vol_percent:3d}%{marker}")

        # Draw text
        line_height = 20
        text_color = (200, 200, 200)  # Light gray

        for i, line in enumerate(info_lines):
            y_pos = panel_y + 25 + i * line_height

            # Set color based on content
            if "✓" in line:
                color = (0, 255, 0)  # Green
            elif "✗" in line:
                color = (0, 0, 255)  # Red
            elif "===" in line:
                color = (0, 255, 255)  # Yellow (title)
            elif "*" in line:
                color = (255, 100, 255)  # Purple (violin solo)
            else:
                color = text_color

            cv2.putText(frame, line, (panel_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time

    def cleanup(self):
        """Cleanup resources"""
        print("\n" + "="*60)
        print("Cleaning up resources...")
        print("="*60 + "\n")

        # Cleanup audio system
        try:
            self.audio_controller.cleanup()
            print("✓ Audio system cleaned up")
        except Exception as e:
            print(f"⚠ Audio cleanup warning: {e}")

        # Stop camera
        try:
            self.pose_detector.stop_camera()
            print("✓ Camera stopped")
        except Exception as e:
            print(f"⚠ Camera stop warning: {e}")

        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            print("✓ Windows closed")
        except Exception as e:
            print(f"⚠ Window close warning: {e}")

        # Quit pygame
        try:
            pygame.quit()
            print("✓ Pygame exited")
        except Exception as e:
            print(f"⚠ Pygame exit warning: {e}")

        print("\n" + "="*60)
        print("Application completely exited, thank you for using!")
        print("="*60 + "\n")


def main():
    """Main function entry point"""
    print("\n" + "="*70)
    print(" "*10 + "BWV_29_in_D Body Pose Audio Control System")
    print(" "*20 + "Version 1.0 - 2024")
    print("="*70)

    try:
        # Create and start application
        app = EMajorApp()
        app.start()

    except KeyboardInterrupt:
        print("\n\nUser interrupted (Ctrl+C), exiting...")

    except Exception as e:
        print(f"\n✗ Application error: {e}")
        import traceback
        print("\nComplete error stack:")
        traceback.print_exc()

    finally:
        print("\nProgram ended.")


if __name__ == "__main__":
    main()
