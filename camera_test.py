"""
Camera and Hand Gesture Test Tool
Quick test to verify camera functionality and hand gesture recognition
"""

import cv2
import mediapipe as mp
import numpy as np

class CameraTest:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def test_camera_access(self):
        """Test camera access and functionality"""
        print("🎥 Testing Camera Access...")
        print("-" * 40)
        
        # Test multiple camera indices
        working_cameras = []
        for camera_index in range(4):
            print(f"Testing camera index {camera_index}...", end=" ")
            
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"✅ WORKING - {width}x{height}, FPS: {fps}")
                    working_cameras.append({
                        'index': camera_index,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                else:
                    print("❌ No frames")
            else:
                print("❌ Cannot open")
            cap.release()
        
        return working_cameras
    
    def test_hand_gesture_recognition(self):
        """Test hand gesture recognition with live camera feed"""
        print("\n🖐️  Testing Hand Gesture Recognition...")
        print("-" * 40)
        
        # Find working camera
        working_cameras = self.test_camera_access()
        if not working_cameras:
            print("❌ No working cameras found. Cannot test gesture recognition.")
            return False
        
        # Use first working camera
        camera_info = working_cameras[0]
        camera_index = camera_info['index']
        
        print(f"\n📹 Opening camera {camera_index} for gesture test...")
        print("Controls:")
        print("  - Move your hand in front of the camera")
        print("  - Press 'q' to quit")
        print("  - Press 's' to take a screenshot")
        print()
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        hand_detected_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Flip frame horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    hand_detected_count += 1
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Get key points for rotation calculation
                        wrist = hand_landmarks.landmark[0]
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        pinky_tip = hand_landmarks.landmark[20]
                        
                        # Calculate palm center
                        palm_x = (wrist.x + thumb_tip.x + pinky_tip.x) / 3
                        palm_y = (wrist.y + thumb_tip.y + pinky_tip.y) / 3
                        
                        # Calculate rotation values
                        rotation_x = (0.5 - palm_y) * 120
                        rotation_y = (palm_x - 0.5) * 180
                        
                        # Display rotation info
                        cv2.putText(frame, f"X-Rotation: {rotation_x:.1f}°", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Y-Rotation: {rotation_y:.1f}°", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display status
                detection_rate = (hand_detected_count / frame_count) * 100 if frame_count > 0 else 0
                cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", 
                          (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", 
                          (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Camera & Gesture Test', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f"gesture_test_screenshot_{frame_count}.png"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 Screenshot saved: {screenshot_name}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 Test Results:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Frames with hand detected: {hand_detected_count}")
            print(f"   Detection rate: {detection_rate:.1f}%")
            
            if detection_rate > 50:
                print("✅ Hand gesture recognition is working well!")
                return True
            else:
                print("⚠️  Hand gesture recognition may need improvement.")
                print("   Try better lighting or clearer hand visibility.")
                return False

def main():
    print("🧪 Camera and Hand Gesture Test Tool")
    print("=" * 50)
    
    tester = CameraTest()
    
    # Test camera access
    working_cameras = tester.test_camera_access()
    
    if not working_cameras:
        print("\n❌ No working cameras found!")
        print("\n💡 Troubleshooting tips:")
        print("   - Check camera permissions in System Preferences")
        print("   - Make sure no other applications are using the camera")
        print("   - Try reconnecting external cameras")
        return
    
    print(f"\n✅ Found {len(working_cameras)} working camera(s)")
    for cam in working_cameras:
        print(f"   Camera {cam['index']}: {cam['resolution']}, {cam['fps']} FPS")
    
    # Ask user if they want to test gesture recognition
    print("\nTest hand gesture recognition? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', '']:
            success = tester.test_hand_gesture_recognition()
            if success:
                print("\n🎉 All tests passed! Your camera and gesture recognition are ready.")
                print("   You can now run: python euler_spiral_test.py")
            else:
                print("\n⚠️  Some issues detected. Check camera positioning and lighting.")
        else:
            print("Skipping gesture recognition test.")
    except (EOFError, KeyboardInterrupt):
        print("\nTest cancelled by user.")

if __name__ == "__main__":
    main()