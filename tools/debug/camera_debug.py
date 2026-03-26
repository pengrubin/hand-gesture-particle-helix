"""
Debug camera initialization to identify the issue
"""

import cv2
import mediapipe as mp

def debug_camera_init():
    print("🔍 Debug Camera Initialization")
    print("=" * 40)
    
    try:
        print("Step 1: Testing basic OpenCV camera access...")
        cap = cv2.VideoCapture(0)
        
        print(f"Camera opened: {cap.isOpened()}")
        
        if cap.isOpened():
            print("Step 2: Testing frame reading...")
            ret, frame = cap.read()
            print(f"Frame read result: {ret}")
            if ret and frame is not None:
                print(f"Frame shape: {frame.shape}")
                print("✅ Basic camera access works!")
            else:
                print("❌ Cannot read frames")
        else:
            print("❌ Cannot open camera")
        
        cap.release()
        
        print("\nStep 3: Testing MediaPipe initialization...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_camera_init()