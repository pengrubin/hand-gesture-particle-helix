"""
欧拉公式螺旋测试
基于复数公式的3D螺旋轨迹可视化

公式: z(θ) = 11e^{i(11θ)} + 14sin(10θ)e^{iθ} + 13e^{iθ}

学术署名:
- Animation: Patrick Georges, University of Ottawa
- Original Idea: Chirag Dudhat

这是一个基于欧拉公式的复杂螺旋轨迹可视化，结合手势控制实现3D交互体验。
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
import cv2
import mediapipe as mp
import sys

# Ensure output is not buffered
sys.stdout.flush()

class EulerSpiralVisualizer:
    def __init__(self, width=1200, height=800):
        print("Initializing Euler Spiral Visualizer...", flush=True)
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        
        # Initialize pygame and OpenGL
        print("Setting up pygame window...", flush=True)
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Euler Formula Spiral - Hand Gesture Control")
        
        # OpenGL设置
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        
        # 设置投影矩阵 - 使用正交投影避免透视畸变
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # 正交投影：避免远近物体大小差异
        view_size = 8.0  # 视窗大小
        aspect = width / height
        if aspect >= 1.0:  # 宽屏
            glOrtho(-view_size * aspect, view_size * aspect, -view_size, view_size, -50.0, 50.0)
        else:  # 高屏
            glOrtho(-view_size, view_size, -view_size / aspect, view_size / aspect, -50.0, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Spiral parameters - balanced for better visualization
        self.time_step = 0.0
        self.spiral_points = []  # Store trajectory points
        self.max_points = 10000  # Good amount of trajectory points
        self.time_speed = 0.001  # Half speed for much smoother curves
        self.z_scale = 1.0       # Default Z-axis scaling - can be increased to 3
        
        # Coordinate axis rotation parameters 
        self.gesture_rotation_x = 0  # Hand gesture rotation
        self.gesture_rotation_y = 0
        self.gesture_rotation_z = 0
        self.mouse_rotation_x = 0    # Mouse rotation (additive)
        self.mouse_rotation_y = 0
        
        # Camera display control
        self.show_camera_view = False  # Will be enabled if camera is active
        
        # Camera following parameters for XYZ tracking
        self.camera_x_target = 0.0
        self.camera_y_target = 0.0
        self.camera_z_target = -8.0
        self.camera_x_current = 0.0
        self.camera_y_current = 0.0
        self.camera_z_current = -8.0
        self.follow_speed = 0.3   # Much faster camera following speed
        
        # Hand gesture recognition initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera with proper error handling
        print("🎥 Initializing camera...")
        self.init_camera()
        
        # Mouse control state
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
    
    def init_camera(self):
        """Initialize camera with proper error handling"""
        self.camera_active = False
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    # Configure camera
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Test again after configuration
                    ret2, frame2 = self.cap.read()
                    if ret2 and frame2 is not None:
                        self.camera_active = True
                        self.show_camera_view = True  # Enable camera view display
                        print(f"✅ Camera initialized successfully!")
                        print(f"   Resolution: {frame2.shape[1]}x{frame2.shape[0]}")
                        print("   Hand gesture control is ready!")
                        print("   Camera view window will open")
                        return True
                        
                print("⚠️  Camera opened but unstable frame reading")
            else:
                print("⚠️  Cannot open camera device") 
                
        except Exception as e:
            print(f"⚠️  Camera initialization error: {e}")
        
        # Cleanup on failure
        if hasattr(self, 'cap'):
            self.cap.release()
        
        print("🖱️  Using mouse control (drag to rotate)")
        print("💡 Press 'C' during runtime to retry camera connection")
        return False
    
    def reconnect_camera(self):
        """Try to reconnect camera during runtime"""
        print("\n🔄 Attempting to reconnect camera...")
        
        # Close existing camera if open
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Use the same initialization method
        return self.init_camera()
        
    def euler_spiral_formula(self, theta):
        """
        欧拉公式螺旋: 11*exp(1j*(11*theta)) + 14*sin(10*theta)*exp(1j*theta) + 13*exp(1j*theta)
        返回复数结果
        """
        term1 = 11 * np.exp(1j * (11 * theta))
        term2 = 14 * np.sin(10 * theta) * np.exp(1j * theta)  
        term3 = 13 * np.exp(1j * theta)
        
        return term1 + term2 + term3
    
    def generate_spiral_point(self, time_val):
        """根据时间值生成螺旋上的点"""
        # 使用时间作为theta参数
        theta = time_val * 2 * np.pi
        
        # 计算复数结果
        complex_result = self.euler_spiral_formula(theta)
        
        # 提取实部和虚部作为x, y坐标
        x = complex_result.real * 0.1  # 缩放以适应屏幕
        y = complex_result.imag * 0.1
        z = time_val * self.z_scale * 3.0   # 时间作为z轴，加速Z轴移动
        
        return (x, y, z)
    
    def update_hand_gesture(self):
        """Update hand gesture - coordinate system follows hand like a glove"""
        if not self.camera_active:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠️  Warning: Failed to read camera frame")
                return
            
            # Flip frame horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)
            
            # Convert color space for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame for hand detection
            results = self.hands.process(rgb_frame)
            
            # Display camera view with hand landmarks if enabled
            if self.show_camera_view:
                display_frame = frame.copy()
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Add text overlay with direction guides
                cv2.putText(display_frame, "Hand + Mouse Controller", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                total_x = self.gesture_rotation_x + self.mouse_rotation_x
                total_y = self.gesture_rotation_y + self.mouse_rotation_y
                cv2.putText(display_frame, f"Hand: {self.gesture_rotation_x:.0f},{self.gesture_rotation_y:.0f},{self.gesture_rotation_z:.0f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Total: {total_x:.0f},{total_y:.0f},{self.gesture_rotation_z:.0f}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                cv2.putText(display_frame, "Tilt Up/Down | Turn Left/Right | Rotate CW/CCW", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Draw direction arrows as visual guides
                height, width = display_frame.shape[:2]
                center_x, center_y = width - 100, height - 100
                
                # Draw coordinate system reference
                cv2.arrowedLine(display_frame, (center_x, center_y), 
                               (center_x + 30, center_y), (0, 0, 255), 2)  # X - Red
                cv2.arrowedLine(display_frame, (center_x, center_y), 
                               (center_x, center_y - 30), (0, 255, 0), 2)  # Y - Green
                cv2.putText(display_frame, "X", (center_x + 35, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(display_frame, "Y", (center_x - 5, center_y - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Show camera window
                cv2.imshow("Hand Gesture Camera View", display_frame)
                cv2.waitKey(1)  # Required for OpenCV window to update
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get key hand landmarks for orientation calculation
                    wrist = hand_landmarks.landmark[0]
                    index_base = hand_landmarks.landmark[5]    # Index finger MCP
                    middle_base = hand_landmarks.landmark[9]   # Middle finger MCP
                    ring_base = hand_landmarks.landmark[13]    # Ring finger MCP
                    pinky_base = hand_landmarks.landmark[17]   # Pinky MCP
                    index_tip = hand_landmarks.landmark[8]     # Index finger tip
                    middle_tip = hand_landmarks.landmark[12]   # Middle finger tip
                    
                    # Calculate hand coordinate system vectors
                    # X-axis: From index to pinky (across the palm)
                    hand_x_vector = np.array([
                        pinky_base.x - index_base.x,
                        pinky_base.y - index_base.y,
                        pinky_base.z - index_base.z
                    ])
                    
                    # Y-axis: From wrist to middle finger (along the hand)
                    hand_y_vector = np.array([
                        middle_tip.x - wrist.x,
                        middle_tip.y - wrist.y,
                        middle_tip.z - wrist.z
                    ])
                    
                    # Z-axis: Cross product (perpendicular to palm)
                    hand_z_vector = np.cross(hand_x_vector, hand_y_vector)
                    
                    # Normalize vectors
                    hand_x_vector = hand_x_vector / (np.linalg.norm(hand_x_vector) + 1e-6)
                    hand_y_vector = hand_y_vector / (np.linalg.norm(hand_y_vector) + 1e-6)
                    hand_z_vector = hand_z_vector / (np.linalg.norm(hand_z_vector) + 1e-6)
                    
                    # Convert hand orientation to Euler angles
                    # Corrected for natural left/right and flip directions
                    
                    # Calculate hand orientation relative to camera view
                    # Note: frame is already flipped horizontally for mirror effect
                    
                    # Pitch (X rotation) - hand tilting up/down
                    pitch = math.asin(np.clip(-hand_z_vector[1], -1, 1)) * 180 / math.pi
                    
                    # Yaw (Y rotation) - hand turning left/right
                    # Negate to fix left/right direction (mirror correction)
                    yaw = -math.atan2(hand_z_vector[0], abs(hand_z_vector[2])) * 180 / math.pi
                    
                    # Roll (Z rotation) - hand rotating clockwise/counter-clockwise
                    # Use consistent vector calculation for stable roll
                    roll = math.atan2(-hand_x_vector[1], hand_x_vector[0]) * 180 / math.pi
                    
                    # Apply scaling and proper direction correction
                    target_rotation_x = pitch * 1.5       # Natural up/down tilt
                    target_rotation_y = yaw * 1.8         # Corrected left/right turn
                    target_rotation_z = roll * 1.0        # Natural roll rotation
                    
                    # Fix angle wrapping issues - prevent sudden jumps at -180/180 boundary
                    def smooth_angle_transition(current, target, smoothing):
                        # Calculate the shortest angular distance
                        diff = target - current
                        while diff > 180:
                            diff -= 360
                        while diff < -180:
                            diff += 360
                        
                        # Apply smoothing to the difference
                        return current + diff * smoothing
                    
                    # Apply smoothing with angle wrapping protection
                    smoothing_factor = 0.15  # Slightly more conservative for stability
                    
                    self.gesture_rotation_x = smooth_angle_transition(self.gesture_rotation_x, target_rotation_x, smoothing_factor)
                    self.gesture_rotation_y = smooth_angle_transition(self.gesture_rotation_y, target_rotation_y, smoothing_factor)  
                    self.gesture_rotation_z = smooth_angle_transition(self.gesture_rotation_z, target_rotation_z, smoothing_factor)
                    
                    # Feedback for first detection
                    if not hasattr(self, 'gesture_feedback_counter'):
                        self.gesture_feedback_counter = 0
                        print("🧤 Hand detected! Coordinate system locked to hand orientation")
                        print("   Rotate your hand to rotate the 3D view")
                    
                    self.gesture_feedback_counter += 1
                    
                    # Periodic status update
                    if self.gesture_feedback_counter % 120 == 0:
                        print(f"🔄 Hand tracking active - Natural 3D rotation")
            
            else:
                # No hand detected - gradually return to front view (Z-axis perpendicular to screen)
                target_front_x = 0.0   # No tilt
                target_front_y = 0.0   # No turn  
                target_front_z = 0.0   # No roll
                
                # Smoothly transition to front view
                return_speed = 0.02  # Slow return to front view
                self.gesture_rotation_x = self.gesture_rotation_x * (1 - return_speed) + target_front_x * return_speed
                self.gesture_rotation_y = self.gesture_rotation_y * (1 - return_speed) + target_front_y * return_speed
                self.gesture_rotation_z = self.gesture_rotation_z * (1 - return_speed) + target_front_z * return_speed
                
                if hasattr(self, 'gesture_feedback_counter'):
                    delattr(self, 'gesture_feedback_counter')
                    print("👋 Hand lost - returning to front view (Z-axis perpendicular to screen)")
        
        except Exception as e:
            print(f"⚠️  Hand gesture processing error: {e}")
            pass
    
    def handle_mouse_control(self):
        """Handle mouse control (works alongside gesture control)"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        
        if mouse_pressed[0]:  # Left click drag
            if self.mouse_down:
                # Calculate mouse movement
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                
                # Add to mouse rotation (additive with gesture control)
                self.mouse_rotation_y += dx * 0.5
                self.mouse_rotation_x += dy * 0.5
            
            self.mouse_down = True
            self.last_mouse_pos = mouse_pos
        else:
            self.mouse_down = False
    
    def draw_coordinate_axes(self):
        """Draw coordinate axes - now only shows Z axis"""
        glLineWidth(1.0)
        
        # Only draw Z-axis as a reference - blue color
        glColor4f(0.0, 0.0, 1.0, 0.3)  # Semi-transparent blue
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 10)  # Longer Z-axis line
        glEnd()
    
    def draw_spiral_trail(self):
        """绘制螺旋轨迹"""
        if len(self.spiral_points) < 2:
            return
        
        glLineWidth(1.5)
        
        # 绘制轨迹线条，颜色随时间变化
        glBegin(GL_LINE_STRIP)
        for i, (x, y, z) in enumerate(self.spiral_points):
            # 颜色随轨迹进展变化
            progress = i / len(self.spiral_points)
            
            # 彩虹色谱
            hue = (progress + self.time_step * 0.1) % 1.0
            r = 0.5 + 0.5 * math.sin(hue * 6.28)
            g = 0.5 + 0.5 * math.sin((hue + 0.33) * 6.28)  
            b = 0.5 + 0.5 * math.sin((hue + 0.67) * 6.28)
            
            # 透明度随距离衰减
            alpha = 0.3 + 0.7 * progress
            
            glColor4f(r, g, b, alpha)
            glVertex3f(x, y, z)
        glEnd()
        
        # 绘制当前点（更亮）
        if self.spiral_points:
            x, y, z = self.spiral_points[-1]
            glPointSize(8.0)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_POINTS)
            glVertex3f(x, y, z)
            glEnd()
    
    def update(self):
        """更新螺旋状态"""
        # 推进时间
        self.time_step += self.time_speed
        
        # 生成新的螺旋点
        new_point = self.generate_spiral_point(self.time_step)
        self.spiral_points.append(new_point)
        
        # 限制轨迹点数量
        if len(self.spiral_points) > self.max_points:
            self.spiral_points.pop(0)
        
        # Update hand gesture control
        self.update_hand_gesture()
        
        # Update mouse control (always active, works alongside gestures)
        self.handle_mouse_control()
    
    def render(self):
        """Render scene with camera following spiral center in 3D"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Calculate target camera position to follow spiral center in XYZ
        if self.spiral_points:
            # Focus on the very latest points for immediate following
            latest_points = self.spiral_points[-min(3, len(self.spiral_points)):]  # Only last 3 points for tight following
            if latest_points:
                # Get the XYZ center position of the newest points
                latest_x = sum(p[0] for p in latest_points) / len(latest_points)
                latest_y = sum(p[1] for p in latest_points) / len(latest_points)
                latest_z = sum(p[2] for p in latest_points) / len(latest_points)
                
                # Camera targets: keep XY at origin, only follow Z
                self.camera_x_target = 0.0          # Keep X at origin
                self.camera_y_target = 0.0          # Keep Y at origin  
                self.camera_z_target = -latest_z - 6.0  # Follow Z with viewing distance
                
                # Debug: print Z tracking every few frames
                if not hasattr(self, 'debug_frame_count'):
                    self.debug_frame_count = 0
                self.debug_frame_count += 1
                if self.debug_frame_count % 120 == 0:  # Every 2 seconds at 60fps
                    print(f"Z tracking - Latest: {latest_z:.3f}, Target: {self.camera_z_target:.3f}, Current: {self.camera_z_current:.3f}")
            else:
                self.camera_x_target = 0.0
                self.camera_y_target = 0.0
                self.camera_z_target = -6.0
        else:
            self.camera_x_target = 0.0
            self.camera_y_target = 0.0
            self.camera_z_target = -6.0
        
        # Smooth camera movement toward target
        self.camera_x_current = (self.camera_x_current * (1 - self.follow_speed) + 
                                 self.camera_x_target * self.follow_speed)
        self.camera_y_current = (self.camera_y_current * (1 - self.follow_speed) + 
                                 self.camera_y_target * self.follow_speed)
        self.camera_z_current = (self.camera_z_current * (1 - self.follow_speed) + 
                                 self.camera_z_target * self.follow_speed)
        
        # 计算动点的Z轴位置作为旋转中心
        current_z_center = 0.0
        if self.spiral_points:
            latest_points = self.spiral_points[-min(3, len(self.spiral_points)):]
            if latest_points:
                current_z_center = sum(p[2] for p in latest_points) / len(latest_points)
        
        # Set view position - basic viewing distance
        glTranslatef(0, 0, -6.0)
        
        # Move rotation center to current spiral Z position  
        glTranslatef(0, 0, current_z_center)
        
        # Apply combined rotation (gesture + mouse control) around dynamic center
        total_rotation_x = self.gesture_rotation_x + self.mouse_rotation_x
        total_rotation_y = self.gesture_rotation_y + self.mouse_rotation_y
        total_rotation_z = self.gesture_rotation_z  # Only gesture control for Z-axis
        
        glRotatef(total_rotation_x, 1, 0, 0)
        glRotatef(total_rotation_y, 0, 1, 0) 
        glRotatef(total_rotation_z, 0, 0, 1)
        
        # Move back to scene position
        glTranslatef(0, 0, -current_z_center)
        
        # Draw coordinate axes (now only Z-axis)
        self.draw_coordinate_axes()
        
        # Draw spiral trail
        self.draw_spiral_trail()
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        print("="*50)
        print("🧬 Euler Formula Spiral Trajectory Visualization")
        print("="*50)
        print("Formula: z(θ) = 11e^(i11θ) + 14sin(10θ)e^(iθ) + 13e^(iθ)")
        print()
        print("Academic Credits:")
        print("- Animation: Patrick Georges, University of Ottawa")
        print("- Original Idea: Chirag Dudhat")
        print()
        if self.camera_active:
            print("🧤 Hand Control (Natural 3D Controller):")
            print("- Palm facing camera = Looking down at XY plane")
            print("- Tilt hand up/down → Pitch rotation")
            print("- Turn hand left/right → Yaw rotation")  
            print("- Rotate hand clockwise/counter-clockwise → Roll")
            print("- Camera tracks spiral center in 3D space")
            print("🖱️ Mouse Control (additive with hand):")
            print("- Left click + drag → Additional rotation")
        else:
            print("🖱️ Control Methods:")
            print("- Left click + drag → Rotate view")
        print("- ESC key → Exit")
        print("- Space key → Clear trajectory")
        print("- Up/Down arrows → Adjust time speed")
        print("- +/- keys → Adjust Z-axis spacing")
        print("- F/G keys → Adjust camera follow speed")
        print("- T key → Toggle tight following mode")
        print("- M key → Reset mouse rotation")
        print("- R key → Reset to default speed")
        print("- C key → Try to reconnect camera")
        print("="*50)
        
        running = True
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        # Clear trajectory
                        self.spiral_points.clear()
                        self.time_step = 0.0
                        print("Trajectory cleared")
                    elif event.key == K_UP:
                        # Increase speed
                        self.time_speed = min(self.time_speed * 1.5, 0.1)
                        print(f"Speed increased to {self.time_speed:.4f}")
                    elif event.key == K_DOWN:
                        # Decrease speed
                        self.time_speed = max(self.time_speed * 0.67, 0.0001)
                        print(f"Speed decreased to {self.time_speed:.4f}")
                    elif event.key == K_r:
                        # Reset to default speed
                        self.time_speed = 0.001
                        print(f"Speed reset to default: {self.time_speed:.4f}")
                    elif event.key == K_EQUALS or event.key == K_PLUS:
                        # Increase Z-axis spacing
                        self.z_scale = min(self.z_scale * 1.2, 3.0)  # Increased limit to 3
                        print(f"Z-axis spacing increased to {self.z_scale:.3f}")
                    elif event.key == K_MINUS:
                        # Decrease Z-axis spacing
                        self.z_scale = max(self.z_scale * 0.83, 0.01)
                        print(f"Z-axis spacing decreased to {self.z_scale:.3f}")
                    elif event.key == K_c:
                        # Try to reconnect camera
                        self.reconnect_camera()
                    elif event.key == K_f:
                        # Increase camera follow speed
                        self.follow_speed = min(self.follow_speed * 1.2, 0.5)
                        print(f"Camera follow speed increased to {self.follow_speed:.3f}")
                    elif event.key == K_g:
                        # Decrease camera follow speed
                        self.follow_speed = max(self.follow_speed * 0.8, 0.01)
                        print(f"Camera follow speed decreased to {self.follow_speed:.3f}")
                    elif event.key == K_t:
                        # Toggle tight following mode
                        self.follow_speed = 0.8 if self.follow_speed < 0.5 else 0.3
                        print(f"Tight following mode: {self.follow_speed:.1f} ({'ON' if self.follow_speed > 0.5 else 'OFF'})")
                    elif event.key == K_m:
                        # Reset mouse rotation
                        self.mouse_rotation_x = 0
                        self.mouse_rotation_y = 0
                        print("Mouse rotation reset to neutral")
            
            # Update and render
            self.update()
            self.render()
            self.clock.tick(60)
        
        # Clean up resources
        if self.camera_active:
            self.cap.release()
            cv2.destroyAllWindows()  # Close camera view window
        pygame.quit()

if __name__ == "__main__":
    visualizer = EulerSpiralVisualizer()
    visualizer.run()