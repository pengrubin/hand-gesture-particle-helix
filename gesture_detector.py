"""
纯Python手势识别系统
使用MediaPipe进行实时手势检测，无需TouchDesigner
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time

class GestureDetector:
    def __init__(self):
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 手势数据
        self.gesture_data = {
            'hands_detected': 0,
            'left_hand': {
                'detected': False, 
                'landmarks': [], 
                'gesture': 'none', 
                'openness': 0.0, 
                'center': [0.5, 0.5],
                'rotation_angle': 0.0,  # 手掌旋转角度（弧度）
                'palm_direction': [0.0, 1.0],  # 手掌朝向向量
                'rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 3D旋转矩阵
            },
            'right_hand': {
                'detected': False, 
                'landmarks': [], 
                'gesture': 'none', 
                'openness': 0.0, 
                'center': [0.5, 0.5],
                'rotation_angle': 0.0,
                'palm_direction': [0.0, 1.0],
                'rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 3D旋转矩阵
            },
            'gesture_strength': 0.0,
            'timestamp': 0.0,
            'combined_rotation': 0.0,  # 综合旋转角度
            'combined_rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 综合3D旋转矩阵
        }
        
        # 相机设置
        self.camera = None
        self.is_running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
    def start_camera(self, camera_id=0):
        """启动摄像头"""
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id}")
        
        self.is_running = True
        self.thread = threading.Thread(target=self._camera_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()
    
    def _camera_loop(self):
        """摄像头循环线程"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # 水平翻转以获得镜像效果
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(frame)
                
                with self.frame_lock:
                    self.current_frame = processed_frame
            
            time.sleep(1/30)  # 30 FPS
    
    def get_current_frame(self):
        """获取当前处理后的帧"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def process_frame(self, frame):
        """处理单帧图像"""
        if frame is None:
            return None
        
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # 重置检测数据
        self.gesture_data['hands_detected'] = 0
        self.gesture_data['left_hand']['detected'] = False
        self.gesture_data['right_hand']['detected'] = False
        self.gesture_data['timestamp'] = time.time()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            self.gesture_data['hands_detected'] = len(results.multi_hand_landmarks)
            
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # 获取手部关键点
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # 确定左右手
                hand_label = handedness.classification[0].label.lower()
                
                # 计算手势特征
                gesture_type = self.detect_gesture_type(landmarks)
                openness = self.calculate_hand_openness(landmarks)
                center = self.calculate_hand_center(landmarks)
                rotation_angle, palm_direction, rotation_matrix = self.calculate_palm_rotation(landmarks)
                
                # 存储数据
                hand_data = {
                    'detected': True,
                    'landmarks': landmarks,
                    'gesture': gesture_type,
                    'openness': openness,
                    'center': center,
                    'rotation_angle': rotation_angle,
                    'palm_direction': palm_direction,
                    'rotation_matrix': rotation_matrix  # 新增：3D旋转矩阵
                }
                
                if hand_label == 'left':
                    self.gesture_data['left_hand'] = hand_data
                else:
                    self.gesture_data['right_hand'] = hand_data
                
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        # 计算综合手势强度和旋转角度
        left_strength = self.gesture_data['left_hand']['openness'] if self.gesture_data['left_hand']['detected'] else 0
        right_strength = self.gesture_data['right_hand']['openness'] if self.gesture_data['right_hand']['detected'] else 0
        self.gesture_data['gesture_strength'] = max(left_strength, right_strength)
        
        # 计算综合旋转角度和矩阵
        self.gesture_data['combined_rotation'], self.gesture_data['combined_rotation_matrix'] = self.calculate_combined_rotation()
        
        # 添加文本信息
        self.draw_info_on_frame(frame)
        
        return frame
    
    def detect_gesture_type(self, landmarks):
        """检测手势类型"""
        if not landmarks or len(landmarks) < 21:
            return 'none'
        
        # 获取指尖和指关节的位置
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        
        # 检测手指是否伸直
        fingers_up = []
        
        # 拇指 (特殊处理，更准确的检测)
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        
        # 改进的拇指检测：比较拇指尖相对于拇指IP关节的位置
        # 如果拇指尖距离手腕更远，则认为拇指伸直
        thumb_tip_to_wrist = math.sqrt(
            (thumb_tip[0] - wrist[0])**2 + (thumb_tip[1] - wrist[1])**2
        )
        thumb_ip_to_wrist = math.sqrt(
            (thumb_ip[0] - wrist[0])**2 + (thumb_ip[1] - wrist[1])**2
        )
        thumb_up = thumb_tip_to_wrist > thumb_ip_to_wrist * 1.1
        fingers_up.append(thumb_up)
        
        # 其他四个手指：更准确的检测方法
        finger_landmarks = [
            (index_tip, index_pip, landmarks[5]),    # 食指：tip, pip, mcp
            (middle_tip, middle_pip, landmarks[9]),   # 中指：tip, pip, mcp  
            (ring_tip, ring_pip, landmarks[13]),     # 无名指：tip, pip, mcp
            (pinky_tip, pinky_pip, landmarks[17])    # 小指：tip, pip, mcp
        ]
        
        for tip, pip, mcp in finger_landmarks:
            # 方法1：指尖相对于PIP关节的Y坐标（基础检测）
            basic_up = tip[1] < pip[1]
            
            # 方法2：指尖相对于MCP关节的距离（增强检测）
            tip_to_wrist = math.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
            mcp_to_wrist = math.sqrt((mcp[0] - wrist[0])**2 + (mcp[1] - wrist[1])**2)
            distance_up = tip_to_wrist > mcp_to_wrist * 1.1
            
            # 综合判断：两种方法都满足才认为手指伸直
            finger_up = basic_up and distance_up
            fingers_up.append(finger_up)
        
        fingers_count = sum(fingers_up)
        
        # 调试信息：打印手指检测状态
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        debug_info = []
        for i, (name, up) in enumerate(zip(finger_names, fingers_up)):
            if up:
                debug_info.append(name)
        
        # 偶尔打印调试信息（避免输出过多）
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 60 == 0:  # 每60帧打印一次
            print(f"检测到 {fingers_count} 个手指: {', '.join(debug_info) if debug_info else '无'}")
        
        gesture_map = {
            0: 'fist',
            1: 'one',
            2: 'two', 
            3: 'three',
            4: 'four',
            5: 'open_hand'
        }
        
        return gesture_map.get(fingers_count, 'unknown')
    
    def calculate_hand_openness(self, landmarks):
        """计算手部张开程度"""
        if not landmarks or len(landmarks) < 21:
            return 0.0
        
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        total_distance = 0
        count = 0
        
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                dist = math.sqrt(
                    (finger_tips[i][0] - finger_tips[j][0])**2 + 
                    (finger_tips[i][1] - finger_tips[j][1])**2
                )
                total_distance += dist
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        return min(avg_distance / 0.3, 1.0)
    
    def calculate_hand_center(self, landmarks):
        """计算手部中心点"""
        if not landmarks or len(landmarks) < 21:
            return [0.5, 0.5]
        
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return [center_x, center_y]
    
    def calculate_palm_rotation(self, landmarks):
        """计算手掌的3D旋转矩阵"""
        if not landmarks or len(landmarks) < 21:
            identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            return 0.0, [1.0, 0.0], identity_matrix
        
        try:
            # 使用MediaPipe的3D坐标信息计算完整的手掌姿态
            wrist = landmarks[0]        # 手腕
            index_mcp = landmarks[5]    # 食指根部  
            middle_mcp = landmarks[9]   # 中指根部
            pinky_mcp = landmarks[17]   # 小指根部
            
            # 构建手掌坐标系的三个轴
            # X轴：从食指指向小指（手掌宽度方向）
            x_axis = [pinky_mcp[0] - index_mcp[0], 
                      pinky_mcp[1] - index_mcp[1], 
                      pinky_mcp[2] - index_mcp[2]]
            
            # Y轴：从手腕指向中指（手掌长度方向）  
            y_axis = [middle_mcp[0] - wrist[0], 
                      middle_mcp[1] - wrist[1], 
                      middle_mcp[2] - wrist[2]]
            
            # 正规化向量
            def normalize(v):
                length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                return [v[0]/length, v[1]/length, v[2]/length] if length > 0.0001 else [0, 0, 1]
            
            x_axis = normalize(x_axis)
            y_axis = normalize(y_axis)
            
            # Z轴：通过叉积计算（手掌法向量）
            z_axis = [
                x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
                x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2], 
                x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0]
            ]
            z_axis = normalize(z_axis)
            
            # 重新正交化Y轴以确保坐标系正交
            y_axis = [
                z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
                z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
                z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0]
            ]
            y_axis = normalize(y_axis)
            
            # 构建3D旋转矩阵
            rotation_matrix = [
                [x_axis[0], y_axis[0], z_axis[0]],
                [x_axis[1], y_axis[1], z_axis[1]], 
                [x_axis[2], y_axis[2], z_axis[2]]
            ]
            
            # 为了兼容性，仍然返回一个角度值（Z轴旋转的近似）
            angle = math.atan2(x_axis[1], x_axis[0])
            
            return angle, x_axis, rotation_matrix
            
        except (IndexError, ZeroDivisionError, ValueError):
            # 返回单位矩阵
            identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            return 0.0, [1.0, 0.0], identity_matrix
    
    def calculate_combined_rotation(self):
        """计算综合旋转角度和3D旋转矩阵"""
        left_detected = self.gesture_data['left_hand']['detected']
        right_detected = self.gesture_data['right_hand']['detected']
        
        identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        if left_detected and right_detected:
            # 双手都检测到：使用主手（右手优先）的旋转矩阵
            # 也可以计算两个矩阵的平均，但这更复杂
            primary_matrix = self.gesture_data['right_hand']['rotation_matrix']
            primary_angle = self.gesture_data['right_hand']['rotation_angle']
            
            return primary_angle, primary_matrix
            
        elif left_detected:
            left_angle = self.gesture_data['left_hand']['rotation_angle']
            left_matrix = self.gesture_data['left_hand']['rotation_matrix']
            return left_angle, left_matrix
            
        elif right_detected:
            right_angle = self.gesture_data['right_hand']['rotation_angle']
            right_matrix = self.gesture_data['right_hand']['rotation_matrix']
            return right_angle, right_matrix
            
        else:
            return 0.0, identity_matrix
    
    def draw_info_on_frame(self, frame):
        """在帧上绘制信息"""
        h, w = frame.shape[:2]
        
        # 背景框
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # 检测信息
        info_lines = [
            f"Hands: {self.gesture_data['hands_detected']}",
            f"Strength: {self.gesture_data['gesture_strength']:.2f}",
        ]
        
        if self.gesture_data['left_hand']['detected']:
            left = self.gesture_data['left_hand']
            info_lines.append(f"Left: {left['gesture']} ({left['openness']:.2f})")
        
        if self.gesture_data['right_hand']['detected']:
            right = self.gesture_data['right_hand']
            info_lines.append(f"Right: {right['gesture']} ({right['openness']:.2f})")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
    
    def get_gesture_data(self):
        """获取当前手势数据"""
        return self.gesture_data.copy()

if __name__ == "__main__":
    # 测试代码
    detector = GestureDetector()
    
    try:
        detector.start_camera(0)
        print("手势检测启动成功！按 'q' 退出")
        
        while True:
            frame = detector.get_current_frame()
            if frame is not None:
                cv2.imshow('Gesture Detection', frame)
                
                # 打印手势数据
                data = detector.get_gesture_data()
                if data['hands_detected'] > 0:
                    print(f"检测到 {data['hands_detected']} 只手，强度: {data['gesture_strength']:.2f}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()