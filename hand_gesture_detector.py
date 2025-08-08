"""
TouchDesigner Hand Gesture Recognition Script
使用MediaPipe进行实时手势识别，输出手势数据用于控制粒子效果

在TouchDesigner中使用：将此脚本放在Text DAT中，在其他DAT中调用相关函数
"""

import cv2
import mediapipe as mp
import numpy as np
import math

class HandGestureDetector:
    def __init__(self):
        # 初始化MediaPipe手部模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=3,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 手势状态变量 - 支持最多3只手
        self.gesture_data = {
            'hands_detected': 0,
            'hands': [],  # 存储所有检测到的手
            'digit_gestures': [],  # 存储数字手势 [1, 2, 3]
            'active_audio_tracks': set(),  # 当前应该播放的音轨
            'gesture_type': 'none',
            'finger_distances': [],
            'hand_openness': 0.0,
            'hand_center': [0.5, 0.5],
            'gesture_strength': 0.0
        }
        
    def detect_gesture_type(self, landmarks):
        """根据关键点检测手势类型，专门识别数字1、2、3"""
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
        thumb_ip = landmarks[3]
        
        # 检测手指是否伸直
        fingers_up = []
        
        # 拇指检测 (改进的检测方法)
        thumb_up = thumb_tip[0] > thumb_ip[0] if landmarks[4][0] > landmarks[3][0] else thumb_tip[0] < thumb_ip[0]
        fingers_up.append(thumb_up)
        
        # 其他四个手指 (垂直比较，指尖高于第二关节)
        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                        (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            fingers_up.append(tip[1] < pip[1])
        
        # 根据伸直的手指数量和组合判断数字手势
        fingers_count = sum(fingers_up)
        
        # 专门识别数字1、2、3的逻辑（更灵活的识别）
        
        # 数字1：食指伸出（可以包含拇指）
        if fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return 1
            
        # 数字2：食指和中指都伸出（可以包含拇指）
        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return 2
            
        # 数字3：食指、中指、无名指都伸出（可以包含拇指）
        if fingers_up[1] and fingers_up[2] and fingers_up[3] and not fingers_up[4]:
            return 3
        
        # 如果不是目标数字手势，返回none
        return 'none'
    
    def calculate_hand_openness(self, landmarks):
        """计算手部张开程度 (0-1)"""
        if not landmarks or len(landmarks) < 21:
            return 0.0
        
        # 计算手指间的平均距离
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
        # 标准化到0-1范围 (经验值)
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
    
    def process_frame(self, frame):
        """处理视频帧并检测手势"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # 重置检测数据
        self.gesture_data['hands_detected'] = 0
        self.gesture_data['hands'] = []
        self.gesture_data['digit_gestures'] = []
        self.gesture_data['active_audio_tracks'] = set()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            self.gesture_data['hands_detected'] = len(results.multi_hand_landmarks)
            
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # 获取手部关键点
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # 确定左右手
                hand_label = handedness.classification[0].label.lower()
                gesture_number = self.detect_gesture_type(landmarks)
                
                # 存储手部数据
                hand_data = {
                    'id': i,
                    'label': hand_label,
                    'detected': True,
                    'landmarks': landmarks,
                    'gesture_number': gesture_number,
                    'openness': self.calculate_hand_openness(landmarks),
                    'center': self.calculate_hand_center(landmarks)
                }
                
                self.gesture_data['hands'].append(hand_data)
                
                # 如果是数字手势，添加到数字手势列表
                if gesture_number in [1, 2, 3]:
                    self.gesture_data['digit_gestures'].append(gesture_number)
                    self.gesture_data['active_audio_tracks'].add(gesture_number)
                
                # 绘制关键点 (可选)
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # 计算综合手势强度
        total_strength = 0
        for hand in self.gesture_data['hands']:
            total_strength += hand['openness']
        
        self.gesture_data['gesture_strength'] = total_strength / max(1, len(self.gesture_data['hands']))
        
        return frame

# TouchDesigner接口函数
def get_gesture_data():
    """获取当前手势数据，供TouchDesigner调用"""
    if hasattr(op, 'detector'):
        return op.detector.gesture_data
    return None

def initialize_detector():
    """初始化手势检测器"""
    if not hasattr(op, 'detector'):
        op.detector = HandGestureDetector()
    return True

def process_camera_frame(frame_data):
    """处理摄像头帧数据"""
    if not hasattr(op, 'detector'):
        initialize_detector()
    
    # 这里需要根据TouchDesigner的具体数据格式进行调整
    # frame_data应该是从Video In TOP获得的图像数据
    processed_frame = op.detector.process_frame(frame_data)
    return processed_frame