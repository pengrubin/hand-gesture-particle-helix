#!/usr/bin/env python3
"""
手势识别测试脚本
专门用于测试和调试手势识别的准确性
"""

import cv2
import time
from gesture_detector import GestureDetector

def test_gesture_detection():
    """测试手势检测准确性"""
    print("=== 手势识别测试模式 ===")
    print("这个模式专门用于测试手势识别准确性")
    print("请尝试以下手势：")
    print("- 握拳（0个手指）")  
    print("- 伸出1个手指（食指）")
    print("- 伸出2个手指（食指+中指）")
    print("- 伸出3个手指（食指+中指+无名指）")
    print("- 伸出4个手指")
    print("- 张开手掌（5个手指）")
    print("按 'q' 退出测试\n")
    
    detector = GestureDetector()
    
    try:
        detector.start_camera(0)
        print("摄像头启动成功，开始手势检测测试...\n")
        
        gesture_history = []
        
        while True:
            frame = detector.get_current_frame()
            if frame is not None:
                # 获取手势数据
                gesture_data = detector.get_gesture_data()
                
                if gesture_data and gesture_data.get('hands_detected', 0) > 0:
                    left_hand = gesture_data.get('left_hand', {})
                    right_hand = gesture_data.get('right_hand', {})
                    
                    # 显示检测结果
                    current_gesture = None
                    if left_hand.get('detected', False):
                        current_gesture = left_hand.get('gesture', 'unknown')
                        openness = left_hand.get('openness', 0)
                        print(f"左手: {current_gesture} (张开程度: {openness:.2f})")
                        
                    if right_hand.get('detected', False):
                        current_gesture = right_hand.get('gesture', 'unknown')
                        openness = right_hand.get('openness', 0)
                        print(f"右手: {current_gesture} (张开程度: {openness:.2f})")
                    
                    # 记录手势历史
                    if current_gesture:
                        gesture_history.append(current_gesture)
                        if len(gesture_history) > 10:
                            gesture_history.pop(0)
                    
                    # 添加测试信息到图像
                    add_test_info(frame, gesture_data)
                else:
                    print("未检测到手部")
                
                # 显示图像
                cv2.imshow('Hand Gesture Test', frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 显示手势统计
                if gesture_history:
                    from collections import Counter
                    gesture_counts = Counter(gesture_history)
                    print(f"\n最近10次手势统计: {dict(gesture_counts)}\n")
            
            time.sleep(0.03)  # ~30fps
                
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()
        print("测试结束")

def add_test_info(frame, gesture_data):
    """在测试图像上添加详细信息"""
    h, w = frame.shape[:2]
    
    # 背景
    cv2.rectangle(frame, (10, h - 200), (400, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - 200), (400, h - 10), (255, 255, 255), 2)
    
    # 标题
    cv2.putText(frame, "Gesture Detection Test", (20, h - 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 检测信息
    info_lines = []
    
    hands_count = gesture_data.get('hands_detected', 0)
    info_lines.append(f"Hands detected: {hands_count}")
    
    if hands_count > 0:
        left_hand = gesture_data.get('left_hand', {})
        right_hand = gesture_data.get('right_hand', {})
        
        if left_hand.get('detected', False):
            gesture = left_hand.get('gesture', 'unknown')
            openness = left_hand.get('openness', 0)
            info_lines.append(f"Left: {gesture} ({openness:.2f})")
            
        if right_hand.get('detected', False):
            gesture = right_hand.get('gesture', 'unknown')
            openness = right_hand.get('openness', 0)
            info_lines.append(f"Right: {gesture} ({openness:.2f})")
        
        strength = gesture_data.get('gesture_strength', 0)
        info_lines.append(f"Strength: {strength:.2f}")
    
    # 绘制信息
    for i, line in enumerate(info_lines):
        y = h - 140 + i * 25
        cv2.putText(frame, line, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # 添加按键提示
    cv2.putText(frame, "Press 's' for statistics, 'q' to quit", 
                (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def calibrate_gesture_thresholds():
    """标定手势检测阈值"""
    print("=== 手势检测标定模式 ===")
    print("这个模式用于标定和优化手势检测参数")
    print("请按以下步骤进行标定：")
    print("1. 握紧拳头，观察检测结果")
    print("2. 逐个伸出手指，观察计数变化")
    print("3. 记录不准确的情况\n")
    
    detector = GestureDetector()
    
    try:
        detector.start_camera(0)
        
        while True:
            frame = detector.get_current_frame()
            if frame is not None:
                cv2.imshow('Gesture Calibration', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"标定错误: {e}")
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    print("选择测试模式：")
    print("1. 手势识别准确性测试")
    print("2. 手势检测标定模式")
    print("3. 退出")
    
    try:
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            test_gesture_detection()
        elif choice == '2':
            calibrate_gesture_thresholds()
        elif choice == '3':
            print("退出测试")
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n用户中断退出")
    except Exception as e:
        print(f"测试错误: {e}")

if __name__ == "__main__":
    main()