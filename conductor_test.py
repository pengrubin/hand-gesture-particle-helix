#!/usr/bin/env python3
"""
指挥家手势识别测试脚本
测试修改后的 hand_gesture_detector.py 文件的功能
"""

import cv2
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_gesture_detector import HandGestureDetector

def test_conductor_gesture_detection():
    """测试指挥家手势识别功能"""

    # 初始化检测器
    detector = HandGestureDetector()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("指挥家手势识别测试")
    print("=" * 50)
    print("区域布局：")
    print("[1-Tromba] [2-Violins] [3-Viola]")
    print("[4-Oboe]   [中央控制]   [5-Continuo]")
    print("[6-Organo] [7-Timpani]")
    print("=" * 50)
    print("手势识别：")
    print("- open_hand: 开手 (强奏)")
    print("- fist: 握拳 (停止)")
    print("- pointing: 指向 (特定指挥)")
    print("- conducting: 一般指挥")
    print("=" * 50)
    print("按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            break

        # 处理帧并检测手势
        processed_frame = detector.process_frame(frame, show_regions=True)

        # 获取手势数据
        gesture_data = detector.gesture_data

        # 在屏幕上显示状态信息
        info_y = 30
        cv2.putText(processed_frame, f"Hands detected: {gesture_data['hands_detected']}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        info_y += 30
        cv2.putText(processed_frame, f"Central control: {gesture_data['central_control_active']}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示激活的区域
        info_y += 30
        active_regions = list(gesture_data['active_regions'].keys())
        if active_regions:
            regions_text = "Active regions: " + ", ".join([r.split('_')[0] for r in active_regions])
            cv2.putText(processed_frame, regions_text,
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 获取指挥命令
        commands = detector.get_conductor_commands()
        info_y += 30
        cv2.putText(processed_frame, f"Volume: {commands['volume_change']:.2f}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        info_y += 25
        cv2.putText(processed_frame, f"Expression: {commands['expression_level']:.2f}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        info_y += 25
        cv2.putText(processed_frame, f"Stop: {commands['stop_command']}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0) if commands['stop_command'] else (255, 255, 0), 2)

        # 显示处理后的帧
        cv2.imshow('Conductor Gesture Recognition', processed_frame)

        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重置检测器
            detector = HandGestureDetector()
            print("检测器已重置")

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("测试结束")

def print_region_info():
    """打印区域信息"""
    detector = HandGestureDetector()
    region_info = detector.get_region_info()

    print("\n区域信息：")
    print("=" * 60)

    print("声部区域：")
    for name, data in region_info['voice_regions'].items():
        bounds = data['bounds']
        print(f"  {data['id']}. {name}")
        print(f"     边界: ({bounds['x1']:.2f}, {bounds['y1']:.2f}) -> ({bounds['x2']:.2f}, {bounds['y2']:.2f})")
        print(f"     中心: ({data['center'][0]:.2f}, {data['center'][1]:.2f})")
        print(f"     颜色: {data['color']}")
        print()

    print("中央控制区域：")
    central = region_info['central_control_region']
    bounds = central['bounds']
    print(f"     边界: ({bounds['x1']:.2f}, {bounds['y1']:.2f}) -> ({bounds['x2']:.2f}, {bounds['y2']:.2f})")
    print(f"     中心: ({central['center'][0]:.2f}, {central['center'][1]:.2f})")
    print(f"     颜色: {central['color']}")

if __name__ == "__main__":
    print("指挥家手势识别系统测试")
    print_region_info()

    # 询问是否开始摄像头测试
    response = input("\n是否开始摄像头测试？(y/n): ").lower().strip()
    if response == 'y' or response == 'yes':
        test_conductor_gesture_detection()
    else:
        print("测试结束")