#!/usr/bin/env python3
"""
摄像头诊断工具
用于排查摄像头被占用或无法打开的问题
"""

import cv2
import platform
import subprocess
import sys
import time

def get_system_info():
    """获取系统信息"""
    print("=" * 50)
    print("系统信息")
    print("=" * 50)

    system = platform.system()
    print(f"操作系统: {system}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")

    if system == "Darwin":
        try:
            macos_ver = platform.mac_ver()[0]
            print(f"macOS版本: {macos_ver}")

            # 检查是否是老版本
            parts = macos_ver.split('.')
            major = int(parts[0]) if parts else 0
            minor = int(parts[1]) if len(parts) > 1 else 0

            if major < 10 or (major == 10 and minor < 14):
                print("  [警告] 您使用的是老版本 macOS (< 10.14)")
                print("  [提示] 摄像头权限API在 macOS 10.14+ 有变化")
        except:
            pass

    return system

def check_camera_processes(system):
    """检查占用摄像头的进程"""
    print("\n" + "=" * 50)
    print("检查摄像头相关进程")
    print("=" * 50)

    if system == "Darwin":
        # macOS
        processes = ['VDCAssistant', 'AppleCameraAssistant', 'avconferenced']
        for proc in processes:
            try:
                result = subprocess.run(['pgrep', '-l', proc],
                                       capture_output=True, text=True)
                if result.stdout.strip():
                    print(f"[运行中] {proc}: {result.stdout.strip()}")
                else:
                    print(f"[未运行] {proc}")
            except:
                pass

        # 检查占用摄像头的应用
        print("\n可能占用摄像头的应用:")
        apps = ['zoom.us', 'FaceTime', 'Skype', 'Teams', 'Discord', 'OBS']
        for app in apps:
            try:
                result = subprocess.run(['pgrep', '-l', app],
                                       capture_output=True, text=True)
                if result.stdout.strip():
                    print(f"  [占用中] {app}")
            except:
                pass

    elif system == "Windows":
        print("Windows: 请在任务管理器中检查占用摄像头的应用")

    elif system == "Linux":
        try:
            result = subprocess.run(['fuser', '/dev/video0'],
                                   capture_output=True, text=True)
            if result.stdout.strip():
                print(f"占用 /dev/video0 的进程: {result.stdout.strip()}")
        except:
            pass

def test_camera_backends():
    """测试不同的摄像头后端"""
    print("\n" + "=" * 50)
    print("测试摄像头后端")
    print("=" * 50)

    backends = [
        (cv2.CAP_ANY, "CAP_ANY (自动)"),
        (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION (macOS)"),
    ]

    # 添加平台特定后端
    if platform.system() == "Windows":
        backends.extend([
            (cv2.CAP_DSHOW, "CAP_DSHOW (DirectShow)"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
        ])
    elif platform.system() == "Linux":
        backends.append((cv2.CAP_V4L2, "CAP_V4L2"))

    working_backend = None

    for backend_id, backend_name in backends:
        print(f"\n测试 {backend_name}...")
        try:
            cap = cv2.VideoCapture(0, backend_id)

            if cap.isOpened():
                print(f"  [OK] 摄像头已打开")

                # 尝试读取帧
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  [OK] 可以读取帧 ({frame.shape[1]}x{frame.shape[0]})")
                    working_backend = backend_name
                else:
                    print(f"  [失败] 无法读取帧")

                cap.release()
                print(f"  [OK] 摄像头已释放")

                # 等待一下确保完全释放
                time.sleep(0.5)
            else:
                print(f"  [失败] 无法打开摄像头")

        except Exception as e:
            print(f"  [异常] {e}")

    return working_backend

def release_camera_macos():
    """尝试释放被占用的macOS摄像头"""
    print("\n" + "=" * 50)
    print("尝试释放macOS摄像头")
    print("=" * 50)

    if platform.system() != "Darwin":
        print("此功能仅适用于macOS")
        return

    print("运行 killall VDCAssistant...")
    try:
        result = subprocess.run(['killall', 'VDCAssistant'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("  [OK] VDCAssistant 已重启")
        else:
            print("  [INFO] VDCAssistant 未在运行或无需重启")
    except Exception as e:
        print(f"  [失败] {e}")
        print("  [提示] 尝试手动运行: sudo killall VDCAssistant")

    print("\n运行 killall AppleCameraAssistant...")
    try:
        result = subprocess.run(['killall', 'AppleCameraAssistant'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("  [OK] AppleCameraAssistant 已重启")
        else:
            print("  [INFO] AppleCameraAssistant 未在运行")
    except Exception as e:
        print(f"  [INFO] {e}")

    print("\n等待摄像头服务重启...")
    time.sleep(2)

def main():
    print("\n摄像头诊断工具")
    print("用于排查摄像头被占用的问题\n")

    # 获取系统信息
    system = get_system_info()

    # 检查进程
    check_camera_processes(system)

    # 测试后端
    working = test_camera_backends()

    if working:
        print(f"\n[成功] 找到可用后端: {working}")
    else:
        print("\n[失败] 没有找到可用的摄像头后端")

        if system == "Darwin":
            print("\n是否尝试释放摄像头? (y/n): ", end="")
            try:
                response = input().strip().lower()
                if response == 'y':
                    release_camera_macos()
                    print("\n重新测试...")
                    working = test_camera_backends()
                    if working:
                        print(f"\n[成功] 释放后找到可用后端: {working}")
            except:
                pass

    # 总结
    print("\n" + "=" * 50)
    print("诊断总结")
    print("=" * 50)

    if working:
        print("[OK] 摄像头可以正常工作")
        print(f"[推荐后端] {working}")
    else:
        print("[问题] 摄像头无法使用")
        print("\n可能的解决方案:")
        print("1. 关闭其他使用摄像头的应用 (Zoom, FaceTime, Teams 等)")
        print("2. 运行: sudo killall VDCAssistant (macOS)")
        print("3. 检查系统偏好设置中的摄像头权限")
        print("4. 重启电脑")

        if system == "Darwin":
            try:
                macos_ver = platform.mac_ver()[0]
                parts = macos_ver.split('.')
                major = int(parts[0]) if parts else 0
                minor = int(parts[1]) if len(parts) > 1 else 0

                if major < 10 or (major == 10 and minor < 14):
                    print(f"\n[老版本macOS注意事项] 您的系统是 macOS {macos_ver}")
                    print("- 考虑升级到 macOS 10.14+ 以获得更好的摄像头支持")
                    print("- 尝试使用 CAP_ANY 后端而非 CAP_AVFOUNDATION")
            except:
                pass

if __name__ == "__main__":
    main()
