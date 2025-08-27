#!/usr/bin/env python3
"""
跨平台安装脚本
自动检测平台并安装合适的依赖包
"""

import sys
import platform
import subprocess
import os
from typing import List, Dict, Any

def get_platform_info() -> Dict[str, Any]:
    """获取平台信息"""
    system = platform.system()
    machine = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if system == "Darwin":  # macOS
        if machine == "arm64":
            processor_type = "Apple Silicon"
        else:
            processor_type = "Intel"
    elif system == "Windows":
        processor_type = "x86_64" if machine == "AMD64" else machine
    else:
        processor_type = machine
    
    return {
        'system': system,
        'machine': machine,
        'processor_type': processor_type,
        'python_version': python_version,
        'python_executable': sys.executable
    }

def run_command(cmd: List[str], description: str = "") -> bool:
    """运行命令并返回是否成功"""
    try:
        print(f"正在执行: {' '.join(cmd)}")
        if description:
            print(f"  {description}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ 命令未找到: {cmd[0]}")
        return False

def install_core_dependencies(platform_info: Dict[str, Any]) -> bool:
    """安装核心依赖"""
    print("\n=== 安装核心依赖 ===")
    
    core_packages = [
        "numpy>=1.21.0,<2.0.0",
        "opencv-python>=4.5.0,<5.0.0", 
        "mediapipe>=0.9.0,<=0.10.21"
    ]
    
    pip_cmd = [platform_info['python_executable'], '-m', 'pip', 'install']
    
    # 根据平台调整安装策略
    if platform_info['system'] == 'Darwin' and platform_info['processor_type'] == 'Apple Silicon':
        print("检测到 Apple Silicon Mac，使用优化的安装策略...")
        # Apple Silicon 优先使用预编译二进制
        pip_cmd.extend(['--only-binary=all'])
    elif platform_info['system'] == 'Darwin' and platform_info['processor_type'] == 'Intel':
        print("检测到 Intel Mac，使用兼容性安装策略...")
        # Intel Mac 可能需要特殊处理
        pass
    elif platform_info['system'] == 'Windows':
        print("检测到 Windows，使用标准安装策略...")
        pass
    
    # 尝试安装核心包
    success = True
    for package in core_packages:
        cmd = pip_cmd + [package]
        if not run_command(cmd, f"安装 {package}"):
            success = False
    
    return success

def install_optional_dependencies(platform_info: Dict[str, Any]) -> bool:
    """安装可选依赖"""
    print("\n=== 安装可选依赖 ===")
    
    optional_packages = [
        ("pygame>=2.1.0,<3.0.0", "音频处理支持"),
        ("PyOpenGL>=3.1.5,<4.0.0", "3D渲染支持"),
        ("mido>=1.2.0,<2.0.0", "MIDI处理支持")
    ]
    
    pip_cmd = [platform_info['python_executable'], '-m', 'pip', 'install']
    
    if platform_info['system'] == 'Darwin' and platform_info['processor_type'] == 'Apple Silicon':
        pip_cmd.extend(['--only-binary=all'])
    
    success_count = 0
    for package, description in optional_packages:
        cmd = pip_cmd + [package]
        if run_command(cmd, f"安装 {package} ({description})"):
            success_count += 1
        else:
            print(f"  警告: {description} 安装失败，但不影响核心功能")
    
    # PyOpenGL-accelerate 特殊处理（Intel Mac 经常失败）
    if platform_info['system'] == 'Darwin' and platform_info['processor_type'] == 'Intel':
        print("Intel Mac 跳过 PyOpenGL-accelerate（经常导致安装失败）")
    else:
        accelerate_cmd = pip_cmd + ["PyOpenGL-accelerate>=3.1.5,<4.0.0"]
        if run_command(accelerate_cmd, "安装 PyOpenGL 加速支持"):
            success_count += 1
        else:
            print("  警告: PyOpenGL 加速支持安装失败，将使用软件渲染")
    
    print(f"可选依赖安装完成: {success_count}/{len(optional_packages)+1} 成功")
    return True

def test_installation(platform_info: Dict[str, Any]) -> bool:
    """测试安装是否成功"""
    print("\n=== 测试安装 ===")
    
    tests = [
        ("import cv2; print(f'OpenCV: {cv2.__version__}')", "OpenCV"),
        ("import mediapipe as mp; print(f'MediaPipe: {mp.__version__}')", "MediaPipe"),
        ("import numpy as np; print(f'NumPy: {np.__version__}')", "NumPy")
    ]
    
    optional_tests = [
        ("import pygame; print(f'Pygame: {pygame.version.ver}')", "Pygame"),
        ("import OpenGL; print('PyOpenGL: OK')", "PyOpenGL"),
        ("import mido; print('Mido: OK')", "Mido")
    ]
    
    # 测试核心模块
    core_success = True
    for test_code, name in tests:
        cmd = [platform_info['python_executable'], '-c', test_code]
        if run_command(cmd, f"测试 {name}"):
            pass
        else:
            print(f"✗ {name} 测试失败")
            core_success = False
    
    if not core_success:
        return False
    
    # 测试可选模块
    optional_success = 0
    for test_code, name in optional_tests:
        cmd = [platform_info['python_executable'], '-c', test_code]
        if run_command(cmd, f"测试 {name}"):
            optional_success += 1
    
    print(f"\n核心模块: {'✓ 全部正常' if core_success else '✗ 有问题'}")
    print(f"可选模块: {optional_success}/{len(optional_tests)} 可用")
    
    return core_success

def test_camera_access():
    """测试摄像头访问"""
    print("\n=== 测试摄像头访问 ===")
    
    test_code = """
import cv2
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('✓ 摄像头访问正常')
        else:
            print('✗ 摄像头无法读取帧')
        cap.release()
    else:
        print('✗ 摄像头无法打开')
except Exception as e:
    print(f'✗ 摄像头测试错误: {e}')
"""
    
    cmd = [sys.executable, '-c', test_code]
    run_command(cmd, "测试摄像头访问")

def print_platform_specific_notes(platform_info: Dict[str, Any]):
    """打印平台特定说明"""
    print(f"\n=== 平台特定说明 ({platform_info['processor_type']}) ===")
    
    if platform_info['system'] == 'Darwin':  # macOS
        print("macOS 使用说明:")
        print("1. 如果摄像头无法访问，请检查:")
        print("   系统偏好设置 > 安全性与隐私 > 隐私 > 相机")
        print("   确保 Terminal 或您的 Python IDE 有摄像头权限")
        print("2. 重新启动终端或 IDE 可能有助于权限生效")
        
        if platform_info['processor_type'] == 'Intel':
            print("3. Intel Mac 特别说明:")
            print("   - GPU 加速可能不可用，系统将自动使用 CPU 模式")
            print("   - 如果遇到 PyOpenGL-accelerate 问题，可以跳过不安装")
        else:
            print("3. Apple Silicon Mac 特别说明:")
            print("   - 支持完整的 GPU 加速")
            print("   - 如果遇到包安装问题，考虑使用 conda")
    
    elif platform_info['system'] == 'Windows':
        print("Windows 使用说明:")
        print("1. 如果摄像头无法访问，请检查:")
        print("   设置 > 隐私 > 相机")
        print("   确保应用有摄像头权限")
        print("2. 如果遇到 OpenCV 问题，可尝试:")
        print("   pip uninstall opencv-python")
        print("   pip install opencv-python-headless")
        print("3. 某些情况下需要安装 Microsoft Visual C++ Redistributable")

def main():
    print("跨平台手势识别系统安装程序")
    print("=" * 50)
    
    # 获取平台信息
    platform_info = get_platform_info()
    print(f"检测到平台: {platform_info['system']} {platform_info['processor_type']}")
    print(f"Python版本: {platform_info['python_version']}")
    print(f"Python路径: {platform_info['python_executable']}")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("✗ 错误: 需要 Python 3.8 或更高版本")
        sys.exit(1)
    elif sys.version_info >= (3, 13):
        print("⚠ 警告: Python 3.13+ 未完全测试，可能存在兼容性问题")
    
    # 升级pip
    print("\n=== 升级 pip ===")
    run_command([platform_info['python_executable'], '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # 安装核心依赖
    if not install_core_dependencies(platform_info):
        print("✗ 核心依赖安装失败，无法继续")
        sys.exit(1)
    
    # 安装可选依赖
    install_optional_dependencies(platform_info)
    
    # 测试安装
    if not test_installation(platform_info):
        print("✗ 安装测试失败")
        sys.exit(1)
    
    # 测试摄像头
    test_camera_access()
    
    # 打印平台说明
    print_platform_specific_notes(platform_info)
    
    print("\n" + "=" * 50)
    print("✓ 安装完成！")
    print("现在可以运行:")
    print("  python cross_platform_gesture_detector.py")
    print("或者:")
    print("  python gesture_detector.py")

if __name__ == "__main__":
    main()