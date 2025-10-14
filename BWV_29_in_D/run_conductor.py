#!/usr/bin/env python3
"""
BWV_29_in_D 指挥家控制系统启动器
简化的启动脚本，自动检测配置并启动系统

使用方法:
python run_conductor.py

Author: Claude Code
Date: 2025-10-05
"""

import os
import sys
import subprocess
import platform


def check_dependencies():
    """检查依赖项是否安装"""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'pygame': 'pygame'
    }

    missing_packages = []

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_audio_files():
    """检查音频文件是否存在"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "Tromba_I_in_D.mp3",
        "Tromba_II_in_D.mp3",
        "Tromba_III_in_D.mp3",
        "Violins_in_D.mp3",
        "Viola_in_D.mp3",
        "Oboe_I_in_D.mp3",
        "Continuo_in_D.mp3",
        "Organo_obligato_in_D.mp3",
        "Timpani_in_D.mp3"
    ]

    missing_files = []
    found_files = []

    for filename in required_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            found_files.append(filename)
            print(f"✓ {filename}")
        else:
            missing_files.append(filename)
            print(f"✗ {filename} (missing)")

    print(f"\n音频文件状态: {len(found_files)}/{len(required_files)} 已找到")

    if missing_files:
        print("\n警告: 缺少音频文件，某些功能可能无法正常工作")
        print("缺少的文件:")
        for filename in missing_files:
            print(f"  - {filename}")

        response = input("\n是否继续启动? (y/n): ").lower().strip()
        return response in ['y', 'yes', '是']

    return True


def check_camera():
    """检查摄像头是否可用"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ 摄像头可用")
                return True
            else:
                print("✗ 摄像头无法读取画面")
                return False
        else:
            print("✗ 摄像头无法打开")
            return False
    except Exception as e:
        print(f"✗ 摄像头检查失败: {e}")
        return False


def get_system_info():
    """获取系统信息"""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0]
    }

    print("\n系统信息:")
    print(f"  平台: {info['platform']}")
    print(f"  Python版本: {info['python_version']}")
    print(f"  架构: {info['architecture']}")

    return info


def main():
    """主函数"""
    print("BWV_29_in_D 指挥家控制系统启动器")
    print("=" * 50)

    # 获取系统信息
    get_system_info()

    print("\n正在检查系统要求...")

    # 检查依赖项
    print("\n1. 检查Python依赖项:")
    if not check_dependencies():
        sys.exit(1)

    # 检查摄像头
    print("\n2. 检查摄像头:")
    if not check_camera():
        response = input("\n摄像头不可用，是否继续? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '是']:
            sys.exit(1)

    # 检查音频文件
    print("\n3. 检查音频文件:")
    if not check_audio_files():
        sys.exit(1)

    print("\n系统检查完成，正在启动指挥家控制系统...")
    print("\n控制说明:")
    print("  ESC - 退出程序")
    print("  SPACE - 暂停/恢复")
    print("  R - 重置手势检测")
    print("  D - 切换调试信息")
    print("  F - 切换全屏")
    print("  1-9 - 直接音量控制")
    print("  0 - 静音所有声部")

    print("\n请将手掌放在摄像头前开始指挥...")
    input("按Enter键开始启动 (Ctrl+C 取消)...")

    try:
        # 启动主控制程序
        script_path = os.path.join(os.path.dirname(__file__), 'conductor_control.py')
        subprocess.run([sys.executable, script_path], check=True)

    except KeyboardInterrupt:
        print("\n启动已取消")

    except subprocess.CalledProcessError as e:
        print(f"\n程序运行出错: {e}")
        print("请检查错误信息并重试")

    except Exception as e:
        print(f"\n未知错误: {e}")


if __name__ == "__main__":
    main()