#!/usr/bin/env python3
"""
E_Major 跨平台启动脚本
支持 Windows、macOS、Linux
"""

import os
import sys
import platform
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("✅ 需要Python 3.7或更高版本")
        print("\n请访问 https://www.python.org/downloads/ 下载最新版本")
        return False

    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查依赖包"""
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'pygame': 'pygame'
    }

    missing = []

    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - 未安装")
            missing.append(package_name)

    return missing

def install_dependencies(missing_packages):
    """安装缺失的依赖包"""
    if not missing_packages:
        return True

    print(f"\n检测到 {len(missing_packages)} 个缺失的依赖包")
    print("尝试自动安装...\n")

    try:
        # 使用pip安装
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        print(f"执行: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("\n✅ 依赖包安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ 自动安装失败")
        print("\n请手动安装依赖包：")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\n或使用requirements.txt：")
        print("  pip install -r requirements.txt")
        return False

def check_audio_files():
    """检查音频文件"""
    audio_dir = os.path.dirname(os.path.abspath(__file__))

    required_files = [
        "violin_in_E.mp3",
        "Violas_in_E.mp3",
        "Violins_1_in_E.mp3",
        "Violins_2_in_E.mp3",
        "Oboe_1_in_E.mp3",
        "Oboe_2_in_E.mp3",
        "Organ_in_E.mp3",
        "Timpani_in_E.mp3",
        "Trumpet_in_C_1_in_E.mp3",
        "Trumpet_in_C_2_in_E.mp3",
        "Trumpet_in_C_3_in_E.mp3"
    ]

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(audio_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename}")
        else:
            print(f"⚠️  {filename} - 缺失")
            missing_files.append(filename)

    if missing_files:
        print(f"\n⚠️  警告: {len(missing_files)} 个音频文件缺失")
        print("程序可以运行，但部分音轨无声音")
        print("请将音频文件放在E_Major目录下")

    return len(missing_files) == 0

def display_system_info():
    """显示系统信息"""
    print("\n" + "="*60)
    print("  E_Major 五乐器姿态识别系统")
    print("="*60)

    system = platform.system()
    if system == "Darwin":
        system_name = "macOS"
        machine = platform.machine()
        if machine == "arm64":
            processor = "Apple Silicon"
        else:
            processor = "Intel"
    elif system == "Windows":
        system_name = "Windows"
        processor = platform.machine()
    elif system == "Linux":
        system_name = "Linux"
        processor = platform.machine()
    else:
        system_name = system
        processor = platform.machine()

    print(f"操作系统: {system_name} ({processor})")
    print(f"Python:   {platform.python_version()}")
    print("="*60 + "\n")

def main():
    """主函数"""
    display_system_info()

    print("正在检查运行环境...\n")

    # 1. 检查Python版本
    if not check_python_version():
        sys.exit(1)

    print("\n正在检查依赖包...")
    # 2. 检查依赖
    missing = check_dependencies()

    if missing:
        print("\n是否尝试自动安装缺失的依赖包？")
        response = input("输入 y/yes 确认，其他键取消: ").lower()

        if response in ['y', 'yes']:
            if not install_dependencies(missing):
                sys.exit(1)
        else:
            print("\n❌ 缺少必要依赖，无法启动")
            sys.exit(1)

    print("\n正在检查音频文件...")
    # 3. 检查音频文件
    check_audio_files()

    print("\n" + "="*60)
    print("环境检查完成！正在启动应用...")
    print("="*60 + "\n")

    # 4. 启动应用
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
    main_script = os.path.join(code_dir, "main_e_major.py")

    if not os.path.exists(main_script):
        print(f"❌ 找不到主程序: {main_script}")
        sys.exit(1)

    # 切换到code目录并运行
    os.chdir(code_dir)

    try:
        # 使用subprocess运行主程序
        subprocess.run([sys.executable, main_script])
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
