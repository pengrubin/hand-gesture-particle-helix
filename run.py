#!/usr/bin/env python3
"""
手势控制粒子球形效果 - 启动脚本
纯Python实现，无需TouchDesigner
"""

import sys
import os
import subprocess

def check_dependencies():
    """检查依赖库"""
    print("检查Python依赖库...")
    
    required_packages = [
        'cv2',
        'mediapipe', 
        'numpy',
        'pygame',
        'OpenGL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ OpenCV: {cv2.__version__}")
            elif package == 'mediapipe':
                import mediapipe as mp
                print(f"✓ MediaPipe: {mp.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✓ NumPy: {np.__version__}")
            elif package == 'pygame':
                import pygame
                print(f"✓ Pygame: {pygame.__version__}")
            elif package == 'OpenGL':
                import OpenGL
                print(f"✓ PyOpenGL: {OpenGL.__version__}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}: 未安装")
    
    return missing_packages

def install_dependencies():
    """安装缺失的依赖库"""
    print("\n安装依赖库...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ 依赖库安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖库安装失败: {e}")
        return False

def check_camera():
    """检查摄像头"""
    print("\n检查摄像头...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ 无法打开摄像头")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ 无法读取摄像头数据")
            return False
        
        print("✓ 摄像头正常")
        return True
        
    except Exception as e:
        print(f"✗ 摄像头检查失败: {e}")
        return False

def show_help():
    """显示帮助信息"""
    print("""
=== 手势控制粒子球形效果应用 ===

启动方法：
    python run.py              # 启动完整应用
    python run.py test          # 启动波浪形状测试模式

系统要求：
    - Python 3.7+
    - 摄像头设备
    - OpenGL支持的显卡

控制说明：
    鼠标控制：
    - 左键拖拽：旋转3D视角
    
    键盘控制：
    - R：重置视角
    - C：切换摄像头窗口显示/隐藏
    - W：切换线框球体显示/隐藏
    - I：切换性能信息显示/隐藏
    - 1-5：调整粒子数量 (20%, 40%, 60%, 80%, 100%)
    - ESC：退出应用
    
    手势控制 → 波浪形状：
    - 握拳 → 锯齿波：尖锐的锯齿状波浪
    - 1个手指 → 正弦波：经典的平滑波浪线  
    - 2个手指 → 双重波浪：两层叠加的波浪
    - 3个手指 → 螺旋线：3D螺旋曲线
    - 4个手指 → 心形曲线：浪漫的心形轨迹
    - 张开手掌 → 3D螺旋：立体螺旋上升
    - 双手同时 → 多条平行线：5条平行波浪线
    - 手势强度：控制波浪幅度、频率、速度
    - 手部位置：影响颜色和波浪参数

故障排除：
    1. 如果摄像头无法打开：
       - 检查摄像头权限设置
       - 确保摄像头未被其他程序占用
       - 尝试不同的摄像头ID (0, 1, 2...)
    
    2. 如果渲染性能差：
       - 按数字键1-3降低粒子数量
       - 按W键关闭线框显示
       - 确保显卡驱动最新
    
    3. 如果手势识别不准确：
       - 确保光线充足
       - 手部完整在摄像头视野内
       - 避免复杂背景

参数调整：
    可以编辑 main_app.py 中的参数来调整效果：
    - max_particles: 最大粒子数量
    - sensitivity: 手势敏感度
    - smoothing: 数据平滑程度
    """)

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return
    elif len(sys.argv) > 1 and sys.argv[1] in ['-t', '--test', 'test']:
        print("启动波浪形状测试模式...")
        try:
            from test_shapes import main as test_main
            test_main()
        except ImportError as e:
            print(f"测试模式启动失败: {e}")
        return
    
    print("=== 手势控制粒子波浪效果应用启动器 ===")
    print("纯Python实现，无需TouchDesigner")
    print("支持9种波浪形状！\n")
    
    # 检查依赖
    missing = check_dependencies()
    
    if missing:
        print(f"\n缺少依赖库: {', '.join(missing)}")
        response = input("是否自动安装依赖库? (y/n): ").lower().strip()
        
        if response in ['y', 'yes', '']:
            if not install_dependencies():
                print("依赖库安装失败，请手动安装:")
                print("pip install -r requirements.txt")
                return
            
            # 重新检查
            missing = check_dependencies()
            if missing:
                print(f"仍有依赖库缺失: {', '.join(missing)}")
                return
        else:
            print("请手动安装依赖库:")
            print("pip install -r requirements.txt")
            return
    
    # 检查摄像头
    if not check_camera():
        response = input("摄像头检查失败，是否继续启动? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    
    # 启动主应用
    print("\n正在启动应用...")
    
    try:
        from main_app import main as app_main
        app_main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有文件都在同一目录中")
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()