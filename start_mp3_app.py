#!/usr/bin/env python3
"""
启动MP3版本的手势粒子应用
"""

import sys
import os

def main():
    print("🎵 启动MP3版本的手势粒子应用")
    print("="*50)
    
    # 检查MP3文件
    mp3_files = [
        "Fugue in G Trio violin-Violin.mp3",
        "Fugue in G Trio-Tenor_Lute.mp3", 
        "Fugue in G Trio Organ-Organ.mp3"
    ]
    
    print("检查MP3文件:")
    all_exist = True
    for mp3_file in mp3_files:
        if os.path.exists(mp3_file):
            print(f"✅ {mp3_file}")
        else:
            print(f"❌ 缺失: {mp3_file}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 缺少MP3文件，无法启动")
        return
    
    print("\n🎮 手势控制说明:")
    print("- 1个手指 → 小提琴音轨")
    print("- 2个手指 → 鲁特琴音轨")  
    print("- 3个手指 → 管风琴音轨")
    print("- 张开手掌 → 所有音轨")
    print("- 无手势 → 静音")
    print()
    print("键盘控制:")
    print("- R: 重置")
    print("- C: 切换摄像头显示") 
    print("- M: 音频开关")
    print("- P: 暂停/继续")
    print("- ESC: 退出")
    print()
    
    try:
        print("正在启动MP3版本应用...")
        from main_app import main as original_main
        original_main()
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 尝试直接运行:")
        print("python3 main_app.py")

if __name__ == "__main__":
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()