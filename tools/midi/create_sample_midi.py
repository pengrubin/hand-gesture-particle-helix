#!/usr/bin/env python3
"""
创建示例MIDI文件用于3D可视化测试
生成简单的巴赫风格和弦进行
"""

import pretty_midi
import numpy as np

def create_sample_midi():
    """创建一个简单的示例MIDI文件"""
    
    # 创建MIDI对象
    midi = pretty_midi.PrettyMIDI()
    
    # 创建钢琴音轨
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # 定义简单的和弦进行（C大调）
    # C Major - Am - F Major - G Major
    chords = [
        [60, 64, 67],  # C Major (C-E-G)
        [57, 60, 64],  # A minor (A-C-E) 
        [53, 57, 60],  # F Major (F-A-C)
        [55, 59, 62],  # G Major (G-B-D)
    ]
    
    # 添加低音线
    bass_notes = [48, 45, 41, 43]  # C, A, F, G (低八度)
    
    # 添加旋律线
    melody_notes = [72, 74, 76, 77, 79, 77, 76, 74]  # C5-D5-E5-F5-G5-F5-E5-D5
    
    current_time = 0.0
    
    # 添加和弦（每个和弦持续1秒）
    for i, chord in enumerate(chords):
        for pitch in chord:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=current_time + i * 1.0,
                end=current_time + i * 1.0 + 0.8  # 稍微短一点，避免重叠
            )
            piano.notes.append(note)
    
    # 添加低音（每个低音持续1秒）
    for i, pitch in enumerate(bass_notes):
        note = pretty_midi.Note(
            velocity=90,
            pitch=pitch,
            start=current_time + i * 1.0,
            end=current_time + i * 1.0 + 1.0
        )
        piano.notes.append(note)
    
    # 添加旋律线（每个音符0.5秒）
    for i, pitch in enumerate(melody_notes):
        note = pretty_midi.Note(
            velocity=70,
            pitch=pitch,
            start=current_time + i * 0.5,
            end=current_time + i * 0.5 + 0.4
        )
        piano.notes.append(note)
    
    # 添加乐器到MIDI
    midi.instruments.append(piano)
    
    return midi

def main():
    """主函数"""
    print("创建示例MIDI文件...")
    
    # 创建MIDI
    midi = create_sample_midi()
    
    # 保存文件
    output_path = "sample_music.mid"
    midi.write(output_path)
    
    print(f"✅ 示例MIDI文件已创建: {output_path}")
    print(f"   总时长: {midi.get_end_time():.2f}秒")
    print(f"   音符数量: {len(midi.instruments[0].notes)}")
    print(f"   音高范围: {min(n.pitch for n in midi.instruments[0].notes)} - {max(n.pitch for n in midi.instruments[0].notes)}")

if __name__ == "__main__":
    main()