#!/usr/bin/env python3
"""
创建一个更复杂的MIDI文件，包含多个乐器和更丰富的音乐结构
模拟巴赫风格的赋格曲片段
"""

import pretty_midi
import numpy as np

def create_complex_midi():
    """创建一个复杂的多声部MIDI文件"""
    
    # 创建MIDI对象
    midi = pretty_midi.PrettyMIDI()
    
    # 创建多个乐器音轨
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    violin_program = pretty_midi.instrument_name_to_program('Violin')
    violin = pretty_midi.Instrument(program=violin_program)
    
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    
    # 定义音阶和和弦
    c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C大调音阶
    
    # 1. 钢琴：主旋律
    melody_pattern = [
        (0, 60, 0.5, 80),   # C
        (0.5, 64, 0.5, 85), # E
        (1.0, 67, 0.5, 90), # G
        (1.5, 72, 1.0, 95), # C高八度
        (2.5, 69, 0.5, 80), # A
        (3.0, 67, 0.5, 85), # G
        (3.5, 64, 0.5, 75), # E
        (4.0, 60, 1.0, 70), # C
    ]
    
    for start, pitch, duration, velocity in melody_pattern:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + duration
        )
        piano.notes.append(note)
    
    # 2. 小提琴：装饰性旋律（高音区）
    violin_pattern = [
        (1.0, 76, 0.25, 70),  # E5
        (1.25, 77, 0.25, 72), # F5
        (1.5, 79, 0.25, 75),  # G5
        (1.75, 81, 0.25, 80), # A5
        (2.0, 79, 0.5, 85),   # G5
        (2.5, 77, 0.25, 75),  # F5
        (2.75, 76, 0.25, 70), # E5
        (3.0, 74, 0.5, 80),   # D5
        (3.5, 72, 0.5, 75),   # C5
    ]
    
    for start, pitch, duration, velocity in violin_pattern:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + duration
        )
        violin.notes.append(note)
    
    # 3. 大提琴：低音基础
    cello_pattern = [
        (0, 36, 2.0, 90),    # C2 - 长低音
        (2.0, 31, 1.0, 85),  # G1
        (3.0, 33, 1.0, 80),  # A1
        (4.0, 36, 1.0, 90),  # C2
    ]
    
    for start, pitch, duration, velocity in cello_pattern:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + duration
        )
        cello.notes.append(note)
    
    # 4. 添加一些和弦（钢琴）
    chord_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    chord_progressions = [
        [48, 52, 55],  # C大三和弦低音位
        [47, 50, 54],  # Dm和弦
        [45, 48, 52],  # F大三和弦
        [43, 47, 50],  # G大三和弦
        [48, 52, 55],  # C大三和弦
    ]
    
    for i, (time, chord) in enumerate(zip(chord_times, chord_progressions)):
        for pitch in chord:
            note = pretty_midi.Note(
                velocity=60,  # 较轻的和弦背景
                pitch=pitch,
                start=time,
                end=time + 0.8
            )
            piano.notes.append(note)
    
    # 5. 添加装饰音（小提琴）
    ornament_times = np.linspace(0.5, 4.5, 8)
    ornament_pitches = [84, 83, 81, 79, 77, 76, 74, 72]  # 下行装饰音
    
    for time, pitch in zip(ornament_times, ornament_pitches):
        note = pretty_midi.Note(
            velocity=50,
            pitch=pitch,
            start=time,
            end=time + 0.1  # 很短的装饰音
        )
        violin.notes.append(note)
    
    # 添加所有乐器到MIDI
    midi.instruments.extend([piano, violin, cello])
    
    return midi

def main():
    """主函数"""
    print("创建复杂的多声部MIDI文件...")
    
    # 创建MIDI
    midi = create_complex_midi()
    
    # 保存文件
    output_path = "complex_music.mid"
    midi.write(output_path)
    
    total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
    
    print(f"✅ 复杂MIDI文件已创建: {output_path}")
    print(f"   总时长: {midi.get_end_time():.2f}秒")
    print(f"   乐器数量: {len(midi.instruments)}")
    print(f"   总音符数: {total_notes}")
    
    for i, instrument in enumerate(midi.instruments):
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        pitch_min = min(n.pitch for n in instrument.notes) if instrument.notes else 0
        pitch_max = max(n.pitch for n in instrument.notes) if instrument.notes else 0
        print(f"   {instrument_name}: {len(instrument.notes)}音符, 音高{pitch_min}-{pitch_max}")

if __name__ == "__main__":
    main()