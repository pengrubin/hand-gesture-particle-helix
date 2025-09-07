#!/usr/bin/env python3
"""
分析MIDI文件的详细结构，包括音轨和声部信息
"""

import pretty_midi
import sys

def analyze_midi_structure(midi_file_path):
    """详细分析MIDI文件的结构"""
    print(f"\n🎼 分析MIDI文件: {midi_file_path}")
    print("=" * 60)
    
    try:
        # 加载MIDI文件
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        
        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"   总时长: {midi_data.get_end_time():.2f}秒")
        print(f"   乐器数量: {len(midi_data.instruments)}")
        print(f"   节拍数: {len(midi_data.get_beats())}")
        print(f"   速度变化: {len(midi_data.get_tempo_changes()[0])}次")
        
        # 详细的乐器/音轨信息
        print(f"\n🎹 乐器/音轨详细信息:")
        total_notes = 0
        
        for i, instrument in enumerate(midi_data.instruments):
            # 获取乐器名称
            if instrument.is_drum:
                instrument_name = "Drum Kit"
            else:
                instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            
            # 统计音符信息
            notes = instrument.notes
            total_notes += len(notes)
            
            if notes:
                pitch_min = min(n.pitch for n in notes)
                pitch_max = max(n.pitch for n in notes)
                velocity_min = min(n.velocity for n in notes)
                velocity_max = max(n.velocity for n in notes)
                
                # 分析音符分布（可能的声部）
                pitch_ranges = analyze_pitch_ranges(notes)
                
                print(f"\n   音轨 {i+1}: {instrument_name}")
                print(f"      程序号: {instrument.program}")
                print(f"      音符数: {len(notes)}")
                print(f"      音高范围: {pitch_min}-{pitch_max} (MIDI编号)")
                print(f"      音高范围: {pretty_midi.note_number_to_name(pitch_min)}-{pretty_midi.note_number_to_name(pitch_max)}")
                print(f"      力度范围: {velocity_min}-{velocity_max}")
                
                # 显示可能的声部
                if len(pitch_ranges) > 1:
                    print(f"      🎵 检测到 {len(pitch_ranges)} 个可能的声部:")
                    for j, (low, high, count) in enumerate(pitch_ranges):
                        print(f"         声部{j+1}: {pretty_midi.note_number_to_name(low)}-{pretty_midi.note_number_to_name(high)} ({count}个音符)")
            else:
                print(f"\n   音轨 {i+1}: {instrument_name} (无音符)")
        
        print(f"\n   📝 总音符数: {total_notes}")
        
        # 分析同时发声的音符（和弦/复调）
        print(f"\n🎶 复调分析:")
        polyphony_info = analyze_polyphony(midi_data)
        print(f"   最大同时发声数: {polyphony_info['max_polyphony']}")
        print(f"   平均同时发声数: {polyphony_info['avg_polyphony']:.2f}")
        
        if polyphony_info['max_polyphony'] > 1:
            print(f"   → 这是一个复调/多声部作品")
            if len(midi_data.instruments) == 1:
                print(f"   → 虽然只有一个乐器，但包含多个声部（如钢琴的左右手）")
        
        return midi_data
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def analyze_pitch_ranges(notes):
    """分析音符的音高范围，识别可能的声部"""
    if not notes:
        return []
    
    # 按音高分组
    pitch_groups = {}
    for note in notes:
        pitch = note.pitch
        if pitch not in pitch_groups:
            pitch_groups[pitch] = []
        pitch_groups[pitch].append(note)
    
    # 尝试识别声部（基于音高范围的间隙）
    sorted_pitches = sorted(pitch_groups.keys())
    ranges = []
    current_range_start = sorted_pitches[0]
    current_range_end = sorted_pitches[0]
    current_count = len(pitch_groups[sorted_pitches[0]])
    
    for i in range(1, len(sorted_pitches)):
        # 如果音高间隔大于12（一个八度），可能是不同声部
        if sorted_pitches[i] - current_range_end > 12:
            ranges.append((current_range_start, current_range_end, current_count))
            current_range_start = sorted_pitches[i]
            current_range_end = sorted_pitches[i]
            current_count = len(pitch_groups[sorted_pitches[i]])
        else:
            current_range_end = sorted_pitches[i]
            current_count += len(pitch_groups[sorted_pitches[i]])
    
    # 添加最后一个范围
    ranges.append((current_range_start, current_range_end, current_count))
    
    return ranges

def analyze_polyphony(midi_data):
    """分析复调信息"""
    # 获取所有音符的开始和结束时间
    all_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, note.end))
    
    if not all_notes:
        return {'max_polyphony': 0, 'avg_polyphony': 0}
    
    # 按时间排序
    all_notes.sort()
    
    # 计算每个时间点的同时发声数
    max_polyphony = 0
    total_polyphony = 0
    sample_points = 100  # 采样点数
    
    duration = midi_data.get_end_time()
    for i in range(sample_points):
        time_point = (duration / sample_points) * i
        concurrent_notes = 0
        
        for start, end in all_notes:
            if start <= time_point < end:
                concurrent_notes += 1
        
        max_polyphony = max(max_polyphony, concurrent_notes)
        total_polyphony += concurrent_notes
    
    avg_polyphony = total_polyphony / sample_points if sample_points > 0 else 0
    
    return {
        'max_polyphony': max_polyphony,
        'avg_polyphony': avg_polyphony
    }

def main():
    """主函数"""
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
    else:
        # 默认分析文件
        midi_file = "vp3-1pre.mid"
    
    if not midi_file.endswith(('.mid', '.midi')):
        print(f"⚠️ 警告: {midi_file} 可能不是MIDI文件")
    
    # 分析MIDI结构
    midi_data = analyze_midi_structure(midi_file)
    
    if midi_data:
        print("\n✅ 分析完成!")
        print("\n💡 解释:")
        print("   - 一个乐器可以演奏多个声部（如钢琴的左右手）")
        print("   - 声部通过音高范围区分（低音、中音、高音等）")
        print("   - 复调音乐中，同一时间会有多个音符同时发声")

if __name__ == "__main__":
    main()