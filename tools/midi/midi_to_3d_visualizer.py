#!/usr/bin/env python3
"""
MIDI音乐转三维建筑结构可视化器
将MIDI文件中的音符转换为3D方块建筑，支持导出和预览
"""

import pretty_midi
import trimesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Dict, Tuple
import argparse

class MidiTo3DVisualizer:
    def __init__(self, midi_file_path: str):
        """
        初始化MIDI转3D可视化器
        
        Args:
            midi_file_path: MIDI文件路径
        """
        self.midi_file_path = midi_file_path
        self.midi_data = None
        self.notes_df = None
        self.blocks = []
        self.merged_model = None
        
        # 可视化参数
        self.time_scale = 5.0      # 时间轴缩放（X轴）
        self.pitch_scale = 0.5     # 音高轴缩放（Y轴）
        self.block_width = 0.8     # 方块宽度
        self.block_height = 1.0    # 方块基础高度
        self.min_duration = 0.05   # 最小持续时间，避免太薄的块
        
        print(f"🎼 初始化MIDI转3D可视化器")
        print(f"   输入文件: {midi_file_path}")
    
    def load_midi(self) -> bool:
        """加载MIDI文件"""
        try:
            print("📂 加载MIDI文件...")
            self.midi_data = pretty_midi.PrettyMIDI(self.midi_file_path)
            print(f"✅ MIDI文件加载成功")
            print(f"   总时长: {self.midi_data.get_end_time():.2f}秒")
            print(f"   乐器数量: {len(self.midi_data.instruments)}")
            return True
        except Exception as e:
            print(f"❌ MIDI文件加载失败: {e}")
            return False
    
    def extract_notes_data(self) -> pd.DataFrame:
        """提取音符数据到DataFrame"""
        print("🎵 提取音符数据...")
        
        notes_list = []
        
        for instrument_idx, instrument in enumerate(self.midi_data.instruments):
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            
            for note in instrument.notes:
                notes_list.append({
                    'start': note.start,
                    'end': note.end,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': note.end - note.start,
                    'instrument': instrument_name,
                    'instrument_idx': instrument_idx
                })
        
        self.notes_df = pd.DataFrame(notes_list)
        
        if len(self.notes_df) > 0:
            print(f"✅ 提取完成:")
            print(f"   音符总数: {len(self.notes_df)}")
            print(f"   时间范围: {self.notes_df['start'].min():.2f}s - {self.notes_df['end'].max():.2f}s")
            print(f"   音高范围: {self.notes_df['pitch'].min()} - {self.notes_df['pitch'].max()}")
            print(f"   持续时间范围: {self.notes_df['duration'].min():.3f}s - {self.notes_df['duration'].max():.3f}s")
            
            # 显示乐器信息
            instrument_counts = self.notes_df.groupby('instrument').size()
            print("   乐器分布:")
            for instrument, count in instrument_counts.items():
                print(f"     {instrument}: {count}个音符")
        else:
            print("⚠️ 未找到音符数据")
        
        return self.notes_df
    
    def notes_to_blocks(self) -> List[trimesh.Trimesh]:
        """将音符转换为3D方块"""
        print("🧱 将音符转换为3D方块...")
        
        if self.notes_df is None or len(self.notes_df) == 0:
            print("❌ 没有音符数据可转换")
            return []
        
        blocks = []
        
        # 为不同乐器设置不同颜色
        colors = [
            [255, 100, 100, 255],  # 红色
            [100, 255, 100, 255],  # 绿色
            [100, 100, 255, 255],  # 蓝色
            [255, 255, 100, 255],  # 黄色
            [255, 100, 255, 255],  # 紫色
        ]
        
        for idx, note in self.notes_df.iterrows():
            # 计算3D坐标和尺寸
            x = note['start'] * self.time_scale  # 时间 → X轴
            y = note['pitch'] * self.pitch_scale  # 音高 → Y轴  
            z = 0  # 基础Z位置
            
            # 方块尺寸
            duration = max(note['duration'], self.min_duration)  # 确保最小厚度
            length = duration * self.time_scale  # X方向长度（持续时间）
            width = self.block_width  # Y方向宽度
            height = self.block_height * (note['velocity'] / 127.0)  # Z方向高度（基于力度）
            
            # 创建方块
            box = trimesh.creation.box(
                extents=[length, width, height],
                transform=trimesh.transformations.translation_matrix([
                    x + length/2,  # 中心位置调整
                    y, 
                    z + height/2
                ])
            )
            
            # 设置颜色（根据乐器）
            color_idx = note['instrument_idx'] % len(colors)
            box.visual.face_colors = colors[color_idx]
            
            blocks.append(box)
        
        self.blocks = blocks
        print(f"✅ 创建了 {len(blocks)} 个3D方块")
        
        return blocks
    
    def create_3d_model(self) -> trimesh.Trimesh:
        """合并所有方块为单一3D模型"""
        print("🏗️ 合并3D模型...")
        
        if not self.blocks:
            print("❌ 没有方块可合并")
            return None
        
        try:
            # 合并所有方块
            self.merged_model = trimesh.util.concatenate(self.blocks)
            
            print(f"✅ 3D模型创建完成:")
            print(f"   总顶点数: {len(self.merged_model.vertices)}")
            print(f"   总面数: {len(self.merged_model.faces)}")
            print(f"   边界框: {self.merged_model.bounds}")
            
            return self.merged_model
        
        except Exception as e:
            print(f"❌ 模型合并失败: {e}")
            return None
    
    def preview_matplotlib(self, sample_ratio: float = 0.1):
        """使用matplotlib预览3D结构（采样显示）"""
        print("👁️ 生成matplotlib 3D预览...")
        
        if self.notes_df is None or len(self.notes_df) == 0:
            print("❌ 没有数据可预览")
            return
        
        # 采样数据以提高性能
        if len(self.notes_df) > 1000:
            sample_size = max(100, int(len(self.notes_df) * sample_ratio))
            df_sample = self.notes_df.sample(n=sample_size)
            print(f"   采样显示 {sample_size}/{len(self.notes_df)} 个音符")
        else:
            df_sample = self.notes_df
        
        # 创建3D图
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为不同乐器设置不同颜色
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        instruments = df_sample['instrument'].unique()
        
        for i, instrument in enumerate(instruments):
            instrument_notes = df_sample[df_sample['instrument'] == instrument]
            
            x = instrument_notes['start'] * self.time_scale
            y = instrument_notes['pitch'] * self.pitch_scale
            z = np.zeros(len(instrument_notes))  # 基础Z位置
            
            # 方块大小基于持续时间和力度
            sizes = (instrument_notes['duration'] * 100 + 50) * (instrument_notes['velocity'] / 127.0)
            
            ax.scatter(x, y, z, 
                      c=colors[i % len(colors)], 
                      s=sizes,
                      alpha=0.6,
                      label=instrument)
        
        # 设置标签和标题
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pitch (MIDI note)', fontsize=12) 
        ax.set_zlabel('Height (velocity)', fontsize=12)
        ax.set_title('MIDI Notes as 3D Architecture Preview', fontsize=16)
        
        # 添加图例
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片而不是显示
        preview_path = "midi_3d_preview.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"✅ 3D预览已保存: {preview_path}")
        plt.close()  # 关闭图形，不显示
    
    def export_obj(self, output_path: str = "midi_3d_model.obj") -> bool:
        """导出3D模型为OBJ文件"""
        print(f"💾 导出3D模型到: {output_path}")
        
        if self.merged_model is None:
            print("❌ 没有3D模型可导出")
            return False
        
        try:
            # 导出OBJ文件
            self.merged_model.export(output_path)
            
            # 获取文件信息
            file_size = os.path.getsize(output_path)
            print(f"✅ 导出成功:")
            print(f"   文件路径: {os.path.abspath(output_path)}")
            print(f"   文件大小: {file_size/1024:.1f} KB")
            print(f"   格式: OBJ")
            print(f"   可用软件打开: Blender, Rhino, MeshLab等")
            
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    def export_stats(self) -> Dict:
        """导出统计信息"""
        if self.notes_df is None:
            return {}
        
        stats = {
            'total_notes': len(self.notes_df),
            'duration': self.notes_df['end'].max(),
            'pitch_range': (self.notes_df['pitch'].min(), self.notes_df['pitch'].max()),
            'instruments': list(self.notes_df['instrument'].unique()),
            'avg_note_duration': self.notes_df['duration'].mean(),
            'total_blocks': len(self.blocks) if self.blocks else 0
        }
        
        return stats
    
    def process_full_pipeline(self, preview: bool = True, export_obj: bool = True, 
                             output_path: str = None):
        """执行完整的处理流程"""
        print("\n🚀 开始MIDI转3D建筑结构完整流程")
        print("=" * 60)
        
        # 1. 加载MIDI
        if not self.load_midi():
            return False
        
        # 2. 提取音符数据
        self.extract_notes_data()
        
        # 3. 转换为3D方块
        self.notes_to_blocks()
        
        # 4. 创建3D模型
        self.create_3d_model()
        
        # 5. 预览（可选）
        if preview:
            self.preview_matplotlib()
        
        # 6. 导出OBJ（可选）
        if export_obj:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(self.midi_file_path))[0]
                output_path = f"{base_name}_3d_model.obj"
            self.export_obj(output_path)
        
        # 7. 打印统计信息
        stats = self.export_stats()
        print(f"\n📊 处理统计:")
        print(f"   总音符数: {stats.get('total_notes', 0)}")
        print(f"   音乐时长: {stats.get('duration', 0):.2f}秒")
        print(f"   音高范围: {stats.get('pitch_range', (0,0))}")
        print(f"   乐器: {', '.join(stats.get('instruments', []))}")
        print(f"   3D方块数: {stats.get('total_blocks', 0)}")
        
        print("\n✅ 完整流程处理完成!")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MIDI转3D建筑结构可视化器')
    parser.add_argument('midi_file', help='输入的MIDI文件路径')
    parser.add_argument('--no-preview', action='store_true', help='跳过matplotlib预览')
    parser.add_argument('--no-export', action='store_true', help='跳过OBJ导出')
    parser.add_argument('--output', '-o', help='输出OBJ文件路径')
    parser.add_argument('--time-scale', type=float, default=5.0, help='时间轴缩放系数')
    parser.add_argument('--pitch-scale', type=float, default=0.5, help='音高轴缩放系数')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.midi_file):
        print(f"❌ MIDI文件不存在: {args.midi_file}")
        return
    
    # 创建可视化器
    visualizer = MidiTo3DVisualizer(args.midi_file)
    
    # 设置参数
    visualizer.time_scale = args.time_scale
    visualizer.pitch_scale = args.pitch_scale
    
    # 执行处理流程
    visualizer.process_full_pipeline(
        preview=not args.no_preview,
        export_obj=not args.no_export,
        output_path=args.output
    )

def demo():
    """演示函数，使用示例MIDI文件"""
    print("🎯 MIDI转3D建筑结构可视化 - 演示模式")
    
    # 检查示例文件是否存在
    sample_file = "sample_music.mid"
    if not os.path.exists(sample_file):
        print(f"❌ 示例MIDI文件不存在: {sample_file}")
        print("   请先运行 python create_sample_midi.py 创建示例文件")
        return
    
    # 创建并运行可视化器
    visualizer = MidiTo3DVisualizer(sample_file)
    visualizer.process_full_pipeline()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # 无参数时运行演示
        demo()
    else:
        # 有参数时解析命令行
        main()