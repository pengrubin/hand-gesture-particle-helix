# MIDI音乐转三维建筑结构可视化器

将MIDI音乐文件转换为3D"音符方块建筑"，每个音符表示为一个长方体，可导出为OBJ文件在3D软件中查看。

## 🎯 功能特点

- **3D映射**: 
  - X轴：时间（音符开始时间）
  - Y轴：音高（MIDI音符编号）
  - Z轴：高度（基于音符力度）
  
- **方块尺寸**:
  - 长度：音符持续时间
  - 宽度：固定值（可调）
  - 高度：音符力度映射

- **多乐器支持**: 不同乐器用不同颜色区分
- **预览功能**: matplotlib 3D散点图预览
- **格式导出**: 支持OBJ格式，可在Blender、Rhino等软件打开

## 📦 安装依赖

```bash
pip install pretty_midi trimesh pandas matplotlib numpy
```

## 🚀 使用方法

### 基本用法

```bash
# 使用示例MIDI文件（演示模式）
python midi_to_3d_visualizer.py

# 使用指定的MIDI文件
python midi_to_3d_visualizer.py your_music.mid
```

### 命令行参数

```bash
python midi_to_3d_visualizer.py [MIDI文件] [选项]

选项:
  --no-preview          跳过matplotlib预览
  --no-export          跳过OBJ文件导出
  --output, -o         指定输出OBJ文件路径
  --time-scale         时间轴缩放系数 (默认: 5.0)
  --pitch-scale        音高轴缩放系数 (默认: 0.5)
```

### 使用示例

```bash
# 生成3D模型并保存预览图
python midi_to_3d_visualizer.py song.mid --output song_3d.obj

# 只生成模型，跳过预览
python midi_to_3d_visualizer.py song.mid --no-preview

# 调整缩放参数
python midi_to_3d_visualizer.py song.mid --time-scale 8.0 --pitch-scale 1.0
```

## 📁 文件结构

```
midi_to_3d_visualizer.py    # 主程序
create_sample_midi.py       # 创建示例MIDI文件
create_complex_midi.py      # 创建复杂多声部MIDI文件
sample_music.mid            # 简单示例MIDI
complex_music.mid           # 复杂示例MIDI
*.obj                       # 生成的3D模型文件
midi_3d_preview.png         # 预览图像
```

## 🎼 示例文件

### 创建简单示例

```bash
python create_sample_midi.py
```

生成包含：
- 基本和弦进行（C大调）
- 简单旋律线
- 低音伴奏

### 创建复杂示例

```bash
python create_complex_midi.py  
```

生成包含：
- 多乐器编排（钢琴、小提琴、大提琴）
- 巴赫风格对位
- 装饰音和和弦

## 🏗️ 3D建筑效果

生成的3D模型呈现出类似建筑的效果：

- **低音部分**: 底部厚重的"基础建筑"
- **中音部分**: 中层的"主体结构"  
- **高音部分**: 顶部的"塔尖装饰"
- **和弦**: 形成"建筑群组"
- **旋律**: 连贯的"建筑线条"

## 🛠️ 技术实现

### 核心库

- `pretty_midi`: MIDI文件解析
- `trimesh`: 3D几何体创建和导出
- `pandas`: 数据处理
- `matplotlib`: 3D可视化预览

### 处理流程

1. **MIDI解析**: 提取音符的时间、音高、持续时间、力度
2. **数据映射**: 转换为3D坐标和方块尺寸
3. **模型构建**: 使用trimesh创建每个方块
4. **模型合并**: 将所有方块合并为单一模型
5. **预览生成**: matplotlib 3D散点图
6. **文件导出**: 保存为OBJ格式

## 🎨 可视化效果

### 颜色方案

- 红色：乐器1（通常是钢琴）
- 绿色：乐器2（如小提琴）
- 蓝色：乐器3（如大提琴）
- 黄色：乐器4
- 紫色：乐器5

### 在3D软件中查看

**Blender (免费)**:
1. 打开Blender
2. File → Import → Wavefront (.obj)
3. 选择生成的.obj文件

**Rhino**:
1. 打开Rhino
2. File → Import
3. 选择.obj文件

**在线查看**:
- 使用在线3D查看器如：
  - 3D Viewer Online
  - Sketchfab

## 📊 输出信息

程序会显示详细的处理信息：

```
📊 处理统计:
   总音符数: 44
   音乐时长: 5.00秒  
   音高范围: (31, 84)
   乐器: Acoustic Grand Piano, Violin, Cello
   3D方块数: 44
```

## ⚙️ 参数调节

### 时间缩放 (`--time-scale`)
- 较大值：音符间距更宽，适合长乐曲
- 较小值：音符紧密排列，适合短片段

### 音高缩放 (`--pitch-scale`)  
- 较大值：音高差异更明显
- 较小值：音高更紧凑

## 🔧 自定义参数

编辑`midi_to_3d_visualizer.py`中的参数：

```python
self.time_scale = 5.0      # 时间轴缩放
self.pitch_scale = 0.5     # 音高轴缩放  
self.block_width = 0.8     # 方块宽度
self.block_height = 1.0    # 方块基础高度
self.min_duration = 0.05   # 最小持续时间
```

## 🎵 支持的MIDI格式

- 标准MIDI文件(.mid, .midi)
- 多音轨MIDI
- 多种乐器编排
- 不同力度和时值

## ⚠️ 注意事项

- 大型MIDI文件可能生成很大的3D模型
- 建议先用预览模式检查效果
- OBJ文件可能需要在3D软件中调整材质
- 复杂音乐可能产生密集的几何体

## 📈 性能优化

- 使用`sample_ratio`参数控制预览采样率
- 调整`min_duration`避免过薄方块
- 大文件建议使用`--no-preview`跳过预览

## 🤝 扩展功能

可以进一步开发：
- 材质和纹理支持
- 动画导出
- VR/AR兼容性
- 更多3D格式支持
- 实时MIDI输入

---

🎼 **享受将音乐转化为建筑艺术的过程！** 🏗️