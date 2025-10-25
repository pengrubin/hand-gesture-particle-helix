# 📦 E_Major 安装指南

## 快速安装（适合所有用户）

### Windows用户

1. **安装Python**
   - 访问 https://www.python.org/downloads/
   - 下载Python 3.9或更高版本
   - 安装时勾选 **"Add Python to PATH"**

2. **下载项目**
   ```cmd
   git clone https://github.com/yourusername/hand-gesture-particle-helix.git
   cd hand-gesture-particle-helix\E_Major
   ```

3. **运行程序**
   ```cmd
   python run.py
   ```
   脚本会自动检查并安装所需依赖

---

### macOS用户

1. **检查Python**（macOS自带Python 3）
   ```bash
   python3 --version
   ```
   如果版本低于3.7，访问 https://www.python.org/downloads/ 安装

2. **下载项目**
   ```bash
   git clone https://github.com/yourusername/hand-gesture-particle-helix.git
   cd hand-gesture-particle-helix/E_Major
   ```

3. **运行程序**
   ```bash
   python3 run.py
   ```

4. **授权摄像头**（首次运行）
   - 系统会弹出摄像头权限请求
   - 点击"允许"
   - 如果没弹出，手动设置：
     - 系统偏好设置 > 安全性与隐私 > 隐私 > 摄像头
     - 勾选"终端"或"Python"

---

### Linux用户（Ubuntu/Debian）

1. **安装依赖**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git
   ```

2. **下载项目**
   ```bash
   git clone https://github.com/yourusername/hand-gesture-particle-helix.git
   cd hand-gesture-particle-helix/E_Major
   ```

3. **运行程序**
   ```bash
   python3 run.py
   ```

4. **摄像头权限**（如果无法访问）
   ```bash
   sudo usermod -a -G video $USER
   # 注销后重新登录
   ```

---

## 手动安装（高级用户）

如果自动安装脚本遇到问题，可以手动安装：

### 1. 创建虚拟环境（推荐）

**Windows**:
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 手动运行

```bash
cd code
python main_e_major.py
```

---

## 依赖包说明

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| opencv-python | >= 4.8.0 | 摄像头和图像处理 |
| mediapipe | >= 0.10.0 | 人体姿态估计 |
| numpy | >= 1.24.0 | 数值计算 |
| pygame | >= 2.5.0 | 音频播放和窗口管理 |

---

## 音频文件准备

### 文件列表

程序需要11个MP3文件（E大调管弦乐编制）：

**主旋律**：
- `violin_in_E.mp3`

**小提琴组**：
- `Violas_in_E.mp3`
- `Violins_1_in_E.mp3`
- `Violins_2_in_E.mp3`

**单簧管组**：
- `Oboe_1_in_E.mp3`
- `Oboe_2_in_E.mp3`

**钢琴组**：
- `Organ_in_E.mp3`

**鼓组**：
- `Timpani_in_E.mp3`

**小号组**：
- `Trumpet_in_C_1_in_E.mp3`
- `Trumpet_in_C_2_in_E.mp3`
- `Trumpet_in_C_3_in_E.mp3`

### 文件放置位置

```
E_Major/
├── violin_in_E.mp3
├── Violas_in_E.mp3
├── ... (其他9个文件)
├── code/
└── run.py
```

**注意**：
- 文件必须放在 `E_Major/` 目录（不是 `code/` 目录）
- 文件名必须严格匹配（区分大小写）
- 如果缺少音频文件，程序仍可运行但无声音

---

## 验证安装

运行以下命令检查环境：

```bash
python3 -c "import cv2, mediapipe, numpy, pygame; print('✅ All dependencies OK')"
```

如果看到 `✅ All dependencies OK`，说明安装成功！

---

## 常见安装问题

### 问题1：pip不是有效命令

**解决方案**：
- 确保Python安装时勾选了"Add Python to PATH"
- 重启命令行窗口
- 使用 `python -m pip` 代替 `pip`

### 问题2：mediapipe安装失败

**Windows**：
```cmd
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mediapipe
```

**macOS Apple Silicon**：
```bash
# 确保使用ARM版Python
arch -arm64 pip3 install mediapipe
```

### 问题3：opencv无法导入

```bash
# 卸载重装
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### 问题4：pygame音频无声

**Linux**：
```bash
# 安装音频库
sudo apt install python3-pygame libsdl2-mixer-2.0-0
```

---

## 性能优化建议

### macOS Apple Silicon用户

代码已针对Apple Silicon优化，会自动：
- 降低frame_skip（提高识别精度）
- 使用AVFoundation摄像头后端
- 启用平台特定优化

### Windows用户

- 关闭不必要的后台程序
- 确保摄像头驱动是最新的
- 使用DirectShow摄像头后端（自动）

### 低配置电脑

如果运行卡顿，可以修改 `code/pose_body_detector.py`：
```python
# 第683行附近
self.config = {
    'frame_skip': 4,  # 从2改为4，降低检测频率
    # ...
}
```

---

## 下一步

安装完成后，阅读 [README.md](README.md) 了解：
- 使用指南
- 手势识别技巧
- 键盘控制

或查看 [QUICKSTART.md](code/QUICKSTART.md) 快速上手！

---

## 获取帮助

- 查看 [常见问题](README.md#常见问题)
- 提交 [GitHub Issue](https://github.com/yourusername/hand-gesture-particle-helix/issues)
- 阅读详细文档：[GESTURE_RECOGNITION_STANDARDS.md](code/GESTURE_RECOGNITION_STANDARDS.md)
