# 📤 GitHub上传指南

## 准备工作清单

在上传到GitHub之前，确保完成以下准备：

### ✅ 文件检查

- [x] `requirements.txt` - Python依赖列表
- [x] `.gitignore` - Git忽略文件配置
- [x] `README.md` - 项目说明文档
- [x] `INSTALL.md` - 安装指南
- [x] `run.py` - 跨平台启动脚本
- [x] `code/` 目录 - 源代码
- [ ] 音频文件（可选，如果太大可以不上传）

### ✅ 代码检查

- [x] 跨平台摄像头支持（Windows/macOS/Linux）
- [x] 路径使用os.path（跨平台兼容）
- [x] 平台检测代码（platform.system()）
- [x] 依赖包版本兼容

---

## 方式一：使用GitHub Desktop（推荐新手）

### 1. 安装GitHub Desktop

下载并安装 [GitHub Desktop](https://desktop.github.com/)

### 2. 创建GitHub仓库

1. 登录 [GitHub网站](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写信息：
   - **Repository name**: `hand-gesture-particle-helix`
   - **Description**: `Real-time Musical Instrument Gesture Recognition System`
   - **Public/Private**: 选择Public（公开）
   - ✅ 勾选 "Add a README file"（可选）
   - 选择许可证：MIT License（推荐）
4. 点击 "Create repository"

### 3. 使用GitHub Desktop上传

1. 打开GitHub Desktop
2. 点击 "File" → "Add local repository"
3. 选择 `E_Major` 文件夹
4. 点击 "Publish repository"
5. 确认信息并发布

---

## 方式二：使用Git命令行（推荐开发者）

### 1. 初始化Git仓库

```bash
cd /path/to/hand-gesture-particle-helix/E_Major

# 初始化git（如果还没有）
git init

# 添加所有文件
git add .

# 查看状态
git status
```

### 2. 创建首次提交

```bash
# 提交文件
git commit -m "Initial commit: E_Major 5-instrument gesture recognition system

- 5 instrument gesture recognition (Piano, Violin, Clarinet, Drum, Trumpet)
- 11-track orchestral audio control
- Cross-platform support (Windows/macOS/Linux)
- MediaPipe Pose integration
- Real-time gesture detection with 92% accuracy"
```

### 3. 在GitHub创建远程仓库

1. 访问 https://github.com/new
2. 填写仓库名：`hand-gesture-particle-helix`
3. 选择Public
4. **不要**勾选"Initialize this repository with a README"（因为本地已有）
5. 点击"Create repository"

### 4. 连接远程仓库并推送

```bash
# 添加远程仓库（替换yourusername为你的GitHub用户名）
git remote add origin https://github.com/yourusername/hand-gesture-particle-helix.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

---

## 音频文件处理建议

### 选项1：不上传音频文件（推荐）

**优点**：
- 仓库体积小
- 下载快速
- 节省GitHub空间

**操作**：
音频文件已在`.gitignore`中被注释，如果要排除：

```bash
# 编辑 .gitignore，取消注释这两行：
# *.mp3
# *.wav
```

然后在README.md中说明用户需要自行准备音频文件。

---

### 选项2：使用Git LFS上传音频（适合完整体验）

如果想提供完整可运行的项目：

```bash
# 安装Git LFS
# Windows: 下载 https://git-lfs.github.com/
# macOS: brew install git-lfs
# Linux: sudo apt install git-lfs

# 初始化LFS
git lfs install

# 追踪音频文件
git lfs track "*.mp3"

# 添加.gitattributes
git add .gitattributes

# 提交
git add *.mp3
git commit -m "Add audio files with Git LFS"
git push
```

**注意**：GitHub免费账户LFS存储限额为1GB

---

### 选项3：提供下载链接

在README.md中添加音频下载链接：

```markdown
## 音频文件下载

由于文件较大，音频文件托管在以下位置：
- [百度网盘](链接) 提取码: xxxx
- [Google Drive](链接)
- [OneDrive](链接)

下载后解压到 `E_Major/` 目录
```

---

## 推荐的仓库结构

```
hand-gesture-particle-helix/
├── E_Major/                    # 主项目
│   ├── code/                   # 源代码
│   ├── *.mp3                   # 音频（可选）
│   ├── README.md               # 项目说明
│   ├── INSTALL.md              # 安装指南
│   ├── requirements.txt        # 依赖
│   ├── run.py                  # 启动脚本
│   └── .gitignore              # Git忽略
│
├── LICENSE                     # MIT许可证
└── README.md                   # 总项目说明（可选）
```

---

## 完成上传后

### 1. 添加主题标签（Topics）

在GitHub仓库页面，点击"Add topics"添加：
- `gesture-recognition`
- `mediapipe`
- `music`
- `computer-vision`
- `real-time`
- `python`
- `audio-control`
- `pose-estimation`

### 2. 编辑仓库描述

```
Real-time Musical Instrument Gesture Recognition System - Control 11-track orchestra with body gestures using MediaPipe Pose
```

### 3. 启用GitHub Pages（可选）

如果想展示项目：
1. Settings → Pages
2. Source选择"main"分支
3. 文件夹选择"/ (root)"
4. Save

### 4. 创建Release（可选）

发布第一个版本：
1. 点击"Releases" → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: `E_Major v1.0.0 - Initial Release`
4. 描述功能特性
5. 发布

---

## 后续更新代码

### 修改代码后提交

```bash
# 查看修改
git status

# 添加修改的文件
git add code/main_e_major.py

# 或添加所有修改
git add .

# 提交
git commit -m "Fix: 修复摄像头在Windows下的兼容性问题"

# 推送到GitHub
git push
```

### 常用Git命令

```bash
# 查看提交历史
git log --oneline

# 撤销未提交的修改
git checkout -- filename

# 查看差异
git diff

# 创建新分支
git checkout -b feature/new-instrument

# 合并分支
git checkout main
git merge feature/new-instrument
```

---

## 常见问题

### Q: 提示"Permission denied (publickey)"

**解决**：配置SSH密钥
```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 查看公钥
cat ~/.ssh/id_ed25519.pub

# 复制公钥到GitHub设置 > SSH and GPG keys > New SSH key
```

或使用HTTPS URL：
```bash
git remote set-url origin https://github.com/yourusername/hand-gesture-particle-helix.git
```

### Q: 文件太大无法上传

**解决**：
1. 检查`.gitignore`是否正确配置
2. 使用Git LFS处理大文件
3. 或提供外部下载链接

### Q: 如何删除已提交的大文件

```bash
# 使用git filter-branch（危险操作，谨慎使用）
git filter-branch --tree-filter 'rm -f large_file.mp3' HEAD
git push --force
```

---

## 最佳实践

### 1. 提交信息规范

使用清晰的提交信息：
```
feat: 添加鼓手势识别
fix: 修复Windows摄像头兼容性
docs: 更新README安装说明
perf: 优化姿态检测性能
refactor: 重构音频控制器代码
```

### 2. 分支策略

- `main` - 稳定版本
- `develop` - 开发分支
- `feature/xxx` - 功能开发
- `fix/xxx` - Bug修复

### 3. 代码审查

使用Pull Request进行代码审查：
```bash
# 创建功能分支
git checkout -b feature/add-flute

# 开发完成后推送
git push -u origin feature/add-flute

# 在GitHub创建Pull Request
# 审查通过后合并到main
```

---

## 下一步

上传完成后，你可以：

1. ⭐ 在README.md中添加徽章（Badges）
2. 📝 编写详细的Wiki文档
3. 🐛 创建Issue模板
4. 🤝 添加CONTRIBUTING.md贡献指南
5. 📊 集成GitHub Actions进行自动化测试

---

## 获取帮助

- GitHub文档：https://docs.github.com/
- Git教程：https://git-scm.com/book/zh/v2
- GitHub Desktop：https://desktop.github.com/

祝你上传顺利！🚀
