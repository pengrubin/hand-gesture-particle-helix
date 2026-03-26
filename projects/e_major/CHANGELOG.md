# 更新日志

## [1.0.0] - 2025-10-25

### 跨平台支持
- ✅ 添加Windows/macOS/Linux摄像头后端自动选择
- ✅ DirectShow (Windows)、AVFoundation (macOS)、V4L2 (Linux)
- ✅ 跨平台路径处理（os.path）
- ✅ 平台特定优化（Apple Silicon、Intel）

### 文档完善
- ✅ 创建详细README.md（中英文）
- ✅ 添加INSTALL.md安装指南
- ✅ 创建GITHUB_UPLOAD_GUIDE.md上传指南
- ✅ 完善.gitignore配置

### 启动脚本
- ✅ 创建跨平台run.py启动脚本
- ✅ 自动依赖检查和安装
- ✅ 音频文件检测
- ✅ 系统信息显示

### 依赖管理
- ✅ 创建requirements.txt
- ✅ 指定最低版本要求
- ✅ 确保跨平台兼容性

### 核心功能
- ✅ 5种乐器姿态识别（Piano、Violin、Clarinet、Drum、Trumpet）
- ✅ 11轨管弦乐队音频控制
- ✅ 1.5秒手势确认机制
- ✅ 人体检测自动开始/暂停
- ✅ 92%识别准确率
- ✅ <35ms延迟
