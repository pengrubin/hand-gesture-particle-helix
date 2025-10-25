# 🎵 E_Major 五种乐器识别标准完整文档

**文件位置**: `/Users/hongweipeng/hand-gesture-particle-helix/E_Major/code/pose_body_detector.py`

**最后更新**: 2024-10-16

---

## 📋 目录

1. [Piano（钢琴）](#1-piano钢琴)
2. [Violin（小提琴）](#2-violin小提琴)
3. [Clarinet（单簧管）](#3-clarinet单簧管)
4. [Drum（鼓）](#4-drum鼓)
5. [Trumpet（小号）](#5-trumpet小号)
6. [冲突解决机制](#6-冲突解决机制)
7. [识别难度对比](#7-识别难度对比)

---

## 1. Piano（钢琴）

**代码位置**: `pose_body_detector.py:333-387`

### 识别条件

| 条件 | 标准 | 代码行 |
|------|------|--------|
| **双手水平对齐** | `abs(left_wrist.y - right_wrist.y) < 0.1` | 353 |
| **键盘高度** | `0.4 < wrist.y < 0.7`（双手） | 356 |
| **速度上限** | `vertical_velocity < 0.12` | 374 |

### 置信度计算

```python
# 有按键动作（0.02 < velocity < 0.15）: confidence = 0.9
# 只有位置满足: confidence = 0.6
# 都不满足: confidence = 0.0
```

### 通过阈值

```python
if is_piano and conf_piano > 0.5:  # 最低阈值
    detected_instruments['piano'] = conf_piano
```

### 标准姿势

```
     👤 正面

  🖐️ ━━━━━ 🖐️  双手水平，y坐标差 < 10%
     (腰到胸部高度，y=0.4-0.7)

  ↕️ 小幅上下运动（velocity < 0.12）
```

### 关键参数解释

- **y坐标**: 0=头顶，1=脚底，0.4-0.7约为腰到胸部
- **水平对齐**: 双手y坐标差小于10%屏幕高度（约48px）
- **速度上限**: 排除快速运动（防止误判为Drum）

---

## 2. Violin（小提琴）

**代码位置**: `pose_body_detector.py:175-288`

### 识别条件

| 条件 | 标准 | 代码行 |
|------|------|--------|
| **左手持琴** | `abs(left_wrist.y - left_shoulder.y) < 0.4` | 221 |
| **左手前伸** | `abs(left_wrist.x - left_shoulder.x) < 0.3` | 227 |
| **右手拉弓** | `horizontal_velocity > 0.05` | 251 |
| **双手距离** | `distance < 0.5`（可选加分） | 262 |

### 置信度计算

```python
confidence = (
    left_hand_position_score * 0.4 +    # 左手持琴
    right_hand_bowing_score * 0.4 +     # 右手拉弓
    right_hand_position_score * 0.2     # 右手位置
)
```

### 通过阈值

```python
if is_violin and conf_violin > 0.6:
    detected_instruments['violin'] = conf_violin
```

### 标准姿势

```
     👤 正面

    🖐️ ← 左手举起（接近肩膀高度）
        |
        ↔️ → 🖐️ 右手左右拉弓（横向运动）
```

### 关键参数解释

- **持琴**: 左手腕y坐标接近左肩（差值 < 40%屏幕高度）
- **拉弓**: 右手横向速度 > 0.05（轻微移动即可）
- **宽松标准**: 适合业余学习者

---

## 3. Clarinet（单簧管）

**代码位置**: `pose_body_detector.py:291-330`

### 识别条件

| 条件 | 标准 | 代码行 |
|------|------|--------|
| **双手垂直对齐** | `abs(left_wrist.x - right_wrist.x) < 0.15` | 311 |
| **中间高度** | `0.3 < wrist.y < 0.6`（双手） | 314 |
| **垂直排列** | `abs(left_wrist.y - right_wrist.y) > 0.15` | 318 |

### 置信度计算

```python
# 全部满足: confidence = 0.8
# 否则: confidence = 0.0
```

### 通过阈值

```python
if is_clarinet and conf_clarinet > 0.6:
    detected_instruments['clarinet'] = conf_clarinet
```

### 标准姿势

```
     👤 正面

      🖐️ ← 上手（胸部高度）
       |
       |  x坐标差 < 15%
       ↓
      🖐️ ← 下手（腹部高度）

  y坐标差 > 15%（约10-15cm）
```

### 关键参数解释

- **垂直对齐**: 双手x坐标差 < 15%屏幕宽度（约96px）
- **垂直间距**: 双手y坐标差 > 15%屏幕高度
- **高度范围**: y=0.3-0.6（胸部到腹部）

---

## 4. Drum（鼓）

**代码位置**: `pose_body_detector.py:389-453`

### 识别条件（最新修改）

| 条件 | 标准 | 代码行 |
|------|------|--------|
| **快速上下运动** | `max(left_velocity, right_velocity) > 0.05` | 428-431 |
| **鼓的高度** | `0.3 < wrist.y < 0.6`（任一手） | 434 |
| **双手不水平** | `abs(left_wrist.y - right_wrist.y) > 0.15` | 437 |

### 双手检测机制

```python
# 分别计算左右手速度
right_velocity = abs(curr_y_r - prev_y_r) / dt_r
left_velocity = abs(curr_y_l - prev_y_l) / dt_l

# 使用最大速度
vertical_velocity = max(right_velocity, left_velocity)
```

### 置信度计算

```python
confidence = min(1.0, vertical_velocity * 5) if is_drum else 0.0
```

### 通过阈值

```python
if is_drum and conf_drum > 0.6:
    detected_instruments['drum'] = conf_drum
```

### 标准姿势

```
     👤 正面

      ↑
    🖐️  ← 快速上下挥动（velocity > 0.05）
      ↓

    🖐️  ← 另一只手可以静止或不同高度

  双手y坐标差 > 15%（防止误判为Piano）
```

### 关键参数解释

- **速度要求**: > 0.05（中等速度，比Piano快但不需极快）
- **双手支持**: 任一手达到速度即可
- **区分Piano**: 双手y坐标差必须 > 15%

---

## 5. Trumpet（小号）

**代码位置**: `pose_body_detector.py:455-488`

### 识别条件

| 条件 | 标准 | 代码行 |
|------|------|--------|
| **双手高位** | `left_wrist.y < 0.4 AND right_wrist.y < 0.4` | 454 |
| **双手靠近** | `hand_distance < 0.25` | 461 |

### 距离计算

```python
hand_distance = sqrt(
    (left_wrist.x - right_wrist.x)^2 +
    (left_wrist.y - right_wrist.y)^2
)
```

### 置信度计算

```python
# 全部满足: confidence = 0.85
# 否则: confidence = 0.0
```

### 通过阈值

```python
if is_trumpet and conf_trumpet > 0.55:  # 降低后的阈值
    detected_instruments['trumpet'] = conf_trumpet
```

### 标准姿势

```
     👤 正面

    🖐️🖐️ ← 双手高举（胸部以上，y < 0.4）
      ||    双手靠近（间距 < 25%屏幕对角线）
```

### 关键参数解释

- **高度要求**: y < 0.4（胸部以上到头部）
- **靠近距离**: 欧几里得距离 < 0.25
- **降低阈值**: 从0.7降到0.55，更容易识别

---

## 6. 冲突解决机制

### Piano vs Drum 冲突

**问题场景**: 双手上下挥动

| 乐器 | 判定条件 | 结果 |
|------|----------|------|
| Piano | `hands_level (y差<0.1)` + `velocity<0.12` | ✓ 慢速水平 |
| Drum | `hands_not_level (y差>0.15)` + `velocity>0.05` | ✓ 快速不水平 |

**解决方案**:
- Piano添加速度上限（< 0.12）→ 排除快速运动
- Drum添加"双手不水平"条件 → 排除水平姿势

### 速度区间对比

```
Violin横向: ────────────────> 0.05
Piano按键:  ─────> 0.02-0.15 (上限0.12)
Drum打击:   ────────────────> 0.05
            |       |        |
            慢      中等      快
```

**关键**:
- Piano和Drum的速度有重叠（0.05-0.12）
- 通过"双手水平/不水平"区分

---

## 7. 识别难度对比

### 难度排行

| 排名 | 乐器 | 阈值 | 主要条件 | 难度 |
|------|------|------|----------|------|
| 1 | **Piano** | 0.5 | 双手水平 + 高度 | ⭐ 最容易 |
| 2 | **Violin** | 0.6 | 左手举 + 右手动 | ⭐⭐ 容易 |
| 3 | **Clarinet** | 0.6 | 双手垂直对齐 | ⭐⭐⭐ 适中 |
| 4 | **Drum** | 0.6 | 双手快速 + 不水平 | ⭐⭐⭐ 适中 |
| 5 | **Trumpet** | 0.55 | 双手高位 + 靠近 | ⭐⭐⭐ 适中 |

### 阈值对比

```
Piano:    0.5 ████████████████
Trumpet:  0.55 █████████████████
Violin:   0.6 ██████████████████
Clarinet: 0.6 ██████████████████
Drum:     0.6 ██████████████████
```

### 条件数量对比

| 乐器 | 必要条件数 | 可选条件数 | 总复杂度 |
|------|-----------|-----------|----------|
| Piano | 3 | 1 | 中 |
| Violin | 2 | 1 | 中 |
| Clarinet | 3 | 0 | 高 |
| Drum | 3 | 0 | 高 |
| Trumpet | 2 | 0 | 低 |

---

## 8. MediaPipe坐标系统

### 坐标范围

```
x: 0.0 (左) ──────→ 1.0 (右)
y: 0.0 (上) ──────→ 1.0 (下)

       0.0 (头顶)
        |
      0.3 (胸部)
        |
      0.5 (腰部)
        |
      0.7 (膝盖)
        |
       1.0 (脚底)
```

### 关键点索引

| 索引 | 名称 | 用途 |
|------|------|------|
| 11 | left_shoulder | Violin左手持琴基准 |
| 12 | right_shoulder | Violin右手拉弓基准 |
| 15 | left_wrist | 所有乐器左手检测 |
| 16 | right_wrist | 所有乐器右手检测 |

---

## 9. 速度计算方法

### 横向速度（Violin）

```python
horizontal_velocity = abs(curr_x - prev_x) / dt
```

### 垂直速度（Piano, Drum）

```python
vertical_velocity = abs(curr_y - prev_y) / dt
```

### 时间间隔

```python
dt = curr_time - prev_time  # 通常约33ms (30 FPS)
```

### 速度单位

```
velocity = 坐标变化量 / 秒
例如: velocity = 0.1 表示每秒移动屏幕高度的10%
```

---

## 10. 实际测试建议

### 测试顺序

1. **Piano** - 最简单，先测试
2. **Violin** - 较简单
3. **Trumpet** - 中等难度
4. **Clarinet** - 较难（需要x轴对齐）
5. **Drum** - 中等难度（需要速度和不水平）

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Piano误判为Drum | 速度太快 | 放慢手部运动 |
| Drum识别不出 | 双手太水平 | 增加双手高度差 |
| Clarinet识别不出 | x轴对齐不够 | 双手更垂直对齐 |
| Trumpet识别不出 | 手不够高 | 举到胸部以上 |
| Violin识别不出 | 速度太慢 | 增加拉弓速度 |

---

## 11. 参数调整建议

如果某个乐器还是难识别，可以调整以下参数：

### Clarinet x轴对齐宽松度
```python
# 第311行
hands_vertical = abs(left_wrist.x - right_wrist.x) < 0.20  # 从0.15改为0.20
```

### Drum速度进一步降低
```python
# 第431行
fast_drumming = vertical_velocity > 0.03  # 从0.05改为0.03
```

### Trumpet高度进一步降低
```python
# 第454行
both_hands_high = (left_wrist.y < 0.45) and (right_wrist.y < 0.45)  # 从0.4改为0.45
```

### Piano速度上限放宽
```python
# 第374行
not_too_fast = vertical_velocity < 0.15  # 从0.12改为0.15
```

---

**生成时间**: 2024-10-16
**代码版本**: E_Major v2.0 - Multi-Instrument Recognition
**作者**: Claude Code with 方案1（直接修改）
