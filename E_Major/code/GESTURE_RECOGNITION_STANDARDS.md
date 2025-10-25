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
| **速度上限** | `vertical_velocity < 0.12`（必要条件） | 374 |

### 置信度计算

```python
# 有按键动作（0.02 < velocity < 0.15）且速度不太快: confidence = 0.9
# 只有位置满足且速度不太快: confidence = 0.6
# 不满足速度上限条件: confidence = 0.0
```

### 通过阈值

```python
if is_piano and conf_piano > 0.5:  # 最低阈值
    detected_instruments['piano'] = conf_piano
```

### 判定逻辑（更新）

```python
is_piano = hands_level and keyboard_height and not_too_fast
# not_too_fast = vertical_velocity < 0.12（必要条件，防止误判为Drum）
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
| **右手拉弓** | `horizontal_velocity > 0.05` | 269 |
| **运动方向占比** | `horizontal/vertical > 2.0`（必要条件） | 265 |
| **双手距离** | `distance < 0.5`（可选加分） | 274 |

### 置信度计算

```python
confidence = (
    left_hand_position_score * 0.4 +    # 左手持琴
    right_hand_bowing_score * 0.4 +     # 右手拉弓（含方向占比）
    right_hand_position_score * 0.2     # 右手位置
)

# 右手拉弓判定（更新）
right_hand_bowing = (horizontal_velocity > 0.05) AND (motion_ratio > 2.0)
# motion_ratio = horizontal_velocity / vertical_velocity
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
- **运动方向占比**: 横向速度必须是垂直速度的2倍以上（区分打鼓）
- **宽松标准**: 适合业余学习者

### 运动方向占比检测（新增）

**目的**: 区分小提琴拉弓（横向）和打鼓击打（垂直）

```python
# 计算速度比例
if vertical_velocity < 0.01:
    motion_ratio = inf  # 纯横向运动
    is_mainly_horizontal = True
else:
    motion_ratio = horizontal_velocity / vertical_velocity
    is_mainly_horizontal = motion_ratio > 2.0
```

**效果对比**:

| 动作 | horizontal_velocity | vertical_velocity | motion_ratio | is_mainly_horizontal | 判定结果 |
|------|---------------------|-------------------|--------------|---------------------|---------|
| 拉小提琴 | 0.15 | 0.02 | 7.5 | ✅ YES | Violin ✅ |
| 打鼓（有抖动） | 0.06 | 0.15 | 0.4 | ❌ NO | Drum ✅ |
| 打鼓（交替） | 0.08 | 0.12 | 0.67 | ❌ NO | Drum ✅ |

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

**代码位置**: `pose_body_detector.py:262-349`

### 识别条件（2024-10-16 优化后）

| 条件 | 标准 | 代码行 | 更新说明 |
|------|------|--------|---------|
| **快速上下运动** | `max(left_velocity, right_velocity) > 0.08` | 317 | 使用5帧滑动窗口最大值 |
| **鼓的高度** | `0.2 < wrist.y < 0.7`（任一手） | 320 | 扩大范围：0.3-0.6 → 0.2-0.7 |
| **双手不水平** | `abs(left_wrist.y - right_wrist.y) > 0.05`（必要条件） | 324 | 放宽条件：0.15 → 0.05 |

### 双手检测机制（优化后）

```python
# 使用5帧滑动窗口计算最大速度（捕捉瞬时峰值）
right_velocities = []
for i in range(1, min(5, len(right_hand_history))):
    velocity = abs(curr_y - prev_y) / dt
    right_velocities.append(velocity)

right_velocity = max(right_velocities) if right_velocities else 0.0
# 同样处理左手...

# 使用双手中的最大速度
vertical_velocity = max(right_velocity, left_velocity)
```

### 置信度计算

```python
confidence = min(1.0, vertical_velocity * 3) if is_drum else 0.0
```

### 通过阈值（降低后）

```python
if is_drum and conf_drum > 0.35:  # 从0.6降低到0.35
    detected_instruments['drum'] = conf_drum
```

### 标准姿势（更新后）

```
     👤 正面

      ↑
    🖐️  ← 快速上下挥动（velocity > 0.08）
      ↓  （高度范围：0.2-0.7，更宽松）

    🖐️  ← 另一只手可以同时击打（允许同时）

  双手y坐标差 > 5%（放宽条件，允许同时击打）
```

### 关键参数解释（优化后）

- **速度要求**: > 0.08（中等速度，使用5帧窗口捕捉峰值）
- **高度范围**: 0.2-0.7（扩大范围，从腹部到肩部）
- **双手支持**: 任一手达到速度即可
- **同时击打**: y坐标差 > 5%即可（允许同时或交替击打）
- **区分Piano**: Piano要求双手水平(y差<10%)，Drum要求y差>5%
- **滑动窗口**: 使用5帧最大值而非2帧，更容易捕捉瞬时速度峰值

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

### Violin vs Drum 冲突（已解决 ✅）

**问题场景**: 打鼓动作被误识别成小提琴

| 乐器 | 判定条件 | 结果 |
|------|----------|------|
| Violin | `left_hand_holding` + `horizontal_velocity > 0.05` + `motion_ratio > 2.0` | ✓ 横向主导 |
| Drum | `fast_drumming (> 0.08)` + `hands_not_level (> 0.15)` | ✓ 垂直主导 |

**解决方案（方案1：运动方向占比检测）**:
- Violin添加`motion_ratio > 2.0`作为**必要条件**
- 确保横向运动**显著大于**垂直运动
- 打鼓时的横向抖动或交替击打不会触发Violin

**区分效果**:
- 横向主导(ratio > 2.0) + 速度 > 0.05 → Violin ✓
- 垂直主导(ratio < 2.0) + 速度 > 0.08 → Drum ✓

---

### Piano vs Drum 冲突（已解决 ✅，2024-10-16优化）

**问题场景**: 双手上下挥动

| 乐器 | 判定条件 | 结果 |
|------|----------|------|
| Piano | `hands_level (y差<0.1)` + `velocity<0.12`（必要） | ✓ 慢速水平 |
| Drum | `hands_not_level (y差>0.05)`（必要，放宽） + `velocity>0.08` + `conf>0.35`（降低） | ✓ 快速不水平 |

**解决方案（方案1+方案2组合+优化）**:
1. Piano添加速度上限（< 0.12）作为**必要条件** → 排除快速运动
2. Drum放宽"双手不水平"（y差>0.05，从0.15降低）→ 允许同时击打
3. Drum降低置信度阈值（0.35，从0.6降低）→ 更容易触发
4. Drum使用5帧滑动窗口 → 更容易捕捉峰值速度

**区分效果（优化后）**:
- 双手水平(y差<0.05) + 慢速(0.02-0.12) → Piano ✓
- 双手略有高度差(0.05<y差<0.10) + 快速(>0.08) → Drum ✓（新增支持）
- 双手明显不水平(y差>0.10) + 快速(>0.08) → Drum ✓
- 双手水平(y差<0.05) + 快速(>0.12) → 都不触发 ✗

### 速度区间对比

```
Violin横向: ────────────────> 0.05
Piano垂直:  ─────────> 0.02-0.12 (必须<0.12)
Drum垂直:   ──────────────────> 0.08
            |       |    |   |
            慢    Piano Drum  快
                   区   区
```

**关键**:
- Piano和Drum的速度区间通过"双手水平/不水平"区分
- Piano: 0.02-0.12, 必须双手水平(y差<0.1)
- Drum: >0.08, 必须双手不水平(y差>0.15)
- 速度0.08-0.12是重叠区，但姿态不同：
  - 双手水平 → Piano
  - 双手不水平 → Drum

---

## 7. 识别难度对比

### 难度排行（2024-10-16更新）

| 排名 | 乐器 | 阈值 | 主要条件 | 难度 |
|------|------|------|----------|------|
| 1 | **Drum** | 0.35 | 双手快速 + 不水平（放宽） | ⭐ 最容易（优化后） |
| 2 | **Piano** | 0.5 | 双手水平 + 高度 | ⭐⭐ 容易 |
| 3 | **Trumpet** | 0.55 | 双手高位 + 靠近 | ⭐⭐⭐ 适中 |
| 4 | **Violin** | 0.6 | 左手举 + 右手动 | ⭐⭐⭐ 适中 |
| 5 | **Clarinet** | 0.6 | 双手垂直对齐 | ⭐⭐⭐⭐ 较难 |

### 阈值对比（更新后）

```
Drum:     0.35 ███████████ （大幅降低，最容易触发）
Piano:    0.5  ████████████████
Trumpet:  0.55 █████████████████
Violin:   0.6  ██████████████████
Clarinet: 0.6  ██████████████████
```

### 条件数量对比（更新）

| 乐器 | 必要条件数 | 可选条件数 | 总复杂度 | 备注 |
|------|-----------|-----------|----------|------|
| Piano | 3 | 1 | 中 | 新增速度上限(必要) |
| Violin | 2 | 1 | 中 | - |
| Clarinet | 3 | 0 | 高 | - |
| Drum | 3 | 0 | 高 | 新增双手不水平(必要) |
| Trumpet | 2 | 0 | 低 | - |

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

### Drum速度进一步降低（如需更宽松）
```python
# 第317行
fast_drumming = vertical_velocity > 0.06  # 从0.08改为0.06
# 注意：已使用5帧滑动窗口，通常0.08已足够宽松
```

### Drum双手不水平条件进一步宽松（不建议）
```python
# 第324行
hands_not_level = abs(left_wrist.y - right_wrist.y) > 0.03  # 从0.05改为0.03
# 警告：过低可能与Piano冲突（Piano要求y差<0.10）
# 建议保持0.05-0.10之间
```

### Drum置信度阈值进一步降低（如需更灵敏）
```python
# detect_instrument方法，约第578行
if is_drum and conf_drum > 0.25:  # 从0.35改为0.25
# 警告：过低可能误触发
```

### Trumpet高度进一步降低
```python
# 第454行
both_hands_high = (left_wrist.y < 0.45) and (right_wrist.y < 0.45)  # 从0.4改为0.45
```

### Piano速度上限放宽（慎重）
```python
# 第374行
not_too_fast = vertical_velocity < 0.15  # 从0.12改为0.15（可能与Drum冲突）
```

**注意**: Piano速度上限是区分Piano和Drum的关键，不建议超过0.12

---

**生成时间**: 2024-10-16
**代码版本**: E_Major v2.0 - Multi-Instrument Recognition
**作者**: Claude Code with 方案1（直接修改）
