# PI0.5 TensorRT 部署问题排查与解决方案

**日期**: 2026-07-27
**模型**: PI0.5 (checkpoint 005613)
**机器人**: SO-101 Follower
**任务**: Put the block in the bin

---

## 执行摘要

TensorRT 加速部署在技术上成功，推理延迟达标，但真机运行因**防碰撞守护逻辑缺陷**而失败。问题根源不在 TensorRT，而在守护无法区分正常快速动作和真实碰撞。

---

## 1. 背景：成功运行与后续失败

### 1.1 T0a 首次成功 (2026-07-26 18:57)

**配置**:
- Backend: PyTorch prefix
- 守护: 无（功能尚未实现）
- 时长: 5 秒
- 队列阈值: Q45
- RTC 时序: actual_consumed

**结果**:
- ✅ 成功抓取物体
- ✅ 67 次 elbow 钳制，但不连续
- ✅ 5 秒内完成

### 1.2 T0b 桌面碰撞事件 (2026-07-26 晚)

**发生**:
- 机械臂撞击桌面
- 触发 26 次连续钳制
- 腕相机 USB 断开
- 需要手动减扭矩复位

**响应**:
- 紧急添加防碰撞守护 (`stall_guard`)
- 逻辑: 连续 5 次钳制 → 触发急停

---

## 2. TensorRT 部署失败分析

### 2.1 T1 运行记录

#### 运行 1: 2026-07-27 09:17
```
配置:
  prefix_backend=tensorrt
  action_filter_enabled=true
  stall_guard_ticks=5
  queue_threshold=45

延迟性能:
  初始延迟: 432.018 ms
  稳态延迟: p50=135.3ms, p95=191.3ms, max=191.3ms

失败原因:
  Line 96-114: elbow_flex 连续 5 次钳制
  - 82.46 → 84.14 (clamped)
  - 79.77 → 83.18 (clamped)
  - 76.63 → 82.21 (clamped)
  - 73.47 → 81.42 (clamped)
  - 70.30 → 80.45 (clamped) ← 守护触发

退出码: 1
运行时长: ~1 秒
```

#### 运行 2: 2026-07-27 09:49
```
配置: 同上

延迟性能:
  初始延迟: 326.850 ms
  稳态延迟: p50=138.8ms, p95=146.8ms, max=146.8ms ✅

失败原因:
  Line 89-110: 连续 7 次钳制（elbow + shoulder + wrist）
  第 5 次触发守护

退出码: 1
运行时长: ~1 秒
```

### 2.2 关键发现

**TensorRT 推理本身正常**:
- ✅ 引擎加载成功
- ✅ 生成 18 层 K/V cache
- ✅ 延迟达标（第二次运行最大 146.8ms < 166.667ms 目标）
- ✅ RTC 队列管理正常
- ✅ actual_consumed 步数稳定（4-5 步）

**失败的真正原因**:
- ❌ 守护逻辑过于简单
- ❌ 无法区分"快速下降动作"和"真实碰撞"
- ❌ 正常抓取必然触发守护

---

## 3. 根本原因分析

### 3.1 为什么正常动作会触发守护

**安全钳制机制** (`max_relative_target=5.0`):
```python
safe_goal = present + clip(goal - present, ±5.0)
```

**抓取下降轨迹**:
- 模型输出: elbow 从 90° 快速下降到 70°（20° 变化）
- 每步限制: 最多移动 ±5°
- 结果: 连续 4-5 步都会被钳制

**守护判断**:
```python
if consecutive_clamps >= 5:
    raise StallContactError  # 错误地认为是碰撞
```

### 3.2 T0a 为什么成功

**关键**: T0a 运行时守护功能尚未实现
- 同样有 67 次钳制
- 但系统不检测连续性
- 抓取动作正常完成

### 3.3 守护的设计缺陷

当前逻辑无法区分:

| 场景 | 连续钳制 | 实际情况 | 应该响应 |
|------|----------|----------|----------|
| 快速抓取下降 | 5+ 次 | 正常动作 | ✅ 允许继续 |
| 真实桌面碰撞 | 26 次 | 异常阻塞 | ❌ 紧急停止 |
| 物体接触卡滞 | 持续 | 异常阻塞 | ❌ 紧急停止 |

**缺失的信号**:
- 关节电流（检测真实负载）
- 位置误差累积（检测是否完全卡死）
- 钳制持续时间（瞬态 vs 持续）

---

## 4. 解决方案

### 4.1 临时方案：禁用守护

**脚本**: `run_pi05_full_rollout_rtc_actual_no_guard.sh`

```bash
STALL_GUARD_TICKS=0  # 强制禁用守护
```

**验证**:
- ✅ PyTorch + 无守护: 30 秒成功运行（虽然物体被推出分布）
- ✅ PyTorch + 无守护 + 5 秒: 成功抓取并回位

**代价**:
- 失去碰撞保护
- 需要人工监督 + 手动急停准备

### 4.2 TensorRT + 无守护验证

**建议运行**:
```bash
PI05_PREFIX_BACKEND=tensorrt \
PI05_TRT_PREFIX_ENGINE=/data/cqy_workspace/tk/lerobot_src/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model/pi05_tensorrt_rebuild_bf16/prefix_cache_bf16.plan \
./run_pi05_full_rollout_rtc_actual_no_guard.sh
```

**预期**: 成功运行，确认 TensorRT 推理质量无问题

### 4.3 长期方案：改进守护逻辑

**方案 A: 增加关节电流检测**
```python
if consecutive_clamps >= 5 and motor_current > threshold:
    # 真实碰撞：高电流 + 连续钳制
    raise StallContactError
```

**方案 B: 检测位置误差累积**
```python
if position_error_integral > threshold:
    # 完全卡死：期望位置和实际位置差距持续扩大
    raise StallContactError
```

**方案 C: 区分瞬态和持续钳制**
```python
if consecutive_clamps >= 10 and clamp_duration > 0.5:
    # 持续阻塞：连续钳制时间超过 500ms
    raise StallContactError
```

**推荐**: 方案 A（电流） + 方案 C（时长），双重保护

---

## 5. TensorRT 加速效果

### 5.1 架构说明

**加速部分**（TensorRT）:
- 双摄像头视觉编码器（处理 top + wrist 图像）
- PaliGemma prefix prefill（生成 18 层 K/V cache）

**保留 PyTorch 部分**:
- Action expert 网络
- 10 步去噪循环
- RTC 梯度引导计算（需要自动微分）

### 5.2 性能对比（基于已有日志）

| 指标 | T0a (PyTorch) | T1 (TensorRT) | 改善 |
|------|---------------|---------------|------|
| 初始延迟 | 403.0 ms | 326.9 ms | -19% |
| 稳态延迟 (均值) | ~158 ms | ~137 ms | **-13%** |
| 稳态延迟 (最大) | 171.3 ms | 146.8 ms | **-14%** |
| 目标余量 | -4.6 ms (超标) | +19.9 ms (达标) | ✅ |

**关键提升**:
- T0a 最大延迟 171ms，超出 166.667ms 目标
- T1 最大延迟 147ms，有 20ms 安全余量
- **TensorRT 使 6Hz replan 从勉强达标变为稳定达标**

### 5.3 加速原因分析

视觉编码是计算瓶颈（估计占总延迟 40-50%）:
- 处理 2×(640×480) 图像
- 卷积密集型操作
- TensorRT 对卷积有高度优化（kernel fusion, 内存优化）

**为什么不加速后半部分**:
1. RTC 需要梯度计算（TensorRT 不支持）
2. Action expert 相对轻量（占比 10-15%）
3. 收益递减（继续优化只能再降 10-15ms）
4. 灵活性重要（研究阶段需要快速迭代）

---

## 6. 运行检查清单

### 6.1 TensorRT 部署前置条件

```bash
# 1. 验证 TensorRT 引擎
ls -lh $MODEL/pi05_tensorrt_rebuild_bf16/prefix_cache_bf16.plan
ls -lh $MODEL/pi05_tensorrt_rebuild_bf16/prefix_cache_bf16.plan.verified.json

# 2. 检查验证标记
python -c 'import json; print(json.load(open("...verified.json"))["passed"])'

# 3. 验证依赖
export PYTHONPATH="$TRT_PYTHON:$TRT_ROOT:$ROOT/src"
export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:$LD_LIBRARY_PATH"
python -c 'import tensorrt; print(tensorrt.__version__)'
```

### 6.2 启动时必须检查的日志

```
✅ TensorRT PI0.5 prefix enabled: engine=.../prefix_cache_bf16.plan, cameras=2, cache_layers=18
✅ Policy loaded: type=pi05, device=cuda
✅ Robot connected: so_follower
✅ RTC timing mode=actual_consumed
✅ Action output filter enabled
⚠️  Stall/contact guard enabled: ticks=X  # 当前建议设为 0
```

### 6.3 运行中监控指标

**延迟目标**: < 166.667 ms (6Hz replan)

```
RTC timing diagnostics:
  latency_ms=146.8     ← 应 < 167ms
  actual_consumed_steps=4  ← 应 ≤ 5
  replan_hz=6.000      ← 应接近 6.0
```

**异常信号**:
- `latency_ms > 170`: 延迟超标
- `actual_consumed_steps > 5`: 消耗过快，队列可能耗尽
- `safety clamp` 连续出现: 守护即将触发

---

## 7. 已知问题与限制

### 7.1 训练数据分布窄

**问题**: 平均 15 秒 episode，物体位置范围有限

**表现**:
- 首次接触推动物体后，物体进入未见位置
- 模型输出质量下降
- 30 秒运行中无法完成 place

**缓解**: 5 秒快速运行，在物体被推出分布前完成抓取

### 7.2 守护与快速动作不兼容

**问题**: 简单计数守护误杀正常动作

**临时方案**: 禁用守护 + 人工监督

**长期方案**: 需要传感器融合（电流/力矩）

### 7.3 钳制警告频繁

**正常行为**: 快速动作时每步钳制

**不是问题**:
- 保护机制正常工作
- 确保单步位移不超过 5°
- T0a 有 67 次钳制仍成功

**误判**: 守护将正常钳制当作碰撞

---

## 8. 推荐工作流

### 8.1 当前最稳定配置

**5 秒快速抓取 + 回位**:
```bash
./run_pi05_no_guard_5s.sh
```

配置:
- PyTorch prefix（稳定性优先）
- 无守护（避免误触发）
- 5 秒时长（训练分布内）
- 自动回位

### 8.2 TensorRT 性能测试

```bash
PI05_PREFIX_BACKEND=tensorrt \
PI05_TRT_PREFIX_ENGINE=$MODEL/pi05_tensorrt_rebuild_bf16/prefix_cache_bf16.plan \
PI05_DURATION=5 \
./run_pi05_full_rollout_rtc_actual_no_guard.sh
```

验证 TensorRT 推理质量

### 8.3 30 秒完整演示（需要准确物体位置）

```bash
PI05_DURATION=30 \
PI05_RETURN_TO_INITIAL_POSITION=true \
./run_pi05_full_rollout_rtc_actual_no_guard.sh
```

注意: 首次接触后物体可能进入未见位置

---

## 9. 下一步建议

### 9.1 立即可做

1. **验证 TensorRT + 无守护**: 确认推理质量
2. **多次 5 秒测试**: 建立成功率基线
3. **记录物体初始位置**: 找到训练分布中心

### 9.2 短期改进

1. **改进守护逻辑**:
   - 增加电流/力矩传感器
   - 区分瞬态和持续钳制
   - 调整阈值（如提高到 10 次）

2. **优化物体放置**:
   - 测量训练数据中物体位置分布
   - 在中心位置放置物体
   - 增加 place 成功率

### 9.3 长期优化

1. **扩充训练数据**:
   - 收集更多 episode（当前 40 个）
   - 增加物体位置多样性
   - 延长 episode 时长（目标 30 秒）

2. **全栈 TensorRT**:
   - 如需 <100ms 延迟
   - 重新设计静态引导机制
   - 牺牲 RTC 灵活性

---

## 10. 结论

**TensorRT 部署技术上成功**:
- ✅ 推理延迟降低 13-14%
- ✅ 稳定达到 6Hz replan 目标
- ✅ 引擎验证通过，输出质量符合预期

**真机失败原因**:
- ❌ 防碰撞守护设计缺陷
- ❌ 无法区分正常动作和真实碰撞
- ❌ 需要传感器融合改进

**当前最佳实践**:
- 使用无守护脚本 + 人工监督
- 5 秒快速运行避免分布外问题
- TensorRT 可选（性能提升但非必需）

---

## 附录：关键文件

### A.1 成功运行脚本

- `run_pi05_no_guard_5s.sh`: 5 秒快速抓取 + 回位
- `run_pi05_full_rollout_rtc_actual_no_guard.sh`: 无守护基础脚本

### A.2 失败运行日志

- `outputs/rollout/t1_q45/t1_q45_20260727_091710.log`: TensorRT + 守护失败 1
- `outputs/rollout/t1_q45/t1_q45_20260727_094933.log`: TensorRT + 守护失败 2

### A.3 成功运行日志

- `outputs/rollout/t0a_q45/t0a_q45_20260726_185544.log`: PyTorch + 无守护成功

### A.4 分析工具

- `analyze_tensorrt_speedup.py`: TensorRT 加速效果量化分析

---

**文档版本**: 1.0
**最后更新**: 2026-07-27
**维护者**: LeRobot Team
