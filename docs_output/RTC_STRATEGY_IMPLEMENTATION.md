# RTC (Real-Time Chunking) 策略实现原理

## 概述

RTC (Real-Time Chunking) 是 Physical Intelligence 开发的一种实时动作分块技术，用于解决 VLA (Vision-Language-Action) 模型推理慢但需要高频控制的问题。

**核心论文**: [Real-Time Chunking](https://www.physicalintelligence.company/download/real_time_chunking.pdf)

---

## 问题背景

### 传统 Action Chunking 的困境

**典型场景**:
- 模型生成 50 步动作 chunk（chunk_size=50）
- 30Hz 控制频率
- 每步耗时 33.3ms
- **但模型推理需要 140ms！**

**问题**:
1. 控制频率 30Hz，每 33ms 需要一个新动作
2. 推理 140ms 期间，需要消耗 4-5 个动作
3. 如果等推理完成再 replan，动作会不连续

**传统方案的局限**:
```
Time:  0ms    50ms   100ms  150ms  200ms
       |------|------|------|------|
Queue: 50 → 45 → 40 → 35 → 30 (等推理)
                          ↑
                      推理完成，queue 还有 30 个
                      如何平滑过渡？
```

---

## RTC 解决方案

### 核心思想

**动态引导新推理结果，使其与已执行的动作平滑衔接**

```
旧 chunk: [已执行] [队列中剩余] ← 这部分是"prefix"
新 chunk: [............生成中............]
            ↓ RTC 引导
新 chunk: [与 prefix 平滑衔接] [新的规划]
```

### 关键机制

#### 1. **Prefix Leftover（前缀剩余）**

```python
# 推理开始时，记录队列剩余
prev_chunk_left_over = queue.get_left_over()  # [B, T_prev, A]
# 例如：queue 还有 30 个动作未执行
```

**物理意义**: 这 30 个动作是机器人"承诺"要执行的轨迹

#### 2. **Guidance via Autograd（梯度引导）**

核心代码 (`modeling_rtc.py:212-219`):

```python
with torch.enable_grad():
    v_t = original_denoise_step_partial(x_t)  # 原始去噪输出
    x_t.requires_grad_(True)

    # 计算如果执行 v_t，预测的最终状态
    x1_t = x_t - time * v_t

    # 计算与 prefix 的误差（带权重）
    err = (prev_chunk_left_over - x1_t) * weights

    # 梯度反传：如何修改 x_t 来减小误差
    correction = torch.autograd.grad(x1_t, x_t, err)[0]

# 应用引导
result = v_t - guidance_weight * correction
```

**数学原理**:
1. `x1_t = x_t - time * v_t`: 欧拉步预测的最终动作
2. `err = (prefix - x1_t) * weights`: 与历史承诺的偏差
3. `∂err/∂x_t`: 如何调整当前状态来减小偏差
4. `v_t - guidance_weight * correction`: 修正后的速度

#### 3. **Prefix Attention Weights（前缀注意力权重）**

控制哪些历史动作被强制对齐，哪些允许偏离。

**EXP 调度**（当前配置，`modeling_rtc.py:264-268`）:

```python
# inference_delay=5, execution_horizon=10, chunk_size=50
weights = get_prefix_weights(start=5, end=10, total=50)

# 结果:
# [1, 1, 1, 1, 1,  # 前 5 步：强制对齐（已在执行）
#  0.9, 0.7, 0.5, 0.2, 0.1,  # 5-10 步：指数衰减（窗口内）
#  0, 0, ..., 0]  # 10 步后：完全自由
```

**权重物理意义**:
- `weight=1.0`: 强制对齐（已在机器人执行路径上）
- `0 < weight < 1`: 软引导（可以偏离但有惩罚）
- `weight=0`: 完全自由（允许重新规划）

#### 4. **Guidance Weight（引导强度）**

随去噪时间动态调整（`modeling_rtc.py:221-227`）:

```python
tau = 1 - time  # time 从 1 → 0，tau 从 0 → 1
c = (1 - tau) / tau  # 早期小，后期大
inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
guidance_weight = min(c * inv_r2, max_guidance_weight)
```

**时间演化**:
- `time=1.0` (去噪开始): `guidance_weight ≈ 0`，几乎不引导
- `time=0.5` (中期): `guidance_weight ≈ 2-3`，中等引导
- `time=0.1` (接近完成): `guidance_weight → 10.0`，强引导

**原因**: 去噪早期需要探索空间，晚期需要精确对齐

---

## 完整推理流程

### 在推理循环中的调用

**去噪步骤** (`modeling_pi05.py:854-882`):

```python
x_t = noise  # 初始噪声
for step in range(num_steps):  # 10 步去噪
    time = 1.0 + step * dt

    if self._rtc_enabled():
        # RTC 包装的去噪
        v_t = self.rtc_processor.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev_actions,  # 历史剩余
            inference_delay=delay,  # 延迟估计
            time=time,
            original_denoise_step_partial=denoise_step_partial,
            execution_horizon=execution_horizon,
        )
    else:
        # 原始去噪（无引导）
        v_t = denoise_step_partial(x_t)

    x_t = x_t + dt * v_t  # 更新状态
```

### 后台推理线程

**RTC 引擎** (`rtc.py:654-849`):

```python
while not shutdown:
    # 1. 检查队列是否需要 replan
    if queue.qsize() <= queue_threshold:  # 例如 ≤ 45

        # 2. 获取当前观测
        obs = get_current_observation()

        # 3. 估计延迟
        estimated_delay = guidance_estimator.estimate()  # 例如 5 步

        # 4. 获取队列剩余（prefix）
        if timing_mode == "actual_consumed":
            snapshot = queue.snapshot()
            prev_actions = snapshot.original_leftover  # 未执行的原始动作

        # 5. 调用策略（包含 RTC 引导）
        actions = policy.predict_action_chunk(
            obs,
            inference_delay=estimated_delay,
            prev_chunk_left_over=prev_actions,  # 传入 prefix
        )

        # 6. 合并新 chunk 到队列
        queue.merge_actual_consumed(
            original=actions,
            processed=postprocess(actions),
            snapshot=snapshot,
        )
```

### Actual Consumed 时序模式

**关键创新**（`rtc.py:682-696`）:

```python
# 推理开始时刻，记录队列状态
inference_start = queue.snapshot()
idx_before = inference_start.next_action_index  # 例如 20

# 推理耗时 140ms，期间执行了 4 个动作
# 推理完成时，队列已到 index=24

# 计算实际消耗
actual_consumed = current_index - idx_before  # 4 步

# 从新 chunk 跳过前 4 个，直接从第 5 个开始入队
merge_actual_consumed(..., max_actual_consumed_steps=10)
```

**优势**: 精确同步，不会出现 replan 延迟累积

---

## 参数配置

### 当前 PI0.5 配置

```python
# chunk_size=50, fps=30, queue_threshold=45
execution_horizon = 10          # 引导窗口：前 10 步
max_guidance_weight = 10.0      # 最大引导强度
prefix_attention_schedule = EXP  # 指数衰减权重
guidance_delay_mode = "fixed"   # 固定延迟估计
fixed_guidance_delay_steps = 5  # 固定 5 步延迟

# 队列管理
queue_threshold = 45            # 剩余 ≤45 时 replan
# 名义 replan 间隔 = (50 - 45) / 30 = 5 步 / 0.167s = 6Hz
```

### 参数调优指南

#### `execution_horizon`
- **物理意义**: 未来多少步内强制对齐
- **太小** (e.g., 5): 引导不足，动作可能跳变
- **太大** (e.g., 20): 过度约束，无法重新规划
- **推荐**: 10-15，覆盖 1-2 个 replan 间隔

#### `max_guidance_weight`
- **物理意义**: 最大引导力度
- **太小** (e.g., 3.0): 对齐不足
- **太大** (e.g., 50.0): 可能震荡
- **推荐**: 5.0-10.0

#### `queue_threshold`
- **物理意义**: 提前多久开始新推理
- **Q45**: replan 间隔 5 步 (6Hz)
- **Q40**: replan 间隔 10 步 (3Hz)
- **Q20**: replan 间隔 30 步 (1Hz)
- **推荐**: 根据推理延迟动态调整

#### `guidance_delay_mode`
- **fixed**: 固定延迟步数（当前使用）
- **estimated**: 动态估计（基于历史延迟）
- **推荐**: 固定模式更稳定

---

## 技术细节

### 为什么需要梯度？

```python
# 目标：让新 chunk 的前几步接近 prefix
# 方法：最小化 loss = ||new_chunk[:10] - prefix[:10]||^2

# 传统方法：直接覆盖
new_chunk[:10] = prefix[:10]  # 简单粗暴

# RTC 方法：软约束 + 梯度引导
correction = ∂loss/∂x_t  # 如何调整输入来减小 loss
v_t = v_t - guidance_weight * correction  # 沿梯度方向修正
```

**优势**:
- 保持去噪过程的连续性
- 允许在对齐 prefix 和重新规划之间平衡
- 梯度信息比硬替换更平滑

### 为什么 TensorRT 无法加速这部分？

**RTC 核心依赖** (`modeling_rtc.py:212-219`):

```python
with torch.enable_grad():
    x_t.requires_grad_(True)
    x1_t = x_t - time * v_t
    correction = torch.autograd.grad(x1_t, x_t, err)[0]
```

**TensorRT 的限制**:
- ❌ 不支持 `torch.autograd`
- ❌ 不支持动态计算图
- ❌ 只能做前向推理

**如果强行转 TensorRT**:
- 失去梯度计算能力
- RTC 引导机制完全失效
- 动作会不连续跳变

---

## 实际效果

### 对比实验（6Hz replan）

| 配置 | 动作连续性 | 任务成功率 | 推理频率 |
|------|-----------|-----------|----------|
| **无 RTC** | 跳变明显 | 低 | 固定间隔 |
| **RTC (Q45)** | 平滑 | 高 | 自适应 6Hz |
| **RTC (Q20)** | 极平滑 | 最高 | 自适应 1Hz |

### 当前系统表现

```
配置: Q45 + actual_consumed + fixed_delay=5
结果:
  - nominal_replan_hz=6.000
  - 实际 replan 间隔: 4-5 步
  - 延迟峰值: 191ms (超标但可控)
  - 动作连续性: 优秀
```

---

## 关键代码位置

### 核心实现
- `src/lerobot/policies/rtc/modeling_rtc.py:117-249` - RTC 引导逻辑
- `src/lerobot/rollout/inference/rtc.py:654-849` - 后台推理线程
- `src/lerobot/policies/rtc/action_queue.py` - 动作队列管理

### 调用路径
```
rollout loop (strategies/core.py)
  ↓ notify_observation
RTCInferenceEngine (inference/rtc.py)
  ↓ _policy_loop (后台线程)
  ↓ predict_action_chunk
PI05Policy (modeling_pi05.py)
  ↓ sample_actions
  ↓ 去噪循环 (10 步)
    ↓ rtc_processor.denoise_step
RTCProcessor (modeling_rtc.py)
  ↓ torch.autograd.grad (计算引导)
  ↓ 返回引导后的 v_t
```

---

## 总结

**RTC 的本质**:
- 用梯度引导让慢速推理生成的动作平滑衔接快速控制循环
- 是一种"软约束"的在线优化方法
- 依赖自动微分，因此必须保留 PyTorch

**关键创新**:
1. **Prefix guidance via autograd** - 梯度引导而非硬替换
2. **Actual consumed timing** - 精确同步队列消耗
3. **Adaptive prefix weights** - 指数衰减的注意力权重
4. **Time-varying guidance strength** - 动态调整引导强度

**工程价值**:
- 使 VLA 模型可以用于高频实时控制（30Hz）
- 推理延迟从瓶颈变为可管理的因素
- 为 TensorRT 等加速技术提供了集成空间

**这就是为什么 TensorRT 只能加速 prefix，而 action 部分必须保留 PyTorch 的根本原因。**
