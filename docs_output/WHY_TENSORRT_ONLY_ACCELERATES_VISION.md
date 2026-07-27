# 为什么 TensorRT 只加速视觉部分而不加速 Action 部分

## 技术架构分析

基于代码分析 (`src/lerobot/policies/pi05/modeling_pi05.py`)，PI0.5 模型的完整推理流程如下：

### 完整推理流程

```python
def sample_actions(self, images, img_masks, tokens, masks, ...):
    """完整的推理流程"""

    # ========== 第一阶段：Prefix 计算 (TensorRT 加速) ==========
    # 1. 视觉编码 (SigLIP)
    img_emb = self.paligemma_with_expert.embed_image(img)

    # 2. 语言编码
    lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)

    # 3. PaliGemma prefix prefill (生成 18 层 K/V cache)
    _, past_key_values = self.paligemma_with_expert.forward(
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    # 输出: past_key_values (18 层，每层包含 key 和 value)

    # ========== 第二阶段：去噪循环 (PyTorch) ==========
    x_t = noise  # 初始噪声
    for step in range(num_steps):  # 默认 10 步
        time = 1.0 + step * dt

        # 4. Action embedding
        action_emb = self.action_in_proj(x_t)  # [B, 50, 2304]

        # 5. Time embedding
        time_emb = create_sinusoidal_pos_embedding(time, ...)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)

        # 6. Action expert forward (使用缓存的 K/V)
        suffix_out, _ = self.paligemma_with_expert.forward(
            past_key_values=past_key_values,  # 复用第一阶段的 cache
            inputs_embeds=[None, suffix_embs],
            adarms_cond=[None, adarms_cond],
        )

        # 7. Action projection
        v_t = self.action_out_proj(suffix_out)  # [B, 50, 7]

        # 8. 更新噪声
        x_t = x_t + dt * v_t

    return x_t  # 最终的动作序列
```

---

## 为什么不加速 Action 部分的 5 大原因

### 1. **RTC 动态引导需要梯度计算**

**核心约束**: TensorRT 不支持自动微分

```python
# RTC 的核心机制 (在 denoise_step 内部)
if self._rtc_enabled():
    v_t = self.rtc_processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inference_delay,
        time=time,
        original_denoise_step_partial=denoise_step_partial_call,
        execution_horizon=execution_horizon,
    )
```

**RTC 引导的工作原理**:
- 每次 replan 时，计算当前观测相对于历史动作的梯度
- 用梯度信息调整新生成的动作，使其与已执行的动作平滑衔接
- 这需要 PyTorch 的 `torch.autograd` 机制

**如果转为 TensorRT**:
- ❌ 无法计算 VJP (Vector-Jacobian Product)
- ❌ RTC 的引导机制完全失效
- ❌ 动作会出现明显的不连续跳变

**替代方案的代价**:
- 需要重新设计静态引导方案（可能降低质量）
- 或者完全放弃 RTC（退回固定 chunk 执行）

---

### 2. **去噪循环的动态性**

**10 步去噪循环的特点**:

```python
x_t = noise
for step in range(num_steps):  # 循环 10 次
    time = 1.0 + step * dt

    # 每步的输入都依赖上一步的输出
    v_t = denoise_step(x_t, time, past_key_values)
    x_t = x_t + dt * v_t  # 动态更新
```

**动态依赖链**:
- 步骤 N 的输入 `x_t` 来自步骤 N-1 的输出
- 无法提前知道中间状态
- TensorRT 擅长的是固定计算图，不适合动态循环

**TensorRT 转换的挑战**:
- 需要 unroll 整个循环（10 步 × action expert forward）
- 模型大小膨胀 10 倍
- 失去调整 `num_inference_steps` 的灵活性（5/10/15 步）

---

### 3. **计算瓶颈分布不均**

从延迟数据反推各部分耗时：

| 模块 | 估计耗时 | 计算特征 | TensorRT 加速潜力 |
|------|----------|----------|-------------------|
| **视觉编码** | ~50-70ms | 2×(640×480) 图像卷积 | ⭐⭐⭐⭐⭐ 极高 |
| **PaliGemma prefix** | ~40-50ms | 大型 VLM，生成 18 层 cache | ⭐⭐⭐⭐ 高 |
| **10 步去噪循环** | ~30-40ms | 循环 10 次，但每步轻量 | ⭐⭐ 低 |
| **Action expert** (单步) | ~3-4ms | 相对小的网络 | ⭐ 极低 |
| **Action in/out proj** | ~1-2ms | 线性层 | ⭐ 极低 |

**实际测量验证**:
- PyTorch 全流程: ~158ms
- TensorRT prefix: ~137ms
- **加速了 21ms (13%)**

**如果继续优化后半部分**:
- 假设 10 步去噪 + action expert 共 35ms
- 即使全部加速 50%，只能再降 17-18ms
- **总延迟从 137ms → 120ms (额外 12% 提升)**

**边际收益递减**:
- 第一次优化（视觉）：20% 工作量 → 13% 提升
- 第二次优化（action）：80% 工作量 → 12% 提升
- **性价比急剧下降**

---

### 4. **当前性能已经满足需求**

**实际数据** (T1 运行 2):
```
稳态延迟: p50=138.8ms, max=146.8ms
目标上限: 166.667ms (6Hz replan)
余量: 19.9ms
```

**性能已达标**:
- ✅ 最大延迟有 12% 安全余量
- ✅ 满足 6Hz replan 的生产要求
- ✅ 即使偶尔波动到 160ms 仍在目标内

**继续优化的必要性**:
- 如果目标是 6Hz → 不需要
- 如果目标是 10Hz (100ms) → 需要全栈优化
- 如果目标是 15Hz (67ms) → 需要模型蒸馏或量化

**工程决策**:
- 当前瓶颈不在推理速度，而在守护逻辑和训练数据
- 投入资源优化 action 部分的 ROI 很低

---

### 5. **灵活性的巨大价值**

保留 PyTorch 后半部分的战略意义：

#### A. 研究迭代灵活性

```python
# 可以快速实验的参数
num_inference_steps = 5 / 10 / 15  # 去噪步数
execution_horizon = 10 / 15 / 20   # RTC 窗口
max_guidance_weight = 5.0 / 10.0   # 引导强度
```

**PyTorch**: 修改配置，立即运行
**TensorRT**: 重新导出 → 编译（10-30 分钟）→ 验证 → 部署

#### B. 模型架构演进

当前可能的改进方向：
- 调整 action expert 的层数（当前基于 Gemma 2B）
- 实验不同的时间编码方式
- 尝试条件引导的变体
- 增加状态信息输入

**PyTorch**: 改代码，训练，测试
**TensorRT**: 每次修改都要重新走完整个 TensorRT 转换流程

#### C. 调试和诊断

```python
# 研究阶段常见的调试需求
if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
    self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
```

**PyTorch**: 可以随时插入 print、可视化、梯度检查
**TensorRT**: 黑盒推理，难以诊断问题

#### D. 多后端兼容性

当前代码支持：
- CUDA (生产)
- CPU (开发测试)
- MPS (Mac 开发)
- 未来可能的 AMD ROCm

**PyTorch**: 跨平台无缝
**TensorRT**: 需要为每个平台单独编译引擎

---

## 什么情况下需要加速 Action 部分

### 场景 1: 延迟目标 < 100ms

如果需要 10Hz 或更高的 replan 频率：
- 当前 137ms → 需要降到 100ms 以下
- 必须优化后半部分

**技术方案**:
1. Action expert 转 TensorRT (需要放弃 RTC 或重新设计)
2. 降低去噪步数 (10 → 5 步，可能影响质量)
3. 模型量化 (BF16 → INT8)

### 场景 2: 研究阶段结束，进入生产部署

当模型架构完全冻结，不再需要快速迭代：
- 追求极致性能
- 降低推理成本
- 部署到边缘设备

**技术方案**:
1. 全栈 TensorRT 或 ONNX Runtime
2. 去噪循环 unroll 并融合
3. 静态引导替代 RTC 动态引导

### 场景 3: 硬件资源受限

在算力有限的边缘设备（Jetson Orin）上部署：
- GPU 内存紧张
- 功耗预算有限
- 需要最大化吞吐量

**技术方案**:
1. 模型蒸馏（小模型模仿大模型）
2. INT8 量化 + TensorRT
3. 剪枝不重要的权重

---

## 当前架构的优势总结

### ✅ 当前混合架构 (TensorRT Prefix + PyTorch Action)

**优点**:
1. **性能达标**: 6Hz replan，19.9ms 余量
2. **快速迭代**: 修改 action expert 无需重新导出
3. **RTC 完整功能**: 动态引导、梯度计算
4. **灵活调参**: 去噪步数、RTC 窗口可调
5. **易于调试**: PyTorch 全部诊断工具可用
6. **低维护成本**: 一次 TensorRT 转换，长期使用

**代价**:
- 只优化了 13%，不是理论最大值

### ❌ 全栈 TensorRT 架构

**优点**:
1. **性能更高**: 可能再降 10-15ms
2. **部署简单**: 单一推理引擎

**代价**:
1. ❌ **失去 RTC**: 动作不连续，质量下降
2. ❌ **失去灵活性**: 每次改模型重新编译
3. ❌ **维护困难**: TensorRT 调试困难
4. ❌ **开发效率低**: 迭代周期从分钟级变成小时级

---

## 结论

**TensorRT 只加速视觉部分是深思熟虑的工程决策，不是技术限制**：

1. **技术约束**: RTC 需要梯度，TensorRT 不支持
2. **性能瓶颈**: 视觉编码占 40-50%，action 只占 10-15%
3. **收益递减**: 继续优化只能再降 12%，但失去所有灵活性
4. **当前够用**: 已达到 6Hz 目标，有 12% 安全余量
5. **研究阶段**: 需要频繁迭代，PyTorch 的灵活性价值巨大

**80/20 法则的完美体现**: 用 20% 的工作量（只转 prefix）解决了 80% 的问题（达到性能目标），保留了 100% 的灵活性。

---

## 附录：代码证据

### Prefix 可以替换为 TensorRT

```python
def _compute_prefix_cache(self, images, img_masks, tokens, masks):
    if self._prefix_cache_backend is not None:
        # 使用外部后端（TensorRT）
        return self._prefix_cache_backend(images, img_masks, tokens, masks)

    # 回退到 PyTorch
    if self.paligemma_with_expert.paligemma is None:
        raise RuntimeError("PyTorch prefix modules were released")

    # PyTorch prefix 计算
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    _, past_key_values = self.paligemma_with_expert.forward(...)
    return prefix_pad_masks, past_key_values
```

### Action 部分依赖动态循环

```python
@torch.no_grad()
def sample_actions(self, images, img_masks, tokens, masks, ...):
    # Prefix 计算（可替换）
    prefix_pad_masks, past_key_values = self._compute_prefix_cache(...)

    # 去噪循环（动态，依赖上一步）
    x_t = noise
    for step in range(num_steps):  # 10 次循环
        time = 1.0 + step * dt

        # RTC 需要梯度计算
        if self._rtc_enabled():
            v_t = self.rtc_processor.denoise_step(
                x_t=x_t,
                original_denoise_step_partial=denoise_step_partial_call,
                ...
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        # 动态更新
        x_t = x_t + dt * v_t

    return x_t
```

### Denoise step 内部

```python
def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
    # 1. Action embedding
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

    # 2. Action expert forward（复用 prefix cache）
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        past_key_values=past_key_values,  # 来自 prefix
        inputs_embeds=[None, suffix_embs],
        adarms_cond=[None, adarms_cond],
    )

    # 3. Action projection
    suffix_out = outputs_embeds[1][:, -self.config.chunk_size:]
    return self.action_out_proj(suffix_out)
```

**关键观察**:
- `past_key_values` 在 10 步循环中**不变**（只计算一次）
- `x_t` 在每步都**变化**（依赖上一步）
- RTC 的引导在循环内部**需要梯度**

这就是为什么可以把 prefix 转 TensorRT（固定计算），但 action 部分必须保留 PyTorch（动态计算 + 梯度）。
