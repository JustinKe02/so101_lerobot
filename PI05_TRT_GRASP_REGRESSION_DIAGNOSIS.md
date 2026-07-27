# PI0.5 TensorRT 抓取退化诊断

更新时间：2026-07-24

兼容优先的后续实施方案见：`PI05_TRT_RTC_COMPATIBILITY_PLAN.md`。该计划保留
`legacy + PyTorch` 为默认回滚路径，新 RTC 时序与 action TensorRT 均通过独立开关验证。

## 结论

当前 TensorRT prefix 的“固定 delay 数值 parity”基本正常，但它不是当前 RTC 闭环的等价替换。最可能的退化来源是推理耗时跨过了 RTC 的离散延迟桶：

```text
PyTorch 约 140 ms -> ceil(140/33.33) = 5 步
TensorRT 约 125 ms -> ceil(125/33.33) = 4 步
```

RTC 随后分别执行新 chunk 的 `actions[5:]` 和 `actions[4:]`，并使用不同的 guidance delay。抓取阶段（数据集帧 170 -> 200）固定 delay 时机器人单位最大误差只有 `0.20`，按各自实测 delay 时首个执行动作最大误差为 `4.24`，完整 chunk 最大误差为 `1.55`。

此外，必须确认 policy 和 engine 来自同一 checkpoint。把旧全量 policy 与 S1 engine 混用时，离线 action 最大误差达到 `100.01` 个机器人单位，肯定不能上真机。

## 已执行的无硬件验证

所有报告都标记 `hardware_action_sent: false`：

```text
examples/inference/verify_pi05_tensorrt_dataset.py
examples/inference/verify_pi05_tensorrt_rtc.py
```

S1 普通 action parity（5 帧）：

```text
action mean abs       0.003396
action max abs        0.021890
robot-unit max        1.32294
prefix speedup        1.50x
full-action speedup   1.18x
```

报告文件：

```text
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model/pi05_tensorrt/prefix_cache_bf16.plan.dataset_parity.json
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model/pi05_tensorrt/prefix_cache_bf16.plan.rtc_parity.json
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model/pi05_tensorrt/prefix_cache_bf16.plan.rtc_parity_grasp.json
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model/pi05_tensorrt/old_full_model_with_s1_engine.dataset_parity.json
```

核心测试结果：

```text
RTC 核心单元测试：113 passed
PI0.5 RTC 单元测试：5 passed（HF 离线缓存环境）
```

## 代码层面的风险

1. `src/lerobot/rollout/inference/rtc.py` 用 `LatencyTracker.max()` 计算 guidance delay，却用当前调用的 `new_latency` 计算 queue merge delay。历史最大值只增不减，一次冷启动尖峰可能污染整次 rollout。
2. `src/lerobot/policies/rtc/action_queue.py` 已计算实际消费步数 `indexes_diff`，但不一致时只记录 warning，仍返回耗时估算的 `real_delay`。
3. `src/lerobot/policies/rtc/modeling_rtc.py` 在 `v_t=f(x_t)` 之后才设置 `x_t.requires_grad_(True)`，当前 correction 不包含模型 Jacobian；`max_guidance_weight=10` 会放大 prefix 微扰。
4. 现有普通 parity 不启用 RTC，也不验证不同 delay、leftover chunk、真实机器人单位或实际首个执行 action；因此不能作为 RTC 真机安全门槛。
5. 官方 engine verification 曾因 KV max error 超过 `2.0` 失败。S1 脚本使用自定义 dataset parity 报告，未覆盖该 KV max 门槛。

## 训练模型的额外变量

S1 只训练 action expert。最终训练日志约为 `0.043`；旧全量训练 checkpoint 最终日志约为 `0.009`。训练 loss 不能直接等价为真机成功率，但如果现场比较的是“旧全量 PyTorch”与“S1 TensorRT”，模型训练策略和推理后端同时变化，不能把退化单独归因于 TensorRT。

## 当前建议

在 RTC delay/queue 修正和 RTC parity 门禁完成前，不把当前 TensorRT engine 当作等价真机部署路径。现场应先确认同一个 S1 checkpoint 的 PyTorch baseline：

```bash
cd /data/cqy_workspace/tk/lerobot_src
PI05_USE_TRT=0 PI05_PREFLIGHT_ONLY=1 ./run_pi05_s1_rollout.sh
```

只有在明确记录 `Loading policy from` 与 `TensorRT PI0.5 prefix enabled: engine=` 后，才进行短时人工值守 A/B。TensorRT 预检命令为：

```bash
cd /data/cqy_workspace/tk/lerobot_src
PI05_USE_TRT=1 PI05_PREFLIGHT_ONLY=1 ./run_pi05_s1_rollout.sh
```

这两个命令只做资产、权限、相机和 parity 检查；不会发送机械臂动作。当前诊断没有自动启动真机。
