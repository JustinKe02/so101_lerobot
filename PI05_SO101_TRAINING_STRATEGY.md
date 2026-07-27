# PI0.5 SO-101 训练与部署方案

状态：S1/S2 训练均已完成，但多 seed 离线稳定性门禁失败，暂停真机。
旧全量模型恢复为当前唯一部署基线。

更新时间：2026-07-24

说明：本文的训练策略和 S1 结果继续有效；TensorRT/RTC 的后续部署、兼容性、
回滚和 action TensorRT 实施顺序，以 `PI05_TRT_RTC_COMPATIBILITY_PLAN.md` 为准。
在该计划的 T1 门禁通过前，不再沿用本文早期的 TensorRT 真机试运行顺序。

## 1. 当前基线

- 数据集：`data/so101_test_data`
- 数据规模：40 episodes，17,960 frames，30 Hz
- 输入：top/wrist 两路 640x480 图像和 6 维关节状态
- 输出：6 维绝对关节位置，action chunk size 为 50
- 任务：`Put the block in the bin`
- 基础模型：`/data/cqy_workspace/tk/model_assets/lerobot/pi05_base`
- 当前全量训练模型：
  `outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model`
- 当前全量训练配置：batch size 32，5,613 steps（10 epochs），学习率 2.5e-5
- 当前训练更新了约 41.43 亿参数，没有冻结视觉或 PaliGemma，也没有使用 PEFT。

当前全量模型保留为基线，不覆盖、不续训。新的训练候选仍使用全部 40 episodes，最终使用未参与训练的新真机摆放条件测试。

## 2. 目标

1. 减少小数据全量微调造成的过拟合和预训练能力退化风险。
2. 降低真机运行中 `max_relative_target` 的触发比例。
3. 保持动作专家对 SO-101 动作空间的适应能力。
4. 与现有全量模型进行单变量、可复现的对照。
5. 将训练策略收益与输出滤波收益分开评估。

## 3. 训练方案对比

| 方案 | 约可训练参数 | 核心设置 | 优点 | 主要风险 | TensorRT prefix |
| --- | ---: | --- | --- | --- | --- |
| S1 动作专家训练 | 6.93 亿标记可训练，约 4.30 亿有效更新 | 冻结整个 PaliGemma | 数据规模匹配较好，变量清晰 | 可能欠缺视觉域适配 | PaliGemma 固定，可为该系列复用同一个新 engine |
| S2 LoRA | 128.72 万（r=16） | 动作专家 Q/V 与动作输入/输出投影 LoRA | 显存、存储和过拟合风险最低 | 容量可能不足，默认不训练 time MLP | PaliGemma 固定，可复用新 engine |
| S3 分阶段训练 | 先约 4.30 亿有效更新，再全量解冻 | 专家训练后低学习率短暂全量解冻 | 适应能力和稳定性较均衡 | 流程更复杂，第二阶段仍可能破坏 VLM | 第二阶段完成后必须重建 |
| S4 低学习率全量训练 | 41.43 亿 | 全模型训练 5 epochs | 适应能力上限高 | 40 episodes 下仍容易过拟合 | 每个候选 checkpoint 均需重建 |
| 仅冻结视觉编码器 | 37.31 亿 | 视觉冻结，语言模型继续训练 | 保留视觉表征 | 仍训练约 90% 参数，收益有限 | 必须重建 |

## 4. 已完成但暂不部署的方案：S1

> 2026-07-24 补充：固定训练帧的三 seed 离线复现中，S1 的 50 步 MAE 为
> `4.937/21.917/15.529`，S2 为 `3.460/21.714/12.688`，旧全量模型为
> `0.191/0.210/0.199`。因此本节保留训练配置记录，但 S1 不再是当前 RTC/TensorRT
> 修复的真机基线。RTC 修复完成也不会自动解除该模型门禁。

从原始 `pi05_base` 开始训练，而不是从当前全量模型继续训练，从而与当前基线保持相同初始化。

```text
dataset episodes              全部 40 episodes
policy.train_expert_only      true
policy.freeze_vision_encoder  false
policy.gradient_checkpointing true
policy.dtype                  bfloat16
batch_size                    32
steps                         5613（10 epochs）
optimizer                     AdamW
peak learning rate            2.5e-5
weight decay                  0.01
warmup config / effective     1000 / 187（5613 steps 时自动缩放）
seed                          1000
save frequency                1123 steps（约每个 2 epochs）
```

`scheduler_decay_steps=30000` 大于本次总步数，LeRobot 会按 `5613/30000` 自动缩放调度器，因此实际 warmup 为 187 steps、实际 decay 为 5613 steps。该行为与当前全量训练基线完全一致，用于保持单变量对照。

S1 会冻结以下模块：

- SigLIP vision tower
- multimodal projector
- PaliGemma Gemma-2B 语言模型与 token embedding

S1 会训练以下模块：

- Gemma action expert
- action input/output projection
- time MLP

这里的“动作专家训练”不是只训练最后一个很小的 action output head。当前实现会把约 6.93 亿个参数标为可训练，但其中动作专家的 `lm_head` 约 2.63 亿参数不参与 PI0.5 动作前向、不会获得梯度；实际会更新约 4.30 亿个原始参数。除这个未使用的 `lm_head` 外，动作专家内部的 attention、MLP、归一化和 AdaRMS，以及动作与时间投影都会做 dense full fine-tuning。

计划输出目录：

```text
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000
```

预计单卡训练耗时约 4-7 小时。实际耗时以首个 100-200 step 的吞吐为准。

## 5. 备选方案

### S2：LoRA

```text
pretrained model  pi05_base
method            LoRA
rank              16
alpha             16
learning rate     1e-4
steps             5613
seed              1000
```

LeRobot 当前 PI0.5 默认 LoRA target 实际命中以下 38 个线性层：

- 18 层 Gemma action expert 的 `q_proj` 和 `v_proj`，共 36 层
- `action_in_proj`
- `action_out_proj`

rank 16 时，新增并训练的 LoRA 参数精确为 1,287,168。基础模型的原始权重全部冻结，训练的是每个目标线性层旁路上的低秩增量，而不是直接更新原始矩阵。

S2 必须显式传入 `lora_alpha=16`；若省略该参数，当前配置会落到 PEFT 默认值 8，缩放系数会从计划的 `16/16=1.0` 变成 `8/16=0.5`。

当前默认 target 正则写的是 `action_time_mlp_in/out`，但 PI0.5 模型实际模块名是 `time_mlp_in/out`，因此默认配置不会命中 time MLP。若采用 S2，需要在启动前显式修正 target；是否把 time MLP 设为 LoRA 或完整训练模块，应作为 S2 的固定配置记录。PaliGemma 在该默认范围内不更新。

### S1 与 S2 的本质区别

| 对比项 | S1 动作专家训练 | S2 LoRA r=16 |
| --- | --- | --- |
| 更新方式 | 直接更新动作专家及投影层的原始权重 | 冻结原始权重，学习低秩增量 `BA` |
| 可训练规模 | 6.93 亿标记可训练，约 4.30 亿获得梯度 | 1,287,168（按当前实际 target） |
| 动作专家覆盖 | attention、MLP、norm、AdaRMS 等完整覆盖 | 默认仅 attention Q/V |
| action projection | 原始矩阵完整更新 | 低秩更新 |
| time MLP | 原始矩阵完整更新 | 当前默认未命中 |
| 优化器/梯度显存 | 较高 | 显著更低 |
| 40 episodes 风险 | 表达能力强，但更易过拟合 | 更保守，但可能欠拟合动作域 |
| 保存与加载 | 完整 PI0.5 checkpoint | base model 加 adapter；部署可选择合并 |

### S3：分阶段

第一阶段：

```text
train_expert_only=true
steps=4490（约 8 epochs）
learning_rate=2.5e-5
```

第二阶段：

```text
从第一阶段最佳 checkpoint 继续
train_expert_only=false
steps=1123（约 2 epochs）
learning_rate=2.5e-6
```

第二阶段不应在没有检查第一阶段结果的情况下自动启动。

### S4：低学习率全量训练

```text
train_expert_only=false
freeze_vision_encoder=false
steps=2806（约 5 epochs）
learning_rate=5e-6
seed=1000
```

## 6. Action 输出滤波

输出滤波不改变模型的约 140 ms 推理时间。RTC 已经按照延迟进行 action chunk 对齐；滤波仅用于抑制突跳、限制速度和加速度。

推荐在 RTC action queue 输出之后、电机发送之前加入状态相关的二阶限幅器：

```text
v_target  = clamp((q_target - q_command) / dt, -v_max, v_max)
v_command = clamp(v_target, v_previous - a_max * dt, v_previous + a_max * dt)
q_command = q_command + v_command * dt
```

依据 40 episodes 的 P95 动作变化，初始上限为：

| 关节 | 速度上限 | 加速度上限 |
| --- | ---: | ---: |
| shoulder_pan | 66 deg/s | 317 deg/s^2 |
| shoulder_lift | 95 deg/s | 317 deg/s^2 |
| elbow_flex | 95 deg/s | 396 deg/s^2 |
| wrist_flex | 53 deg/s | 475 deg/s^2 |
| wrist_roll | 40 deg/s | 396 deg/s^2 |
| gripper | 102 unit/s | 924 unit/s^2 |

`robot.max_relative_target=5.0` 继续作为最终硬件安全门限。第一轮不对整条机械臂增加重 EMA，避免额外相位延迟和抓取时序偏移。

## 7. 评估方案

原始 40 episodes 已全部参与现有基线训练，因此不再把其中任何 episode 声称为独立测试集。

每个候选模型使用新的真机条件测试：

- 至少 20 次独立试验
- 覆盖训练分布内位置和轻微位置偏移
- 每个条件对“无滤波”和“有滤波”各运行相同次数

记录以下指标：

```text
抓取成功率
放入盒子成功率
每次任务完成时间
max_relative_target 触发次数和触发比例
P95/P99 关节速度
P95/P99 关节加速度
RTC mean/P95 inference latency
action queue 最小长度
```

模型选择优先级：安全失败为零，然后依次比较任务成功率、限幅比例、动作平滑度和完成时间。

## 8. TensorRT 约束

- 当前 TensorRT engine 来自全量微调后的 `005613`，不能用于从 `pi05_base` 训练的 S1/S2 模型。
- S1/S2 不修改 PaliGemma，因此可在候选确定后基于其 PaliGemma 构建一次新 prefix engine，并在同类候选间复用。
- 当前 TensorRT exporter 只按完整 PI0.5 checkpoint 加载，不能直接把 LoRA adapter 目录当作 checkpoint 导出。S2 部署时需要加载 base 加 adapter，或先合并 adapter；默认 LoRA 不修改 prefix，因此无需为动作侧 adapter 重建 prefix engine。
- S3 第二阶段和 S4 会修改 PaliGemma，每个最终 checkpoint 都需要重新导出、构建和验证 engine。
- TensorRT engine 上真机前仍需进行真实帧 action parity 验证。

S1 最终 checkpoint 的 BF16 engine 已构建：

```text
outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/
  pretrained_model/pi05_tensorrt/prefix_cache_bf16.plan
```

默认随机合成输入验证结果：

```text
KV worst mean abs       0.085741（通过 0.1 门限）
KV worst max abs        5.500000（未通过固定 2.0 门限）
action mean abs         0.000810（通过 0.01 门限）
action max abs          0.006532（通过 0.1 门限）
```

5 个真实数据帧（indices 0、4490、8980、13470、17959）验证结果：

```text
prefix masks            全部一致
KV worst mean abs       0.086205
KV worst max abs        6.000000
action worst mean abs   0.003396
action worst max abs    0.021890
PyTorch prefix          45.06 ms
TensorRT prefix         30.00 ms（1.50x）
PyTorch full action     108.84 ms
TensorRT full action    92.28 ms（1.18x）
```

真实帧 action parity 通过，但随机 KV 单点 max 门限未通过，因此不生成官方 `.verified.json`。首轮现场试运行默认使用 PyTorch prefix；TensorRT 必须通过 `PI05_USE_TRT=1` 显式启用，且启动脚本会检查真实帧 parity 报告。

## 9. 训练结果与状态

S1 于 2026-07-23 22:28:47 CST 启动，于 2026-07-24 04:22:25 CST 完成：

```text
job       pi05_so101_expert_only_10epochs_bs32_seed1000
steps     5613 / 5613
epochs    10.00
loss      0.333 -> 0.043（最后一个完整日志窗口）
runtime   约 5 小时 54 分钟（含模型加载和 checkpoint 保存）
run.log   outputs/train/.run_state/pi05_so101_expert_only_10epochs_bs32_seed1000/run.log
status    outputs/train/.run_state/pi05_so101_expert_only_10epochs_bs32_seed1000/status
model     outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model
```

最终 checkpoint 的模型 safetensors、优化器状态、预处理器、scheduler 状态和 `training_step=5613` 均已通过结构检查。原 `exit_code=2` 来自 Python 训练正常结束后的外层 Bash 记账错误；状态目录同时保留 `launcher_exit_code=2` 和经日志、checkpoint 确认的 `training_exit_code=0`。

训练集 loss 不能代替真机成功率。下一阶段依次执行最终 checkpoint 加载检查、TensorRT prefix 构建、合成输入和真实数据帧 action parity，然后才进入现场真机评估。

上述离线阶段现已完成。独立启动文件为：

```text
run_pi05_s1_rollout.sh
pi05_s1_rollout.json
```

PyTorch 和 TensorRT 两种模式的 `PI05_PREFLIGHT_ONLY=1` 均已通过，未连接机器人、未打开相机、未发送硬件动作。现场顺序为先运行 5-10 秒 PyTorch prefix 试验，确认姿态和限幅行为后，再执行 30 秒正式试验；TensorRT 对照放在 PyTorch 真机行为确认之后。
