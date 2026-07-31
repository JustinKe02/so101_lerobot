# PI0.5 + VLASH 训练与推理链路报告（2026-07-30）

## 1. 结论摘要

本轮 PI0.5 VLASH 专家分支微调已正常完成，共训练 10 个 epoch。训练过程稳定，未出现
NaN、Inf、OOM、梯度爆炸或数据读取错误。最终 checkpoint 可以严格加载，VLASH 异步推理
链路在数据集前、中、后段回放中均通过时延和 deadline 检查。

当前不能把这些结果等同于“已通过真机安全门”：最终权重的快速多种子动作质量检查仍未
通过，主要风险是首动作的 gripper 离群值，以及 action 13 的 shoulder/elbow 尾部误差。
此外，当前用户 `cqy` 不属于 `dialout` 组，对 follower 串口没有读写权限。

本报告末尾保留了完整的 5 秒受保护真机执行命令。该命令只有在明确接受当前模型风险并修复
串口权限后才可执行。

## 2. 数据集

- 数据集：`admin123/so101_test_data`
- 本地路径：`/data/cqy_workspace/tk/lerobot_src/data/so101_test_data`
- Episode 数量：40
- 总帧数：17,960
- 采样频率：30 Hz
- 相机：`top`、`wrist`
- 状态维度：6
- 动作维度：6，绝对关节目标
- 任务：`Put the block in the bin`
- 数据划分：40 个 episode 全部训练，验证集为 0

由于没有独立验证集，本报告中的动作误差检查属于训练分布内稳定性检查，不能证明对新物体
位置、光照、相机扰动或未见过初始状态的泛化能力。

## 3. 训练配置

- 基础 checkpoint：
  `outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model`
- 输出目录：
  `outputs/train/pi05_so101_vlash_expert_only_10epochs_bs32_seed1000`
- 最终 checkpoint：
  `outputs/train/pi05_so101_vlash_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model`
- 训练步数：5,613
- Epoch：10
- Batch size：32
- 随机种子：1000
- 学习率：`5e-6`，最终衰减到 `2.5e-6`
- 数据类型：bfloat16
- `state_cond=true`
- `temporal_offset_max_steps=8`
- `train_expert_only=true`
- `gradient_checkpointing=true`
- `compile_model=false`
- `fuse_qkv=false`
- `fuse_gate_up=false`
- `eval_split=0`
- `eval_steps=0`

最终 checkpoint 包含完整的：

- `model.safetensors`：9,362,583,312 bytes
- optimizer state：2,212,202,056 bytes
- scheduler state
- RNG state
- policy preprocessor / postprocessor
- `pi05_temporal_offset_processor_step`

整个训练目录约 54 GB。

## 4. 训练曲线分析

训练于 2026-07-30 16:20:10 正常结束，总耗时约 5 小时 52 分钟。

| Epoch 区间 | Loss 均值 | Loss 最小值 | Loss 最大值 | Loss 标准差 | 梯度均值 | 梯度最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 0-2 | 0.014991 | 0.011 | 0.028 | 0.002772 | 0.176000 | 0.344 |
| 2-4 | 0.013478 | 0.009 | 0.020 | 0.001725 | 0.160549 | 0.206 |
| 4-6 | 0.013223 | 0.010 | 0.017 | 0.001462 | 0.162214 | 0.206 |
| 6-8 | 0.012920 | 0.010 | 0.016 | 0.001364 | 0.160688 | 0.204 |
| 8-10 | 0.012752 | 0.009 | 0.017 | 0.001508 | 0.159735 | 0.196 |

最后一个 epoch 的 loss 均值为 `0.012772`，最终记录为：

- loss：`0.012`
- gradient norm：`0.181`
- learning rate：`2.5e-6`
- 单步耗时：`3.747 s`

训练 loss 从前两个 epoch 到最后两个 epoch 下降约 14.9%。epoch 6 之后改善速度明显变慢，
已经进入平台区。继续单纯增加 epoch 预计只能带来有限收益，不能保证消除动作尾部离群值。

## 5. Checkpoint 动作质量对比

快速检查使用相同的 5 个数据帧和 3 个随机种子（0、42、1000），所有预测均为有限值。

| Checkpoint | MAE 均值 | MAE P95 | Action 0 P95 | Action 13 P95 | Seed Std P95 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Epoch 4 / step 2246 | 1.512 | 5.442 | 6.459 | 5.531 | 2.078 | 未通过 |
| Epoch 6 / step 3369 | 1.409 | 5.231 | 5.049 | 5.527 | 2.074 | 未通过 |
| Epoch 10 / step 5613 | 1.397 | 5.003 | 4.817 | 4.990 | 1.937 | 未通过 |

最终权重比 epoch 4 和 epoch 6 更好，但仍有以下必需检查失败：

- MAE P95：`5.003 > 5.0`
- Action 0 P95：`4.817 > 3.0`
- Action 0 max：`12.738 > 5.0`
- Action 13 P95：`4.990 > 3.0`
- Action 13 max：`6.077 > 5.0`

主要关节风险：

- `gripper.pos` action 0：P95 `11.038`，max `12.738`
- `elbow_flex.pos` action 13：P95 `5.758`，max `6.077`
- `shoulder_lift.pos` action 13：P95 `4.893`，max `4.924`

最终质量报告：

`outputs/eval/pi05_vlash_epoch10_quick.json`

## 6. VLASH 推理链路

部署参数：

- `execution_horizon=10`
- `inference_overlap_steps=8`
- `max_future_state_delta=5.0`
- `deadline_miss_limit=1`
- 目标 FPS：30
- PyTorch prefix/action backend
- QKV 与 gate/up fusion 关闭

最终 checkpoint 在 episode 0、20、39 上各回放 100 个控制步：

| Episode | 控制步 | 异步推理次数 | Deadline miss | P50 | P95 / Max | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 100/100 | 11 | 0 | 111.5 ms | 115.3 ms | 通过 |
| 20 | 100/100 | 11 | 0 | 111.6 ms | 114.6 ms | 通过 |
| 39 | 100/100 | 11 | 0 | 112.9 ms | 114.2 ms | 通过 |

三次回放共执行 300 个控制步、33 次异步推理，没有 deadline miss、非有限动作或控制周期
overrun。VLASH 软件链路和推理性能已经打通。

回放报告：

- `outputs/eval/pi05_vlash_epoch10_replay.json`
- `outputs/eval/pi05_vlash_epoch10_replay_ep20.json`
- `outputs/eval/pi05_vlash_epoch10_replay_ep39.json`

## 7. 当前硬件状态

- Follower：`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00`
- Leader：`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00`
- Top camera：`/dev/video4`
- Wrist camera：`/dev/video6`
- Follower 标定：
  `/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower/tk_follower.json`

两个相机当前可读写且未被占用，标定文件存在。Follower 串口权限为 `root:dialout 660`，
当前用户 `cqy` 不属于 `dialout`，因此当前命令会在连接机械臂之前失败。

推荐永久修复：

```bash
sudo usermod -aG dialout cqy
```

执行后退出当前登录会话并重新登录，再确认：

```bash
id
test -r /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
test -w /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
```

仅用于当前设备会话的临时权限方案：

```bash
sudo chmod a+rw /dev/ttyACM0
```

## 8. 上机前检查

执行真机命令前必须确认：

1. 机械臂工作空间内无人手、线缆和硬障碍物。
2. 机械臂已放在数据集覆盖的初始姿态和物体布局附近。
3. Top 与 wrist 相机位置、方向、焦距和训练数据一致。
4. 操作者可以立即断电或按下急停。
5. 串口和相机没有被其他进程占用。
6. 明确接受当前动作质量 smoke 未通过的风险。

设备占用检查：

```bash
fuser /dev/ttyACM0 /dev/video4 /dev/video6
```

没有输出才表示设备未被其他进程占用。

## 9. 反馈闭环修订后的 60 秒真机命令

以下命令使用最终 epoch 10 checkpoint、VLASH overlap 5、每关节最大相对目标 5.0、动作
过滤器和 5 秒相机预热。VLASH 使用实际下发动作和关节跟踪反馈校正未来状态。由于当前数据的
正常 action/state lag 会持续触发旧 stall guard，本次保持 `stall_guard_ticks=0`；deadline
fail-closed 和机器人侧 `max_relative_target=5.0` 仍然启用。

```bash
cd /data/cqy_workspace/tk/lerobot_src

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/cqy_workspace/tk/lerobot_src/src
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}

MODEL=/data/cqy_workspace/tk/lerobot_src/outputs/train/pi05_so101_vlash_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
CALIB_DIR=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower
CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"backend":200},"wrist":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"backend":200}}'

/home/cqy/miniconda3/envs/lerobot_tk/bin/python -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --inference.type=vlash \
  --inference.execution_horizon=10 \
  --inference.inference_overlap_steps=5 \
  --inference.max_future_state_delta=5.0 \
  --inference.deadline_miss_limit=1 \
  --inference.timing_diagnostics=true \
  --inference.require_state_conditioning=true \
  --inference.require_offset_training=true \
  --camera_warmup_s=5.0 \
  --interpolation_multiplier=1 \
  --action_filter.enabled=true \
  --stall_guard_ticks=0 \
  --stall_guard_tolerance=0.001 \
  --pi05_prefix_backend=pytorch \
  --pi05_action_backend=pytorch \
  --policy.path="$MODEL" \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --policy.fuse_qkv=false \
  --policy.fuse_gate_up=false \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=tk_follower \
  --robot.calibration_dir="$CALIB_DIR" \
  --robot.max_relative_target=5.0 \
  --robot.cameras="$CAMERAS" \
  --task="Put the block in the bin" \
  --duration=60 \
  --fps=30 \
  --seed=1000 \
  --device=cuda \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=false
```

运行时如出现以下任一情况，应立即停止并断电检查：

- 连续 `max_relative_target` clamp
- `stall/contact guard tripped`
- `VLASH missed chunk deadline`
- 相机读取失败或控制频率明显低于 30 Hz
- gripper、shoulder 或 elbow 出现与任务无关的大幅跳变

完成上面的环境变量、`MODEL`、`ROBOT_PORT`、`CALIB_DIR` 和 `CAMERAS` 定义后，也可以使用
仓库 launcher 执行同一组 VLASH 核心参数：

```bash
VLASH_EXECUTION_HORIZON=10 \
VLASH_INFERENCE_OVERLAP_STEPS=5 \
VLASH_MAX_FUTURE_STATE_DELTA=5.0 \
VLASH_DEADLINE_MISS_LIMIT=1 \
VLASH_CAMERA_WARMUP_S=5.0 \
VLASH_DURATION_S=60 \
LEROBOT_PYTHON=/home/cqy/miniconda3/envs/lerobot_tk/bin/python \
examples/inference/run_pi05_vlash.sh "$MODEL" \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --action_filter.enabled=true \
  --stall_guard_ticks=0 \
  --stall_guard_tolerance=0.001 \
  --pi05_prefix_backend=pytorch \
  --pi05_action_backend=pytorch \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=tk_follower \
  --robot.calibration_dir="$CALIB_DIR" \
  --robot.max_relative_target=5.0 \
  --robot.cameras="$CAMERAS" \
  --task="Put the block in the bin" \
  --fps=30 \
  --seed=1000 \
  --device=cuda \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=false
```

## 10. 后续建议

不建议直接增加更多 epoch。当前训练 loss 已进入平台区，而动作尾部风险主要集中在少数关节和
少量样本。更有效的下一步是：

1. 定位 quick report 中触发 gripper action 0 离群值的帧和随机种子。
2. 增加这些初始状态附近的数据，并统一 gripper 开合时机。
3. 保留独立验证 episode，不再把全部数据用于训练。
4. 对 epoch 8 和 epoch 10 做更大规模的多种子、future-state-aware 对比。
5. 质量门通过后，再从 5 秒、低风险物体布局开始真机测试。

## 11. 2026-07-30 动作反馈闭环修订

针对 30 秒真机运行中大量动作被滤波器和机器人 clamp 改写、VLASH 仍按原始动作外推未来状态的
问题，推理链路完成以下修订：

- steady-state 推理改为在本周期动作实际发送后触发。
- 使用机器人返回的实际发送目标，而不是仅使用原始策略动作。
- 根据最近 8 个控制周期的实测状态变化，分别估算各关节跟踪增益。
- 未来状态由“当前实测状态 + 当前实际发送目标 + 剩余动作 + 跟踪增益”计算。
- `inference_overlap_steps` 改为 `5`；包含当前已发送动作后，实际状态投影为 6 步，仍在 checkpoint
  的 `temporal_offset_max_steps=8` 训练范围内。
- VLASH 强制 `interpolation_multiplier=1`，避免动作反馈与策略 action index 错位。
- 新增 `camera_warmup_s`，launcher 默认在全部相机连接后额外预热 5 秒。
- `max_future_state_delta=5.0` 和机器人 `max_relative_target=5.0` 保持不变。

聚焦回归结果：`149 passed`；ruff、shell 语法和 diff whitespace 检查通过。

最终 checkpoint 在 episode 0 上按 30 Hz 回放 100 步：

| 控制步 | 推理次数 | Deadline miss | P50 | P95 / Max | 结果 |
|---:|---:|---:|---:|---:|---|
| 100/100 | 11 | 0 | 111.27 ms | 113.50 ms | 通过 |

回放参数为 `execution_horizon=10`、`inference_overlap_steps=5`、
`max_future_state_delta=5.0`。反馈源为数据集记录动作，不会连接或驱动机械臂。报告文件：

`outputs/eval/pi05_vlash_epoch10_feedback_replay_ep0.json`

下一次真机运行需要与旧日志对比以下指标：

1. `hardware_clamps / action_feedback_count` 是否显著低于旧运行的约 72%。
2. `tracking_gain_mean` 是否保持有限且随关节运动更新。
3. gripper clamp 次数和闭合时间是否下降。
4. P95 是否继续低于 150 ms，最大时延是否低于 167 ms，deadline miss 是否为 0。

## 12. 反馈闭环 30 秒真机日志分析

原始 30 秒日志已在完成统计后按清理请求删除；本节保留其分析结果。当前真机基线日志为：

`outputs/rollout/vlash_epoch10/vlash_60s_feedback.log`

运行正常达到 30 秒时长并退出，完成相机和机器人断开；没有 traceback、ERROR、CRITICAL、
deadline miss、控制周期过慢或相机错误。5 秒相机预热已实际执行。

| 指标 | 结果 |
|---|---:|
| VLASH 推理次数 | 89 |
| steady-state 推理次数 | 88 |
| steady-state 平均时延 | 120.91 ms |
| steady-state P95 | 126.11 ms |
| steady-state 最大时延 | 130.29 ms |
| 首次推理时延 | 382.93 ms |
| 最后一次反馈统计 | 878 |
| 最后一次 hardware clamp 统计 | 646（73.6%） |
| 最后一次 filter rewrite 统计 | 857（97.6%） |
| 跟踪增益均值 / 中位数 | 0.203 / 0.204 |
| 跟踪增益范围 | 0.024-0.315 |

日志中共有 651 个控制周期打印硬件 clamp，总关节改写次数为 1,256：

| 关节 | Clamp 次数 | 平均残差 | P95 残差 | 最大残差 |
|---|---:|---:|---:|---:|
| shoulder lift | 408 | 10.999 | 28.673 | 32.939 |
| gripper | 377 | 12.070 | 21.152 | 24.247 |
| elbow flex | 295 | 11.648 | 25.127 | 26.543 |
| shoulder pan | 149 | 4.212 | 8.172 | 8.851 |
| wrist flex | 27 | 0.398 | 1.038 | 1.206 |
| wrist roll | 0 | - | - | - |

与修订前约 72% 的 clamp 周期相比，本次约 73.6%，没有下降。反馈闭环已经生效，
`tracking_gain_mean` 持续更新且 future projection 为 6 步，但它主要修正下一 chunk 的状态条件，
不能改变当前 chunk 中已经生成的远距离绝对目标。夹爪 clamp 数量有所下降，但 shoulder lift、
elbow flex 和 shoulder pan 的改写增加，因此当前主要限制仍然是策略目标、动作滤波器和真实电机
可达轨迹不一致，而不是 VLASH 推理时延。

launcher 当前默认 `VLASH_DURATION_S=60`。60 秒运行主要用于继续验证长时稳定性和任务循环，
不能仅凭延长时间预期抓取准确率提升。
