# SO-101 ACT / PI0.5 训练与真机推理交接文档

本文记录 `tk_follower` 数据集、ACT 历史基线、PI0.5 本地训练结果，以及当前 SO-101 真机标定、遥操和推理流程。文中的当前配置对应 2026-07-21 的仓库状态。

## 1. 当前状态摘要

- 当前真机推理模型：本地训练完成的 PI0.5 `005613` 检查点。
- 当前推理入口：`lerobot-rollout --config_path=pi05_rollout.json`。
- 当前推理方式：本机 CUDA + RTC 后台推理线程，不依赖远程 `PolicyServer`、SSH 隧道或 `robot_client`。
- 当前推理时长：60 秒，控制频率 30 FPS。
- 当前相对目标限幅：`max_relative_target=6.0`。
- 当前动作插值倍率：`interpolation_multiplier=1`，即不额外插入中间动作。
- ACT 20k 模型仍然保留，作为已经验证过的历史训练基线。

关键路径：

```text
仓库：      /data/cqy_workspace/tk/lerobot_src
数据集：    /data/cqy_workspace/tk/lerobot_src/data/so101_test_data
PI0.5 模型：/data/cqy_workspace/tk/lerobot_src/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
推理配置：  /data/cqy_workspace/tk/lerobot_src/pi05_rollout.json
标定根目录：/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
```

## 2. 本机环境与硬件

### 2.1 软件环境

- LeRobot 源码：`/data/cqy_workspace/tk/lerobot_src`
- Conda 环境：`/home/cqy/miniconda3/envs/lerobot_tk`
- Hugging Face 缓存：`/data/cqy_workspace/tk/hf_cache/huggingface`
- 当前机器使用 CUDA 进行 PI0.5 训练和推理。
- 真机运行默认关闭数据显示和声音。

### 2.2 稳定设备路径

不要依赖可能在重启或重新插拔后变化的 `/dev/ttyACM0`、`/dev/ttyACM1`。当前稳定路径如下：

```text
Follower：/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
Leader：  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00
Top 相机：/dev/video4
Wrist 相机：/dev/video6
```

机械臂 ID：

```text
Follower：tk_follower
Leader：  tk_leader
```

相机规格均为 640x480、30 FPS。使用 `/dev/video4`、`/dev/video6` 这类绝对路径时，OpenCV 配置必须包含 V4L2 后端 `"backend": 200`。

## 3. 标定文件与电机配置

### 3.1 当前标定文件

标定文件已经复制到项目使用的 Hugging Face 缓存中：

```text
/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower/tk_follower.json
/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/teleoperators/so_leader/tk_leader.json
```

旧目录中的备份仍然保留：

```text
/home/cqy/.cache/huggingface/lerobot/calibration/...
```

迁移时已经完成以下验证：

- 新旧文件的 SHA-256 一致。
- 两个 JSON 文件均能正常解析。
- LeRobot 能从每个文件中读取 6 个电机的标定数据。
- 当前不存在 `tk_follower_v2.json` 或 `tk_leader_v2.json`。

`robot.calibration_dir` 和 `teleop.calibration_dir` 应指向包含对应 JSON 的具体目录，而不是标定根目录：

```text
Follower calibration_dir：/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower
Leader calibration_dir：  /data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/teleoperators/so_leader
```

标定文件名由设备 ID 决定。命令使用 `--robot.id=tk_follower` 时会加载 `tk_follower.json`；改变 ID 会寻找另一个文件。

### 3.2 标定文件对各流程的影响

- 离线训练只读取数据集，不读取当前机械臂的标定 JSON。迁移标定文件不会改变已训练模型。
- 标定、遥操、数据采集、回放和真机推理会读取标定文件。
- 后续命令显式传入 `calibration_dir` 时，不需要另外设置 `HF_LEROBOT_CALIBRATION`。
- 若命令不显式传入 `calibration_dir`，则需要确保默认缓存路径或 `HF_LEROBOT_CALIBRATION` 与文件位置一致。

### 3.3 是否需要同时重新标定两个臂

不需要。只重新组装、换电机、改变零位或运动范围的机械臂需要重新标定。Follower 改动时只标定 Follower；Leader 未变化时可以继续使用原来的 `tk_leader.json`。遥操前应分别确认两份标定文件与实体机械臂匹配。

## 4. 可直接运行的标定与遥操命令

以下代码块只包含命令本身。不要复制终端提示符，例如 `(base) cqy@...$` 或续行提示符 `>`。

### 4.1 标定 Follower

```bash
cd /data/cqy_workspace/tk/lerobot_src
conda activate lerobot_tk
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 --robot.id=tk_follower --robot.calibration_dir=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower
```

### 4.2 标定 Leader

```bash
cd /data/cqy_workspace/tk/lerobot_src
conda activate lerobot_tk
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00 --teleop.id=tk_leader --teleop.calibration_dir=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/teleoperators/so_leader
```

### 4.3 遥操

```bash
cd /data/cqy_workspace/tk/lerobot_src
conda activate lerobot_tk
lerobot-teleoperate --teleop.type=so101_leader --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00 --teleop.id=tk_leader --teleop.calibration_dir=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/teleoperators/so_leader --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 --robot.id=tk_follower --robot.calibration_dir=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration/robots/so_follower
```

标定前先断开另一条臂，确认端口对应关系，并检查 6 个电机的供电和串联线。标定过程中按程序提示移动关节，不要强行越过机械限位。

## 5. 数据集

### 5.1 路径与标识

```text
本机路径：/data/cqy_workspace/tk/lerobot_src/data/so101_test_data
数据标识：admin123/so101_test_data
```

历史 ACT 服务器副本位于：

```text
/workspace/tk/lerobot_src_act/data/so101_test_data
```

### 5.2 数据概况

- 40 个 episode
- 每个 episode 449 帧
- 总帧数 17,960
- 30 FPS
- 每个 episode 约 15 秒
- 任务文本：`Put the block in the bin`
- 两路 H.264 视频：`observation.images.top`、`observation.images.wrist`
- 状态维度：6
- 动作维度：6
- 关节顺序：`shoulder_pan.pos`、`shoulder_lift.pos`、`elbow_flex.pos`、`wrist_flex.pos`、`wrist_roll.pos`、`gripper.pos`

### 5.3 已完成的数据验证

- 本机与历史服务器副本的 7 个数据文件 SHA-256 一致。
- 40 个 episode 均能由 `LeRobotDataset` 加载。
- Parquet、时间戳、状态、动作和两路视频均可读取。
- 没有 NaN 或 Inf。
- 所有 episode 均为 449 帧。
- `stats.json` 包含 mean/std 和 q01/q10/q50/q90/q99。

该数据集已经用于 ACT 和 PI0.5 训练。

## 6. ACT 历史训练基线

本节保留 ACT 的已验证结果，但 ACT 已不再是当前默认真机推理入口。

### 6.1 训练配置

- 策略：ACT，约 52M 参数
- 视觉骨干：ImageNet 预训练 ResNet18
- `chunk_size=100`
- `n_action_steps=100`
- 状态、动作和图像归一化：MEAN_STD
- 条件 VAE：启用
- Optimizer：AdamW
- 学习率：`1e-5`
- Weight decay：`1e-4`
- Batch size：8
- 总步数：20,000，约 8.9 epoch
- 每 5,000 step 保存检查点
- 不使用 AMP、学习率调度器、WandB 或 Hub 上传
- 没有单独验证集

### 6.2 训练结果

| Step | Total loss |
| ---: | ---: |
| 100 | 9.090 |
| 1,000 | 1.921 |
| 5,000 | 约 0.355 |
| 10,000 | 约 0.190 |
| 16,700 | 0.128 |
| 20,000 | 0.119 |

最终指标：

```text
loss     = 0.119
l1_loss  = 0.105
kld_loss = 0.001
```

本机最终模型：

```text
/data/cqy_workspace/tk/lerobot_src/outputs/train/act_so101_test_data_20k/checkpoints/020000/pretrained_model
```

模型已与服务器版本逐文件校验。`model.safetensors` 大小约 207 MB。

## 7. PI0.5 初始权重与训练配置

### 7.1 训练开始时加载的权重

训练脚本通过以下参数加载 PI0.5 基础预训练目录：

```text
--policy.pretrained_path=/data/cqy_workspace/tk/model_assets/lerobot/pi05_base
```

实际基础权重文件：

```text
/data/cqy_workspace/tk/model_assets/lerobot/pi05_base/model.safetensors
```

该文件为 14,467,165,872 bytes，约 14.47 GB，权重为 float32。它是微调起点，不是当前真机推理加载的最终训练权重。

### 7.2 本地训练配置

完成训练的脚本：

```text
/data/cqy_workspace/tk/lerobot_src/src/lerobot/scripts/train_pi05_so101_local_10epochs.sh
```

复现同一套训练配置时执行：

```bash
cd /data/cqy_workspace/tk/lerobot_src
conda activate lerobot_tk
bash /data/cqy_workspace/tk/lerobot_src/src/lerobot/scripts/train_pi05_so101_local_10epochs.sh
```

主要配置：

- 策略：PI0.5
- 设备：本机 CUDA GPU 0
- 模型计算 dtype：bfloat16
- Batch size：32
- Worker：2
- 总步数：5,613
- 保存频率：1,123 step
- 启用 gradient checkpointing
- 不冻结视觉编码器
- 不只训练 expert
- 不启用模型编译
- 不进行环境评估
- 不使用 WandB 或 Hub 上传
- Hugging Face 和 Transformers 使用离线模式

`train_pi05_so101_local_2500.sh` 是 2,500 step 的阶段性训练配置；当前最终 `005613` 检查点来自上面的 10 epoch 脚本。

epoch 估算：

```text
steps_per_epoch = 17,960 / 32 = 561.25
epochs          = 5,613 / 561.25 ≈ 10.00
```

### 7.3 训练结果

- 训练完成标志：日志包含 `End of training`。
- 总训练时间：约 7 小时 47 分钟。
- 训练末尾 loss：约 `0.009`。
- 训练末尾显存记录：约 39.55 GB。
- 检查点：`001123`、`002246`、`003369`、`004492`、`005613`。

训练日志：

```text
/data/cqy_workspace/tk/lerobot_src/logs/pi05_so101_local_10epochs_bs32.log
```

最终训练权重：

```text
/data/cqy_workspace/tk/lerobot_src/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model/model.safetensors
```

最终 `model.safetensors` 为 9,354,050,752 bytes，约 9.35 GB，保存为 bfloat16。推理加载的是包含该文件的 `pretrained_model` 目录，而不是基础权重目录。

## 8. 当前 PI0.5 真机推理

### 8.1 当前架构

```text
本机相机和关节状态 -> 本机 PI0.5 / CUDA / RTC -> 本机动作队列 -> Follower
```

RTC 在本机后台线程持续生成动作，因此模型计算与控制循环是并行的；它不等同于历史 ACT 的远程 gRPC 异步推理。当前流程不需要登录服务器，也不需要启动 SSH 隧道。

### 8.2 当前配置文件

配置文件：

```text
/data/cqy_workspace/tk/lerobot_src/pi05_rollout.json
```

当前关键参数：

| 配置 | 当前值 | 作用 |
| --- | ---: | --- |
| `strategy.type` | `base` | 基础 rollout 策略 |
| `inference.type` | `rtc` | RTC 后台推理 |
| `execution_horizon` | 10 | 每轮计划的执行窗口 |
| `max_guidance_weight` | 10.0 | RTC guidance 权重上限 |
| `prefix_attention_schedule` | `EXP` | RTC prefix attention 调度 |
| `queue_threshold` | 10 | 动作队列补充阈值 |
| `interpolation_multiplier` | 1 | 不额外插值 |
| `robot.max_relative_target` | 6.0 | 单次相对目标安全限幅 |
| `duration` | 60 | 推理 60 秒 |
| `fps` | 30 | 控制频率 30 Hz |
| `device` | `cuda` | 本机 GPU 推理 |

该 JSON 已通过配置解析验证，模型、设备、相机和标定均使用绝对路径。

### 8.3 可直接复制的启动命令

```bash
cd /data/cqy_workspace/tk/lerobot_src
conda activate lerobot_tk
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
lerobot-rollout --config_path=pi05_rollout.json
```

这组命令不依赖 `run_pi05_rollout.sh`，也不使用临时路径变量。硬件连接前，程序加载 PI0.5 权重可能需要约一分钟。

`HF_HOME` 指向已迁移的缓存位置；离线变量防止运行时访问 Hub。标定路径已经写在 JSON 中，因此这里不需要 `HF_LEROBOT_CALIBRATION`。

### 8.4 `max_relative_target` 的含义

`max_relative_target=6.0` 不是任务目标，也不会改变模型想抓取的物体。它限制每个控制步中目标关节位置相对当前关节位置的最大变化，用于降低突然大幅运动的风险。

如果完全不配置 `robot.max_relative_target`，当前 SO follower 配置默认值为 `None`，不会执行该相对目标限幅。这样可能减少 clamp 警告，但同时移除了重要的软件安全保护，不建议直接用于首次真机测试。

日志中的：

```text
Relative goal position magnitude had to be clamped to be safe.
```

表示模型给出的目标超过当前限幅，程序已把它改成 `safe goal_pos`。大量持续限幅说明模型动作、初始姿态、机械臂实际跟随速度或控制参数之间存在较大偏差，不能只通过继续增大限幅值来隐藏。

### 8.5 插值与动作顺滑度

当前 `interpolation_multiplier=1`，不会在两个策略动作之间额外生成插值点。大于 1 会增加中间控制点，可能让命令轨迹更细，但也会改变动作执行节奏，不能保证解决由模型输出跳变、持续限幅、机械延迟或标定误差导致的抖动。

RTC 的动作融合参数也会影响轨迹。调整 `execution_horizon`、`max_guidance_weight`、`queue_threshold` 或插值倍率时，应一次只修改一个参数，并以固定初始场景重复评估。

## 9. 相机配置与验证

当前 JSON 中两路相机使用：

```json
{
  "top": {
    "type": "opencv",
    "index_or_path": "/dev/video4",
    "width": 640,
    "height": 480,
    "fps": 30,
    "backend": 200
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": "/dev/video6",
    "width": 640,
    "height": 480,
    "fps": 30,
    "backend": 200
  }
}
```

两路相机已经分别验证可以通过 V4L2 以 640x480、30 FPS 打开。若绝对设备路径未指定 `backend=200`，OpenCV 可能选择 FFMPEG，随后出现以下误导性错误：

```text
failed to set capture_width=640 (actual_width=640, width_success=False)
ioctl(VIDIOC_QBUF): Bad file descriptor
```

该报错不表示 640 宽度本身不受支持，首先检查是否明确选择了 V4L2 后端，以及设备是否被其他进程占用。

## 10. 当前代码改动对流程的影响

### 10.1 Feetech 电机通信重试

`SerialMotorsBus.torque_disabled()` 新增 `num_retry` 参数，并将它同时传给关闭和重新启用扭矩操作。SO follower 在电机配置阶段使用 `num_retry=3`。

该改动用于缓解 USB 串口偶发丢失状态包时的以下错误：

```text
Failed to write 'Lock' ... Incorrect status packet!
Failed to write 'Lock' ... There is no status packet!
```

重试只处理偶发通信丢包。若错误持续出现在同一电机，应检查：

1. 电机供电、电压规格和红色状态灯。
2. 串联线与接头是否松动。
3. 电机 ID 是否正确且没有重复。
4. 串口是否被另一个进程占用。
5. 使用的 `by-id` 路径是否确实属于目标机械臂。

### 10.2 PI0.5 多卡 FSDP 兼容处理

`lerobot_train.py` 在 FSDP 包装前检查浮点参数 dtype。若 PI0.5 参数包含混合 dtype 且 Accelerate 使用 bf16/fp16 mixed precision，先把参数统一为 fp32，再由 FSDP 在前向和反向时按 mixed precision 计算，避免 FSDP flatten 混合 dtype 参数失败。

相关实验文件：

```text
/data/cqy_workspace/tk/lerobot_src/src/lerobot/scripts/pi05_fsdp_5gpu.yaml
/data/cqy_workspace/tk/lerobot_src/src/lerobot/scripts/train_pi05_so101_1k.sh
```

当前 `005613` 最终模型来自本机单 GPU 训练脚本，不依赖这套 5 GPU FSDP 配置。

## 11. 常见错误与复制粘贴规则

### 11.1 路径被复制成多条命令

以下写法是错误的：

```text
--policy.path=/data/.../pi05_so101_local_10epochs_bs32/
checkpoints/005613/pretrained_model
```

Bash 会把第二行的 `checkpoints/...` 当成新命令，因此报告：

```text
bash: checkpoints/005613/pretrained_model: 没有那个文件或目录
```

同样，不能在路径中间用反斜杠换行后再添加缩进。反斜杠只能放在一条命令的参数边界处，并且必须是该行最后一个字符。当前推荐使用短的 `--config_path=pi05_rollout.json` 命令，避免复制长参数列表。

### 11.2 复制了终端提示符

不要复制：

```text
(base) cqy@ubuntu-Z790-EAGLE-AX:/data/...$
>
```

它们是 Bash 显示的提示符，不是命令内容。若提示符已经变成 `>`，通常表示引号、反斜杠或命令尚未闭合；按一次 `Ctrl+C` 回到正常 `$` 提示符，再复制完整代码块。

### 11.3 命令提前结束后缺少 `--robot.type`

如果长命令在 `--policy.path` 后被实际换行终止，Python 只收到前半段参数，随后会报告：

```text
ValueError: --robot.type is required for rollout
```

这不是 robot 配置丢失，而是后半段从未传给同一个 Python 进程。使用 JSON 配置文件可以避免该问题。

### 11.4 RTC 队列延迟警告

```text
Indexes diff is not equal to real delay. indexes_diff=4, real_delay=5
```

该警告表示 RTC 估算的动作索引差与实际队列延迟不完全一致。偶发一条不等同于硬件故障；如果持续出现并伴随停顿或不动作，应检查实际控制频率、GPU 推理速度、相机读取延迟和队列参数。

### 11.5 `libtinfo.so.6` 警告

从包含 Conda `lib` 的 `LD_LIBRARY_PATH` 中再次启动 Bash，可能出现：

```text
libtinfo.so.6: no version information available
```

这通常是 Bash 使用了 Conda 版本的 `libtinfo`，不是模型或机械臂错误。当前直接运行 `lerobot-rollout`，不需要再通过 `bash run_pi05_rollout.sh` 启动。

### 11.6 不要粘贴完整环境变量输出

`env` 输出可能包含 API key、访问令牌和其他凭据。排查路径问题时只输出目标变量，例如 `echo "$HF_HOME"`。如果凭据已出现在共享日志中，应立即撤销并轮换。

## 12. 真机安全与评估

1. 每次运行前将机械臂、方块和盒子恢复到采集时的初始分布。
2. 确认加载的是 `tk_follower.json`，并先通过遥操验证关节方向和范围。
3. 第一次使用新标定或新检查点时降低运行时长，并保留 `max_relative_target`。
4. 操作人员必须在机械臂旁，随时可以 `Ctrl+C` 或物理断电。
5. `return_to_initial_position=true` 会在正常结束时尝试返回初始位置，但不是硬件急停机制。
6. 不要连续发送多次 `Ctrl+C`；第一次中断会进入清理和返回初始位置，第二次可能触发强制退出。
7. 训练 loss 只反映训练集拟合程度。当前没有独立验证集，不能用低 loss 代替真机成功率。
8. 固定初始条件至少测试 10 次，记录抓取成功、放置成功、碰撞、抖动和限幅频率。
9. 若最终检查点不稳定，应比较中间检查点，而不是直接延长训练或放宽安全限幅。

## 13. 历史 ACT 远程异步推理归档

2026-07-20 因本机 GPU 被占用，ACT 曾使用以下架构：

```text
本机相机与关节状态 -> SSH 隧道 -> 服务器 gRPC PolicyServer -> ACT/GPU
本机 Follower 动作 <- SSH 隧道 <- 服务器动作块
```

历史环境：

```text
服务器：root@10.0.0.30:2233
项目：  /workspace/tk/lerobot_src_act
Python：/opt/conda/envs/lerobot/bin/python
端口：  28080
模型：  /workspace/tk/lerobot_src_act/outputs/train/act_so101_test_data_20k/checkpoints/020000/pretrained_model
```

该流程曾达到约 28.3 Hz，并使用 `actions_per_chunk=50`、`chunk_size_threshold=0.5`、`aggregate_fn_name=weighted_average`。它仍可作为 ACT 的历史调试参考，但不是当前 PI0.5 的启动方式。当前不要为运行 PI0.5 而启动 `policy_server`、SSH 隧道或 `lerobot.async_inference.robot_client`。

## 14. 后续变更原则

- 设备路径、模型路径、相机配置、推理参数和时长优先集中修改 `pi05_rollout.json`。
- 改用新 Follower ID 时，必须同时生成同名标定 JSON，并更新配置中的 `robot.id`。
- 改变相机名称或顺序前，应确认训练数据特征名称仍是 `top` 和 `wrist`。
- 改变模型时，应同时核对输入特征、归一化文件、动作维度和任务文本。
- 每次只改一个主要控制参数，并保留对应日志和真机结果，避免无法判断变化来源。
- 训练和真机推理命令不要记录访问令牌，也不要将完整 `env` 输出写入交接文档。
