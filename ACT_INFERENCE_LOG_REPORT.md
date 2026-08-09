# SO-101 ACT 推理日志报告

本文总结 2026-08-09 在 SO-101 Follower 机械臂上完成的两次本地 ACT 真机推理，重点分析控制循环时序和动作块行为。两次推理均未记录任务成功率，因此本文不能用于判断策略精度或任务成功率是否提升。

## 共同配置

- 策略检查点：`outputs/train/act_so101_test_data_20k/checkpoints/020000/pretrained_model`
- 策略：ACT，约 5160 万参数
- 输入：`observation.state`、`observation.images.top` 和 `observation.images.wrist`
- 相机：两路 OpenCV 相机，640 x 480，30 FPS
- 机械臂：SO-101 Follower，6 维动作
- 推理后端：同步推理（`inference.type=sync`）
- 目标控制频率：30 FPS
- 运行时间：无限（`duration=0`），使用 `Ctrl+C` 终止
- 动作插值：关闭（`interpolation_multiplier=1`）
- 相对目标安全限幅：关闭（未配置 `max_relative_target`）
- 计算设备：NVIDIA RTX 4090，CUDA

该检查点配置为 `chunk_size=100`，关闭 temporal ensemble，训练时使用 `n_action_steps=100`。ACT 内部动作队列为空时，模型会一次预测完整的 100 步动作块。

## 第一次推理：每 10 步强制重规划

第一次推理增加了以下同步推理参数：

```text
inference.max_actions_per_chunk=10
```

策略每次模型调用仍然生成 100 步动作，但同步推理引擎执行 10 步后便重置策略队列。在 30 FPS 下，这会约每 0.33 秒强制调用一次模型，并丢弃每个预测动作块中约 90% 的动作。

退出时的关键统计为：

```text
SyncInferenceEngine stopped (horizon_replans=317, clamp_replans=0)
```

日志中出现的低速控制循环包括：

```text
29.1 Hz
21.3 Hz
20.4 Hz
7.4 Hz
```

`Sync guarded replan: execution prefix reached 10 actions` 日志约每秒出现三次。可见卡顿由两个因素共同造成：

1. 每次动作队列被重置后，模型在控制线程内同步推理，期间控制线程被阻塞。
2. 新动作块的第一个动作没有与上一个动作块的最后一个动作进行平滑衔接。

使用 `Ctrl+C` 后，本次推理正常结束。机械臂返回初始位置，两路相机正常断开，Follower 在断开连接时关闭扭矩。整个过程没有发生限幅触发的重规划。

## 第二次推理：自然执行 50 步动作队列

第二次推理删除了 `inference.max_actions_per_chunk`，改为直接覆盖策略执行步数：

```text
policy.n_action_steps=50
```

ACT 动作队列会自然耗尽，不再由同步推理引擎强制调用 `policy.reset()`。模型约每 1.67 秒生成一次新动作块。

初始化日志确认参数已经生效：

```text
SyncInferenceEngine initialized (
    device=cuda,
    action_keys=6,
    max_actions_per_chunk=None,
    replan_on_clamp=False
)
```

本次推理运行约 126 秒，退出统计为：

```text
SyncInferenceEngine stopped (horizon_replans=0, clamp_replans=0)
```

日志中共有 13 个控制循环低于 30 Hz，约占预期控制循环总数的 0.34%。具体为：

```text
5.5 Hz   # 第一个控制循环
13.3 Hz
11.4 Hz
12.5 Hz
26.5 Hz
26.2 Hz
29.5 Hz
19.8 Hz
21.9 Hz
20.8 Hz
20.2 Hz
26.1 Hz
29.9 Hz
```

第一个控制循环只有 5.5 Hz，符合 CUDA 和卷积算法首次预热的特征。后续绝大部分控制循环满足 33.3 ms 的时间预算，其余低速循环属于孤立的延迟尖峰，而不是持续性的控制频率下降。

本次推理同样在 `Ctrl+C` 后正常结束。机械臂返回初始位置，两路相机正常断开，日志中没有相机、串口、CUDA 或限幅重规划错误。

## 离线模型耗时

在不连接机械臂和相机、输入张量已经位于 GPU 的条件下，对 ACT 模型进行了离线耗时测试。

| 模式 | 中位耗时 | 观测 p95 | 输出类型 |
| --- | ---: | ---: | --- |
| FP32 | 4.76 ms | 4.77 ms | `torch.float32` |
| CUDA autocast | 4.17 ms | 4.24 ms | `torch.float16` |

Autocast 仅将模型耗时降低约 0.6 ms，无法解释或解决真机控制循环中 75-180 ms 的延迟尖峰，并且会改变推理输出的数据类型。因此第二次推理继续采用检查点原有的 FP32 推理行为。

## 分析结论

第一次推理中的 10 步强制重规划是周期性卡顿的直接配置原因。第二次推理已经成功移除这一行为：`max_actions_per_chunk=None`、`horizon_replans=0`，并且不再出现重复的 `Sync guarded replan` 日志。

剩余延迟不能完全归因于 ACT 模型计算。离线模型耗时低于 5 ms，而 30 Hz 控制循环的时间预算为 33.3 ms。真机控制循环还需要完成以下操作：

- 同步读取机械臂关节状态；
- 获取两路相机图像；
- 将图像从 uint8 转换为 float32；
- 将图像转换为通道优先格式并生成连续内存副本；
- 将图像同步传输到 CUDA；
- 对模型动作进行后处理；
- 通过串口向电机发送动作。

当前 INFO 级别的警告只记录整个控制循环的总耗时，无法判断每个孤立延迟尖峰具体来自哪个环节。

运动不连续和计算延迟是两个不同的问题。当前检查点关闭 temporal ensemble，推理时也没有使用动作插值，因此新预测的 50 步动作块不保证从上一个动作块的最终指令连续开始。移除 `max_relative_target` 后，这类动作跳变会在没有相对运动裁剪的情况下直接进入电机指令路径。

## 当前基线与后续测量

第二次推理更适合作为当前 ACT 真机部署基线：

```text
policy.n_action_steps=50
inference.max_actions_per_chunk=None
interpolation_multiplier=1
fps=30
duration=0
```

后续工作应分别处理控制循环延迟和轨迹连续性：

1. 分别记录关节读取、每路相机读取、预处理、模型执行、后处理和串口动作写入的耗时。
2. 在当前动作队列耗尽前，使用后台工作线程提前生成下一个 ACT 动作块。
3. 在不改变示范数据 30 Hz 动作时序的前提下，对相邻动作块的边界进行平滑融合。
4. 记录每个 episode 的任务成功率、失败原因、完成时间和可见停顿；仅凭时序日志无法评价策略质量。
