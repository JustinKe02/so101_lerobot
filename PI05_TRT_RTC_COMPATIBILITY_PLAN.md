# PI0.5 TensorRT 与 RTC 兼容实施计划 V2

状态：Phase 0-1 的 CPU 实施与门禁已完成；Phase 2 的 T0 独立启动资产已创建，
仍等待离线门禁后才允许人工值守真机。旧全量模型作为唯一 T0 基线，S1/S2 在
通过多帧多 seed 门禁前禁止真机。

更新时间：2026-07-25

## 0. 当前实施记录

2026-07-24 已完成：

- 保存 22 条 legacy golden 测试，覆盖空队列、消费 0/1/3/5、delay 一致与不一致、30 Hz delay 桶、EXP 权重和 prefix normalize。
- `ActionQueue` 增加只读原子 snapshot、generation 和累计消费计数；默认 merge/delay 路径未改变。
- 增加 RTC timing/guidance 配置表面，默认固定为 `legacy + legacy_max`；Phase 2 模式在当前版本立即拒绝。
- 增加 PI0.5 prefix/action backend 显式选择和 PyTorch 强制回滚；action TensorRT 在实现前立即拒绝。
- 默认配置不导入 TensorRT runtime；prefix attach 在连接机器人之前且只出现一次。

CPU 门禁结果：

```text
legacy golden                         22 passed
RTC + rollout compatibility suite    269 passed, 3 skipped
py_compile                            passed
ruff                                 passed
git diff --check                     passed
```

本阶段没有修改 `modeling_pi05.py`、训练脚本或 PEFT 路径，也没有使用 GPU、TensorRT engine 或真机。Python 主路径的 TensorRT artifact 指纹/验证报告门禁和运行期 fatal backend 立即停机仍按计划留在后续 TensorRT 重验阶段；在这些门禁完成前不启用 T1/T2。

2026-07-24 真机失败复盘后的计划修订：

- S2 与 S1 在同一个训练帧上的三 seed 50 步 MAE 分别为
  `3.460/21.714/12.688` 与 `4.937/21.917/15.529`；旧全量模型为
  `0.191/0.210/0.199`。
- T0 基线从 S1 改为旧全量模型
  `outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model`。
- RTC 修复不能解除 S1/S2 的模型门禁；二者在完成多帧多 seed 离线验证前不再上机。
- 新时序路径需要额外记录限幅后的实际下发 action。持续严重限幅时使旧 prefix
  失效并无 guidance 重规划，不把 robot-unit action 直接写回模型空间 prefix。

2026-07-25 T0 启动资产：

- 新增 `run_pi05_full_rollout_rtc_actual.sh`，只加载旧全量 `005613`。
- 脚本显式固定 `actual_consumed + fixed=5 + PyTorch prefix/action`、30 Hz、
  chunk 50、horizon 10、queue 20 和 `max_relative_target=5.0`。
- `PI05_PREFLIGHT_ONLY=1` 只验证本地 checkpoint 和 RTC 配置并打印最终 CLI，
  不打开机器人串口或相机；正式运行默认 30 秒且默认不回初始位置。
- prefix health 由 `PI05_PREFIX_HEALTH_ENABLED` 显式控制且默认关闭，保证首轮 T0
  只改变时序；T0 通过后才作为 T0b 启用。残差阈值、连续严重次数和安全停止
  重规划次数均由脚本传入并在 preflight 中打印。
- 新脚本已通过 `bash -n` 和无硬件 preflight；验证过程中未加载模型权重到 GPU、
  未连接机器人、未打开相机，也未发送任何 action。
- 旧 `run_pi05_rollout.sh` 与 `run_pi05_s1_rollout.sh` 未修改。T0 的 CPU、真实帧
  与现场安全门禁尚未全部完成，因此创建脚本不代表已授权无人值守真机运行。

2026-07-25 RTC deployment-window 修复与 T0a 离线门禁：

- 确认 `chunk_size=50 / queue_threshold=20 / fps=30` 的重规划频率是
  `30 / (50 - 20) = 1 Hz`。在实际消费 4/5 步时，一个新 chunk 常态会执行
  `model index 4/5..33/34`；`execution_horizon=10` 不会自动截断队列。
- 新增 `examples/inference/evaluate_pi05_rtc_replay.py`，使用生产
  `ActionQueue.merge_actual_consumed`、relative-action re-anchor、固定 guidance delay 5
  和相同 noise，重放 Q20/Q30/Q35/Q40/Q45。报告始终标记
  `hardware_action_sent=false`，并原子记录 running/complete/failed。
- 风险样本 episode 7/frame 112/seed 1000 在 Q40 执行 `5..14` 时，误差从
  index 10 开始持续放大，index 14 最大误差为 `23.47`。Q45 只执行
  `4..8` 或 `5..9`，同一样本最大误差降至 `2.03/3.95`，且无新增限幅。
- 最终 Q45 报告覆盖 6 个定点、7 seeds、首 chunk 与稳态、actual-consumed 4/5，
  共 84 条记录且无跳过：

```text
report          outputs/eval/pi05_full_rtc_replay_q45_final_v1.json
deployment      passed
queue/software  nonfinite=0, invariant failures=0, repeat max=0
command delta   max=4.645 < 5.0
splice jump     max=2.959 < 5.0
latency P95     4.44 control steps
latency max     4.74 control steps
hardware        action_sent=false
```

- 单 demo 的远期逐点 GT 指标仍未全部通过，继续作为 semantic diagnostic，不改写
  原 formal v2 的 `formal_passed=false`。RTC 硬门禁使用真实出队命令连续性；这是因为
  counterfactual policy 轨迹在第一步后不再等价于 demo future state。
- actual-consumed 6/7/9 的压力测试均失败；6 步已经可能发送无 guidance 的 index 10。
  因此新增 opt-in `enforce_guided_execution_window`：T0a 在出队 index 10 前立即
  fatal、清队列并触发全局停止。legacy/Q20 默认行为不变。
- 新增 `run_pi05_full_rollout_rtc_q45.sh`。它固定 Q45、6 Hz nominal cadence、
  guided-window fail-closed 和 PyTorch/PyTorch，绑定最终 replay report、checkpoint
  指纹、evaluator SHA256 与配置合同。默认 duration=5 秒；无硬件 preflight 已通过。
- 新增 `run_pi05_full_rollout_rtc_q45_logged.sh` 与
  `examples/inference/analyze_pi05_t0a_log.py`，用于现场唯一的 5 秒人工值守门禁。
  启动器保存完整日志并自动检查 exit code、配置合同、实际消费/merge、稳态延迟、
  safety clamp、fatal、queue 错误和 TensorRT 误启用；分析器测试为 `5 passed`。
- 扩大测试在正确 HF cache 环境下等价为 `421 passed, 1 skipped`；PI0.5 RTC
  真实模型测试单独为 `5 passed`。未启动训练、TensorRT 或真机动作。

2026-07-26 T1 prefix 引擎离线校验、§9.1 指纹门禁与 KV 诊断重校准：

- 校验环境修复：`lerobot_tk` 本体无 tensorrt，须按 `docs/pi05_tensorrt.md` 配方挂载
  `PYTHONPATH=.pi05_tensorrt/python:third_party/tensorrt_10_13_0_35:src` 与对应
  `LD_LIBRARY_PATH`（TensorRT 10.13.0.35 / CUDA 12.8 / RTX 4090 sm_8.9）。
- `verify_engine` 升级 v2：`.plan.verified.json` 记录 engine sha256、prefix 权重指纹、
  架构指纹、TRT/CUDA/GPU 环境、全部阈值与实测值；attach 侧
  `validate_prefix_engine_verification` 在 `release_torch_prefix_modules()` 之前
  fail-closed 校验（缺报告/旧格式/未通过/引擎被改/权重或架构指纹不符/相机数、
  TRT 版本、算力不符全部拒绝），单测 17 passed。真实工件端到端 attach 门禁 PASS。
- KV 诊断重校准（经用户 2026-07-26 确认）：`kv_max` 2.0→3.0，新增离群元素占比
  ≤5% 检查（atol=rtol=5e-2）。依据：strongly-typed bf16 引擎误差集中于
  `value_14..17` 且随深度增长（worst 2.4375 @ value_17，离群 3.07%），属 TRT 融合
  注意力与 PyTorch SDPA 的 bf16 累加顺序差异；动作级三项独立测量无功能足迹
  （10 步去噪 max 0.00716，门禁 0.1；RTC 归一化 max ≤0.0039，门禁 0.020；
  机器人单位 max 0.24）。动作级阈值与 §8.2 门禁均未改动，证据记入
  `docs/pi05_tensorrt.md` 与 verified.json。
- §8.2 真实帧 RTC parity（旧全量 005613 + 7月23日重建引擎，固定同 delay）：
  delay 2-6 归一化 mean `0.00098-0.00119`、max `0.00316-0.00389` 全部通过；
  TRT 中位延迟 `120.8-125.0 ms`（PyTorch `138.4-141.4 ms`），TRT 推理稳定落入
  4 tick（固定 guidance delay=5 保持不变，余量增大）。报告
  `pi05_tensorrt_rebuild_bf16/rtc_parity_oldfull.json`；覆盖为脚本内置单帧对
  ×5 delay，完整 40 帧×5 seed 矩阵留待 T1 门禁报告阶段扩展。
- S1 checkpoint 引擎不可复用旧全量 verified.json；每个 checkpoint 的引擎须各自
  `--verify-engine` 后方可 attach。未连接机器人、未发送任何 action。

2026-07-26 T1 滤波回放门禁与 30-60 秒长时启动资产:

- 滤波回放门禁报告 `outputs/eval/pi05_full_rtc_replay_q45_filter_v1.json`
  (`--queue-thresholds 45 --action-filter`):84/84 记录、0 跳过,
  `deployment_passed=true`,契约含 `action_filter_enabled: true`。硬性安全项对
  无滤波基线全面改善:command_delta_max `4.645→2.200`(门禁 5.0)、
  splice_jump_max `2.959→0.591`、latency 持平(max 4.67 步);语义诊断项同向改善
  (error_p95 `9.256→6.851`、excess_clamp_residual `15.189→10.743`)。滤波器干预
  264/483 步(54.7%),峰值修正 8.62°,速度/加速度峰值恰被压至各关节包络上限
  (139.4→66.0 °/s、1715→475 °/s²),即真机 77 次安全钳制的源头(超包络指令)
  已在下发前消除。
- `run_pi05_full_rollout_rtc_actual.sh` 参数化:新增 `PI05_PREFIX_BACKEND`
  (默认 pytorch)、`PI05_TRT_PREFIX_ENGINE`、`PI05_ACTION_FILTER_ENABLED`
  (默认 false),默认值下行为与 T0 完全一致(preflight 复验通过)。tensorrt
  模式要求引擎与 `.plan.verified.json`(passed=true)存在,并自动按
  `docs/pi05_tensorrt.md` 配方挂载 TRT overlay 与运行库;draccus 预检新增
  prefix backend/engine 路径/action filter 三项解析一致性断言。
- 新增 T1 合约启动器 `run_pi05_full_rollout_rtc_q45_t1.sh`:上机前串联三重
  离线门禁——滤波回放报告(含 evaluator sha256 与 checkpoint 指纹)、引擎校验
  标记(格式 v2/passed/引擎与 checkpoint 路径/size+mtime_ns 重建检测/相机数
  与层数/三项指纹形态),以及 §8.2 parity 报告(delay 2-6 全覆盖、归一化
  mean ≤0.005、max ≤0.020);深层 sha256/指纹绑定仍由进程内 attach 门禁
  fail-closed 复验。三重门禁已对真实工件端到端预检通过。
- 新增 `run_pi05_full_rollout_rtc_q45_t1_logged.sh`:强制 30≤duration≤60
  (默认 45),日志与分析报告写入 `outputs/rollout/t1_q45/`;延迟门禁保持
  T0a 的 166.667 ms 不放宽,由 TensorRT 把峰值压入门内。
- `analyze_pi05_t0a_log.py` 参数化:`--duration`(令牌按 `%.0f` 生成)、
  `--expect-prefix-backend`(tensorrt 时要求 `TensorRT PI0.5 prefix enabled`
  出现,pytorch 时要求缺席)、`--expect-action-filter`(双向硬检查
  `Action output filter enabled` 出现/缺席)、`--run-kind`。旧 T0a 日志回归:
  默认参数下判定与既有报告完全一致(仅新增良性 `action_filter_not_enabled`
  通过项);合成 T1 日志在 T1 期望下通过、在 T0a 期望下正确拒绝。
- 已知状态:旧 `run_pi05_full_rollout_rtc_q45.sh` 现 fail-closed(基线报告记录的
  evaluator sha256 早于滤波功能引入),属契约设计行为;T0a 历史结果不受影响,
  如需重跑 T0a 须以当前 evaluator 重新生成无滤波基线报告。本日未连接机器人、
  未发送任何 action;T1 真机运行仍需用户到场并逐次授权。

## 1. 目标

在不破坏现有 PI0.5 RTC PyTorch 行为的前提下，依次完成：

1. 保留可随时回滚的旧 RTC 路径。
2. 修复 TensorRT 加速后 delay、queue merge 和实际动作消费不一致的问题。
3. 重新验证现有 VLM prefix TensorRT 的闭环等价性。
4. 在 RTC 时序稳定后，再实现 action expert TensorRT。
5. 将论文式精确 VJP RTC 留作独立研究分支，不混入本轮部署修复。
6. 在真机前建立独立于 RTC/TensorRT 的 PI0.5 多 seed 输出稳定性门禁。

本计划不重新训练 40 episodes，不修改 S1 checkpoint，不先增加动作滤波。

## 2. 兼容性合同

修改后必须继续支持当前基线：

```text
RTC timing       legacy
PI0.5 prefix     PyTorch
PI0.5 action     PyTorch
RTC guidance     当前 Jacobian-free 语义
```

兼容性要求：

- 所有新配置默认关闭。
- 未传入任何新参数时，现有 CLI、JSON 和 shell 脚本保持旧行为。
- `legacy + PyTorch` 的 seeded replay 中，action、delay、queue index 和 queue size 与修改前一致。
- 现有 `run_pi05_s1_rollout.sh` 在新路径通过门禁前不改变默认行为。
- 显式请求 TensorRT 时，engine 缺失、指纹不匹配或验证报告不合格必须直接退出；不得静默回退到 PyTorch。
- 新时序模式出现异常时，只需切回 `legacy`，不需要替换模型或重新导出 checkpoint。

## 3. 配置设计

时序配置属于 rollout inference，不放入模型训练使用的 `RTCConfig`。计划在 `RTCInferenceConfig` 中增加：

```text
timing_mode                    legacy | actual_consumed
guidance_delay_mode            legacy_max | fixed | rolling_p95
fixed_guidance_delay_steps     int
latency_warmup_inferences      int
latency_window_size            int
latency_percentile             float
delay_hysteresis_steps         float
delay_change_confirmations     int
timing_diagnostics             bool
```

第一版默认值：

```text
timing_mode                    legacy
guidance_delay_mode            legacy_max
fixed_guidance_delay_steps     5
latency_warmup_inferences      5
latency_window_size            32
latency_percentile             0.95
delay_hysteresis_steps         0.25
delay_change_confirmations     3
timing_diagnostics             false
```

PI0.5 backend 增加显式选择，同时保留旧 engine 参数的兼容语义：

```text
pi05_prefix_backend            auto | pytorch | tensorrt
pi05_action_backend            pytorch | tensorrt
pi05_tensorrt_prefix_engine    optional path
pi05_tensorrt_action_engine    optional path
```

默认值：

```text
pi05_prefix_backend            auto
pi05_action_backend            pytorch
```

`prefix_backend=auto` 保持旧语义：提供旧的 prefix engine 参数时使用 TensorRT，否则使用 PyTorch。显式 `prefix_backend=pytorch` 是强制回滚开关，即使配置中残留 engine 路径也不得加载 TensorRT，并打印 engine 被回滚配置忽略。显式选择 `tensorrt` 时，engine 缺失、加载失败或运行中失败都必须 fail closed 并安全停止，不自动换后端。

`action_backend=pytorch` 是 action expert 的默认旧行为。显式选择 `action_backend=tensorrt` 时必须提供 action engine。

## 4. 验证矩阵

| 编号 | RTC timing | Prefix | Action | 用途 | 初始状态 |
| --- | --- | --- | --- | --- | --- |
| B0 | legacy | PyTorch | PyTorch | 原有代码行为与回滚路径 | 必须保持 |
| B1 | legacy | TensorRT | PyTorch | 复现当前抓取退化 | 仅诊断 |
| T0 | actual_consumed | PyTorch | PyTorch | 旧全量模型单独验证时序修复 | 启动资产已就绪，待离线/真机门禁 |
| T1 | actual_consumed | TensorRT | PyTorch | VLM prefix 加速目标 | 待实现 |
| T2 | actual_consumed | TensorRT | TensorRT | 双端加速目标 | 后续阶段 |

禁止直接从 B0 跳到 T2。每次只改变一个变量。

## 5. Phase 0：冻结旧基线

### 5.1 保存基线信息

固定以下内容：

```text
model        outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
fps          30
chunk        50
denoise      10 steps
horizon      10
guidance     10.0
schedule     EXP
queue        20
interpolate  1
```

保存当前版本的：

- CLI 展开配置。
- seeded dataset replay action。
- 注入固定 latency 后的 guidance delay、merge delay、queue index 和首个执行 action。
- 当前 RTC 单元测试结果。
- 当前 TensorRT parity JSON。

### 5.2 Legacy golden trace

至少覆盖：

```text
首次队列为空
队列剩余 20 actions
推理期间消费 0/1/3/5 actions
wall delay 与实际消费相等
wall delay 与实际消费不相等
queue underflow
reset/clear
```

门禁：Phase 1 结束后，B0 golden trace 必须完全一致。Fake policy/CPU trace 要求逐元素、逐事件一致；同一 GPU 与固定 seed 的真实 PI0.5 trace 优先要求 `torch.equal`，若底层算子确有非确定性，最大误差不得超过 `1e-6`。不得新增 RNG 调用，稳态 latency P95 增幅不得超过 2%。

还需要回归历史实际使用过的配置：

```text
30 Hz / 10 steps / queue 10 / horizon 10 / interpolation 1
30 Hz / 10 steps / queue 20 / horizon 10 / interpolation 1
30 Hz / 10 steps / queue 30 / horizon 10 / interpolation 1
20 Hz /  5 steps / queue 30 / horizon 10 / interpolation 2
```

## 6. Phase 1：只增加观测，不改变行为

### 6.1 明确三个 delay

日志和代码变量改为：

```text
guidance_delay_estimate  推理开始前可用，用于 RTC prefix guidance
wall_latency_steps       本轮耗时换算值，仅用于诊断
actual_consumed_steps    推理期间真实取出的 action 数
```

legacy 模式仍按旧逻辑运行，不能因为变量改名而改变 merge 结果。

### 6.2 原子 queue snapshot

新增一次锁内获取的 snapshot：

```text
queue_generation
next_action_index
total_consumed
original_leftover
processed_leftover
queue_size
```

`clear` 和成功 `merge` 增加 generation。实验模式在 merge 前检查 generation；结果过期时丢弃本次 inference 并立即重规划。

### 6.3 诊断记录

仅当 `timing_diagnostics=true` 时记录：

```text
latency_ms
guidance_delay_estimate
wall_latency_steps
actual_consumed_steps
merge_skip
generation_before/after
index_before/after
queue_before/after
measured_control_fps
clamp_count
warmup_state
```

诊断默认关闭，避免日志 I/O 改变 legacy 延迟。

## 7. Phase 2：实现 actual-consumed 时序

该阶段只启用 T0，不加载任何 TensorRT engine。

### 7.1 Merge 的唯一权威值

在 `timing_mode=actual_consumed` 时：

```text
merge_skip = actual_consumed_steps
```

`wall_latency_steps` 不再决定 queue 切片。

关键规则：

- 首次推理期间队列为空，实际消费为 0，因此 merge skip 必须为 0。
- 推理期间执行了旧 queue 的 N 个 action，新 chunk 才跳过 N 个时间位置。
- generation 已改变的 inference result 不允许 merge。
- `actual_consumed_steps` 越界时 clamp 并记录错误；连续发生则停止 rollout。
- 第一版 experimental mode 只支持 `interpolation_multiplier=1`；其他值必须 fail fast，不能静默近似部分插值进度。

### 7.2 Guidance delay

推理开始前不知道本轮最终消费数，因此 guidance 仍需要估计。验证顺序为：

1. `fixed=5`：用于 PyTorch/TensorRT 单变量 A/B。
2. `rolling_p95`：用于生产候选。
3. `legacy_max`：只保留给旧路径。

rolling P95 规则：

- 前 5 次 inference 为 warmup，不写入窗口。
- 窗口保存最近 32 次稳态 latency。
- 少于 5 个有效样本时使用 fixed delay。
- 越过整数 delay 桶至少 0.25 step，且连续 3 次满足才切换。
- actual-consumed merge 和 stale-generation gate 负责兜底偶发尾延迟。

## 8. Phase 3：RTC 离线门禁

### 8.1 单元测试

必须新增：

- legacy 默认值与旧行为测试。
- snapshot 原子性与 generation 测试。
- 首轮空 queue 的 `actual_consumed=0` 测试。
- 并发消费 0/1/3/5 步的 merge 测试。
- stale result 被丢弃测试。
- warmup 不污染 P95 测试。
- 旧峰值退出窗口后 P95 可恢复测试。
- fixed guidance 与 actual merge 相互独立测试。
- TensorRT 未请求时不会导入 TensorRT runtime 测试。
- 旧 JSON 缺少新字段时解析为 `legacy + prefix auto + action pytorch` 测试。
- 显式 legacy 与省略 timing 字段的行为一致测试。
- 显式 prefix PyTorch 覆盖残留 engine 路径测试。
- 显式 TensorRT 缺 engine/report 时在 `robot.connect()` 前失败测试。
- TensorRT 运行中失败后安全停止且不切换 backend 测试。
- timing 配置传给 sync inference、未知枚举值和冲突配置时 fail fast 测试。

### 8.2 真实帧 RTC parity

数据矩阵：

```text
40 个 episode 过程帧
每帧 5 个 noise seed
delay 3/4/5/6/7
execution_horizon 10
EXP schedule
```

固定相同 delay 的首轮门槛：

```text
normalized mean abs   <= 0.005
normalized max abs    <= 0.020
robot-unit mean abs   <= 0.100
robot-unit max abs    <= 0.500
```

时序门槛：

```text
merge_skip == actual_consumed      100%
首次空 queue merge_skip             0
stale merge                         0 次
estimate 与 actual 相差 <= 1 step   >= 95%
estimate 与 actual 相差 > 2 steps   0 次
```

延迟边界扫频：在 120-150 ms 范围按 1 ms 注入 latency，相邻 1 ms 不得让时间对齐后的首动作跳变超过 1 robot unit。

## 9. Phase 4：重新启用 VLM prefix TensorRT

只有 T0 通过后才执行 T1。

### 9.1 Engine 指纹

prefix engine metadata 新增：

```text
prefix_weight_fingerprint
model_architecture_fingerprint
num_cameras
input/output shapes
TensorRT version
CUDA version
GPU compute capability
precision
```

不能只比较 checkpoint 路径。S1/S2 可以共享 prefix engine 的前提是 prefix 权重指纹完全一致。

### 9.2 A/B 规则

按以下顺序比较：

```text
T0 actual_consumed + PyTorch prefix + PyTorch action
T1 actual_consumed + TensorRT prefix + PyTorch action
```

两组使用同一旧全量 checkpoint、同一 seed、同一 observation replay 和同一 fixed guidance delay。通过后再切 rolling P95。

每次启动必须打印最终解析后的 policy、prefix/action backend、engine、timing mode、FPS、denoise steps、queue、horizon 和 interpolation，作为现场日志的配置凭证。

## 10. Phase 5：Action TensorRT

action TensorRT 不是 RTC 修复的替代品，只在 T1 通过后开始。

### 10.1 第一版导出边界

导出单个 `denoise_step`：

```text
inputs
  x_t                 [1, 50, 32]
  timestep            [1]
  prefix_pad_masks    [1, 712]
  18 x key/value      [1, 1, 712, 256]

output
  v_t                 [1, 50, 32]
```

engine 包含 action input/output projection、time MLP、18 层 Gemma action expert 和 AdaRMS。DynamicCache 必须展开为显式 K/V tensor。

### 10.2 RTC 语义边界

当前仓库和现有单元测试锁定的是 Jacobian-free guidance：

```text
correction = err
```

因此第一版 action TensorRT 只输出 `v_t`，RTC weights、err、guidance 和 queue merge 继续留在 PyTorch，可以复刻当前行为。

论文式精确 VJP：

```text
correction = d(x - t * denoiser(x)) / dx ^ T * err
```

不属于本阶段。普通 TensorRT forward 无法提供该 VJP。

### 10.3 Action engine 门禁

action engine 必须绑定 action 权重指纹，不能跨 S1 checkpoint 复用。依次验证：

1. 单步 `v_t` parity。
2. 固定 noise 的 10 步 final action parity。
3. 带 previous chunk 的 RTC parity。
4. 真实数据帧 robot-unit parity。
5. 时序修复后的 replay parity。

第一版仍由 Python 调用单步 engine 10 次。通过后再评估固定 10 步静态展开或 CUDA Graph。

## 11. 真机验证顺序

真机必须现场人工值守，Codex 不后台自动驱动机械臂。

### 11.1 安全试运行

每个候选先做 5 次无物体、5-10 秒运行：

```text
B0 legacy + PyTorch
T0 actual_consumed + PyTorch
T1 actual_consumed + prefix TensorRT
T2 actual_consumed + prefix/action TensorRT
```

任一候选出现非有限 action、stale merge、queue underflow、连续 clamp 或异常姿态时立即停止，不进入抓取测试。

### 11.2 抓取 A/B

每个通过安全门禁的候选至少 10 次，正式结论建议 20 次。使用 ABBA 顺序交错运行，保持：

```text
同一旧全量 checkpoint
同一场景和初始姿态范围
30 Hz
10 denoise steps
horizon 10
EXP guidance
max_relative_target 5.0
```

记录：

```text
抓取成功率
放置成功率
完成时间
实际控制 FPS
inference P50/P95/P99
guidance/actual delay 偏差
queue underflow/stale merge
安全限幅比例
P95/P99 速度与加速度
```

TensorRT 候选的工程验收目标为成功率下降不超过 10 个百分点、不得新增安全错误，并至少保留 1.1x 完整重规划加速。20 次试验只能作为工程筛选，不能解释为具有严格统计置信度的非劣性证明。

## 12. 回滚设计

任何阶段均可回到：

```text
timing_mode=legacy
pi05_prefix_backend=pytorch
pi05_action_backend=pytorch
pi05_tensorrt_prefix_engine=null
pi05_tensorrt_action_engine=null
```

显式 PyTorch backend 优先于配置中残留的 engine 路径，因此回滚不需要修改 JSON、删除 engine、替换模型或重做 calibration。回滚日志中不得出现 `TensorRT PI0.5 prefix enabled`，golden hash 必须恢复到 B0。

部署脚本在计划实施期间分为：

```text
run_pi05_s1_rollout.sh                   旧 S1 脚本，默认不变但暂停真机
run_pi05_full_rollout_rtc_actual.sh      已创建；旧全量模型 T0 新时序实验
run_pi05_full_rollout_rtc_actual_trt.sh  通过门禁后创建
```

在 T1 真机门禁通过前，不把新模式合入旧基线脚本的默认参数。

## 13. 计划修改文件

Phase 1-3 预计修改：

```text
src/lerobot/rollout/inference/factory.py
src/lerobot/rollout/inference/rtc.py
src/lerobot/policies/rtc/action_queue.py
src/lerobot/policies/rtc/latency_tracker.py
tests/policies/rtc/test_action_queue.py
tests/policies/rtc/test_latency_tracker.py
tests/test_rollout.py
examples/inference/verify_pi05_tensorrt_rtc.py
```

Phase 4 预计修改：

```text
src/lerobot/policies/pi05/tensorrt_prefix.py
src/lerobot/scripts/lerobot_export_pi05_tensorrt.py
run_pi05_full_rollout_rtc_actual_trt.sh
```

Phase 5 预计新增或修改：

```text
src/lerobot/policies/pi05/tensorrt_action.py
src/lerobot/scripts/lerobot_export_pi05_action_tensorrt.py
src/lerobot/policies/pi05/modeling_pi05.py
tests/policies/pi0_pi05/test_pi05_action_tensorrt.py
```

## 14. 时间估计与停止条件

```text
Phase 0-1  基线与观测             2-3 小时
Phase 2-3  actual-consumed 与测试  4-7 小时
Phase 4    prefix TRT 重验         2-3 小时
Phase 5    单步 action TRT 原型     1-3 天
真机 A/B                           1-2 小时现场时间
```

停止条件：

- B0 无法保持旧行为时，不进入 Phase 2。
- T0 的 PyTorch 抓取低于 B0 时，不启用任何 TensorRT。
- T1 固定-delay parity 不通过时，不进入 action TensorRT。
- T2 未保留实际端到端加速时，不继续做 10 步静态展开。

## 15. 实施起点

Phase 0-1 已完成并证明 B0 代码行为未变化。Phase 2 的 T0 独立启动脚本已创建，
原子的 actual-consumed merge、fixed guidance、实际下发 action 反馈和 prefix-health
仍需按门禁逐项确认。CPU 与真实帧门禁通过前，只允许无硬件 preflight，不运行
真机动作；T1/T2 脚本和 engine 继续后置。

## 16. 实施记录

### 16.1 T1 首次监督运行(2026-07-26)— 撞桌事件与根因

运行日志 `outputs/rollout/t1_q45/t1_q45_20260726_211300.log`(目标 45 s,约 9 s 中止,exit 1)。

事件链(全部有日志证据):

1. 策略在从未验证过的 >5 s 轨迹区域输出快速下压(包络内、滤波器限速内),
   shoulder_lift 实际位置到达约 -99(量程下限 -100,即触桌)。
2. 接触后关节堵转,目标仍以每 tick +5 前进,产生连续 26 次
   max_relative_target 钳制(21:14:35-36;另有启动瞬态 2 次微钳制,无害)。
3. 冲击使腕部相机 /dev/video6 掉出 USB(errno=19),读线程死亡,
   get_observation 抛错,rollout 中止。
4. teardown 按 disable_torque_on_disconnect=True 释放扭矩,机械臂瘫在桌面。

同一日志同时证明 T1 两项核心收益成立:TRT prefix 延迟 p50 131.7 / p95 139.9 /
max 141.5 ms(全部低于 166.667 门禁;T0a 为 191.5),自由空间钳制 0 次(T0a 为
77 次)。钳制门禁本身不放宽:本次 28 次钳制全部为接触诱发或启动瞬态,输出
滤波器无法也不应吸收接触误差。

### 16.2 事件后新增守护(改系统,不放宽门禁)

- `src/lerobot/rollout/stall_guard.py`:StallContactGuard 比较策略请求的
  action 与实际下发的 action,连续 `stall_guard_ticks`(默认 5,约 167 ms@30fps)
  个 tick 偏差超过 `stall_guard_tolerance` 即判定堵转/意外接触:先命令保持
  当前位置,再抛 StallContactError 中止。本次事件的 26 连钳制会在第 5 个 tick
  触发;启动瞬态 2 连不会误触。五种 strategy 共用 send_next_action,一处生效。
- 异常退出扭矩保持:run_error 时置 ctx.hardware.abnormal_shutdown = True,
  teardown 跳过回初始位,并临时改写 disable_torque_on_disconnect=False,
  机械臂保位不塌落;操作员先扶住、手动断扭矩后再触碰。KeyboardInterrupt
  仍走正常 teardown 语义。
- 配置与启动链:RolloutConfig 新增 stall_guard_ticks / stall_guard_tolerance
  (__post_init__ 校验);run_pi05_full_rollout_rtc_actual.sh 环境旋钮
  PI05_STALL_GUARD_TICKS/TOLERANCE、显式命令行参数、preflight 内解析断言、
  配置回显 stall_guard_ticks= 行;T1 启动器强制 ticks >= 1;logged 包装器
  分析器新增 --expect-stall-guard true,fail-closed 要求日志出现
  "Stall/contact guard enabled"。
- 测试:tests/test_stall_guard.py 8 项(阈值触发、清洁复位、容差、保位下发、
  异常/正常 teardown 扭矩语义),连同 rollout 与分析器共 58 项通过;
  无硬件 preflight 三门禁全绿。

### 16.3 下次运行前置条件(未满足不得启动)

1. 用户现场检查:腕部相机 USB 连接器与支架(冲击后已重新枚举,需加固/理线),
   各关节齿轮有无异响或损伤。
2. 用户重新逐次授权(上次授权已被失败运行消耗)。
3. 待议(需用户决定):基于遥操作实测桌面高度的工作空间下限/软限位;
   单关节下限无法表达桌面平面,需要物理几何数据。
