# Realtime-VLA V2 to SO-101 Adaptation

## Scope

This branch adapts the useful deployment ideas from
`dexmal/realtime-vla-v2` to the existing LeRobot PI0.5 and SO-101 runtime.
The reference was reviewed at commit `a36d02a7b241de1129af2048e749de58f95ead9c`.
Its source is MIT licensed.

The target is not a line-for-line copy. The reference runtime assumes an
AIRBOT W1 dual-arm system, three RealSense cameras, 14-dimensional actions,
remote HTTP inference, and optional acados MPC. This repository already has
the appropriate SO-101 hardware abstraction, policy processors, RTC queue,
TensorRT prefix path, safety limits, and rollout lifecycle.

## Reference Architecture

The reference implementation combines four independent mechanisms:

1. Action-prefill inference conditions the next action chunk on a clean prefix
   selected from the action trajectory expected to execute during inference.
2. A server-side time-axis optimizer changes waypoint timing while preserving
   the spatial path and applying speed/acceleration objectives.
3. A client-side delay-aligned queue predicts how far execution advances while
   inference is running, then merges only the still-relevant postfix.
4. A local executor applies either smoothing/forward tracking or an acados MPC
   before dispatching commands at a fixed heartbeat.

The approximately 3x task-speed result reported by the project is an outcome
of the complete stack and task-specific tuning. It is not an inference-latency
claim that can be transferred directly to SO-101.

## LeRobot Mapping

| Reference component | SO-101 implementation target |
| --- | --- |
| `server/model.py` action prefill | PI0.5 training-time RTC and trained-prefix denoising |
| `server/optimizer.py` time-axis QP | Optional chunk-level time-axis planner before dispatch |
| `client/local_client.py` inference thread | Existing `RTCInferenceEngine` background thread |
| `client/executor.py` delay-aligned queue | Existing actual-consumed RTC queue and merge diagnostics |
| AIRBOT observer/actuator | Existing `SO101Follower` and LeRobot camera processors |
| acados MPC | Deferred; use existing second-order output filter first |
| HTTP + pickle transport | Omitted for the local-GPU path; add a structured remote transport only if needed |
| Rerun/JSONL traces | Extend current rollout timing and action diagnostics |

## Target Runtime

```text
SO-101 cameras + joint state
            |
            v
LeRobot policy preprocessor
            |
            v
PI0.5 asynchronous chunk inference
  - guided RTC for ordinary checkpoints
  - trained-prefix RTC for compatible checkpoints
            |
            v
optional time-axis chunk planner
            |
            v
second-order output filter + robot safety limits
            |
            v
SO101Follower command dispatch
```

## Implementation Phases

### Phase 1: Training-Time RTC

- Add `policy.rtc_training_max_delay`, defaulting to zero.
- Sample a clean prefix length independently for every training sample.
- Use per-action flow timesteps: zero for the clean prefix and the sampled
  flow timestep for the postfix.
- Exclude prefix positions from the flow loss.
- Preserve bit-for-bit behavior when `rtc_training_max_delay=0`.

### Phase 2: Trained-Prefix Inference

- Add an explicit RTC mode: `guided` or `trained`.
- Hard-inpaint the committed action prefix at every denoising step in trained
  mode; do not run the guided RTC backward pass in that mode.
- Reject checkpoints without positive `rtc_training_max_delay`.
- Fail closed if measured delay exceeds the checkpoint's trained capacity.
- Require queue threshold and execution horizon to cover the trained delay.
- Keep the existing actual-consumed queue diagnostics and prefix-health gates.

### Phase 3: Time-Axis Planning

- Port the reference time-axis objective as a chunk transform, not as a new
  robot backend.
- Configure reference, minimum, and maximum control intervals explicitly.
- Use per-joint velocity and acceleration limits in SO-101 units.
- Preserve chunk endpoints and fall back to the original chunk on solver
  failure or non-finite output.
- Apply the existing per-command safety clamp after planning.

### Phase 4: Replay and Diagnostics

- Record raw chunk, trained-prefix chunk, time-parameterized chunk, filtered
  command, applied command, measured state, inference start/end, and queue
  indices on one monotonic timeline.
- Add dataset replay tests for boundary position/velocity/acceleration.
- Compare guided RTC, trained-prefix RTC, and VLASH with the same checkpoint,
  observation sequence, and random seed.

### Phase 5: Supervised Robot Evaluation

- Run preflight and offline replay before opening hardware.
- Start with short, supervised rollouts and normal SO-101 safety limits.
- Evaluate task success, p50/p95 inference latency, chunk-boundary jump,
  velocity/acceleration, clamp count, queue underruns, and fatal stops.

## Checkpoint Contract

The existing PI0.5 expert-only checkpoints were trained without action-prefix
conditioning. They may use guided RTC but must not use trained-prefix mode.
A compatible checkpoint must serialize a positive `rtc_training_max_delay` and
must be trained from the start with the Phase 1 objective.

For 30 Hz SO-101 control, a maximum delay of four to six steps covers roughly
133 to 200 ms. The initial experiment should use a conservative value within
that range and must keep it smaller than the 50-step action horizon.

## Acceptance Gates

- Existing PI0.5 tests pass with training-time RTC disabled.
- Unit tests cover clean-prefix construction, postfix-only loss, trained-mode
  configuration validation, prefix capacity, and delay overflow.
- Replay tests contain no NaN/Inf and introduce no queue underrun.
- Time-axis planning never violates configured interval bounds.
- Hardware is not connected during configuration or checkpoint validation.
- A trained-prefix checkpoint is not promoted unless it improves boundary
  continuity without reducing controlled task success.

## Deliberately Deferred

- AIRBOT SDK integration and 14-dimensional dual-arm assumptions.
- Three-camera RealSense capture code.
- Unauthenticated pickle-over-HTTP transport.
- acados code generation and task-specific contact MPC.
- Claims of reference-project speedup before SO-101 A/B measurements exist.
