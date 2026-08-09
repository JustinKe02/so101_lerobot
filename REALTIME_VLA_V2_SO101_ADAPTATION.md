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
| acados MPC | Optional research backend; local path uses a bounded planner plus second-order filter |
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

## Current Implementation Status

### Deployment status correction

The branch contains code paths for the adapted runtime, but the recommended
August 9 robot configuration does not enable the complete stack. It enables
RTC6 trained-prefix inference, actual-consumed queue alignment, the time-axis
planner, Triton inference, and schema-v2 tracing. Dynamic prefill, measured
sensor timing, checksum-pinned joint constraints, the calibrated fixed-heartbeat
executor, and the trained speed adapter remain disabled or unavailable.

Triton replaces the PI0.5 model inference backend; it does not replace the
SO-101 motor command backend and is not itself a Realtime-VLA V2 algorithmic
feature. The current deployment must therefore be described as a partial
SO-101 adaptation, not as a paper-ready implementation.

The earlier seven-step dynamic-prefill result used uncalibrated delay constants.
It does not justify RTC15 training. Measure the real sensor and actuator timing
first, keep the RTC6 checkpoint if the resulting prefix fits its capacity, and
only retrain to the smallest measured capacity plus margin if it does not.

The first training/inference compatibility slice is now implemented in the
LeRobot runtime:

- `PI05Config.rtc_training_max_delay` defaults to zero for exact backward
  compatibility.
- Positive values sample a clean action prefix and apply postfix-only flow loss
  with per-sample normalization.
- RTC inference exposes `guided` and `trained_prefix` modes.
- `trained_prefix` hard-inpaints the committed prefix after every denoising
  step, rejects checkpoints without positive training capacity, and fails closed
  when measured delay exceeds that capacity.
- `src/lerobot/scripts/train_pi05_so101_realtime_vla_v2_full.sh` starts the
  primary 40-episode full-unfreeze run with `rtc_training_max_delay=6`.
- `src/lerobot/scripts/train_pi05_so101_realtime_vla_v2_expert_only.sh` is kept
  only as an expert-only ablation for controlled comparison.

The local path now includes a solver-independent time-axis planner,
delay-aligned state/action history, and a unified JSONL trace. It does not
pretend to include the reference repository's AIRBOT HTTP transport or an
acados-generated contact MPC, which remain hardware-specific extensions.

The runtime configuration intentionally keeps
`realtime_executor.actuator_calibration.enabled=false` until an offline SO-101
trace has produced a checksum-pinned calibration artifact. In that state the
manually configured actuator constants are compatibility defaults, not SO-101
measurements, and the configuration must not be described as paper-ready.
Enabling calibration requires both an artifact path and its SHA-256; the
runtime validates schema, source provenance, action dimension, and exact joint
order before it can construct the executor.

### Phase 1: Training-Time RTC

- Add `policy.rtc_training_max_delay`, defaulting to zero. (Implemented.)
- Sample a clean prefix length independently for every training sample.
- Keep the committed prefix clean in the flow input and exclude it from the
  normalized flow loss. Per-token flow-time conditioning remains a follow-up
  because the current PI0.5 expert uses one AdaRMS timestep per sample.
- Preserve bit-for-bit behavior when `rtc_training_max_delay=0`.

### Phase 2: Trained-Prefix Inference

- Add an explicit RTC mode: `guided` or `trained_prefix`. (Implemented.)
- Hard-inpaint the committed action prefix at every denoising step in trained
  mode; do not run the guided RTC backward pass in that mode. (Implemented.)
- Reject checkpoints without positive `rtc_training_max_delay`. (Implemented.)
- Fail closed if measured delay exceeds the checkpoint's trained capacity.
  (Implemented.)
- Require queue threshold and execution horizon to cover the trained delay.
- Keep the existing actual-consumed queue diagnostics and prefix-health gates.

### Phase 3: Time-Axis Planning

- Port the reference time-axis objective as a chunk transform, not as a new
  robot backend. (Implemented in `src/lerobot/rollout/time_axis.py`.)
- Configure reference, minimum, and maximum control intervals explicitly.
- Use bounded policy-coordinate velocity limits; robot-coordinate velocity and
  acceleration shaping remains the second-order output filter.
- Preserve chunk endpoints and fall back to the original chunk on solver
  failure or non-finite output.
- Apply the existing per-command safety clamp after planning. (Implemented.)

### Phase 4: Replay and Diagnostics

- Record raw chunk, trained-prefix chunk, time-parameterized chunk, filtered
  command, applied command, measured state, and inference timing in one JSONL
  stream when `trace.enabled=true`.
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

## Offline Speed Labels

Create an annotation template from a schema-v2 realtime trace; the template
contains no generated beta, failure, or include labels:

```bash
lerobot-prepare-speed-throttle \
  --trace outputs/traces/pi05_realtime_vla_v2_full_rtc15.jsonl \
  --export-annotation-template outputs/traces/speed_annotations.template.jsonl
```

The exported rows already use the final annotation schema, but all human
fields are `null`; an untouched template is rejected by conversion.

After a human fills every annotation, create the strict per-segment dataset:

```bash
lerobot-prepare-speed-throttle \
  --trace outputs/traces/pi05_realtime_vla_v2_full_rtc15.jsonl \
  --annotations outputs/traces/speed_annotations.jsonl \
  --output outputs/traces/speed_throttle.jsonl
```

Use `lerobot-prepare-speed-throttle --print-annotation-schema` for the exact
annotation contract. The converter requires complete robot-space action chunks
and writes trace and annotation SHA-256 provenance beside the output.

Train only after reviewing that provenance and selecting explicit deployment
bounds:

```bash
lerobot-train-speed-adapter \
  --data outputs/traces/speed_throttle.jsonl \
  --output-dir outputs/speed_adapter \
  --beta-min 0.5 --beta-max 1.5
```

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
- acados code generation and task-specific contact MPC. The current planner
  has a deterministic bounded fallback and must be tuned against SO-101 joint
  limits before claiming parity with the reference paper.
- Claims of reference-project speedup before SO-101 A/B measurements exist.
