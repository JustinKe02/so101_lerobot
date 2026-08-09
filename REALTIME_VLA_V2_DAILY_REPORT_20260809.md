# PI0.5 Realtime-VLA V2 training and rollout report (2026-08-09)

## Outcome

The 40-episode PI0.5 RTC6 full-unfreeze run completed successfully and produced the
checkpoint used for today's robot rollouts. The complete Realtime-VLA V2 runtime,
Triton backend, calibration tools, parity gates, trace validator, configs, and tests
were committed as `6c3860cc` on `realtime-vla-v2`.

Triton is the preferred deployment backend. It preserved approximately the same
observed task behavior as PyTorch while reducing mean full-model inference latency
from `136-138 ms` to `45-49 ms`. Disabling the time-axis planner or changing fixed
prefix length 5 to rolling-P95 did not produce an obvious task-quality improvement
in the supervised runs. The remaining quality gap is therefore more likely to be
model/data or physical actuator tracking than the inference backend.

## Training

| Run | Result | Progress | Time | Final loss | Last-10 mean loss | Peak GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTC6 full unfreeze | Completed | 5613 steps, 10.00 epochs | 7 h 50 min | 0.009 | 0.0091 | 39.55 GB |
| RTC15 full unfreeze | Stopped intentionally | 596/5613 steps, 1.05 epochs | 54 min 40 s | 0.041 at step 590 | 0.0473 | 39.55 GB |

RTC6 configuration:

- Dataset: `admin123/so101_test_data`, 40 episodes, no evaluation split.
- Full unfreeze: `freeze_vision_encoder=false`, `train_expert_only=false`.
- Training RTC capacity: `rtc_training_max_delay=6`.
- Batch size 32, BF16, gradient checkpointing, seed 1000.
- Approximately 180K samples were processed at 4.96 seconds/update.
- Checkpoints were saved at steps 2806, 5612, and final step 5613.
- Final checkpoint: `outputs/train/pi05_so101_realtime_vla_v2_full_10epochs_rtc6_seed1000/checkpoints/005613/pretrained_model`.

RTC15 was an exploratory capacity-extension run. It was stopped by the user to
release GPU memory for RTC6 validation before reaching the first checkpoint, so it
did not replace the completed RTC6 checkpoint.

## Export and parity

The RTC6 checkpoint was exported to the static BF16 Triton layout. The export SHA-256
is `8bb1dcd1e44e035c8a473863a0633cbc8cc63d4946d9a5d41e226080bf7dc088`.

Fixed-noise parity passed for every trained prefix length from 0 through 6. The worst
reported normalized max absolute error was `0.04940`; at prefix length 5 it was
`0.02236` with mean absolute error `0.00586`.

## Robot rollout comparison

All latency values below cover complete image encoder, VLM, action expert, and ten
denoising steps. Completed sessions ended through the normal cleanup path.

| Variant | Sessions | Mean latency | P95 latency | Max latency | Queue minimum | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Triton + uncalibrated Executor + rolling-P95 | 1 | 47.64 ms | 58.36 ms | 84.29 ms | 48 | Completed; executor altered model commands |
| Triton direct + planner + fixed prefix 5 | 1 | 47.51 ms | 57.77 ms | 85.08 ms | 48 | Completed; model command was sent directly |
| Triton direct + no planner + fixed prefix 5 | 1 | 45.15 ms | 57.94 ms | 88.29 ms | 48 | Completed; task effect similar, higher peak dynamics |
| PyTorch direct + no planner + fixed prefix 5 | 2 | 136-138 ms | 144-145 ms | 397-418 ms | 45 | Completed; task effect similar, approximately 3x slower |
| Triton direct + planner + rolling-P95 | 1 | 48.65 ms | 57.85 ms | 73.65 ms | 48 | Completed; task effect similar |

The direct variants produced zero difference between the selected model action and
the dispatched command. PyTorch consumed mostly 3-4 control actions during each
inference; Triton normally consumed 1-2, leaving more feedback and queue headroom.

Three early integration sessions ended abnormally and are retained in the report:

1. The uncalibrated executor repeatedly rewrote goals after the arm failed to track
   shoulder/elbow targets, triggering the consecutive safety-clamp stop.
2. Full dynamic prefill required 7 steps, exceeding the RTC6 checkpoint capacity.
3. A sub-microsecond floating-point scheduler regression was rejected as backwards
   time. Later sessions used the corrected scheduler path.

## Interpretation

- Triton is a faithful acceleration path, not a different trained model. It uses the
  same RTC6 weights, 50-action chunk, and ten Euler denoising steps.
- Time-axis planning was not the primary source of the observed task-quality gap.
- The uncalibrated Realtime Executor should remain disabled for current robot use.
- Fixed prefix 5 and rolling-P95 produced similar supervised task behavior. More
  repeated trials with task-success labels are required before claiming a success-rate
  difference.
- Trace acceptance reports show `overall_pass=false` because each rollout records one
  startup empty-queue event and because paper-level calibration/speed-adapter features
  are intentionally disabled. This is not a runtime exception; selected sessions have
  `terminal.status=completed`.

## Recommended current configuration

Use `pi05_realtime_vla_v2_40ep_rtc6_triton_direct_rolling_p95.json`. It keeps the
completed RTC6 weights and Triton backend, disables the uncalibrated executor and
action filter, keeps the time-axis planner, and adapts prefix length to rolling P95
latency. The configuration intentionally retains the previously requested unlimited
relative target setting (`max_relative_target=null`); supervised operation and an
accessible emergency stop remain required.

## Archived reports

Compact, reviewable logs are under `reports/realtime_vla_v2/2026-08-09/`. Raw traces,
checkpoints, optimizer state, and the multi-gigabyte Triton export remain under local
`outputs/` and are identified by SHA-256 in the manifest rather than committed to Git.
