# SO-101 ACT Inference Log Report

This report summarizes two local ACT inference runs performed on 2026-08-09 with the SO-101 follower arm.
It focuses on control-loop timing and action-chunk behavior. The runs did not record task success metrics, so
the results do not establish whether policy accuracy or task success improved.

## Shared setup

- Policy checkpoint: `outputs/train/act_so101_test_data_20k/checkpoints/020000/pretrained_model`
- Policy: ACT, approximately 51.6 million parameters
- Inputs: `observation.state`, `observation.images.top`, and `observation.images.wrist`
- Cameras: two OpenCV cameras at 640 x 480 and 30 FPS
- Robot: SO-101 follower, six action dimensions
- Inference backend: synchronous (`inference.type=sync`)
- Target control frequency: 30 FPS
- Runtime: infinite (`duration=0`) until interrupted with `Ctrl+C`
- Action interpolation: disabled (`interpolation_multiplier=1`)
- Relative target safety clamp: disabled (`max_relative_target` was not configured)
- Device: NVIDIA RTX 4090 with CUDA

The checkpoint uses `chunk_size=100`, has temporal ensembling disabled, and was trained with
`n_action_steps=100`. It predicts a complete 100-action chunk whenever its internal action queue is empty.

## Run 1: forced replan every 10 actions

The first run added the following synchronous inference setting:

```text
inference.max_actions_per_chunk=10
```

The policy still generated 100 actions per model call, but the synchronous inference engine reset the policy
queue after executing 10 actions. At 30 FPS, this forced a new model call approximately every 0.33 seconds and
discarded about 90% of each predicted chunk.

Relevant shutdown statistics:

```text
SyncInferenceEngine stopped (horizon_replans=317, clamp_replans=0)
```

Observed slow control-loop samples included:

```text
29.1 Hz
21.3 Hz
20.4 Hz
7.4 Hz
```

The repeated `Sync guarded replan: execution prefix reached 10 actions` messages occurred about three times
per second. The resulting behavior combined two sources of visible interruption:

1. Inline model inference blocked the control thread whenever the queue was reset.
2. The first action of each new chunk was not blended with the last action of the previous chunk.

The run completed cleanly after `Ctrl+C`, returned the arm to its initial position, disconnected both cameras,
and disabled follower torque during disconnection. No clamp-triggered replan occurred.

## Run 2: natural 50-action queue

The second run removed `inference.max_actions_per_chunk` and overrode the policy execution horizon directly:

```text
policy.n_action_steps=50
```

This allowed the ACT queue to empty naturally instead of forcing `policy.reset()`. A new action chunk was
generated approximately every 1.67 seconds.

The initialization log confirmed the intended configuration:

```text
SyncInferenceEngine initialized (
    device=cuda,
    action_keys=6,
    max_actions_per_chunk=None,
    replan_on_clamp=False
)
```

The run lasted approximately 126 seconds. Shutdown statistics were:

```text
SyncInferenceEngine stopped (horizon_replans=0, clamp_replans=0)
```

There were 13 logged iterations below the 30 Hz target, approximately 0.34% of the expected control ticks.
The slow samples were:

```text
5.5 Hz   # first control iteration
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

The first 5.5 Hz iteration is consistent with one-time CUDA and convolution warmup. Most later iterations met
the 33.3 ms control budget. The remaining slow iterations were isolated latency spikes rather than a sustained
control-rate collapse.

This run also completed cleanly after `Ctrl+C`, returned to the initial pose, disconnected both cameras, and
reported no camera, serial, CUDA, or clamp-replan errors.

## Offline model timing

The ACT model was also benchmarked without robot or camera I/O, using tensors already resident on the GPU.

| Mode | Median | Observed p95 | Output dtype |
| --- | ---: | ---: | --- |
| FP32 | 4.76 ms | 4.77 ms | `torch.float32` |
| CUDA autocast | 4.17 ms | 4.24 ms | `torch.float16` |

Autocast reduced the model-only latency by approximately 0.6 ms. This was too small to explain or resolve the
75-180 ms control-loop spikes, and it changed the inference output dtype. The second run therefore retained the
checkpoint's FP32 inference behavior.

## Findings

The forced 10-action replan in Run 1 was a configuration-level source of periodic interruption. Run 2 removed
that behavior successfully: `max_actions_per_chunk=None`, `horizon_replans=0`, and no repeated guarded-replan
messages.

The remaining latency cannot be attributed solely to ACT model execution. The model-only benchmark is below
5 ms, while a 30 Hz loop has a 33.3 ms budget. The real loop also performs synchronous joint-state reads,
retrieves two camera frames, converts both images from uint8 to float32, makes contiguous channel-first copies,
transfers the images to CUDA, postprocesses actions, and sends commands over the serial motor bus. The current
INFO-level warning reports only aggregate loop time, so it cannot identify which component caused each isolated
spike.

Motion discontinuity is separate from timing delay. The checkpoint has temporal ensembling disabled and the
rollout used no action interpolation. A newly predicted 50-action chunk is therefore not guaranteed to begin
continuously from the final command of the previous chunk. Removing `max_relative_target` allows any such jump
to reach the motor command path without relative-motion clipping.

## Current baseline and next measurements

Run 2 is the better current ACT deployment baseline:

```text
policy.n_action_steps=50
inference.max_actions_per_chunk=None
interpolation_multiplier=1
fps=30
duration=0
```

Further work should separate control-loop latency from trajectory continuity:

1. Add component-level timing for joint reads, each camera read, preprocessing, model execution, postprocessing,
   and serial action writes.
2. Prefetch the next ACT action chunk on a background worker before the current queue is exhausted.
3. Blend the boundary between consecutive action chunks without changing the demonstrated 30 Hz action timing.
4. Record episode-level success, failure reason, completion time, and visible pauses; timing logs alone cannot
   evaluate policy quality.
