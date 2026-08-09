# SO-101 40-Episode Realtime-VLA V2 Inference Plan

## 1. Goal and Current Assets

This plan targets the local SO-101 dataset and the task `Put the block in the bin`.

| Item | Value |
| --- | --- |
| Dataset | `data/so101_test_data` |
| Episodes | 40 |
| Frames | 17,960, 449 frames per episode |
| Frequency | 30 FPS |
| Observation | 6-D joint state, `top` and `wrist` RGB cameras |
| Action | 6-D joint target |
| Current checkpoint | `outputs/train/pi05_so101_expert_only_10epochs_5epoch_ckpt_seed1000/checkpoints/005613/pretrained_model` |
| Training type | PI0.5 expert-only, 10 epochs |

All 40 episodes were used for training. Dataset replay is therefore a numerical
and continuity check, not an unbiased measurement of task generalization.

The current checkpoint has no training-time RTC/action-prefix conditioning.
It must use guided RTC. It must not be described or run as a trained-prefix
Realtime-VLA V2 checkpoint.

## 2. Phase A: Immediately Available Guided RTC Baseline

The checked-in baseline is `pi05_realtime_vla_v2_40ep_guided.json`.

| Parameter | Initial value | Reason |
| --- | --- | --- |
| Timing | `actual_consumed` | Align replacement chunks to commands actually consumed during inference |
| Guidance delay | fixed 5 steps | Matches the observed steady 147-157 ms latency at 30 Hz |
| Execution horizon | 20 steps | Covers the observed 19-step cold-start outlier without disabling the guard |
| Queue threshold | 20 | Preserve the previously used stable queue cadence for the first comparison |
| Action filter | enabled | Limit command velocity and acceleration at chunk boundaries |
| Relative target limit | 6.0 | Keep the existing robot-side per-command safety clamp |
| Prefix health | stop after 2 severe replans | Abort repeated high-residual guided-prefix failures |
| Backend | PyTorch | Establish behavior before adding TensorRT as another variable |
| Duration | 30 seconds | Short supervised trial; never use an unattended first rollout |

`enforce_guided_execution_window` remains disabled for this Q20 baseline because
the nominal replan interval is 30 steps while the guided horizon is 20. The
runtime still rejects a measured inference delay of 20 steps or more. If that
happens, optimize inference or raise the horizon only after offline replay; do
not remove the bound.

### 2.1 Offline Replay

Start with three representative episode midpoints (episodes 0, 20, and 39):

```bash
cd /data/cqy_workspace/tk/lerobot_src
export PYTHONPATH="$PWD/src:$PWD"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

/home/cqy/miniconda3/envs/lerobot_tk/bin/python \
  examples/inference/evaluate_pi05_rtc_replay.py \
  --model outputs/train/pi05_so101_expert_only_10epochs_5epoch_ckpt_seed1000/checkpoints/005613/pretrained_model \
  --dataset-root data/so101_test_data \
  --repo-id admin123/so101_test_data \
  --absolute-indices 224 9204 17735 \
  --seeds 0 42 1000 \
  --queue-thresholds 20 45 \
  --guidance-delay 5 \
  --actual-consumed-steps 4 5 6 19 \
  --execution-horizon 20 \
  --fps 30 \
  --max-relative-target 6.0 \
  --action-filter \
  --output-json outputs/eval/pi05_expert_only_40ep_rtc_smoke.json
```

Then cover the midpoint of every episode. Since each episode has exactly 449
frames, the following sequence produces all 40 midpoint indices:

```bash
/home/cqy/miniconda3/envs/lerobot_tk/bin/python \
  examples/inference/evaluate_pi05_rtc_replay.py \
  --model outputs/train/pi05_so101_expert_only_10epochs_5epoch_ckpt_seed1000/checkpoints/005613/pretrained_model \
  --dataset-root data/so101_test_data \
  --repo-id admin123/so101_test_data \
  --absolute-indices $(seq 224 449 17735) \
  --seeds 0 42 1000 \
  --queue-thresholds 20 45 \
  --guidance-delay 5 \
  --actual-consumed-steps 4 5 6 19 \
  --execution-horizon 20 \
  --fps 30 \
  --max-relative-target 6.0 \
  --action-filter \
  --output-json outputs/eval/pi05_expert_only_40ep_rtc_all.json
```

The existing replay evaluator reports its top-level deployment gate for Q45.
For this Q20 baseline, inspect the `q20_g5_a*` groups as well as the top-level
Q45 result. Do not promote the configuration if either candidate has NaN/Inf,
queue underflow, repeated clamp events, or a failed splice/command-delta gate.

### 2.2 Supervised Robot Rollout

Preconditions: the arm workspace is clear, the emergency stop is reachable,
the calibration file and both camera device paths match the connected hardware,
and an operator remains beside the robot for the entire run.

```bash
cd /data/cqy_workspace/tk/lerobot_src
export PYTHONPATH="$PWD/src"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}

/home/cqy/miniconda3/envs/lerobot_tk/bin/python -m lerobot.scripts.lerobot_rollout \
  --config_path=/data/cqy_workspace/tk/lerobot_src/pi05_realtime_vla_v2_40ep_guided.json
```

Run five trials with identical initial arm and block placement. Record task
success, p50/p95/max inference latency, `actual_consumed_steps`, prefix residual,
chunk-boundary command delta, safety clamp count, stall-stop count, and queue
underflow. Stop immediately on repeated clamp warnings, a severe prefix-health
stop, unexpected contact, camera loss, or an inference fatal state.

The Phase A candidate passes only if it does not reduce success relative to the
existing RTC baseline and improves continuity without increasing safety events.
Lower latency alone is not sufficient because the previously observed problem
was reduced latency accompanied by worse action continuity.

## 3. Controlled A/B Matrix

Use the same scene, seed, duration, safety settings, and five trials per cell.

| Candidate | Checkpoint | Runtime | Purpose |
| --- | --- | --- | --- |
| A0 | Existing full PI0.5 | Existing RTC | Current on-robot reference |
| A1 | Expert-only `005613` | Guided RTC Q20/H20 + filter | Immediate 40-episode baseline |
| A2 | Expert-only `005613` | Guided RTC Q45/H20 + filter | Test whether more frequent replanning helps reactivity |
| B1 | RTC-trained expert-only | Trained-prefix runtime | Isolate the value of action-prefix training |
| B2 | RTC-trained full model | Trained-prefix runtime | Test whether full unfreezing adds visual/language adaptation |

Do not compare candidates from different random seeds or initial object
placements as if the difference came only from the runtime.

## 4. Phase B: Training-Compatible Realtime-VLA V2

Phase B is now implemented in the PI0.5 policy/runtime. The primary training
run is full-unfreeze and starts with
`src/lerobot/scripts/train_pi05_so101_realtime_vla_v2_full.sh`:

1. The script sets `rtc_training_max_delay=6` in PI0.5 configuration and
   checkpoint metadata.
2. It samples a clean prefix of 0-6 actions for each training sample.
3. It keeps the clean prefix in the flow input and computes normalized loss only
   on the postfix. Postfix tokens receive sampled flow time while clean prefix
   tokens receive time zero; AdaRMS carries that per-token time signal through
   every action-expert layer.
4. It sets both `train_expert_only=false` and `freeze_vision_encoder=false`, so
   the VLM, vision encoder, projections, and action expert are all trainable.
5. The expert-only script remains available only as a controlled ablation with
   the same data order, seed, epochs, and checkpoint cadence.
6. The runtime rejects trained-prefix inference for any checkpoint without a positive serialized
   `rtc_training_max_delay`.

Start with `rtc_training_max_delay=6` because it covers the normal 4-5 consumed
steps plus margin. The 19-step cold-start case must be handled by warmup and the
runtime queue; it should not dictate the training prefix distribution.

After a compatible checkpoint exists, select `--inference.mode=trained_prefix`;
the runtime hard-inpaints the prefix at every denoising step and rejects delay
overflow. The paper-style local stack is configured in
`pi05_realtime_vla_v2_40ep_trained_prefix.json`: it enables the bounded
time-axis planner, second-order action filter, delayed trajectory history, and
JSONL trace. Keep the planner enabled only after offline endpoint/bounds checks
pass; it falls back to the raw model chunk on numerical failure.

Start the training run only after checking that the output directory does not
already exist:

```bash
cd /data/cqy_workspace/tk/lerobot_src
bash src/lerobot/scripts/train_pi05_so101_realtime_vla_v2_full.sh
```

The expert-only ablation is optional and must use its separate launcher and
output directory; it is not a prerequisite for the full-unfreeze run.

## 5. Decision Rule

Do not start another full-unfreeze VLASH/Realtime-VLA V2 training run solely
because the expert-only rollout looks similar to action-head fine-tuning. Full
training is justified when the controlled results show a perception or task
grounding error that action-prefix training and runtime smoothing do not fix.

Conversely, if A1/A2 choose the right motion but show chunk-boundary discontinuity,
the next useful experiment is B1 training-time RTC, not broader unfreezing. If B1
is continuous but still selects the wrong grasp/placement behavior, B2 full
unfreezing is the appropriate next experiment.
