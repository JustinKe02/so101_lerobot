# PI0.5 Realtime-VLA V2 Triton rollout template

The checked-in machine rollout JSON files intentionally keep
`"pi05_action_backend": "pytorch"`. Do not select the Triton backend until the
export below has completed and its SHA-256 has been recorded.

## 1. Export the trained checkpoint

```bash
cd /data/cqy_workspace/tk/lerobot_src

CHECKPOINT="$PWD/outputs/train/pi05_so101_realtime_vla_v2_full_10epochs_rtc6_seed1000/checkpoints/005613/pretrained_model"
TRITON_WEIGHTS="$PWD/outputs/realtime_vla_v2/pi05_rtc6_triton.pkl"
TOKENIZER="google/paligemma-3b-pt-224"

lerobot-export-pi05-realtime-vla-v2 \
  --checkpoint "$CHECKPOINT" \
  --output "$TRITON_WEIGHTS" \
  --prompt "Put the block in the bin" \
  --tokenizer-path "$TOKENIZER"

TRITON_SHA256="$(sha256sum "$TRITON_WEIGHTS" | awk '{print $1}')"
printf '%s\n' "$TRITON_SHA256"
```

## 2. Copy and edit the full V2 rollout config

Start from `pi05_realtime_vla_v2_40ep_rtc6_triton_preflight.json`. It pins the
validated RTC6 export and keeps unmeasured calibration and learned-speed
features disabled. Do not enable them without checksum-pinned real artifacts.

```json
{
  "pi05_prefix_backend": "pytorch",
  "pi05_action_backend": "triton",
  "pi05_triton_export_weights": "REPLACE_WITH_TRITON_WEIGHTS_ABSOLUTE_PATH",
  "pi05_triton_weights_sha256": "REPLACE_WITH_TRITON_SHA256",
  "pi05_triton_model_config": "REPLACE_WITH_CHECKPOINT_CONFIG_JSON_ABSOLUTE_PATH",
  "pi05_triton_tokenizer_path": "google/paligemma-3b-pt-224",
  "pi05_triton_camera_keys": [
    "observation.images.top",
    "observation.images.wrist"
  ],
  "pi05_triton_prompt_capacity": 64,
  "pi05_triton_tokenizer_max_length": 200,
  "pi05_triton_min_free_cuda_gib": 18.0
}
```

The rollout config rejects missing files, a missing or malformed SHA-256,
TensorRT mixing, guided/legacy RTC, wrong camera order, and non-CUDA devices.
The runtime verifies the full export hash and warms every trained prefix length
before the robot object is constructed or connected.

## 3. Run serial fixed-noise parity first

```bash
PARITY_DIR="$PWD/outputs/realtime_vla_v2/parity"
mkdir -p "$PARITY_DIR"

lerobot-pi05-realtime-vla-v2-parity make-request \
  --output "$PARITY_DIR/request.npz" \
  --prompt "Put the block in the bin" \
  --prefill-length 6

lerobot-pi05-realtime-vla-v2-parity run-pytorch \
  --checkpoint "$CHECKPOINT" \
  --request "$PARITY_DIR/request.npz" \
  --output "$PARITY_DIR/pytorch.npz" \
  --tokenizer-path "$TOKENIZER"

# Run only after the PyTorch command exits and releases its model memory.
lerobot-pi05-realtime-vla-v2-parity run-triton \
  --weights "$TRITON_WEIGHTS" \
  --weights-sha256 "$TRITON_SHA256" \
  --model-config "$CHECKPOINT/config.json" \
  --request "$PARITY_DIR/request.npz" \
  --output "$PARITY_DIR/triton.npz" \
  --tokenizer-path "$TOKENIZER"

lerobot-pi05-realtime-vla-v2-parity compare \
  --pytorch-output "$PARITY_DIR/pytorch.npz" \
  --triton-output "$PARITY_DIR/triton.npz"
```

Each GPU phase performs a free-memory preflight. It fails without loading a
model while another training or inference process leaves less than the
configured free-memory threshold.
