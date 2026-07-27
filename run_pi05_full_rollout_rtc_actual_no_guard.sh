#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
CALIB_ROOT=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
CALIB_DIR=$CALIB_ROOT/robots/so_follower
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
TOP_CAMERA=/dev/video4
WRIST_CAMERA=/dev/video6

DURATION=${PI05_DURATION:-30}
SEED=${PI05_SEED:-1000}
PREFLIGHT_ONLY=${PI05_PREFLIGHT_ONLY:-0}
RETURN_TO_INITIAL_POSITION=${PI05_RETURN_TO_INITIAL_POSITION:-false}
PREFIX_HEALTH_ENABLED=${PI05_PREFIX_HEALTH_ENABLED:-false}
ENFORCE_GUIDED_EXECUTION_WINDOW=${PI05_ENFORCE_GUIDED_EXECUTION_WINDOW:-false}
PREFIX_HEALTH_SEVERE_RESIDUAL_THRESHOLD=${PI05_PREFIX_HEALTH_SEVERE_RESIDUAL_THRESHOLD:-5.0}
PREFIX_HEALTH_CONSECUTIVE_SEVERE=${PI05_PREFIX_HEALTH_CONSECUTIVE_SEVERE:-3}
PREFIX_HEALTH_SAFETY_STOP_REPLANS=${PI05_PREFIX_HEALTH_SAFETY_STOP_REPLANS:-0}
PREFIX_BACKEND=${PI05_PREFIX_BACKEND:-pytorch}
TRT_PREFIX_ENGINE=${PI05_TRT_PREFIX_ENGINE:-}
ACTION_FILTER_ENABLED=${PI05_ACTION_FILTER_ENABLED:-false}
STALL_GUARD_TICKS=0  # Force disable guard to replicate T0a conditions
STALL_GUARD_TOLERANCE=${PI05_STALL_GUARD_TOLERANCE:-0.001}

FPS=30
CHUNK_SIZE=50
NUM_INFERENCE_STEPS=10
QUEUE_THRESHOLD=${PI05_QUEUE_THRESHOLD:-20}
EXECUTION_HORIZON=10
MAX_GUIDANCE_WEIGHT=10.0
FIXED_GUIDANCE_DELAY_STEPS=5
MAX_RELATIVE_TARGET=5.0

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

normalize_bool() {
  case "${1,,}" in
    1 | true) echo true ;;
    0 | false) echo false ;;
    *) fail "Expected a boolean (0/1/false/true), got: $1" ;;
  esac
}

[[ -x "$PY" ]] || fail "Python executable not found: $PY"
[[ -f "$MODEL/config.json" ]] || fail "Policy config not found: $MODEL/config.json"
[[ -f "$MODEL/model.safetensors" ]] || fail "Full model weights not found: $MODEL/model.safetensors"
[[ -f "$MODEL/policy_preprocessor.json" ]] || fail "Policy preprocessor not found: $MODEL"
[[ -f "$MODEL/policy_postprocessor.json" ]] || fail "Policy postprocessor not found: $MODEL"
[[ -f "$CALIB_DIR/tk_follower.json" ]] || fail "Calibration not found: $CALIB_DIR/tk_follower.json"
[[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "PI05_DURATION must be a non-negative number"
[[ "$SEED" =~ ^-?[0-9]+$ ]] || fail "PI05_SEED must be an integer"
PREFLIGHT_ONLY=$(normalize_bool "$PREFLIGHT_ONLY")
RETURN_TO_INITIAL_POSITION=$(normalize_bool "$RETURN_TO_INITIAL_POSITION")
PREFIX_HEALTH_ENABLED=$(normalize_bool "$PREFIX_HEALTH_ENABLED")
ENFORCE_GUIDED_EXECUTION_WINDOW=$(normalize_bool "$ENFORCE_GUIDED_EXECUTION_WINDOW")
ACTION_FILTER_ENABLED=$(normalize_bool "$ACTION_FILTER_ENABLED")
[[ "$QUEUE_THRESHOLD" =~ ^[0-9]+$ ]] || fail "PI05_QUEUE_THRESHOLD must be a non-negative integer"
[[ "$STALL_GUARD_TICKS" =~ ^[0-9]+$ ]] || fail "PI05_STALL_GUARD_TICKS must be a non-negative integer"
[[ "$STALL_GUARD_TOLERANCE" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "PI05_STALL_GUARD_TOLERANCE must be a non-negative number"

case "$PREFIX_BACKEND" in
  pytorch | tensorrt) ;;
  *) fail "PI05_PREFIX_BACKEND must be 'pytorch' or 'tensorrt', got: $PREFIX_BACKEND" ;;
esac
if [[ "$PREFIX_BACKEND" == tensorrt ]]; then
  [[ -n "$TRT_PREFIX_ENGINE" ]] || fail "PI05_PREFIX_BACKEND=tensorrt requires PI05_TRT_PREFIX_ENGINE"
  [[ -f "$TRT_PREFIX_ENGINE" ]] || fail "TensorRT prefix engine not found: $TRT_PREFIX_ENGINE"
  TRT_VERIFIED_MARKER="$TRT_PREFIX_ENGINE.verified.json"
  [[ -f "$TRT_VERIFIED_MARKER" ]] || fail "TensorRT engine verification marker not found: $TRT_VERIFIED_MARKER"
  "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' \
    "$TRT_VERIFIED_MARKER" \
    || fail "TensorRT engine verification marker does not record a passing verification: $TRT_VERIFIED_MARKER"
elif [[ -n "$TRT_PREFIX_ENGINE" ]]; then
  fail "PI05_TRT_PREFIX_ENGINE is set but PI05_PREFIX_BACKEND is not 'tensorrt'"
fi

CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"backend":200},"wrist":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"backend":200}}'

export CUDA_VISIBLE_DEVICES=0
if [[ "$PREFIX_BACKEND" == tensorrt ]]; then
  # TensorRT deployment recipe from docs/pi05_tensorrt.md: site-packages overlay
  # (tensorrt bindings, onnx, ml_dtypes) + TensorRT runtime libraries.
  TRT_PYTHON_OVERLAY=$ROOT/.pi05_tensorrt/python
  TRT_RUNTIME_ROOT=/data/cqy_workspace/third_party/tensorrt_10_13_0_35
  [[ -d "$TRT_PYTHON_OVERLAY" ]] || fail "TensorRT python overlay not found: $TRT_PYTHON_OVERLAY"
  [[ -d "$TRT_RUNTIME_ROOT/tensorrt_libs" ]] || fail "TensorRT runtime libs not found: $TRT_RUNTIME_ROOT/tensorrt_libs"
  export PYTHONPATH="$TRT_PYTHON_OVERLAY:$TRT_RUNTIME_ROOT:$ROOT/src"
  export LD_LIBRARY_PATH="$TRT_RUNTIME_ROOT/tensorrt_libs:/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}"
else
  export PYTHONPATH="$ROOT/src"
  export LD_LIBRARY_PATH="/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION="$CALIB_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROLLOUT_ARGS=(
  --strategy.type=base
  --inference.type=rtc
  --inference.timing_mode=actual_consumed
  --inference.guidance_delay_mode=fixed
  --inference.fixed_guidance_delay_steps="$FIXED_GUIDANCE_DELAY_STEPS"
  --inference.timing_diagnostics=true
  --inference.enforce_guided_execution_window="$ENFORCE_GUIDED_EXECUTION_WINDOW"
  --inference.prefix_health_enabled="$PREFIX_HEALTH_ENABLED"
  --inference.prefix_health_severe_residual_threshold="$PREFIX_HEALTH_SEVERE_RESIDUAL_THRESHOLD"
  --inference.prefix_health_consecutive_severe="$PREFIX_HEALTH_CONSECUTIVE_SEVERE"
  --inference.prefix_health_safety_stop_replans="$PREFIX_HEALTH_SAFETY_STOP_REPLANS"
  --inference.rtc.execution_horizon="$EXECUTION_HORIZON"
  --inference.rtc.max_guidance_weight="$MAX_GUIDANCE_WEIGHT"
  --inference.rtc.prefix_attention_schedule=EXP
  --inference.queue_threshold="$QUEUE_THRESHOLD"
  --interpolation_multiplier=1
  --action_filter.enabled="$ACTION_FILTER_ENABLED"
  --stall_guard_ticks="$STALL_GUARD_TICKS"
  --stall_guard_tolerance="$STALL_GUARD_TOLERANCE"
  --pi05_prefix_backend="$PREFIX_BACKEND"
  --pi05_action_backend=pytorch
  --policy.path="$MODEL"
  --policy.gradient_checkpointing=false
  --policy.num_inference_steps="$NUM_INFERENCE_STEPS"
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id=tk_follower
  --robot.calibration_dir="$CALIB_DIR"
  --robot.max_relative_target="$MAX_RELATIVE_TARGET"
  --robot.cameras="$CAMERAS"
  --task="Put the block in the bin"
  --duration="$DURATION"
  --fps="$FPS"
  --seed="$SEED"
  --device=cuda
  --display_data=false
  --play_sounds=false
  --return_to_initial_position="$RETURN_TO_INITIAL_POSITION"
)
if [[ "$PREFIX_BACKEND" == tensorrt ]]; then
  ROLLOUT_ARGS+=(--pi05_tensorrt_prefix_engine="$TRT_PREFIX_ENGINE")
fi

validate_t0_config() {
  "$PY" - "$MODEL/config.json" "$CHUNK_SIZE" "$NUM_INFERENCE_STEPS" "$QUEUE_THRESHOLD" "$ENFORCE_GUIDED_EXECUTION_WINDOW" "$PREFIX_BACKEND" "$ACTION_FILTER_ENABLED" "$TRT_PREFIX_ENGINE" -- "${ROLLOUT_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.configs import RTCAttentionSchedule
from lerobot.configs import parser as lerobot_parser
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout import RolloutConfig
from lerobot.rollout.inference.factory import RTCInferenceConfig
from lerobot.scripts import lerobot_rollout as _rollout_registrations  # noqa: F401

config_path = Path(sys.argv[1])
expected_chunk_size = int(sys.argv[2])
expected_inference_steps = int(sys.argv[3])
expected_queue_threshold = int(sys.argv[4])
expected_guided_window_enforcement = sys.argv[5] == "true"
expected_prefix_backend = sys.argv[6]
expected_action_filter = sys.argv[7] == "true"
expected_trt_engine = sys.argv[8] or None
separator = sys.argv.index("--")
rollout_args = sys.argv[separator + 1 :]
policy = json.loads(config_path.read_text())

expected_policy = {
    "type": "pi05",
    "chunk_size": expected_chunk_size,
    "n_action_steps": expected_chunk_size,
    "num_inference_steps": expected_inference_steps,
    "use_peft": False,
    "train_expert_only": False,
}
for key, expected in expected_policy.items():
    actual = policy.get(key)
    if actual != expected:
        raise SystemExit(f"Unexpected policy config {key}: expected {expected!r}, got {actual!r}")

rtc = RTCInferenceConfig(
    rtc=RTCConfig(
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    ),
    queue_threshold=expected_queue_threshold,
    timing_mode="actual_consumed",
    guidance_delay_mode="fixed",
    fixed_guidance_delay_steps=5,
    timing_diagnostics=True,
    enforce_guided_execution_window=expected_guided_window_enforcement,
    prefix_health_enabled=False,
)
if rtc.fixed_guidance_delay_steps >= rtc.rtc.execution_horizon:
    raise SystemExit("Fixed guidance delay must remain below execution horizon")
if expected_guided_window_enforcement:
    replan_interval = expected_chunk_size - expected_queue_threshold
    if replan_interval + rtc.fixed_guidance_delay_steps > rtc.rtc.execution_horizon:
        raise SystemExit(
            "Guided execution window does not cover the nominal dispatched chunk window: "
            f"replan_interval={replan_interval}, fixed_delay={rtc.fixed_guidance_delay_steps}, "
            f"execution_horizon={rtc.rtc.execution_horizon}"
        )


def capture_config(config: RolloutConfig) -> RolloutConfig:
    return config


original_argv = sys.argv
try:
    sys.argv = ["t0-preflight", *rollout_args]
    resolved = lerobot_parser.wrap()(capture_config)()
finally:
    sys.argv = original_argv

return_to_initial_arg = next(
    arg for arg in rollout_args if arg.startswith("--return_to_initial_position=")
)
expected_return_to_initial = return_to_initial_arg.split("=", 1)[1] == "true"
def arg_value(name: str) -> str:
    prefix = f"--{name}="
    return next(arg for arg in rollout_args if arg.startswith(prefix)).split("=", 1)[1]


expected_prefix_health = arg_value("inference.prefix_health_enabled") == "true"
expected_prefix_threshold = float(arg_value("inference.prefix_health_severe_residual_threshold"))
expected_prefix_consecutive = int(arg_value("inference.prefix_health_consecutive_severe"))
expected_prefix_stop_replans = int(arg_value("inference.prefix_health_safety_stop_replans"))
expected_seed = int(arg_value("seed"))
resolved_checks = {
    "policy path": (str(resolved.policy.pretrained_path), str(config_path.parent)),
    "policy type": (resolved.policy.type, "pi05"),
    "chunk size": (resolved.policy.chunk_size, 50),
    "denoise steps": (resolved.policy.num_inference_steps, 10),
    "timing mode": (resolved.inference.timing_mode.value, "actual_consumed"),
    "guidance mode": (resolved.inference.guidance_delay_mode.value, "fixed"),
    "fixed guidance delay": (resolved.inference.fixed_guidance_delay_steps, 5),
    "execution horizon": (resolved.inference.rtc.execution_horizon, 10),
    "max guidance weight": (resolved.inference.rtc.max_guidance_weight, 10.0),
    "prefix attention schedule": (
        resolved.inference.rtc.prefix_attention_schedule,
        RTCAttentionSchedule.EXP,
    ),
    "queue threshold": (resolved.inference.queue_threshold, expected_queue_threshold),
    "timing diagnostics": (resolved.inference.timing_diagnostics, True),
    "guided execution window enforcement": (
        resolved.inference.enforce_guided_execution_window,
        expected_guided_window_enforcement,
    ),
    "prefix backend": (resolved.pi05_prefix_backend.value, expected_prefix_backend),
    "tensorrt prefix engine": (resolved.pi05_tensorrt_prefix_engine, expected_trt_engine),
    "action backend": (resolved.pi05_action_backend.value, "pytorch"),
    "action filter enabled": (resolved.action_filter.enabled, expected_action_filter),
    "stall guard ticks": (resolved.stall_guard_ticks, int(arg_value("stall_guard_ticks"))),
    "stall guard tolerance": (resolved.stall_guard_tolerance, float(arg_value("stall_guard_tolerance"))),
    "fps": (resolved.fps, 30),
    "interpolation multiplier": (resolved.interpolation_multiplier, 1),
    "max relative target": (resolved.robot.max_relative_target, 5.0),
    "return to initial position": (
        resolved.return_to_initial_position,
        expected_return_to_initial,
    ),
    "prefix health enabled": (
        resolved.inference.prefix_health_enabled,
        expected_prefix_health,
    ),
    "prefix health residual threshold": (
        resolved.inference.prefix_health_severe_residual_threshold,
        expected_prefix_threshold,
    ),
    "prefix health consecutive severe": (
        resolved.inference.prefix_health_consecutive_severe,
        expected_prefix_consecutive,
    ),
    "prefix health safety stop replans": (
        resolved.inference.prefix_health_safety_stop_replans,
        expected_prefix_stop_replans,
    ),
    "seed": (resolved.seed, expected_seed),
}
for label, (actual, expected) in resolved_checks.items():
    if actual != expected:
        raise SystemExit(f"Unexpected resolved {label}: expected {expected!r}, got {actual!r}")
if resolved.duration <= 0:
    raise SystemExit("T0 duration must be positive")
PY
}

print_final_config() {
  printf '%s\n' \
    "T0 rollout configuration validated" \
    "model=$MODEL" \
    "policy_kind=old_full_005613" \
    "rtc_timing=actual_consumed" \
    "guidance_delay=fixed:$FIXED_GUIDANCE_DELAY_STEPS" \
    "prefix_backend=$PREFIX_BACKEND" \
    "tensorrt_prefix_engine=${TRT_PREFIX_ENGINE:-none}" \
    "action_backend=pytorch" \
    "action_filter_enabled=$ACTION_FILTER_ENABLED" \
    "stall_guard_ticks=$STALL_GUARD_TICKS" \
    "stall_guard_tolerance=$STALL_GUARD_TOLERANCE" \
    "fps=$FPS" \
    "chunk_size=$CHUNK_SIZE" \
    "denoise_steps=$NUM_INFERENCE_STEPS" \
    "queue_threshold=$QUEUE_THRESHOLD" \
    "nominal_replan_hz=$(awk -v fps="$FPS" -v chunk="$CHUNK_SIZE" -v queue="$QUEUE_THRESHOLD" 'BEGIN { printf "%.3f", fps / (chunk - queue) }')" \
    "execution_horizon=$EXECUTION_HORIZON" \
    "max_guidance_weight=$MAX_GUIDANCE_WEIGHT" \
    "prefix_attention_schedule=EXP" \
    "interpolation_multiplier=1" \
    "max_relative_target=$MAX_RELATIVE_TARGET" \
    "duration=$DURATION" \
    "seed=$SEED" \
    "return_to_initial_position=$RETURN_TO_INITIAL_POSITION" \
    "timing_diagnostics=true" \
    "enforce_guided_execution_window=$ENFORCE_GUIDED_EXECUTION_WINDOW" \
    "prefix_health_enabled=$PREFIX_HEALTH_ENABLED" \
    "prefix_health_severe_residual_threshold=$PREFIX_HEALTH_SEVERE_RESIDUAL_THRESHOLD" \
    "prefix_health_consecutive_severe=$PREFIX_HEALTH_CONSECUTIVE_SEVERE" \
    "prefix_health_safety_stop_replans=$PREFIX_HEALTH_SAFETY_STOP_REPLANS" \
    "robot_port=$ROBOT_PORT" \
    "top_camera=$TOP_CAMERA" \
    "wrist_camera=$WRIST_CAMERA" \
    "hardware_connected=false"
  printf 'command='
  printf '%q ' "$PY" -m lerobot.scripts.lerobot_rollout "${ROLLOUT_ARGS[@]}"
  printf '\n'
}

validate_t0_config
print_final_config

if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "T0 preflight complete; no robot or camera was opened."
  exit 0
fi

[[ -e "$ROBOT_PORT" ]] || fail "Robot port not found: $ROBOT_PORT"
[[ -r "$ROBOT_PORT" && -w "$ROBOT_PORT" ]] || fail "Robot port is not readable and writable: $ROBOT_PORT"
[[ -r "$TOP_CAMERA" ]] || fail "Top camera is not readable: $TOP_CAMERA"
[[ -r "$WRIST_CAMERA" ]] || fail "Wrist camera is not readable: $WRIST_CAMERA"

if fuser "$ROBOT_PORT" >/dev/null 2>&1; then
  fail "Robot port is already in use: $ROBOT_PORT"
fi
if fuser "$TOP_CAMERA" >/dev/null 2>&1; then
  fail "Top camera is already in use: $TOP_CAMERA"
fi
if fuser "$WRIST_CAMERA" >/dev/null 2>&1; then
  fail "Wrist camera is already in use: $WRIST_CAMERA"
fi

cd "$ROOT"
exec "$PY" -m lerobot.scripts.lerobot_rollout "${ROLLOUT_ARGS[@]}"
