#!/usr/bin/env bash
set -euo pipefail

# T1 supervised long-run launcher: TensorRT prefix + per-joint action output
# filter + Q45 fixed-guidance RTC. Binds three offline gates before any
# hardware is touched:
#   1. the filtered RTC replay deployment report (action_filter_enabled=true),
#   2. the TensorRT engine offline verification marker (<engine>.plan.verified.json),
#   3. the plan §8.2 real-frame RTC parity report (normalized mean <= 0.005, max <= 0.020).
# The deep engine/model binding (sha256 + weight/architecture fingerprints) is
# re-validated fail-closed in-process at attach time by tensorrt_prefix.py.

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
GATE_REPORT=${PI05_RTC_GATE_REPORT:-$ROOT/outputs/eval/pi05_full_rtc_replay_q45_filter_v1.json}
EVALUATOR=$ROOT/examples/inference/evaluate_pi05_rtc_replay.py
TRT_PREFIX_ENGINE=${PI05_TRT_PREFIX_ENGINE:-$MODEL/pi05_tensorrt_rebuild_bf16/prefix_cache_bf16.plan}
TRT_VERIFIED_MARKER=$TRT_PREFIX_ENGINE.verified.json
RTC_PARITY_REPORT=${PI05_RTC_PARITY_REPORT:-$(dirname "$TRT_PREFIX_ENGINE")/rtc_parity_oldfull.json}

[[ -x "$PY" ]] || { echo "Python executable not found: $PY" >&2; exit 1; }
[[ -f "$GATE_REPORT" ]] || { echo "RTC replay gate report not found: $GATE_REPORT" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "RTC replay evaluator not found: $EVALUATOR" >&2; exit 1; }
[[ -f "$TRT_PREFIX_ENGINE" ]] || { echo "TensorRT prefix engine not found: $TRT_PREFIX_ENGINE" >&2; exit 1; }
[[ -f "$TRT_VERIFIED_MARKER" ]] || { echo "TensorRT engine verification marker not found: $TRT_VERIFIED_MARKER" >&2; exit 1; }
[[ -f "$RTC_PARITY_REPORT" ]] || { echo "RTC parity report not found: $RTC_PARITY_REPORT" >&2; exit 1; }

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
"$PY" - "$GATE_REPORT" "$EVALUATOR" "$MODEL" "$TRT_PREFIX_ENGINE" "$TRT_VERIFIED_MARKER" "$RTC_PARITY_REPORT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from examples.inference.evaluate_pi05_multiseed import verify_checkpoint_asset_fingerprint

report_path = Path(sys.argv[1]).resolve()
evaluator_path = Path(sys.argv[2]).resolve()
expected_model = Path(sys.argv[3]).resolve()
engine_path = Path(sys.argv[4]).resolve()
marker_path = Path(sys.argv[5]).resolve()
parity_path = Path(sys.argv[6]).resolve()

# --- Gate 1: filtered RTC replay deployment report -------------------------
report = json.loads(report_path.read_text())

if report.get("report_status") != "complete" or report.get("deployment_passed") is not True:
    raise SystemExit("RTC replay report has not passed the deployment safety gate")
if report.get("hardware_action_sent") is not False:
    raise SystemExit("RTC replay report has an invalid hardware_action_sent marker")
if report.get("deployment_candidate") != "q45_fixed_guidance_5":
    raise SystemExit("RTC replay report was produced for a different deployment candidate")

descriptor = report.get("checkpoint", {})
if Path(descriptor.get("checkpoint", "")).resolve() != expected_model:
    raise SystemExit("RTC replay report references a different checkpoint")
verify_checkpoint_asset_fingerprint(descriptor)

evaluator = report.get("evaluator", {})
if Path(evaluator.get("path", "")).resolve() != evaluator_path:
    raise SystemExit("RTC replay report references a different evaluator")
digest = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
if evaluator.get("sha256") != digest:
    raise SystemExit("RTC replay evaluator changed after the gate report was generated")

expected_config = {
    "timing_mode": "actual_consumed",
    "guidance_delay_mode": "fixed",
    "fixed_guidance_delay_steps": 5,
    "queue_threshold": 45,
    "execution_horizon": 10,
    "max_guidance_weight": 10.0,
    "prefix_attention_schedule": "EXP",
    "enforce_guided_execution_window": True,
    "prefix_backend": "pytorch",
    "action_backend": "pytorch",
    "action_filter_enabled": True,
}
if report.get("required_rollout_config") != expected_config:
    raise SystemExit("RTC replay report has a different rollout configuration contract")

print(f"RTC replay gate verified: {report_path}")

# --- Gate 2: TensorRT engine offline verification marker --------------------
marker = json.loads(marker_path.read_text())

if marker.get("format") != "lerobot_pi05_tensorrt_prefix_verification_v2":
    raise SystemExit("TensorRT verification marker has an unexpected format")
if marker.get("passed") is not True:
    raise SystemExit("TensorRT verification marker does not record a passing verification")
if Path(marker.get("engine", "")).resolve() != engine_path:
    raise SystemExit("TensorRT verification marker references a different engine")
if Path(marker.get("checkpoint", "")).resolve() != expected_model:
    raise SystemExit("TensorRT verification marker references a different checkpoint")
stat = os.stat(engine_path)
if marker.get("engine_size") != stat.st_size or marker.get("engine_mtime_ns") != stat.st_mtime_ns:
    raise SystemExit(
        "TensorRT engine changed after verification; rerun --verify-engine "
        f"(size {stat.st_size} vs verified {marker.get('engine_size')}, "
        f"mtime_ns {stat.st_mtime_ns} vs verified {marker.get('engine_mtime_ns')})"
    )
if marker.get("num_cameras") != 2 or marker.get("num_layers") != 18:
    raise SystemExit("TensorRT verification marker has unexpected camera/layer counts")
for key in ("engine_sha256", "prefix_weight_fingerprint", "model_architecture_fingerprint"):
    value = marker.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"TensorRT verification marker is missing a usable {key}")

print(f"TensorRT engine verification gate verified: {marker_path}")

# --- Gate 3: plan §8.2 real-frame RTC parity report -------------------------
parity = json.loads(parity_path.read_text())

if Path(parity.get("model", "")).resolve() != expected_model:
    raise SystemExit("RTC parity report references a different checkpoint")
if Path(parity.get("engine", "")).resolve() != engine_path:
    raise SystemExit("RTC parity report references a different engine")
if parity.get("hardware_action_sent") is not False:
    raise SystemExit("RTC parity report has an invalid hardware_action_sent marker")

delays = parity.get("delays") or []
covered = {entry.get("delay") for entry in delays}
if not {2, 3, 4, 5, 6}.issubset(covered):
    raise SystemExit(f"RTC parity report does not cover delays 2-6 (got {sorted(covered)})")
for entry in delays:
    normalized = entry.get("normalized_action") or {}
    mean_abs = normalized.get("mean_abs")
    max_abs = normalized.get("max_abs")
    if mean_abs is None or max_abs is None:
        raise SystemExit(f"RTC parity report is missing normalized metrics for delay {entry.get('delay')}")
    if mean_abs > 0.005 or max_abs > 0.020:
        raise SystemExit(
            "RTC parity gate failed for delay "
            f"{entry.get('delay')}: normalized mean {mean_abs} (max 0.005), max {max_abs} (max 0.020)"
        )

print(f"RTC parity gate verified: {parity_path}")
PY

export PI05_PREFIX_BACKEND=tensorrt
export PI05_TRT_PREFIX_ENGINE="$TRT_PREFIX_ENGINE"
export PI05_ACTION_FILTER_ENABLED=true
export PI05_STALL_GUARD_TICKS=${PI05_STALL_GUARD_TICKS:-5}
if ! [[ "$PI05_STALL_GUARD_TICKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "The T1 run requires the stall/contact guard (PI05_STALL_GUARD_TICKS >= 1), got: $PI05_STALL_GUARD_TICKS" >&2
  exit 1
fi
export PI05_QUEUE_THRESHOLD=45
export PI05_ENFORCE_GUIDED_EXECUTION_WINDOW=true
export PI05_DURATION=${PI05_DURATION:-45}

exec "$ROOT/run_pi05_full_rollout_rtc_actual.sh" "$@"
