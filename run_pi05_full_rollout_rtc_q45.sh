#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
GATE_REPORT=${PI05_RTC_GATE_REPORT:-$ROOT/outputs/eval/pi05_full_rtc_replay_q45_final_v1.json}
EVALUATOR=$ROOT/examples/inference/evaluate_pi05_rtc_replay.py

[[ -x "$PY" ]] || { echo "Python executable not found: $PY" >&2; exit 1; }
[[ -f "$GATE_REPORT" ]] || { echo "RTC replay gate report not found: $GATE_REPORT" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "RTC replay evaluator not found: $EVALUATOR" >&2; exit 1; }

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
"$PY" - "$GATE_REPORT" "$EVALUATOR" "$MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from examples.inference.evaluate_pi05_multiseed import verify_checkpoint_asset_fingerprint

report_path = Path(sys.argv[1]).resolve()
evaluator_path = Path(sys.argv[2]).resolve()
expected_model = Path(sys.argv[3]).resolve()
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
}
if report.get("required_rollout_config") != expected_config:
    raise SystemExit("RTC replay report has a different rollout configuration contract")

print(f"RTC replay gate verified: {report_path}")
PY

export PI05_QUEUE_THRESHOLD=45
export PI05_ENFORCE_GUIDED_EXECUTION_WINDOW=true
export PI05_DURATION=${PI05_DURATION:-5}

exec "$ROOT/run_pi05_full_rollout_rtc_actual.sh" "$@"
