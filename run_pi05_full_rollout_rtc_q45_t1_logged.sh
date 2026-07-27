#!/usr/bin/env bash
set -euo pipefail

# Supervised T1 long-run wrapper: 30-60 s duration, TensorRT prefix + action
# output filter, full log capture, and automatic gate analysis. The latency
# gate stays at the T0a value (166.667 ms = 30 fps replan budget); TensorRT
# is expected to bring the peak under it, not the other way around.

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
LOG_DIR=${PI05_T1_LOG_DIR:-$ROOT/outputs/rollout/t1_q45}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH=$LOG_DIR/t1_q45_${TIMESTAMP}.log
REPORT_PATH=$LOG_DIR/t1_q45_${TIMESTAMP}.analysis.json

if [[ "${PI05_PREFLIGHT_ONLY:-0}" == 1 || "${PI05_PREFLIGHT_ONLY:-0}" == true ]]; then
  echo "Use run_pi05_full_rollout_rtc_q45_t1.sh directly for a no-hardware preflight." >&2
  exit 1
fi

DURATION=${PI05_DURATION:-45}
if ! [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PI05_DURATION must be a non-negative number, got: $DURATION" >&2
  exit 1
fi
if ! awk -v d="$DURATION" 'BEGIN { exit !(d >= 30 && d <= 60) }'; then
  echo "The supervised T1 long run requires 30 <= PI05_DURATION <= 60 (got $DURATION)." >&2
  exit 1
fi
export PI05_DURATION="$DURATION"

mkdir -p "$LOG_DIR"
echo "T1 supervised rollout log: $LOG_PATH"

set +e
"$ROOT/run_pi05_full_rollout_rtc_q45_t1.sh" 2>&1 | tee "$LOG_PATH"
ROLLOUT_STATUS=${PIPESTATUS[0]}
set -e

set +e
"$PY" "$ROOT/examples/inference/analyze_pi05_t0a_log.py" \
  --log "$LOG_PATH" \
  --rollout-exit-code "$ROLLOUT_STATUS" \
  --output-json "$REPORT_PATH" \
  --duration "$DURATION" \
  --expect-prefix-backend tensorrt \
  --expect-action-filter true \
  --expect-stall-guard true \
  --run-kind supervised_real_robot_t1
ANALYSIS_STATUS=$?
set -e

echo "T1 rollout exit code: $ROLLOUT_STATUS"
echo "T1 analysis report: $REPORT_PATH"

if [[ "$ROLLOUT_STATUS" -ne 0 ]]; then
  exit "$ROLLOUT_STATUS"
fi
exit "$ANALYSIS_STATUS"
