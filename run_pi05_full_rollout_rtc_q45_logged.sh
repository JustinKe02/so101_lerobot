#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
LOG_DIR=${PI05_T0A_LOG_DIR:-$ROOT/outputs/rollout/t0a_q45}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH=$LOG_DIR/t0a_q45_${TIMESTAMP}.log
REPORT_PATH=$LOG_DIR/t0a_q45_${TIMESTAMP}.analysis.json

if [[ "${PI05_PREFLIGHT_ONLY:-0}" == 1 || "${PI05_PREFLIGHT_ONLY:-0}" == true ]]; then
  echo "Use run_pi05_full_rollout_rtc_q45.sh directly for a no-hardware preflight." >&2
  exit 1
fi
if [[ -n "${PI05_DURATION:-}" && "${PI05_DURATION}" != 5 ]]; then
  echo "The supervised T0a gate requires PI05_DURATION=5." >&2
  exit 1
fi
export PI05_DURATION=5

mkdir -p "$LOG_DIR"
echo "T0a supervised rollout log: $LOG_PATH"

set +e
"$ROOT/run_pi05_full_rollout_rtc_q45.sh" 2>&1 | tee "$LOG_PATH"
ROLLOUT_STATUS=${PIPESTATUS[0]}
set -e

set +e
"$PY" "$ROOT/examples/inference/analyze_pi05_t0a_log.py" \
  --log "$LOG_PATH" \
  --rollout-exit-code "$ROLLOUT_STATUS" \
  --output-json "$REPORT_PATH"
ANALYSIS_STATUS=$?
set -e

echo "T0a rollout exit code: $ROLLOUT_STATUS"
echo "T0a analysis report: $REPORT_PATH"

if [[ "$ROLLOUT_STATUS" -ne 0 ]]; then
  exit "$ROLLOUT_STATUS"
fi
exit "$ANALYSIS_STATUS"
