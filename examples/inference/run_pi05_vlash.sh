#!/usr/bin/env bash

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 POLICY_PATH [lerobot-rollout robot/camera arguments...]" >&2
  exit 2
fi

POLICY_PATH=$1
shift

EXECUTION_HORIZON=${VLASH_EXECUTION_HORIZON:-10}
OVERLAP_STEPS=${VLASH_INFERENCE_OVERLAP_STEPS:-5}
MAX_FUTURE_STATE_DELTA=${VLASH_MAX_FUTURE_STATE_DELTA:-5.0}
DEADLINE_MISS_LIMIT=${VLASH_DEADLINE_MISS_LIMIT:-1}
CAMERA_WARMUP_S=${VLASH_CAMERA_WARMUP_S:-5.0}
DURATION_S=${VLASH_DURATION_S:-60}
TIMING_DIAGNOSTICS=${VLASH_TIMING_DIAGNOSTICS:-true}
FUSE_QKV=${PI05_FUSE_QKV:-false}
FUSE_GATE_UP=${PI05_FUSE_GATE_UP:-false}

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run lerobot-rollout)
else
  LEROBOT_PYTHON=${LEROBOT_PYTHON:-/home/cqy/miniconda3/envs/lerobot_tk/bin/python}
  if [[ ! -x "$LEROBOT_PYTHON" ]]; then
    echo "Neither uv nor an executable LEROBOT_PYTHON is available" >&2
    exit 1
  fi
  RUNNER=("$LEROBOT_PYTHON" -m lerobot.scripts.lerobot_rollout)
fi

exec "${RUNNER[@]}" \
  --strategy.type=base \
  --inference.type=vlash \
  --inference.execution_horizon="$EXECUTION_HORIZON" \
  --inference.inference_overlap_steps="$OVERLAP_STEPS" \
  --inference.max_future_state_delta="$MAX_FUTURE_STATE_DELTA" \
  --inference.deadline_miss_limit="$DEADLINE_MISS_LIMIT" \
  --inference.timing_diagnostics="$TIMING_DIAGNOSTICS" \
  --inference.require_state_conditioning=true \
  --inference.require_offset_training=true \
  --policy.path="$POLICY_PATH" \
  --policy.fuse_qkv="$FUSE_QKV" \
  --policy.fuse_gate_up="$FUSE_GATE_UP" \
  --camera_warmup_s="$CAMERA_WARMUP_S" \
  --duration="$DURATION_S" \
  --interpolation_multiplier=1 \
  "$@"
