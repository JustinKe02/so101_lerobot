#!/usr/bin/env bash
set -Eeuo pipefail

export RTC_MAX_DELAY=15
export JOB_NAME=pi05_so101_realtime_vla_v2_full_10epochs_rtc15_seed1000

exec bash /data/cqy_workspace/tk/lerobot_src/src/lerobot/scripts/train_pi05_so101_realtime_vla_v2_full.sh
