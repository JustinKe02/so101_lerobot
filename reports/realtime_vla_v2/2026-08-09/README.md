# Realtime-VLA V2 reports for 2026-08-09

This directory contains compact, reviewable evidence derived from today's local
training and robot rollout logs.

- `training_summary.json`: RTC6 completed run and intentionally stopped RTC15 run.
- `training_key_events.txt`: selected original training status and metric lines.
- `raw_artifacts.sha256`: hashes and local sizes for raw logs and traces.
- `triton_parity_rtc6.json`: fixed-noise PyTorch/Triton parity for prefix lengths 0-6.
- `trace_rtc6_triton_all.json`: five Triton integration sessions, including three
  fail-closed debugging sessions and two completed sessions.
- `trace_triton_direct_no_time_axis.json`: direct Triton, fixed prefix 5, planner off.
- `trace_pytorch_direct_prefix5.json`: two direct PyTorch sessions.
- `trace_triton_direct_rolling_p95.json`: direct Triton, planner on, rolling-P95 prefix.

The raw JSONL traces total more than 180 MB and are excluded by `.gitignore`. They
remain under `outputs/traces/`; the SHA-256 manifest binds these reports to those
exact source files. Model checkpoints, optimizer state, and the 6.7 GB Triton export
are also intentionally not stored in Git.

The trace validator's default acceptance result is false for these ablations because
of one startup empty-queue event per session and intentionally disabled paper-level
calibration features. Check `terminal.status` and the individual checks when
distinguishing a completed run from an integration failure.
