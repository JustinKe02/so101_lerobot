# 2026-08-09 Realtime-VLA V2 机器日志说明

本目录保存由校验工具生成的机器可读日志。JSON 字段名、枚举值和检查描述属于固定数据格式，
为了保持分析器兼容性不做翻译；对应的中文结论请查看：

- `REALTIME_VLA_V2_DAILY_REPORT_20260809.md`
- `reports/realtime_vla_v2/2026-08-09/README.md`
- `reports/realtime_vla_v2/2026-08-09/training_summary.json`

文件说明：

- `trace_rtc6_triton_all.json`：Triton 集成阶段的 5 次会话。
- `trace_triton_direct_no_time_axis.json`：关闭时间轴规划器的 Triton 直出会话。
- `trace_pytorch_direct_prefix5.json`：两次 PyTorch 直出会话。
- `trace_triton_direct_rolling_p95.json`：滚动 P95 前缀的 Triton 直出会话。
- `triton_parity_rtc6.json`：RTC6 权重在 0-6 步前缀上的 PyTorch/Triton 一致性。
- `raw_artifacts.sha256`：本机原始日志、轨迹与一致性文件的 SHA-256 和字节数。
