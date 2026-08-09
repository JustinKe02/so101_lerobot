# 2026-08-09 Realtime-VLA V2 中文报告归档

> 状态说明：本归档记录的是 RTC6 training-RTC、Triton 推理和部分 Realtime-VLA V2 运行组件
> 的集成结果，不代表完整论文级运行栈已经启用。动态 action-prefill、实测传感器与电机标定、
> 固定心跳 Realtime Executor 和 Speed Adapter 在归档会话中均未完整启用。

本目录只保存面向人员阅读的中文报告内容：

- `training_summary.json`：RTC6 完整训练与 RTC15 主动停止训练的中文结构化摘要。
- `training_key_events.txt`：训练开始、结束、损失和权重保存等关键事件。
- 仓库根目录的 `REALTIME_VLA_V2_DAILY_REPORT_20260809.md`：当天训练、推理和上机结论。

校验器输出采用固定英文数据格式，不能翻译字段名，否则会破坏后续工具读取。这些文件已移动到
`artifacts/realtime_vla_v2/2026-08-09/`，作为机器日志而不是面向人员的报告：

- `triton_parity_rtc6.json`：0-6 步前缀的 PyTorch/Triton 固定噪声一致性结果。
- `trace_rtc6_triton_all.json`：5 次 Triton 集成运行，包含 3 次故障关闭调试和 2 次正常完成。
- `trace_triton_direct_no_time_axis.json`：Triton 直出、固定前缀 5、关闭规划器。
- `trace_pytorch_direct_prefix5.json`：两次 PyTorch 直出运行。
- `trace_triton_direct_rolling_p95.json`：Triton 直出、启用规划器、滚动 P95 前缀。
- `raw_artifacts.sha256`：本机原始日志和轨迹的文件大小及 SHA-256。

原始 JSONL 轨迹合计超过 180 MB，继续保存在本机 `outputs/traces/`，不直接进入 Git。模型
权重、优化器状态和 6.7 GB 的 Triton 导出也不提交到代码仓库。

机器日志默认显示 `overall_pass=false`，主要原因是每次运行启动时有一次空队列事件，以及论文级
标定功能被有意关闭。判断运行是否正常时，应结合 `terminal.status` 和各项子检查；日报已用中文
给出解释。Triton 延迟降低只代表模型推理加速，不应解释为完整 Realtime-VLA V2 已经实现或
任务速度已经获得论文中的同等提升。
