# vLLM 运行入口

社区完整流程见 [快速开始](../../../doc/getting-started.md) 和
[Tutti × vLLM 操作手册](../../../doc/intergration/tutti-vllm-manual.md)。这里不复制第二套安装流程。

| 文件 | 职责 |
|---|---|
| `profile-env.sh` | 从用户环境获得 Python、NVMe scratch，设置 JIT/cache/PYTHONPATH；不修改 GPU 或编译器 |
| `serve-8gpu-striped.sh` | 生成并校验 connector 配置，启动在线 API；`--dry-run` 打印真实配置，不加载模型 |
| `bench-8gpu-striped.sh` | 离线 A→B、HBM 对照、nsys wrapper；TP 可覆盖，历史名称不代表强制 8 卡 |
| `../vllm_profile_offline.py` | 直接 `LLM.generate()`，参数和 profiler 门控见 `--help` |
| `../vllm_online_longrun.py` | 在线 trace 客户端，保存客户端侧请求/延迟统计 |
| `nsys-bubble.py` | SQLite 时间线分析；按报告中的进程和 kernel 语义解释结果 |

## 必须由调用者指定的环境

```bash
export TUTTI_PYTHON="$(command -v python)"
export MODEL=/path/to/model
export TUTTI_PROFILE_ROOT=/path/on/nvme/tutti-profile
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export DAEMON_CONFIG="$PWD/config/local/tutti_daemon_kv.yaml"
export DEVICE_GROUPS="0,1,2,3"
export KV_MOUNT=/mnt/your-kv
```

`DAEMON_CONFIG`、盘组和 mount 的未覆盖默认值来自维护者的双阵列部署，**不是任意新机器
的可用配置**。上例的`tutti_daemon_kv.yaml`须自行创建并审核，不是预装文件。
`MODEL`和NVMe scratch必须明确提供。权重加载默认为`auto`；Phoenix
loader 仅在显式选择 `LOAD_FORMAT=phxsafetensors` / `PHXLOADER_DIR` 时使用。

参数的模块归属、默认/单位/优先级见 [配置与性能指南](../../../doc/configuration-and-performance.md)。
`NUM_QUEUES` 支持serve/bench覆盖，默认8；这是每rank每设备值，不是全服务队列总量。
服务用`TOTAL_CAPACITY="128 GiB"`指定服务总预算；离线用`CAPACITY="64 GiB"`或
`--capacity "64 GiB"`指定每rank预算，HBM baseline无需KV容量。
模型 block/context/KV dtype/parser 的选择和 SSE 验收在操作手册中。启动前必须先
`serve-8gpu-striped.sh --dry-run` 检查，脚本不会自动准备驱动或接管 RAID。

## 日志和测试纪律

- 临时文件、缓存、报告放 `TUTTI_PROFILE_ROOT`，不使用根盘 `/tmp`。
- 新 `POOL_TAG` 只隔离索引，**不隔离** `<mount>/r<rank>/segments` 的数据文件。
  测试与在线服务不得同时占用同一 mount/rank 工作区。
- benchmark 默认在 A/B 间只清本地 HBM cache，保留 connector cache。检查真实读计划和
  输出一致后才报告性能；首轮与稳态分开。
- 报告名包含 TP/长度/复用率/tag/时间戳，另存运行参数；不能拿被覆盖的日志作证据。
- nsys 需要匹配驱动和工具版本。系统范围 metrics 可能需要额外权限，不自动提权。

## nsys 最小流程

```bash
bash scripts/vllm/run/bench-8gpu-striped.sh \
  --capacity "64 GiB" --nsys --tokens 10000 --reuse-pct 90 --rounds 3 --pool-tag profile-check
nsys stats --report cuda_gpu_kern_sum,nvtx_sum /path/to/report.nsys-rep
```

读/写方向以 `tutti.*|op=read` / `|op=write` 和实际 stream/请求路径为准，不使用 grid 大小
猜测。同步路径是 fused submit-and-wait；实验异步路径包含 submit-only 与 `wait_batches`。
等待 kernel 的耗时不能当成独立 NVMe 数据吞吐。

需要跨 CPU/GPU 异步边界比较同一步时，使用 launch correlation 归属而非 CPU step 时间窗。
异步功能状态与实测见 [async-nvme-completion.md](../../../doc/design/async-nvme-completion.md)。
