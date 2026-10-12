# Tutti × vLLM 操作手册

这是可复制的部署与验收流程，不是维护者机器的历史日志。先完成
[`快速开始`](../getting-started.md) 的构建、三个 Python 包、宿主安全配置及硬件验证。
Connector 的入口是 `tutti.integration.vllm.connector.TuttiConnectorV1`；不再使用
旧的 `adapter.connector`、`integration/vllm-connector` 或 `/data2` 路径。

## 1. 明确运行环境

在仓库根、同一 Python 环境中设置：

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

以上值是示例，必须匹配已经启动的daemon。`tutti_daemon_kv.yaml`表示部署者自行创建并
审核的配置，不是仓库预装文件；先按快速开始创建它。`MODEL`也可以是Hugging Face ID。
默认权重加载为 vLLM 的 `auto`，**KV offload 不依赖 Phoenix**。

| 参数 | 含义 |
|---|---|
| `DAEMON_CONFIG` | 当前运行的 daemon YAML；多个文件以逗号分隔，合并成客户端盘清单 |
| `DEVICE_GROUPS` | 每组以逗号分隔 `device_id`，组间分号；不是 `/dev/nvmeN` 编号 |
| `TP_SIZE` | vLLM TP rank 数；须能被盘组数量整除 |
| `KV_MOUNT` | 各 rank checkpoint/索引根所在挂载点；段文件位于所分配设备的 mount 下 |
| `TUTTI_PROFILE_ROOT` | 日志、JIT cache、临时文件、profiler 报告的 NVMe 目录 |
| `TUTTI_PYTHON` | 已安装 torch、vLLM、两个原生扩展的解释器 |

两个 NUMA 盘组的例子：`TP_SIZE=8 DEVICE_GROUPS="0,1,2,3;4,5,6,7"`，rank 0–3
使用第一组，rank 4–7 使用第二组。每组 rank 都访问该组设备。PCI ACL 必须允许这些访问，
不能仅凭“GPU i 对 NVMe i”推断其它成员盘的 P2P 可达性。

## 2. 配置模型几何与容量

原生Runtime YAML与Python preset的区别、参数默认/单位/优先级和性能选择，见
[配置与性能指南](../configuration-and-performance.md)。最小connector JSON示例位于
`config/integration/vllm/kv_transfer.json`；必须修改其中的本机路径、设备ID与每rank容量。
逐项注释见 [Python用户参数说明](../../config/integration/vllm/README.md)。直接JSON可写
`"capacity": "64 GiB"`，并省略`max_chunks_per_wave`、`io_stream`和`num_queues`；
在线wrapper使用`TOTAL_CAPACITY="128 GiB"`指定服务总payload预算，先除TP再传每rank。
离线wrapper使用`CAPACITY`或`--capacity`指定每rank预算，不按模型名字猜测槽数。

运行入口仍名为 `serve-8gpu-striped.sh` / `bench-8gpu-striped.sh`，名称是历史名称；
`TP_SIZE`、GPU 列表和盘组可覆盖，并不强制 8 卡。它们的预置几何来自 Hy3-FP8 验证环境，
不是所有模型的安全默认。

- `BLOCK_SIZE` 必须被所选 vLLM attention backend 支持，且 `CHUNK_TOKENS` 是其整数倍。
- `KV_CACHE_DTYPE=fp8` 必须有可用且正确的 KV scales；未经验证的模型先使用 `auto`。
- `MAX_MODEL_LEN` 不超过模型和显存允许的上限；prompt 加生成 token 也不能超限。
- `TOOL_CALL_PARSER` 必须符合模型工具格式；不把 Hermes 当作所有模型的兼容承诺。
- 当前仅支持单一、定宽 KV cache group，多组/hybrid 布局会拒绝启动。
- `TOTAL_CAPACITY` 是带单位的服务总payload预算，除以TP后传每rank；不是每盘上限。
  首个段文件组仍需要足够空间，不能将预建开销误认为完全按需零成本。

段文件实际位于 `<mount>/r<rank>/segments/`。**不同 `POOL_TAG` 只隔离索引，不隔离这些
段文件**；不得在同一挂载点/rank 上并发运行多个实例，也不得并发 benchmark 与线上服务。
模型/TP/几何改变时应使用独立的数据部署或先停止并有计划地清理旧 KV。

## 3. 离线 A→B 复用测试

先停占用同一组 GPU / KV 段文件的服务，从较小上下文开始：

```bash
bash scripts/vllm/run/bench-8gpu-striped.sh \
  --capacity "64 GiB" --tokens 10000 --reuse-pct 90 --rounds 3 --pool-tag reuse-check
```

这个 wrapper 的离线几何固定为 `block-size=64`，KV dtype 使用 vLLM 默认；不要把它与在线
脚本的 FP8/block=512 数字直接混在一起比较。需要调整离线几何时调用 driver 的 `--help`。

- A 计算并写入，B 复用 A 的一部分前缀。
- Tutti 跑法在 A/B 间只清本地 HBM prefix cache，保留外部缓存。否则 B 可能只从 HBM 命中。
- 不同轮次改变首 token，保持 shape 一致且避免轮间重复命中。
- 检查外部命中及 `DIRECT_READ_PLAN_READY`、失败记录和生成 token，再比较 `steady_mean`。
  首轮包含 JIT、注册等一次性成本；HTTP 200 或“速度快”本身不能证明 KV 直读成功。

同日 HBM 对照：

```bash
bash scripts/vllm/run/bench-8gpu-striped.sh \
  --baseline --tokens 10000 --reuse-pct 90 --rounds 3
```

这是 HBM prefix-cache 参考，不是 B 全量重算参考。不要同时运行两组测试。

## 4. 在线服务：先预览、再启动

例子使用兼容性较保守的 KV dtype 与 context 长度，须依据目标模型调整 block/parser：

```bash
export PORT=8000
export SERVED_NAME=tutti
export KV_CACHE_DTYPE=auto
export BLOCK_SIZE=64
export CHUNK_TOKENS=512
export MAX_MODEL_LEN=32768
export MAX_PROMPT_TOKENS=32768
export TOTAL_CAPACITY="128 GiB"
export POOL_TAG=online-v020
bash scripts/vllm/run/serve-8gpu-striped.sh --dry-run
```

预览会校验 daemon 清单、盘组 ID 和 TP，打印真实 connector JSON 及执行命令，不加载模型。
确认后以前台方式启动，方便记录并管理进程：

```bash
bash scripts/vllm/run/serve-8gpu-striped.sh
```

需要后台启动时，将输出明确放到 NVMe，记录 API 进程 PID，并在停机时给该进程一次
SIGTERM、等待全部 worker 退出；不要用宽泛 `pkill` 杀其它用户任务。此脚本不自动加载
驱动、重建 RAID、格式化、清 KV 或释放控制器。

### 验收 API 与流式输出

在另一个终端：

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
curl --fail -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tutti","prompt":"Explain what a KV cache is.","max_tokens":32,"temperature":0,"stream":true}'
curl --fail http://localhost:8000/metrics
```

成功标准：health=200、model 列表含服务名、流里有 token 并以 `[DONE]` 结束。
同 prompt 再次调用可能命中 HBM；证明 NVMe 复用仍以第 3 节的离线隔离测试为准。
长 trace 使用 `scripts/vllm/vllm_online_longrun.py --help` 选择数据集和并发，并单独保存客户端
TTFT、错误码及运行参数；系统日志不一定能解释分词/排队/客户端延迟的全部开销。

### 停止和恢复 Linux 驱动

顺序是停止 API/worker → 确认无 IO 客户端 → 停 daemon/退回阵列。不要先解绑定 SSD。
已登记的阵列通过 `tutti-env.sh daemon stop <name>` 交回 Linux；`protected` 级别还需要
`--confirm <name>`，这不是绕过使用者检查的强制选项。不要对 daemon 连发两次 SIGTERM，
也不要用 lazy unmount 当作安全回退。

## 5. nsys 与实验异步完成

```bash
bash scripts/vllm/run/bench-8gpu-striped.sh \
  --nsys --tokens 10000 --reuse-pct 90 --rounds 3 --pool-tag nsys-check
```

日志与 `.nsys-rep` 位于 `$TUTTI_PROFILE_ROOT/logs` / `reports`。CUDA/NVTX 必须覆盖
真正的请求；按 launch correlation 和进程归属比较同一步，不能只按 CPU 时间窗截取 GPU
kernel。单次 kernel 求和不是墙钟，也不能仅凭 kernel 名认定读写方向。

`TUTTI_ASYNC_COMPLETION=1` 是可选实验模式：提交与 CQ 回收分离；默认关闭。不改
vLLM/FlashInfer。正常路径及 64K/256K 性能结果见
[`异步完成设计`](../design/async-nvme-completion.md)，但超时/迟到完成/资源隔离门禁尚未
完整验收，不作为 v0.2.0 的默认安全路径。

## 6. 排障入口

| 症状 | 先查 |
|---|---|
| import 失败 | 当前 Python ABI、三个包、`build/` 库与 RUNPATH；`tutti-env.sh status` |
| 服务启动拒绝 group/layout | 模型真实 KV cache spec；不可伪造层宽或禁用 guard |
| B 无外部读 | HBM reset、A 提交成功、namespace、盘池恢复、读计划记录 |
| `EAGAIN` 创建队列 | 每盘 ranks × 每 rank queues 与 daemon 实际可用配额 |
| 慢首轮/启动 | JIT、权重加载、预建和注册；区分首轮与稳态 |
| IO 失败/设备身份错配 | 立即停止测试，核对 PCI/namespace/字符设备，不能换编号“试一下” |
| ENOSPC/根盘写入 | `findmnt`、KV mount、每 rank 容量与预建空间；不自动删业务文件 |

详细判别方法随 [`runtime skill`](../../Skills/tutti-runtime/SKILL.md) 的 references 提供。
