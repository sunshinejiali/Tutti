# MiniMax TP8 KV 副本读放大与 GPU 直通协同恢复方案

日期：2026-10-14。

状态：**v0.2.1 后续设计提案，尚未实现、尚未验收，不属于 v0.2.0 已交付能力。** 本文不改变当前在线服务、模型计算路径或磁盘布局。

## 1. 问题与范围

当前 MiniMax-M3-MXFP8 在线配置为 TP8、EP8、主 KV FP8 E4M3、index-K BF16，最大上下文 1,048,576 tokens。Tutti 的 NVMe 队列由 GPU kernel 驱动，数据通过 DMA 直接进入各 worker 的 vLLM KV 缓存。

模型包含：

- 60 层主 K/V，4 个全局 KV heads，每 head 128 维。
- 57 层 index-K，每 token 一个 128 维 key-only 向量；不是临时 top-k 编号。
- TP8 下主 KV 的每个全局 head 由两个 rank 持有；index-K 在八个 rank 持有副本。
- EP8 分布专家权重，不消除这些 KV 副本。

当前各 rank 独立保存、恢复自己的实体，尚无跨 rank 读取合并和副本分发。需要区分两个量：

1. 模型去重后的唯一 KV 状态。
2. 当前八个 GPU 实际消费的物理副本总量。

对于完整 1M 上下文，按有效 payload 计算，不含段头、对齐、文件系统和临时内存：

| 数据 | 唯一模型状态 | 当前 TP8 物理状态 |
|---|---:|---:|
| 主 K/V，FP8 | 60 GiB | 120 GiB，复制 2 倍 |
| index-K，BF16 | 14.25 GiB | 114 GiB，复制 8 倍 |
| 合计 | 74.25 GiB | 234 GiB |

每 rank 为 15 GiB 主 KV + 14.25 GiB index-K，即 29.25 GiB。相对唯一状态，完整恢复的副本读放大约为：

\[
\frac{234}{74.25}\approx 3.15
\]

**这是当前计算布局与独立恢复方式产生的流量，不等于发现了重复提交 IO 的 bug。** 正常读取路径按块、实体页宽生成请求，extent/MDTS 拆分本身不增加有效字节。异常重投、跨请求重复恢复等额外流量仍需独立字节计数确认。

### 1.1 线上观测的含义

当前 block 为 128 tokens，每个主 KV/index 实体页均为 32 KiB；Tutti chunk 为 512 tokens，包含 4 个 block。

每 chunk、每 rank 的完整有效恢复量为：

\[
4\times117\times32\ \mathrm{KiB}=14.625\ \mathrm{MiB}
\]

例如 1,828 chunks 对应 935,936 tokens：单 rank 26.109 GiB，八 rank 约 208.875 GiB。若各 rank 读取相同 chunk 集且窗口近似同步，以 rank0 的 4.112 秒窗口换算，全机等效速率约 50.8 GiB/s。

因此近 1M 前缀的 4.1～4.6 秒读窗口与约 200 GiB 的全机恢复量相符，并不说明只读了几十 GiB 却异常缓慢。

这仍不是精确的全机带宽测量：日志窗口包含提交、排队和逐层推进；rank 窗口未必完全一致。`READ_TAIL`、GPU 利用率 100%、单 rank 窗口乘八，都不能单独证明 IO 饱和或计算完全被阻塞。GPU-direct 流量也不保证出现在普通 `iostat`/`diskstats` 中。

## 2. 设计目标与非目标

### 目标

- 对相同 KV payload 合并读取，尽量只发生一次 NVMe → HBM 传输。
- 保留现有 vLLM GPU 缓存和模型算子，不删除计算所需副本。
- 利用 GPU 互联完成分发，不引入 KV 的 D2H/H2D、CPU memcpy 或统一 DRAM KV 池。
- 不额外分配一整份 GPU KV staging；直接使用 owner 的现有 KV 页。
- 保留按实体/层的流水线，不等待整个请求的 KV 全部恢复后才计算。
- CPU 只处理有界批次的地址描述、映射、提交和生命周期，不搬运 KV 内容。

### 非目标

首版不同时引入共享 DRAM、新 attention 算法、远程 KV 直接寻址、先算 top-k 再按需读主 KV或全 GPU 常驻调度器。

统一的是**对象身份和恢复计划**，不是强制把 KV 数据放到 CPU 内存统一管理。Mooncake/LMCache 的统一存储管理可参考，但不能据此认定其当前实现已自动支持 MiniMax 的两份主 KV、八份 index 去重复用。

## 3. 推荐数据路径

```text
唯一恢复对象
    │
    │ owner GPU 的 Tutti IO kernel 提交 NVMe 命令
    │ SSD DMA
    ▼
owner 的现有 vLLM KV 页
    │
    │ 真实 IO 完成依赖
    ▼
GPU peer scatter/fanout kernel，经 NVLink
    ├── 消费 rank 的现有 KV 页
    ├── 消费 rank 的现有 KV 页
    └── ...
    │
    │ 消费者数据就绪依赖
    ▼
各 rank 原有 attention/index 算子
```

一次普通 NVMe 命令只有一个目的缓冲，不假设 SSD 能直接广播到八个目的 GPU。方案是一次直读后 GPU 分发。

本机只读拓扑检查显示八卡之间为 `NV18`，P2P read/write 均为 `OK`。这是可行性前提，不是跨进程映射成功或 fanout 带宽验收；互联也会与 TP/EP 通信竞争。

## 4. 对象、owner 与地址映射

### 4.1 对象身份与副本身份分离

唯一对象需要包含模型/版本、前缀 chunk、层/实体、全局 KV head，以及 dtype、布局、量化尺度等正确性信息。消费副本另包含 rank、目的 block、偏移和本次 generation。

- 主 KV：每个全局 head 一份唯一对象，对应两个消费者。
- index-K：一份唯一对象，对应八个消费者。
- rank → global head 的映射从实际 TP 权重/缓存布局核对，不仅凭 rank 号猜测。
- 将两个缓存称为副本之前，必须验证有效字节和量化 scale 一致；如不一致，不能共享。

**不能直接删除现有文件名/key 中的 rank 标识。** 不同 rank 可能持有不同 head，未经上述映射合并会覆盖数据；跨 TP 布局复用也需要明确格式兼容性。

### 4.2 读取负责人均衡

- 主 KV 在其同 head 消费组中选择 owner。
- index 以 chunk/批次为单位在可访问的消费 GPU 间分配 owner，避免固定 GPU0 读取并发送全部 index。
- owner 选择同时考虑盘组、KV 可用副本、拓扑和当前批次；新 rank 路由必须验证能够解析和打开选定 canonical 盘目标。
- 所有消费者协调同一份 owner 计划；不得多个 rank 各自决定并重复读。
- 若 owner 本身已有相同有效 HBM 页，可直接分发，但需验证该页的 key、generation 和保活状态。

首版可保留现有多 rank 磁盘文件，只选择一份正确副本读取。此时减少读流量但不减少落盘容量/写流量。后续唯一化落盘才引入 canonical 对象和新的持久化提交规则。

### 4.3 跨进程 GPU 映射

在 KV 绑定期一次性交换 CUDA allocation IPC handle、设备身份、allocation 大小、tensor/storage offset 和 stride，导入消费者现有缓存并保留映射表。

- 不能使用另一进程的裸 `data_ptr()` 作为有效地址。
- 多个 tensor 可能共享底层 allocation；应按 allocation 管理导出和导入，不假定每实体独占一个可导出 allocation。
- 必须验证当前 vLLM allocator 的可导出性、P2P 方向、offset 边界与释放契约；不能导出时明确拒绝该模式，不暗中引入 staging。
- 当前本地布局仍要求 64 KiB 缓存基址、4 KiB IO 对齐和已支持的 packed 布局；IPC 不放宽这些契约。

目的 GPU 的 block ID 不一定与 owner 相同。运行时下发小型、批量描述符：源 block/offset、目的 rank/block/offset、长度、epoch。描述符可以有小型复用的 host/GPU 缓冲，**禁止按 KV 数据量增长的 CPU bounce buffer**。

## 5. GPU 分发与同步

每个 owner、每层或有界波次启动一个 fused peer scatter/fanout kernel，处理该批多个块与消费者。避免每页一次 Python 调用、跨进程消息或 `cudaMemcpy`。

流水线为：

```text
层 L：     DMA → fanout → compute
层 L+1：         DMA → fanout → compute
```

消费者必须等待本实体自己的目的页就绪，不一定等待整个 fanout 批次或所有层。可以先用清晰的批次 fence，验证正确性后再细化到子批次；不承诺细粒度分发必然更快。

### 5.1 真实完成，不是仅提交完成

当前 store 有 read/read-copy 流和 compute event 依赖，可复用为本进程依赖链。但现有事件不是完整的跨进程协同恢复协议。

- 同步完成路径：在 NVMe 真正完成后才能启动 peer 读取源页。
- `TUTTI_ASYNC_COMPLETION=1`：提交 kernel 退出仅表示命令已发出，普通 read fence 不是 DMA 完成。必须在 fanout 流上接入 runtime 的真实 completion wait，而不是先在 compute 流等待或直接 scatter。
- fanout 后记录消费者可等待的跨进程事件；事件需使用支持 IPC 的创建/导入方式。
- 消费者在收到对应 generation 的事件已经 record 的控制通知后，再排入等待；防止未记录事件或事件复用导致误放行。

CUDA event 的完成只说明执行顺序，**不说明 NVMe 返回成功**。需接入成功状态检查/error gating：坏页不得被当成有效 KV 使用。正常数据链可 GPU 串联，错误仍要传播到所有消费者并 fail closed。

首版选择小型 host IPC + CUDA event IPC，不新增常驻 GPU flag 轮询。完全设备侧通知需额外验证 system-scope 内存可见性、peer atomic、generation/ABA、超时与失败传播，还可能占用 SM 干扰 GEMM，不作为第一阶段目标。

## 6. 生命周期、提交与失败契约

- owner 源页从读之前保留到 fanout 真正完成；owner 自身计算使用也不能被提前回收。
- 消费者目的页从恢复前保留到其计算完成。
- 对每批记录 key/block/generation，避免异步消息将旧数据写进已重分配页。
- 注销/关闭顺序：停止新计划，drain DMA、fanout、consumer compute，再关闭 IPC imports，最后允许释放 exporter allocation。
- 现有 runtime 的 IO inflight/注销保护不覆盖新增 peer kernel 和跨进程消费者，需要显式扩展页 lease。
- partial acceptance 必须检查 `initial_states`，只处理已接受请求并正确补投被拒请求；不能把一个 IO handle 存在当成全批成功。
- owner 失败、消费者退出、超时均不得永久等待或静默发布；初版保持失败关闭，不未经协调自行改选 owner。

唯一化写盘后，将两个门禁区分开：

1. 唯一对象所需全部 head、主 KV/index 实体均已成功持久化。
2. 本次恢复的所有消费 GPU 页均已成功就绪。

持久化提交由真实唯一对象清单判定；计算等待第二个门禁。不能只降低当前跨 rank 提交计数阈值，或把 fanout event 完成当作持久化成功。

## 7. 预计收益与代价

对于完整 1M 恢复，假定副本字节一致，且全部需要恢复：

| 阶段 | NVMe 有效读量 | 补齐副本的逻辑 GPU 分发量 |
|---|---:|---:|
| 当前独立读 | 234 GiB | 无专门 fanout |
| 仅 index 单读分发 | 134.25 GiB | 99.75 GiB |
| 再做主 KV 成对分发 | 74.25 GiB | 159.75 GiB 合计 |

GPU 分发量是逻辑副本 payload，不是含拓扑路由后的 NVLink 线缆计数；广播算法、P2P kernel 和并发决定实际开销。GPU 缓存仍保留 234 GiB 物理副本，**本方案不降低 HBM 驻留量**。

以历史全机盘探针约 50.9 GiB/s 计算，单纯 NVMe 传输下界约从 4.60 秒变为 1.46 秒。该下界不是端到端预测：owner 的盘访问并行度、peer fanout、HBM 竞争、TP/EP 通信和同步可能抵消部分收益。

统一 DRAM 只有在唯一对象身份、并发读取合并和共享驻留同时成立时才消除重复读盘；仍会产生多 GPU H2D。它不是当前 GPU-direct 去重的必要条件。

## 8. 实施顺序与验收标准

### 阶段 A：index 读取合并与 GPU 分发

保留现有磁盘布局、主 KV 独立读取和模型算子。先实现 KV allocation IPC、index owner 计划、批量 fanout、就绪/失败协议及 lease。

验收：

- 不同消费者目的 block ID、非连续块、HBM 部分命中、generation 更新下八卡恢复字节正确。
- 统计真正提交并完成的 NVMe bytes 和 fanout bytes，确认 index 不再八次恢复；日志中的 chunk 数不替代此计数。
- 用完成后的逐字节校验及同精度 deterministic 输出验证；FP8 对 BF16 的质量变化另行评价。
- host 内存只有有界描述符、IPC/lease 状态，无 KV host staging；GPU 无额外整份 KV pool。
- owner/consumer 退出、IO 错误、部分拒绝、超时、页复用与关闭均可收敛，不使用坏页，不泄漏映射或挂起。
- nsys 检查逐层 DMA/fanout/compute 依赖、实际 overlap 和 GPU 空等；不能仅以 GPU 100% 或 `READ_TAIL` 判词验收。
- 同负载多轮 clean wall 与独立 trace 分开，检查与 EP 通信竞争后的端到端收益，不只看盘读量减少。

### 阶段 B：主 KV 成对分发

验证实际 rank/head 映射及副本一致性，再把同 head 两次读取合并。重复阶段 A 的正确性、错误、生命周期和性能验证。

### 阶段 C：唯一化持久化

引入 canonical 对象目录、容量核算、owner 写入和完整对象提交/恢复规则。使用新布局版本和新池，**不直接合并现有 rank 文件或改变旧池 stride**。

### 后续独立议题：top-k 后按需读主 KV

先恢复 index，计算每次 query 的选块集合，再只恢复实际使用的主 KV，需要改变模型内部加载边界。前三个 dense 层仍需完整 KV，多 query/top-k 并集可能覆盖大量块；不能把 top-k=16 当成只需保存或读取 16 个历史 index keys。该优化不与首版副本分发同时实施。

## 9. 当前代码接入位置与缺口

| 文件 | 当前可复用能力 / 需补充内容 |
|---|---|
| `tutti/storage/tutti_nvme/layerwise_direct.py` | 逐实体注册现有 vLLM GPU tensor；`_page_requests()` 直接按 block offset 生成直读请求。需增加 canonical 读计划到 peer 目的页的映射，而非把别的进程指针直接注册。 |
| `tutti/storage/tutti_nvme/store.py` | `stream_context()`、read/read-copy event 和 compute wait；需真实完成后的 fanout、IPC fence、成功门禁与 lease。 |
| `tutti/engine/core.py` | 分层 read plan、feeder、等待与结算；需把消费者“可计算”定义扩展为其 fanout 完成。 |
| `tutti/integration/vllm/minimax.py` | eager pre-hook 等主 KV/index，post-hook 保存；可保留模型回调边界，改变回调背后的恢复实现。 |
| `tutti/integration/vllm/worker_meta.py` | 跨 rank 提交反馈；唯一化落盘后不能原样按 rank 副本计数判定完整对象。 |
| `csrc/include/tutti/storage_runtime.h` | IO submit/completion/`enqueue_wait`；保留 GPU-direct 核心，新增跨进程分发不应绕过 partial-commit 或注销保护。 |

外部模型依据是本机 vLLM 的 `vllm/models/minimax_m3/common/indexer.py`、`vllm/models/minimax_m3/nvidia/model.py` 和 `vllm/model_executor/layers/linear.py`；其版本和缓存布局变化后须重新确认上述副本关系。

**当前仓库没有完整的跨进程 KV allocation 映射、peer fanout 和协调恢复生命周期实现。本文只记录方案，不宣称已经降低线上读量、解决 overlap 或支持无 CPU 的全设备失败处理。**
