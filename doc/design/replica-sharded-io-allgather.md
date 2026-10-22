# 副本组分片 IO 与 GPU all-gather 实验

创建：2026-10-16；更新：2026-10-21。状态：**默认关闭的实验后端；owner整实体直接读最终KV页，消费者单kernel恢复远端离散页，无wave或payload staging。正常恢复通过CUDA IPC事件排设备依赖，不再host等待GPU完成；仍有peer事件已record的CPU通知门槛，尚未完全等同原Tutti纯本地编排。后台观察结果，步末排空传播迟到IO错误并保护事件/源页寿命。Python 739 passed / 3 skipped，140-chunk八卡真实NVMe多轮、GPU未完成即返回fence、预提交失败通过。2026-10-21同配置256K/99%四轮ON/OFF 8/8输出一致，稳态B 2.643s/2.852s（延迟下降7.3%）；nsys首prefill 1087ms/1360ms，读跨度502ms，较既有50GiB/s八盘探针对应367ms下界仍多约135ms。仍有CPU通知、首步和结算空档；详见§6末尾对照记录。不是 v0.2.0 默认路径。** §4 描述历史 NCCL 版本，当前实现及未验收的失败边界见 §6 的 2026-10-20 记录。Hy4 不在本机 H20 测试范围内，仅作为软件栈边界的资料对照。

## 1. 原因与目标

MiniMax M3 TP8 的主 KV 有四个全局 head，各 head 两份 GPU 副本；57 个 index-key 实体在八个 rank 复制。当前 FP8 主 KV/BF16 index 下，1,048,576 tokens 唯一有效 payload 为 74.25 GiB，独立 rank 读写为 234 GiB。GPU 上保留计算需要的副本，不代表 SSD 必须保存和恢复全部副本。

NVMe 是较慢的一段。目标是在既有 GPU→device_groups→阵列→rank 段文件部署上，让各卡分摊唯一 payload 读写，并用 GPU all-gather 补齐消费副本。不引入 CPU KV 中转，不修改 vLLM allocator、attention 算法或模型 forward 源码。

## 2. 首版范围与开关

- Connector extra config：`replica_io`，布尔值，默认 `false`。
- 首版仅 MiniMax M3、单机 TP8、DP/PP/DCP/PCP=1、NVIDIA packed NHD 主 KV及连续 index、eager、单逻辑 KV group。
- `load_mode=sync`，`TUTTI_ASYNC_COMPLETION=0`。跨步异步读计划和异步 NVMe completion 暂不支持，明确拒绝而不是静默退回。
- 原路径开关关闭时不创建通信组，不改变 namespace。
- 开启时 namespace 增加 fragment 布局版本，必须使用独立元数据根和独立 payload 目录，不能复用旧池或并发读写线上段文件。
- 模型权重使用 `phxsafetensors`，同时开启 `enable_expert_parallel` 与 `enable_ep_weight_filter`。

## 3. 稳定编排与物理放置

现有 vLLM 单调度域向 TP workers 下发 Tutti metadata。Tutti 复用各 rank 自己的 block 表，不重新分配 vLLM block；请求开始按实体计划批量交换各 rank block IDs，不假设 owner 和消费者的物理 block ID 相同，也不进行逐页 CPU 往返。物理地址按 owner/importer 实际 allocation、实体 offset、page stride 和对应 rank 的 block ID 推导。

主 KV 副本组是 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)`；index 副本组为 `(0..7)`。该关系来自当前 MiniMax QKV 分片实现，不推广到所有模型；相同 shape 不证明 payload 相同。

每个对象取 chunk key 前八字节的小端整数，模副本组成员数得到 owner。不是对临时 GPU block ID、磁盘 slot 或本次请求位置取模。相同 chunk 各层 owner 相同，各卡只读写自己负责的实体部分。

沿用各 rank 本地对象池、target ticket 和 segment 内偏移；不让一个 rank 打开其它 rank 的盘组或文件。每 rank 仍为共同的 chunk 预留完整 slot，只有 owned 实体生成有效 IO。

**首版降低有效读写字节，但尚不压缩 slot 预建空间。** 这是与旧布局兼容的实验折中，不能把减少写流量宣称为磁盘容量同比下降。没有 owned payload 的实体执行合法空批结算，标记含义为“本 rank 负责的 fragment 完整”，而不是“本 rank 有完整可独立读取的副本”。所有 rank fragment 成功后才发布 chunk；恢复时必须协同 all-gather。

## 4. 通信与完成依赖

使用 Tutti 独立 NCCL 进程组，初始化时按固定顺序创建八卡组和四个两卡组。独立进程组不免除与模型 TP/EP communicator 的跨 rank 提交次序约束。

恢复按统一的请求 key、实体、wave 排列。每波各 owner 按共同描述表读取自己负责的 chunk。普通 all-gather 输入长度一致：不足及空 owned 分片参与 padding，只有有效 payload 才进入 scatter，不读 padding，不将 padding 发布为 KV。

数据路径：

```text
本 rank owned 段文件
    ↓ GPU NVMe kernel / SSD DMA
有界连续 HBM send buffer
    ↓ 副本组 NCCL all-gather
有界连续 HBM receive buffer
    ↓ GPU 批量 scatter
原有 vLLM paged KV
    ↓ 实体 ready fence
attention / index
```

原位页与连续 collective 缓冲间有额外 HBM 搬运，KV payload 没有 D2H/H2D；错误状态仍有少量 D2H。缓冲仅为有界波次，不复制整份 1M KV池。首版同步检查真实 NVMe 完成后才能启动通信，不能把提交完成当成 DMA 完成。方法返回前需要保证 read fence 覆盖 all-gather 与 scatter；缓冲再用及注销前必须覆盖 GPU 消费完成。

IO、计划不一致、通信失败必须 fail closed；一个成员不能跳过其它成员正在执行的 collective。现有跨 rank commit/forget 路径继续负责调度门禁，但不替代 collective 的错误同步。

2026-10-19 最小实现采用 `async_op=True` 的 NCCL `all_gather_into_tensor`，在提交它的 read stream 上调用 `Work.wait()` 建立接收完成依赖，再执行本地 scatter。NCCL 的CPU调用返回不等于GPU数据完成；同一rank只需等待自己接收缓冲和scatter完成，不需要额外CPU进程通知，也不要求所有peer的CPU已经返回。

为先跑通实验，`requires_ordered_collectives` 后端不启动跨层 feeder，后续实体在主线程消费回调中按计划恢复；进入恢复前同步本地compute stream，全部波次结束后同步read stream，再返回并记录既有ready event。此版明确串行化模型通信与恢复通信，不宣称保持原有overlap。不同stream上的后续操作必须等待相应event；当前同步排空不是未来流水线优化的最终方案。

不新增共享主机控制区、CUDA IPC地址协议或CPU完成广播。共同计划校验和波次IO失败同步仍通过NCCL完成。当前预collective准入失败、rank退出、不同rank pin结果不一致的故障路径仍须专门验证；成功冒烟不能代替这些边界。

## 5. 非侵入性边界

改动在 Tutti 的配置、direct backend 和离线 driver/wrapper。保持 `register_kv_caches`、connector metadata、`start_load_kv`、`wait_for_layer_load`、`save_kv_layer` 等既有边界，MiniMax 继续使用外部 eager hook。不修改 vLLM 源码；必要 debug 日志不作为功能依赖。

首版不宣称支持任意 MLA、Hy3 或多机拓扑。后续按副本关系增加适配，不能直接把所有缓存八卡广播。

## 6. 实验与验证记录

### 软件检查

2026-10-16：副本配置/成员/namespace/不支持模式拒绝/调度侧不导入CUDA测试：12 passed。运行时隐藏 GPU，暂存位于 `/mnt/nvme4/tutti-profile/replica-layout-20261016-test`。

### 离线比较计划

同一 MiniMax checkpoint、TP8/EP8、Phoenix加载与EP过滤、FP8主KV/BF16index、block128、eager、同步完成，分别跑 `replica_io=false/true`。确定性采样保持一致；每轮扰动首token防止跨轮命中，在 A/B 间清 HBM prefix cache保留外部池。两种模式各用新 payload目录，不与在线20TB池混用。

先小上下文验证模型输出一致和真实外部命中，再增加上下文和复用比例；clean多轮墙钟与独立 nsys 分开记录。本轮实验前已保存在线恢复信息，并在无在途请求时 SIGTERM 停止自有 API，未动 daemon 或阵列；实验暂停后没有自动恢复在线服务。后续状态须重新检查，不能把本记录当成实时状态。恢复信息存于 `/mnt/nvme4/tutti-profile/replica-allgather-20261016/online-restore.json`，包含进程环境，不应提交该文件或复制完整环境到公开论文附件。

必须报告：A首次/稳态写入、B首次/稳态恢复、实际NVMe已完成字节、all-gather有效字节/padding、HBM缓冲预算、owner负载、层ready窗口、生成token一致性。不能用普通diskstats替代GPU-direct字节计数；不能将token命中比例当作计算时间比例。

2026-10-16 已完成旧路径 16K/80% 两轮基线：A-cold 为 4.435、3.931 秒，B 为 1.981、1.941 秒；B 每 rank 恢复 51 个 256-token chunks，rank0 读窗口 189.679、144.301 ms，没有 read failure。生成长度 16 tokens，TP8/EP8、Phoenix EP filter、FP8/BF16、block128、最大 context 1,048,576、HBM预算0.95、最大batch8192/seq8、每rank容量16 GiB。此数据仅作为本次开关对比基线，不是协同模式的性能结果。日志及启动参数：`/mnt/nvme4/tutti-profile/replica-allgather-20261016/baseline16k.log`、`baseline16k-launch.json`。

软件扩展检查：146 个 geometry/deploy/replica配置/scripts 测试通过，GPU 隐藏；这是当时子集结果，不代替新增代码的最终软件回归或协同 GPU 数据面验收。

旧路径 64K/95% 三轮基线已完成：A 为 12.654、12.405、12.548 秒，稳态均值 12.477 秒；B 为 2.181、2.130、2.137 秒，稳态均值 2.134 秒。参数记录 `baseline64k-launch.json` 的复用率为 95%；该历史日志 summary 仍硬编码 `B-80pct`，不能据标签认定实际复用率。每 rank 容量 64 GiB，其余精度/并行/最大 context/调度预算与 16K 基线相同。日志：`/mnt/nvme4/tutti-profile/replica-allgather-20261016/baseline64k.log`。

协同 16K 第二次实验未通过：`enabled16k-r2.log` 记录 `ALLREDUCE` 的 120 秒 NCCL watchdog 超时，未得到完整 A/B 正确性与性能对比。worker 栈显示主线程在 read/event 路径等待，feeder 在协同恢复的错误同步 collective 等待。跨 stream/communicator 与模型通信的顺序交叉是排查假设，**尚未证明唯一根因**。

2026-10-16 暂停前草稿增加 `requires_ordered_collectives`，尝试禁用该后端跨层 feeder，并在实体恢复前同步 compute stream；当时未重新进行模型验收。2026-10-19 加上显式 NCCL `Work.wait()` 和实体返回前 read stream 排空后，16K/80% 两轮已通过恢复冒烟，但仍牺牲原有读算 overlap，不证明先前超时的唯一根因或所有故障路径已解决。实验保持默认关闭。

### 2026-10-19：仅 NCCL 的同步实验

本机检查八卡空闲、两个daemon/阵列正常，无在线服务进程。默认in-place runtime扩展是缺少新工厂符号的旧构建产物；备份到本轮NVMe目录后同步已经验证的 `build/raid0-python` 扩展，`tutti-env.sh status` 完整通过。未修改vLLM源码、daemon配置、挂载或旧缓存。所有实验新建独立池。

16K/80% 同日开关对比，2轮×A/B，确定性16-token输出4/4完全一致，无读失败或NCCL watchdog超时：

| 时延 | 关闭 `replica_io` | 开启，同步NCCL |
|---|---:|---:|
| A 首轮 | 4.179 s | 4.185 s |
| A 后续轮 | 3.962 s | 3.969 s |
| B 首轮 | 2.001 s | 4.253 s |
| B 后续轮 | 1.956 s | 2.209 s |

B后续轮慢12.9%，因此只证明协同恢复跑通和字节减少，**没有端到端加速**。首轮B额外耗时尚未分解，不能直接归因于NCCL初始化。

- B每轮恢复51 chunks。全机协同 `confirmed_bytes` 为 **0.9244995 GiB**；旧完整副本布局由几何推算为 **2.9135742 GiB**，减少68.27%。旧路径本轮没有新增native字节计数，必须把推算与实测区分。
- 第一次B最后实体读统计时，全机写 `confirmed_bytes` 为 **1.16015625 GiB**，吻合A的64 chunks唯一有效payload；旧完整布局对应 **3.65625 GiB**（几何推算）。这是完成计数快照，不是整轮A/B全部写IO总量。
- 波次常驻send/receive分别为4 MiB/32 MiB，每rank36 MiB；`index_select`另产生有界临时HBM payload，36 MiB不是总峰值。
- 写/读统计已改用completion的 `confirmed_bytes`，不会把请求构造或仅受理字节当成功；`gather_bytes`是rank级名义副本补齐量（含padding），不是NVLink链路实测字节。
- 软件检查：分片/配置/脚本/页寻址子集92 passed；direct KV旧feeder与新增有序主线程分支等89 passed。子集有重叠，不累加成总测试数。
- 日志/启动参数/比较JSON：`/mnt/nvme4/tutti-profile/replica-allgather-20261019/{enabled16k,baseline16k}.log`、对应 `*-launch.json`、`comparison16k.json`。

64K/95% 同日3轮×A/B也完成，6/6请求的16-token输出与关闭开关一致，没有读失败或NCCL watchdog超时；主KV副本组的恢复跨多个波次，验证有界send/receive复用。

| 时延 | 关闭 `replica_io` | 开启，同步NCCL |
|---|---:|---:|
| A 首轮 | 12.633 s | 12.647 s |
| A 后两轮均值 | 12.4445 s | 12.4665 s |
| B 首轮 | 2.124 s | 4.552 s |
| B 后两轮均值 | 2.1475 s | 2.537 s |

每轮B恢复243 chunks：全机实际确认读 **4.4049683 GiB**，三轮相同；旧完整布局推算 **13.8823242 GiB**，符合68.27%减少。首次B读统计时A的确认写 **4.640625 GiB**，旧布局256 chunks对应 **14.625 GiB**（推算）。16K/64K合计10个请求输出匹配只构成冒烟证据，不等于全页byte-exact、数值质量或异常生命周期验收。

同步版64K稳态B慢18.1%，没有打满NVMe或加速的实测证明。逐实体同步、NCCL通信与scatter的总开销及失去跨层预取需后续分解；不能把差值直接归为某一个因素。本段为clean墙钟实验，未包含nsys；后续同日nsys对照单独记录如下。不新增共享控制区，不继续扩大模型/上下文配置。64K日志、对应启动参数、`enabled64k-io.json`、`comparison64k.json`在同一实验目录。

旧路径和协同路径均出现vLLM退出阶段宽限期后由自身process manager强制清理EngineCore的日志；部分旧路径还报resource_tracker对象清理警告。这不是新增CPU控制区，也不认定只由NCCL造成。实验结束已核对无推理进程、八卡显存归零，两个daemon与挂载正常；没有自动恢复旧在线服务。开关默认关闭，故障/重启/关闭等生产门禁仍未完成。

### 2026-10-19：nsys 对照定位同步版变慢

用户要求不以同步/失去流水线的推测代替证据。保持上述64K/95%、TP8/EP8、FP8/BF16、Phoenix加载/EP过滤、1M最大context、8192 batch、seq8等条件，各用新池采集on/off三轮A/B；不修改vLLM，不调整通信算法。只在实验后端增加opt-in NVTX阶段：`compute_sync`、`plan_check`、`read_owned`（含完整runtime等待）、`control_sync`、`all_gather`、`scatter`、`read_sync`。

Nsight Systems版本 `2025.3.2.367`；`--trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=tutti.bench@tutti --capture-range-end=none`。报告与SQLite全部放NVMe。模型GPU初始化不在capture范围内，但首次使用恢复NCCL进程组在B首轮，因此仍被捕获。两个profile模式六个请求生成token6/6一致，无read failure或watchdog超时。

统计以B的driver NVTX时间窗为界；GPU kernel按CUPTI launch correlation关联同一进程/线程的NVTX，不用kernel执行时刻是否处于host范围来猜方向。读写按真实IO launch范围分类，模型NCCL与恢复NCCL分开；全部kernel均找到launch关联。以下每GPU值是八卡、两个稳态B的区间并集平均，不是八卡累计墙钟：

| 项目 | 旧路径 OFF | 同步分片 ON |
|---|---:|---:|
| B稳态profile墙钟 | 2.442 s | 2.930 s |
| NVMe读kernel驻留 | 336.5 ms | 241.9 ms |
| 读IO与模型kernel的交叠/读IO时间 | 93.4% | 0.0% |
| 未与模型重叠的读kernel时间 | 22.0 ms | 241.9 ms |
| 新增KV状态NCCL kernel | 0 | 55.5 ms |
| KV payload all-gather kernel | 0 | 10.7 ms |
| KV scatter kernel | 0 | 9.5 ms |
| 模型侧kernel并集（含TP/EP通信） | 2121.5 ms | 2082.6 ms |
| CUDA无活动区间（扣除kernel/memcpy/memset） | 294.4 ms | 521.1 ms |

**证据支持：减少读量并不等于缩短关键路径。** 旧路径约314.5ms读kernel驻留被模型活动覆盖，暴露仅22ms；同步版先等当前compute、再读/状态同步/gather/scatter、再继续模型，241.9ms读驻留全部暴露，暴露读时间增加约220ms。新增GPU状态通信与payload/scatter约76ms；同时CUDA无活动时间平均增加约227ms。各阶段可能交叠，不能把这几个数字直接相加当精确因果分解。

模型侧平均kernel并集没有整体增加，非通信模型侧kernel亦没有变慢（约632.6→567.4ms）。单rank模型all-reduce驻留差异很大，会包含等其它rank的时间；不能把rank0单独的all-reduce增长当成模型算力下降。比如B第2轮rank0的模型活动OFF1223ms/ON2348ms，但八卡平均为OFF2112ms/ON2095ms。

**NVLink payload传输不是本轮主要耗时项。** rank0每个稳态B有117个缓存实体、177个IO波次/对应payload all-gather（60个主KV实体各2波、57个index实体各1波）。状态collective还有一次八卡计划校验。payload GPU通信约11ms，scatter约10ms；这不证明NVLink带宽打满，也不是完整恢复wall只有21ms，因为IO、CPU描述表准备、状态同步和层间串行仍存在。

B第2轮rank0的CUDA `cudaStreamSynchronize`共945次，归属为：

| host阶段 | 次数 | API区间累计 |
|---|---:|---:|
| 等前一实体模型计算 `compute_sync` | 117 | 504.9 ms |
| 每波状态检查 `control_sync` | 354 | 53.1 ms |
| 每波scatter索引准备 `scatter` | 354 | 2.5 ms |
| 每实体末尾 `read_sync` | 117 | 3.7 ms |
| 计划校验/其它 | 3 | 0.02 ms |

`control_sync`中的CPU标量→GPU构造与GPU结果`.item()`、`scatter`中两份CPU索引→GPU tensor构造，各波都触发同步。scatter host范围累计约40ms，但GPU实际scatter kernel约9.6ms，两个层级不能相加。`compute_sync`的505ms大部分是在等正在执行的模型工作，不是额外505ms空转；它的重要副作用是禁止同时预取下一实体。945次也不能乘某个固定延迟当新增总开销。

**首轮B另有可定位的一次性开销。** ON首B4.968s，rank0在layer0 `plan_check`中停1001.7ms，在layer0/wave0 `control_sync`中停1065.9ms；对应GPU kernel空档分别约1008.6ms、1065.9ms。CUDA trace在这两个恢复NCCL阶段出现大量`cuMemExportToShareableHandle`、`cuMemSetAccess`、`cuMulticastBindMem`等API，支持首次通信组/缓冲注册初始化成本的判断，而不是后续每层NVLink数据传输慢。两段合计约2.068s，后两轮消失；这是本轮首B额外开销的主要已定位来源，不声称排除所有一次性成本。

Profile让两侧都变慢；clean稳态2.1475→2.537s与profile2.442→2.930s分别报告，不能交叉相减。没有CPU采样，本轮可以证明读算串行与CUDA空档增加，但尚未将空档逐毫秒归因于GIL/OS调度/锁。

原始报告约482MiB/份：`/mnt/nvme4/tutti-profile/replica-nsys-20261019/{off,on}.nsys-rep`。同目录有 `{off,on}.sqlite`、各自`*-launch.json`/日志、`analyse.py`、`render.py`、`summary.json`、`steady-summary.json`、`stream-sync-stages.json`及 `steady-b-timeline.svg`（真实rank0 B第2轮完整时间线＋前180ms放大；OFF的额外CPU阶段未打点，空白不等于无等待）。

下一步候选而非本次改动：首次collective资源在warmup期预热；scatter索引和状态缓冲复用，减少每波CPU↔GPU同步；再以本地event依赖恢复有序双缓冲/读算重叠，单独验证模型通信次序。不能直接删除正确性门禁或把独立communicator当成无序并发许可。本轮只profiling，不改变算法，结束后八卡释放、daemon与阵列保持正常。

### 2026-10-19：第一轮流水线优化

按照用户要求优化当前实验后端，不修改vLLM、不新增跨进程共享控制区，owner、盘组、段布局及fragment namespace保持不变。

实现变更：

1. `_DirectAllLayerReadPlan`通过可选 `begin_prefetch/join_prefetch/stop_prefetch` hooks管理owned读取；普通后端仍走原feeder。实验NCCL只由主线程按相同callback/wave顺序提交，不恢复过去跨线程调用完整load的做法。
2. 后台线程只使用 `read_copy` 流读NVMe，主线程使用 `read` 流执行控制collective、payload all-gather和scatter。先测两槽（send/receive共72MiB每rank），实际交叠约42.6%后增加到四个固定wave槽（send/receive共144MiB每rank，额外workspace/索引/对齐另计），跨过主KV两波＋index的预读边界。不分配整层或1M历史staging。
3. scatter之后记录slot free event，producer等其真正完成再zero/read；read返回scatter fence完成句柄，不能用空IO句柄提前完成。请求取消停止producer，关闭发现CUDA仍在途时返回BUSY并保留registered backing，避免无期限流同步或提前释放。
4. 当前compute前缀通过event→read stream依赖，而非CPU `compute.synchronize()`；不再实体末尾read stream排空。owned预读独立于这条依赖，可和当前模型计算并行。
5. 副本NCCL两卡/八卡组在初始化阶段预热。send ticket在registration预热。每批各副本组的ownership/waves/scatter索引只构造一次并上传GPU，重复实体复用；状态GPU/pinned-host缓冲复用。
6. IO真实完成观察和slot回收用100us休眠观察间隔并检查取消/120s deadline；超时或取消不释放仍在途IO backing。初版1ms观察已暴露逐wave额外延迟，缩短后另跑独立池验证。

**仍保留的限制：** 每波NCCL失败共识仍将GPU状态拷回pinned host并等待event，确保失败时不发布页。它也可能等待已经提交的compute前缀，因此这是“NVMe预读恢复交叠＋通信/scatter异步返回”，不是已完成跨层后台NCCL全流水。不能把删除两次stream synchronize描述成消除了所有host等待。生产者线程是本进程队列，不是CPU进程间IPC或DRAM KV缓存。

clean实验均新池、Phoenix加载/EP过滤、与之前精度/TP8/context/batch保持相同：

| 配置 | 旧路径OFF | 初版同步协同 | 本轮优化 |
|---|---:|---:|---:|
| 16K/80% B首轮 | 2.001 s | 4.253 s | 2.070 s（1ms观察版） |
| 16K/80% B后续轮 | 1.956 s | 2.209 s | 2.066 s（1ms观察版） |
| 64K/95% B首轮 | 2.124 s | 4.552 s | 2.198 s（四槽） |
| 64K/95% B后两轮均值 | 2.1475 s | 2.537 s | 2.1955 s（四槽） |

64K最终四槽B三轮为2.198、2.213、2.178s，较同步协同稳态改善13.5%，本次样本仍比旧OFF慢约2.2%，**尚未证明端到端优于独立读取**，接近旧路径也不能据两轮样本宣称打平。中间两槽100us观察版稳态2.2575s、1ms观察版约2.37s；各版本不同新池/样本，不将所有差值归因于某一个旋钮。

16K四个、64K六个请求的16-token输出分别与旧路径一致。最终64K三轮每轮真实确认读4.4049683GiB，没有read failure、NCCL watchdog或generation不一致。payload减少幅度不变；slot预建空间仍未压缩。软件子集160 passed，包括scatter完成门禁、producer失败/停止、预取hooks及BUSY关闭保活；这不代替真机故障注入和全页byte-verify。

日志/参数/clean计数：`/mnt/nvme4/tutti-profile/replica-overlap-20261019/{on16k,on64k,fast64k,four64k}.log`、对应启动JSON、`clean-comparison.json`。优化版nsys与clean分开，不依据host线程并发推算。

最终四槽 `four-optimized.nsys-rep` 已采集，约326MiB，4/4个profile请求token与旧路径一致，无读失败/超时。沿用上一节launch-correlation分类脚本，全部GPU kernel找到对应launch。取第2轮B，按八卡区间并集平均：

| 指标 | 初版同步协同（前述两稳态轮平均） | 最终四槽（一个稳态轮） |
|---|---:|---:|
| 读IO与模型kernel交叠比例 | 0% | **65.9%**（各卡62.7%～68.0%） |
| 读kernel驻留 | 241.9 ms | 219.3 ms |
| 未与模型重叠的读kernel时间 | 241.9 ms | 74.7 ms |
| KV状态通信kernel | 55.5 ms | 12.2 ms |
| payload all-gather kernel | 10.7 ms | 16.9 ms |
| scatter kernel | 9.5 ms | 9.8 ms |

两槽中间版实际交叠42.6%，四槽提高到65.9%；旧独立读取仍有约93.4%交叠，剩余暴露读和控制开销是下一步优化对象。四槽payload通信时间较同步版增加，不能隐去或将其视为随机噪声；总clean改善不代表每个GPU阶段都更快。两轮/一轮profile样本只是机制验证，不是统计充分的性能排名。

rank0第2轮B的 `cudaStreamSynchronize` **945次→3次**（残留累计约0.020ms），但 `cudaEventSynchronize`仍有210次、累计约461.6ms。大部分状态门禁等待的区间包含前一实体已经提交的模型工作，不能当成461.6ms新增空转，亦不能宣称零host同步。每批scatter映射复用后，rank0 scatter host范围由同步版约40ms降到约13.2ms（GPUscatter仍约9.8ms）。四槽rank0等待ready队列约4.5ms，两槽约73ms，支持扩大固定预读窗口的方向，不证明可无限增加槽位。

Profile B首轮/后续轮分别2.515/2.559s，首轮已无之前两段秒级恢复NCCL注册空窗。clean数据仍使用2.198/2.1955s，不和profile跨口径相减。首轮通信初始化前移由初始化warmup完成，未证明消除了所有其他新shape首次成本。

原始报告：`/mnt/nvme4/tutti-profile/replica-overlap-20261019/four-optimized.nsys-rep`；SQLite同名；统计 `four-optimized.analysis.json`、`final-profile-summary.json`；启动记录 `four-optimized-launch.json`。中间两槽报告 `optimized.nsys-rep`/`two-slot-profile-summary.json`保留。分析脚本使用 `/mnt/nvme4/tutti-profile/replica-nsys-20261019/analyse.py`，未修改旧报告。

上述64K轮次到此停止；按用户后续要求另做256K/512K验证，记录如下。最终默认仍关闭。没有修改vLLM、daemon或挂载，没有恢复旧在线服务。异常清理、全页byte-verify及跨请求更复杂并发仍需独立验证，不能将成功的有限冒烟变成生产支持声明。

### 2026-10-19：长上下文与完整 feeder 小试

用户指出仅NVMe预读不等于原有完整恢复feeder，并要求256K/512K。核对代码确认：当前 `requires_ordered_collectives=True` 时，`core.py` 不启动 `tutti-direct-read-feeder`，后续实体由compute回调驱动NCCL/scatter；因此完整恢复仍在计算下发路径，不能称为已经恢复原feeder。

尝试完整feeder的独立64K准入小试：暂用 `TUTTI_REPLICA_FEEDER=1` 允许原feeder逐实体load，`begin_prefetch` 记录请求开始时的compute前缀，后续所有实体只等该固定前缀，不从feeder默认stream记录未来模型工作。此小试A正常12.614s，首B无进展。rank0栈显示 `tutti-direct-read-feeder` 在 `_report_io_failure → cudaEventSynchronize`，主线程在 `marlin_gemm → cudaLaunchKernel → pthread_rwlock_rdlock/libcuda`，owned producer在runtime poll。GPU利用率100%不能视作有效模型计算。

**固定前缀未解决跨线程/communicator的并发阻塞。** 没有足够证据断言唯一根因就是NCCL顺序、驱动锁或SM占满；`_direct_submit_lock` 还覆盖整实体恢复，与写准备/提交共享，可能放大host阻塞，不能仅删除模型依赖就保证安全。没有产生有效B或端到端对照结果，不将其标为成功恢复feeder。

先保存 `feeder-rank0-stack.txt`，再核对PID后SIGTERM只停止自有小试及子worker；八卡归零，daemon/阵列未动。小试开关、request prefix代码和专用测试随后撤销，避免保留失败模式让用户误启。日志/启动记录保存在 `/mnt/nvme4/tutti-profile/replica-long-20261019/feeder64k.log`/`feeder64k-launch.json`，该启动JSON仅是失败历史，不可据此认为当前开关有效。成功主线程路径软件子集重新检查160 passed。

长上下文保持相同MiniMax模型、TP8/EP8、Phoenix/EP过滤、FP8主KV/BF16index、block128/chunk256、1M context、HBM0.95、batch8192、seq8、maxnew16、95%复用，明确failure policy=fail；每次新KV/payload池，清HBM本地前缀。两轮每轮A/B，per-rank池capacity在256K设256GiB、512K设512GiB，容量是上限而非全量预建。这个维度从64K扩展，不混用不同模型精度或最大context。

| clean请求 | 256K旧路径OFF | 256K四wave协同ON | 512K旧路径OFF | 512K协同ON |
|---|---:|---:|---:|---:|
| A首轮 | 52.335s | 52.446s | 119.990s | 119.945s |
| A后续轮 | 52.179s | 52.169s | 119.858s | 120.008s |
| B首轮 | 5.278s | 6.136s | 11.404s | 12.476s |
| B后续轮 | 5.055s | 5.770s | 11.553s | 12.731s |

256K两轮4/4 token输出匹配，四wave模式没有读失败或watchdog，B每轮972 chunks确认读17.619873GiB；旧完整布局对应约55.529297GiB（几何推算）。稳态B慢14.1%，**扩大context没有自动消除当前编排/小wave开销**。rank0两轮读窗口为3261.4/2898.8ms，包含提交/同步/设备等待，不是纯SSD吞吐。512K两轮也4/4 token匹配，无读失败/watchdog；B每轮1945 chunks真实确认读35.257874GiB，旧完整布局对应111.115723GiB（几何推算），稳态B慢10.2%。长上下文仍减少68.27% payload，但未获得端到端加速。每rank wave容量64 chunks、144MiB总基础缓冲保持不变；长请求每实体需要更多wave，不能假设四槽足以跨层覆盖。256K稳态B的on/off nsys另行采集，避免用64K交叠比例外推512K。

长实验日志/启动/输出比较统一在 `/mnt/nvme4/tutti-profile/replica-long-20261019/`：`{off,on}{256k,512k}.log`、对应 `*-launch.json`、`comparison.json`。原数据和失败小试均保留，不改默认配置，不使用/tmp暂存。

256K稳态B专用nsys捕获完成，避免采集两个约52s冷A的全部kernel：`--capture-range=nvtx --nvtx-capture="tutti.request|B-95pct|r1@tutti" --capture-range-end=stop`，两侧同参数、新池、各执行2轮A/B。报告每份约48～53MiB，仅第2轮B窗口；不能用它作为512K的profile结果。

按八卡区间并集平均、CUPTI launch correlation分类（所有kernel关联成功）：

| 256K capture内指标 | 旧路径OFF | 四wave协同ON |
|---|---:|---:|
| NVTX捕获窗口 | 5.339s | 6.264s |
| 读IO/模型交叠比例 | 93.8% | **22.2%** |
| NVMe读kernel驻留 | 1083.0ms | 741.3ms |
| 未与模型交叠的读驻留 | 66.8ms | 576.5ms |
| 新增KV状态kernel | — | 184.9ms |
| KV payload all-gather kernel | — | 169.3ms |
| KV scatter kernel | — | 39.8ms |
| 模型kernel（含通信） | 4725.4ms | 4676.2ms |
| CUDA无活动区间 | 536.7ms | 826.5ms |

相比64K交叠65.9%，256K下降至22.2%，**支持用户指出的“完整feeder没有恢复、四wave难以跨长实体”的判断**。旧路径每rank117个整实体读kernel，协同路径651个小wave读kernel/651个all-gather/1302个状态kernel，且主线程按compute回调处理每实体，读量虽然减少但更多等待与小批下发进入关键路径。具体wave数取决于key哈希分布，不能用完全均分向上取整公式代替实测。

rank0本次有686次 `cudaEventSynchronize`、累计约1927ms，`control_sync` host范围约2103ms，仍混合等待模型前缀/通信，不能当成2.1s新增空转。all-gather本次约169ms GPU平均（含等慢rank及调度），不能沿用64K的11ms宣称256K通信可忽略；没有链路带宽/纯payload microbenchmark，不能把169ms全部归因于NVLink搬运。

**测量边界：** profile driver墙钟ON8.570s/OFF7.612s，而export中的NVTX捕获窗口为6.264/5.339s，两侧差约2.3s，capture开启/结束等开销未完全处于可分析区间；未额外定位该差异。上述阶段统计只针对capture内，不与clean5.770/5.055s混算，也不能解释driver完整延迟的全部时间。

原始文件：`on256k-steady.nsys-rep`、`off256k-steady.nsys-rep`及同名SQLite，`{on,off}-analysis.txt`、各自 `*.analysis.json`、`profile-comparison.json`、对应启动JSON，均在上述NVMe目录。512K完成clean对照但未采集trace，明确不外推其overlap。结束后核对八卡显存归零、无推理/profiler进程，daemon/阵列正常。

后续需要改变完整恢复的提交协议或减少每实体的wave数量，不能靠将 `requires_ordered_collectives` 改False解决。跨线程NCCL的阻塞尚未闭环，失败共识不能直接删除；此次未解决完整feeder。

### 2026-10-19：层级失败共识

用户要求将传输纳入 IO/event 流水，并先实施层级共识。当前改动只落在实验副本后端：每 wave 仍按共同计划提交 all-gather、`Work.wait()` 建 read stream 依赖、scatter 和 slot reuse event；本地 IO 错误累积到整个物理缓存实体末尾，共识成功才发布 ready。失败波次不能让本 rank 提前跳过 collective。MiniMax 此配置有 117 个物理缓存实体，不是此前口头估计的约 80 层；上一份 256K trace 的共识次数从 651/wave 改为 117/entity，波数及 payload 不变。

共识 D2H 的 `cudaEventSynchronize` 改为带取消/120s deadline 的 event query，只有 event ready 后才读取 pinned-host 标量。**这仍是每实体一次 host 等待，不是零同步或无限超前**；NCCL 完成可以编入 event，event 本身不能替 CPU 判断 IO 错误，也不解除共享缓冲复用和多 communicator 的跨 rank 次序约束。

没有重新打开完整跨层 NCCL feeder：上一次固定 request prefix 小试已经超时，Tutti 的 `_direct_submit_lock` 不覆盖模型 TP/EP 通信，不能凭换成 query 推断无环。保留后台 owned NVMe producer 和主线程有序恢复；更深流水需要另行设计覆盖模型通信的提交协议，不新增失败开关。

新增测试覆盖多 wave 一次共识、早/晚波失败不发布 ready、状态未就绪不读取、取消/超时。相关 Python 门禁 **167 passed**。真机复用此前 MiniMax TP8/EP8、FP8/BF16、256K/512K 两轮、95%复用及 failure-policy=fail 口径，新池、模型实验日志、JIT 缓存和后续测试临时目录放 NVMe。首轮启动因 PATH 中缺已安装的 ninja 失败，补齐 PATH 后重跑；未进入 A/B 的失败日志保留。首次 pytest 误用 /tmp 的本轮目录已清除，之后显式指定 NVMe basetemp。

两轮 clean 运行的第 2 轮 B（非 nsys）结果：

| prompt | 历史独立读取 OFF | 历史逐 wave 共识 ON | 本轮逐实体共识 ON |
|---|---:|---:|---:|
| 256K | 5.055s | 5.770s | **5.854s** |
| 512K | 11.553s | 12.731s | **12.693s** |

OFF/旧 ON 是上一轮同口径记录，没有重新测量；每种长度本轮只有两轮，不能将 1% 左右变化视为加速。256K/512K 各 4/4 输出 token IDs 与历史 OFF 一致，逐轮确认读取量分别 17.619873/35.257874 GiB；未出现 failed-closed 或 NCCL watchdog 错误。故障注入仍只有 CPU mock 门禁，不是多 rank 真机异常验收。

本轮 256K 稳态 B trace wall **6283.6ms**，rank0 read/model overlap **22.8%**，未恢复独立读路径约 94% 的交叠。SQL 核实全 rank 层共识 NVTX **936=117×8**，每 rank 从 651 次降至 117 次；rank0 `cudaEventSynchronize` 降至 35 次（包含其它路径），但 `control_sync` host 区间仍有 **1607.4ms**，event query 仍然在前向线程等待。

**结论：减少门禁次数已验证，不构成端到端优化成功。** 上次将逐 wave 门禁称为唯一元凶、预测层共识即可恢复流水的判断过强；本轮未解除逐实体 compute-prefix → read/NCCL → CPU 状态检查 → compute 的关键路径，也未恢复完整 feeder。

产物：`/mnt/nvme4/tutti-profile/replica-layer-20261019/`，包括 clean 日志/对照 JSON、`on256k-steady.nsys-rep`、SQLite、分析输出和 trace 次数/输出核验。实验已结束，八卡释放；未修改 daemon、阵列、vLLM 模型实现，未提交 git。

### 2026-10-20：方案 B 的安全修复与分级验证

用户恢复 daemon/Phoenix 后继续优化，不代加载/卸载模块或操作阵列。旧 P2P 原型的 GPU 控制块与 PyTorch 跨设备 `copy_` 已移除：

- exporter 使用独立 `cudaMalloc`，不把缓存分配器整块 storage 通过 PyTorch reduction 一发多收；importer 的 native `cudaMemcpyAsync(cudaMemcpyDefault)` 只提交到显式本地 read stream。原生调用释放 GIL。
- 控制面复用现有 rendezvous `PrefixStore`。plan/ready/ACK 带严格递增 epoch；原子 terminal 保留首个失败。KV payload 不经过 CPU，CPU 通知可有额外等待，不能称为零开销。
- 四个 send slots，单份 receive；仅拷贝有效片段、去掉 padding zero；分配前检查空闲显存并保留余量。send slot 在所有消费者已完成设备拷贝后才重用。
- 所有在途 tensor/fence 保留至设备完成；异常 sticky、停止新工作；quiescent/closing/released 握手完成后才能关闭 mapping、注销注册和释放 exporter。native tensor alias 存活时拒绝 free；不协调的 rank 退出只能 timeout 保留，不宣称 crash recovery 完成。
- 拒绝 detached read 与全层计划混用；首层初始化失败停止 producer；回调顺序、重复回调、跨 epoch 故障均有软件门禁。

隔离硬件验证：两卡/八卡 IPC 小缓冲多轮（包含 delayed peer、预提交失败、bounds/view guard、默认流工作）通过；八卡真实 Runtime/NVMe 小型恢复包含异构 4/8KiB 页、非连续 block 表、每 owner 一 chunk 的多 wave、两轮 byte-verify、第三轮预提交失败注入和协调 close，通过。该测试不加载模型或改旧池；尚未注入接受后的 NVMe kernel 卡死、rank SIGKILL 或驱动失联。

模型分级验证始终使用新池和 85% 显存预算。Phoenix 未加载时的首次模型启动失败保留；恢复后声明 1M context 与 85% 预算不相容（需要 29.25GiB KV、可用 26.98GiB），故改声明上限为 768K，精度与 TP8/EP8 不变。**不能直接与历史 95%/1M 对照比较。**

16K 首 B 曾再次阻塞，但驱动响应、无 D 状态/NVRM 异常。两次栈定位出了确定的 host 等待耦合：①背景 P2P 恢复持有 `_direct_submit_lock` 等远端，save callback 同样等这把锁；仅该独立读写后端改读专用锁。②`latest_read_event()` 为 snapshot 获取 `_state_lock`，而 feeder 长恢复也持该锁；snapshot 改用事件发布的 `_ready_cond` 短锁。两条回归均覆盖“feeder 未完成时前向写/状态查询立即推进”，其它 direct 后端仍保留原串行锁。cuBLAS `cuKernelGetFunction` 同时出现在栈中，但不能凭此认定唯一根因为 CUDA 懒加载。全模块 EAGER 尝试停在父进程 CUDA 初始化六分钟，已 SIGTERM 结束，未用于后续测量。

修复后 16K/95%、2轮×A/B，A=4.139/3.977s，B=1.663/1.669s，无 failed-closed/watchdog。每 B 全机确认读量 1,167,851,520 bytes（不是请求 wall）；该模型的输出与同配置 OFF 对照另行比较。vLLM shutdown 有 EngineCore grace-period force-kill/shared-memory warning，最终 GPU 客户端与 D 状态为空；隔离测试的协调 close 验收不能替代模型退出路径完全验收。

软件门禁 **177 passed**，CUDA transfer 原生扩展编译成功；IDE 的 torch includePath 缺失属于已有诊断，非编译错误。产物位于 `/mnt/nvme4/tutti-profile/replica-safe-20261020/`，包括失败/成功冒烟、同配置对照、线程栈及 `p2p256k-steady.nsys-rep`/SQLite。

同日两轮 clean 对照（8 卡处理单个请求，95%复用、16 token 输出、85%显存/768K声明上限；下表为第二轮 B 整请求 wall，不是 IO 时间）：

| 输入 | 原路径 OFF | P2P ON | ON 全机确认 NVMe 读取量/单 B |
|---|---:|---:|---:|
| 256K | 5.116s | 5.251s（+2.6%） | 17.619873GiB |
| 512K | 11.441s | 12.050s（+5.3%） | 35.257874GiB |

两档各 4/4 token IDs 与 OFF 完全一致，未出现 failed-closed/watchdog。OFF 的完整副本读取量按几何推算为 55.529297/111.115723GiB；未增加 native 字节计数，不能标作确认读量。模型 A 第二轮耗时分别：256K ON/OFF=52.101/52.207s；512K ON/OFF=120.007/119.907s。两轮不足以推断尾延迟或稳定均值。

256K 稳态 B 的 ON trace：wall=5680.057ms；rank0 NVMe read kernel 活跃并集=759.262ms（含 GPU轮询，不是SSD内部服务时间），八卡为693.42–759.26ms；首次读开始至末读结束跨度=2413.677ms，包含无读间隙。read/model交叠=84.5%，历史串行共识ON约22.8%（旧配置，仅定性参考）。按 launch NVTX 对应的 P2P memcpy：全机40,704,933,888 bytes，rank0 4,958,060,544 bytes/18.334ms；scatter=59.517ms。KV payload 没有 CPU bounce，不代表 metadata 零 H2D/D2H。

rank0 host 区间 `peer_wait`=715.645ms、`prefetch_wait`=578.822ms；这些后台等待可能互相/与计算重叠，不可相加到 wall。**本轮恢复了大部分读算流水和正确性，但没有净加速；单份receive每wave设备完成查询、全组ACK及CPU store通知仍是后续优化候选，不能仅凭等待区间认定唯一瓶颈。** 512K未采trace，不外推其纯IO或重叠率。所有本轮模型已结束，GPU客户端/D状态为空；没有新NVRM/Oops/Xid，未改daemon/挂载/阵列、未提交git。

### 2026-10-20：单 kernel 直达 KV 页

按用户要求继续实现无自旋的融合拷贝。每个 ready wave 一次 `replica_page_scatter_kernel`，直接从本地send或本设备打开的远端IPC地址读取，写最终非连续KV页；删除 receive 分配、逐peer memcpy、`index_select` 临时payload及 `index_copy_`。四个发送slots、epoch ready/ACK、错误门禁保持不变，ACK仍须等设备完成。

`ReplicaScatterPlan` 预构建 `(源member,槽内页,目的block)` 描述符，按副本组及页形状复用到多个物理实体；GPU寻址为 `imported_base + allocation_offset + slot*send_bytes + source_page*page_bytes`。一launch，最多256个CTA、每CTA128线程，有16字节向量路径及非对齐字节尾路径，不轮询IO ready。kernel占用SM，而旧P2P memcpy可以走拷贝路径，是否利于模型必须实测。

安全边界：源/目的/slot/member越界、重复目的页（含跨wave）、源目的重叠及launch形状检查；lease使mapping在计划存活时不能显式close。计划持有local/remote/目的storage、pinned及GPU描述符；设备工作未完成或fence记录失败保留backing，不发布ACK。quiescent后先close计划，再closemapping，最后released与exporter free。补修评审发现的deferred部分提交handle收尾、lease取得后vector分配异常、CPU描述符默认设备及正常析构设备guard。

门禁：软件 **180 passed**；最终两卡/八卡IPC（多epoch、delayed peer、失败、非连续目的页、非法slot/member/page、mapping live-plan守卫、17byte尾页/storage offset、pending close）通过；真实NVMe小型异构页/多wave/两轮byte-verify/第三轮预提交失败通过。native CUDA扩展编译成功。仍未验收rank崩溃、接受后IO kernel卡死/驱动失联。

融合版16K两轮B=1.618/1.601s；256K两轮A=52.485/52.228s、B=5.575/5.304s，均无failed-closed/watchdog。256K四个输出token IDs与当日同配置OFF完全一致，单B全机确认NVMe读取17.619873GiB。第二轮B旧memcpy/scatter版=5.251s、OFF=5.116s来自当日先前同配置测试，非本轮重新采集；融合版5.304s无净加速，约1%与上一版差异不足以判定稳定回退。85%显存、768K声明上限、TP8/EP8、95%复用、输出16token不变，使用全新池。

256K稳态B融合trace的区间wall=5635.733ms（不是带capture-stop开销的日志7.970s）。rank0融合kernel **651次/27.556ms**，八卡均651次（27.134–28.140ms）；旧路径同wave数下有1302个scatter kernels/59.517ms＋KV P2P memcpy18.334ms。融合确实减去中间搬运与launch，但read kernel仍651次，活跃756.830ms、首读到末读2411.757ms，与旧版759.262/2413.677ms几乎相同；read/model交叠86.0%（旧84.5%）。不相加多流累计时长来冒充请求收益。

**结论：实现了单wave单kernel直达，GPU恢复段缩短，但细波次、CPU ready/ACK和发送slot背压仍在，端到端未加速。** 本轮没有继续扩大512K以重复已知未获益的路径；后续批量提交/ACK流水需单独实现与验证。本轮全部模型已结束，最终GPU客户端/D状态为空、无新NVRM/Oops/Xid。根盘 `/` 可用空间降至0，停止新模型实验；本轮日志/临时/trace都指定NVMe，未清理用户根盘、未改模块/daemon/阵列、未提交git。

产物：`/mnt/nvme4/tutti-profile/replica-fused-20261020/`；报告 `fused256k-steady.nsys-rep`，SQLite/analysis/detail/trace-verification及clean比较JSON同目录，旧报告保留。

### 2026-10-20：删除 wave 和 staging，整实体直读最终页

用户明确要求恢复整层批量提交，并直接从 owner 已读出的最终 KV block 分发。当前 `TuttiReplicaDirectBackend` 已删除 64-chunk 切分、四个发送槽、发送显存注册、receive 和逐 wave 描述符。每个实体的全部 owned 页一次 `_submit_retry`，使用已有最终 memory ticket 和实际 block offset；本地 owned 页不再执行恢复拷贝。

`ReplicaIpcExport` 零拷贝导出现有 CUDA allocation，保留对应 tensor 引用，不自行释放 vLLM 显存；按 `(peer, allocation handle)` 去重 import，共享 allocation 内的实体保留独立 offset/size。不可导出的 VMM 明确 fail-closed，不通过 staging 回退。`ReplicaPeerPagePlan` 每实体一个 kernel，CPU int64 描述符为 `(peer, owner block, local destination block)`，只有元数据 H2D，没有 payload 临时缓冲。共同计划校验实体身份和逻辑 keys，各 rank 物理 block ID 可不同；owner 分组顺序、页范围、重复目的页与重叠实体检查不可省略。

保留实体级 ready 与源页释放：producer 先等待此前 compute-prefix event，再 DMA 写最终 owned 页；owned IO 真正完成才发布 ready。消费者 peer kernel 完成后 ACK，只等待该实体实际副本组，再把页交给 compute，避免最终源页被写入或复用时仍有 peer 在读取。预读可继续其它实体，不再被四槽 ACK 限制。即使 producer 已退出，未完整消费的请求中止也必须发布 terminal；stream/inflight/deferred 排空及全局 quiescent/closing/released 仍保护 IPC 关闭。

验收：原生扩展编译成功；最终两卡/八卡 IPC **2 passed**；真实 NVMe **140 chunks、三个 4/8/4KiB 实体、每 rank 不同非连续页表、两轮逐字节恢复、第三实体预提交失败和协调关闭**通过。主 KV 每 owner 70 chunks（超过旧 64 上限），全部 140 页在一个最终 ticket 提交；测试捕获所有请求的 ticket、offset、长度。Python 全套 **713 passed / 3 skipped**，跳过项另行通过硬件门禁。

小型 trace `direct-pages-140-v2.nsys-rep` 按 CUDA launch correlation 校验：八 rank 每个正常 `read_owned` 实体区间一个 NVMe kernel，每个正常 scatter 实体区间一个恢复 kernel；两轮每 rank 六个读和六个恢复，第三轮失败区间允许没有 launch，未发现实体内拆分。首份 v1 未开启 Python NVTX，仅保留原生 IO 标注；采用 v2 与 `trace-verification.json` 为逐实体证据。产物均在 `/mnt/nvme4/tutti-profile/replica-pages-20261020/`，未改 daemon、阵列、模块或旧数据。

**当前版本没有重跑模型和端到端对照，不宣称净加速或继承旧 wave 版本的 token 验证。** 根盘 `/` 可用空间仍为0，停止新模型实验；只做隔离小型门禁及 trace，全部临时/编译/日志放 NVMe。rank 崩溃、驱动失联和接受后 IO kernel 卡死仍未验收，未提交 git。

### 2026-10-20：无 staging 模型 clean 对照与 Nsight 根因

用户要求端到端验证后，重新检查daemon/两个阵列/GPU，显式把CUDA/Triton/FlashInfer/HF/vLLM缓存、日志、tmp都放NVMe；未清理根盘或变更硬件。新16K两轮4/4 token与历史一致。256K/95%复用/输出16 tokens，TP8/EP8、FP8主KV/BF16 index、block128/chunk256、max-batched8192、eager、同步IO、85%显存和768K声明上限不变；每侧四轮，全新metadata/payload池。rank0–3落 `/mnt/tutti_md0` 的四盘RAID0，rank4–7落 `/mnt/nvme4` 的四盘RAID0，确实八盘。

clean后三轮：普通Tutti OFF A=52.373s/B=5.094s；去冗余 ON A=52.352s/B=5.217s（慢123ms/2.4%）。两侧8/8输出逐token一致，ON各B确认读取17.619873GiB，无read-plan failure/recompute/drift；退出仍有vLLM EngineCore强制清理/shared_memory告警，但GPU最终释放。旧wave融合版5.304s只是历史单稳态样本，不把87ms差异报告为稳定收益。clean产物 `/mnt/nvme4/tutti-profile/replica-pages-e2e-20261020/`。

随后同参数独立两轮，只采稳态 `tutti.request|B-95pct|r1@tutti`，CUDA/NVTX/OSRT（10µs阈值），模型加载/首轮不进报告。ON/OFF四个输出一致。日志7.956/7.719s含capture启停，分析采用NVTX窗口5.672/5.427s；不能代替clean均值。每rank两侧均117个NVMe读kernel、234个写kernel和24776个模型kernel，ON另117个恢复kernel；无多读wave或额外模型重算。

| 指标 | 普通直读OFF | 去冗余ON |
|---|---:|---:|
| 全机NVMe读GiB | 55.529297（936次×1944页×32KiB，几何核算） | 17.619873（确认完成计数） |
| 全机B写GiB | 2.970703（97344页，trace请求数核算） | 0.942627（30888页，同口径核算） |
| 额外远端页分发GiB | 0 | 37.909424（确认恢复计数） |
| rank0读kernel活跃ms | 1073.417 | 311.752 |
| 八rank读kernel活跃并集ms | 1145.035 | 842.835 |
| 八rank首读至末读跨度ms | 1228.086 | 1614.291 |
| rank0读算交叠 | 95.60% | 96.88% |
| rank0新增恢复kernel活跃ms | 0 | 17.502 |
| rank0首次load范围ms | 17.065 | 356.493 |

IO kernel含GPU轮询，活跃并集不是SSD内部服务时间；全rank活跃并集也不等于每盘连续服务。去冗余确实缩短实际读驻留，但读更稀疏，结束反而更晚。八盘NVMe读kernel活跃并集已小于1s，1.6s恢复跨度不能称为纯磁盘耗时。

关键路径：rank0 first-owned范围前341.525ms没有本请求owned IO，八rank共同pre-owned窗口310.043ms；整个最大无kernel空档371.004ms包含少量descriptor H2D，不能称完全无GPU工作。首次load内仅0.801ms为 `_ensure_targets`；117次descriptor H2D为3.631MB/0.396ms，CUDA API约1.966ms，不是payload搬运，也不能解释300多ms。

为避免只凭静态代码归因，额外用外部惰性import NVTX探针（不修改生产代码、不让scheduler导入CUDA）分开 `_check_plan`、`_build_scatter_plans`、`peer_page_rows` 与原生构造。验证trace rank0：计划检查/表交换10.157ms；全117实体构建308.313ms，其中重复Python页表验证/owner映射212.824ms、117次native构造17.852ms、剩余边界检查/CPU tensor构造等77.637ms。八rank构建297.952–320.330ms。代码只存在两种owner组（本head两卡、全八卡），但按实体117次重复生成布局；先完成所有描述计划才启动producer，产生显著前向停顿。

首8192-token prefill step：OFF 2229.950ms/ON 2360.084ms。rank0模型kernel活跃（含通信等待）2095.207→1897.969ms，非模型IO暴露47.276→10.228ms，却新增GPU无活动87.333→451.372ms；这些互斥区间解释约130ms首步退化，不能将后台prefetch_wait/peer_wait相加。模型算术和通信分别观察：主要Marlin驻留缩短，但index-score几乎不变；整个request不同rank通信等待与GPU空闲会互换，不能将rank0通信驻留大降当作有效计算大幅加速。第二5120-token step约1.34s不变；此单trace decode累计额外121ms，IO已结束，未进一步确定其主机/通信归因，不能都算在盘或copy上。

ON producer每实体仍遍历全部keys/页后才过滤owned；rank0 owned范围start→launch累计352.791ms，可含Python构建、pybind parse和调度，与模型重叠。consumer背景等待prefetch/peer/source_release分别749.744/214.701/126.547ms，不可相加为request开销。按实际descriptor字节长度验证的实体分组（不是前60个全main）：两卡source-release60次33.830ms，八卡57次92.717ms；合计约49.834ms等待慢peer scatter尚未完成、28.193ms到全部peer已发布ACK的可见边界、48.520ms观察/退出。ACK本身无独立时间戳，以peer source_release入口作上界，后两项不是纯RPC耗时。本地scatter完成到进入source_release另累计58.186ms。这些门禁在本层compute fence发布前执行；本地页就绪不等于源页已可回收，不能为提速直接删除保护。

建议下一轮优先：共享同组页描述布局（每rank两种分组，不按117实体重复生成；实体只换真实base），请求级预算owned集合，避免每实体逐页重算owner；之后再测native共享内存/单调水位替换Store通知及完成查询，保留epoch/错误/源页lease。当前仅分析未改实现，不将优化候选报告为收益。

报告目录 `/mnt/nvme4/tutti-profile/replica-pages-nsys-20261020/`：`off256k-steady.nsys-rep`、`on256k-steady.nsys-rep`（主对照），`on256k-setup.nsys-rep`（额外细分验证），对应SQLite/analysis/detail/phases JSON、`setup-attribution.json`、`source-release-decomposition.json`、`comparison256k-nsys.json`及外部探针。root可用约303MB，全部新产物均NVMe；采集结束无模型GPU客户端，出现的 `sap1001` D状态不是本次模型PID，未操作该进程。未提交代码。

### 2026-10-20：页描述复用、owned预预算与99% overlap验收

用户要求继续优化并提高复用率至99%，本轮只改 `replica_direct.py` 两处已证实的CPU重复工作：请求内按有序 `tuple(members)` 生成一次CPU `(peer,srcblock,dstblock)`描述，共两组；每个实体仍独立原生地址计划、逐实体IPC size边界检查，不能把绝对地址缓存成跨实体描述。各rank最大block仅请求级扫描一次。producer启动先预检全部entries，然后请求级按组预算owned indices，各实体只生成本owner请求；实际提交前仍验证owned entry的ticket/generation，最终页ticket/stride不变。normal write owner过滤不改。begin重置索引，quiesce/close后释放索引，空owner仍发布完成metadata，不改Store ready/ACK或源页保护。

先新增重复构建复现，再实现：软件718 passed / 3 skipped；新增117实体仅两次rows构造且native117、成员顺序区分、cache-hit小source范围失败（含emptyrows）、不同请求页表更新、wholeentity owner索引及保留preflight等门禁。八卡140-chunk真实NVMe再次通过（异构实体、不同rank离散页、两轮byteverify、第三实体预提交失败和协调关闭）。无需重编native，无staging、无wave，未改模型forward/allocator或daemon/阵列。

99% workload沿用256K/TP8/EP8/八盘及上述精度/容量，chunk对齐实际1013 chunks=259328 tokens，命中98.92578125%，剩2816-token单步prefill。ON/OFF各四轮clean且8/8输出逐token一致：

| B请求 | r0 | r1 | r2 | r3 | 后三轮均值 |
|---|---:|---:|---:|---:|---:|
| 普通Tutti OFF | 3.433s | 2.867s | 2.969s | 2.861s | 2.899s |
| 优化去冗余 ON | 2.722s | 2.589s | 2.614s | 2.582s | 2.595s |

同99%对照延迟减少304ms/10.486%，A均值52.321/52.333s基本不变；ON每B完成确认读18.363098GiB，OFF页几何57.871582GiB。只有三稳态样本，不扩展成普遍结论；95%→99%改变prefill形状，不能把5.217→2.595s声称纯代码优化收益。

额外ON/OFF相同稳态B-only nsys包含外部惰性NVTX探针，不改生产代码；四个profile输出相同。profile日志5.179/5.569s含capture开销，采用NVTX窗口ON2.912s/OFF3.270s，不替代clean均值。八rank rows均2次，native计划仍117次：rank0 rows3.757ms、native13.976ms、总构建20.207ms（全rank19.865–20.875ms）；此前95%同分片工作集rank0构建308.313ms/rows212.824ms。请求首owned开始前38.898ms（之前341.525ms），首次load53.956ms。rank0每实体owned区间至launch累计194.426ms（之前352.791ms）；这些是后台边界，可含Python/pybind/调度，不能与request wall相加。

| 99% trace指标（rank0示例） | OFF | ON |
|---|---:|---:|
| 读kernel | 117 | 117 |
| 模型kernel | 22315 | 22315 |
| 写kernel | 117 | 117 |
| 额外恢复kernel | 0 | 117／16.998ms |
| NVMe读活跃 | 1110.580ms | 358.605ms |
| 首读到末读跨度 | 1153.570ms | 773.130ms |
| 2816-token prefill wall | 1387.898ms | 1052.891ms |
| prefill无模型kernel的IO-only并集 | 380.842ms | 13.914ms |
| prefill GPU无活动 | 86.611ms | 143.037ms |

全八rank ON prefill wall约1053–1066ms，OFF约1383–1391ms；ON IO-only并集约12.0–18.6ms，OFF约31.8–380.8ms。OFF其它rank等待慢rank的通信kernel驻留会覆盖IO，不能用rank4的96%交叠宣称全系统IO已被隐藏。ON read/model时间交叠95.3–96.9%；剔除已识别allreduce/NCCL等通信名称后，与其余模型kernel交叠70.6–76.3%，OFF55.5–58.4%。这些均是驻留交叠，不证明全部有用算术、更不是SSD内部时间。结论是优化ON在99%仍有较好的pipeline overlap，NVMe读大部分不单独暴露；不是完全覆盖，剩CPU空档、算术/通信资源争用及ready/ACK仍有空间。

产物 `/mnt/nvme4/tutti-profile/replica-opt99-20261020/`：`on99-clean.log`/`off99-clean.log`、`comparison99-clean.json`；`on99-profile.nsys-rep`/`off99-profile.nsys-rep`及SQLite、analysis/detail/phases；`setup99-attribution.json`、`comparison99-nsys.json`。启动环境和metadata/payload全新，缓存/tmp/log/pytest均NVMe，旧报告未覆盖。两侧仍有已知EngineCore grace-period强制清理/shared_memory告警发生于输出完成之后；最终GPU客户端和D状态均为空，根盘约300MB未清理，未提交git。

### 2026-10-20：整实体提前下发与独立完成观察（负收益实验）

用户要求“不要每实体主机确认后才提交下一实体”，本轮拆分为每rank一个producer和一个completion线程。producer仍整实体生成最终页请求，提交后立即保留completion并record该实体专属CUDA event，下一实体提交不能位于该event之前；禁止completion线程事后向read_copy尾部record旧实体fence。只在Runtime真实在途配额满时等待进度，不切wave、不新增payload staging。completion先查询该提交边界event，然后检查Runtime全handles终态及confirmed bytes，成功才发布ready；异步Runtime下event可能只代表发出命令，结果检查仍必须保留。源页ACK与compute handover契约不改。

`_submit_retry` 仅增加可选 `on_backpressure`：只对 `in-flight operation capacity exhausted` 调用，默认普通读写保持旧行为；回调query本entity尚未移交的最后handle、并poll已跟踪读写completion，防止并发写抢配额而query旧terminalhandle不驱动progress。callback不release当前聚合handle，不重复取得所有权；异常仍由既有deferred唯一drain。宽度拒绝、永久错误不当作背压。超过terminal-result预算的单entity超多切分仍可能安全失败，不宣称本轮已支持任意无界批宽。双线程首错加锁保留，start失败不join未启动对象，stop/join两个线程后才排空stream/inflight/deferred与关闭IPC。队列只存实体/完成/fence元数据。

软件先复现再修复：新增producer在任何poll之前提交三个完整实体，独立event/顺序ready；部分接受只重发rejected，背压反复零接受能推进当前handle，callback抛错保留一个deferredowner；已terminal当前handle情况下推进抢额度写completion；双线程join/完成后释放。最终723 passed / 3 skipped，八卡140-chunk异构实体/不同页表/两轮byteverify/第三实体提交前失败通过。尝试把硬件quota减到2时绑定准入按读写2×实体要求拒绝（required6），未发生数据IO；保留原quota8并用新池通过，不放松生产准入护栏。

初版先poll Runtime再event，99%四轮B为2.798/2.689/2.762/2.617s，后三轮2.689s；它已实际提前下发，八rank峰值17–40实体在GPU完成前enqueue，rank0读跨度773→478ms，但prefill IO-only14→120ms。为减少未完成时频繁Runtime query，改为先查已record的CUDAevent，再校验Runtime结果；这没有新增实际收益，最终四轮B2.720/2.625/2.860/2.741s，后三轮2.742s（此前确认版2.595s）。各版本8/8输出与此前99%基线完全一致，每B confirmed read仍18.363098GiB，无read-plan failure或recompute。A稳态52.283s基本不变。历史基线与本轮未交错、且当前三稳态波动较大，不能把差异外推到所有负载。最终检查还发现其它用户rsync于19:59:50启动，源为 `/data/home/haodedu/...`、目标远端 `/apdcephfs_qyqx/...`，累计物理读约9.8GB；与最终r2/r3及profile重叠。它未显示直接访问本轮NVMe池，但CPU/系统IO负载是污染因素，不能把147ms差异唯一归因于代码；仅只读核对，未停止该进程。

最终nsys稳态B：每rank117读/117恢复/117写、22315模型kernel不变；八rank峰值23–39完整实体已提交尚未GPU完成，111–115个“下一实体开始”早于上个read kernel结束，证明不是仅换线程。全机首读至末读跨度471.875ms，对18.363GiB的有效跨度吞吐约38.915GiB/s（不是SSD内部service测量）。rank0读跨度467.054ms，读kernel活跃416.679ms，copy17.361ms；较此前确认版773.130/358.605ms，IO间隙明显减少但单IO驻留更长。

| 99%主对照（rank0示例） | 此前逐实体确认 | 最终提前下发 |
|---|---:|---:|
| clean B稳态 | 2.595s | 2.742s |
| trace NVTX B | 2.912s | 3.199s |
| first prefill wall | 1052.891ms | 1168.777ms |
| prefill模型kernel活跃（含通信） | 892.554ms | 860.783ms |
| prefill非模型IO-only | 13.914ms | 167.448ms |
| prefillGPU无活动 | 143.037ms | 136.882ms |
| rank0 read/model交叠 | 96.88% | 60.09% |

全rank提前下发的read/model交叠53.8–60.1%，prefill IO-only167.4–198.9ms，prefill wall1168–1175ms。IO更集中在计算尚未推进/等待恢复的阶段，读窗口短不等于关键路径被隐藏；此处“全部model”包含通信等待，不能当作有效算术率。独立completion线程仍有约114ms pthread_mutex等待等控制开销，但这些线程区间与producer/compute重叠，不能加成request退化。恢复copy本身约17ms未变；当前未实证将全部差异唯一归因于HBM或SM竞争，不作该断言。**本轮实现了NVMe提前enqueue，但没有交付额外端到端加速。后续用户指出主机完成门槛破坏了feeder，专项审计确认计算fence迟交；应优先恢复event-only依赖链，不能先把限read-ahead深度当成已确定的修复。详见下节。**

产物：初版 `/mnt/nvme4/tutti-profile/replica-ahead99-20261020/`（negative clean/trace/queue证据）；最终 `/mnt/nvme4/tutti-profile/replica-ahead99-event-20261020/`，`on99-clean.log`、`on99-profile.nsys-rep`/SQLite/analysis/phases、`comparison99-clean.json`、`comparison99-nsys.json`。profile日志5.536s包含capture开销，采用NVTX窗口3.199s而非日志wall。两个版本均有已知输出完成后的vLLM shutdown强制清理/shared_memory警告，最终模型GPU客户端为空，root约87MiB，所有新临时/缓存/报告NVMe；未改daemon/阵列/模块，未提交git。

### 2026-10-20：显式同步、计算气泡与feeder迟交专项审计

用户指出kernel间只能设备event依赖，任何主机完成等待都会破坏正常流水线。此次仅只读源码、重新解析三份既有99% trace，并补采一份外部惰性NVTX＋CUDA sync回溯验证trace；未修改生产实现或同步行为。补采的4/4输出与当前实现相同；采样和回溯改变调度，不用新trace与旧trace比较性能。

**确实存在违反enqueue-only设计的host gate，但不能把所有 `cudaEventSynchronize` 都归为新读feeder。** 既有普通OFF、原去冗余逐实体确认、提前下发三份trace，每rank均33次EventSync。prefill主线程仅1次24.7µs（OFF15.0µs），没有新增读feeder/producer/completion线程EventSync。补采用 `tutti.audit.vllm_output_copy_wait` 包住 `AsyncGPUModelRunnerOutput.get_output`，全8rank每rank16次EventSync准确落在该函数；原生回溯含 `THCPEvent_synchronize`。它位于vLLM输出拷贝线程，不能把后台输出等待总时长当成本层compute被sync阻塞。剩17次为模型主线程既有同步；prefill的1次不解释100ms级气泡。profiler CUDA回溯会同时导出无版本interpose和 `_v3020` 实际API、且窗口外有额外收尾事件，统计只取真实请求窗口内versionedAPI，避免双计数。

真正阻断feeder的链路：`_complete_owned` 主机 `_wait_event(done)`循环，检查Runtime结果后才CPU ready；`_recover_batch`等待ready，再launch scatter，随后主机 `_wait_event(free)`、ACK和 `wait_acks`，最后才 `record_read_event`并返回；core `_submit_layer` 等这个返回，之后才登记 `read_ready_events`。首实体还在构造 `_DirectAllLayerReadPlan` 的主线程同步执行，首层恢复返回后才创建core feeder。因此提前的是NVMe owned提交，不是“恢复→compute”的事件依赖链；换成query而不用API synchronize，仍然是host completion gate。

关键对照：普通OFF rank0的117/117计算读fence，都在对应NVMe read kernel结束前record。提前下发去冗余全8rank每rank117/117计算fence，均在scatter GPU完成后才record；各rank“scatter完成→core record”累计262–290ms（重叠区间，不是可直接加到request的净开销）。`cudaStreamWaitEvent`虽仍存在，但被喂入的event常已完成，无法替代被延迟的host enqueue。

既有低开销trace按模型主计算stream统计>50µs空档（其他流可正在IO，不称整GPU闲置）：原确认版本rank0约145.6ms，提前下发293.2ms，全8rank293–318ms；其中rank0约292.8ms发生时下一kernel尚未发出，不是已排好的kernel在GPU等待。最明显空档：request相对484.041→610.719ms，126.677ms；下一Marlin launch直到610.694ms才出现，期间实体25/26计算fence分别602.873/609.065ms才record。新探针直接包 `_await_ready_event`，rank0主线程确实出现157.1ms等待feeder发布event，其他rank166–181ms同类等待。新探针改变时序，仅用来确认调用归属，不替代旧126.7ms低开销时间证据。

另一个已有同步问题：Runtime `query_io→progress→StripedDataPath::aggregate_completion_status_`在event query success后执行同步 `cudaMemcpy(...DeviceToHost)`取status。此前OFF也有，同步D2H不是本次新增EventSync；提前下发rank0 prefill234次约4.42ms，153次completion线程/81次主线程。CPU采样也见该聚合函数，但本次CUDA memory回溯未得到此调用栈，不假称逐次native栈确认。它可能阻塞进度/锁，应改为预分配状态metadata＋异步D2H/event观察，不能把event已完成当作同步拷贝不存在，也不能说它单独解释126ms空档。析构、错误恢复和vLLM输出兑现同步必须与正常kernel间依赖分别审查。

修复边界：应先恢复 `owned IO event→peer scatter event→compute wait_event` 的设备依赖，feeder在enqueue/record后立即发布计算fence；CPU完成/失败观察独立，不可成为逐层正常下发的触发器。源页ACK不应挡本地计算，应以独立源页lease和peer完成依赖保护释放；跨进程event必须已record且epoch匹配，不能让GPU等待尚未record的复用event（CUDA会视作已完成）。IO错误、partial acceptance和源页寿命仍需fail-closed，不能只删除所有wait。当前仅定位未实现该协议，不宣称已修复。

证据目录 `/mnt/nvme4/tutti-profile/replica-sync-audit-20261020/`：`on99-sync-audit.nsys-rep`、SQLite、`annotated-sync-attribution.json`（输出同步与主机等待直接标注）、`off-sync-audit.json`/`serial-sync-audit.json`/`ahead-sync-audit.json`（三侧原trace）、`sync-conclusion.json`、外部探针与分析器。prefill低开销原报告仍位于先前opt99/ahead99-event目录。profile完成无GPU客户端或D状态，root约85MiB；所有新缓存/tmp/报告NVMe，未改daemon/模块/阵列，未提交代码。

### 2026-10-21：恢复设备事件链与当天普通Tutti同配置验收

按用户“每轮必须nsys对比新增开销、气泡、退化及性能极限差距”的要求，本轮正常恢复不再使用 `_wait_event` 或CPU结果确认作为逐层下发条件。使用torch内置 `Event(interprocess=True)`，无需增加原生事件抽象：producer整实体提交后record本epoch独立owned事件，发布IPC handle并马上交接consumer；独立completion只负责最终IO错误/字节统计，不能触发正常ready。scatter读流排本地owned与peer owned事件依赖，launch后record并发布消费者IPC event。收到实际副本组“已record”的通知后，在读流排peer scatter事件wait，然后record本层compute fence、返回core feeder；不等peer GPU完成。把peer消费完成依赖纳入本层fence，保护本层计算可能更新的末尾共享页；事件、源tensor、native plan与mapping一直保留至全rank quiescent。仍然有record通知CPU门槛，这不是完全无主机协调，不把它报告成原Tutti纯本地调度。

只有后台结果观察与步末排空可等待终态：core join_prefetch只join提交线程，正常quiesce才join观察线程、检查迟到错误并排空。读取completion发现peer故障时mark本地poison，防止finally abort覆盖原IO错误。async submit-only模式先排真实IO completion wait，再record owned事件；恢复completion有fence时不让async空handles捷径跳过scatter。IO错误/partial accept/源页释放护栏没有删除。

验证：739 passed / 3 skipped。八卡140-chunk真实NVMe，两轮不同rank非连续页逐字节一致；第二轮故意延迟读流4e9 cycles，首实体返回时fence仍未完成，证明enqueue不等GPU；第三实体提交失败和协调排空通过。首次pending门禁失败，nsys小报告定位首次scatter `cudaLaunchKernel`懒加载被人为sleep阻塞，不是IPC等待；检查移到预热第二轮后pending通过，再修复peer错误被本地abort掩盖的结算bug。`cudaIpcOpenEventHandle`同一进程的测试报invalid context，不代表独立进程不可用，正式门禁用8个独立worker验证，未修改CUDA驱动行为。

当天新的256K/99%四轮clean：ON B2.635/2.640/2.682/2.607s，稳态2.643s；普通Tutti OFF B3.083/2.797/2.914/2.845s，稳态2.852s，延迟下降209ms/7.328%。两侧8/8输出逐token一致，ON每B确认18.363098GiB。A稳态52.374/52.315s基本不变。昨日host确认版本2.742s和组复用版2.595s仅历史参照，不能声称本轮超过所有旧版本。

ON/OFF同配置稳态B-only nsys主报告：NVTX window2.935s/3.157s；profile日志ON5.541s含capture开销，不代替clean。两侧每rank22315模型、117读、117写kernel，ON另117恢复kernel。rank0正常recover、peer_ready、source_release范围内EventQuery计数为0；后台观察仍query终态。此前请求相对484→611ms的126.7ms中途计算气泡在本轮不存在；ON计算流>50µs空档24次/196.930ms（全rank16–26次、186.7–199.5ms），旧ahead为56次/293.156ms，OFF rank0为82次/409.295ms。不能只拿rank0宣称整卡气泡全部消失：ON其它rank约187–199ms，OFF其它rank约135–215ms，首步setup和步末结算仍有新增空档。

| rank0稳态trace指标 | 普通OFF | 新事件链ON | 昨日host确认ahead（历史） |
|---|---:|---:|---:|
| 2816-token prefill wall | 1359.905ms | 1087.065ms | 1168.777ms |
| prefill IO-only并集 | 336.675ms | 35.440ms | 167.448ms |
| prefillGPU无活动 | 87.471ms | 173.433ms | 136.882ms |
| read kernel活跃 | 1116.453ms | 425.556ms | 416.679ms |
| rank0 read跨度 | 1146.655ms | 496.942ms | 467.054ms |

全rankON prefill1085–1092ms、IO-only22–52ms；OFF prefill1353–1360ms。新事件链恢复计算交叠但未消除首load59.349ms（OFF21.317ms）和两个大空档：rank0首步114.820ms、步末74.780ms。source_release记录通知117次累计195.292ms仍让host按组取得事件handle再返回，不是GPU完成等待但可以进一步减少。113/117本层fence在短scatter完成后record是自然调度时序，不能像旧代码一样仅看晚record就证明host等待；pending门禁和正常recover范围零EventQuery才是行为证据。ON read/model交叠96.2%，但通信驻留混入该口径，不代表全部有用算术。

新增事件API：rank0ON比OFF多469次EventCreate、469次RecordWithFlags、234次普通Record、1155次StreamWaitEvent；这些API累计host时间增量约6.9ms（不同线程非净request开销）。恢复kernel约23ms。所有GPU完成query在观察/结算路径，正常recovery没有新增 `cudaEventSynchronize`。双方每rankEventSync仍33次，输出线程归属见前一审计；两侧都还有每rank1次约8–10µs的StreamSync。专项全部sync/memory回溯确认栈 `Tensor.__setitem__ → dispatch_set_item → copy_ → copy_kernel_cuda → cudaStreamSynchronize`，不是IPC事件API。静态首选候选是外部vLLM `minimax_m3/common/sparse_attention.py` 的 `prefill_cu_seqlens_k[0]=0`；Python行号未捕获，不直接改外仓。Runtime已有同步status D2H未消除：ON rank0preifll234次约4.448ms（observer168/main66），不能宣称正常进度所有隐式sync均已清零，仍需专项处理。

性能极限差距：全八rank首read→末read跨度501.896ms，18.363GiB/跨度约36.587GiB/s；此前独立八盘探针50GiB/s对应367.262ms，当前多134.634ms/36.659%，有效吞吐约探针73.2%。该跨度包含GPU排队、host提交空隙和推理争用，不称SSD内部时间；并发推理不保证复现独立探针50GiB/s，不能声称还有135ms等额request收益。没有实测同形状2816-token prefill＋长前缀的纯HBM对照，本轮不编造精确端到端性能极限。

产物全部 `/mnt/nvme4/tutti-profile/replica-event99-20261021/`：`on99-clean.log`/`off99-clean.log`、`comparison99-clean.json`；`on99-profile.nsys-rep`/`off99-profile.nsys-rep`（主对照）、对应SQLite/analysis/phases、`event-sync-audit.json`/`off-sync-audit.json`、`comparison99-nsys.json`；`on99-sync.nsys-rep`（sync归属回溯，不做性能比较）、`stream-sync-callchains.json`、`event-api-overhead.json`；小型失败pending trace和通过门禁日志亦保留。两侧输出结束后的EngineCore强制清理/shared_memory告警仍出现；最终GPU客户端和D状态为空。root可用约43MiB，所有新tmp/cache/log/pytest/report均NVMe，未改daemon、阵列、模块，未提交。

### 2026-10-21：64/128KiB逻辑IO页与同形HBM基线

用户允许调整vLLM block以减少小命令成本，并要求区分约1s prefill原生计算/存储增量、讨论盘闲时服务其他等待请求。本轮支持逻辑block256/512和chunk参数；vLLM MiniMax attention实际kernel block仍128，将逻辑block拆成2/4个连续小页。`layerwise_direct.page_geometry`严格验证compact NHD/index stride与完整整数分组，按真实连续字节返回逻辑page stride；不重分配、不copy、不改vLLM kernel。scheduler下发逻辑block IDs，Tutti不做二次展开。新增driver `--chunk-tokens`、`--shared-tokens`、block512 choice；wrapper实际透传。显式共享前缀保证准确分叉，日志记录原生 `num_cached_tokens`，避免纯HBM命中多128tokens造成非同形比较。

软件753 passed / 3 skipped（最终仅fixture逻辑视图修正后未重复全套）；新增连续2/4kernel页分组、拒绝不完整页、driver/wrapper参数透传。真实八rank4小页组成逻辑页门禁通过，包含非连续块、多轮byteverify、GPU pending返回及第三实体失败。首次硬件失败是fixture仍按kernel tensor轴0填充/验证逻辑ID，改为零copy逻辑页view后通过；生产地址逻辑未再改动。

实验均256K/99%名义复用、TP8/EP8、八盘和相同FP8主KV/BF16 index：

| 模式 | block/chunk tokens | 实际命中/剩余prefill | clean B后三轮规则中的r1/r2平均 | 全机read跨度 | 跨度带宽 |
|---|---|---|---:|---:|---:|
| 32KiB旧事件链（历史） | 128/256 | 259328/2816 | 2.643s（旧四轮三稳态） | 501.896ms | 36.587GiB/s |
| 64KiB本轮 | 256/256 | 259328/2816 | 2.591s（本轮三轮两稳态） | 478.776ms | 38.354GiB/s |
| 128KiB本轮 | 512/512 | 259072/3072 | 2.695s（本轮三轮两稳态） | 673.927ms | 27.221GiB/s |
| 纯HBM准确前缀 | 128，无connector | 259328/2816 | 2.014s | 无NVMe IO | 不适用 |

64KiB与HBM各6/6请求输出逐token一致；HBM三个B均原生cached_tokens=259328。64KiB每B确认读18.363098GiB，128KiB为18.344971GiB；每rank均117读/恢复/写kernel。64KiB相对旧32KiB跨度追回23.120ms，距50GiB/s参考367.262ms仍111.514ms；不能称追回全部135ms。128KiB这轮读窗口显著变长，但命中向下多取整256tokens、chunk与owner集合改变，不能将全部端到端差异唯一归为IO页大小；物理命令仍可能被extent/RAID条带/MDTS拆开，当前报告未计真实SQE拆分数量，不能宣称128KiB总是较差。

关于“1s是不是原本如此”，本轮纯HBM确实没有任何Tutti/NVMe读写kernel，但必须按launch correlation归属prefill GPU工作，不按CPU NVTX的403ms截断：异步GPU执行跨到下一个host decode范围。两侧都1732个prefill下发模型kernel。rank0 HBM从prefill下发开始至最后该prefill GPU kernel结束795.903ms，GPU首末跨度789.092ms（模型kernel并集779.182ms，空档9.909ms）；64KiB Tutti对应1070.204/1063.679ms（并集876.282ms，空档187.397ms）。因此约0.8s本就存在，另约0.274s为本对照Tutti增量；其中约97.099ms更多模型kernel驻留、177.488ms更多模型流间隙，不能把全部称IO或CPU成本。

rank0按prefill launch归属的family：HBM/Tutti64KiB，index-score218.244/218.711ms（+0.467）、sparse-attention111.061/112.667ms（+1.606）、Marlin102.420/138.356ms（+35.936）、MoE164.562/191.462ms（+26.900）、通信138.163/168.113ms（+29.950）。数值是驻留，通信包括等peer，不能唯一证明HBM或SM竞争；HBM/Tutti物理attention kernel block都128，但逻辑block不同，完整同block归因仍须补测。HBM clean B2.014s与Tutti64KiB2.591s差577ms，包含decode/调度/写入，不等于prefill额外274ms，也不能相加到读跨度。

本轮32KiB准确前缀补测在初始化阶段失败：启动时显式创建的隔离payload目录后来已不存在，两个RAID used同时约16T降至3T。没有捕获删除动作或操作者，原因不明，不认定是谁清理；该样本没有数据IO，不作为带宽/正确性结论。后续停止模型升级，不重新创建/复用被破坏池，也未再跑128KiB同形HBM或新增调度策略。已完成的日志和nsys保留，20T空闲不是本轮agent清盘所得。环境只读检查没有变更daemon/挂载，用户取消命令后不继续启动实验。

等待请求复用空闲盘的边界：可以提前解析/pin对象、准备ticket等控制工作；payload直读只能针对已准入且持有独立稳定最终GPU blocks的请求。未分配blocks的waiting请求不能猜未来block ID。现有async接口能在调度准入后分配blocks并挂WAITING_FOR_REMOTE_KVS，但不保证能在正在执行的prefill中途为任意排队请求下发；replica backend当前还不支持detached async。因此本轮仅讨论，不新增调度。即使NVMe空闲，GPU-direct submit/poll、DMA写HBM、IPC恢复仍共享SM/HBM/NVLink、队列与控制锁；低优先级流或green context不能严格隔离这些资源，不能承诺“不影响当前prefill”。后续必须固定当前请求比较有无背景预读的kernel时长和prefill p50/p99，在事先指定误差界内无退化，同时下一请求等待下降、取消排空正确，才可启用；不能仅以盘更忙/总吞吐变高通过验收。

产物 `/mnt/nvme4/tutti-profile/replica-block-study-20261021/`：`hbm-b128-clean.log`及`hbm-b128-profile.nsys-rep`；`on-b256-clean.log`/`on-b256-profile.nsys-rep`、`on-b512-clean.log`/`on-b512-profile.nsys-rep`；对应SQLite和`*.launch-analysis.json`、`comparison-blocks.json`。分析器按launch归属避免异步边界漏计，第一次深NVTX父范围搜索截断漏掉Tutti计算，已修复并重新解析，不沿用错误15-kernel结论。报告未覆盖历史记录，未提交代码。

### 2026-10-21：64KiB独立kernel、177ms间隙归属及等待任务list边界

用户要求64KiB独立IO kernel，并追问计算延长是否冲突、IO逐层与多请求预读编排。本轮未改生产代码：独立测量仅用新隔离文件、8GPU/2×四盘RAID0、worker256、64KiB×1024请求（每kernel64MiB）、每rank256批。每批一个fused IO kernel，无模型或peer scatter。8rank测量kernel共2048（每rank排除1个warmup）；三个clean样本共读128GiB。probe验证accepted/COMPLETED，不做内容byteverify，重复128MiB backing可能受SSD内部缓存影响，不把它当完全同生产LBA/owner布局。

最初 `build/bin` 的Oct9旧产物不识别当前device-group schema，失败于runtime配置、未IO；切换与当前pybind RUNPATH相同的 `build/config-cleanup/bin` Oct12现有产物，新建目录测量，保留失败证据。普通128KiB header+256KiB slot stride读payload布局测得20.553/19.690/19.393GiB/s，均值19.879GiB/s，nsys19.313GiB/s；改为同文件大小、同64KiB请求和批宽的连续payload（metadata_zone），均值42.814GiB/s，nsys42.759GiB/s，kernel mean11.151ms。该差异说明RAID覆盖/物理放置对独立probe很重要，不能将任意64KiB请求都当50GiB/s；没有成员级计数，不断言具体只有几块盘工作。本轮同大小连续probe参考18.363GiB约需428.903ms，而此前带模型64KiB窗口478.776ms，约多49.874ms；50GiB/s参考仍是不同probe边界，不再把111/135ms全称“纯推理争用”。

按先前HBM和64KiB模型同形trace（1732个prefill launch kernel）逐name＋occurrence全量配对，rank0间隙总账：HBM9.909ms、Tutti187.397ms，额外177.488ms中176.591ms为host尚未enqueue下一模型kernel，额外已enqueue设备等待仅0.897ms。两个最大空档94.335ms和77.894ms，合计172.229ms：前者在首读准备/写目标准备窗口，load范围覆盖41.035ms、CUDA描述H2D117次总API0.606ms；后者在最后norm后到下一模型kernel之间，主线程177次flock累计30.852ms、2pread5.661ms、少量pwrite/fsync。没有该版本Python逐行标注，不能将剩余时间全部归唯一函数；当前store已与trace时不同（对象提交/槽位元数据重构），需新trace才能指定当前修补位置。CPU source candidate在Worker.wait_for_save/engine.wait_idle/完成settle/对象头或checkpoint，不以计数相似证明逐chunk manifest重复。

kernel驻留的增加则主要在IO活跃期：200个Marlin（HBM44.929/Tutti80.271ms，+35.343）、43个MoE（59.139/85.832ms，+26.693），合计+62.036ms；读结束后的280Marlin＋71MoE合计仅+0.801ms。支持IO/恢复与计算资源竞争是重要候选；不区分SM/HBM/NVLink的因果，不把同时发生当唯一根因。通信活跃期增加21.942ms、之后8.007ms（包含TP等待，单列），index/attention基本不变。177ms主要是CPU下发/结算迟滞，不能归成NVMe kernel抢SM导致已排队compute没运行。

请求调度与任务list：Runtime队列承载页IO而不是完整vLLM请求；Tutti的read计划按callback对应cache实体推进（117实体=60main+57index，不是117 transformer层）。如果40层只准备前20层，可开始算已就绪的前层，计算到未就绪层通过device event等待并让feeder持续读，不必等全部40层完成；预读“跳过”只能跳重复IO，仍须保留正确layer回调与事件等待。已enqueue到CUDA流/NVMe的命令不能指望由新请求抢占或重排，优先级选择应发生在host下发前。

用任务list让当前IO feeder读后续已准入请求的前若干层，在Tutti内部可设计，但当前不是追加list即可：必须存活跨step的request/epoch/最终目的block IDs、GPU块ownership lease与每实体事件/失败状态（磁盘generation和memory registration都不能代替GPU逻辑块寿命）。请求尚未分配最终GPU块时只能提前lookup/目标解析等metadata，不可直接payload DMA。现有vLLM async入口能让已分配块请求WAITING_FOR_REMOTE_KVS，但全请求finished_recving才回到调度；当前replica backend不支持detached async，单active epoch和步末quiesce会清理。若只服务已被scheduler交给connector的请求，可从已有入口扩Tutti；要在当前prefill预算已用尽时主动拿尚未被scheduler看见的waiting请求或支持半层完成提前重新准入，不能承诺完全不用vLLM侧改动。M3当前hook在整个attention forward前等main/index，QKV位于forward内部，不能声称已经将wait移到attention核心入口之后。

背景预读预算不应由“不固定的prefill秒数”估一个层数；采用前台缺失层优先、后台每次只下发少量完整实体、检查取消/请求切换后不再追加已失效任务，不引入wave payload staging或GPU host sync。正在运行的GPU-direct poll/DMA/peer scatter仍共享SM/HBM/NVLink；低优先级流无法保证绝对无干扰。零/近零退化必须以固定前台请求的有无背景预读nsys kernel时长与p50/p99在指定噪声界内通过，不能只看盘更忙。此轮仅核实设计，未实现跨请求list，也未声称它可以不用块管理协调。

产物：`/mnt/nvme4/tutti-profile/replica-kernel64-current-20261021/pure64.nsys-rep`（slot stride probe）、`replica-kernel64-dense-20261021/pure64.nsys-rep`（连续payload）；`replica-kernel64-20261021/gap-and-contention-summary.json`、`pure64-device-summary.json`、`summary.json`及只读分析器。当前软件套件753 passed/3 skipped。独立probe完成时无GPU进程；结束检查发现16:32:48另一用户启动Hy3 vLLM服务、16:33:22八worker出现，不属于本轮probe，未停止且未继续新模型测量。root约43MiB，全部probe准备/日志/trace/temp均NVMe，未改daemon/阵列/模块，未提交。

### 2026-10-21：已交connector请求的前台优先层预读（软件阶段）

两种probe布局都只有一个固定大文件并按slot分配，差异不是“独立小文件对大文件”。以独立64KiB×2、1个实体、128KiB header、512 slots参数为例：普通模式offset=`slot×256KiB+128KiB+ordinal×64KiB`，每槽128KiB头区后放128KiB数据；metadata_zone模式offset=`64MiB+slot×128KiB+ordinal×64KiB`，把全部头区集中到前64MiB，后面数据无头区间隔。文件大小均128MiB。前者仅访问每256KiB的后128KiB，会改变两个RAID的条带相位覆盖；后者连续扫数据区。20/43GiB/s不能证明生产大segment架构错误，生产117实体的slot payload和层偏移不同，不据单实体probe直接改盘格式或删除头区。

用户限定只处理已交connector、已有最终GPU块的IO-waiting请求。本轮首先在普通direct backend已有detached async入口上落实FIFO与层进度：Worker不再合并整条等待list为超大批，逐请求FIFO，重复metadata/尚未回报finished的请求不再pin或重读；取消待启动请求先过滤再kick，避免先启动后取消。Engine._AsyncReadPlan在跨step存活的plan中保留已提交实体，每次仅一个后台完整实体在途，正常前向不等它；每次新提交重新检查前台优先。前台首层构建pending或最后实体fence尚未登记/尚未完成时后台不提交，前台已读完但还计算的窗口才可投后台层，前台新计划到来暂停尚未下发的后台层。只后台poll/query，未增加前台CUDA同步，没有改变同步read feeder连续enqueue。vLLM仍要求全部层/全部TP rank finished_recving才恢复请求调度；未声称把读20/40层请求提前从waiting调入compute。

安全回归同时覆盖：部分submit接受后异常将同一个completion移交给异步plan，DMA终态前不回报失败finished，也不提前释放块；`read_async` completion采用保留未知观察模式，query/invalid observation不等于设备终态，不releasehandle；仅已观察所有handle COMPLETED/FAILED才允许异常关闭清理。真正设备失败可以结算，未知观察可暂停后续层并等待再次查询。abort/drain失败保留engine._async_read_plans，close在仍有后台DMA时不释放store/目标显存。线程创建或辅助计时失败在未IO时关闭detached计划；成功close之后不再调用engine.unpin。普通非async完成默认失败语义未改。

软件764 passed / 3 skipped。新增前台尾fence gate不调用synchronize、每层只提交一次跨step继续、新前台优先、取消部分加载排空、FIFO/metadata重复、未知观察不释放、线程/计时失败、close保持未完成IObacking。新增NVTX `tutti.async.layer_submit` 供后续硬件对照。

**未交付完整去冗余后台队列或前台无退化结论。** 当前 `TuttiReplicaDirectBackend.supports_async_read=False`，本轮保持该护栏，不能以普通backend的detached读当成去冗余多epoch事件/源页lease已完成。原同步event链和单活跃epoch不变。还需为去冗余后端单独验证多request事件通知隔离、源页保护和不同rank计划一致；不靠简单打开flag放行。

本轮没有执行新的模型/硬件nsys：其他用户17:28启动Hy3 TP8服务并占用全部8GPU（每卡约38GiB），agent仅只读检查，不停止其它任务；并发测量不能证明当前prefill无退化。后续需独占GPU时成对采有/无后台层预读trace，核对前台kernel驻留、气泡与prefill p50/p99，下一请求等待减少、已读层不重复IO、取消异常排空也通过才验收。软件日志 `/mnt/nvme4/tutti-profile/connector-layer-queue-20261021/safety-final2.log`，全部tmp/pytest放NVMe，未修改vLLM外仓、daemon、模块或生产文件布局，未提交。

### 2026-10-21：机会预读模型验证与用户接受约4.7%前台代价

前述GPU占用解除后继续了本轮验证。八卡64KiB六实体真实门禁通过：前台最后读fence之前后台无提交；后台每实体exact-once、不同目的页byteverify正确，取消只排空已提交实体。sleep对照仅证明编排，不能当资源不争用。进一步40实体BF16 GEMM对照，8rank计算耗时变化约-0.17%至+0.79%，只两个交替样本，不能外推MiniMax prefill。

模型采用普通direct backend、MiniMaxM3 TP8/EP8、256K/99%共享259328 tokens、block256/chunk256；同批两个请求，外部探针固定一前台同步读、一已准入IO-waiting请求。最初晚读探针误用长期活跃plan指针导致自身停止推进，保存本人py-spy栈并SIGTERM仅本人进程组；改为真实前台步开始/结束边界。v2随后发现第二请求在前台prefill之后才被准入，后台并未进入首prefill；该组不能作为填补prefill盘闲窗口验收。最终v3仅实验钩子等两请求同时到达再由正常scheduler准入（未改vLLM源码或生产配置），两侧同角色、相同prompt和事件路径。

v3稳态nsys真正验证了机会窗口：前台最后NVMe读结束后才启动后台，每rank117实体exact-once；其中16–21个后台实体在前台prefill期间执行。前台1732个模型kernel相同，GPU跨度控制约1409.8–1410.9ms，机会预读1474.7–1476.9ms，增加约65–67ms，4.567–4.724%。rank0为1409.935→1475.491ms（+4.650%）。整批两请求capture窗口6035.408→5920.317ms，缩短115.091ms（1.907%）；所有rank后台GPU最后读完成最晚时刻3067.369→2974.891ms，提前92.478ms。它不是scheduler finished_recving的精确时间，不报告为等同请求等待收益。4个请求range日志的首输出token IDs一致、每range32输出tokens；不是完整重复clean p50/p99。

用户明确接受提前下一请求导致当前prefill约4.6–4.7%增长。本次撤销刚加入而尚未验收的全面prefill禁读条件、metadata标志和额外计算尾事件；保持原已测的机会策略：前台读优先、后台每次一个完整实体、下一提交再检查新前台优先，DMA未排空不释放目标。实测上界4.724%，这是单profile对照，不承诺所有负载都低于4.7%或自动把5%设为配置。

本轮撤销后connector/storage软件528 passed / 3 skipped，诊断0；此前全套764 passed/3 skipped，继续阶段另一次全套有2个既有pipeline异步观察测试偶发失败，未修改无关测试。当前仍只有普通direct支持detached后台读，`TuttiReplicaDirectBackend.supports_async_read=False`保持护栏，不把普通模式验证当去冗余多请求已支持。前台容忍标准由零退化调整为有限代价换队列收益，后续需多轮clean和p95/p99、负载矩阵及去冗余扩展分别验收。

最终成对报告 `/mnt/nvme4/tutti-profile/connector-layer-queue-20261021/model-v3-deferred.nsys-rep` 与 `model-v3-opportunistic.nsys-rep`；`*.step-analysis.json`逐实例区分前台prefill/后续请求，`accepted-prefetch-comparison.json`保存本次接受标准、吞吐窗口和逐rank前台代价。早期v1停滞、v2到达不匹配和小kernel/GEMM报告保留，不覆盖或合并成虚假收益。本次撤销临时禁读未改变v3机会实现，无需以重复硬件测试宣称新收益；未提交。

## 7. 已知困难与调整依据

1. 非连续 KV 页已改由 kernel 直接按 owner/local 页表分发；NCCL 的连续输入限制不再成为 staging 理由。
2. 稳定 hash 不保证恰好均分：每 owner 按真实 owned 集合整实体读取，允许空 owner，不 padding 或切 wave。
3. 实体级 ready/源页 ACK 仍有 CPU 等待：先验证正确性，再测共享内存通知是否值得；不将仅提交完成当作设备完成。
4. TP/EP也用NVLink：看消费者就绪和端到端时间，不只看NVMe字节减少。
5. slot仍完整预建：容量收益需要后续专门compact布局，不能在这轮偷偷改变所有段偏移和恢复格式。
6. 当前模型副本字节及scale一致性需要实测。FP8 scale=1.0的数值质量与协同IO正确性分别验证。
7. 关闭/异常退出、空分片、多wave、非连续blocks、partial acceptance和重启恢复需要独立门禁，未完成的项必须保留在结果中。
8. 首次离线环境准备发现：现有外部 `build/minimax-m3-test/sitecustomize.py` 只允许 `minimax-m3-20261013-` 前缀的隔离 tag；非法 tag 使 Python 的 sitecustomize 加载失败而程序继续，随后落回默认 `/r<rank>/segments`，触发旧段大小与本次配置不匹配。未修改/清除旧段文件。调整为合法独立 tag，并在模型启动前验证 `_derive_isolated` 已生效；后续实验启动入口必须做此检查，不能只检查目录存在。

## 8. 必要性：解决计算副本映射到存储的放大，不是创造新 attention

### 8.1 四种“重复”不能混为一谈

| 情况 | 是否为本文目标 |
|---|---|
| 同一前缀、同一实体、同一 global KV head 的多个 TP 副本 | 是：验证字节/scale 一致后唯一持久化，并合并恢复 IO |
| 同一 prompt 的不同 KV heads 或不同层 | 否：真实不同的数据，不能删 rank/head 身份后合并 |
| 不同 DP 组的不同 prompt | 否：KV 和 index 通常不同，EP/DP 大小不能作为副本倍率 |
| 同一对象为可靠性/放置策略保留的存储副本 | 独立策略：不能当成错误冗余直接删除 |

相同前缀在不同 DP 组之间可能复用，但要求前缀、模型、精度、布局与数据等价性成立；这是额外场景，不作为本文收益的普遍依据。

GPU 计算仍需要副本时，直接把每 rank HBM 原样映射为磁盘对象会重复写入/读取。NVMe 比 GPU 互联慢，使长命中前缀更可能受存储流量限制。本文不是认为 NVLink 无代价：需要比较 NVMe 读、GPU scatter、padding、TP/EP 竞争和同步的端到端代价。

### 8.2 同一模型的不同缓存可以有不同复制范围

| 模型/缓存 | TP8、DCP1 下的逻辑 KV 结构 | 已核实的复制范围/限制 |
|---|---|---|
| MiniMax M3 主 KV | 4 个全局 KV heads | 同 head 两卡复制；不同 head 必须分开 |
| MiniMax M3 index-key | 单个共享 key head | 八卡复制；不是每次 query 的 top-k 整数结果 |
| Qwen3.8-Flash-Next QSA 主 KV | 2 个全局 KV heads，12 个 QSA 层 | 同 head 四卡复制；另有36个 GatedDeltaNet 层，state 不能按完整逐 token KV 外推 |
| Qwen3.8-Flash-Next QSA index | 单共享 key head、存在 index 压缩 | 八卡复制；大小不能套 MiniMax 的57层/BF16公式 |
| Hy3 主 KV | 8 个全局 KV heads | TP8 已无该类 head 副本；不应添加无收益的广播 |
| Hy4-preview/GLM/DeepSeek DSA | MLA 压缩主缓存及 sparse index | 用于考察软件栈适用边界，不视为多个展开 GQA heads；Hy4 不在本机H20测试范围 |

两个已确认的异构 GQA/index 架构是 MiniMax 与 Qwen QSA。没有实测 Qwen，不宣称 Tutti 已支持其混合布局。Hy4 的配置 `num_key_value_heads=8` 不表示实际保存八个展开 KV heads：其代码使用 MLA 压缩 latent+RoPE cache。硬件型号影响吞吐和 backend 可用性，不改变复制身份本身。

## 9. 上层软件栈：DCP 已解决什么，未覆盖什么

### 9.1 DCP 是计算与缓存联合分片

常规 TP 下，设 TP 大小为 P、全局 KV heads 为 H；在 P>H、可整除且 backend 支持的情况下，每个 head 有 P/H 个副本。DCP 在这些 rank 中沿 context/token 轴分片，分片度 D 后剩余等价副本约为 P/(H·D)。P<=H 时常规 GQA 已按不同 heads 切分，没有本文的复制放大。

例如 TP8/H4 的主 KV 用 DCP2 可消除成对副本；共享单head MLA 在完整支持时用 DCP8 可消除八份副本。DCP 降低 HBM 常驻量，存储正确映射其真实分片后，写/读有效 payload 也减少。这优先于存储恢复后再次补齐已被 DCP 消除的副本。

DCP 不只是 key 或对象编号：同时改变 token 到本地 page/slot 的映射、attention 的局部输出和合并、index 的评分/top-k，以及跨 rank 通信。cache group 主要描述缓存布局/分配；它不是自动得到一套可独立运行的 context-parallel 通信算法。

### 9.2 当前上游并非“所有 sparse attention 都不支持 DCP”

截至2026-10-16、vLLM上游 `074c83a09a0b872cec92e244d91adf2b3c55e41b`：

- `KVCacheSpec.dcp_sharded` 可描述某缓存是否参与 DCP，`max_tp_shards` 描述不同TP数据分片数。这是现有工作，不应声称上层完全不知道复制关系。
- worker仍使用模型的全局 `decode_context_parallel_size` 与 `get_dcp_group()`；没有本文需要的“主KV DCP2、同模型index独立DCP8”完整配置/执行协议。不是理论不能分组，而是异构分组必须有匹配的算法、page映射和同步支持。
- 当前MiniMax自定义 sparse 路径没有完整DCP执行支持；Qwen QSA明确 `supports_dcp=False`。不能删除准入检查强行开启，也不能仅凭MLA cache spec假定indexer能分布式评分。
- FlashMLA sparse 已声明 `supports_dcp=True`，共享 sparse indexer 已实现分片评分及全局top-k合并。这一体系当前DCP要求 `fp8_ds_mla`，BF16 sparse路径被拒绝。Hy4使用相关基础及attention-sink扩展，但未验证其特定组合；不运行Hy4实验。

因此，文章的边界不是“DCP无效”，而是“已有DCP优先使用；模型/backend或异构实体分组未覆盖时，存储层可在不改模型计算的前提下降低IO放大”。本机用于实验的fork `897ff4f39` 与最新main不同，两者能力不能混写。

### 9.3 MiniMax 容量推算：条件值而非已跑通 DCP

按1,048,576 tokens、主KV FP8/index BF16、无元数据/对齐开销：

| 状态 | 主KV | index | 总payload | 相对唯一状态的冗余 |
|---|---:|---:|---:|---:|
| TP8/DCP1完整副本 | 120 GiB | 114 GiB | 234 GiB | 159.75 GiB |
| 假设只有主KV正确DCP2，index仍全复制 | 60 GiB | 114 GiB | 174 GiB | 99.75 GiB |
| 假设主KV/index都正确参与同一DCP2 | 60 GiB | 57 GiB | 117 GiB | 42.75 GiB |
| 所有唯一对象只持久化一份 | 60 GiB | 14.25 GiB | 74.25 GiB | 0 |

最后一行是磁盘唯一payload，不自动意味着模型HBM能只保留同量。完整DCP2是理论条件；当前MiniMax不能直接启用它验收。若保持主KV分组，index在不同head组间仍有副本；需要独立的分布式index算法才能从计算侧完全消除，connector不改算法则走一次恢复后分发。

## 10. LMCache 与 Mooncake：已有去重能力必须作为对照

结论先行：**二者不是天然完全解决，也不是都完全没有解决。存储唯一化、消费端读取合并与计算HBM分片是三个不同能力。** 容量节省不能代替实际磁盘/网络读字节测试。

### 10.1 LMCache

核对 `LMCache/LMCache` 的 `dev` 提交 `1ed386c4988b9190161daed9ec54ac690a41119e`：

- 通用 `CacheEngineKey` 包含模型、`world_size`、`worker_id`、chunk hash、dtype；rank不同通常生成不同身份，以免合并真正不同的head数据。
- 已有 `save_only_first_rank`，在该代码路径受 `metadata.use_mla` 限制，默认值从该模型级标记推导。普通保存时非首rank直接跳过；普通恢复时首rank加载，随后通过GPU广播把数据送到其它rank。
- 代码在广播后保留首rank的HBM对象，避免首rank随后再读同一CPU L1缓冲。这证明“读一次后GPU分发”并非本文新概念。
- 源码仍有 `TODO: support save_only_first_rank when use layerwise`，不能把非layerwise MLA优化宣称为所有逐层路径已覆盖。
- 这是模型级MLA特例；不能因为MiniMax辅助index的spec是MLA，就把整个GQA主KV+index模型切成首rank保存。未验证该版本能按实体同时处理2卡和8卡副本组，也未运行MiniMax/QSA兼容实验。
- LMCache有不同存储与传输后端；不能把所有路径都描述为CPU bounce。本文区别是Tutti指定的GPU kernel驱动NVMe、无CPU KV中转以及多owner波次恢复，而不是笼统的“LMCache一定经过CPU”。

### 10.2 MooncakeStoreConnector 与 Mooncake Store

必须区分 vLLM `MooncakeConnector` 的TransferEngine传输，与 `MooncakeStoreConnector` 的对象存储适配。核对最新vLLM `074c83a09a0b` 的 `mooncake/store/worker.py`：

- 已按cache group计算复制因子：DCP>1或含Mamba状态时保守取1；纯MLA packed group取TP大小；普通GQA取 `max(1, TP/总KVheads)`。
- 对已知副本，把key中的TP shard编号规范化为 `tp_rank // replication_factor`；不同head仍有不同key。按 `put_step` 与rank/group相位分摊chunk PUT，不是固定GPU0写全部。
- 因而纯MLA和常规GQA的**重复写入/对象容量**已有显式优化，不能声称Mooncake总按rank重复存相同KV。
- 接收路径仍由各消费worker对自己的GPU目的地址调用 `batch_get_into_multi_buffers`；这里没有“一个rank GET再GPU广播”的协同恢复。底层DRAM命中、NVMe回填/读取合并等由Store和后端决定，本次未证明各GET必然重复读盘，也未证明已合并成一次读盘。
- packed group同时包含主GQA和辅助MLA index时，公共复制因子不能自动等于每个实体的复制因子；例如MiniMax主KV2份/index8份，按共同factor2处理不等于index已唯一化。若能分成独立纯index group，其对象去重机会不同。必须核对实际注册、打包边界和模型callback覆盖，不能从复制因子函数直接宣称端到端MiniMax已支持。
- 上游也有不同store-TP拓扑布局路径，本段描述的是常规group复制因子路径，不以有限检索否定所有后端和配置。

另核对 `kvcache-ai/Mooncake` 提交 `c3aea6550e3d5f1fa302d84f1a3385d50f5ffc99`：master检查同tenant/key的metadata并返回 `OBJECT_ALREADY_EXISTS`。这是按对象身份检查，不是对不同key内容自动hash去重。存储可靠性/放置副本与模型TP副本不同；并发PUT幂等性也不保证所有消费GET只有一次数据传输。

### 10.3 对本文贡献范围的约束

| 已有路径 | 唯一化写入/对象身份 | 消费端一次读后分发 | 计算HBM去重 |
|---|---|---|---|
| vLLM可用的完整DCP | 按实际分片存取可实现 | 按其计算通信协议处理，不需恢复完整副本 | 是 |
| LMCache普通MLA首rank路径 | 已有 | 已有首rank加载后广播 | 否，计算仍有副本 |
| MooncakeStore常规副本group路径 | 已有规范化key与分摊PUT | 各worker仍GET；底层磁盘合并未核实 | 否 |
| 当前Tutti旧路径 | 未跨rank唯一化 | 各rank独立直读 | 否 |
| Tutti本文实验目标 | 各实体副本组唯一writer | 多owner直读＋组内all-gather/scatter | 否 |

不能把“canonical key”或“读取后广播”宣称为全新能力。需要验证的区别是：**异构主KV/index副本组、拓扑/盘组均衡的多owner、GPU-direct NVMe 与模型通信共存、有限缓冲及非侵入生命周期协议**。这些仍是研究目标，不是已证明优于LMCache/Mooncake的结果。

## 11. Tutti 需要做什么，以及不该做什么

1. 明确每个缓存实体的逻辑身份、真实head分片和字节/scale一致性，不把所有rank都合并，不把不同DP请求当副本。
2. 若上层DCP已去重，按DCP物理页/slot映射保存其实际分片；不能再广播成旧完整布局。该适配独立验证，当前Tutti尚无通用DCP支持。
3. 对残余副本按实体建立不同副本组，唯一持久化对象与消费rank分开；复用已有block表和GPU→盘组映射，不修改vLLM allocator/model forward。
4. 让各owner读写本盘组的稳定分片，使NVMe并行度保持足够大；读取一次后在有界HBM波次上完成all-gather/scatter，不增加DRAM KV池。
5. 相同hash模owner只有统计均衡，不保证任意短batch/盘组均匀；必须测每owner和每阵列有效字节、在途深度，避免去重后IO工作集过小反而打不满盘。
6. 发布fragment完整门禁与消费页就绪门禁分别维护；零owned分片仍参与collective，IO失败、重复generation、页复用和关闭必须一致收敛。
7. 解决恢复collective和模型TP/EP的跨stream顺序问题后再恢复流水线；同步整卡或禁用feeder只能用作受限诊断，不能作为overlap性能成果。
8. compact持久化布局在正确性通过后另做；首版slot仍完整预建，不报告磁盘容量同比下降。

非目标：修改vLLM/attention算法、强行开启不支持的DCP、任意不同TP格式转换、跨DP全组KV广播、所有模型兼容，以及Hy4在H20上的测试。Hy4/DSA只作原生DCP已有能力对照。

## 12. 后续验收与证据来源

### 12.1 必须分别证明的事项

- **正确性**：副本组有效字节和scale一致；所有消费者恢复到原block页；同精度确定性token输出一致；失败/空分片/重启恢复不发布坏页。
- **写流量**：统计实际成功IO payload，而不是仅累计构造的请求。唯一writer应减少有效写字节；slot容量另报。
- **读流量**：统计NVMe真正完成bytes，区分all-gather有效bytes、padding和HBM scatter。多个客户端GET不等于必然多个NVMe读。
- **端到端性能**：同模型/精度/EP/调度/缓存命中条件的开关对比；独立clean多轮墙钟与nsys报告，包含TP/EP干扰和消费者就绪窗口。
- **与已有方案比较**：DCP可用时必须列原生DCP；LMCache比较对应的MLA首rank和layerwise配置；Mooncake比较实际DRAM/SSD命中层与group布局。无法兼容的模型应标注缺失，不编造基线。
- **限制与停止条件**：若去重节省被额外同步抵消，或correctness/communicator顺序未通过，保持默认关闭，不以理论带宽比代替实验。

### 12.2 固定版本参考（源码阅读，不代表运行认证）

- [MiniMax模型配置](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)、[Qwen QSA配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)、[QSA说明](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md)。模型main配置可变化，本次参数为2026-10-16读取值。
- vLLM `074c83a09a0b`：[缓存spec的DCP属性](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/v1/kv_cache_interface.py#L512)、[全局DCP组](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/distributed/parallel_state.py)、[QSA不支持DCP](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/models/qwen4_exp/nvidia/qsa.py#L212)、[DSA支持DCP及格式限制](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/v1/attention/backends/mla/flashmla_sparse.py#L710)、[共享index分布式top-k](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/model_executor/layers/sparse_attn_indexer.py#L1040)。
- LMCache `1ed386c4988b`：[key身份](https://github.com/LMCache/LMCache/blob/1ed386c4988b9190161daed9ec54ac690a41119e/lmcache/utils.py#L418)、[MLA首rank限制及layerwise TODO](https://github.com/LMCache/LMCache/blob/1ed386c4988b9190161daed9ec54ac690a41119e/lmcache/v1/cache_engine.py#L116)、[被动rank不保存](https://github.com/LMCache/LMCache/blob/1ed386c4988b9190161daed9ec54ac690a41119e/lmcache/v1/cache_engine.py#L424)、[读取广播](https://github.com/LMCache/LMCache/blob/1ed386c4988b9190161daed9ec54ac690a41119e/lmcache/v1/cache_engine.py#L1845)、[模型级use_mla](https://github.com/LMCache/LMCache/blob/1ed386c4988b9190161daed9ec54ac690a41119e/lmcache/integration/vllm/vllm_service_factory.py#L94)。
- vLLM MooncakeStore：[canonical TP shard key](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1625)、[group复制因子](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1675)、[分摊PUT](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L789)、[消费端GET](https://github.com/vllm-project/vllm/blob/074c83a09a0b872cec92e244d91adf2b3c55e41b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1261)。
- Mooncake `c3aea6550e3d`：[master同key检查](https://github.com/kvcache-ai/Mooncake/blob/c3aea6550e3d5f1fa302d84f1a3385d50f5ffc99/mooncake-store/src/master_service.cpp#L5368)。可靠性复制不纳入模型冗余收益，本轮未跑第三方后端。
