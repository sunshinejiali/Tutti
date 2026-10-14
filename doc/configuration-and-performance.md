# 配置与性能：从 vLLM 到控制器

本指南回答：选择哪条路径、每层模块做什么、参数在哪里写、哪些应自动推导、如何找性能瓶颈。
这里描述当前实现，不承诺同一参数在不同模型/TP/GPU/SSD 上都最优。
文件入口见 [config/README.md](../config/README.md)，部署/验收见
[快速开始](getting-started.md)、[vLLM 手册](intergration/tutti-vllm-manual.md)。

## 1. 一次 KV 读写经过哪些模块

```text
vLLM 模型 / HBM KVCacheConfig（层数、KV dtype、block 几何）
  → TuttiConnectorV1 / WorkerImpl：key、请求区间、TP 汇报、逐层回调
  → SchedulerMetadataIndex / ChunkIndex：resident/pending/pin、LRU、全 rank 提交门禁
  → KVEngine：读写计划、逐层 feeder、准入与完成
  → TuttiKVStore / TuttiDirectBackend：注册 HBM 池、block table→字节 IO、票据缓存
  → ObjectLayout / native ObjectStore：槽位/段文件、对象头、bitmap、checkpoint、恢复
  → StorageRuntime：Resolver 与 DataPath 路由、受理状态、引用/终态生命周期
  → LocalFile/MultiMount/md Resolver：文件 extent→控制器/namespace/LBA
  → LocalNvmeDataPath 或 StripedDataPath：PRP、队列、提交 kernel、CQ 完成
  → libnvm / snvme：控制器身份、队列内存、peer DMA、内核权威 queue depth/MDTS
```

旁路：`tutti_daemon` 控制设备 ownership、挂载/阵列、ACL、Resource grant 和 lease。
它不处理 KV key、LRU、模型层数，也不搬运每个请求的 KV。

- Scheduler 热路径以内存为权威；worker 成功提交增量汇报，不每次命中查询扫盘。
- 生产 `tutti_nvme` 直连路径注册完整 CUDA KV 池，按页/层地址读写，不经过 CPU staging。
  layout、对齐或容量准入不满足时显式失败，不静默“换个慢路径”。
- gather/scatter、RingWindow 和 staged 用例仍支持其它后端/测试能力；不是当前生产故障回退。
- direct 读 feeder 是 Python 主机线程逐层提交；首层先提交，余层提前起跑。它不是常驻 GPU
  IO service，也没有 `TUTTI_USE_STEP_FEEDER` / `TUTTI_FEEDER_WINDOW_OPS` 调参入口。

对应测试：[测试架构表](../tests/README.md)。不要用某层的 fake 通过代替另一层的 DMA 验收。

## 2. 单盘、多盘与 RAID0：区别只在哪几层

| 部署 | 对象放置 | Resolver | 数据面 |
|---|---|---|---|
| 单独 NVMe mount | 显式 `options.mounts` 或默认 root 下的段文件；local preset 的 device.mount_path 不自动用于对象放置 | LocalFileResolver | LocalNvmeDataPath，Python `preset.type=local` |
| 多个独立 NVMe mounts | 整槽位在 mounts 轮转，长 prompt 多个槽位并行 | MultiMountLocalFileResolver 分派 | 显式`preset.type=nvme-device-group`，多控制器数据面 |
| md RAID0 一个 mount（默认） | 一个文件系统内的段文件 | md resolver核对完整namespace成员并映射extent | `preset.type=raid0`，复用多控制器数据面 |

用户配置不再用`striped`。Python默认`raid0`：至少两个namespace且必须共用同一个
真实阵列mount；native resolver核对md级别、完整成员和offset，失败不降级。原生YAML用
`resolver.config.topology=raid0`强制该约束，数据面仍为`nvme-device-group`。
md stripe是阵列超级块物理事实，不能用connector参数覆盖，不把共享目录视作合法md。
同一runtime选择多个md mounts、或md与独立盘混合时，当前不能按mount聚合成员并分别
创建md resolver。应让不同rank/runtime选择各自独立的完整阵列组，不把多个阵列成员
全集塞进同一个runtime。

多个 daemon 控制器集合必须互不重叠。Python 合并多份 daemon YAML 时 `device_id` 跨文件
唯一；`DEVICE_GROUPS="0,1;2,3"` 在 TP4 下让 rank0–1 使用第一组、rank2–3 使用第二组。
选择组不建立 P2P 可达性，也不修改 ACL。

## 3. 两种配置装配面不是同一份配置

### 原生 C++ Runtime YAML

推荐`config/storage_runtime/raid0.yaml`，单盘/独立多盘分别用`local_nvme.yaml`、
`nvme_device_group.yaml` → parser →
`TuttiRuntimeSpec` → Resource allocation → Resolver/DataPath factory → Runtime。
`storage.backends[]` 用 ID 连接 resource/resolver/datapath；实际设备路径、MDTS、队列 grant
来自 daemon 返回的资源。仅改 datapath 参数不会改变挂载/设备 identity。

### Python connector preset

`config/integration/vllm/kv_transfer.json` 或 serve 生成的 JSON → geometry/factory →
store options → preset → native preset factory。
Python 读 daemon YAML 推导 `device_id` 的 PCI/mount/namespace，再构造 native runtime。
**preset 路径不是原生 Resource RPC 装配图**：它通过 libnvm 创建用户队列，不能宣称
`queue_pool` 的 RPC 策略已替它完成所有 rank 总预算检查。以实际内核队列容量/可用 QID 和
构建结果核验；daemon YAML 的 ACL/设备映射仍须人工审核。

不能将 native `datapaths[].config` 粘进 Python preset；后者对未知键会拒绝。
`TUTTI_NVME_PRESET` 仅是无显式 runtime/preset 时的后备装配入口，不读取原生Runtime YAML。

## 4. 路径的职责与隔离

| 路径 | 所有者/内容 | 不能误解成什么 |
|---|---|---|
| daemon `backing_mount_path` | 实际独立盘或 md 文件系统 | 不等于 GPU index 或设备编号推导路径 |
| daemon `view_root` | 每 accelerator 发布的文件视图 | 不等于 KV 索引根；RPC 消费路径不能自己拼 |
| Python `store.options.root` | 每 rank 索引/checkpoint；未推导mounts时也承载payload | 多盘preset换root/tag通常不隔离payload；local须核对实际mounts |
| `<有效mount或root>/r<rank>/segments/*.seg` | 真实 KV 槽位数据 | 多实例共用路径/rank会互相覆盖 |
| `TUTTI_PROFILE_ROOT` | logs/JIT/临时文件/nsys | 不是 store 容量、不是部署设备 identity |
| `config/local/` | 已审核本机配置 | 是项目数据，不是临时缓存 |
| `examples/**.yaml`、`tests/**.yaml` | workload/test 专用预算 | 不是生产推荐值，例如探针的大 PRP 容量 |

对象放置优先级：显式 `options.mounts` → 多盘 `preset.devices[].mount_path`（去重）→
`root`。单盘 `preset.device.mount_path` 不参与该推导；未显式设置mounts时，root必须确实
在该单盘文件系统上，payload也放在root下。不要同时让benchmark和线上实例使用同一
有效payload路径/rank；多盘wrapper不同 `POOL_TAG` 也不安全。

## 5. 参数目录：在哪里配置、单位和默认

### 5.1 控制器与 daemon（`device_manager/nvme_daemon.yaml`）

| 参数 | 单位/职责 | 当前缺省或建议 |
|---|---|---|
| `grpc.endpoint` | 服务地址/端口 | parser 默认 `127.0.0.1:50051`；多 daemon 不同端口 |
| `accelerators[].accel_id`、`nvmes[].device_id` | 显式身份 | 必须审核，不凭数组顺序推导 |
| `pci_addr`、`namespace_id` | 控制器 BDF / namespace | namespace 默认1；核验真实设备 |
| `backing_mount_path`、`view_root` | 文件系统路径 | 必须审核当前占用与已有数据 |
| `allowed_accel_ids` | accelerator ACL | 空集合会扩展到已配置accelerators，不等于安全“无需检查” |
| `auto_mount` | 挂载/ownership | parser 默认true；md成员应false，阵列入口负责挂载 |
| `raid0_arrays[].device_ids/mount_path` | 阵列成员和mount | 不创建/格式化阵列；保持已有superblocks |
| `kernel_ioq_cap` | 内核 block-IO 队列 hint | 默认0；只在bind前使用，非用户SQ深度 |
| `queue_pool.default_per_client/max_per_client` | Resource RPC每控制器队列政策 | parser默认4/32；模板显式16/32，不替代内核可用总量 |
| `lease.heartbeat_interval_sec/timeout_sec` | 秒 | 默认10/30，不为吞吐任意调低 |
| `unmount_retry.interval_ms/max` | ms / 次数 | 默认1000/30；不靠第二次信号强退忙盘 |

**不可覆盖的事实**：queue depth、user QID range、MDTS、namespace block size、PCI/chrdev
身份、md stripe。用户环深度与内核不同会造成回卷/CQ phase 不一致，不能作为调优旋钮。

### 5.2 Runtime 与 DataPath（native YAML 或 Python preset）

| 参数 | 作用域/单位 | 当前行为 |
|---|---|---|
| native `runtime.accel_id` / preset `gpu_id` | 本地逻辑accelerator ordinal | 与可见GPU及绑定pool一致 |
| native `allocation.device_ids` / preset `device_id`、`devices` | daemon显式ID | 不是 `/dev/nvmeN` 编号；设备事实和chrdev仍验证 |
| native `queues_per_controller` / preset `num_queues` | 每客户端/rank每设备SQ/CQ队列对 | connector字典preset缺省8，低层直调默认不变；两种schema都不把0当auto；不是queue depth |
| `max_in_flight_operations` | 每DataPath未完成操作上限，不是NVMe命令数 | vLLM缺省按加载模式推导2L/3L；原生无模型默认16；高级覆盖保留 |
| `max_batch_entries` | 每次native提交展开后的sub-IO上限 | sync按模型上下文、序列数和KV block/page估算并保持有限上限；auto/async不按单步缩减；原生默认256 |
| PRP缓存 | host pinned地址表，按控制器分池 | runtime自动按实际页大小、控制器数和有效batch/in-flight定容；不再接受容量配置 |
| handle缓存 | local低层GPU handle及host快照 | 应用factory不启用重复缓存；RAID0本来不使用该L1/L2；不再接受容量配置 |
| `TUTTI_POOL_WORKERS` | GPU提交worker数 | multi默认256且按条目裁剪；local默认2048不按条目裁剪；同步0用legacy burst，实验async的0仍取256 |
| `TUTTI_ASYNC_COMPLETION` | 设备完成模式 | 默认关；1只对multi DataPath启用实验submit/wait分离 |
| Runtime `max_terminal_results` | 保留的完成结果数量 | native Runtime默认64；preset按in-flight自动抬到至少2倍 |

`prp_cache_capacity`、`handle_cache_capacity`、`handle_cache_l2_capacity`已从native YAML和
Python preset删除；未被执行路径消费的YAML `io_granularity`也已删除。底层显式注册粒度和
cache契约测试接口保留，它们不是应用配置。

Python preset的batch/in-flight显式0在装配前转换为有效256/16，caps与terminal结果预算
使用有效值，不再用0跳过准入。建议省略这些高级字段，让connector计算模型所需下限。

### 5.3 Connector、KVEngine 和对象层

| 参数 | 单位/范围 | 缺省/配置位置 |
|---|---|---|
| `chunk_tokens` | token/chunk | connector必填；示例512；必须是vLLM block token数的整数倍 |
| `max_chunks_per_wave` | chunk/引擎工作窗口 | 可省略，缺省最多512；有权威block几何和字典preset预算时按能力缩小，显式值保持原准入 |
| `num_layers/chunk_kv_bytes/segment_bytes` | 层/字节 | 从vLLM权威KVCacheConfig推导，显式值只用于兼容且必须一致 |
| `store.options.root` / `mounts` | 路径 / 路径列表 | root是每rank索引根，mounts决定放置；local未传mounts时payload也回退root，见§4 |
| `capacity` | 带单位的每rankpayload预算 | 唯一用户容量入口，例如`"64 GiB"`；GB用1000、GiB用1024，内部再换算字节和槽数 |
| `io_stream` | CUDA有序任务流，`auto`或native句柄 | vLLM NVMe worker缺省auto；直接store仍None；不是网络流/线程数；不手拼0默认流 |
| `segment_file_slots` | 每段文件槽数 | 默认2048；影响预建组成本、共享布局，改动需部署隔离 |
| `segment_header_bytes` | 每槽前缀字节 | 默认128KiB，≥4096且对齐；属于持久布局，不是免费调优 |
| `defer_writes_after_reads` | 写流排序bool | options > `TUTTI_DEFER_WRITES_AFTER_READS` > true |
| `load_mode` | 请求sync/async/auto | `TUTTI_LOAD_MODE` > extra > auto；auto未测得成本时仍sync |
| `async_load_free_ratio` | (0,1] | 对应env > extra > 0.5；约束后台读持有HBM blocks |
| `lookahead_k` | staging层窗口 | 默认2；只影响staging路径，不限制direct逐层feeder；没有其它配置别名 |
| `direct_transfer` | 能力要求bool | true要求direct；false不关闭已有direct能力，不作强制staged开关 |
| `min_retrieve_tokens/max_tokens_per_load` | token | 默认0，不过滤最小加载/不封顶；与吞吐及命中收益有关 |
| `TUTTI_CHECKPOINT_INTERVAL_S` | 秒 | 默认60，0关闭周期checkpoint；改变恢复窗口而非IO正确性 |
| `TUTTI_PRECREATE_MIN_FREE_BYTES` | 字节 | 默认min(32GiB,mount总容量5%)，0关闭护栏；别为速度关闭 |
| `TUTTI_HEALTH_INTERVAL_S` | 秒 | 默认30，最少1；日志观测非吞吐旋钮 |

load-policy async是vLLM把请求暂缓直到所有rank加载完成；completion async是GPU IO提交后退出，
由消费流回收CQ。二者独立，不能看到“async”就同设或把配额规则混用。

### 5.4 服务/离线wrapper

MODEL、TP_SIZE、CUDA_VISIBLE_DEVICES、DAEMON_CONFIG、DEVICE_GROUPS、KV_MOUNT、scratch
须用户明确审核。服务历史默认Hy3-FP8/TP8、FP8 KV、block512、chunk512、总量8TiB，不是
任意模型/机器推荐；未知模型优先验证`auto` KV dtype，FP8需要正确scales。

`NUM_QUEUES` serve/bench默认8，覆盖进入Python preset；`MAX_IN_FLIGHT`离线缺省不传，
由connector自动算。在线`TOTAL_CAPACITY="8 TiB"`为服务总payload预算；离线
`CAPACITY="1 TiB"`或`--capacity "1 TiB"`为每rank预算，必须明确设置。
HBM `NUM_GPU_BLOCKS_OVERRIDE`服务缺省不传，保留vLLM自然池。
离线driver默认block64且KV dtype为vLLM默认；与在线不同，不能混算比较。
改变block同时保持固定GPU block数会改变HBM容量，不能称作同口径性能实验。

## 6. 已自动推导的值与不能省略的预算

### 缓存预算，不是按显卡型号猜测

PRP缓存存储NVMe读取的地址表，实际backing在host pinned内存，不占HBM；key由已注册
内存的起始page和IO页数决定，与文件大小无直接比例关系。设有效batch上限为E、在飞
上限为I、控制器页大小为P、控制器数为N，每控制器缓存页数为
`min(2*I*E, floor(64 MiB / N / P))`。该64MiB是每个DataPath的固定保守缓存backing预算，
不是系统检测出的最优值，不包含CPU索引、fallback池、队列或其它进程；TP总量仍要乘rank数。

缓存耗尽继续走原host PRP fallback。自动缓存初始化失败时退到同一路径，启动日志
`PRP_CACHE_AUTO`报告实际容量（0表示未启用）；不会降低IO准入或放松pin/释放约束。
自动策略在shutdown/重新initialize后仍保留，容量重新按控制器事实计算。

handle L2只在local低层使用，它不能解决所有L1项仍被open引用而无法驱逐的问题。
应用对象池已经长期缓存段文件票据，因此factory不再额外启用有限handle L1/L2，避免
文件数超过缓存时open失败及重复快照成本。RAID0使用共享成员handle，本来没有这套L2。
这不是依据文件shape自动选出最优L2，而是移除该路径不必要的缓存层。

`allocator_enabled`及未被native消费的`background_reclaim`传递链也已删除；后台预建
仍由对象层根据分配前沿、ready余量和磁盘空间护栏驱动，不是这些空开关控制的。

### MiniMax M3单组异构实体

MiniMax-M3-MXFP8的主KV和稀疏attention index是同一逻辑block表下的独立实体，page大小
不同。Tutti用固定的最大page宽保留磁盘segment，而每实体只搬实际page字节；capacity
包含未使用padding，不能用60个模型层估算实际117实体占用。支持NVIDIA packed NHD及
连续index页，eager；split K/V、HND、graph和多group尚未支持。

worker的子spec与scheduler简化后的spec不同；两侧几何使用保留的tensor allocation
恢复真实page宽。namespace包含实体page序列和NHD布局指纹，避免假驻留和错格式读回。
自定义attention前后hook一起等待/保存main和index；写计划缺失任何callback都不能提交
成功。当前模型参数和验证范围见[vLLM手册](intergration/tutti-vllm-manual.md#minimax-m3-mxfp8)。

### 模型及资源推导

`max_in_flight_operations`与`max_batch_entries`仍是必要的内部边界，而不是必填项。
原生`NvmeDataPathTuning`没有模型信息，默认16操作/256条目；模板用`config: {}`。
vLLM先取得权威KV block token数、每层block字节和层数，再传入实际装配预算。

sync缺省batch预算：令R为`ceil(max_model_len / block_tokens) * max_num_seqs`，
F为`ceil(kv_page_bytes / 4096)`，W为一波chunk对应block请求数，取
`max(W, min(local 4096或设备组16384, R*F))`。4KiB仅是工作量估算粒度，不修改
真实namespace/MDTS。小模型可缩小批宽；缺少权威信息时保留有限的既有preset默认。
这个估算不是全请求保证：实际extent、md边界及后台合并还会改变submit次数。

auto/async读计划可跨调度步合并，不能用单步`max_num_seqs`缩小batch，否则每层分成
多个operation会吃完3L配额。因此本轮只对sync做模型批宽缩减；不宣称所有模式均已最小化。
`TUTTI_IO_BUDGET`启动日志输出最终层数、in-flight及batch预算；显式高级覆盖不被悄悄改写。

arena槽数为2倍in-flight，GPU entry/status/descriptor空间随槽数×batch增长；设备组还
有target/identity指针表。PRP LIST backing在host pinned内存，不是每entry额外占4KiB HBM。
因此不存在“很大且不消耗资源”的通用最大值；把workspace改成动态池或改变完成回收
需要独立的生命周期设计，不在本轮隐式改变。

单盘现在也检查展开后的entry数，超过工作区时在租赁/launch前整体拒绝；输入请求
数量小不代表MDTS展开后仍小。新硬件反例只编译，本轮未执行真实NVMe。

Python入门参数的直白说明、盘组分配示例和单位规则，见
[connector参数注释](../config/integration/vllm/README.md)。波次是引擎/staging工作窗口，
不是direct整请求大小上限；direct整层提交与backend分段逻辑保持不变。
缺省取`min(512, max_batch_entries // blocks_per_chunk)`，无已知额度时取512；
显式值不裁剪，原准入仍检查。容量先精确转换字节，旧槽数公式不变。

令层数为 L、每层每chunk字节为 S、chunk token为 C、block token为 B。

- blocks/chunk = C/B；IO page由实际每rank KV pool形状和dtype得出，不是模型权重dtype。
- chunk payload = L×S；`capacity`先解析成字节数D，内部槽数为
  `num_chunks=floor(D/(L×S))`。**仍未扣128KiB前缀/预建组取整/文件系统开销**，
  不是严格物理上限；保留空间，实际容量不可只看这个数。
- wrapper服务总量 T 传每rank T/TP；原生ObjectStore/单Store的预算没有服务总量语义。
- 未显式配置的dict preset：`load_mode=sync` 注入in-flight=2L；auto/async注入3L。
  80层分别160/240。该下限仅缺省时补齐，**不覆盖显式值**；环境preset不保证补齐。
- direct bind仅在caps为正数时检查≥2L和
  `max_batch_entries≥blocks_per_chunk×max_chunks_per_wave`；caps为0会跳过检查，绝非无限能力。
  该宽度只是输入block请求数，MDTS/extent/md边界可fan-out成更多sub-IO entries；实际native
  batch上限仍可能拒绝，不把最小准入公式当无限并发保证。
- arena槽数=2×in-flight。每槽含entries/status/dev_table等，内存随batch×in-flight线性涨；
  multi表还有`max(2048,entries)`下限。大于够用配额会争HBM，不能照搬探针大值。

设备字段、namespace、mount/chrdev身份在装配和实际打开时检查，不为减少配置而推测。
`config/storage_runtime/*.yaml`是独立C++工作负载示例，不知道模型L，所以不能替connector
设置最小整层配额。

## 7. 怎样选择性能起点

| 观察 | 先检查 | 再尝试，保持正确性门禁 |
|---|---|---|
| 没有外部hit、B与A一样 | worker提交完整性、scheduler恢复、HBM是否承担全部hit | 先修配置/实验口径，不增加队列 |
| RESOURCE_EXHAUSTED/准入失败 | in-flight、batch宽度、rank×QP、终态handle释放 | 优先让模型自动推导，不把所有预算设巨大 |
| 首请求很慢 | JIT/模型加载/peer注册/首组零填 | 比较稳态；启动预热而非调高所有预算 |
| 大读带宽低 | 真实P2P、IO大小/MDTS、盘是否同刻工作、extent、队列grant | 受控扫每rank队列4/8/16；实际总预算先审核 |
| 计算GEMM变慢、NCCL变长 | IO kernel驻留、各rank到达差、SM/HBM争用 | nsys确认；实验async需新代码真实回归后再尝试 |
| 写入跟不上/尾延迟变差 | 读写混跑、后台预建、容量/pin、写流依赖 | 对比defer=true/false；相同输入与缓存状态 |
| 预建抢盘/启动占用高 | segment_file_slots、工作集、首文件组成本 | 先核对容量/并发实例，不关闭空间护栏 |

保留各路径现有默认：multi worker池256、local池2048、段头128KiB、HBM自然池、自动几何/
缺省配额。常用block32线程；显式worker<32时local会缩小block，异步launcher另有处理；
保留必须审核：设备、模型、TP、容量和queue预算。不要用0、永久高QD、重建RAID或降低lease
来“自动优化”。本机4–32队列收益相同、某模型block512收益好，均不构成新机器保证。

## 8. 自动找最优：能做什么，不能做什么

当前已有两类自动行为：vLLM几何/准入推导；load_mode=auto在线采样read/compute/step EWMA
后选择请求加载方式。它们不是所有配置的黑盒搜索。

**本轮没有新增宣称全局最优的autotuner。** 一个安全调优器需要明确模型/长度/复用/并发、
目标TTFT p90或吞吐、HBM缓存状态、可丢弃且物理隔离的KV工作区，以及显存/queue约束；
缺这些条件，自动跑会覆盖段文件或选中只在单请求有效的参数。

使用现有bench做受控队列扫参即可：设置全部已审核部署环境，停相同设备的服务；串行使用
NUM_QUEUES=4/8/16，每个候选同输入至少3轮，保存不同tag日志，并穿插重复基线。不要并行，
不同POOL_TAG不隔离段文件；执行前必须允许写该测试盘。
比较真实完成吞吐、客户端尾延迟、输出/逐字节正确性、失败率、峰值显存和实际QP申请，
不是只取某轮最快B。跨候选控制温度、首轮JIT、缓存和对象预建成本。

后续脚本可自动**生成候选/读取结果并推荐**，执行搜索须单独显式授权；不得自动改内核、
格式化、调整md stripe、切驱动或把实验完成模式设为生产默认。

## 9. 实现核对点

- daemon schema：`csrc/device_manager/nvme/nvmeservice/src/nvmeservice_config.h/.cpp`
- native图/schema：`csrc/config/`；装配消费：`csrc/data_paths/data_path_factory.cpp`
- preset预算/绑定：`csrc/include/tutti/presets/local_nvme.h`、`csrc/python/src/_core.cpp`
- Python几何/配额：`tutti/integration/vllm/geometry.py`、`factory.py`
- 请求auto策略：`tutti/integration/vllm/load_policy.py`
- IO能力/布局：`tutti/storage/tutti_nvme/store.py`、`object_layout.py`、`runtime_factory.py`
- kernel/caps：`csrc/data_paths/local_nvme/`、`striped_local_nvme/`

新模型布局、持久布局、异步故障路径和真实性能分别验收，不用“配置解析成功”替代。
本轮保持当前默认同步完成；生命周期修正的状态见
[async-nvme-completion.md](design/async-nvme-completion.md)。
