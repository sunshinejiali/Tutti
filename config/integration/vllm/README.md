# Python 用户：KV connector 参数说明

`kv_transfer.json` 是 vLLM 可消费的 JSON，不能加入 `//`、`#` 或说明性未知字段。
因此注释放在本文件，运行配置保持干净。完整分层参数见
[配置指南](../../../doc/configuration-and-performance.md)。

## 入门只关心这几项

| 配置 | 意义 | 用户需要做什么 |
|---|---|---|
| `chunk_tokens` | 一个缓存对象对应多少 token | 默认示例512；必须是vLLM block大小的整数倍，改变会改变缓存key布局 |
| `options.root` | 当前rank的索引/checkpoint目录 | 改成NVMe上的路径，TP多rank保留`{LOCAL_RANK}`；多盘时payload仍由设备mount决定 |
| `options.capacity` | **每rank**的KV payload预算 | 用`"64 GiB"`、`"1 TB"`等带单位字符串；不要填裸数 |
| `preset.daemon_config` | 本机已审核的设备清单配置 | 改成实际daemon YAML；不启动daemon、不格式化磁盘 |
| `preset.gpu_id` | 当前worker使用的逻辑GPU序号 | `{LOCAL_RANK}`由connector展开，不是PCI地址 |
| `preset.device_groups` | 将连续TP ranks分配给哪些存储设备组 | 按部署意图指定；ID来自daemon的`device_id`，不是`/dev/nvmeN`后缀 |

`preset.type`默认`raid0`，本例也明确写`raid0`：选择**已存在的同一md RAID0的完整成员**。
运行时验证实际md级别、成员集合和offset；不同mount/缺成员/非md会拒绝，不降级到独立盘。
不会自动创建阵列、格式化或调整条带。单盘用`local`，独立多盘用`nvme-device-group`，
`nvme-device-group`表示一次IO使用的NVMe控制器集合，不表示RAID类型；共享md挂载点
仍会核验阵列。旧`multi-nvme`、`striped`名称不再接受，普通用户沿用默认`raid0`即可。

## 容量单位不再需要手算

- `"64 GiB"` = 64×1024³字节；`"64 GB"` = 64×1000³字节。MiB/GiB/TiB用1024，MB/GB/TB用1000。
- 支持大小写、空格和小数，如`"1.5 GiB"`；必须能精确换成正整数字节。
- 用户配置只接受`capacity`；不再支持`capacity_bytes`或`num_chunks`。底层对象层仍内部使用字节和槽数。
- TP8时每rank`"64 GiB"`意味着服务总payload预算512GiB，**不是每盘64GiB**。
- 预算仍未包含槽位前缀、文件组取整或文件系统开销，不保证物理空间严格封顶；必须留余量。
- 这是connector的配置便利项，底层直接构造store仍按`num_chunks/segment_bytes`工作。

## 为什么还保留 max_chunks_per_wave

它是**引擎每个工作窗口最多处理的chunk数**，用于分批和兼容staging路径的缓冲预算。
512个chunk、每chunk512token，窗口对应262144token；这不是单请求长度上限，也不是队列深度。

该参数来自通用引擎/暂存路径，直接暴露为所有vLLM用户必填没有必要。现在可省略，起点512；
有权威block几何和已知preset批次额度时，取
`min(512, max_batch_entries // blocks_per_chunk)`。不能装下一chunk就报错。

当前direct路径可以提交整层，并由backend按能力分段；wave也参与direct准入检查，
**不代表每个direct请求都机械切成这么宽**。MDTS/extent/md边界仍会增加native sub-IO条目，
默认公式不是完整物理准入证明。

高级用户仍可显式设正整数；旧显式值不自动裁剪，保留原准入错误。普通用户省略即可。

## io_stream 是什么：不是网络流

CUDA stream是GPU上的有序任务队列。Tutti用独立读/写流与模型计算建立event依赖，
让IO可以重叠，且在读取KV之前保证数据就绪。

vLLM的`tutti_nvme` worker缺省使用`auto`：在runtime绑定的GPU上创建IO流，用户无需
提供整数CUDA指针。scheduler不会因为这个默认创建流；通用store/fake调用保持原默认。
这不开启实验异步完成，也不关闭整层同步门禁。

`io_stream`仅保留给低层集成的高级覆盖。不要写`0`、随便复制另一GPU的句柄或把它当Python
线程数。显式`None`或句柄不会被自动覆盖，但可能无法满足生产direct独立流准入。

## device_groups：部署者配置一次，应用用户复用

TP rank是参与同一模型的worker编号，从0开始。盘组按rank连续均分：

| 配置与TP | 实际分配 |
|---|---|
| `[[0, 1]]`，TP4 | rank0–3都使用设备0、1 |
| `[[0, 1], [2, 3]]`，TP4 | rank0–1使用设备0、1；rank2–3使用设备2、3 |
| `[[0], [1], [2], [3]]`，TP4，`type=nvme-device-group` | 每rank只使用对应的一组设备；不是RAID0配置 |

组数必须整除TP；每组中的全部设备都要允许该组GPU访问，并核实P2P。完整md阵列的成员
应在同一runtime组内。单盘local preset不需device_groups，显式`device_id`即可。
不能自动把“GPU i”与“磁盘i”配对，Linux编号、PCI亲和和数据ownership并不等价。

这一项不是模型调参，应由部署者提供已审核模板；Python应用用户复用模板，而不是理解
所有PCI细节。不同root/POOL_TAG不隔离同一mount/rank的payload，别并发跑测试和线上。

## num_queues：普通用户不设，高级调参时再展开

一对NVMe SQ/CQ用于提交命令并收完成。`num_queues`表示**每rank、每设备**申请多少队列对，
不是每个chunk多少队列，也不是queue depth。

connector的字典preset缺省8；native/preset低层直调原默认不变。4个ranks共享一设备时，
缺省一共申请4×8=32队列对，还要考虑其它进程。真实可用QID由内核/部署决定，不能仅凭
daemon每客户端策略认为全服务预算必定够。

更多队列不一定更快，先用默认；确有测量依据时用`preset.num_queues`或wrapper的
`NUM_QUEUES`覆盖。配置本身不改变内核queue depth，也不会替你切驱动扩队列。

## 缓存容量：不用再填写

`prp_cache_capacity`、`handle_cache_capacity`、`handle_cache_l2_capacity`已从用户配置删除。
PRP是host pinned地址表，不是GPU里的KV数据缓存。runtime依据控制器真实页大小和有效
batch/in-flight计算容量，每个runtime的缓存backing总预算最多64MiB，按控制器分摊。
不足或无法初始化时继续原PRP fallback，不代表整个host内存占用有64MiB硬上限。

RAID0并不使用handle L2；local对象池已长期持有文件票据，不再叠加有限L1/L2缓存。
盲目按显存、文件大小放大L2不会让仍处于open状态的L1可驱逐。
`allocator_enabled`空开关也已删除，实际预建仍自动跟随分配前沿并遵守磁盘空间护栏。

层数、KV字节、in-flight和wave已有推导，普通用户不必填batch/in-flight。
sync模式按上下文、序列数及KV block/page估算batch，小模型不再固定分配16384宽度；
auto/async可能跨步合并，先保留原有限批宽。两项内部上限仍保留高级覆盖，因为实际fan-out
还依赖md条带、MDTS及文件extents；不能给出廉价且覆盖任意请求的最大值。
队列、IO worker数和异步模式也不是显存越大就应越大的参数，不自动跑写盘扫参。

## MiniMax M3的KV布局

MiniMax-M3-MXFP8在NVIDIA路径有主KV及独立稀疏attention index缓存，均必须offload。
同一group内page宽不同仍使用固定宽磁盘segment，按每实体真实字节读写；不是把index
当主KV或跳过它。配置几何由cache spec/tensor分配推导，不能按模型60层手填117实体。

已验证TP8、block128、KV dtype `auto`、eager、16K/80%复用。要求NHD packed页、64KiB
base对齐；CUDA graph、split K/V、HND与多group未认证且会拒绝。
运行参数和安全边界见[MiniMax操作说明](../../../doc/intergration/tutti-vllm-manual.md#minimax-m3-mxfp8)。

## 一条规则

**应用用户选择模型、容量和缓存目录；部署者提供GPU/存储映射；引擎负责IO窗口、流和
缺省队列。** 性能调参不改变数据格式、控制器身份验证或同步正确性门禁。
