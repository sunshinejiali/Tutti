# 配置：按模块选入口，不按磁盘昵称猜路径

本目录分开存放 **daemon 模板、原生 Runtime 装配、Python connector 配置和本机部署事实**。
完整能力/参数/调优说明见 [配置与性能指南](../doc/configuration-and-performance.md)。

| 文件/目录 | 消费者 | 用途 |
|---|---|---|
| `device_manager/nvme_daemon.yaml` | `tutti_daemon --config` | daemon schema 模板：PCI/namespace/mount/ACL/队列策略/lease |
| `storage_runtime/local_nvme.yaml` | `TuttiRuntime::create` / C++ example 的 `--config` | 单控制器 Resource→Resolver→DataPath→Backend 图 |
| `storage_runtime/raid0.yaml` | 同上，推荐默认 | 已存在md RAID0；resolver强制检查实际阵列及完整成员，不允许降级 |
| `storage_runtime/nvme_device_group.yaml` | 同上，显式可选 | 多个独立挂载点，复用同一多控制器数据面 |
| `integration/vllm/kv_transfer.json` | `vllm --kv-transfer-config` | 最小 Python connector/preset 示例；逐项说明见 [参数注释](integration/vllm/README.md) |
| `local/tutti_daemon_<name>.yaml` | daemon、Python 硬件事实推导、宿主脚本 | 已审核的本机设备集合；仓库的 kv0/md0 只描述维护者机器 |

旧模板别名已删除。daemon必须显式传入`--config config/local/tutti_daemon_<name>.yaml`，
不自动查找旧路径或接管模板中的设备。Runtime YAML同样由调用者传入，不会隐式影响
Python服务；daemon只接受显式ID的`accelerators`/`nvmes` schema。

## 最少需要确定哪些值

1. 已审核 daemon 的 PCI、namespace、ACL、endpoint、数据 mount 和已有 RAID UUID。
2. Python 模型、TP/GPU 可见性、block/KV dtype 的兼容性。
3. 每组使用哪些显式 `device_id`；组数整除 TP。`device_id` 不是 Linux 设备编号。
4. 每个 rank 独立索引 root、服务总容量与实际物理空间、日志/JIT scratch 位置。
5. 初次部署审核总队列预算；connector默认每rank每设备8，不必每个应用填写。

示例 JSON 中 `capacity="64 GiB"` 是 **每 rank 64 GiB**，不是服务总量；总量
为该值乘 TP，且物理占用还包括前缀/文件组取整/文件系统元数据。先修改 `root`、
`daemon_config` 和 `device_groups`，再传给 vLLM。例子只提供 connector 参数，不替代
`--model`、`--tensor-parallel-size`、attention backend 等 vLLM 设置。

不要手工填写已由vLLM推导的`num_layers`、`chunk_kv_bytes`、`segment_bytes`。
`max_chunks_per_wave`、`io_stream`、`num_queues`均可省略；分别使用自动窗口、auto CUDA流
和8队列起点。容量只写`capacity`，支持MB/GB/TB与MiB/GiB/TiB；不接受用户配置的
`capacity_bytes`或`num_chunks`。不要增加说明性未知键；JSON注释放在旁边的README。
Python与native是两套schema。

## 当前数据面类型名称

名称直接区分两件事：**存储拓扑**与**参与IO的设备集合**。

- `raid0`：默认存储拓扑，使用已存在的md RAID0，核验完整成员。
- `local`：单盘入口。
- `nvme-device-group`：NVMe设备组入口，常用于独立挂载点；共享md挂载点仍会核验阵列。
- 原生`selection: device-group`：一次申请整个设备组，不表示文件放置或RAID类型。
- 原生`datapath.type/contract: nvme-device-group`：设备组IO实现，RAID0和独立盘都复用它。
- `resolver.config.topology: raid0`：要求该设备组构成真实RAID0，错误时不降级。

旧`multi`、`multi-nvme`和`striped`配置名不再接受。普通Python用户只需默认`raid0`，
不需要理解设备组数据面的内部装配。
`gpu_id`选择Python preset绑定GPU，对应C++ `accel_id`。

单盘不需要 `device_groups`：使用 `preset.type=local`、`device_id` 与 `gpu_id`。
多个独立 mounts 和一个 md mount 均用多控制器路径；不要为 RAID 复制第三套 DataPath 调参。
同一runtime混合多个md mounts或md＋独立盘尚不支持，双阵列应按rank/runtime分别选完整组。
local preset未显式传 `options.mounts` 时payload回退到root下，必须让root位于正确单盘mount。

## 默认与本机覆盖

- 模型几何和缺省in-flight由connector推导；sync的batch根据上下文、序列数和KV page缩减。
  auto/async可能跨步合并，先保留已有有限批宽。普通用户不需要填这两个高级字段。
- 原生无模型入口使用16操作/256条目，模板只留`config: {}`；它不是无限请求保证。
- IO worker 池、段文件槽数/前缀等已有默认保持不变；不把本机测量升级为全局最优值。
- 缓存容量不再由用户填写：PRP按实际控制器页和在飞预算自动分配；应用factory不启用
  重复handle L1/L2缓存。旧缓存容量键会被拒绝，详见指南§6。
- batch/in-flight的0在preset装配前变成有效256/16，caps同步有效值；它不是无限能力。
- 删除未生效的`allocator_enabled`、`background_reclaim`及YAML `io_granularity`选项；
  对象层实际后台预建及磁盘护栏保持不变。
- `load_mode=auto` 是请求级调度选择；`TUTTI_ASYNC_COMPLETION=1` 是实验的设备完成模式，
  不是同一个“async”开关。
- `NUM_QUEUES` 可覆盖 serve/bench 默认 8；不改变内核 queue depth。

修改前阅读 [指南](../doc/configuration-and-performance.md)，验收按
[快速开始](../doc/getting-started.md) 和 [测试模块索引](../tests/README.md)。
