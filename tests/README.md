# 测试与架构对应表

测试按**责任模块**落目录，依赖级别和部署拓扑另行标注。不要按任务编号、round 或
单次事故建立新的顶层目录；回归补在对应模块。文件数量不是覆盖质量，只有相同输入、
相同被测实现和相同断言才算可删除的重复。

## 1. 模块归属

| 测试目录 | 对应实现 | 主要验证内容 |
|---|---|---|
| `api/` | `csrc/include/tutti/` | Status、IO/Memory 类型、CUDA-like/device guard、公开 API、SPI 消费和头文件隔离 |
| `config/` | `csrc/config/` | YAML 解析、引用关系、schema、调试输出和配置头消费 |
| `device_manager/` | `csrc/device_manager/nvme/`、`csrc/common/snvme_identity.h` | daemon/grant、挂载管理、UAPI、字符设备与 PCI 身份 |
| `resource/` | `csrc/resource/` | 资源能力、描述与预算契约 |
| `resolvers/` | `csrc/resolvers/` | SPI、factory、真实文件 FIEMAP 解析 |
| `payloads/` | `csrc/payloads/` | payload 类型协议、memfs 扩展示例 |
| `data_paths/` | `csrc/data_paths/` | SPI 与 mock kit、factory、单设备/多设备提交、异步完成与退休 |
| `storage_runtime/` | `csrc/include/tutti/storage_runtime.h`、`csrc/tutti_runtime/` | 组件装配、partial acceptance、引用生命周期、单 NVMe 与 md RAID0 全链路 |
| `storage_objects/` | `csrc/storage_objects/` | 对象 SPI、digest/codec、分配、提交/恢复、media、固定段文件 |
| `python/` | `tutti/`、`csrc/python/`、`csrc/kv_transfer/`、`scripts/` | Python 对应模块及扩展/框架接线，见 [Python 测试](python/README.md) |

公共测试头 `hardware_test_directory.h`、`nvme_test_cli.h` 保留在此目录，避免为归组
改变 include 契约。C++ 叶目录保留独立 target：不同链接依赖、HOST/CUDA 构建边界、
故障隔离和写盘权限不应为减少目录数合成一个巨型 executable。

### 为什么相似测试不能合并删除

- Index 的 LRU/pin 与 Runtime 的 IO 引用退休，是不同层的保证。
- Mock/SPI 自测试不替代真实 NVMe DMA；GPU 合成 CQ 不替代控制器错误注入。
- 对象头、checkpoint、bitmap、段文件预建与整层重启，各有不同的恢复故障面。
- 公开 API、SPI consumer、header hygiene 虽都包含基础类型断言，但**链接面和 include
  可见性不同**，分别保留。
- Python `storage/test_store_contract.py` 使用同一套参数化契约验证 Memory/Tutti store；
  实现专属错误、目标票据和提交行为仍在专项文件。

## 2. 拓扑不是模块，也不是完成模式

| 拓扑/能力 | 测试位置 | 说明 |
|---|---|---|
| 独立单 NVMe | `data_paths/local_nvme/`、`storage_runtime/local_nvme/` | LocalFileResolver → LocalNvmeDataPath |
| 多个独立挂载点 | `data_paths/multi_nvme/` | 整个对象槽位在 mounts 轮转；StripedDataPath 是保留的实现名，不是细条带切文件 |
| md RAID0 | `storage_runtime/raid0/` | 一个阵列挂载点，由 resolver 映射到成员 namespace；以 Linux md 文件 IO 作字节 oracle |
| 字符设备身份 | `device_manager/snvme_identity/` | 核对 PCI 与 chrdev，不凭设备编号匹配 |
| 异步完成 | `data_paths/async_completion/` | 完成模式，不是另一种磁盘拓扑；合成 CQ/多流/迟到完成验证 |

md 测试 YAML 留在 `storage_runtime/raid0/`，只属于该用例，不是部署默认配置。

## 3. 按模块和依赖选择

先按 [快速开始](../doc/getting-started.md) 构建并选择解释器，设置 NVMe 上的 `TMPDIR`。

```bash
# 当前兼容入口：Python 全套 + C++ 非硬件
bash scripts/tutti-env.sh test all

# 查注册清单，不执行
ctest --test-dir build -N
ctest --test-dir build -N -L hardware

# C++ 模块软件门禁：保留所有旧测试名，追加 module_<目录> 标签
ctest --test-dir build -L '^module_storage_objects$' -LE hardware --output-on-failure
ctest --test-dir build -L '^module_storage_runtime$' -LE hardware --output-on-failure
ctest --test-dir build -L '^module_device_manager$' -LE hardware --output-on-failure

# Python 模块（当前环境须已安装所需依赖）
python -m pytest tests/python/index tests/python/engine -q
python -m pytest tests/python/integration_vllm -q

# 显存合成故障测试：需要 GPU，不打开 SSD，不需 daemon；不是 SSD DMA 认证
ctest --test-dir build -L '^gpu_only$' --output-on-failure
```

C++ 模块标签为 `module_api`、`module_config`、`module_device_manager`、
`module_resource`、`module_resolvers`、`module_payloads`、`module_data_paths`、
`module_storage_runtime`、`module_storage_objects`。原标签 `hardware`、`raid0`、
`local_nvme`、`runtime_e2e`、`async_completion` 等仍保留，历史 `striped` 标签仅用于兼容。

**`test all` 不意味着零 GPU/零文件写入**：Python 仍包含可用 GPU 时执行的 transfer、
回调接线、staging 用例；对象 media 测试在临时目录创建小数据文件。非硬件 C++ 表示
不需要真实 NVMe 控制器，部分环境用例可能因权限/设备缺失跳过。
报告应区分 passed、skipped 和构建时未启用的用例；仅数 CTest target 不等于数内部断言。

### 真实盘验收与性能

真实 NVMe/RAID 测试必须先停相同设备上的服务，检查 daemon/GPU/namespace/目录，确认
测试盘允许写入。不要在维护者示例配置上直接运行所有 `hardware` 用例，也不要把
`POOL_TAG` 当作共享段文件的物理隔离。目录审核和启动顺序见快速开始。

- `data_paths/perf/batch_open/`、`data_paths/perf/register/`：手动性能 executable，
  不注册 CTest，不并入默认 pass/fail 测试。
- `examples/nvme_bw_probe/`：带宽与异步计算重叠基准。
- `scripts/vllm/run/`：真实模型 offline/在线/NSys 验证，见 [运行说明](../scripts/vllm/run/README.md)。
- `device_manager/service_client/`：历史手动 attach harness，带维护者设备映射，不属于默认
  测试；不能把“无 block IO”当作“无设备接管副作用”，未经配置审核不要执行。

## 4. 修改测试的规则

1. 归属首先按被测实现，跨层测试按主要断言所在层；例如 Runtime→md 全链路归 Runtime。
2. unit/contract/integration/perf 是验证性质，不再作为混装所有模块的顶层类别。
3. 合并相同断言时参数化不同输入，不删除 singleton、多层、per-layer/cross-layer 等边界。
4. 文件迁移同时更新 CMake、pytest 跨文件 imports、相对仓库路径和文档/skill。
5. CTest executable 名称、add_test 名称与原 binary 工作目录保持稳定；不要顺带移动生产库。
6. 回归确认：迁移前后注册清单一致、收集差异可解释、软件门禁通过。需要硬件才成立的
   正确性与性能仍须另行验收。
