# Tutti 快速开始

本文是社区部署的入口：**确认硬件 → 构建和安装 → 配置宿主 → 软件/硬件验收 → 离线复用 → 在线服务**。
不要在未知机器上直接执行 `bootstrap`。它会加载驱动并接管配置中的 NVMe。

## 1. 支持范围与前置条件

- Linux 裸机，NVIDIA GPU、匹配的 CUDA toolkit、当前运行内核的 headers/devel。
- GPU 与所用 NVMe 必须能够 PCIe peer-to-peer DMA。优先同一 switch / 同 NUMA；跨 switch 必须实际验证。
- IOMMU 使用 passthrough 或关闭；不能把普通 IOMMU 地址转换当作已支持路径。
- CMake ≥ 3.21，兼容 CUDA/PyTorch 的 C++ 编译器，以及 gRPC、Protobuf、yaml-cpp、uuid 开发包。
- Python ≥ 3.10；先在同一虚拟环境安装 CUDA 版 PyTorch 和 vLLM，再编译 Tutti 扩展。
- 选择允许接管的 **KV 专用数据盘**。不使用系统盘，不默认与模型、Docker、业务数据共盘。

内核版本的“代码 baseline 存在”不等于已完成真机认证；检查
[`kernel_modules/PORTING.md`](../csrc/device_manager/nvme/kernel_modules/PORTING.md)。
其他 GPU 厂商的构建边界见 [`advanced-build.md`](advanced-build.md)。

### Connector 兼容性

当前支持 Hy3-FP8 的统一跨层缓存路径，以及 MiniMax-M3-MXFP8 主 KV 和稀疏 index 的
逐实体 GPU-direct 路径；后者允许同一逻辑 block 表下不同实体 page 宽度，仅接受 NVIDIA
packed NHD 主 KV / 连续 index 页并要求 eager。CUDA graph、split K/V、HND 及多 KV group
尚未支持；DeepSeek 仍需单独适配。模型参数、实测范围与限制见
[vLLM 操作手册](intergration/tutti-vllm-manual.md#minimax-m3-mxfp8)，
**不应仅因 vLLM 可以加载就声称 Tutti 已支持**。

维护者验证环境：Python 3.11.13、PyTorch 2.13.0+cu130、vLLM
`0.23.1rc1.dev2467+g897ff4f39`、FlashInfer `0.6.15.post1`。这是一组验证记录，不是所有版本
的兼容承诺；目前尚未发布经过完整 CI 验证的 vLLM 版本矩阵。Tutti 通过外部
`TuttiConnectorV1` 接入，不需要修改 vLLM 或 FlashInfer。

## 2. 选择 Python 与 NVMe 暂存目录

在克隆后的仓库根执行，下面的 `/path/...` 必须换成真实路径：

```bash
cd /path/to/Tutti
source /path/to/venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export TUTTI_PYTHON="$PYTHON_BIN"
export TUTTI_PROFILE_ROOT=/path/on/nvme/tutti-profile
export TMPDIR="$TUTTI_PROFILE_ROOT/tmp"
export PIP_CACHE_DIR="$TUTTI_PROFILE_ROOT/cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python -c 'import torch,vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)'
```

暂存、pytest、JIT cache、日志和 nsys 导出都留在所选 NVMe；不要挤占根盘 `/tmp`。
`profile-env.sh` 不再改写 `PATH`、编译器或 `CUDA_VISIBLE_DEVICES`，这些由用户环境负责。

## 3. 构建和安装

`prepare_env.sh` 会检查系统依赖，缺失时可能安装软件；在共享主机上先取得管理员许可。
它不会替你安装正确版本的 CUDA/PyTorch/vLLM，也不会创建虚拟环境。

```bash
bash scripts/prepare_env.sh
bash scripts/tutti-env.sh build -j 8
python -m pip install 'pybind11>=2.11' 'PyYAML>=6.0' pytest
bash scripts/tutti-env.sh build-ext
python -m pip install -e . --no-build-isolation
```

Tutti 有三个独立 distribution：

| 包 | 构建/安装位置 |
|---|---|
| `tutti` | 根目录 `pyproject.toml`，Python engine、store、connector |
| `tutti_runtime` | `csrc/python`，链接已构建的 C++ Runtime |
| `tutti_kv_transfer` | `csrc/kv_transfer`，PyTorch gather/scatter CUDA 扩展 |

`build-ext` 构建的是 in-place 扩展；从仓库运行时加载路径由下面命令设置：

```bash
eval "$(bash scripts/tutti-env.sh env)"
python -c 'import tutti,tutti_runtime,tutti_kv_transfer; print("imports OK")'
```

需要安装两个扩展到当前环境时，可以分别执行：

```bash
python -m pip install -e csrc/python --no-build-isolation
python -m pip install -e csrc/kv_transfer --no-build-isolation
```

它们依赖 CMake 的 `build/` 库和同一 Python/PyTorch ABI。**不能复制 `.so` 后删除 build 树，
也不能在服务运行时覆盖它依赖的库。** 无 GPU 的构建机器应显式设置目标
`TORCH_CUDA_ARCH_LIST`；不再默认为 H20 的 `9.0`。

## 4. 宿主配置与安全启动

入口文件与消费者见 [配置目录索引](../config/README.md)，从模型到控制器的参数、
路径及调优边界见 [配置与性能指南](configuration-and-performance.md)。

先按 [`tutti_daemon.md`](tutti_daemon.md) 核对 PCI BDF、namespace、序列号、容量和挂载。
从 `config/device_manager/nvme_daemon.yaml` 的 schema 创建自己的
`config/local/tutti_daemon_<name>.yaml`，而不是直接照抄维护者的 `kv0` / `md0`。

- `pci_addr` 是物理控制器身份；`nvmeN`、`snvmeN`、`ssnvmeN` 的数字不是身份。
- 每个 daemon 的 gRPC endpoint 唯一。客户端合并多份配置时，`device_id` 也必须跨文件唯一。
- 独立盘与 RAID0 都须在接管前准备好文件系统。格式化不是默认安装步骤。
- 已有 md 阵列必须先安全卸载/停阵列，不能让 daemon 从仍挂载的阵列下直接拿走成员。

模块加载和设备接管需管理员明确批准，顺序不可交换：

```text
内核模块 → daemon bring-up（产生 snvme 块设备、挂载、发布视图）→ 测试/服务
```

**独立的 KV 专用 NVMe**，在确认配置的盘空闲后：

```bash
bash scripts/tutti-env.sh --dry-run daemon start <name>
sudo bash scripts/tutti-env.sh modules load
bash scripts/tutti-env.sh daemon start <name>
TUTTI_DAEMONS=<name> bash scripts/tutti-env.sh status
```

**已有 RAID0**：`scripts/tutti-md.sh` 当前的阵列表只登记了维护者机器的 `kv0` / `md0`，
不是任意机器的自动部署器。新阵列须先登记自己的 PCI 集合、阵列 UUID、挂载点和保护级别，
再演练回退。含 `raid0_arrays` 而未登记的配置会被 `tutti-env.sh` 拒绝启动。
`normal` / `protected` 是数据安全策略，不是磁盘名称；后者手动切驱动还须 `--confirm <name>`。
详见 daemon 指南的 RAID0 章节。Phoenix 权重加载是可选项，默认不加载其内核模块。

## 5. 分层验收

按模块选择、拓扑与依赖范围见 [测试架构对应表](../tests/README.md)。测试目录已按
API/config/device_manager/resolver/data_path/Runtime/对象层和 Python 模块归组。

### 软件测试（不执行真实 NVMe IO）

```bash
bash scripts/tutti-env.sh test all
```

默认 C++ 排除 `hardware` 标签；Python 测试使用当前解释器，设置了 `TMPDIR` 时 pytest
暂存也放在那里。不要把“软件测试通过”当作 PCIe/P2P 真机认证。

### 硬件测试（显式选择，会写数据盘）

先检查相应用例的配置与 `--directory`，将所有测试文件指向允许写入的 KV 专用盘。部分
用例配置是维护者示例，必须按本机 endpoint、GPU、`device_id` 和挂载点修改。
不要直接对含主机数据的阵列运行全部硬件用例。

```bash
ctest --test-dir build -N -L hardware
# 配置审核完成、服务已停且测试盘允许写入后：
bash scripts/tutti-env.sh test cpp --hardware
```

至少完成设备身份错配拒绝、GPU 写/读与内核路径逐字节一致的用例，再运行 LLM。

## 6. 离线复用与在线服务

继续阅读 [`Tutti × vLLM 操作手册`](intergration/tutti-vllm-manual.md)，完成：

1. 明确 `MODEL`、TP、block size、KV dtype 与 context 长度；
2. 配置 `DAEMON_CONFIG`、`DEVICE_GROUPS`、`KV_MOUNT`；
3. 离线 A→B，清掉本地 HBM prefix cache 后确认 B 真的从 Tutti 读取；
4. 在线启动预览 `--dry-run`、健康检查、推理和 SSE 流式验收。

Agent 用户可复制完整的 [`Skills/tutti-runtime`](../Skills/tutti-runtime) 目录，安装方法见
[`Skills/README.md`](../Skills/README.md)。Skill 要求先确认硬件与数据安全边界，不会把
“运行部署”理解成“自动格式化、卸载或绑定任意磁盘”。

## 7. 发布状态

v0.2.0 提供外部 vLLM connector、固定段文件/RAID0 部署及 MiniMax 主 KV/index 逐实体路径。
默认仍是同步完成路径。跨 rank 副本读取合并与 GPU 分发计划在 v0.2.1 实现，
当前未实现；设计见 [KV 副本读放大提案](vllm-connector/kv-replica-read-amplification.md)。
`TUTTI_ASYNC_COMPLETION=1` 仅用于实验：生命周期修正已通过软件与 GPU-only 故障测试，
修正后的真实 NVMe/LLM 回归及控制器永久故障恢复尚未验收，不能仅凭之前的性能结果升级为生产默认。
