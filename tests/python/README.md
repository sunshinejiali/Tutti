# Python 测试：按责任模块归组

总测试分层见 [tests/README.md](../README.md)。这里不再用 `unit/contract/adapter/overlap`
混合承载不同模块；文件名与场景保留，按实际实现责任落目录。

| 目录 | 对应实现 | 内容/依赖 |
|---|---|---|
| `common/` | `tutti/common/` | NVTX 与轻依赖基础逻辑 |
| `index/` | `tutti/index/` | 链式 key/namespace、LRU/pin、驻留压力；namespace 文件包含配置接线断言 |
| `engine/` | `tutti/engine/` | 读写计划、完成/容量闭环、staging 生命周期；后者需要 GPU |
| `storage/` | `tutti/storage/` | 共享 Store SPI、FakeRuntime、ObjectLayout/恢复/预建、多设备目标；部分用例依赖原生对象层 |
| `integration_vllm/` | `tutti/integration/vllm/` | connector/worker/scheduler、模型几何、IPC/TP、读写接线与跨层往返；需要 vLLM/torch，部分接线用例用 GPU |
| `bindings/` | `csrc/python/` | `tutti_runtime` stub 绑定；局部 conftest 不上移，避免整个 suite 收集时触发构建检查 |
| `kv_transfer/` | `csrc/kv_transfer/` | gather/scatter 精度、CPU 布局发现与 staged 全链路；数据搬运用例需要 GPU |
| `scripts/` | `scripts/vllm/run/` | 服务 dry-run、配置拒绝、调用者环境保留、monitor 日志解析；不加载模型 |

`integration_vllm/test_direct_kv.py` 保留 worker→engine→store 的跨层断言，不为目录纯度
拆散共享 fake 或重复支架。`storage/test_store_contract.py` 的参数化工厂与
`storage/test_tutti_store.py` 仍在同目录。层往返复用 vLLM harness 的导入已同步。

## 运行

先按 [快速开始](../../doc/getting-started.md) 安装所选环境和原生扩展。

在仓库根，选择 Python 后设置 NVMe 暂存目录：

```bash
# 包装入口维护 tutti 与两个原生扩展的 PYTHONPATH；未启动 daemon 或服务
bash scripts/tutti-env.sh test py

# 直接 pytest：先让环境能 import tutti/tutti_runtime/tutti_kv_transfer
python -m pytest tests/python -q
python -m pytest tests/python/index -q
python -m pytest tests/python/storage -q
python -m pytest tests/python/integration_vllm -q
```

若直接运行且未做 editable 安装，可按快速开始设置仓库和两个扩展目录的 PYTHONPATH。
指定 `--basetemp=<NVMe 临时目录>`，避免根盘 `/tmp`；不要指定存放重要数据的目录，pytest
会清理 basetemp。使用 `tutti-env.sh` 且已设置 `TMPDIR` 时会传入专用 basetemp。

完整收集仍由 `pyproject.toml` 的 `testpaths=["tests/python"]` 控制，目录重组不改变默认
测试选择。没有 GPU 的相关用例 visibly skip；有 GPU 时全套会运行这些用例，不能称作
纯 CPU suite。FakeRuntime/对象层用例不证明真实 DMA 正确；真实模型、性能和压力回放
使用 [vLLM 运行脚本](../../scripts/vllm/run/README.md)（仓库根下）。

性能脚本在 `scripts/vllm/run/` 与 `examples/nvme_bw_probe/`，不再保留空 `perf/scale/`
目录或已经失效的 stripe-unit 教学模拟器。容量压力回归仍在 `index/` 和 `engine/`。
