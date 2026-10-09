#!/usr/bin/env bash
#
# Tutti 在线服务：Hy3-FP8 × TP8 × 两个 4 盘 RAID0（共 8 盘），与 8 卡离线基准同构。
#
# 几何与部署形态对齐 scripts/vllm/run/bench-8gpu-striped.sh，可直接配合
# vllm_online_longrun.py 做在线长跑。
#
# 用法：
#   RUN_DIR=/mnt/nvme4/tutti-profile/online/<tag> \
#     nohup bash scripts/vllm/run/serve-8gpu-striped.sh > $RUN_DIR/server.log 2>&1 &
#
# 数据布局：每 rank 的槽位在 4 盘间轮转，打包进固定大段文件
# <mount>/r<rank>/segments/<id>.seg（每文件 2048 槽 ≈ 20 GiB）。冷池启动时
# 每 rank 同步建首个文件组（每盘一个段文件，约 1 分钟），之后由后台线程在
# 分配前沿逼近时补建下一组；写路径永不建文件。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PORT="${PORT:-8192}"
# 工具调用（tool choice=auto 必需）。默认 hermes，与平台此前的 Mooncake 基线
# 一致：本服务只测 KV，不解析 Hy3 的工具调用。hermes 不认识 Hy3 的
# <tool_calls:opensource>，会把它当正文透传；hy_v3（Hy3 专用）会缓存它等参数，
# 在平台 max_tokens=1 的测法下正文为空、被判"答案为空"。需要结构化 tool_calls
# 时改用 TOOL_CALL_PARSER=hy_v3。不影响 Tutti 的 key 命名空间，旧池数据可复用。
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
MODEL="${MODEL:-/mnt/nvme4/models/Hy3-FP8}"
# 对外服务名：流量平台按服务组名 base_model_zh7 路由请求，必须挂成 vLLM
# 别名，否则 404 "model does not exist"。tutti 保留给本地脚本（driver 默认
# --model tutti）。vLLM 的 --served-model-name 接受空格分隔的多名字。
SERVED_NAME="${SERVED_NAME:-tutti base_model_zh7}"
TP_SIZE="${TP_SIZE:-8}"
# KV 精度。默认 fp8：Hy3-FP8 checkpoint 自带 KV scale（kv_cache_scheme: static），
# 每 token 字节减半 → HBM KV 容量约 2.4M token（bf16 时 1.23M），盘读量也减半。
# 设 auto 回到 bf16。KV 命名空间含 dtype，切换后旧池数据不会被误读（但也不能复用）。
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
# vLLM 的 KV block 大小（token 数）。决定 Tutti 侧的两条几何：
#   page_bytes       = BLOCK_SIZE × 每 token 字节（TP8：bf16 512 B / fp8 256 B，单条 IO 的大小）
#   blocks_per_chunk = CHUNK_TOKENS / BLOCK_SIZE（必须整除）
# 取 512：bf16 单条 IO 256 KiB、fp8 128 KiB，一个 chunk = 1 条 IO。
# 改这个值必须同步改 NUM_GPU_BLOCKS_OVERRIDE —— 后者的单位是 block 而非
# token，block 定义翻倍而该值不变会让 HBM KV 池占用翻倍（反之则缩水）。
BLOCK_SIZE="${BLOCK_SIZE:-512}"
# 在线驱动只发不超过该长度的 prompt（脚本默认 --max-prompt-tokens 与之一致）。
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-262144}"
SAMPLES="${SAMPLES:-4}"
POOL_TAG="${POOL_TAG:-online1}"
# 模型上限来自 config.json 的 max_position_embeddings；MAX_MODEL_LEN 允许比
# prompt 上限多 16（vLLM 惯例），但不得超过模型上限（264144 会直接启动失败）。
MODEL_MAX_LEN="${MODEL_MAX_LEN:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_TOKENS + 16 < MODEL_MAX_LEN ? MAX_PROMPT_TOKENS + 16 : MODEL_MAX_LEN ))}"

# HBM KV 池上限（单位：block）。默认**不传**：用 vLLM 自然池。
# FP8 的可用 token 总量约 2.46M；512-token block 的块数由 vLLM 实测决定。
#
# 只有"隔离 Tutti 路径做实验"时才需要压小它：本机 vLLM 未注册
# /reset_prefix_cache 端点，无法清 HBM 前缀缓存，把池压到装不下工作集才能
# 逼 HBM 淘汰、让复用落到 Tutti（离线测试的做法：1040 blocks = 1.01 并发
# 128K）。⚠️ 压小 = 并发被锁死（请求排队），线上服务不要压。
# 取值约束：≥ ceil(MAX_PROMPT_TOKENS / BLOCK_SIZE)（512 for 256K/512），否则
# 调度器无法容纳一个请求。
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"

# 数据盘物理总量（方案 A）：**服务级总量**（8 卡加起来的总占用），默认 8 TiB。
# 注意 apply_capacity_bytes 的语义是"每个 rank 的容量"，而 8 个 rank 共享同一
# 组盘、各占一份 KV 分片，所以这里必须除以 TP_SIZE 再传给 store，否则每个
# rank 都按整份总量建槽 = TP 倍超发，盘会被预建线程铺满（2026-09-22 事故：
# 8TiB × 8 rank = 64TiB 需求铺满 4×5.8TB 盘，ENOSPC 后 KV 写全失败）。
# 换算示例：8 TiB 总量 → 每 rank 1 TiB → 每盘（8 rank / 4 盘）2 TiB。
TOTAL_CAPACITY_BYTES="${TOTAL_CAPACITY_BYTES:-$(( 8 * 1024 * 1024 * 1024 * 1024 ))}"

# 盘组部署（默认：两个 md RAID0，各由独立 daemon 管理）：
#   DAEMON_CONFIG  客户端设备清单 = 各 daemon 配置，逗号分隔（device_id 全局唯一）
#   DEVICE_GROUPS  按 rank 分组的 device_id，分号分组：rank 0-3 → kv0（NVMe 0-3，
#                  /mnt/tutti_md0），rank 4-7 → md0（NVMe 4-7，/mnt/nvme4），与 GPU
#                  所在 NUMA/PCIe switch 对应
#   KV_MOUNT       各 rank 池根（索引/检查点）所在挂载点
# md0 需先用 scripts/tutti-md0.sh up 交给 Tutti，kv0 daemon 单独启动。
DAEMON_CONFIG="${DAEMON_CONFIG:-$REPO_ROOT/config/local/tutti_daemon_kv0.yaml,$REPO_ROOT/config/local/tutti_daemon_md0.yaml}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3;4,5,6,7}"
KV_MOUNT="${KV_MOUNT:-/mnt/tutti_md0}"
DEVICE_GROUPS_JSON="[[${DEVICE_GROUPS//;/], [}]]"
DAEMON_CONFIG_JSON="[\"${DAEMON_CONFIG//,/\", \"}\"]"
CAPACITY_BYTES=$(( TOTAL_CAPACITY_BYTES / TP_SIZE ))
# 工作集（仅用于日志展示）
CHUNKS_PER_REQ=$(( (MAX_PROMPT_TOKENS + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
WORKING_SET=$(( SAMPLES * CHUNKS_PER_REQ ))

echo "[serve] model=$MODEL tp=$TP_SIZE port=$PORT max_model_len=$MAX_MODEL_LEN (prompt≤$MAX_PROMPT_TOKENS) kv_cache_dtype=$KV_CACHE_DTYPE"
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    echo "[serve] HBM KV 池: ${NUM_GPU_BLOCKS_OVERRIDE} blocks（覆盖值）" \
         "(${NUM_GPU_BLOCKS_OVERRIDE} × ${BLOCK_SIZE} = $(( NUM_GPU_BLOCKS_OVERRIDE * BLOCK_SIZE )) tokens)"
else
    echo "[serve] HBM KV 池: vLLM 自然池（未覆盖；实际容量见 vLLM 启动日志）"
fi
echo "[serve] chunks_per_req=$CHUNKS_PER_REQ working_set=$WORKING_SET"
echo "[serve] IO 几何: block_size=$BLOCK_SIZE tokens/block" \
     "blocks_per_chunk=$(( CHUNK_TOKENS / BLOCK_SIZE ))" \
     "device_groups=$DEVICE_GROUPS_JSON kv_mount=$KV_MOUNT"
echo "[serve] daemon_config=$DAEMON_CONFIG"
echo "[serve] pool: 服务级总量 $(( TOTAL_CAPACITY_BYTES / 1024 / 1024 / 1024 / 1024 )) TiB" \
     "(每 rank $(( CAPACITY_BYTES / 1024 / 1024 / 1024 )) GiB，容量仅为上限)"
echo "[serve] pool_tag=$POOL_TAG (换 tag = 全新索引；段文件按 rank 共享，脚本不删)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/profile-env.sh"
PYTHON="${TUTTI_PYTHON:?profile-env.sh did not set TUTTI_PYTHON}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Phoenix GDS 权重加载器（phxloader）：直接从 NVMe 读到显存，绕过 page cache。
# 冷缓存下标准 safetensors 加载每 shard 10s+（300GB 要 ~18 分钟），GDS 不受
# page cache 冷热影响。phxloader 未编译/未安装时自动回退 --load-format auto。
PHXLOADER_DIR="${PHXLOADER_DIR:-/data/home/ryeqiu/phoenix/adapters/vLLM/phxloader}"
if [ -f "$PHXLOADER_DIR/phxloader/__init__.py" ] &&
   ls "$PHXLOADER_DIR"/phxloader/_phxloader*.so >/dev/null 2>&1; then
    export PYTHONPATH="$PHXLOADER_DIR${PYTHONPATH:+:$PYTHONPATH}"
    LOAD_FORMAT="${LOAD_FORMAT:-phxsafetensors}"
else
    echo "[serve] 警告：phxloader 不可用（缺 _phxloader*.so）→ 回退标准加载"
    LOAD_FORMAT="${LOAD_FORMAT:-auto}"
fi
echo "[serve] load_format=$LOAD_FORMAT phxloader_dir=$PHXLOADER_DIR"

# 每组 rank 共用一个阵列（DEVICE_GROUPS）；每 rank 自己的池根（检查点/
# 元数据）。{LOCAL_RANK} 由 connector 的 expand_placeholders 展开（大括号、无 $）。
read -r -d '' KV_CONFIG <<JSON || true
{
  "kv_connector": "TuttiConnectorV1",
  "kv_connector_module_path": "tutti.integration.vllm.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "chunk_tokens": $CHUNK_TOKENS,
    "max_chunks_per_wave": 512,
    "store": {
      "type": "tutti_nvme",
      "options": {
        "root": "$KV_MOUNT/tutti-kv-online-${POOL_TAG}-{LOCAL_RANK}",
        "capacity_bytes": $CAPACITY_BYTES,
        "io_stream": "auto",
        "preset": {
          "type": "striped",
          "daemon_config": $DAEMON_CONFIG_JSON,
          "gpu_id": "{LOCAL_RANK}",
          "device_groups": $DEVICE_GROUPS_JSON,
          "num_queues": 8
        }
      }
    }
  }
}
JSON

# HBM 池只在显式给了覆盖值时才传参（默认留空 = vLLM 自然池，生产口径）。
HBM_ARGS=()
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    HBM_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    "${HBM_ARGS[@]}" \
    --model "$MODEL" \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size "$TP_SIZE" \
    --block-size "$BLOCK_SIZE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --load-format "$LOAD_FORMAT" \
    --enable-prefix-caching \
    --enable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --port "$PORT" \
    --no-enable-flashinfer-autotune \
    --kv-transfer-config "$KV_CONFIG"
