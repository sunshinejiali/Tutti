#!/usr/bin/env bash
# Tutti 环境管理：构建、内核模块、daemon、测试的单一入口。
#
# 子命令
#   status                  只读全面检查（构建产物 / Python 扩展 / 内核模块 /
#                           daemon / 挂载）。退出码非 0 表示环境不完整。
#   bootstrap               从零到可运行：build -> build-ext -> up
#   build [--clean]         cmake configure + 构建全部 C++ 目标（含内核模块）
#   build-ext               构建两个 Python 扩展（tutti_runtime、tutti_kv_transfer）
#   modules load|unload|rebuild
#   daemon start|stop|restart|status [NAME...]
#   up [NAME...]            modules load + daemon start + status（**不编译**）
#   down [NAME...]          停止 daemon（不卸载模块）
#   test py|cpp|all         跑测试（内置正确的 PYTHONPATH 与前置检查）
#   env                     打印供 shell eval 的环境变量（PYTHONPATH 等）
#
# 多 daemon：每个盘组一个 daemon，NAME 对应 config/local/tutti_daemon_<NAME>.yaml，
# 不给 NAME 时管理该目录下全部 daemon。端口和应有的挂载点从各自配置读取，进程按
# --config 区分。配置含 raid0_arrays 的 daemon 一律经 scripts/tutti-md.sh 启停：
# 它负责从内核 nvme/md 交接，失败或 daemon 退出时自动退回内核 nvme 驱动 + 原阵列；
# 因此对它 stop/down = 把阵列还给 Linux。
#
# 保护级别（scripts/tutti-md.sh 的 LEVEL，本机 md0 = protected，kv0 = normal）：
# protected 阵列承载主机数据，切换驱动（start/stop/restart）必须 --confirm NAME，
# 且有任何使用者（挂载点、块设备、Tutti 客户端）时直接拒绝、不等待。未确认的
# protected daemon 被跳过（退出码非 0），其余 daemon 照常处理。
#
# 选项
#   --dry-run       只打印将执行的命令
#   --no-phoenix    跳过 phoenixfs（RDMA 场景需卸载它）
#   -j N            构建并行度（默认 nproc 上限 32）
#   --clean         仅 build：先删 build/ 再全新 configure
#   --confirm NAME  允许切换 protected 阵列 NAME 的驱动（可重复）
#
# 环境变量覆盖
#   TUTTI_ROOT TUTTI_BUILD TUTTI_LOG_DIR PHOENIX_BUILD PYTHON_BIN
#   TUTTI_DAEMONS        默认管理的 daemon 名（空格分隔，默认 config/local 下全部）
#   TUTTI_DAEMON_CONFIG  直接给配置路径（逗号分隔），优先于 TUTTI_DAEMONS
#
# 依赖顺序不可交换：/dev/snvme*n1 由 daemon bring-up 后才出现，先 mount 会失败。
#   内核模块 -> tutti_daemon（建块设备节点 + auto_mount + gRPC）-> 负载
#
# 本脚本不签名内核模块。若加载失败且 dmesg 出现 "Required key not available"，
# 请按站点签名流程签名 build/module/*.ko 后重跑。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TUTTI_ROOT="${TUTTI_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
readonly TUTTI_BUILD="${TUTTI_BUILD:-${TUTTI_ROOT}/build}"
readonly MD_SCRIPT="${SCRIPT_DIR}/tutti-md.sh"
readonly PHOENIX_BUILD="${PHOENIX_BUILD:-/data/home/ryeqiu/phoenix/build}"
readonly PHOENIX_KO="${PHOENIX_BUILD}/module/phoenixfs.ko"
readonly LOG_DIR="${TUTTI_LOG_DIR:-${TUTTI_ROOT}/.local/log}"
readonly SNVME_CORE_KO="${TUTTI_BUILD}/module/snvme-core.ko"
readonly SNVME_KO="${TUTTI_BUILD}/module/snvme.ko"
readonly DAEMON_BIN="${TUTTI_BUILD}/bin/tutti_daemon"
readonly LIBNVM_SO="${TUTTI_BUILD}/csrc/device_manager/nvme/libnvm/libnvm.so"
readonly PYBIND_DIR="${TUTTI_ROOT}/csrc/python"
readonly KVTRANSFER_DIR="${TUTTI_ROOT}/csrc/kv_transfer"

# 解释器必须与扩展的 ABI 标签一致（.so 名形如 _core.cpython-311-*.so）。
# 系统 python3 常常过旧（本机 /usr/bin/python3 是 3.6.8），故按此优先级发现：
#   PYTHON_BIN 显式指定 > 已激活的 venv > 仓库同级的 vllm-env（本机 vLLM/torch
#   所在环境）> PATH 中第一个 >= 3.10 的 python3.x > python3。
discover_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then printf '%s' "$PYTHON_BIN"; return; fi
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s' "${VIRTUAL_ENV}/bin/python"; return
    fi
    if [[ -x "${TUTTI_ROOT}/../vllm-env/bin/python" ]]; then
        printf '%s' "$(cd -- "${TUTTI_ROOT}/../vllm-env/bin" && pwd)/python"; return
    fi
    local p
    for p in python3.13 python3.12 python3.11 python3.10; do
        command -v "$p" >/dev/null 2>&1 && { printf '%s' "$p"; return; }
    done
    printf 'python3'
}
readonly PYTHON_BIN="$(discover_python)"

DRY_RUN=0
WANT_PHOENIX=1
DO_CLEAN=0
JOBS=""
SUDO_READY=0
DAEMON_NAMES=()   # 命令行点名的 daemon
CONFIRMED=()      # --confirm 允许切换驱动的 protected 阵列

note() { printf '[tutti] %s\n' "$*"; }
warn() { printf '[tutti] WARN: %s\n' "$*" >&2; }
die()  { printf '[tutti] ERROR: %s\n' "$*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

jobs_arg() { printf '%s' "${JOBS:-$(nproc 2>/dev/null | awk '{print ($1>32)?32:$1}')}"; }
module_loaded() { [[ -d "/sys/module/$1" ]]; }

# ---- daemon 配置 --------------------------------------------------------------

config_of_name() { printf '%s/config/local/tutti_daemon_%s.yaml' "$TUTTI_ROOT" "$1"; }
cfg_name() { local b; b="$(basename "$1" .yaml)"; printf '%s' "${b#tutti_daemon_}"; }

# 要管理的 daemon 配置，每行一个：命令行点名 > TUTTI_DAEMON_CONFIG > TUTTI_DAEMONS。
daemon_configs() {
    local n
    if (( ${#DAEMON_NAMES[@]} )); then
        for n in "${DAEMON_NAMES[@]}"; do config_of_name "$n"; echo; done
    elif [[ -n "${TUTTI_DAEMON_CONFIG:-}" ]]; then
        tr ',' '\n' <<< "$TUTTI_DAEMON_CONFIG" | sed '/^[[:space:]]*$/d'
    elif [[ -n "${TUTTI_DAEMONS:-}" ]]; then
        for n in $TUTTI_DAEMONS; do config_of_name "$n"; echo; done
    else
        ls "${TUTTI_ROOT}"/config/local/tutti_daemon_*.yaml 2>/dev/null || true
    fi
}

cfg_port() {
    local p
    p="$(grep -oE 'endpoint:[^0-9]*[0-9.]+:[0-9]+' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+$' || true)"
    printf '%s' "${p:-50051}"
}
cfg_has_arrays() { grep -qE '^raid0_arrays:' "$1" 2>/dev/null; }

# 经 tutti-md.sh 监管：配置含阵列且 tutti-md.sh 认识这个名字。
cfg_supervised() {
    cfg_has_arrays "$1" && [[ -x "$MD_SCRIPT" ]] &&
        grep -qE "^$(cfg_name "$1")\)" "$MD_SCRIPT"
}

# 配置声明 daemon 应挂上的点："array <mount>"（raid0_arrays，auto_mount）或
# "disk <mount>"（nvmes，auto_mount）。只认本仓库 daemon YAML 的两层结构。
cfg_expected_mounts() {
    awk '
        function flush() { if (am == "true" && mp != "") print kind, mp; mp = ""; am = "" }
        /^[[:space:]]*#/ { next }
        /^[A-Za-z0-9_]+:/ { flush(); sec = $1; sub(/:.*/, "", sec); next }
        sec != "nvmes" && sec != "raid0_arrays" { next }
        /^[[:space:]]*- / { flush() }
        {
            kind = (sec == "nvmes") ? "disk" : "array"
            line = $0; sub(/^[[:space:]]*-?[[:space:]]*/, "", line)
            key = line; sub(/:.*/, "", key)
            val = line; sub(/^[^:]*:[[:space:]]*/, "", val); gsub(/["\047[:space:]]/, "", val)
            if ((sec == "nvmes" && key == "backing_mount_path") ||
                (sec == "raid0_arrays" && key == "mount_path")) mp = val
            if (key == "auto_mount") am = val
        }
        END { flush() }' "$1" 2>/dev/null
}

cfg_level() {  # normal|protected（只有经 tutti-md.sh 监管的阵列有级别）
    if cfg_supervised "$1"; then "$MD_SCRIPT" "$(cfg_name "$1")" level 2>/dev/null || echo protected
    else echo normal; fi
}
confirmed() { local n; for n in "${CONFIRMED[@]}"; do [[ "$n" == "$1" ]] && return 0; done; return 1; }

# protected 阵列未确认时返回 1（调用方跳过它）。
check_level() {  # $1=配置 $2=动作
    local name; name="$(cfg_name "$1")"
    [[ "$(cfg_level "$1")" == protected ]] || return 0
    confirmed "$name" && return 0
    warn "跳过 ${name}：protected 阵列（承载主机数据），${2} 会切换它的驱动。确认无人使用后加 --confirm ${name}。"
    return 1
}

# 该配置的 daemon 进程（按 --config 匹配；锚定行首，排除 sudo/env 包装进程）。
cfg_pid() {
    local b; b="$(basename "$1")"; b="${b//./\\.}"
    pgrep -n -f "^([^ ]*/)?tutti_daemon --config ([^ ]*/)?${b}( |\$)" 2>/dev/null || true
}
all_daemons() { pgrep -a -f '^([^ ]*/)?tutti_daemon --config ' 2>/dev/null || true; }

port_listening() {
    command -v ss >/dev/null 2>&1 || return 0   # 无 ss 时不作为判据
    ss -ltnH 2>/dev/null | awk -v p="$1" '$4 ~ (":" p "$"){f=1} END{exit !f}'
}

as_root() {
    if (( DRY_RUN )); then note "dry-run: $*"; return 0; fi
    if (( EUID == 0 )); then "$@"; else sudo "$@"; fi
}

ensure_sudo() {
    (( SUDO_READY || DRY_RUN )) && return 0
    if (( EUID == 0 )); then SUDO_READY=1; return 0; fi
    command -v sudo >/dev/null 2>&1 || die '需要 root 或 sudo 来加载模块 / 启动 daemon。'
    sudo -v || die '需要 sudo 权限。'
    SUDO_READY=1
}

run() {  # dry-run 感知的执行器
    if (( DRY_RUN )); then note "dry-run: $*"; return 0; fi
    "$@"
}

# 供 pytest / 手动运行使用：仓库根 + 两个扩展的 in-place 产物目录
tutti_pythonpath() {
    printf '%s:%s/src:%s' "$TUTTI_ROOT" "$PYBIND_DIR" "$KVTRANSFER_DIR"
}

python_can_import() {
    PYTHONPATH="$(tutti_pythonpath)" "$PYTHON_BIN" -c "import $1" 2>/dev/null
}

python_version() {
    "$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?"
}

# 区分"扩展未构建"与"解释器与扩展 ABI 不匹配"——后者最容易浪费时间。
# $1=模块名 $2=搜索 .so 的目录
extension_state() {
    local module="$1" dir="$2" so tag
    if python_can_import "$module"; then printf 'ok'; return; fi
    so="$(find "$dir" -name '*.so' 2>/dev/null | head -1)"
    if [[ -z "$so" ]]; then printf 'missing'; return; fi
    tag="$(basename "$so" | sed -n 's/.*cpython-\([0-9]*\).*/\1/p')"
    if [[ -n "$tag" ]]; then
        local want="${tag:0:1}.${tag:1}"
        [[ "$want" != "$(python_version)" ]] && { printf 'abi:%s' "$want"; return; }
    fi
    printf 'broken'
}

# ---- 只读检查 ---------------------------------------------------------------

check_all() {
    local quiet="${1:-}" bad=0 pid nodes mounted expected runpath

    _row() {
        [[ "$quiet" == "--quiet" ]] && return 0
        local mark
        case "$2" in
            ok) mark='  OK  ' ;; skip) mark=' SKIP ' ;; warn) mark=' WARN ' ;;
            *) mark=' FAIL ' ;;
        esac
        printf '[%s] %-24s %s\n' "$mark" "$1" "$3"
    }

    # --- 宿主前置 ---
    # GPU-BAR 直通路径假设 phys == bus。IOMMU 启用但非 passthrough 会破坏该假设，
    # 现象是 DMA 落到错地址而非报错，故必须前置检查。
    if grep -qw 'iommu=pt' /proc/cmdline 2>/dev/null; then
        _row 'IOMMU passthrough' ok 'iommu=pt'
    elif [[ -n "$(ls -A /sys/class/iommu 2>/dev/null)" ]]; then
        _row 'IOMMU passthrough' bad 'IOMMU 已启用但非 passthrough：GPU-BAR 路径假设 phys==bus，需内核参数 iommu=pt'
        bad=$((bad + 1))
    else
        _row 'IOMMU passthrough' skip 'IOMMU 未启用（phys==bus 成立）'
    fi

    # --- 构建产物 ---
    if [[ -x "$DAEMON_BIN" ]]; then
        _row 'tutti_daemon 二进制' ok "$DAEMON_BIN"
    else
        _row 'tutti_daemon 二进制' bad "缺失（跑 $0 build）"; bad=$((bad + 1))
    fi
    if [[ -f "$SNVME_CORE_KO" && -f "$SNVME_KO" ]]; then
        _row '内核模块 .ko' ok 'snvme-core.ko + snvme.ko 已构建'
    else
        _row '内核模块 .ko' bad "缺失（跑 $0 build）"; bad=$((bad + 1))
    fi
    if [[ -f "$LIBNVM_SO" ]]; then
        _row libnvm.so ok "$LIBNVM_SO"
    else
        _row libnvm.so bad '缺失：Python 扩展的 RUNPATH 指向此处，缺则 import 失败'
        bad=$((bad + 1))
    fi

    # --- Python 解释器与扩展 ---
    local pyver state
    pyver="$(python_version)"
    if [[ "$pyver" == "?" ]]; then
        _row 'Python 解释器' bad "${PYTHON_BIN} 不可用"; bad=$((bad + 1))
    elif [[ "${pyver%%.*}" -ge 3 && "${pyver#*.}" -ge 10 ]]; then
        _row 'Python 解释器' ok "${PYTHON_BIN} (${pyver})"
    else
        _row 'Python 解释器' bad "${PYTHON_BIN} 是 ${pyver}，项目要求 >= 3.10（用 PYTHON_BIN= 指定或激活 venv）"
        bad=$((bad + 1))
    fi

    _ext_row() {  # $1=显示名 $2=模块 $3=目录
        local st; st="$(extension_state "$2" "$3")"
        case "$st" in
            ok)      _row "$1" ok '可导入' ;;
            missing) _row "$1" bad "未构建（跑 $0 build-ext）"; bad=$((bad + 1)) ;;
            abi:*)   _row "$1" bad ".so 是 cpython-${st#abi:} 而当前解释器 ${pyver}，ABI 不匹配"
                     bad=$((bad + 1)) ;;
            *)       _row "$1" bad "已构建但导入失败（多为依赖缺失，见 $0 build-ext 输出）"
                     bad=$((bad + 1)) ;;
        esac
    }
    _ext_row 'tutti_runtime 扩展' tutti_runtime "${PYBIND_DIR}/src"
    _ext_row 'tutti_kv_transfer 扩展' tutti_kv_transfer "$KVTRANSFER_DIR"
    # RUNPATH 与实际 .so 位置是否一致（构建树被删/迁移后最常见的坑）
    local core_so
    core_so="$(find "${PYBIND_DIR}/src" -name '_core*.so' 2>/dev/null | head -1)"
    if [[ -n "$core_so" ]] && command -v readelf >/dev/null 2>&1; then
        runpath="$(readelf -d "$core_so" 2>/dev/null |
                   sed -n 's/.*RUNPATH.*\[\(.*\)\]/\1/p' | tr ':' '\n' |
                   grep -m1 libnvm || true)"
        if [[ -z "$runpath" ]]; then
            _row 'RUNPATH -> libnvm' skip '未在 RUNPATH 中声明'
        elif [[ -f "${runpath}/libnvm.so" ]]; then
            _row 'RUNPATH -> libnvm' ok "$runpath"
        else
            _row 'RUNPATH -> libnvm' bad "指向 ${runpath} 但该处无 libnvm.so（构建树被删/迁移过）"
            bad=$((bad + 1))
        fi
    fi

    # --- 内核模块 ---
    if module_loaded phoenixfs; then
        _row phoenixfs ok '已加载'
    elif (( WANT_PHOENIX )); then
        _row phoenixfs skip '未加载（仅 Phoenix GDS 路径需要）'
    else
        _row phoenixfs skip '按 --no-phoenix 跳过'
    fi
    # phoenixfs 把整个 GPU BAR remap 出 struct page，使 NVIDIA peer-memory 的
    # dma_map_resource 拒绝该地址 -> RDMA 注册显存 EIO。本地 NVMe 路径不受影响，
    # 故只告警不计入异常：仅当该机器还要跑 RDMA 时才是阻塞问题。
    if module_loaded phoenixfs && module_loaded nvidia_peermem; then
        _row 'phoenixfs/peermem' warn '两者同时加载：RDMA 注册 GPU 显存会 EIO（本地 NVMe 路径不受影响；跑 RDMA 前 rmmod phoenixfs）'
    fi
    if module_loaded snvme_core && module_loaded snvme; then
        _row 'snvme 模块' ok 'snvme-core + snvme 已加载'
    else
        _row 'snvme 模块' bad "core=$(module_loaded snvme_core && echo Y || echo N) pci=$(module_loaded snvme && echo Y || echo N)"
        bad=$((bad + 1))
    fi
    if [[ -e /dev/snvm_control ]]; then
        _row /dev/snvm_control ok '存在'
    else
        _row /dev/snvm_control bad '缺失（模块未正确初始化）'; bad=$((bad + 1))
    fi

    # --- daemon（逐个配置）---
    local cfg name port kind mp src
    while IFS= read -r cfg; do
        name="$(cfg_name "$cfg")"
        if [[ ! -f "$cfg" ]]; then
            _row "daemon ${name}" bad "缺配置 ${cfg}"; bad=$((bad + 1)); continue
        fi
        port="$(cfg_port "$cfg")"; pid="$(cfg_pid "$cfg")"
        if [[ -z "$pid" ]]; then
            _row "daemon ${name}" bad "未运行（跑 $0 daemon start ${name}）"; bad=$((bad + 1)); continue
        elif port_listening "$port"; then
            _row "daemon ${name}" ok "PID=${pid} gRPC :${port}"
        else
            _row "daemon ${name}" bad "PID=${pid} 但 :${port} 未监听"; bad=$((bad + 1))
        fi
        if cfg_supervised "$cfg"; then
            local st; st="$(cat "/run/tutti-md/${name}/status" 2>/dev/null || echo unknown)"
            if [[ "$st" == up ]]; then
                _row "  ${name} 监管" ok "tutti-md.sh 监管中，级别 $(cfg_level "$cfg")（daemon 退出即自动退回内核 nvme）"
            else
                _row "  ${name} 监管" warn "未经 tutti-md.sh 启动（status=${st}）：daemon 退出后不会自动退回"
            fi
        fi
        # 应有挂载缺失时 KV 会落到根盘，现象只是"慢"，极难察觉
        while read -r kind mp; do
            [[ -n "$mp" ]] || continue
            src="$(findmnt -no SOURCE "$mp" 2>/dev/null || true)"
            if [[ "$kind" == disk && "$src" == /dev/snvme*n* ]]; then
                _row "  ${mp}" ok "$src"
            elif [[ "$kind" == array && "$src" == /dev/md* ]] &&
                 ! ls "/sys/block/$(basename "$(readlink -f "$src")")/slaves" 2>/dev/null | grep -qv '^snvme'; then
                _row "  ${mp}" ok "${src}（成员全为 snvme）"
            else
                _row "  ${mp}" bad "${src:-未挂载}（应为 daemon 管理的 ${kind}）"; bad=$((bad + 1))
            fi
        done < <(cfg_expected_mounts "$cfg")
    done < <(daemon_configs)

    # 块设备节点由 daemon bring-up 创建，是 "daemon 真的干完活了" 的证据
    nodes="$(ls /dev/snvme*n1 2>/dev/null | wc -l)"
    if (( nodes > 0 )); then
        _row '块设备节点' ok "${nodes} 个 /dev/snvme*n1"
    else
        _row '块设备节点' bad '无 /dev/snvme*n1（daemon 未完成 bring-up）'; bad=$((bad + 1))
    fi

    return "$bad"
}

# ---- 构建 -------------------------------------------------------------------

cmd_build() {
    command -v cmake >/dev/null 2>&1 || die '未找到 cmake。'
    if (( DO_CLEAN )); then
        note "删除 ${TUTTI_BUILD} 后全新配置。"
        run rm -rf "$TUTTI_BUILD"
    fi
    note "配置（preset default：RelWithDebInfo + 内核模块 + 硬件测试）。"
    ( cd "$TUTTI_ROOT" && run cmake --preset default ) || die 'cmake 配置失败。'
    note "构建全部目标，并行度 $(jobs_arg)。modules 是 ALL 目标，一并构建。"
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --parallel "$(jobs_arg)" ) \
        || die 'C++ 构建失败。'
    (( DRY_RUN )) && return 0
    [[ -x "$DAEMON_BIN" ]] || warn "构建结束但缺 ${DAEMON_BIN}"
    [[ -f "$LIBNVM_SO" ]] || warn "构建结束但缺 ${LIBNVM_SO}"
    note '构建完成。'
}

cmd_build_ext() {
    [[ -f "$LIBNVM_SO" ]] || die "缺 ${LIBNVM_SO}；先跑 $0 build。"
    # 必须清干净再建：build_ext 会复用旧对象文件，导致 RUNPATH 仍指向旧构建树。
    note '构建 pybind 扩展 tutti_runtime。'
    run rm -rf "${PYBIND_DIR}/build"
    if (( ! DRY_RUN )); then find "${PYBIND_DIR}/src" -name '*.so' -delete; fi
    ( cd "$PYBIND_DIR" && run env TUTTI_ROOT="$TUTTI_ROOT" TUTTI_BUILD_DIR="$TUTTI_BUILD" \
        "$PYTHON_BIN" setup.py build_ext --inplace ) || die 'tutti_runtime 构建失败。'
    note '构建 CUDA 扩展 tutti_kv_transfer。'
    run rm -rf "${KVTRANSFER_DIR}/build"
    if (( ! DRY_RUN )); then find "$KVTRANSFER_DIR" -name '*.so' -delete; fi
    ( cd "$KVTRANSFER_DIR" && run "$PYTHON_BIN" setup.py build_ext --inplace ) \
        || die 'tutti_kv_transfer 构建失败（需要 torch + nvcc）。'
    (( DRY_RUN )) && return 0
    python_can_import tutti_runtime || die 'tutti_runtime 构建后仍不可导入。'
    python_can_import tutti_kv_transfer || die 'tutti_kv_transfer 构建后仍不可导入。'
    note '两个 Python 扩展就绪。'
}

# ---- 内核模块 ---------------------------------------------------------------

modules_rebuild() {
    note '重建内核模块（clean_modules + modules）。'
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target clean_modules ) || true
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target modules \
        --parallel "$(jobs_arg)" ) || die '内核模块构建失败。'
    note '内核模块已重建；load 前请确认已按站点流程签名。'
}

modules_load() {
    local core=0 pci=0
    module_loaded snvme_core && core=1
    module_loaded snvme && pci=1
    (( core && pci )) && { note 'snvme-core 和 snvme 已加载。'; return 0; }
    (( core || pci )) && die '检测到半加载状态；请先 modules unload 清理。'
    [[ -f "$SNVME_CORE_KO" && -f "$SNVME_KO" ]] || die "缺内核模块；先跑 $0 build。"
    ensure_sudo
    note '加载 snvme-core 和 snvme（insmod target 只转发 make insmod，不构建）。'
    if ! ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target insmod ); then
        warn '加载失败。若 dmesg 出现 "Required key not available" / signature，'
        warn "请签名以下模块后重试： ${SNVME_CORE_KO} ${SNVME_KO}"
        die 'SNVMe 模块加载失败。'
    fi
    (( DRY_RUN )) && return 0
    module_loaded snvme_core || die 'snvme-core 未加载。'
    module_loaded snvme || die 'snvme 未加载。'
    [[ -e /dev/snvm_control ]] || die '模块已加载但 /dev/snvm_control 不存在。'
}

modules_unload() {
    module_loaded snvme || module_loaded snvme_core || { note 'snvme 模块未加载。'; return 0; }
    # daemon 持有模块引用；不先停它 rmmod 必然 EBUSY。只自动停本次管理的
    # daemon，其余的（如 md0）必须由人显式停，不能连带着把阵列交回去。
    if [[ -n "$(all_daemons)" ]]; then
        warn 'tutti_daemon 在运行，会持有模块引用。'
        stop_daemons
        if [[ -n "$(all_daemons)" ]] && (( ! DRY_RUN )); then
            all_daemons >&2
            die "仍有 daemon 在运行（未在本次管理范围内）；先 $0 daemon stop <NAME>。"
        fi
    fi
    ensure_sudo
    note '卸载 snvme 与 snvme-core。卸载后 /dev/snvme* 与 /dev/ssnvme* 全部消失。'
    ( cd "$TUTTI_ROOT" && run cmake --build --preset default --target rmmod ) \
        || die '卸载失败（可能仍有进程占用）。'
}

ensure_phoenix() {
    (( WANT_PHOENIX )) || { note '按 --no-phoenix 跳过 phoenixfs。'; return 0; }
    module_loaded phoenixfs && { note 'phoenixfs 已加载。'; return 0; }
    if [[ ! -f "$PHOENIX_KO" ]]; then
        warn "缺 ${PHOENIX_KO}，跳过（仅影响 Phoenix GDS 路径）。"
        warn "需要它请先手动构建： cd ${PHOENIX_BUILD} && make modules"
        return 0
    fi
    ensure_sudo
    note '加载 phoenixfs。'
    if (( EUID == 0 )); then
        run insmod "$PHOENIX_KO" || { warn 'phoenixfs 加载失败（可能需签名）。'; return 0; }
    else
        ( cd "$PHOENIX_BUILD" && run make insmod ) \
            || { warn 'phoenixfs 加载失败（可能需签名）。'; return 0; }
    fi
}

# ---- daemon -----------------------------------------------------------------

start_one() {  # $1=配置
    local cfg="$1" name port pid launcher log
    name="$(cfg_name "$cfg")"; port="$(cfg_port "$cfg")"
    [[ -f "$cfg" ]] || die "缺配置 ${cfg}"
    if pid="$(cfg_pid "$cfg")"; [[ -n "$pid" ]]; then
        note "daemon ${name} 已运行，PID=${pid}，端口=${port}。"; return 0
    fi
    [[ -x "$DAEMON_BIN" ]] || die "缺 ${DAEMON_BIN}；先跑 $0 build。"
    module_loaded snvme || die "snvme 未加载；先跑 $0 modules load。"
    ensure_sudo
    if cfg_supervised "$cfg"; then
        check_level "$cfg" start || return 1
        # 阵列交接（内核 nvme/md -> snvme + daemon）与失败回退都在 tutti-md.sh 里
        note "daemon ${name}（$(cfg_level "$cfg")）：经 tutti-md.sh 交接阵列并监管（日志 /var/log/tutti-md-${name}.log）。"
        as_root "$MD_SCRIPT" "$name" up --confirm "$name" || die "daemon ${name} 未能接管，阵列已退回内核（见 /var/log/tutti-md-${name}.log）。"
        return 0
    fi
    cfg_has_arrays "$cfg" && die "${cfg} 含 raid0_arrays，但 tutti-md.sh 不认识 ${name}：先在其中登记该阵列（PCI/UUID/挂载点），否则不能安全交接。"
    log="${LOG_DIR}/tutti_daemon_${name}.log"
    mkdir -p "$LOG_DIR"
    if (( DRY_RUN )); then
        note "dry-run: nohup sudo env TUTTI_VERBOSE=1 ${DAEMON_BIN} --config ${cfg} >> ${log} 2>&1 &"
        return 0
    fi
    printf '\n===== %s: starting tutti_daemon %s =====\n' "$(date --iso-8601=seconds)" "$name" >> "$log"
    note "启动 daemon ${name}，日志：${log}"
    if (( EUID == 0 )); then
        nohup env TUTTI_VERBOSE=1 "$DAEMON_BIN" --config "$cfg" >> "$log" 2>&1 < /dev/null &
    else
        nohup sudo env TUTTI_VERBOSE=1 "$DAEMON_BIN" --config "$cfg" >> "$log" 2>&1 < /dev/null &
    fi
    launcher=$!
    for _ in $(seq 1 30); do
        pid="$(cfg_pid "$cfg")"
        if [[ -n "$pid" ]] && port_listening "$port"; then
            note "daemon ${name} 就绪，PID=${pid}，端口=${port}。"; return 0
        fi
        if ! kill -0 "$launcher" 2>/dev/null && [[ -z "$pid" ]]; then
            tail -n 80 "$log" >&2 || true
            die "daemon ${name} 启动期间退出；日志：${log}"
        fi
        sleep 1
    done
    tail -n 80 "$log" >&2 || true
    die "daemon ${name} 30 秒内未监听 ${port}；日志：${log}"
}

stop_one() {  # $1=配置
    local cfg="$1" name pid
    name="$(cfg_name "$cfg")"
    if cfg_supervised "$cfg"; then
        [[ -n "$(cfg_pid "$cfg")" ]] || { note "daemon ${name} 未运行。"; return 0; }
        check_level "$cfg" stop || return 1
        # 只发一次 SIGTERM：daemon 卸载、停阵列，挂载点忙就一直等；监管者随后把
        # 成员盘还给内核 nvme 驱动并重新组装/挂载原阵列。仍有 Tutti 客户端在用
        # 这几块盘时 tutti-md.sh 直接拒绝。
        ensure_sudo
        note "daemon ${name}（$(cfg_level "$cfg")）：停止并把阵列还给 Linux（tutti-md.sh down）。"
        as_root "$MD_SCRIPT" "$name" down --confirm "$name" || { warn "daemon ${name} 未完全退回（见 /var/log/tutti-md-${name}.log）。"; return 1; }
        return 0
    fi
    pid="$(cfg_pid "$cfg")"
    [[ -z "$pid" ]] && { note "daemon ${name} 未运行。"; return 0; }
    ensure_sudo
    note "停止 daemon ${name} PID=${pid}（SIGTERM，触发其 umount 流程）。"
    as_root kill -TERM "$pid"
    (( DRY_RUN )) && return 0
    for _ in $(seq 1 30); do
        [[ -z "$(cfg_pid "$cfg")" ]] && { note "daemon ${name} 已停止。"; return 0; }
        sleep 1
    done
    # 不补第二个信号：那会让 daemon 在文件系统仍挂载时释放控制器
    warn "daemon ${name} 30 秒内未退出（多为挂载点被占用，它会继续等待）；见 ${LOG_DIR}/tutti_daemon_${name}.log"
    return 1
}

start_daemons() {
    local c rc=0
    while IFS= read -r c; do start_one "$c" || rc=1; done < <(daemon_configs)
    return "$rc"
}

stop_daemons() {  # 逆序停
    local c rc=0
    while IFS= read -r c; do stop_one "$c" || rc=1; done < <(daemon_configs | tac)
    return "$rc"
}

daemon_status() {
    local cfg name pid
    while IFS= read -r cfg; do
        name="$(cfg_name "$cfg")"; pid="$(cfg_pid "$cfg")"
        printf '%-6s %-8s port=%-6s %-10s %s%s\n' "$name" "${pid:-stopped}" "$(cfg_port "$cfg")" \
            "$(cfg_level "$cfg")" "$(cfg_supervised "$cfg" && echo 'supervised ' || true)" "$cfg"
    done < <(daemon_configs)
    note '本机所有 tutti_daemon 进程：'
    all_daemons | sed 's/^/  /'
}

# ---- 测试 -------------------------------------------------------------------

cmd_test() {
    local what="${1:-all}" rc=0
    case "$what" in
        py|all)
            note '跑 Python 测试（tests/python）。'
            ( cd "$TUTTI_ROOT" && run env -u PYTHONPATH PYTHONPATH="$(tutti_pythonpath)" \
                "$PYTHON_BIN" -m pytest tests/python -q ) || rc=1
            ;;
    esac
    case "$what" in
        cpp|all)
            note '跑 C++ 测试（ctest）。硬件契约测试需要环境就绪。'
            if ! check_all --quiet; then
                warn "环境不完整，硬件契约测试会失败。先跑 $0 status 查看。"
            fi
            ( cd "$TUTTI_BUILD" && run ctest --output-on-failure ) || rc=1
            ;;
    esac
    case "$what" in
        py|cpp|all) ;;
        *) die "test 的参数须为 py|cpp|all，收到：${what}" ;;
    esac
    return "$rc"
}

# ---- 组合动作 ---------------------------------------------------------------

cmd_up() {
    note "Tutti 根目录：${TUTTI_ROOT}"
    ensure_phoenix
    modules_load
    local started=0
    start_daemons && started=1
    (( DRY_RUN )) && { note 'dry-run 结束。'; return 0; }
    (( started )) || die '有 daemon 未启动（见上；protected 阵列需 --confirm NAME）。'
    echo
    if check_all; then note '环境就绪。'; else die '环境仍不完整（见上表）。'; fi
}

cmd_bootstrap() {
    note '从零 bootstrap：build -> build-ext -> up'
    cmd_build
    cmd_build_ext
    cmd_up
}

cmd_env() {
    printf 'export PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}"\n' "$(tutti_pythonpath)"
    printf 'export TUTTI_ROOT="%s"\n' "$TUTTI_ROOT"
    printf 'export TUTTI_BUILD_DIR="%s"\n' "$TUTTI_BUILD"
}

main() {
    local subcmd="" arg=""
    while (( $# )); do
        case "$1" in
            status|bootstrap|build|build-ext|modules|daemon|up|down|test|env)
                if [[ -z "$subcmd" ]]; then subcmd="$1"; else arg="$1"; fi ;;
            load|unload|rebuild|start|stop|restart|py|cpp|all) arg="$1" ;;
            --dry-run)    DRY_RUN=1 ;;
            --no-phoenix) WANT_PHOENIX=0 ;;
            --clean)      DO_CLEAN=1 ;;
            --confirm)    shift; CONFIRMED+=("${1:?--confirm 需要阵列名}") ;;
            -j)           shift; JOBS="${1:?-j 需要数值}" ;;
            -j*)          JOBS="${1#-j}" ;;
            -h|--help)    usage; return 0 ;;
            -*) usage >&2; die "未知参数：$1" ;;
            *)
                # daemon / up / down 之后的裸名字 = daemon 名
                if [[ "$subcmd" =~ ^(daemon|up|down)$ ]] && [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]]; then
                    [[ -f "$(config_of_name "$1")" ]] || die "无配置 $(config_of_name "$1")"
                    DAEMON_NAMES+=("$1")
                else
                    usage >&2; die "未知参数：$1"
                fi ;;
        esac
        shift
    done
    case "${subcmd:-status}" in
        status)    check_all || die "环境不完整（$? 项异常）" ;;
        bootstrap) cmd_bootstrap ;;
        build)     cmd_build ;;
        build-ext) cmd_build_ext ;;
        modules)
            case "${arg:-}" in
                load) modules_load ;; unload) modules_unload ;; rebuild) modules_rebuild ;;
                *) die 'modules 须带 load|unload|rebuild' ;;
            esac ;;
        daemon)
            case "${arg:-}" in
                start) start_daemons || die '有 daemon 未启动（见上）。' ;;
                stop)  stop_daemons || die '有 daemon 未停止（见上）。' ;;
                restart) stop_daemons || die '停止未完成，不重启。'; start_daemons ;;
                status) daemon_status ;;
                *) die 'daemon 须带 start|stop|restart|status [NAME...]' ;;
            esac ;;
        up)   cmd_up ;;
        down) stop_daemons || die '有 daemon 未停止（见上）。' ;;
        test) cmd_test "${arg:-all}" ;;
        env)  cmd_env ;;
    esac
}

main "$@"
