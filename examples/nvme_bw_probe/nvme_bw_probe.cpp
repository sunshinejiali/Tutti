#define _GNU_SOURCE
// nvme_bw_probe.cpp -- single-drive small-IO read-bandwidth probe.
//
// One file on ONE drive, one registered GPU buffer, N x --io-kb KiB random
// (or sequential) reads per batch.  --depth D = one submit() of D entries
// (burst model: all D in flight at once, kernel exits when all D are done).
// Set TUTTI_POOL_WORKERS=W to switch the datapath kernel to the worker-pool
// model (W fixed workers drain the batch; W is the in-flight bound).
//
// Reference points (same drive, fio libaio direct randread):
//   32K QD1   0.70 GB/s    32K QD128  6.12 GB/s    32K QD512 6.75 GB/s
//   16K QD512 5.9  GB/s    64K QD128  6.75 GB/s

#include <tutti/tutti_runtime.h>

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>

#include <tutti/cuda_like.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tutti;

#define PROBE_OK(...) do { std::printf("[ OK ] " __VA_ARGS__); std::printf("\n"); } while (0)
#define PROBE_FAIL(...) do { std::fprintf(stderr, "[FAIL] " __VA_ARGS__); std::fprintf(stderr, "\n"); return 1; } while (0)
#define CUDA_OK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    std::fprintf(stderr,"[FAIL] %s: %s\n",#call,cudaGetErrorString(_e)); return 1;} } while (0)

static double sec_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static bool create_file(const std::string& path, std::uint64_t size) {
    // Project policy: ALL file opens carry O_DIRECT.
    int f = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (f < 0) return false;
    // fallocate pre-reserves blocks as a few large extents; the zero-fill
    // write then converts unwritten -> written WITHOUT adding extents (the
    // resolver rejects unwritten extents and caps the extent count).
    if (::fallocate(f, 0, 0, (off_t)size) != 0) {
        ::close(f);
        return false;
    }
    void* ap = nullptr;
    if (::posix_memalign(&ap, 4096, 1 << 20) != 0) { ::close(f); return false; }
    std::memset(ap, 0, 1 << 20);
    std::uint64_t off = 0;
    while (off < size) {
        size_t n = std::min<std::uint64_t>(1 << 20, size - off);
        ssize_t w = ::pwrite(f, ap, n, (off_t)off);
        if (w != (ssize_t)n) { std::free(ap); ::close(f); return false; }
        off += n;
    }
    std::free(ap);
    ::fsync(f);
    ::close(f);
    return true;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

// Reuse is only safe when the existing file is exactly the size this run
// needs: --kv-targets changes the per-file size, and silently reusing a
// stale smaller file would push reads past EOF.
static bool file_is_size(const std::string& path, std::uint64_t want) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 &&
           (std::uint64_t)st.st_size == want;
}

// Existence only (a barrier flag file is zero-length, so file_exists() above,
// which requires a non-empty file, cannot be reused here).
static bool path_exists(const std::string& path) {
    return ::access(path.c_str(), F_OK) == 0;
}

static std::int64_t epoch_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool ensure_directory(const std::string& path) {
    if (path.empty()) return false;
    if (path == "/") return true;
    for (std::size_t i = 1; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/') continue;
        const std::string partial = path.substr(0, i);
        if (partial.empty() || partial == "/") continue;
        if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            return false;
    }
    return true;
}

int main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> directories;
    std::string mode = "rand";
    std::uint32_t io_kb = 32;
    std::uint32_t depth = 128;
    std::uint32_t total_gb = 32;
    std::uint32_t file_gb = 64;
    std::uint32_t buf_mb = 512;
    bool buf_mb_set = false;
    std::uint32_t write_pct = 0;
    std::uint32_t split_bufs = 0;
    std::uint32_t seed = 7;
    bool fresh = false;
    std::string barrier;
    // ---- KV mode: mirror the production vLLM read plan ----
    // Production geometry (tutti/storage/tutti_nvme/store.py:783-799, confirmed
    // against the NVTX marker "chunks=511|requests=1022"):
    //   one chunk  = one slot file = one drive (rotating placement)
    //   one submit = one LAYER across ALL chunks
    //   per chunk per layer -> blocks_per_chunk requests of page_bytes each
    //   memory_offset = block_id*block_stride + layer*layer_stride
    //   target_offset = header + layer*segment_bytes + ordinal*page_bytes
    // The two blocks of a chunk are contiguous on the drive but
    // block_stride apart in the pool, which is exactly why production
    // cannot coalesce them into one page_bytes*blocks_per_chunk IO.
    std::uint32_t kv_chunks = 0;        // 0 = classic probe mode
    std::uint32_t kv_layers = 80;
    // 0 = no chunking, i.e. one submit carries every chunk of the layer.
    // This is what production does: the direct path (tutti/engine/core.py:1378-1402)
    // hands all keys to load_layer() in one call and returns.  The
    // max_chunks_per_wave loop at core.py:1405 sits *after* that return and
    // only runs for the staging path, which needs waves because the ring
    // window has a bounded capacity_per_wave.  Keep this 0 unless you are
    // deliberately modelling staging.
    std::uint32_t kv_wave_chunks = 0;
    // 0 = one target per chunk (production).  Setting N packs the chunks into
    // N files, ceil(chunks/N) chunks per file at distinct offsets, so batch
    // width, IO size, pool offsets AND the set of distinct LBAs all stay put
    // and only "how many files a kernel touches" changes.  (Folding with a
    // plain modulo would instead collapse every folded chunk onto the same
    // LBA, turning the run into a same-block re-read.)
    std::uint32_t kv_targets = 0;
    // Shuffle the entry order within each submit.  The byte set, the IO size
    // and the batch width are all unchanged -- only which requests are in
    // flight together changes.  Production emits a layer's chunks in chunk
    // order, and chunk files are created in order, so consecutive entries sit
    // at a near-constant stride on the drive; this knob tests whether that
    // regularity is what costs bandwidth.
    bool kv_shuffle = false;
    // Per-slot random padding, in 4 KiB units, added to each slot file's size
    // so consecutive slots no longer sit at a constant physical stride.
    //
    // Measured on this box: 82% of consecutive 10 MiB slot files land exactly
    // 2561 blocks (10.00 MiB) apart, because ext4 packs same-sized files
    // created in sequence.  A read plan touches the same relative offset in
    // every slot, so a constant namespace-LBA stride is visible to the SSD.
    // The NAND channel/die mapping is FTL-private; this knob tests the stride
    // effect without claiming a particular internal mapping.
    std::uint32_t kv_pad_max_kb = 0;
    bool kv_pad_fixed = false;   // pad every slot identically (control)
    std::uint32_t kv_header_skew_kb = 0; // per-slot payload start skew
    bool kv_header_skew_fixed_size = false;
    std::uint32_t kv_blocks_per_chunk = 2;
    std::uint32_t kv_block_stride_kb = 5120;   // 5242880 B
    std::uint32_t kv_layer_stride_kb = 64;     // 65536 B
    std::uint32_t kv_header_kb = 4;            // 4096 B slot header

    for (int i = 1; i < argc;) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--directory") && i + 1 < argc) { directories.emplace_back(argv[++i]); ++i; }
        else if (!std::strcmp(a, "--config") && i + 1 < argc) { config_path = argv[++i]; ++i; }
        else if (!std::strcmp(a, "--io-kb") && i + 1 < argc) { io_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--depth") && i + 1 < argc) { depth = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--total-gb") && i + 1 < argc) { total_gb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--file-gb") && i + 1 < argc) { file_gb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--buf-mb") && i + 1 < argc) { buf_mb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); buf_mb_set = true; ++i; }
        else if (!std::strcmp(a, "--write-pct") && i + 1 < argc) { write_pct = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--split-bufs") && i + 1 < argc) { split_bufs = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--seed") && i + 1 < argc) { seed = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--barrier") && i + 1 < argc) { barrier = argv[++i]; ++i; }
        else if (!std::strcmp(a, "--kv-chunks") && i + 1 < argc) { kv_chunks = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-layers") && i + 1 < argc) { kv_layers = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-wave-chunks") && i + 1 < argc) { kv_wave_chunks = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-targets") && i + 1 < argc) { kv_targets = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-header-kb") && i + 1 < argc) { kv_header_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-shuffle")) { kv_shuffle = true; ++i; }
        else if (!std::strcmp(a, "--kv-pad-max-kb") && i + 1 < argc) { kv_pad_max_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-pad-fixed")) { kv_pad_fixed = true; ++i; }
        else if (!std::strcmp(a, "--kv-header-skew-kb") && i + 1 < argc) { kv_header_skew_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-header-skew-fixed-size")) { kv_header_skew_fixed_size = true; ++i; }
        else if (!std::strcmp(a, "--kv-blocks-per-chunk") && i + 1 < argc) { kv_blocks_per_chunk = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-block-stride-kb") && i + 1 < argc) { kv_block_stride_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--kv-layer-stride-kb") && i + 1 < argc) { kv_layer_stride_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--rand")) { mode = "rand"; ++i; }
        else if (!std::strcmp(a, "--sequential")) { mode = "seq"; ++i; }
        else if (!std::strcmp(a, "--fresh")) { fresh = true; ++i; }
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
            std::printf("usage: %s --directory DIR [--directory DIR ...] [--config PATH]\n"
                        "  [--rand|--sequential] [--io-kb N] [--depth D] [--total-gb N]\n"
                        "  [--file-gb N] [--buf-mb N] [--seed N] [--fresh] [--barrier PATH]\n"
                        "  1 directory  -> single-drive (local-nvme)\n"
                        "  >=2 (power of two) directories -> rotating over drives (striped)\n"
                        "  --config defaults to the matching built-in YAML for each mode\n"
                        "  --barrier PATH: multi-process start barrier.  Writes\n"
                        "    PATH.ready.<pid> after warmup, then blocks until PATH.go\n"
                        "    exists.  Without it, concurrent instances' measured windows\n"
                        "    are skewed by seconds of startup and a summed aggregate\n"
                        "    overcounts.\n"
                        "\n"
                        "KV mode (mirrors the production vLLM read plan):\n"
                        "  --kv-chunks N              N slot files rotating over the drives;\n"
                        "                             enables KV mode (one submit = one layer\n"
                        "                             across all N chunks).  --io-kb becomes\n"
                        "                             page_bytes and --depth is ignored.\n"
                        "  --kv-layers L              layers per chunk (default 80)\n"
                        "  --kv-wave-chunks N         chunks per submit (default 0 = all,\n"
                        "                             which is what the direct path does;\n"
                        "                             waves only exist for staging)\n"
                        "  --kv-targets N             pack chunks into N files (default 0 =\n"
                        "                             one per chunk); isolates target count\n"
                        "                             at constant batch width, offsets and LBAs\n"
                        "  --kv-header-kb N           slot header before the payload\n"
                        "                             (default 4, production's payload_offset).\n"
                        "                             A 4 KiB header leaves every IO 4 KiB- but\n"
                        "                             not page_bytes-aligned; set 0 to measure\n"
                        "                             what that misalignment costs.\n"
                        "  --kv-header-skew-kb N      per-slot deterministic header skew in KiB\n"
                        "                             (0 = disabled; grows each file by its skew)\n"
                        "  --kv-header-skew-fixed-size  all files equally sized at maximum skew;\n"
                        "                             isolates read offsets from file stride\n"
                        "  --kv-blocks-per-chunk B    requests per chunk per layer (default 2)\n"
                        "  --kv-block-stride-kb N     pool stride between blocks (default 5120)\n"
                        "  --kv-layer-stride-kb N     pool stride between layers (default 64)\n"
                        "  Production reference: --kv-chunks 2783 --io-kb 64 --kv-layers 80\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown: %s (try --help)\n", a);
            return 1;
        }
    }
    if (directories.empty()) PROBE_FAIL("--directory is required");
    if (io_kb == 0 || (io_kb * 1024ull) % 4096 != 0) PROBE_FAIL("--io-kb must be 4 KiB-aligned");
    if (depth == 0) PROBE_FAIL("--depth must be > 0");

    const std::size_t ndev = directories.size();
    if (ndev > 1 && ((ndev & (ndev - 1)) != 0))
        PROBE_FAIL("multi-drive mode requires a power-of-two --directory count (got %zu)", ndev);

    const std::uint64_t io_bytes = (std::uint64_t)io_kb * 1024;
    const std::uint64_t buf_bytes = (std::uint64_t)buf_mb << 20;

    // ---- KV mode geometry ----
    const bool kv_mode = (kv_chunks > 0);
    // Distinct files actually opened.  Production is one per chunk; folding
    // them onto fewer files keeps every request byte-identical while shrinking
    // the number of targets a single kernel dereferences.
    const std::uint32_t kv_files =
        (kv_targets == 0 || kv_targets > kv_chunks) ? kv_chunks : kv_targets;
    const std::uint64_t kv_page_bytes    = io_bytes;
    const std::uint64_t kv_segment_bytes = kv_page_bytes * kv_blocks_per_chunk;
    const std::uint64_t kv_block_stride  = (std::uint64_t)kv_block_stride_kb * 1024;
    const std::uint64_t kv_layer_stride  = (std::uint64_t)kv_layer_stride_kb * 1024;
    const std::uint64_t kv_header_bytes  = (std::uint64_t)kv_header_kb * 1024;
    // One slot file holds every layer of one chunk, matching the production
    // object layout (4 KiB header then layer-major segments).
    const std::uint64_t kv_slot_bytes =
        kv_header_bytes + (std::uint64_t)kv_layers * kv_segment_bytes;
    const std::uint64_t kv_header_skew_bytes =
        (std::uint64_t)kv_header_skew_kb * 1024;
    // With --kv-targets the chunks are packed several-per-file, each at its own
    // slot offset, so the distinct-LBA set is unchanged.
    const std::uint32_t kv_slots_per_file =
        kv_mode ? (kv_chunks + kv_files - 1) / kv_files : 1;
    const std::uint64_t kv_file_bytes = kv_slot_bytes * kv_slots_per_file;
    // Pool footprint the requests address into.  Blocks are numbered
    // globally, so the highest offset any request produces is
    // (last_block)*block_stride + (last_layer)*layer_stride + page_bytes.
    const std::uint64_t kv_blocks_total =
        (std::uint64_t)kv_chunks * kv_blocks_per_chunk;
    const std::uint64_t kv_pool_bytes = kv_mode
        ? (kv_blocks_total - 1) * kv_block_stride
          + (std::uint64_t)(kv_layers - 1) * kv_layer_stride + kv_page_bytes
        : 0;

    if (kv_mode) {
        if (kv_header_skew_kb % 4 != 0 || kv_header_kb % 4 != 0 || kv_pad_max_kb % 4 != 0)
            PROBE_FAIL("KV header, skew and padding must be 4 KiB-aligned");
        if (kv_header_skew_fixed_size && (kv_header_skew_kb == 0 || kv_files != kv_chunks))
            PROBE_FAIL("--kv-header-skew-fixed-size requires skew and one file per chunk");
        if (kv_layers == 0 || kv_blocks_per_chunk == 0)
            PROBE_FAIL("--kv-layers and --kv-blocks-per-chunk must be > 0");
        if (kv_layer_stride == 0 || kv_layer_stride % 4096 != 0)
            PROBE_FAIL("--kv-layer-stride-kb must be non-zero and 4 KiB-aligned");
        if (kv_block_stride % 4096 != 0)
            PROBE_FAIL("--kv-block-stride-kb must be 4 KiB-aligned");
        // A layer's slice of one block must fit in the layer stride, else
        // consecutive layers would overlap in the pool.
        if (kv_page_bytes > kv_layer_stride)
            PROBE_FAIL("page_bytes (%llu) exceeds layer stride (%llu)",
                       (unsigned long long)kv_page_bytes,
                       (unsigned long long)kv_layer_stride);
        if ((std::uint64_t)kv_layers * kv_layer_stride > kv_block_stride)
            PROBE_FAIL("layers*layer_stride (%llu) exceeds block stride (%llu)",
                       (unsigned long long)kv_layers * kv_layer_stride,
                       (unsigned long long)kv_block_stride);
    }

    const std::uint64_t file_bytes = kv_mode ? kv_file_bytes
                                             : ((std::uint64_t)file_gb << 30);
    // KV mode reads the whole working set exactly once per round, like a
    // production read plan; --total-gb does not apply.
    const std::uint64_t total_bytes = kv_mode
        ? (std::uint64_t)kv_chunks * kv_layers * kv_segment_bytes
        : ((std::uint64_t)total_gb << 30);
    const std::uint64_t total_ios = total_bytes / io_bytes;
    if (!kv_mode && io_bytes > buf_bytes) PROBE_FAIL("--io-kb exceeds buffer");

    if (config_path.empty()) {
        config_path = (ndev == 1) ? TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG
                                  : TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG_STRIPED;
    }

    const char* pool_env = std::getenv("TUTTI_POOL_WORKERS");
    const long pool_workers = pool_env ? std::atol(pool_env) : 2048;

    // ---- Runtime ----
    auto created = TuttiRuntime::create(config_path);
    if (!created.ok())
        PROBE_FAIL("TuttiRuntime::create(%s): %s", config_path.c_str(),
                   created.status().message().c_str());
    std::unique_ptr<TuttiRuntime> owner = std::move(created).value();
    StorageRuntime* rt = owner->storage_runtime();
    if (rt == nullptr) PROBE_FAIL("no StorageRuntime");
    const int32_t gpu = rt->accel_id();
    if (gpu < 0) PROBE_FAIL("runtime.accel_id unspecified");
    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(gpu));
    PROBE_OK("runtime up (%s, %zu drive%s, config=%s)",
             ndev == 1 ? "local-nvme" : "nvme-device-group", ndev,
             ndev == 1 ? "" : "s", config_path.c_str());

    // ---- Backing files ----
    // Classic mode: one big file per drive.
    // KV mode: kv_chunks per-chunk files rotating over the drives
    // (chunk C -> drive C % ndev, path <mount>/r<gpu>/C.obj).
    // The per-GPU subdirectory is required: with 8 instances sharing one set of
    // drives the slot numbers are independent, and without it they would
    // overwrite each other.
    std::vector<std::string> kv_paths;
    if (!kv_mode) {
        for (std::size_t d = 0; d < ndev; ++d) {
            const std::string fpath = directories[d] + "/nvme_bw_probe.bin";
            if (fresh || !file_exists(fpath)) {
                auto t0 = std::chrono::steady_clock::now();
                if (!create_file(fpath, file_bytes))
                    PROBE_FAIL("create_file %s: %s", fpath.c_str(), std::strerror(errno));
                PROBE_OK("backing file %s (%u GiB) in %.2fs", fpath.c_str(), file_gb,
                         sec_since(t0));
            }
        }
        PROBE_OK("backing files ready (%u GiB each, reuse; --fresh to recreate)", file_gb);
    } else {
        const std::string sub = "/kvprobe_gpu" + std::to_string((int)gpu);
        for (std::size_t d = 0; d < ndev; ++d) {
            const std::string dir = directories[d] + sub;
            if (!ensure_directory(dir))
                PROBE_FAIL("mkdir %s: %s", dir.c_str(), std::strerror(errno));
        }
        // Padding lives in a separate subdirectory so a padded run cannot be
        // confused with an unpadded one: the two need different physical
        // layouts, and file_is_size() would happily reuse the wrong set.
        const std::string layout_tag =
            "_hskew" + std::to_string(kv_header_skew_kb) +
            (kv_header_skew_fixed_size ? "_equal" : "") +
            (kv_pad_max_kb ? (kv_pad_fixed ? "_padfix" : "_pad") +
                              std::to_string(kv_pad_max_kb) : "_pad0") +
            "_s" + std::to_string(seed);
        const std::string sub_pad = sub + layout_tag;
        for (std::size_t d = 0; d < ndev; ++d) {
            const std::string dir = directories[d] + sub_pad;
            if (!ensure_directory(dir))
                PROBE_FAIL("mkdir %s: %s", dir.c_str(), std::strerror(errno));
        }
        kv_paths.reserve(kv_files);
        auto t0 = std::chrono::steady_clock::now();
        std::uint32_t created_n = 0;
        std::mt19937_64 pad_rng(seed ^ 0xC0FFEEull);
        for (std::uint32_t c = 0; c < kv_files; ++c) {
            const std::string fpath = directories[c % ndev] + sub_pad + "/" +
                                      std::to_string(c) + ".obj";
            kv_paths.push_back(fpath);
            // Padding only grows the file past the payload, so every request
            // computed below still lands inside it.
            // Pad in whole 4 KiB pages: create_file writes with O_DIRECT, so
            // a non-page-multiple size fails with EINVAL.
            const std::uint64_t header_skew = kv_header_skew_bytes
                ? ((static_cast<std::uint64_t>(c) * 0x9E3779B97F4A7C15ull + seed) %
                   (kv_header_skew_bytes / 4096 + 1)) * 4096
                : 0;
            std::uint64_t want = kv_file_bytes +
                (kv_header_skew_fixed_size ? kv_header_skew_bytes : header_skew);
            if (kv_pad_max_kb) {
                // Negative max = fixed padding: the files grow by the same
                // amount, so the stride stays constant.  Control for "bigger
                // files" as an explanation of the random-padding result.
                want += kv_pad_fixed
                    ? (kv_pad_max_kb / 4u) * 4096ull
                    : (pad_rng() % (kv_pad_max_kb / 4u + 1u)) * 4096ull;
            }
            if (fresh || !file_is_size(fpath, want)) {
                if (!create_file(fpath, want))
                    PROBE_FAIL("create_file %s: %s", fpath.c_str(),
                               std::strerror(errno));
                ++created_n;
            }
        }
        PROBE_OK("KV slots ready: %u files x %.2f MiB over %zu drive%s "
                 "(%u created in %.2fs, rest reused)",
                 kv_files, (double)kv_file_bytes / (1 << 20), ndev,
                 ndev == 1 ? "" : "s", created_n, sec_since(t0));
        if (kv_files != kv_chunks)
            PROBE_OK("NOTE %u chunks packed into %u file(s), %u slot(s) each; "
                     "batch width, IO size, pool offsets and distinct LBAs "
                     "unchanged -- only the target count differs",
                     kv_chunks, kv_files, kv_slots_per_file);
        PROBE_OK("KV geometry: page=%lluK segment=%lluK layers=%u "
                 "blocks/chunk=%u block_stride=%lluK layer_stride=%lluK "
                 "pool=%.2f GiB working_set=%.2f GiB requests/layer=%llu",
                 (unsigned long long)(kv_page_bytes >> 10),
                 (unsigned long long)(kv_segment_bytes >> 10), kv_layers,
                 kv_blocks_per_chunk,
                 (unsigned long long)(kv_block_stride >> 10),
                 (unsigned long long)(kv_layer_stride >> 10),
                 (double)kv_pool_bytes / (1ull << 30),
                 (double)total_bytes / (1ull << 30),
                 (unsigned long long)kv_blocks_total);
    }

    // ---- Registered GPU buffer(s) (64 KiB aligned, granularity = io_bytes) ----
    // --split-bufs N emulates a fragmented registration footprint (N small
    // cudaMalloc+register chunks, like the layerwise example's per-chunk K/V
    // tensors) instead of one big block.  IOVA/TLB behaviour of scattered
    // registrations is the thing under test.
    //
    // KV mode allocates the real pool footprint by default so the request
    // offsets land exactly where production's do (one registration, requests
    // scattered block_stride apart).  --buf-mb caps it, at the cost of
    // wrapping the offsets and shrinking the distinct-offset count that the
    // PRP cache sees.
    const std::uint64_t pool_bytes =
        (kv_mode && !buf_mb_set) ? kv_pool_bytes : buf_bytes;
    if (kv_mode && buf_mb_set && buf_bytes < kv_pool_bytes)
        PROBE_OK("NOTE --buf-mb caps the pool at %.2f GiB (< %.2f GiB needed); "
                 "offsets wrap, so distinct-offset count (and PRP cache "
                 "pressure) is lower than production",
                 (double)buf_bytes / (1ull << 30),
                 (double)kv_pool_bytes / (1ull << 30));
    std::vector<MemoryHandle> mem_handles;
    if (split_bufs == 0) {
        void* raw = nullptr;
        CUDA_OK(cudaMalloc(&raw, pool_bytes + 65536));
        void* buf = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 65535) &
                                             ~uintptr_t(65535));
        auto reg = rt->register_memory({buf, pool_bytes, MemoryKind::DEVICE,
                                         MemoryOwnership::CALLER_OWNED, gpu,
                                         TUTTI_COMPILED_ACCELERATOR_PROFILE, io_bytes});
        if (!reg.ok()) PROBE_FAIL("register_memory: %s", reg.status().message().c_str());
        mem_handles.push_back(reg.value());
        PROBE_OK("registered %.2f GiB GPU buffer at granularity %u KiB",
                 (double)pool_bytes / (1ull << 30), io_kb);
    } else {
        if (pool_bytes % split_bufs != 0)
            PROBE_FAIL("--buf-mb must divide evenly by --split-bufs");
        const std::uint64_t chunk_bytes = pool_bytes / split_bufs;
        if (chunk_bytes < io_bytes || chunk_bytes % 4096 != 0)
            PROBE_FAIL("split chunk (%llu B) smaller than io or not 4K-aligned",
                       (unsigned long long)chunk_bytes);
        const std::uint32_t slots_per_buf = (std::uint32_t)(chunk_bytes / io_bytes);
        auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t b = 0; b < split_bufs; ++b) {
            void* raw = nullptr;
            CUDA_OK(cudaMalloc(&raw, chunk_bytes + 65536));
            void* buf = reinterpret_cast<void*>(
                (reinterpret_cast<uintptr_t>(raw) + 65535) & ~uintptr_t(65535));
            auto reg = rt->register_memory({buf, chunk_bytes, MemoryKind::DEVICE,
                                            MemoryOwnership::CALLER_OWNED, gpu,
                                            TUTTI_COMPILED_ACCELERATOR_PROFILE,
                                            io_bytes});
            if (!reg.ok()) PROBE_FAIL("register_memory #%u: %s", b,
                                      reg.status().message().c_str());
            mem_handles.push_back(reg.value());
        }
        PROBE_OK("registered %u split buffers (%u KiB each, %u slot(s) per buffer) "
                 "in %.2fs", split_bufs, (unsigned)(chunk_bytes >> 10), slots_per_buf,
                 sec_since(t0));
    }
    const std::uint32_t mem_count = (std::uint32_t)mem_handles.size();
    const std::uint64_t slots_per_mem =
        (split_bufs == 0) ? pool_bytes / io_bytes : (pool_bytes / split_bufs) / io_bytes;
    // KV mode wraps raw pool offsets into the allocated span, io_bytes-aligned.
    const std::uint64_t kv_mem_span =
        (split_bufs == 0) ? pool_bytes : pool_bytes / split_bufs;

    // ---- Targets ----
    // Classic mode: one per drive.  KV mode: one per chunk (a slot file),
    // which is the whole point -- production opens thousands of them while
    // the classic probe only ever touches ndev.
    std::vector<TargetHandle> tgt;
    if (!kv_mode) {
        tgt.resize(ndev);
        for (std::size_t d = 0; d < ndev; ++d) {
            const std::string fpath = directories[d] + "/nvme_bw_probe.bin";
            auto op = rt->open(std::string("file://") + fpath, OpenOptions{"file"});
            if (!op.ok()) PROBE_FAIL("open %s: %s", fpath.c_str(),
                                      op.status().message().c_str());
            tgt[d] = op.value();
        }
    } else {
        tgt.reserve(kv_files);
        auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t c = 0; c < kv_files; ++c) {
            auto op = rt->open(std::string("file://") + kv_paths[c],
                               OpenOptions{"file"});
            if (!op.ok()) PROBE_FAIL("open %s: %s", kv_paths[c].c_str(),
                                      op.status().message().c_str());
            tgt.push_back(op.value());
        }
        PROBE_OK("opened %u KV targets in %.2fs", kv_files, sec_since(t0));
    }

    // ---- Offset table (classic mode only) ----
    // KV mode derives every offset from (chunk, layer, ordinal) instead, so it
    // skips this table: at production scale it would hold 445k entries.
    const std::uint64_t file_slots = file_bytes / io_bytes;
    std::vector<std::uint64_t> file_off;
    if (!kv_mode) {
        file_off.resize(total_ios);
        std::mt19937_64 rng(seed);
        if (mode == "rand") {
            for (std::uint64_t i = 0; i < total_ios; ++i)
                file_off[i] = (rng() % file_slots) * io_bytes;
        } else {
            for (std::uint64_t i = 0; i < total_ios; ++i)
                file_off[i] = (i % file_slots) * io_bytes;
        }
    }

    // ---- KV submit schedule ----
    // Direct mode issues ONE submit per layer covering every chunk
    // (tutti/engine/core.py:1378-1402).  --kv-wave-chunks only exists to model
    // the staging path, whose ring window forces max_chunks_per_wave-sized
    // waves; leave it 0 for production behaviour.
    struct KvSubmit { std::uint32_t wave_start, wave_len, layer; };
    std::vector<KvSubmit> kv_schedule;
    if (kv_mode) {
        const std::uint32_t wave =
            (kv_wave_chunks == 0 || kv_wave_chunks > kv_chunks) ? kv_chunks
                                                               : kv_wave_chunks;
        for (std::uint32_t s = 0; s < kv_chunks; s += wave) {
            const std::uint32_t len = std::min(wave, kv_chunks - s);
            for (std::uint32_t l = 0; l < kv_layers; ++l)
                kv_schedule.push_back({s, len, l});
        }
        const std::uint32_t nwaves = (kv_chunks + wave - 1) / wave;
        PROBE_OK("KV schedule: %zu submits (%u wave%s x %u layers), "
                 "%u requests/submit, %u distinct target(s)/submit%s",
                 kv_schedule.size(), nwaves, nwaves == 1 ? "" : "s", kv_layers,
                 wave * kv_blocks_per_chunk, std::min(wave, kv_files),
                 nwaves == 1 ? " [direct: no wave chunking]" : " [staging model]");
    }

    // ---- Read loop ----
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, gpu, stream};

    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0));
    CUDA_OK(cudaEventCreate(&ev1));

    // ---- Registration warmup (outside the measured window) ----
    // registration_for_ is lazy: the first IO touching a (memory, domain)
    // pair runs the DataPath dma-map slow path (nvm_dma_map ->
    // nvidia_p2p_get_pages, serialized ~0.25ms each).  Touch every
    // (memory, target) combination once here so the steady-state loop
    // measures IO, not one-time registrations.
    {
        auto w0 = std::chrono::steady_clock::now();
        std::uint64_t warmed = 0;
        for (std::uint32_t m = 0; m < mem_count; ++m) {
            for (std::size_t d = 0; d < ndev; ++d) {
                IoRequest wr{IoDirection::READ, mem_handles[m], 0, tgt[d],
                             0, io_bytes};
                auto o = rt->submit(&wr, 1, ctx);
                if (!o.io.has_value()) PROBE_FAIL("warmup submit rejected");
                auto wo = rt->wait(o.io.value(), 60000);
                if (wo.observation_status.code() != StatusCode::OK ||
                    !wo.result || wo.result->state != IoState::COMPLETED)
                    PROBE_FAIL("warmup wait failed");
                rt->release_io(o.io.value());
                warmed++;
            }
        }
        PROBE_OK("registration warmup: %llu (memory,target) pairs in %.2fs",
                 (unsigned long long)warmed, sec_since(w0));
    }

    // ---- Optional multi-process start barrier ----
    // Everything expensive and one-time (CUDA context, 512 MiB registration,
    // target opens, peer-memory dma-map warmup) is now behind us.  With N
    // concurrent instances that startup work takes seconds and differs per
    // process, so without a barrier the measured windows only partly overlap
    // and any cross-process aggregate overcounts (an instance that starts late
    // or finishes early sees the drives to itself).  Releasing all instances
    // from one flag aligns the windows to milliseconds.
    if (!barrier.empty()) {
        const std::string ready =
            barrier + ".ready." + std::to_string((long)::getpid());
        int rf = ::open(ready.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (rf < 0)
            PROBE_FAIL("barrier ready file %s: %s", ready.c_str(),
                       std::strerror(errno));
        ::close(rf);
        const std::string go = barrier + ".go";
        auto b0 = std::chrono::steady_clock::now();
        while (!path_exists(go)) {
            if (sec_since(b0) > 600)
                PROBE_FAIL("barrier timeout waiting for %s", go.c_str());
            ::usleep(1000);
        }
        PROBE_OK("barrier released after %.2fs wait", sec_since(b0));
    }

    const std::int64_t loop_start_ns = epoch_ns();
    auto wall0 = std::chrono::steady_clock::now();

    std::uint64_t done_ios = 0;
    std::uint64_t next_slot = 0;
    double io_ms_total = 0;
    int submit_rounds = 0;
    std::uint32_t cur_depth = depth;
    std::vector<double> batch_ms;  // per-batch wall (submit->wait) for tail stats
    std::size_t kv_idx = 0;
    const std::uint64_t kv_mem_slots = kv_mode ? kv_mem_span / io_bytes : 0;
    std::vector<IoRequest> reqs;

    while (done_ios < total_ios) {
        std::uint32_t n = 0;
        auto bt0 = std::chrono::steady_clock::now();
        if (!kv_mode) {
            n = (std::uint32_t)std::min<std::uint64_t>(cur_depth,
                                                       total_ios - done_ios);
            reqs.resize(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint64_t slot = next_slot++;
                const std::uint32_t m_idx =
                    (std::uint32_t)(split_bufs == 0 ? 0 : slot % split_bufs);
                const std::uint64_t m_off = (slot / mem_count % slots_per_mem) * io_bytes;
                const std::size_t d = (done_ios + i) % ndev;  // rotating placement
                // Mixed R/W emulation (default pure read): interleaved writes
                // force the SSD to interleave directions and the PCIe path to
                // turn around per command, like the layerwise overlap workload.
                const IoDirection dir = (write_pct > 0 && (i % 100) < write_pct)
                                            ? IoDirection::WRITE : IoDirection::READ;
                reqs[i] = {dir, mem_handles[m_idx], m_off, tgt[d],
                           file_off[done_ios + i], io_bytes};
            }
        } else {
            // One submit = one (wave, layer): every chunk of the wave
            // contributes blocks_per_chunk requests of page_bytes.
            const KvSubmit ks = kv_schedule[kv_idx];
            n = ks.wave_len * kv_blocks_per_chunk;
            reqs.resize(n);
            for (std::uint32_t c = 0; c < ks.wave_len; ++c) {
                const std::uint32_t chunk = ks.wave_start + c;
                for (std::uint32_t ord = 0; ord < kv_blocks_per_chunk; ++ord) {
                    const std::uint64_t block_id =
                        (std::uint64_t)chunk * kv_blocks_per_chunk + ord;
                    // Pool side: blocks are block_stride apart, so a chunk's
                    // two halves of one layer are megabytes apart and cannot
                    // be coalesced -- this is why production issues page_bytes
                    // IOs rather than one segment_bytes IO.
                    const std::uint64_t raw =
                        block_id * kv_block_stride +
                        (std::uint64_t)ks.layer * kv_layer_stride;
                    const std::uint64_t m_off =
                        (raw / io_bytes % kv_mem_slots) * io_bytes;
                    const std::uint32_t m_idx =
                        (std::uint32_t)(split_bufs == 0 ? 0 : block_id % split_bufs);
                    // Drive side: layer-major inside the slot, the two ordinals
                    // being adjacent (contiguous LBAs).  chunk/kv_files is the
                    // slot index within its file, which is 0 in production
                    // (one chunk per file) and only non-zero when --kv-targets
                    // packs several chunks together.
                    const std::uint64_t file_slot = chunk / kv_files;
                    const std::uint64_t header_skew = kv_header_skew_bytes
                        ? ((static_cast<std::uint64_t>(chunk) *
                            0x9E3779B97F4A7C15ull + seed) %
                           (kv_header_skew_bytes / 4096 + 1)) * 4096
                        : 0;
                    const std::uint64_t t_off =
                        file_slot * kv_slot_bytes + header_skew + kv_header_bytes +
                        (std::uint64_t)ks.layer * kv_segment_bytes +
                        (std::uint64_t)ord * kv_page_bytes;
                    reqs[c * kv_blocks_per_chunk + ord] =
                        {IoDirection::READ, mem_handles[m_idx], m_off,
                         tgt[chunk % kv_files], t_off, kv_page_bytes};
                }
            }
            if (kv_shuffle) {
                // Deterministic per-submit permutation so runs stay
                // comparable; the request set is identical, only the order
                // (and therefore the concurrently in-flight LBA set) differs.
                std::mt19937_64 srng(seed + 0x9E3779B97F4A7C15ull * (kv_idx + 1));
                std::shuffle(reqs.begin(), reqs.end(), srng);
            }
        }

        CUDA_OK(cudaEventRecord(ev0, stream));
        auto o = rt->submit(reqs.data(), reqs.size(), ctx);
        CUDA_OK(cudaEventRecord(ev1, stream));
        if (!o.io.has_value()) {
            // Batch rejected (e.g. RESOURCE_EXHAUSTED): halve and retry.
            // KV mode cannot shrink -- the width is the wave's, so a rejection
            // means the configured wave exceeds some datapath capacity.
            if (kv_mode)
                PROBE_FAIL("submit rejected with %u entries (%u chunks x %u "
                           "blocks, %u distinct targets): %s",
                           n, kv_schedule[kv_idx].wave_len, kv_blocks_per_chunk,
                           std::min(kv_schedule[kv_idx].wave_len, kv_files),
                           o.status.message().c_str());
            if (cur_depth > 1) { cur_depth /= 2; continue; }
            PROBE_FAIL("submit rejected at depth 1: %s",
                       o.status.message().c_str());
        }
        auto wo = rt->wait(o.io.value(), 60000);
        if (wo.observation_status.code() != StatusCode::OK || !wo.result ||
            wo.result->state != IoState::COMPLETED) {
            PROBE_FAIL("wait failed (round %d)", submit_rounds);
        }
        if (o.initial_states.size() != n)
            PROBE_FAIL("initial_states size mismatch");
        for (std::uint32_t i = 0; i < n; ++i) {
            if (o.initial_states[i].state != IoRequestState::ACCEPTED) {
                PROBE_FAIL("entry %u rejected in round %d: code=%d msg=%s",
                           i, submit_rounds,
                           static_cast<int>(o.initial_states[i].status.code()),
                           o.initial_states[i].status.message().c_str());
            }
        }
        rt->release_io(o.io.value());

        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
        io_ms_total += ms;
        batch_ms.push_back(sec_since(bt0) * 1e3);
        done_ios += n;
        ++submit_rounds;
        if (kv_mode) ++kv_idx;
    }

    const double wall_s = sec_since(wall0);
    const std::int64_t loop_end_ns = epoch_ns();
    const double io_s = io_ms_total / 1e3;
    const double gbps = (double)total_bytes / (1024 * 1024 * 1024) / wall_s;
    const double gbps_io = (double)total_bytes / (1024 * 1024 * 1024) / io_s;
    const double iops = (double)total_ios / wall_s;

    PROBE_OK("mode=%s io=%uK depth=%u pool=%ld%s write-pct=%u | total=%.1f GiB in %.3fs wall",
             mode.c_str(), io_kb, depth, pool_workers,
             pool_workers > 0 ? " (workers)" : "", write_pct,
             (double)total_bytes / (1ull << 30), wall_s);
    PROBE_OK("THROUGHPUT wall=%.2f GB/s io-time=%.2f GB/s | %.0f IOPS | "
             "avg %.1f us/op (io-time)",
             gbps, gbps_io, iops, io_s * 1e6 / (double)total_ios);

    // Absolute window, so a multi-process driver can verify that the
    // instances really overlapped instead of trusting a sum of per-process
    // rates (units: ns since the epoch).
    PROBE_OK("LOOP epoch_start_ns=%lld epoch_end_ns=%lld bytes=%llu",
             (long long)loop_start_ns, (long long)loop_end_ns,
             (unsigned long long)total_bytes);

    // Per-batch wall tail stats (submit->wait): batch tail = slowest batch.
    if (!batch_ms.empty()) {
        std::vector<double> sorted = batch_ms;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t p99 = sorted.size() >= 100
            ? sorted.size() - sorted.size() / 100 : sorted.size() - 1;
        double bavg = 0;
        for (double v : sorted) bavg += v;
        bavg /= sorted.size();
        PROBE_OK("BATCH wall avg=%.2fms p50=%.2fms p99=%.2fms max=%.2fms "
                 "over %zu batches",
                 bavg, sorted[sorted.size() / 2], sorted[p99],
                 sorted.back(), sorted.size());
    }

    CUDA_OK(cudaEventDestroy(ev0));
    CUDA_OK(cudaEventDestroy(ev1));
    CUDA_OK(cudaStreamDestroy(stream));
    return 0;
}
