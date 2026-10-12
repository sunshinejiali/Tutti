// Regression: the preset must drive the controller it was configured for.
//
// 2026-10-07: chrdev_for_bdf() derived /dev/ssnvmeN from the controller
// number snvmeN. The two are numbered by unrelated kernel allocators; once
// two daemons brought controllers up in different orders they diverged and
// GPU IO for one md array was written to the members of the other.
//
// Needs both daemons up (scripts/tutti-md.sh kv0|md0 up), so that controller
// and chrdev numbers really differ. Data-safe: it writes one 4 KiB block
// only into a scratch file it created through the kernel, on each array,
// and verifies through the kernel that exactly that block changed.
//
//   tutti_snvme_identity_contract_test

#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/presets/local_nvme.h>
#include <tutti/storage_runtime.h>

#include "csrc/common/snvme_identity.h"
#include "csrc/data_paths/striped_local_nvme/striped_data_path.h"

#include <fcntl.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace tutti;

namespace {

int g_pass = 0;
int g_fail = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (cond) { ++g_pass; } else {                                \
            ++g_fail;                                                 \
            std::fprintf(stderr, "[FAIL] %s:%d ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

constexpr std::uint64_t kBlock = 4096;
constexpr std::uint64_t kFile = 4u << 20;

struct Group {
    int gpu;
    std::string mount;
    std::vector<std::string> pcis;
};

// Controller number M of /sys/bus/pci/devices/<pci>/snvme/snvmeM, -1 if none.
int controller_number(const std::string& pci) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(
             "/sys/bus/pci/devices/" + pci + "/snvme", ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("snvme", 0) == 0) return std::atoi(n.c_str() + 5);
    }
    return -1;
}

std::string block_of(const std::string& pci) {
    const int m = controller_number(pci);
    return m < 0 ? "" : "/dev/snvme" + std::to_string(m) + "n1";
}

bool kernel_pwrite(const std::string& path, const void* buf, std::uint64_t n,
                   std::uint64_t off, bool create) {
    const int fd = ::open(path.c_str(), O_RDWR | O_DIRECT | (create ? O_CREAT | O_TRUNC : 0), 0644);
    if (fd < 0) return false;
    bool ok = !create || ::fallocate(fd, 0, 0, static_cast<off_t>(kFile)) == 0;
    ok = ok && ::pwrite(fd, buf, n, static_cast<off_t>(off)) == static_cast<ssize_t>(n);
    ok = ok && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

bool kernel_pread(const std::string& path, void* buf, std::uint64_t n) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return false;
    const bool ok = ::pread(fd, buf, n, 0) == static_cast<ssize_t>(n);
    ::close(fd);
    return ok;
}

void run_group(const Group& g) {
    namespace id = tutti::detail::snvme_identity;
    // 1. The preset's chrdev selection must match the kernel's answer.
    for (const auto& pci : g.pcis) {
        const std::string chr = id::chrdev_for_pci(pci);
        CHECK(!chr.empty(), "no chrdev reports %s", pci.c_str());
        CHECK(id::pci_of_chrdev(chr) == pci, "%s does not report %s", chr.c_str(), pci.c_str());
        const int m = controller_number(pci);
        std::printf("  %s: controller snvme%d, chrdev %s%s\n", pci.c_str(), m, chr.c_str(),
                    chr != "/dev/ssnvme" + std::to_string(m) ? "  (numbers differ)" : "");
    }

    // 2. End to end through the preset: the GPU writes one block of a file
    //    the kernel created; the kernel must see exactly that block change.
    presets::StripedNvmePreset p;
    p.accel_id = g.gpu;
    p.num_queues = 2;
    for (const auto& pci : g.pcis) p.devices.push_back({pci, block_of(pci), g.mount, 1, 4096});
    auto built = presets::make_striped_nvme_runtime(p);
    CHECK(built.runtime != nullptr, "gpu%d preset runtime: %s", g.gpu,
          built.creation_status.message().c_str());
    if (!built.runtime) return;
    StorageRuntime* rt = built.runtime.get();
    cudaSetDevice(g.gpu);

    const std::string dir = g.mount + "/snvme_identity." + std::to_string(::getpid());
    ::mkdir(dir.c_str(), 0755);
    const std::string path = dir + "/f.bin";
    void* host = nullptr;
    void* back = nullptr;
    ::posix_memalign(&host, kBlock, kFile);
    ::posix_memalign(&back, kBlock, kFile);
    std::memset(host, 0x11, kFile);
    CHECK(kernel_pwrite(path, host, kFile, 0, true), "create %s", path.c_str());

    auto opened = rt->open("file://" + path, OpenOptions{"file"});
    CHECK(opened.ok(), "open: %s", opened.ok() ? "" : opened.status().message().c_str());
    void* raw = nullptr;
    cudaMalloc(&raw, kBlock + 65536);
    void* gpu = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(raw) + 65535) &
                                        ~std::uintptr_t(65535));
    cudaMemset(gpu, 0xC3, kBlock);
    cudaDeviceSynchronize();
    auto mem = rt->register_memory({gpu, kBlock, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED,
                                    g.gpu, TUTTI_COMPILED_ACCELERATOR_PROFILE, kBlock});
    CHECK(mem.ok(), "register_memory");
    if (opened.ok() && mem.ok()) {
        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);
        const std::uint64_t at = 2u << 20;  // one block in the middle
        IoRequest req{IoDirection::WRITE, mem.value(), 0, opened.value(), at, kBlock};
        auto sub = rt->submit(&req, 1, {ExecutionDomain::DEVICE_EXECUTION, g.gpu, s});
        bool done = sub.io.has_value() && !sub.initial_states.empty() &&
                    sub.initial_states[0].state == IoRequestState::ACCEPTED;
        if (sub.io.has_value()) {
            auto w = rt->wait(sub.io.value(), 30000);
            done = done && w.result && w.result->state == IoState::COMPLETED;
            rt->release_io(sub.io.value());
        }
        CHECK(done, "gpu%d write did not complete", g.gpu);
        CHECK(kernel_pread(path, back, kFile), "read back");
        std::memset(static_cast<char*>(host) + at, 0xC3, kBlock);
        const bool same = std::memcmp(host, back, kFile) == 0;
        CHECK(same, "gpu%d: the file on %s did not get exactly the GPU's block", g.gpu,
              g.mount.c_str());
        std::printf("[%s] gpu%d -> %s: GPU write landed in the right file\n",
                    same && done ? " OK " : "FAIL", g.gpu, g.mount.c_str());
        cudaStreamDestroy(s);
        rt->close(opened.value());
    }
    built.runtime.reset();
    cudaFree(raw);
    std::free(host);
    std::free(back);
    std::filesystem::remove_all(dir);
}

}  // namespace

// The DataPath must refuse a chrdev that drives a different controller than
// the descriptor names (the exact pairing the old number-based lookup could
// produce). Initialization only: no IO is submitted.
void wrong_pairing_is_refused() {
    namespace id = tutti::detail::snvme_identity;
    const std::string a = "0000:08:00.0", b = "0000:d2:00.0";
    const std::string chr_b = id::chrdev_for_pci(b);
    if (chr_b.empty() || id::chrdev_for_pci(a).empty()) return;
    tutti::data_paths::striped_local_nvme::DeviceDescriptor d;
    d.snvme_dev_path = chr_b;          // drives b ...
    d.controller_pci_addr = a;         // ... but claims to be a
    d.cuda_device = 0;
    d.num_user_queues = 1;
    tutti::data_paths::striped_local_nvme::StripedDataPath dp({d}, 0);
    tutti::DataPathConfig cfg{"identity-negative"};
    tutti::ResourceProvider resources;
    const auto st = dp.initialize(cfg, resources);
    CHECK(!st.ok(), "StripedDataPath accepted %s for PCI %s", chr_b.c_str(), a.c_str());
    std::printf("[%s] %s claimed as %s is refused: %s\n", st.ok() ? "FAIL" : " OK ",
                chr_b.c_str(), a.c_str(), st.message().c_str());
    if (st.ok()) dp.shutdown(1000000000);
}

int main() {
    glob_t gl{};
    const bool any = ::glob("/dev/ssnvme*", 0, nullptr, &gl) == 0 && gl.gl_pathc > 0;
    ::globfree(&gl);
    if (!any) {
        std::printf("[SKIP] no snvme character devices\n");
        return 77;
    }
    const std::vector<Group> groups = {
        {0, "/mnt/tutti_md0", {"0000:08:00.0", "0000:4b:00.0", "0000:57:00.0", "0000:63:00.0"}},
        {4, "/mnt/nvme4", {"0000:d2:00.0", "0000:df:00.0", "0000:86:00.0", "0000:c5:00.0"}},
    };
    for (const auto& g : groups) {
        if (controller_number(g.pcis[0]) < 0) {
            std::printf("[SKIP] %s: controllers not on snvme\n", g.mount.c_str());
            continue;
        }
        std::printf("gpu%d / %s\n", g.gpu, g.mount.c_str());
        run_group(g);
    }
    wrong_pairing_is_refused();
    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
