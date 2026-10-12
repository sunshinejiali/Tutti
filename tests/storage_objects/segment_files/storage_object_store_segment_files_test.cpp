#include <tutti/spi/storage_object_store.h>
#include "csrc/storage_objects/slot_media.h"
#include "csrc/storage_objects/slot_placement_policy.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace tutti;
using tutti::storage_objects::FixedSegmentFilePlacement;
using tutti::storage_objects::materialise_segment_file;
using tutti::storage_objects::segment_file_is_ready;

int failures = 0;

void check(bool value, const char* expr, int line) {
    if (!value) {
        std::printf("FAIL line %d: %s\n", line, expr);
        ++failures;
    }
}

#define CHECK(x) check((x), #x, __LINE__)
#define REQUIRE(x) do { if (!(x)) { CHECK(x); return; } } while (0)

std::string make_root() {
    const char* root_env = ::getenv("TUTTI_SEGMENT_TEST_DIR");
    std::string base = root_env != nullptr ? root_env :
#ifdef TUTTI_TEST_TMPDIR_DEFAULT
        TUTTI_TEST_TMPDIR_DEFAULT;
#else
        "/tmp";
#endif
    base += "/tutti_segment_files_XXXXXX";
    std::vector<char> chars(base.begin(), base.end());
    chars.push_back('\0');
    char* made = ::mkdtemp(chars.data());
    return made == nullptr ? std::string() : std::string(made);
}

void remove_tree(const std::string& root) {
    std::string command = "rm -rf '" + root + "'";
    (void)::system(command.c_str());
}

ObjectKey key(std::uint8_t value) {
    ObjectKey out;
    out.bytes.assign(16, value);
    return out;
}

StoreConfig config(const std::string& root) {
    StoreConfig out;
    out.uri = root;
    out.capacity_slots = 5;
    out.layout.segment_bytes = 4096;
    out.layout.segment_count = 2;
    out.segment_file_slots = 2;   // one device -> a file group is 2 slots
    out.segment_header_bytes = 32 * 1024;
    StoreDevice device;
    device.mount_path = root;
    out.devices.push_back(std::move(device));
    out.namespace_fingerprint = {'s', 'e', 'g', 'm', 'e', 'n', 't'};
    out.background_reclaim = false;
    return out;
}

std::unique_ptr<StorageObjectStore> new_store() {
    auto created = create_storage_object_store("local_nvme_file");
    return created.ok() ? std::move(created).value() : nullptr;
}

bool exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// open() builds only the first group; reserve() never grows the pool, it
// rejects until precreate_step() publishes the next group; a reopen adopts
// every group already on media.
void test_group_precreate_and_recovery() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    const StoreConfig cfg = config(root);
    const std::uint64_t slot_bytes =
        cfg.segment_header_bytes + cfg.layout.payload_bytes();
    auto store = new_store();
    REQUIRE(store != nullptr);
    const Status opened = store->open(cfg);
    if (!opened.ok()) std::printf("open failed: %s\n", opened.message().c_str());
    REQUIRE(opened.ok());

    CHECK(store->precreated_slots() == 2);
    CHECK(store->ready_slots() == 2);
    struct stat st{};
    CHECK(::stat((root + "/r0/segments/0.seg").c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == 2 * slot_bytes);
    CHECK(exists(root + "/r0/segments/0.seg.ready"));
    CHECK(!exists(root + "/r0/segments/1.seg"));

    const ObjectKey a = key(0xA1);
    const ObjectKey b = key(0xB2);
    const ObjectKey c = key(0xC3);
    const ObjectKey keys[3] = {a, b, c};
    auto reserved = store->reserve(keys, 3);
    REQUIRE(reserved.ok());
    // c lands beyond the published group: rejected, never created inline.
    REQUIRE(reserved.value().accepted.size() == 2);
    CHECK(reserved.value().rejected_count == 1);
    CHECK(!exists(root + "/r0/segments/1.seg"));
    const auto& pa = reserved.value().accepted[0];
    const auto& pb = reserved.value().accepted[1];
    CHECK(pa.uri == pb.uri);
    CHECK(pa.offset == cfg.segment_header_bytes);
    CHECK(pb.offset == slot_bytes + cfg.segment_header_bytes);
    REQUIRE(store->commit(keys, 2).ok());

    // The group is fully handed out: any headroom asks for one more group.
    auto grown = store->precreate_step(1);
    REQUIRE(grown.ok());
    CHECK(grown.value() == 2);
    CHECK(store->precreated_slots() == 4);
    CHECK(store->precreate_target() == 4);
    CHECK(exists(root + "/r0/segments/1.seg.ready"));
    auto idle = store->precreate_step(1);   // target already met
    REQUIRE(idle.ok());
    CHECK(idle.value() == 0);

    auto again = store->reserve(&c, 1);
    REQUIRE(again.ok());
    REQUIRE(again.value().accepted.size() == 1);
    CHECK(again.value().accepted[0].uri == "file://" + root + "/r0/segments/1.seg");
    CHECK(again.value().accepted[0].offset == cfg.segment_header_bytes);
    REQUIRE(store->commit(&c, 1).ok());

    // The last group is clipped to capacity (one slot), its file still whole.
    auto tail = store->precreate_step(100);
    REQUIRE(tail.ok());
    CHECK(tail.value() == 1);
    CHECK(store->precreated_slots() == 5);
    CHECK(::stat((root + "/r0/segments/2.seg").c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == 2 * slot_bytes);

    REQUIRE(store->checkpoint().ok());
    REQUIRE(store->close().ok());

    auto reopened = new_store();
    REQUIRE(reopened != nullptr);
    REQUIRE(reopened->open(cfg).ok());
    CHECK(reopened->precreated_slots() == 5);
    for (const auto& k : keys) CHECK(reopened->contains(k));
    auto found_a = reopened->lookup(a);
    auto found_b = reopened->lookup(b);
    REQUIRE(found_a.ok());
    REQUIRE(found_b.ok());
    CHECK(found_a.value().uri == found_b.value().uri);
    CHECK(found_a.value().offset != found_b.value().offset);

    // Releasing one slot of a shared file leaves its neighbour live.
    auto released = reopened->release(&a, 1);
    REQUIRE(released.ok());
    CHECK(released.value() == 1);
    CHECK(reopened->contains(b));
    CHECK(reopened->close().ok());
    remove_tree(root);
}

// A group whose file lost its ready marker is not adopted; the next step
// rebuilds it.
void test_unready_group_is_not_adopted() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    const StoreConfig cfg = config(root);
    {
        auto store = new_store();
        REQUIRE(store != nullptr);
        REQUIRE(store->open(cfg).ok());
        REQUIRE(store->precreate_step(3).ok());
        CHECK(store->precreated_slots() == 4);
        REQUIRE(store->close().ok());
    }
    REQUIRE(::unlink((root + "/r0/segments/1.seg.ready").c_str()) == 0);
    auto store = new_store();
    REQUIRE(store != nullptr);
    REQUIRE(store->open(cfg).ok());
    CHECK(store->precreated_slots() == 2);
    REQUIRE(store->precreate_step(3).ok());
    CHECK(store->precreated_slots() == 4);
    CHECK(exists(root + "/r0/segments/1.seg.ready"));
    CHECK(store->close().ok());
    remove_tree(root);
}

// Concurrent growers never build the same group twice.
void test_parallel_precreate_steps() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    auto store = new_store();
    REQUIRE(store != nullptr);
    REQUIRE(store->open(config(root)).ok());
    std::uint64_t made[2] = {0, 0};
    bool ok[2] = {false, false};
    auto grow = [&](int i) {
        auto r = store->precreate_step(3);
        ok[i] = r.ok();
        if (r.ok()) made[i] = r.value();
    };
    std::thread left(grow, 0);
    std::thread right(grow, 1);
    left.join();
    right.join();
    CHECK(ok[0] && ok[1]);
    CHECK(made[0] + made[1] == 2);
    CHECK(store->precreated_slots() == 4);
    CHECK(store->close().ok());
    remove_tree(root);
}

// A failed group build publishes nothing and the next step retries it.
void test_failed_precreate_retries_group() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    StoreConfig cfg = config(root);
    cfg.capacity_slots = 4;
    auto store = new_store();
    REQUIRE(store != nullptr);
    REQUIRE(store->open(cfg).ok());
    CHECK(store->precreated_slots() == 2);
    // A directory where group 1's file must go makes its creation fail.
    const std::string blocked = root + "/r0/segments/1.seg";
    REQUIRE(::mkdir(blocked.c_str(), 0755) == 0);
    CHECK(!store->precreate_step(3).ok());
    CHECK(store->precreated_slots() == 2);
    REQUIRE(::rmdir(blocked.c_str()) == 0);
    auto retried = store->precreate_step(3);
    REQUIRE(retried.ok());
    CHECK(retried.value() == 2);
    CHECK(store->precreated_slots() == 4);
    CHECK(store->close().ok());
    remove_tree(root);
}

void test_partial_file_resumes_without_overwriting_prefix() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    const std::string path = root + "/partial.seg";
    constexpr std::uint64_t page = 4096;
    int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    REQUIRE(fd >= 0);
    std::vector<char> prefix(page, 'X');
    CHECK(::write(fd, prefix.data(), prefix.size()) ==
          static_cast<ssize_t>(prefix.size()));
    CHECK(::close(fd) == 0);
    bool ready = true;
    REQUIRE(segment_file_is_ready(path, 3 * page, &ready).ok());
    CHECK(!ready);
    REQUIRE(materialise_segment_file(path, 3 * page).ok());
    REQUIRE(segment_file_is_ready(path, 3 * page, &ready).ok());
    CHECK(ready);
    {
        std::vector<char> head(page, 0);
        fd = ::open(path.c_str(), O_RDONLY);
        REQUIRE(fd >= 0);
        CHECK(::read(fd, head.data(), head.size()) == static_cast<ssize_t>(page));
        CHECK(::close(fd) == 0);
        CHECK(head[0] == 'X' && head[page - 1] == 'X');
    }
    CHECK(::unlink((path + ".ready").c_str()) == 0);
    REQUIRE(segment_file_is_ready(path, 3 * page, &ready).ok());
    CHECK(!ready);
    REQUIRE(materialise_segment_file(path, 3 * page).ok());
    REQUIRE(segment_file_is_ready(path, 3 * page, &ready).ok());
    CHECK(ready);
    const std::string replacement = root + "/replacement.seg";
    fd = ::open(replacement.c_str(), O_CREAT | O_WRONLY, 0644);
    REQUIRE(fd >= 0);
    REQUIRE(::ftruncate(fd, 3 * page) == 0);
    CHECK(::close(fd) == 0);
    REQUIRE(::rename(replacement.c_str(), path.c_str()) == 0);
    // The old ready marker cannot authorize a same-size replacement inode.
    REQUIRE(segment_file_is_ready(path, 3 * page, &ready).ok());
    CHECK(!ready);
    CHECK(!materialise_segment_file(path, 0).ok());
    CHECK(!materialise_segment_file(path, 1000).ok());
    CHECK(::unlink((path + ".ready").c_str()) == 0);
    CHECK(::unlink(path.c_str()) == 0);
    CHECK(::rmdir(root.c_str()) == 0);
}

void test_parallel_distinct_files_on_one_mount() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    constexpr std::uint64_t bytes = 8 * 1024 * 1024;
    const std::string first = root + "/rank0.seg";
    const std::string second = root + "/rank1.seg";
    bool first_ok = false;
    bool second_ok = false;
    std::thread left([&] { first_ok = materialise_segment_file(first, bytes).ok(); });
    std::thread right([&] { second_ok = materialise_segment_file(second, bytes).ok(); });
    left.join();
    right.join();
    CHECK(first_ok);
    CHECK(second_ok);
    for (const auto& path : {first, second}) {
        bool ready = false;
        REQUIRE(segment_file_is_ready(path, bytes, &ready).ok());
        CHECK(ready);
        CHECK(::unlink((path + ".ready").c_str()) == 0);
        CHECK(::unlink(path.c_str()) == 0);
    }
    CHECK(::rmdir(root.c_str()) == 0);
}

void test_four_device_store_recovers_fixed_files() {
    const std::string root = make_root();
    REQUIRE(!root.empty());
    StoreConfig cfg = config(root);
    cfg.capacity_slots = 8;   // 4 devices x 2 slots per file = one group
    cfg.devices.clear();
    for (int device = 0; device < 4; ++device) {
        const std::string mount = root + "/disk" + std::to_string(device);
        REQUIRE(::mkdir(mount.c_str(), 0755) == 0);
        StoreDevice entry;
        entry.mount_path = mount;
        cfg.devices.push_back(std::move(entry));
    }
    auto store = new_store();
    REQUIRE(store != nullptr);
    REQUIRE(store->open(cfg).ok());
    CHECK(store->precreated_slots() == 8);
    ObjectKey keys[4] = {key(1), key(2), key(3), key(4)};
    auto reserved = store->reserve(keys, 4);
    REQUIRE(reserved.ok());
    REQUIRE(reserved.value().accepted.size() == 4);
    for (int device = 0; device < 4; ++device) {
        const auto& placement = reserved.value().accepted[device];
        CHECK(placement.uri == "file://" + cfg.devices[device].mount_path +
                                   "/r0/segments/0.seg");
        CHECK(placement.offset == cfg.segment_header_bytes);
    }
    REQUIRE(store->commit(keys, 4).ok());
    REQUIRE(store->checkpoint().ok());
    REQUIRE(store->close().ok());

    auto reopened = new_store();
    REQUIRE(reopened != nullptr);
    REQUIRE(reopened->open(cfg).ok());
    for (const auto& object_key : keys) CHECK(reopened->lookup(object_key).ok());
    CHECK(reopened->close().ok());
    remove_tree(root);
}

void test_fixed_file_slot_boundaries() {
    constexpr std::uint64_t slot_bytes = 10 * 1024 * 1024 + 4096;
    for (std::uint64_t slots_per_file : {2048ull, 4096ull}) {
        FixedSegmentFilePlacement placement(
            {"/mnt/a", "/mnt/b"}, "r0/segments",
            slot_bytes, slots_per_file * slot_bytes);
        CHECK(placement.geometry_valid());
        CHECK(placement.file_bytes() == slots_per_file * slot_bytes);
        CHECK(placement.path_for_slot(2 * (slots_per_file - 1)) ==
              "/mnt/a/r0/segments/0.seg");
        CHECK(placement.payload_offset_for_slot(2 * (slots_per_file - 1)) ==
              (slots_per_file - 1) * slot_bytes + ObjectHeaderLayout::kHeaderBytes);
        CHECK(placement.path_for_slot(1) == "/mnt/b/r0/segments/0.seg");
        CHECK(placement.path_for_slot(2 * slots_per_file) == "/mnt/a/r0/segments/1.seg");
        CHECK(placement.header_offset_for_slot(2 * slots_per_file) == 0);
        CHECK(placement.payload_offset_for_slot(2 * slots_per_file) ==
              ObjectHeaderLayout::kHeaderBytes);
        CHECK(placement.path_for_slot(2 * slots_per_file + 1) ==
              "/mnt/b/r0/segments/1.seg");
        CHECK(placement.path_for_slot(4 * slots_per_file) == "/mnt/a/r0/segments/2.seg");
        CHECK(placement.uri_for_slot(1) == "file:///mnt/b/r0/segments/0.seg");

        FixedSegmentFilePlacement four_drives(
            {"/mnt/a", "/mnt/b", "/mnt/c", "/mnt/d"},
            "r0/segments", slot_bytes, slots_per_file * slot_bytes);
        for (std::uint64_t device = 0; device < 4; ++device) {
            const std::string disk = "/mnt/" + std::string(1, 'a' + device);
            CHECK(four_drives.path_for_slot(device) == disk + "/r0/segments/0.seg");
            CHECK(four_drives.path_for_slot(4 * slots_per_file + device) ==
                  disk + "/r0/segments/1.seg");
            CHECK(four_drives.payload_offset_for_slot(4 * slots_per_file + device) ==
                  ObjectHeaderLayout::kHeaderBytes);
        }
    }
    FixedSegmentFilePlacement empty({}, "r0/segments", slot_bytes, slot_bytes);
    CHECK(!empty.geometry_valid());
}

}  // namespace

int main() {
    test_group_precreate_and_recovery();
    test_unready_group_is_not_adopted();
    test_parallel_precreate_steps();
    test_failed_precreate_retries_group();
    test_partial_file_resumes_without_overwriting_prefix();
    test_parallel_distinct_files_on_one_mount();
    test_four_device_store_recovers_fixed_files();
    test_fixed_file_slot_boundaries();
    if (failures != 0) return 1;
    std::printf("storage_object_store segment-files: all checks passed\n");
    return 0;
}
