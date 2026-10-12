// storage_target_resolver_contract_test.cpp
//
// Contract tests for tutti/spi/storage_target_resolver.h.
// Plain C++17 executable, no GTest or third-party deps.
// Returns 0 on full pass, non-zero on any failure.

#include <tutti/spi/storage_target_resolver.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

// =====================================================================
// Fake types for testing
// =====================================================================

// Fake payload — in a real binding this would be e.g. fiemap extents.
struct FakeFilePayload {
    std::string path;
    int placeholder = 42;
};

// Fake owner lease — in a real binding this would own an fd / file lock.
struct FakeOwnerLease {
    std::string description;
    int ref_marker = 1;
};

// Fake resolver implementing the StorageTargetResolver SPI.
class FakeResolver : public tutti::StorageTargetResolver {
public:
    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view uri,
        const tutti::ResolveOptions& /*options*/) override {

        if (uri.substr(0, 7) == "fake://") {
            auto payload = std::make_shared<FakeFilePayload>();
            payload->path = std::string(uri.substr(7));
            payload->placeholder = 99;

            auto lease = std::make_shared<FakeOwnerLease>();
            lease->description = "fake-lease";
            lease->ref_marker = 1;

            return tutti::ResolvedTarget::make<FakeFilePayload, FakeOwnerLease>(
                "fake-resolver",
                "fake-file-payload-v1",
                1,        // source_api_version
                4096,     // logical_size
                "local-nvme-ext4",
                std::move(payload),
                std::move(lease));
        }

        return tutti::Result<tutti::ResolvedTarget>::Failure(
            tutti::Status(tutti::StatusCode::NOT_FOUND,
                          "unknown URI scheme"));
    }
};

// =====================================================================
// Helper
// =====================================================================

static int run_test(int idx, int (*fn)()) {
    int rc = fn();
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: test %d returned %d\n", idx, rc);
    }
    return rc;
}

// =====================================================================
// Tests
// =====================================================================

// 1. Fake resolver implements resolve(uri, options) and returns
//    Result<ResolvedTarget>.
static int test_fake_resolver_success() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://test.txt", opts);

    if (!result.ok()) return 1;
    if (!result.has_value()) return 1;
    if (!result.value().valid()) return 1;
    return 0;
}

// 2. Fake payload type and fake owner lease type are both held by
//    ResolvedTarget (verified indirectly: target is valid and payload
//    is accessible via view, which requires the shared_ptr to be alive).
static int test_payload_and_lease_held() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://data.bin", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    if (!target.valid()) return 1;

    auto v = target.view<FakeFilePayload>("fake-file-payload-v1", 1);
    if (!v.ok()) return 1;
    if (v.value() == nullptr) return 1;
    if (v.value()->path != "data.bin") return 1;
    return 0;
}

// 3. Correct payload_type_id + API version: view returns const payload.
static int test_view_correct_type_and_version() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://correct.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    auto v = target.view<FakeFilePayload>("fake-file-payload-v1", 1);
    if (!v.ok()) return 1;
    if (!v.has_value()) return 1;
    if (v.value() == nullptr) return 1;
    if (v.value()->path != "correct.txt") return 1;
    if (v.value()->placeholder != 99) return 1;
    return 0;
}

// 4. Wrong payload type: view returns UNSUPPORTED.
static int test_view_wrong_payload_type() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://wrong_type.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    auto v = target.view<FakeFilePayload>("wrong-payload-type", 1);
    if (v.ok()) return 1;
    if (v.has_value()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 5. Wrong API version: view returns UNSUPPORTED.
static int test_view_wrong_api_version() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://wrong_ver.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    auto v = target.view<FakeFilePayload>("fake-file-payload-v1", 999);
    if (v.ok()) return 1;
    if (v.has_value()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 6. Error resolver returns non-OK Status, no ResolvedTarget.
static int test_error_resolver_returns_failure() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("error://notfound", opts);

    if (result.ok()) return 1;
    if (result.has_value()) return 1;
    if (result.status().code() != tutti::StatusCode::NOT_FOUND) return 1;
    return 0;
}

// 7. ResolvedTarget does not expose payload mutation.
//    view() returns const Payload*, so writes through the pointer
//    would not compile.  We verify at runtime that the payload data
//    is read-only accessible and unchanged.
static int test_no_payload_mutation() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://immutable.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    auto v = target.view<FakeFilePayload>("fake-file-payload-v1", 1);
    if (!v.ok()) return 1;

    // Read through const pointer — compiles because return type is
    // const FakeFilePayload*.  If it were non-const, the following
    // line would compile and mutate, which we want to prevent:
    //   v.value()->placeholder = 0;  // would NOT compile (const)
    const FakeFilePayload* p = v.value();
    if (p->placeholder != 99) return 1;
    if (p->path != "immutable.txt") return 1;
    return 0;
}

// 8. Move semantics: moved-to target remains valid and viewable;
//    moved-from target has valid() == false.
static int test_move_semantics() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://moveme.txt", opts);
    if (!result.ok()) return 1;

    tutti::ResolvedTarget moved_to = std::move(result.value());

    // Moved-to target is valid and viewable.
    if (!moved_to.valid()) return 1;
    if (moved_to.payload_type_id() != "fake-file-payload-v1") return 1;
    if (moved_to.source_api_version() != 1) return 1;
    if (moved_to.logical_size() != 4096) return 1;

    auto v = moved_to.view<FakeFilePayload>("fake-file-payload-v1", 1);
    if (!v.ok()) return 1;
    if (v.value()->path != "moveme.txt") return 1;

    // Moved-from target has defined state: valid() == false.
    if (result.value().valid()) return 1;
    return 0;
}

// 9. Multiple views from the same target point to the same immutable
//    payload owner (same address).
static int test_multiple_views_same_payload() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://shared.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();

    auto v1 = target.view<FakeFilePayload>("fake-file-payload-v1", 1);
    auto v2 = target.view<FakeFilePayload>("fake-file-payload-v1", 1);

    if (!v1.ok() || !v2.ok()) return 1;
    if (v1.value() == nullptr || v2.value() == nullptr) return 1;

    // Both views point to the same underlying payload object.
    if (v1.value() != v2.value()) return 1;

    // Data is consistent through both views.
    if (v1.value()->path != v2.value()->path) return 1;
    return 0;
}

// 10. Default/empty target: valid() is false, view returns UNSUPPORTED.
static int test_empty_target() {
    tutti::ResolvedTarget empty;

    if (empty.valid()) return 1;
    if (empty.resolver_type_id() != "") return 1;
    if (empty.payload_type_id() != "") return 1;
    if (empty.source_api_version() != 0) return 1;
    if (empty.logical_size() != 0) return 1;
    if (empty.recommended_data_path_key() != "") return 1;

    auto v = empty.view<FakeFilePayload>("fake-file-payload-v1", 1);
    if (v.ok()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 11. Recommended DataPath key is readable as a string, not a closed enum.
static int test_recommended_data_path_key_readable() {
    FakeResolver resolver;
    tutti::ResolveOptions opts{"fake"};
    auto result = resolver.resolve("fake://keytest.txt", opts);
    if (!result.ok()) return 1;

    auto& target = result.value();
    std::string_view key = target.recommended_data_path_key();
    if (key != "local-nvme-ext4") return 1;

    // The key is a plain string_view, not an enum.  This compiles
    // because the return type is std::string_view.
    std::string copied(key);
    if (copied != "local-nvme-ext4") return 1;
    return 0;
}

// 12. No common variant/union payload.
//     This is a compile-time design property: the header uses
//     shared_ptr<void> type erasure, not std::variant or union.
//     The fact that this test compiles and runs without any variant
//     or union in the public API is the proof.
//     Additionally, verify that a second payload type works through
//     the same ResolvedTarget shell, proving the erasure is generic.
struct OtherPayload {
    int x = 7;
};

struct OtherLease {
    int y = 8;
};

static int test_no_common_variant() {
    // Create a target with a different payload/lease type pair.
    auto payload = std::make_shared<OtherPayload>();
    auto lease = std::make_shared<OtherLease>();

    auto result = tutti::ResolvedTarget::make<OtherPayload, OtherLease>(
        "other-resolver",
        "other-payload-v2",
        2,
        8192,
        "some-other-datapath",
        std::move(payload),
        std::move(lease));
    if (!result.ok()) return 1;

    auto& target = result.value();
    if (target.payload_type_id() != "other-payload-v2") return 1;
    if (target.source_api_version() != 2) return 1;

    // View with correct type+version succeeds.
    auto v = target.view<OtherPayload>("other-payload-v2", 2);
    if (!v.ok()) return 1;
    if (v.value() == nullptr) return 1;
    if (v.value()->x != 7) return 1;

    // View with wrong string ID fails (proves string-based type check).
    auto v2 = target.view<OtherPayload>("wrong-payload-type", 2);
    if (v2.ok()) return 1;
    if (v2.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;

    // View with wrong version fails (proves version check).
    auto v3 = target.view<OtherPayload>("other-payload-v2", 999);
    if (v3.ok()) return 1;
    if (v3.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;

    return 0;
}

// =====================================================================
// Main
// =====================================================================

int main() {
    using TestFn = int (*)();
    const TestFn tests[] = {
        test_fake_resolver_success,          // 1
        test_payload_and_lease_held,         // 2
        test_view_correct_type_and_version,  // 3
        test_view_wrong_payload_type,        // 4
        test_view_wrong_api_version,         // 5
        test_error_resolver_returns_failure, // 6
        test_no_payload_mutation,            // 7
        test_move_semantics,                 // 8
        test_multiple_views_same_payload,    // 9
        test_empty_target,                   // 10
        test_recommended_data_path_key_readable, // 11
        test_no_common_variant,              // 12
    };

    const int n = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; ++i) {
        if (run_test(i, tests[i]) != 0) {
            return 1;
        }
    }

    std::printf("All %d storage target resolver contract tests passed.\n", n);
    return 0;
}
