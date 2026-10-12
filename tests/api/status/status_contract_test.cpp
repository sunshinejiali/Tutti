// status_contract_test.cpp -- contract tests for tutti/status.h
//
// Plain C++17 executable, no GTest or third-party deps.
// Returns 0 on full pass, non-zero on any failure.

#include <tutti/status.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

// -------------------------------------------------------------------------
// 1. StatusCode: all 12 categories compile and are distinct.
// -------------------------------------------------------------------------
static int test_status_code_coverage() {
    using SC = tutti::StatusCode;
    const SC codes[] = {
        SC::OK,
        SC::INVALID_ARGUMENT,
        SC::OUT_OF_RANGE,
        SC::NOT_FOUND,
        SC::UNSUPPORTED,
        SC::NOT_READY,
        SC::BUSY,
        SC::RESOURCE_EXHAUSTED,
        SC::TIMEOUT,
        SC::DEVICE_ERROR,
        SC::DATA_LOSS,
        SC::INTERNAL,
    };
    constexpr int n = static_cast<int>(sizeof(codes) / sizeof(codes[0]));
    static_assert(n == 12, "exactly 12 StatusCode values expected");

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (static_cast<int>(codes[i]) == static_cast<int>(codes[j])) {
                std::fprintf(stderr,
                    "FAIL: StatusCode values at index %d and %d are identical\n",
                    i, j);
                return 1;
            }
        }
    }
    return 0;
}

// -------------------------------------------------------------------------
// 2. Default Status is OK.
// -------------------------------------------------------------------------
static int test_default_status_is_ok() {
    tutti::Status s;
    if (!s.ok()) return 1;
    if (s.code() != tutti::StatusCode::OK) return 1;
    if (!s.message().empty()) return 1;
    return 0;
}

// -------------------------------------------------------------------------
// 3. Explicit success factory is OK.
// -------------------------------------------------------------------------
static int test_ok_factory() {
    tutti::Status s = tutti::Status::Ok();
    if (!s.ok()) return 1;
    if (s.code() != tutti::StatusCode::OK) return 1;
    return 0;
}

// -------------------------------------------------------------------------
// 4. Each non-OK status reports the correct code().
// -------------------------------------------------------------------------
static int test_error_codes() {
    using SC = tutti::StatusCode;
    const struct { SC code; const char* msg; } cases[] = {
        {SC::INVALID_ARGUMENT,   "invalid argument"},
        {SC::OUT_OF_RANGE,       "out of range"},
        {SC::NOT_FOUND,          "not found"},
        {SC::UNSUPPORTED,        "unsupported"},
        {SC::NOT_READY,          "not ready"},
        {SC::BUSY,               "busy"},
        {SC::RESOURCE_EXHAUSTED, "resource exhausted"},
        {SC::TIMEOUT,            "timeout"},
        {SC::DEVICE_ERROR,       "device error"},
        {SC::DATA_LOSS,          "data loss"},
        {SC::INTERNAL,           "internal"},
    };
    for (const auto& c : cases) {
        tutti::Status s(c.code, c.msg);
        if (s.ok()) {
            std::fprintf(stderr, "FAIL: status with code should not be ok\n");
            return 1;
        }
        if (s.code() != c.code) {
            std::fprintf(stderr, "FAIL: code mismatch\n");
            return 1;
        }
    }
    return 0;
}

// -------------------------------------------------------------------------
// 5. message is readable and preserved.
// -------------------------------------------------------------------------
static int test_message_preserved() {
    const std::string msg = "device 42 not found";
    tutti::Status s(tutti::StatusCode::NOT_FOUND, msg);
    if (s.message() != msg) return 1;

    // Verify const-ref identity (same object, not a copy).
    const tutti::Status& cs = s;
    if (&cs.message() != &s.message()) return 1;
    return 0;
}

// -------------------------------------------------------------------------
// 6. Status copy/move.
// -------------------------------------------------------------------------
static int test_status_copy_move() {
    tutti::Status original(tutti::StatusCode::TIMEOUT, "timed out");

    // Copy construct
    tutti::Status copied = original;
    if (copied.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (copied.message() != "timed out") return 1;

    // Move construct
    tutti::Status moved = std::move(copied);
    if (moved.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (moved.message() != "timed out") return 1;

    // Copy assign
    tutti::Status assigned;
    assigned = original;
    if (assigned.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (assigned.message() != "timed out") return 1;

    // Move assign
    tutti::Status move_assigned;
    move_assigned = std::move(assigned);
    if (move_assigned.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (move_assigned.message() != "timed out") return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 7. Result<int> success value.
// -------------------------------------------------------------------------
static int test_result_int_success() {
    tutti::Result<int> r = 42;
    if (!r.ok()) return 1;
    if (!r.has_value()) return 1;
    if (r.status().code() != tutti::StatusCode::OK) return 1;
    if (r.value() != 42) return 1;

    // mutable access
    r.value() = 100;
    if (r.value() != 100) return 1;

    // const access
    const tutti::Result<int>& cr = r;
    if (cr.value() != 100) return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 8. Result<std::string> success value.
// -------------------------------------------------------------------------
static int test_result_string_success() {
    tutti::Result<std::string> r = std::string("hello");
    if (!r.ok()) return 1;
    if (!r.has_value()) return 1;
    if (r.value() != "hello") return 1;

    // const access
    const tutti::Result<std::string>& cr = r;
    if (cr.value() != "hello") return 1;

    // move access
    std::string moved_out = std::move(r).value();
    if (moved_out != "hello") return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 9. Error Result<int> keeps code/message and has no value.
// -------------------------------------------------------------------------
static int test_result_int_error() {
    // Via named factory
    tutti::Result<int> r = tutti::Result<int>::Failure(
        tutti::Status(tutti::StatusCode::NOT_FOUND, "no such device"));
    if (r.ok()) return 1;
    if (r.has_value()) return 1;
    if (r.status().code() != tutti::StatusCode::NOT_FOUND) return 1;
    if (r.status().message() != "no such device") return 1;

    // Via explicit constructor
    tutti::Result<int> r2(
        tutti::Status(tutti::StatusCode::BUSY, "device busy"));
    if (r2.ok()) return 1;
    if (r2.has_value()) return 1;
    if (r2.status().code() != tutti::StatusCode::BUSY) return 1;
    if (r2.status().message() != "device busy") return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 10. Result<std::unique_ptr<int>> success construct and move-out.
// -------------------------------------------------------------------------
static int test_result_unique_ptr_success_and_move() {
    auto up = std::make_unique<int>(7);
    tutti::Result<std::unique_ptr<int>> r = std::move(up);
    if (!r.ok()) return 1;
    if (!r.has_value()) return 1;
    if (*r.value() != 7) return 1;

    // Move-out via rvalue value()
    std::unique_ptr<int> extracted = std::move(r).value();
    if (!extracted) return 1;
    if (*extracted != 7) return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 11. Illegal "OK status but no value" cannot become a legal success.
// -------------------------------------------------------------------------
static int test_illegal_ok_without_value() {
    // Via named factory with OK status
    tutti::Result<int> r = tutti::Result<int>::Failure(tutti::Status::Ok());
    if (r.ok()) return 1;          // must NOT be ok
    if (r.has_value()) return 1;   // must NOT have a value
    if (r.status().code() != tutti::StatusCode::INTERNAL) return 1;

    // Via explicit constructor with OK status
    tutti::Result<int> r2(tutti::Status(tutti::StatusCode::OK, "oops"));
    if (r2.ok()) return 1;
    if (r2.has_value()) return 1;
    if (r2.status().code() != tutti::StatusCode::INTERNAL) return 1;

    return 0;
}

// -------------------------------------------------------------------------
// 12. Do not call value() on the error path; verify via ok()/has_value().
// -------------------------------------------------------------------------
static int test_error_path_no_value_access() {
    tutti::Result<int> r = tutti::Result<int>::Failure(
        tutti::Status(tutti::StatusCode::BUSY, "device busy"));

    // Guard: must check before accessing value.
    if (r.ok()) return 1;
    if (r.has_value()) return 1;

    // We intentionally do NOT call r.value() here.
    // The contract says value() on a failure result is undefined.
    return 0;
}

// -------------------------------------------------------------------------

int main() {
    using TestFn = int (*)();
    const TestFn tests[] = {
        test_status_code_coverage,
        test_default_status_is_ok,
        test_ok_factory,
        test_error_codes,
        test_message_preserved,
        test_status_copy_move,
        test_result_int_success,
        test_result_string_success,
        test_result_int_error,
        test_result_unique_ptr_success_and_move,
        test_illegal_ok_without_value,
        test_error_path_no_value_access,
    };

    const int n = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; ++i) {
        int rc = tests[i]();
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: test %d returned %d\n", i, rc);
            return rc;
        }
    }

    std::printf("All %d status contract tests passed.\n", n);
    return 0;
}
