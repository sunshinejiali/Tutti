#pragma once
// tutti/spi/async_completion.h
//
// Optional DataPath capability: submit() returns once the commands are
// issued, and completion is established by a wait the caller enqueues on the
// stream that consumes the data (design: doc/design/async-nvme-completion.md).
//
// A DataPath that implements it also implements DataPath; StorageRuntime
// finds it with dynamic_cast. Host-side progress()/query() keep their usual
// meaning: an op becomes terminal once all of its entries have finished.

#include <cstddef>

#include <tutti/spi/data_path.h>

namespace tutti {

class AsyncCompletion {
public:
    virtual ~AsyncCompletion() = default;

    // Enqueue on `stream` a wait for every op in `ops` (all ops of this
    // DataPath): work enqueued on `stream` after this call runs only after
    // those ops have finished. The caller holds the DataPath's progress gate.
    virtual Status enqueue_wait(const DataPathOp* ops, std::size_t count,
                                void* stream) = 0;

    // Report the irreversible release separately from device-guard restore
    // errors. Runtime must not retry an op already removed by the DataPath.
    virtual Status release_retired(DataPathOp op, bool& retired) {
        const Status result = dynamic_cast<DataPath*>(this)->release(op);
        retired = result.ok();
        return result;
    }
};

} // namespace tutti
