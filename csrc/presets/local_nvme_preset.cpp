// csrc/presets/local_nvme_preset.cpp
//
// Preset assembly layer implementation.
// Includes private headers to construct DataPaths + resolvers, returns
// public types (StorageRuntime + RuntimeTelemetry).

#include <algorithm>
#include <tutti/presets/local_nvme.h>

#include <tutti/storage_runtime.h>
#include <tutti/cuda_like.h>
#include "csrc/common/backend_ids.h"
#include "csrc/common/snvme_identity.h"

#include <cctype>
#include <stdexcept>

// Private headers — included here ONLY, never by consumer code.
#include "csrc/data_paths/local_nvme/local_nvme_data_path.h"
#include "csrc/data_paths/local_nvme/metadata/cache_sizing.h"
#include "csrc/data_paths/striped_local_nvme/striped_data_path.h"
#include "csrc/data_paths/striped_local_nvme/striped_async_data_path.h"
#include "csrc/resolvers/local_file/resolver.h"
#include "csrc/resolvers/local_file/multi_mount_resolver.h"
#include "csrc/resolvers/local_file/raid0_resolver.h"

namespace tutti::presets {

namespace {

std::string normalize_bdf(std::string bdf) {
    while (!bdf.empty() && std::isspace(static_cast<unsigned char>(bdf.back()))) {
        bdf.pop_back();
    }
    std::size_t first = 0;
    while (first < bdf.size() &&
           std::isspace(static_cast<unsigned char>(bdf[first]))) {
        ++first;
    }
    bdf.erase(0, first);
    for (char& c : bdf) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (bdf.size() == 7 && bdf[2] == ':' && bdf[5] == '.') {
        bdf = "0000:" + bdf;
    }
    return bdf;
}

// The character device of the controller at `pci_bdf`, established by the
// kernel (NVM_GET_DEV_INFO -> disk -> PCI), never derived from a number:
// /dev/ssnvmeN and controller snvmeM are numbered by unrelated allocators.
// No match, or more than one, throws -- a wrong guess sends GPU IO to
// another controller's namespace.
std::string chrdev_for_bdf(const std::string& pci_bdf) {
    if (pci_bdf.empty()) {
        throw std::invalid_argument("NVMe device pci_bdf must not be empty");
    }
    const std::string wanted = normalize_bdf(pci_bdf);
    const std::string chrdev = tutti::detail::snvme_identity::chrdev_for_pci(wanted);
    if (chrdev.empty()) {
        throw std::runtime_error(
            "no unique /dev/ssnvme* character device reports PCI " + wanted +
            " (is its controller brought up by tutti_daemon?)");
    }
    return chrdev;
}

} // namespace

RuntimeWithTelemetry make_local_nvme_runtime(const LocalNvmePreset& input) {
    auto p = input;
    if (p.max_batch_entries == 0) p.max_batch_entries = 256;
    if (p.max_in_flight_operations == 0) p.max_in_flight_operations = 16;
    namespace lnvme = tutti::data_paths::local_nvme;
    using tutti::resolvers::local_file::LocalFileResolver;
    using tutti::resolvers::local_file::BackingDeviceConfig;

    // Heap-allocate the DataPath + resolver — injected as raw pointers
    // into RuntimeComponents, lifetime tied to the runtime.
    auto* dp = new lnvme::LocalNvmeDataPath(
        chrdev_for_bdf(p.device.pci_bdf),
        p.accel_id,
        p.num_queues,
        p.device.namespace_id,
        p.device.block_size,
        /*mdts_bytes=*/0,
        p.max_batch_entries,
        /*cq_poll_budget=*/0,
        /*handle_cache_capacity=*/0,
        lnvme::kAutoPrpCacheCapacity,
        p.max_in_flight_operations,
        /*max_batch_requests=*/0,
        /*max_request_bytes_override=*/0,
        /*handle_cache_l2_capacity=*/0,
        p.device.pci_bdf);

    auto* resolver = new LocalFileResolver(
        p.device.pci_bdf,
        p.device.namespace_id,
        p.device.block_size,
        BackingDeviceConfig{p.device.backing_device, 0});

    // Constructed here and owned by nobody else, so hand them to the runtime.
    // Previously these were raw `new`ed and never deleted: leaked on success,
    // and leaked again whenever initialize_components_() failed.
    OwnedComponents owned;
    owned.resolvers.push_back(OwnedResolver{
        "file", std::unique_ptr<StorageTargetResolver>(resolver)});
    owned.data_paths.push_back(OwnedDataPath{
        std::string(tutti::detail::backend_ids::kExt4DataPathKey),
        std::unique_ptr<DataPath>(dp), DataPathConfig{"local_nvme"}});

    RuntimeConfig runtime_config;
    runtime_config.accel_id = p.accel_id;
    // Terminal-but-unreleased IOs are bounded by the DataPath's op arena
    // (2 x max_in_flight_operations slots, held until release). The default
    // 64 is below one whole-layer plan: an async read plan keeps up to
    // num_layers completed handles until its next poll, any other wait drives
    // them terminal, and sync submits arriving meanwhile were rejected.
    runtime_config.max_terminal_results = std::max<std::uint64_t>(
        runtime_config.max_terminal_results,
        2ull * p.max_in_flight_operations);
    auto created = StorageRuntime::create_owning(runtime_config, std::move(owned));
    if (!created.ok()) {
        RuntimeWithTelemetry result;
        result.creation_status = created.status();
        return result;
    }

    RuntimeWithTelemetry result;
    result.runtime = std::move(created).value();
    result.telemetry = RuntimeTelemetry{
        [dp]() -> std::uint64_t { return dp->test_submit_call_count(); },
        [dp]() -> std::uint64_t { return dp->test_kernel_launch_count(); },
        [dp]() { dp->test_reset_submit_counters(); }
    };
    return result;
}

RuntimeWithTelemetry make_striped_nvme_runtime(const StripedNvmePreset& input) {
    auto p = input;
    if (p.max_batch_entries == 0) p.max_batch_entries = 256;
    if (p.max_in_flight_operations == 0) p.max_in_flight_operations = 16;
    namespace snvme = tutti::data_paths::striped_local_nvme;
    using tutti::resolvers::local_file::MultiMountLocalFileResolver;
    using tutti::detail::backend_ids::kStripedDataPathKey;

    if (p.require_raid0) {
        bool one_mount = p.devices.size() >= 2 && !p.devices.front().mount_path.empty();
        for (const auto& device : p.devices) {
            one_mount = one_mount && device.mount_path == p.devices.front().mount_path;
        }
        if (!one_mount) {
            RuntimeWithTelemetry result;
            result.creation_status = Status(StatusCode::INVALID_ARGUMENT,
                "raid0 requires at least two namespaces on one array mount");
            return result;
        }
    }
    // All devices behind one mount: that mount is an md RAID0 over them, and
    // one resolver serves the array (payloads carry the md layout).
    bool one_mount = p.devices.size() > 1;
    for (const auto& d : p.devices) {
        one_mount = one_mount && d.mount_path == p.devices.front().mount_path;
    }
    std::unique_ptr<StorageTargetResolver> resolver;
    if (one_mount) {
        std::vector<tutti::resolvers::local_file::Raid0MemberNamespace> members;
        for (const auto& d : p.devices) {
            members.push_back({d.backing_device, d.pci_bdf, d.namespace_id,
                               d.block_size});
        }
        std::string why;
        auto raid0 = tutti::resolvers::local_file::make_raid0_resolver(
            p.devices.front().mount_path, members,
            std::string(kStripedDataPathKey), &why);
        if (!raid0) {
            RuntimeWithTelemetry result;
            result.creation_status = Status(StatusCode::INVALID_ARGUMENT, why);
            return result;
        }
        resolver = std::move(raid0);
    } else {
        // One "file" resolver dispatching by mount: a slot's file path
        // already names the device it lives on (placement rotated slots
        // across mounts), so resolution is prefix matching plus the full
        // single-device pipeline.
        std::vector<MultiMountLocalFileResolver::MountBinding> bindings;
        bindings.reserve(p.devices.size());
        for (const auto& d : p.devices) {
            bindings.push_back({d.mount_path, d.pci_bdf, d.namespace_id,
                                d.block_size, d.backing_device,
                                std::string(kStripedDataPathKey)});
        }
        resolver = std::make_unique<MultiMountLocalFileResolver>(std::move(bindings));
    }

    std::vector<snvme::DeviceDescriptor> sdevs;
    for (const auto& d : p.devices) {
        sdevs.push_back({chrdev_for_bdf(d.pci_bdf), d.namespace_id,
                         (std::uint32_t)p.accel_id, p.num_queues, d.block_size,
                         d.pci_bdf});
    }

    snvme::StripedDataPath* dp = nullptr;
    if (snvme::StripedAsyncDataPath::requested()) {
        dp = new snvme::StripedAsyncDataPath(
            std::move(sdevs), (std::uint32_t)p.accel_id,
            /*mdts_override=*/0, /*cq_poll_budget=*/0,
            p.max_batch_entries, p.max_in_flight_operations,
            tutti::data_paths::local_nvme::kAutoPrpCacheCapacity);
    } else {
        dp = new snvme::StripedDataPath(
            std::move(sdevs), (std::uint32_t)p.accel_id,
            /*mdts_override=*/0, /*cq_poll_budget=*/0,
            p.max_batch_entries, p.max_in_flight_operations,
            tutti::data_paths::local_nvme::kAutoPrpCacheCapacity);
    }

    // Same reasoning as the single-device factory above: inline construction
    // with no other owner, so the runtime takes them and frees them after
    // shutdown(). The delegates are unique_ptr-held inside the table
    // resolver, so only the top level changes here.
    OwnedComponents owned;
    owned.resolvers.push_back(OwnedResolver{
        "file", std::move(resolver)});
    owned.data_paths.push_back(OwnedDataPath{
        std::string(kStripedDataPathKey),
        std::unique_ptr<DataPath>(dp), DataPathConfig{"nvme-device-group"}});

    RuntimeConfig runtime_config;
    runtime_config.accel_id = p.accel_id;
    // Terminal-but-unreleased IOs are bounded by the DataPath's op arena
    // (2 x max_in_flight_operations slots, held until release). The default
    // 64 is below one whole-layer plan: an async read plan keeps up to
    // num_layers completed handles until its next poll, any other wait drives
    // them terminal, and sync submits arriving meanwhile were rejected.
    runtime_config.max_terminal_results = std::max<std::uint64_t>(
        runtime_config.max_terminal_results,
        2ull * p.max_in_flight_operations);
    auto created = StorageRuntime::create_owning(runtime_config, std::move(owned));
    if (!created.ok()) {
        RuntimeWithTelemetry result;
        result.creation_status = created.status();
        return result;
    }

    RuntimeWithTelemetry result;
    result.runtime = std::move(created).value();
    result.telemetry = RuntimeTelemetry{
        [dp]() -> std::uint64_t { return dp->test_submit_call_count(); },
        [dp]() -> std::uint64_t { return dp->test_kernel_launch_count(); },
        [dp]() { dp->test_reset_submit_counters(); }
    };
    return result;
}

} // namespace tutti::presets
