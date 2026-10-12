#include "nvmeservice_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nvmeservice {
namespace {

template <typename T>
T get_or(const YAML::Node& node, const char* key, T fallback) {
    if (!node || !node[key]) return fallback;
    return node[key].as<T>();
}

int32_t parse_nonnegative_id(const YAML::Node& node, const char* field) {
    if (!node) throw std::runtime_error(std::string(field) + " is missing");
    const int64_t value = node.as<int64_t>();
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string(field) +
                                 " must be in [0, INT32_MAX]");
    }
    return static_cast<int32_t>(value);
}

std::string normalize_pci_bdf(const std::string& input) {
    unsigned int domain = 0;
    unsigned int bus = 0;
    unsigned int device = 0;
    unsigned int function = 0;
    int consumed = 0;
    if (std::sscanf(input.c_str(), "%x:%x:%x.%x%n", &domain, &bus, &device,
                    &function, &consumed) != 4 ||
        consumed != static_cast<int>(input.size()) || domain > 0xffff ||
        bus > 0xff || device > 0x1f || function > 7) {
        throw std::runtime_error("invalid nvmes[].pci_addr: " + input);
    }
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0')
        << std::setw(4) << domain << ':' << std::setw(2) << bus << ':'
        << std::setw(2) << device << '.' << function;
    return out.str();
}

void reject_removed_nvme_fields(const YAML::Node& node, size_t index) {
    if (node["mount_path"] || node["allowed_gpus"]) {
        throw std::runtime_error("nvmes[" + std::to_string(index) +
                                 "] requires backing_mount_path/allowed_accel_ids");
    }
}

void reject_removed_accelerator_fields(const YAML::Node& node, size_t index) {
    if (node["id"] || node["mount_path"]) {
        throw std::runtime_error("accelerators[" + std::to_string(index) +
                                 "] requires accel_id/view_root");
    }
}

// Parse service policies and array declarations.
void parse_common(const YAML::Node& root, ServiceConfig& config) {
    if (root["grpc"]) {
        config.grpc.endpoint =
            get_or<std::string>(root["grpc"], "endpoint", config.grpc.endpoint);
    }
    if (root["queue_pool"]) {
        config.queue_pool.default_per_client = get_or<int32_t>(
            root["queue_pool"], "default_per_client",
            config.queue_pool.default_per_client);
        config.queue_pool.max_per_client = get_or<int32_t>(
            root["queue_pool"], "max_per_client",
            config.queue_pool.max_per_client);
    }
    if (root["lease"]) {
        config.lease.heartbeat_interval_sec = get_or<uint32_t>(
            root["lease"], "heartbeat_interval_sec",
            config.lease.heartbeat_interval_sec);
        config.lease.timeout_sec = get_or<uint32_t>(
            root["lease"], "timeout_sec", config.lease.timeout_sec);
    }
    if (root["unmount_retry"]) {
        config.unmount_retry.interval_ms = get_or<uint32_t>(
            root["unmount_retry"], "interval_ms",
            config.unmount_retry.interval_ms);
        config.unmount_retry.max = get_or<uint32_t>(
            root["unmount_retry"], "max", config.unmount_retry.max);
    }
    if (root["raid0_arrays"]) {
        if (!root["raid0_arrays"].IsSequence()) {
            throw std::runtime_error("raid0_arrays must be a sequence");
        }
        for (const auto& node : root["raid0_arrays"]) {
            Raid0ArrayEntry entry;
            entry.name = get_or<std::string>(node, "name", "");
            entry.mount_path = get_or<std::string>(node, "mount_path", "");
            entry.auto_mount = get_or<bool>(node, "auto_mount", true);
            if (!node["device_ids"] || !node["device_ids"].IsSequence()) {
                throw std::runtime_error("raid0_arrays[].device_ids must be a sequence");
            }
            for (const auto& id : node["device_ids"]) {
                entry.device_ids.push_back(
                    parse_nonnegative_id(id, "raid0_arrays[].device_ids[]"));
            }
            config.raid0_arrays.push_back(std::move(entry));
        }
    }
}

// Canonical schema: explicit accelerator and NVMe identities are mandatory.
void parse_canonical(const YAML::Node& root, ServiceConfig& config) {
    if (!root["accelerators"] || !root["accelerators"].IsSequence()) {
        throw std::runtime_error("accelerators must be a sequence");
    }
    size_t accelerator_index = 0;
    for (const auto& node : root["accelerators"]) {
        reject_removed_accelerator_fields(node, accelerator_index);
        AcceleratorEntry entry;
        entry.accel_id =
            parse_nonnegative_id(node["accel_id"], "accelerators[].accel_id");
        entry.view_root = get_or<std::string>(node, "view_root", "");
        config.accelerators.push_back(std::move(entry));
        ++accelerator_index;
    }

    if (!root["nvmes"] || !root["nvmes"].IsSequence()) {
        throw std::runtime_error("nvmes must be a sequence");
    }
    size_t index = 0;
    for (const auto& node : root["nvmes"]) {
        reject_removed_nvme_fields(node, index);
        NvmeEntry entry;
        entry.device_id =
            parse_nonnegative_id(node["device_id"], "nvmes[].device_id");
        entry.pci_addr = normalize_pci_bdf(
            get_or<std::string>(node, "pci_addr", ""));
        entry.backing_mount_path =
            get_or<std::string>(node, "backing_mount_path", "");
        entry.namespace_id = get_or<uint32_t>(node, "namespace_id", 1u);
        entry.kernel_ioq_cap = get_or<uint32_t>(node, "kernel_ioq_cap", 0u);
        entry.auto_mount = get_or<bool>(node, "auto_mount", true);
        if (node["allowed_accel_ids"]) {
            if (!node["allowed_accel_ids"].IsSequence()) {
                throw std::runtime_error("nvmes[].allowed_accel_ids must be a sequence");
            }
            for (const auto& id : node["allowed_accel_ids"]) {
                entry.allowed_accel_ids.push_back(parse_nonnegative_id(
                    id, "nvmes[].allowed_accel_ids[]"));
            }
        }
        config.nvmes.push_back(std::move(entry));
        ++index;
    }
}

} // namespace

std::optional<ServiceConfig> parse_config_file(
    const std::string& path, std::string* error) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (root["gpus"]) {
            throw std::runtime_error("gpus schema removed; use accelerators with explicit accel_id/view_root");
        }
        ServiceConfig config;
        parse_common(root, config);
        parse_canonical(root, config);

        std::string validation_error;
        if (!validate_config(config, &validation_error)) {
            throw std::runtime_error("validation failed: " + validation_error);
        }

        // Expand an omitted ACL once, then keep the internal snapshot sorted and
        // deterministic for list, allowed selection, and compatibility RPCs.
        std::vector<int32_t> all_accelerators;
        all_accelerators.reserve(config.accelerators.size());
        for (const auto& accelerator : config.accelerators) {
            all_accelerators.push_back(accelerator.accel_id);
        }
        std::sort(all_accelerators.begin(), all_accelerators.end());
        for (auto& nvme : config.nvmes) {
            if (nvme.allowed_accel_ids.empty()) {
                nvme.allowed_accel_ids = all_accelerators;
            } else {
                std::sort(nvme.allowed_accel_ids.begin(),
                          nvme.allowed_accel_ids.end());
            }
        }
        return config;
    } catch (const YAML::Exception& exception) {
        if (error) *error = std::string("YAML parse error: ") + exception.what();
    } catch (const std::exception& exception) {
        if (error) *error = std::string("parse error: ") + exception.what();
    }
    return std::nullopt;
}

bool validate_config(const ServiceConfig& config, std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (config.grpc.endpoint.empty()) return fail("grpc.endpoint is empty");
    if (config.accelerators.empty()) return fail("accelerators list is empty");
    if (config.nvmes.empty()) return fail("nvmes list is empty");

    // Validate identity/path uniqueness before checking ACL references.
    std::set<int32_t> accelerator_ids;
    std::set<std::string> all_paths;
    for (const auto& accelerator : config.accelerators) {
        if (accelerator.accel_id < 0) {
            return fail("accelerators[].accel_id must be >= 0");
        }
        if (!accelerator_ids.insert(accelerator.accel_id).second) {
            return fail("duplicate accelerators[].accel_id: " +
                        std::to_string(accelerator.accel_id));
        }
        if (accelerator.view_root.empty()) {
            return fail("accelerators[accel_id=" +
                        std::to_string(accelerator.accel_id) +
                        "].view_root is empty");
        }
        if (!all_paths.insert(accelerator.view_root).second) {
            return fail("duplicate configured path: " + accelerator.view_root);
        }
    }

    // RAID0 members share their array's mount path; that path is claimed
    // once, by the array.
    std::map<int32_t, const Raid0ArrayEntry*> array_of;
    std::set<std::string> array_names;
    for (const auto& array : config.raid0_arrays) {
        if (array.name.empty() ||
            array.name.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                std::string::npos) {
            return fail("raid0_arrays[].name must be non-empty [A-Za-z0-9_-]");
        }
        if (!array_names.insert(array.name).second) {
            return fail("duplicate raid0_arrays[].name: " + array.name);
        }
        if (array.device_ids.size() < 2) {
            return fail("raid0_arrays[" + array.name + "] needs at least two devices");
        }
        if (array.mount_path.empty()) {
            return fail("raid0_arrays[" + array.name + "].mount_path is empty");
        }
        if (!all_paths.insert(array.mount_path).second) {
            return fail("duplicate configured path: " + array.mount_path);
        }
        for (int32_t id : array.device_ids) {
            if (!array_of.emplace(id, &array).second) {
                return fail("device_id=" + std::to_string(id) +
                            " is listed in more than one raid0 array slot");
            }
        }
    }

    std::set<int32_t> device_ids;
    std::set<std::string> pci_addresses;
    for (const auto& nvme : config.nvmes) {
        if (nvme.device_id < 0) return fail("nvmes[].device_id must be >= 0");
        if (!device_ids.insert(nvme.device_id).second) {
            return fail("duplicate nvmes[].device_id: " +
                        std::to_string(nvme.device_id));
        }
        if (nvme.pci_addr.empty()) return fail("nvmes[].pci_addr is empty");
        if (!pci_addresses.insert(nvme.pci_addr).second) {
            return fail("duplicate nvmes[].pci_addr: " + nvme.pci_addr);
        }
        if (nvme.backing_mount_path.empty()) {
            return fail("nvmes[device_id=" + std::to_string(nvme.device_id) +
                        "].backing_mount_path is empty");
        }
        const auto member = array_of.find(nvme.device_id);
        if (member != array_of.end()) {
            if (nvme.backing_mount_path != member->second->mount_path) {
                return fail("nvmes[device_id=" + std::to_string(nvme.device_id) +
                            "] is a member of raid0 array " + member->second->name +
                            "; its backing_mount_path must be " +
                            member->second->mount_path);
            }
            if (nvme.auto_mount) {
                return fail("nvmes[device_id=" + std::to_string(nvme.device_id) +
                            "] is a raid0 member and must set auto_mount: false");
            }
        } else if (!all_paths.insert(nvme.backing_mount_path).second) {
            return fail("duplicate configured path: " + nvme.backing_mount_path);
        }
        std::set<int32_t> acl;
        for (int32_t accel_id : nvme.allowed_accel_ids) {
            if (!accelerator_ids.count(accel_id)) {
                return fail("nvmes[device_id=" +
                            std::to_string(nvme.device_id) +
                            "].allowed_accel_ids references unknown accel_id=" +
                            std::to_string(accel_id));
            }
            if (!acl.insert(accel_id).second) {
                return fail("nvmes[device_id=" +
                            std::to_string(nvme.device_id) +
                            "].allowed_accel_ids has duplicate accel_id=" +
                            std::to_string(accel_id));
            }
        }
    }
    for (const auto& [id, array] : array_of) {
        if (!device_ids.count(id)) {
            return fail("raid0_arrays[" + array->name +
                        "] references unknown device_id=" + std::to_string(id));
        }
    }

    // Queue policy is admission guidance; kernel-reported capacity is checked
    // later by ServiceState when a controller has been brought up.
    if (config.queue_pool.default_per_client <= 0 ||
        config.queue_pool.max_per_client <= 0 ||
        config.queue_pool.default_per_client > config.queue_pool.max_per_client) {
        return fail("queue_pool.default_per_client must be in (0, max_per_client]");
    }
    if (config.lease.heartbeat_interval_sec == 0 ||
        config.lease.timeout_sec == 0 ||
        config.lease.heartbeat_interval_sec >= config.lease.timeout_sec) {
        return fail("lease.heartbeat_interval_sec must be in (0, timeout_sec)");
    }
    if (config.unmount_retry.interval_ms == 0 || config.unmount_retry.max == 0) {
        return fail("unmount_retry.interval_ms and max must be > 0");
    }
    return true;
}

bool validate_uniform_block_size(const std::vector<uint32_t>& block_sizes,
                                 std::string* error) {
    if (block_sizes.empty()) {
        if (error) *error = "no available NVMe block sizes were reported";
        return false;
    }
    for (size_t index = 1; index < block_sizes.size(); ++index) {
        if (block_sizes[index] == block_sizes.front()) continue;
        if (error) {
            *error = "NVMe block sizes are not uniform: first resource reports " +
                     std::to_string(block_sizes.front()) + ", resource " +
                     std::to_string(index) + " reports " +
                     std::to_string(block_sizes[index]);
        }
        return false;
    }
    return true;
}

} // namespace nvmeservice
