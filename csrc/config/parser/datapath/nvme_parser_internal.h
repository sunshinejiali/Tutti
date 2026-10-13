#pragma once

#include "csrc/config/parser/parser_internal.h"

namespace tutti::config::detail {

template <typename Config>
Status parse_nvme_datapath_tuning(const YAML::Node& config,
                                  const std::string& path,
                                  Config& parsed) {
    if (!config) return Status::Ok();
    Status status = validate_keys(
        config, path,
        {"max_in_flight_operations", "max_batch_entries"});
    if (!status.ok()) return status;
    status = read_optional_u64(config, "max_in_flight_operations", path,
                               parsed.max_in_flight_operations);
    if (!status.ok()) return status;
    status = read_optional_u64(config, "max_batch_entries", path,
                               parsed.max_batch_entries);
    return status;
}

} // namespace tutti::config::detail
