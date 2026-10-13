#include "csrc/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_local_file_resolver(const YAML::Node& config,
                                 const std::string& path,
                                 ResolverSpec& resolver) {
    LocalFileResolverConfig parsed;
    if (config) {
        Status status = validate_keys(config, path, {"topology"});
        if (!status.ok()) return status;
        if (config["topology"]) {
            std::string topology;
            status = read_required_string(config, "topology", path, topology);
            if (!status.ok()) return status;
            if (topology != "auto" && topology != "raid0") {
                return parse_error(path + ".topology must be auto or raid0");
            }
            parsed.require_raid0 = topology == "raid0";
        }
    }
    resolver.config = parsed;
    return Status::Ok();
}
} // namespace tutti::config::detail
