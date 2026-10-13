#include <tutti/config/tutti_runtime_config_parser.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <unistd.h>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,      \
                         #condition);                                          \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

class TempConfig {
public:
    explicit TempConfig(const std::string& contents) {
        char path[] = "/tmp/tutti-config-parser-XXXXXX";
        const int descriptor = ::mkstemp(path);
        CHECK(descriptor >= 0);
        if (descriptor >= 0) ::close(descriptor);
        path_ = path;
        std::ofstream stream(path_);
        stream << contents;
    }
    ~TempConfig() { if (!path_.empty()) ::unlink(path_.c_str()); }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

std::string local_yaml(const std::string& backend_config = "{}") {
    return
        "accelerator: {profile: CUDA}\n"
        "runtime: {accel_id: 0}\n"
        "storage:\n"
        "  resources:\n"
        "    - id: nvme-0\n"
        "      type: nvme\n"
        "      provider: {type: nvme-service, endpoint: endpoint}\n"
        "      allocation: {selection: explicit, device_ids: [2], queues_per_controller: 4}\n"
        "  resolvers:\n"
        "    - {id: resolver-0, type: local-file, scheme: file, config: {}}\n"
        "  datapaths:\n"
        "    - id: datapath-0\n"
        "      type: local-nvme\n"
        "      config: {max_in_flight_operations: 10, max_batch_entries: 11}\n"
        "  backends:\n"
        "    - id: backend-0\n"
        "      contract: ext4-local-nvme\n"
        "      resolver: resolver-0\n"
        "      datapath: datapath-0\n"
        "      resource: nvme-0\n"
        "      config: " + backend_config + "\n";
}

} // namespace

int main() {
    {
        TempConfig file(local_yaml());
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) {
            const auto& spec = parsed.value();
            CHECK(spec.accelerator.profile == "CUDA");
            CHECK(spec.runtime.accel_id == 0);
            const auto* resource = std::get_if<tutti::config::NvmeResourceConfig>(
                &spec.storage.resources.front().config);
            CHECK(resource != nullptr);
            if (resource != nullptr) {
                CHECK(resource->allocation.nvme_device_ids ==
                      std::vector<std::int32_t>{2});
            }
            const auto* datapath =
                std::get_if<tutti::config::LocalNvmeDataPathConfig>(
                    &spec.storage.datapaths.front().config);
            CHECK(datapath != nullptr);
            if (datapath != nullptr) {
                CHECK(datapath->max_in_flight_operations == 10);
                CHECK(datapath->max_batch_entries == 11);
            }
        }
    }
    {
        TempConfig file("storage:\n  resources: {}\n  resolvers: []\n  datapaths: []\n  backends: []\n");
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find("storage.resources") !=
              std::string::npos);
    }
    {
        TempConfig file(local_yaml("{stripe_unit: 4096}"));
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find("unknown field stripe_unit") !=
              std::string::npos);
    }
    {
        // 2026-09-22: the multi-device backend accepts no config keys any
        // more (stripe_unit was removed with the striping itself), so any
        // key under config is rejected by the parser.
        const std::string striped_bad_type =
            "accelerator: {profile: CUDA}\n"
            "runtime: {accel_id: 0}\n"
            "storage:\n"
            "  resources:\n"
            "    - {id: nvme, type: nvme, provider: {type: nvme-service, endpoint: e}, allocation: {selection: device-group, device_ids: [0, 1], queues_per_controller: 2}}\n"
            "  resolvers: [{id: r, type: local-file, scheme: file}]\n"
            "  datapaths: [{id: d, type: nvme-device-group}]\n"
            "  backends: [{id: b, contract: nvme-device-group, resolver: r, datapath: d, resource: nvme, config: {stripe_unit: 65536}}]\n";
        TempConfig file(striped_bad_type);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find("stripe_unit") !=
              std::string::npos);
    }
    {
        std::string yaml = local_yaml();
        const auto position = yaml.find("  resolvers:\n");
        yaml.insert(position,
            "    - id: nvme-0\n"
            "      type: nvme\n"
            "      provider: {type: nvme-service, endpoint: second}\n"
            "      allocation: {selection: explicit, device_ids: [3], queues_per_controller: 1}\n");
        TempConfig file(yaml);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) CHECK(!parsed.value().validate().ok());
    }
    {
        const std::string striped =
            "accelerator: {profile: CUDA}\n"
            "runtime: {accel_id: 0}\n"
            "storage:\n"
            "  resources:\n"
            "    - {id: nvme, type: nvme, provider: {type: nvme-service, endpoint: e}, allocation: {selection: device-group, device_ids: [0, 1], queues_per_controller: 2}}\n"
            "  resolvers: [{id: r, type: local-file, scheme: file}]\n"
            "  datapaths: [{id: d, type: nvme-device-group}]\n"
            "  backends: [{id: b, contract: nvme-device-group, resolver: r, datapath: d, resource: nvme}]\n";
        TempConfig file(striped);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) {
            // 2026-09-22: the multi-device backend has no knobs left, so a
            // parsed StripedLocalNvmeBackendConfig is all there is to check.
            const auto* config =
                std::get_if<tutti::config::StripedLocalNvmeBackendConfig>(
                    &parsed.value().storage.backends.front().config);
            CHECK(config != nullptr);
        }
    }
    {
        for (const char* key : {"prp_cache_capacity", "handle_cache_capacity", "handle_cache_l2_capacity", "io_granularity"}) {
            auto yaml = local_yaml();
            const auto position = yaml.find("max_in_flight_operations: 10");
            yaml.insert(position, std::string(key) + ": 32, ");
            TempConfig file(yaml);
            auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
            CHECK(!parsed.ok());
            CHECK(parsed.status().message().find(key) != std::string::npos);
        }
    }
    {
        const auto source = std::string(TUTTI_TEST_REPOSITORY_ROOT) + "/config/storage_runtime/nvme_device_group.yaml";
        std::ifstream input(source);
        const std::string current((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        for (const auto& replacement : {
                 std::pair<std::string, std::string>{"selection: device-group", "selection: multi"},
                 {"type: nvme-device-group", "type: multi-nvme"}}) {
            auto yaml = current;
            const auto position = yaml.find(replacement.first);
            CHECK(position != std::string::npos);
            if (position == std::string::npos) continue;
            yaml.replace(position, replacement.first.size(), replacement.second);
            TempConfig file(yaml);
            CHECK(!tutti::config::parse_tutti_runtime_config(file.path()).ok());
        }
    }
    {
        for (const char* path : {
                 "config/storage_runtime/local_nvme.yaml",
                 "config/storage_runtime/nvme_device_group.yaml",
                 "config/storage_runtime/raid0.yaml"}) {
            auto parsed = tutti::config::parse_tutti_runtime_config(
                std::string(TUTTI_TEST_REPOSITORY_ROOT) + "/" + path);
            CHECK(parsed.ok());
            if (parsed.ok()) {
                CHECK(parsed.value().validate().ok());
                const auto& resolver = std::get<tutti::config::LocalFileResolverConfig>(
                    parsed.value().storage.resolvers.front().config);
                CHECK(resolver.require_raid0 == (std::string(path) == "config/storage_runtime/raid0.yaml"));
                const auto& datapath = parsed.value().storage.datapaths.front();
                const auto& budget = datapath.type == "local-nvme"
                    ? static_cast<const tutti::config::NvmeDataPathTuning&>(
                        std::get<tutti::config::LocalNvmeDataPathConfig>(datapath.config))
                    : static_cast<const tutti::config::NvmeDataPathTuning&>(
                        std::get<tutti::config::StripedLocalNvmeDataPathConfig>(datapath.config));
                CHECK(budget.max_in_flight_operations == 16);
                CHECK(budget.max_batch_entries == 256);
            }
        }
    }
    std::printf("parser tests: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
