#pragma once

#include <string>
#include <variant>

namespace tutti::config {

struct LocalFileResolverConfig {
    bool require_raid0 = false;
};

struct MemfsResolverConfig {};

using ResolverConfig = std::variant<LocalFileResolverConfig,
                                    MemfsResolverConfig>;

struct ResolverSpec {
    std::string id;
    std::string type;
    std::string scheme;
    ResolverConfig config;
};

} // namespace tutti::config
