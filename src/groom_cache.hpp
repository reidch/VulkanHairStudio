#pragma once

#include "groom_asset.hpp"
#include <cstdint>
#include <filesystem>

namespace vh::groom {

struct DerivedGroomCacheKey {
    uint64_t styleFingerprint = 0u;
    uint64_t meshFingerprint = 0u;
    GroomBuildParams params{};
};

uint64_t fingerprintFile(const std::filesystem::path& file);
bool loadDerivedGroom(const std::filesystem::path& file,
                      const DerivedGroomCacheKey& key,
                      GroomAsset& out,
                      double* loadMs = nullptr);
void saveDerivedGroom(const std::filesystem::path& file,
                      const DerivedGroomCacheKey& key,
                      const GroomAsset& groom,
                      double* saveMs = nullptr);

} // namespace vh::groom
