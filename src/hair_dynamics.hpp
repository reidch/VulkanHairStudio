#pragma once

#include "groom_asset.hpp"
#include <cstdint>

namespace vh::physics {

constexpr uint32_t kVolumeGridDim = 48u;
constexpr uint32_t kVolumeGridCount = kVolumeGridDim * kVolumeGridDim * kVolumeGridDim;

struct alignas(16) VolumeAccumGpu {
    uint32_t occupancy = 0u;
    int32_t momentumX = 0;
    int32_t momentumY = 0;
    int32_t momentumZ = 0;
};

struct alignas(16) VolumeStateGpu { glm::vec4 velocityPacking{0.0f}; };

static_assert(sizeof(VolumeAccumGpu)==16u);
static_assert(sizeof(VolumeStateGpu)==16u);

// User-facing hair material. Solver-level compliance is derived from these
// intuitive controls in the application layer; topology/groom generation does
// not depend on physical tuning.
struct HairMaterial {
    float stretchStiffness = 0.96f;
    float bendStiffness = 0.72f;
    float damping = 0.34f;
    float staticFriction = 0.55f;
    float dynamicFriction = 0.34f;
    float styleMemory = 0.58f;
    float airDrag = 0.46f;
    float bulkFullness = 1.0f;
};

} // namespace vh::physics
