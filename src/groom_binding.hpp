#pragma once

#include <cstdint>
#include <array>
#include <cmath>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include "character_surface.hpp"
#include "scalp_domain.hpp"
#include "groom_space.hpp"

namespace vh::groom {

struct BoundRoot {
    vh::scalp::Sample canonical{};
    vh::character::RootBinding binding{};
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f,1.0f,0.0f};
};

// A similarity transform is deliberate.  A character may move/rotate/scale a
// groom, but cannot anisotropically stretch its curls, wisps or cut silhouette.
struct GroomPlacement {
    glm::vec3 origin{0.0f};
    glm::mat3 worldFromCanonical{1.0f};
    float uniformScale = 1.0f;

    glm::vec3 point(glm::vec3 normalizedCanonical) const {
        return origin + worldFromCanonical*(normalizedCanonical*kCanonicalHeadRadii*uniformScale);
    }
    glm::vec3 direction(glm::vec3 d) const {
        const glm::vec3 q=worldFromCanonical*d;
        const float l2=glm::dot(q,q);
        return l2>1e-12f?q/std::sqrt(l2):glm::vec3(0,1,0);
    }
    glm::vec3 toCanonicalNormalized(glm::vec3 world) const {
        return (glm::transpose(worldFromCanonical)*(world-origin))/(kCanonicalHeadRadii*uniformScale);
    }
};

struct BindingAudit {
    uint32_t rootCount = 0u;
    uint32_t fallbackBindings = 0u;
    uint32_t invalidBindings = 0u;
    float fallbackFraction = 0.0f;
    float minSampledRootSpacing = 1.0f;
    float maxSampledRootSpacing = 0.0f;
    float placementScale = 1.0f;
    float scalpArea = 0.0f;
};

struct GroomBinding {
    std::vector<BoundRoot> roots;
    // Stable spatial order used by every hairstyle. It is computed once from
    // canonical scalp UVs, so render clusters never need to be resorted after
    // a style edit. Root identity remains the canonical Sample::id.
    std::vector<uint32_t> renderOrder;
    GroomPlacement placement{};
    BindingAudit audit{};
};

class GroomRetargeter {
public:
    // One immutable root-ID map per character.  No hairstyle is allowed to
    // resample follicles or derive new section membership from target geometry.
    GroomBinding bind(const vh::character::CharacterSurface& character,
                      uint32_t rootCount,
                      uint32_t seed) const;

    // Root/triangle correspondence depends only on character mesh + canonical
    // follicle stream. Cache it as a derived asset; hairstyle changes never
    // trigger 110k scalp ray casts again.
    GroomBinding loadOrBind(const vh::character::CharacterSurface& character,
                            const std::filesystem::path& cachePath,
                            uint32_t rootCount,
                            uint32_t seed) const;
};

} // namespace vh::groom
