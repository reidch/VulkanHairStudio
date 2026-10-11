#pragma once

#include <cmath>
#include <glm/glm.hpp>

namespace vh::groom {

// Canonical authoring scale.  Character replacement is a similarity placement
// plus exact follicle binding; target skull aspect ratios never stretch curls,
// bundles, or hairstyle silhouette anisotropically.
inline const glm::vec3 kCanonicalHeadRadii{0.145f,0.190f,0.155f};

inline glm::vec3 safeDirection(glm::vec3 d,glm::vec3 fallback={0,1,0}) {
    const float l2=glm::dot(d,d);
    return l2>1e-12f?d/std::sqrt(l2):fallback;
}

} // namespace vh::groom
