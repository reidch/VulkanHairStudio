#pragma once

#include "character_surface.hpp"
#include "groom_asset.hpp"

namespace vh::groom {

struct ValidationReport {
    uint32_t visualInsideSamples = 0u;
    uint32_t baseInsideSamples = 0u;
    uint32_t guideProxyViolations = 0u;
    float minVisualDistance = 1.0e3f;
    float minBaseDistance = 1.0e3f;
    float minGuideProxyClearance = 1.0e3f;
    float deepestPenetration = 0.0f;

    bool contactClean() const {
        return visualInsideSamples==0u && baseInsideSamples==0u && guideProxyViolations==0u;
    }
    bool hasContactIssue() const { return !contactClean(); }
};

struct ContactResolveReport {
    uint32_t correctedGroups = 0u;
    uint32_t projectedConstraints = 0u;
    uint32_t iterations = 0u;
    float maxCorrection = 0.0f;
};

ValidationReport validateGroom(const vh::character::CharacterSurface& character,
                               const GroomAsset& groom,
                               const GroomBuildParams& params);

// Explicit authoring operation. One path-parameter correction field Delta_w(s)
// is solved per contact-coherence group and applied coherently to base, visual and reduced guides.
ContactResolveReport resolveContacts(const vh::character::CharacterSurface& character,
                                     const GroomBuildParams& params,
                                     GroomAsset& groom);

} // namespace vh::groom
