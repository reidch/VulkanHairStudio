#pragma once
#include "groom_asset.hpp"
#include "groom_binding.hpp"

namespace vh::groom {

// Pure authoring pipeline. It never queries collision, projects geometry, or
// decides whether a groom is physics-ready.
class GroomBuilder {
public:
    GroomAsset build(const GroomBinding& binding,
                     const GroomBuildParams& params,
                     const GroomStyle& style,
                     const GroomAsset* reusableBasis = nullptr) const;
};

} // namespace vh::groom
