#pragma once
#include "groom_asset.hpp"
#include "groom_binding.hpp"

namespace vh::groom {

// Constructs a rest groom from style parameters and scalp bindings.
class GroomBuilder {
public:
    GroomAsset build(const GroomBinding& binding,
                     const GroomBuildParams& params,
                     const GroomStyle& style,
                     const GroomAsset* reusableBasis = nullptr) const;
};

} // namespace vh::groom
