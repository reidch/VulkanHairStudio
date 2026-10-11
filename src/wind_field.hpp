#pragma once
#include <glm/glm.hpp>
#include <cstdint>

namespace vh::wind {

enum class Type : uint32_t { Directional=0u, Fan=1u, Vortex=2u };

struct Field {
    Type type = Type::Directional;
    bool enabled = false;
    float speed = 4.0f;
    glm::vec3 direction{1.0f,0.0f,0.0f};
    glm::vec3 position{-0.35f,0.02f,0.20f};
    float range = 0.85f;
    float coneDegrees = 35.0f;
    float gust = 0.12f;
    float turbulence = 0.10f;
};

} // namespace vh::wind
