#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace vh::scalp {

constexpr float kPi = 3.14159265358979323846f;
inline float fract(float x) { return x - std::floor(x); }

// Canonical follicle domain. This is intentionally independent of every
// character mesh: strand i receives the same material/scalp coordinate for the
// same seed, regardless of the head it is later bound to.
inline float thetaMax(float phi) {
    const float front=std::max(0.0f,std::sin(phi));
    const float back =std::max(0.0f,-std::sin(phi));
    const float side =std::abs(std::cos(phi));
    return std::clamp(1.28f-0.14f*front+0.13f*back+0.025f*side,1.10f,1.43f);
}

struct Sample {
    uint32_t id = 0u;
    glm::vec2 uv{0.0f};
    glm::vec3 direction{0,1,0}; // normalized canonical-head coordinate
};

inline Sample sample(uint32_t index, uint32_t seed) {
    constexpr float a1=0.7548776662466927f;
    constexpr float a2=0.5698402909980532f;
    const float u=fract(0.5f+a1*float(index+1u+seed));
    const float v=fract(0.5f+a2*float(index+1u+3u*seed));
    const float phi=-kPi+2.0f*kPi*u;
    constexpr float thetaMin=0.075f;
    const float tmax=thetaMax(phi);
    const float c0=std::cos(thetaMin), c1=std::cos(tmax);
    const float theta=std::acos(std::clamp(c0+(c1-c0)*v,-1.0f,1.0f));
    const float st=std::sin(theta);
    Sample out;
    out.id=index;
    out.uv={u,v};
    out.direction=glm::normalize(glm::vec3(st*std::cos(phi),std::cos(theta),st*std::sin(phi)));
    return out;
}

inline bool containsDirection(glm::vec3 d,float featherRadians=0.0f) {
    const float l2=glm::dot(d,d);
    if(l2<1e-12f) return false;
    d/=std::sqrt(l2);
    const float phi=std::atan2(d.z,d.x);
    const float theta=std::acos(std::clamp(d.y,-1.0f,1.0f));
    return theta>=0.04f && theta<=thetaMax(phi)+featherRadians;
}

} // namespace vh::scalp
