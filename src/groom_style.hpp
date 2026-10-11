#pragma once

#include "groom_asset.hpp"
#include "scalp_domain.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vh::groom {

struct FieldSample {
    float length = 1.0f;
    float density = 1.0f;
    float volume = 1.0f;
    float textureAmplitude = 1.0f;
    float textureFrequency = 1.0f;
    float detail = 1.0f;
};

struct PartSample {
    glm::vec3 flowD{0.0f}; // canonical tangent, points away from the part
    float flowWeight = 0.0f;
    float densityScale = 1.0f;
};

float sampleField(const ScalarField& field, glm::vec3 canonicalDirection);
FieldSample sampleFields(const ScalpFieldAtlas& fields, glm::vec3 canonicalDirection);
PartSample samplePart(const PartCurve& part, glm::vec3 canonicalDirection);
float rootPatchScore(const RootPatch& patch, const vh::scalp::Sample& root);
glm::vec2 rootPatchCoordinate(const RootPatch& patch, const vh::scalp::Sample& root);

struct StyleEntry {
    std::string slug;
    std::string name;
    std::filesystem::path file;
};

std::vector<StyleEntry> discoverStyleAssets(const std::filesystem::path& directory);
GroomStyle loadGroomStyle(const std::filesystem::path& file);
std::optional<size_t> findStyle(const std::vector<StyleEntry>& entries, std::string_view slugOrName);
size_t wrapStyleIndex(int index, size_t count);

} // namespace vh::groom
