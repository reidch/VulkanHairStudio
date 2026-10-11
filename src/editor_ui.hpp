#pragma once

#include "character_surface.hpp"
#include "groom_style.hpp"
#include "groom_asset.hpp"
#include "groom_binding.hpp"
#include "groom_contact.hpp"
#include "hair_dynamics.hpp"
#include "wind_field.hpp"
#include <glm/glm.hpp>
#include <array>
#include <string>
#include <vector>

namespace vh::editor {

enum class StudioMode : uint8_t { Display, Edit, Physics };
enum class ScreenshotFormat : uint8_t { Png=0, Bmp=1, Tga=2 };
enum class EnvironmentPreset : uint8_t { StudioNeutral=0, StudioWarmCool=1, OutdoorHighContrast=2, CinematicDawn=3, SoftMoody=4 };

struct Actions {
    bool groomDirty = false;
    bool physicsDirty = false;
    bool lookDirty = false;
    bool resetPhysics = false;
    bool resolveContacts = false;
    bool requestScreenshot = false;
    bool screenshotIncludeUi = true;
    ScreenshotFormat screenshotFormat = ScreenshotFormat::Png;
    bool modeChangeRequested = false;
    StudioMode requestedMode = StudioMode::Display;
    int requestedStyleIndex = -1;
};

struct StudioLighting {
    // Default portrait setup: one large, soft, slightly elevated key plus environment fill.
    // The second light is a low-energy genuine directional fill for visible hair-shape cues.
    float primaryIntensity = 2.00f;
    float secondaryIntensity = 0.24f;
    float environmentIrradiance = 0.42f;
    float exposureEV = -0.20f;
    float primaryAngularRadius = 0.16f;
    float secondaryAngularRadius = 0.20f;
    float backgroundGray = 0.18f;
    float primaryAzimuthDeg = -35.0f;
    float primaryElevationDeg = 30.0f;
    float secondaryAzimuthDeg = 135.0f;
    float secondaryElevationDeg = 28.0f;
    bool neutralHeadPresentation = true;
    float neutralHeadDomeStrength = 1.35f;
    EnvironmentPreset environmentPreset = EnvironmentPreset::StudioNeutral;
    bool useSkyboxBackground = true;
    bool useSkyboxLighting = true;
    float skyboxRotationDeg = 0.0f;
    float skyboxBackgroundStrength = 1.0f;
};

struct PerformanceStats {
    double cpuFrameMs = 0.0;
    double gpuFrameMs = 0.0;
    double gpuPhysicsMs = 0.0;
    double gpuOpticalMs = 0.0;
    double gpuSceneMs = 0.0;
    double groomBuildMs = 0.0;
    double opticalBuildMs = 0.0;
    uint32_t msaaSamples = 1u;
};

struct State {
    StudioMode mode = StudioMode::Display;
    int selectedBranch = 0;
    int selectedCurve = 0;
    int selectedKey = 0;
    int selectedStyleWarp = 0;
    bool showGuides = true;
    bool keepOutsideSkin = false;
    bool draggingMacroGuide = false;
    bool openHairstyleMenu = false;
    bool openScreenshotMenu = false;
    ScreenshotFormat screenshotFormat = ScreenshotFormat::Png;
    int hairstyleCategory = 0;
    std::array<char, 80> hairstyleSearch{};
    int hairDebugView = 0; // surface/material transport diagnostics; never affects simulation
};

void applyStudioTheme();
Actions draw(State& state,
             const vh::character::CharacterSurface& character,
             const vh::groom::GroomBinding& binding,
             const std::vector<vh::groom::StyleEntry>& catalog,
             size_t styleIndex,
             vh::groom::GroomStyle& style,
             vh::groom::GroomBuildParams& build,
             vh::physics::HairMaterial& material,
             vh::wind::Field& wind,
             StudioLighting& lighting,
             const vh::groom::ValidationReport& validation,
             bool validationPending,
             bool styleBuildPending,
             const PerformanceStats& perf,
             const std::string& assetError,
             const glm::mat4& viewProj,
             uint32_t viewportWidth,
             uint32_t viewportHeight);

} // namespace vh::editor
