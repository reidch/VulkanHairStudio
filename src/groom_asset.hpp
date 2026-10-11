#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "character_surface.hpp"

namespace vh::groom {

// Rendering and dynamics deliberately use different resolutions. The authored
// representation remains compact; dense fibers are a derived runtime asset.
constexpr uint32_t kGuidePoints = 20u;
constexpr uint32_t kRenderCurvePoints = 33u; // maximum visual LOD; runtime also uses 17/9
constexpr uint32_t kSimulationGuideCount = 1600u;
constexpr uint32_t kRenderStrandCount = 110000u;
constexpr uint32_t kClusterSize = 64u;
constexpr uint32_t kOpticalGridDim = 64u;
constexpr uint32_t kCutSurfaceResolution = 4u;

using SweepKeyId = uint16_t;
using StyleCurveId = uint16_t;
using TopologyId = uint16_t;

struct GroomBuildParams {
    float follicleClearance = 0.0012f;
    float minStrandClearance = 0.0015f;
    float fiberDiameter = 0.000070f; // 70 micrometres
    float collisionMargin = 0.00035f;
    float globalLengthScale = 1.0f;
    float globalVolumeScale = 1.0f;
    // Global scale for residual strand-level frizz/flyaway detail. Mesoscopic
    // clump/style structure is unaffected. Values below 1 improve coherent flow.
    float microDetailScale = 0.65f;
    uint32_t seed = 3201u;
};

// A style-volume cross-section. centerD gives the mass-flow centerline and
// axisU/axisV give the actual low-frequency hairstyle cross-section. These are
// design-space quantities, not per-strand offsets.
struct SweepKey {
    glm::vec3 centerD{0.0f};
    glm::vec3 axisUD{0.08f,0.0f,0.0f};
    glm::vec3 axisVD{0.0f,0.08f,0.0f};
    float attachment = 0.0f;
};

struct StyleCurve {
    TopologyId topology = 0u;
    std::vector<SweepKeyId> keys;
};

// A continuous selector on the canonical scalp. Root patches define coherent
// hairstyle masses without baking semantic labels into the runtime.
struct RootPatch {
    std::string name;
    glm::vec3 centerD{0.0f,1.0f,0.0f};
    glm::vec3 invRadius{2.0f};
    glm::vec2 materialExtent{0.60f};
    float bias = 0.0f;
};

// One mesoscopic clump level. The procedural sites are jittered Voronoi-like
// sites in branch material space; crossover softly blends neighboring cells so
// the hierarchy has no hard rectangular bundle seams.
struct ClumpLevel {
    float cellsU = 4.0f;
    float cellsV = 4.0f;
    float siteJitter = 0.22f;
    float rootTightness = 0.04f;
    float midTightness = 0.42f;
    float tipTightness = 0.24f;
    float crossover = 0.16f;
    float drift = 0.05f;
    float waveAmplitude = 0.0f; // dimensionless material-space displacement
    float waveCycles = 0.0f;
};

// Primary clumps shape large locks; secondary clumps organize finer groups.
struct ClumpHierarchy {
    ClumpLevel primary{4.0f,4.0f,0.20f,0.02f,0.34f,0.18f,0.14f,0.055f,0.0f,0.0f};
    ClumpLevel secondary{10.0f,10.0f,0.26f,0.01f,0.18f,0.10f,0.20f,0.028f,0.0f,0.0f};
    float tipSpread = 0.08f;
    float twistTurns = 0.0f;
};

// A smooth cubic Bezier haircut surface over the branch material domain. The
// 4x4 control lattice explicitly represents layered termination instead of
// hiding a haircut inside unrelated local offsets.
struct CutSurface {
    std::array<float,kCutSurfaceResolution*kCutSurfaceResolution> controls{
        1.0f,1.0f,1.0f,1.0f,
        1.0f,1.0f,1.0f,1.0f,
        1.0f,1.0f,1.0f,1.0f,
        1.0f,1.0f,1.0f,1.0f
    };
    float breakup = 0.015f;
    float minLength = 0.58f;
};

// Optional local deformation for asymmetric or editorial styles.
struct StyleWarp {
    glm::vec2 center{0.0f};
    glm::vec2 radius{0.45f};
    glm::vec2 midDrift{0.0f};
    glm::vec2 tipDrift{0.0f};
    float lift = 0.0f;
    float rollTurns = 0.0f;
    float lengthScale = 1.0f;
    float weight = 1.0f;
};

enum class HairTexture : uint8_t { Straight, Wavy, Curly, Coily };

// Fine fiber texture only. Large blowout/wave motion belongs to ClumpLevel so
// strands in one lock move coherently before residual fiber detail is applied.
struct TextureProfile {
    HairTexture type = HairTexture::Straight;
    float amplitude = 0.0f;
    float cycles = 1.0f;
    float start = 0.10f;
    float irregularity = 0.08f;
    float phaseCoherence = 0.78f;
    float secondary = 0.18f;
};

struct DetailProfile {
    float frizzAmplitude = 0.00025f;
    float frizzFrequency = 5.0f;
    float frizzFraction = 0.22f;
    float flyawayAmplitude = 0.0020f;
    float flyawayFraction = 0.018f;
};

// One branch is one coherent hairstyle mass. Macro silhouette comes from the
// swept style volume, mesoscopic organization from clumps, termination from
// the cut surface, and only high-frequency residual structure from texture.
struct GroomBranch {
    StyleCurveId curve = 0u;
    TopologyId topology = 0u;
    RootPatch root{};
    glm::vec2 materialScale{0.72f,0.72f};
    glm::vec2 materialOffset{0.0f};
    ClumpHierarchy clumps{};
    CutSurface cut{};
    std::vector<StyleWarp> styleWarps{};
    TextureProfile texture{};
    DetailProfile detail{};
};

struct GroomTopologyGraph {
    std::vector<SweepKey> sweepKeys;
    std::vector<StyleCurve> curves;
    std::vector<GroomBranch> branches;
};

struct FieldControl {
    glm::vec3 centerD{0.0f,1.0f,0.0f};
    float angularRadius = 0.50f; // radians
    float delta = 0.0f;
};

struct ScalarField {
    float base = 1.0f;
    float minValue = 0.0f;
    float maxValue = 2.0f;
    std::vector<FieldControl> controls;
};

// Scalp fields modulate the style volume, clumping, and fine texture.
struct ScalpFieldAtlas {
    ScalarField length{1.0f,0.30f,1.20f,{}};
    ScalarField density{1.0f,0.0f,1.0f,{}};
    ScalarField volume{1.0f,0.35f,1.70f,{}};
    ScalarField textureAmplitude{1.0f,0.0f,2.0f,{}};
    ScalarField textureFrequency{1.0f,0.25f,2.5f,{}};
    ScalarField detail{1.0f,0.0f,2.0f,{}};
};

struct PartCurve {
    std::vector<glm::vec3> pointsD;
    float flowStrength = 0.0f;
    float influenceRadius = 0.32f; // radians
    float gapRadius = 0.010f;      // radians
};

enum class HairColorModel : uint8_t { DirectColor = 0u, NaturalMelanin = 1u };

struct HairOptics {
    HairColorModel colorModel = HairColorModel::NaturalMelanin;
    glm::vec3 directColor{0.20f,0.055f,0.018f};
    float melanin = 5.00f;
    float melaninRedness = 0.06f;
    float longitudinalRoughness = 0.12f;
    float azimuthalRoughness = 0.28f;
    float ior = 1.55f;
    float cuticleTilt = 0.045f;
    float rootDiameterScale = 1.0f;
    float tipDiameterScale = 0.10f;
    float diameterVariation = 0.14f;
    float tipLightening = 0.08f;
    float volumeScatter = 0.12f; // TT/TRT mixture share (0..0.35); surface BRDF dominates
};

struct GroomStyle {
    std::string slug;
    std::string name;
    GroomTopologyGraph graph;
    ScalpFieldAtlas fields{};
    PartCurve part{};
    HairOptics optics{};
};

struct alignas(16) StrandPointGpu { glm::vec4 position{0.0f}; };

struct SimulationGuideCpu {
    TopologyId topology = 0u;
    // Primary clump identity doubles as the coherent contact-correction group.
    uint64_t contactGroupTag = 0u;
    vh::character::RootBinding rootBinding{};
    glm::vec3 root{0.0f};
    glm::vec3 rootNormal{0.0f,1.0f,0.0f};
    glm::vec3 canonicalDirection{0.0f,1.0f,0.0f};
    std::array<glm::vec3,kGuidePoints> p{};
    std::array<float,kGuidePoints> pathU{};
    std::array<float,kGuidePoints> crossSectionScale{};
    float authoredDensity = 1.0f;
};

// fiber.x = tip diameter scale
// fiber.y = authored local density/visibility in [0,1]
// fiber.zw reserved for render-only quality metadata.
struct alignas(16) StrandDescGpu {
    glm::vec4 rootNormalDiameter{0.0f,1.0f,0.0f,0.000070f};
    glm::uvec4 guides{0u};
    glm::vec4 weights{1.0f,0.0f,0.0f,0.0f};
    glm::vec4 fiber{0.10f,1.0f,0.0f,0.0f};
};

struct alignas(16) GuideFrameGpu { glm::vec4 normal{0.0f,1.0f,0.0f,0.0f}; };
struct alignas(16) GuidePointMetaGpu { glm::vec4 supportVolume{0.0f}; };
struct alignas(16) HairClusterGpu { glm::uvec4 range{0u}; glm::vec4 sphere{0.0f}; };

static_assert(sizeof(StrandPointGpu)==16u);
static_assert(sizeof(StrandDescGpu)==64u);
static_assert(sizeof(GuideFrameGpu)==16u);
static_assert(sizeof(GuidePointMetaGpu)==16u);
static_assert(sizeof(HairClusterGpu)==32u);

struct GroomAudit {
    uint32_t unassignedFollicles = 0u;
    uint32_t crossTopologyBindings = 0u;
    uint32_t crossPrimaryClumpBindings = 0u;
    uint32_t constrainedGuideSamples = 0u;
    uint32_t smallestTopologyPopulation = 0u;
    uint32_t activeContactGroups = 0u;
    float maxRootGuideDistance = 0.0f;
    float restReconstructionError = 0.0f;
    float minGuideSupportRadius = 1.0f;
    float maxGuideSupportRadius = 0.0f;
    float minCrossSectionScale = 1.0f;
    float maxCrossSectionScale = 0.0f;
    float minDensity = 1.0f;
    float maxDensity = 0.0f;
    uint32_t retargetFallbackRoots = 0u;
    float retargetFallbackFraction = 0.0f;
};

struct GroomAsset {
    std::vector<StrandPointGpu> baseStrandPoints;
    std::vector<StrandPointGpu> restStrandPoints;
    std::vector<float> strandPathU;
    std::vector<uint64_t> strandContactGroupTags;
    std::vector<TopologyId> strandTopology;
    std::vector<uint32_t> guideSourceStrands;
    std::vector<SimulationGuideCpu> simulationGuides;
    std::vector<StrandPointGpu> restGuidePoints;
    std::vector<GuideFrameGpu> restGuideFrames;
    std::vector<GuidePointMetaGpu> guidePointMeta;
    std::vector<StrandDescGpu> strands;
    std::vector<HairClusterGpu> clusters;
    GroomAudit audit{};
};

} // namespace vh::groom
