#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "character_surface.hpp"
#include "groom_asset.hpp"
#include "groom_builder.hpp"
#include "groom_cache.hpp"
#include "groom_contact.hpp"
#include "groom_style.hpp"
#include "hair_dynamics.hpp"
#include "wind_field.hpp"
#include "editor_ui.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <future>
#include <mutex>
#include <atomic>
#include <memory>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {

constexpr uint32_t kWindowWidth = 1280;
constexpr uint32_t kWindowHeight = 900;
constexpr float kPi = 3.14159265358979323846f;

void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with VkResult=" + std::to_string(int(r)));
    }
}

std::vector<char> readBinary(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open shader: " + path.string());
    const auto size = static_cast<size_t>(f.tellg());
    std::vector<char> data(size);
    f.seekg(0);
    f.read(data.data(), static_cast<std::streamsize>(size));
    return data;
}

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapSupport {
    VkSurfaceCapabilitiesKHR caps{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

using Vertex = vh::character::Vertex;

namespace {
glm::vec3 azimuthElevationDirection(float azimuthDeg,float elevationDeg) {
    const float az=glm::radians(azimuthDeg);
    const float el=glm::radians(elevationDeg);
    const float ce=std::cos(el);
    return glm::normalize(glm::vec3(ce*std::sin(az),std::sin(el),ce*std::cos(az)));
}
}

struct alignas(16) FrameGpu {
    glm::mat4 viewProj{1.0f};
    glm::vec4 cameraPos{0.0f};
    glm::vec4 frustumPlanes[6]{};
    glm::vec4 screenParams{0.0f}; // width, height, tanHalfFovY, MSAA samples
};

struct alignas(16) HairOpticsGpu {
    glm::vec4 colorMelanin{0.20f,0.055f,0.018f,5.00f};
    glm::vec4 pigmentRoughness{0.18f,0.0f,0.12f,0.28f};
    glm::vec4 fiberOptics{1.55f,0.045f,0.0f,1.0f};
    glm::vec4 ambientParams{0.42f,-0.20f,4.0f,0.0f}; // radiance, exposure EV, active env lobes, diagnostic mode
    glm::vec4 opticalGrid{0.0f,0.0f,0.0f,0.01f};
    glm::vec4 renderParams{64.0f,0.0f,5.0f,4096.0f}; // dim, natural flag, direction count, fixed scale
    glm::vec4 fiberShape{0.000070f,0.10f,1.0f,0.0f}; // root diameter, tip scale, optical support scale
    glm::vec4 primaryDirI{-0.34f,0.72f,0.60f,2.00f};
    glm::vec4 primaryColorShadow{1.00f,1.00f,1.00f,1.00f};
    glm::vec4 envDir0I{ 0.05f, 0.96f, 0.28f,0.38f};
    glm::vec4 envDir1I{ 0.72f, 0.42f, 0.55f,0.24f};
    glm::vec4 envDir2I{-0.74f, 0.36f, 0.57f,0.22f};
    glm::vec4 envDir3I{ 0.00f, 0.18f,-0.98f,0.18f};
    glm::vec4 envColor0{1.00f,0.96f,0.90f,0.0f};
    glm::vec4 envColor1{0.94f,0.98f,1.00f,0.0f};
    glm::vec4 envColor2{0.84f,0.91f,1.00f,0.0f};
    glm::vec4 envColor3{0.72f,0.82f,1.00f,0.0f};
    glm::vec4 secondaryDirI{0.70f,0.48f,-0.52f,0.24f};
    glm::vec4 secondaryColorShadow{1.00f,1.00f,1.00f,1.00f};
    glm::vec4 domParams{256.0f,4.0f,4096.0f,0.0f};
    glm::vec4 lightingParams{0.16f,0.11f,12.0f,0.12f};
    glm::vec4 fiberVariation{0.08f,0.0f,0.0f,0.0f};
    glm::vec4 headParams{1.0f,1.35f,0.0f,0.0f}; // x neutral presentation mode, y isotropic dome strength
    glm::vec4 skyTopColor{0.18f,0.19f,0.22f,1.0f};
    glm::vec4 skyHorizonColor{0.40f,0.40f,0.42f,1.0f};
    glm::vec4 skyBottomColor{0.10f,0.10f,0.11f,1.0f};
    glm::vec4 skyBlob0{0.32f,0.30f,0.22f,1.0f};
    glm::vec4 skyBlob1{0.76f,0.35f,0.28f,0.55f};
    glm::vec4 skyBlob0Color{1.0f,0.98f,0.95f,1.0f};
    glm::vec4 skyBlob1Color{0.92f,0.96f,1.0f,1.0f};
    glm::vec4 skyParams{0.40f,1.0f,0.0f,0.0f};
};

struct alignas(16) SimGpu {
    glm::mat4 headFromRest{1.0f};
    glm::mat4 restFromHead{1.0f};
    glm::vec4 dtSolver{1.0f / 120.0f, 2.0e-8f, 7.0e-6f, 0.0f};
    glm::vec4 wind{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 windPositionType{-0.35f,0.02f,0.20f,0.0f};
    glm::vec4 windShape{0.85f,0.82f,0.12f,0.10f};
    glm::vec4 forces{-9.81f, 1.35f, 0.64f, 0.00030f};
    glm::vec4 contact{0.55f, 0.34f, 0.0014f, 1.0f};
    glm::vec4 style{0.0085f, 4.5f, 9.0f, 1.2f};
    glm::vec4 aero{0.46f, 0.12f, 16.0f, 1.8f};
    glm::vec4 volume{0.58f, 0.24f, 0.55f, 0.85f};
    glm::vec4 grid{-0.42f, -0.56f, -0.42f, 0.018f};
    glm::vec4 bodyField{-0.42f,-0.56f,-0.36f,0.014f};
    glm::vec4 bodyInfo{2.0f,1.0f,0.0f,0.0f};
};
static_assert(sizeof(FrameGpu) % 16 == 0);
static_assert(sizeof(HairOpticsGpu) % 16 == 0);
static_assert(sizeof(SimGpu) % 16 == 0);

struct SkyPresetDef {
    const char* label;
    glm::vec3 topColor;
    glm::vec3 horizonColor;
    glm::vec3 bottomColor;
    glm::vec4 blob0;
    glm::vec4 blob1;
    glm::vec3 blob0Color;
    glm::vec3 blob1Color;
    glm::vec4 envDir[4];
    glm::vec3 envColor[4];
    float vignette;
};

glm::vec3 rotateY(const glm::vec3& v,float degrees) {
    const float a=glm::radians(degrees);
    const float c=std::cos(a), s=std::sin(a);
    return glm::vec3(c*v.x+s*v.z,v.y,-s*v.x+c*v.z);
}

bool nearlyEqual(float a,float b,float eps=1e-4f) {
    return std::abs(a-b)<=eps;
}

bool environmentTransportChanged(const vh::editor::StudioLighting& before,const vh::editor::StudioLighting& after) {
    return before.environmentPreset!=after.environmentPreset
        || before.useSkyboxLighting!=after.useSkyboxLighting
        || !nearlyEqual(before.skyboxRotationDeg,after.skyboxRotationDeg);
}

bool directionalShadowChanged(const vh::editor::StudioLighting& before,const vh::editor::StudioLighting& after) {
    return !nearlyEqual(before.primaryAzimuthDeg,after.primaryAzimuthDeg)
        || !nearlyEqual(before.primaryElevationDeg,after.primaryElevationDeg)
        || !nearlyEqual(before.secondaryAzimuthDeg,after.secondaryAzimuthDeg)
        || !nearlyEqual(before.secondaryElevationDeg,after.secondaryElevationDeg);
}

SkyPresetDef skyPreset(vh::editor::EnvironmentPreset preset) {
    using P=vh::editor::EnvironmentPreset;
    switch(preset) {
        case P::StudioWarmCool: return {
            "Brown Photostudio 02",
            {0.24f,0.22f,0.19f},{0.55f,0.46f,0.34f},{0.14f,0.12f,0.10f},
            {0.26f,0.34f,0.20f,1.10f},{0.78f,0.28f,0.24f,0.72f},
            {1.00f,0.89f,0.74f},{0.76f,0.86f,1.00f},
            {
                glm::vec4(glm::normalize(glm::vec3(-0.26f,0.92f,0.28f)),0.42f),
                glm::vec4(glm::normalize(glm::vec3(0.68f,0.48f,0.55f)),0.24f),
                glm::vec4(glm::normalize(glm::vec3(-0.62f,0.30f,0.72f)),0.20f),
                glm::vec4(glm::normalize(glm::vec3(0.00f,0.20f,-0.98f)),0.14f)
            },
            {{1.00f,0.90f,0.76f},{0.84f,0.90f,1.00f},{0.62f,0.54f,0.46f},{0.35f,0.36f,0.40f}},
            0.34f
        };
        case P::OutdoorHighContrast: return {
            "Kloofendal 48d",
            {0.24f,0.42f,0.74f},{0.82f,0.92f,1.00f},{0.56f,0.60f,0.66f},
            {0.74f,0.18f,0.13f,1.45f},{0.36f,0.46f,0.32f,0.35f},
            {1.00f,0.96f,0.86f},{0.84f,0.92f,1.00f},
            {
                glm::vec4(glm::normalize(glm::vec3(0.72f,0.64f,0.26f)),0.56f),
                glm::vec4(glm::normalize(glm::vec3(-0.38f,0.92f,0.08f)),0.26f),
                glm::vec4(glm::normalize(glm::vec3(-0.82f,0.26f,0.50f)),0.18f),
                glm::vec4(glm::normalize(glm::vec3(0.00f,0.08f,-1.0f)),0.12f)
            },
            {{1.00f,0.97f,0.88f},{0.72f,0.84f,1.00f},{0.50f,0.62f,0.84f},{0.26f,0.30f,0.36f}},
            0.26f
        };
        case P::CinematicDawn: return {
            "Kiara 1 Dawn",
            {0.28f,0.26f,0.40f},{0.98f,0.76f,0.68f},{0.26f,0.20f,0.24f},
            {0.68f,0.26f,0.22f,1.00f},{0.34f,0.30f,0.28f,0.45f},
            {1.00f,0.78f,0.70f},{0.74f,0.70f,0.98f},
            {
                glm::vec4(glm::normalize(glm::vec3(0.34f,0.58f,0.74f)),0.40f),
                glm::vec4(glm::normalize(glm::vec3(-0.46f,0.82f,0.34f)),0.28f),
                glm::vec4(glm::normalize(glm::vec3(0.08f,0.20f,-0.98f)),0.16f),
                glm::vec4(glm::normalize(glm::vec3(-0.74f,0.22f,0.62f)),0.12f)
            },
            {{1.00f,0.80f,0.70f},{0.72f,0.72f,0.98f},{0.44f,0.36f,0.44f},{0.28f,0.24f,0.34f}},
            0.40f
        };
        case P::SoftMoody: return {
            "Misty Pines",
            {0.28f,0.34f,0.36f},{0.62f,0.68f,0.66f},{0.18f,0.22f,0.22f},
            {0.50f,0.24f,0.30f,0.35f},{0.24f,0.32f,0.28f,0.18f},
            {0.88f,0.94f,0.92f},{0.68f,0.78f,0.74f},
            {
                glm::vec4(glm::normalize(glm::vec3(0.02f,0.98f,0.18f)),0.36f),
                glm::vec4(glm::normalize(glm::vec3(-0.62f,0.52f,0.58f)),0.22f),
                glm::vec4(glm::normalize(glm::vec3(0.72f,0.42f,0.55f)),0.18f),
                glm::vec4(glm::normalize(glm::vec3(0.00f,0.14f,-0.99f)),0.10f)
            },
            {{0.86f,0.92f,0.90f},{0.54f,0.64f,0.58f},{0.50f,0.58f,0.56f},{0.28f,0.32f,0.32f}},
            0.24f
        };
        default: return {
            "Studio Small 09",
            {0.16f,0.17f,0.18f},{0.44f,0.44f,0.46f},{0.10f,0.10f,0.11f},
            {0.28f,0.30f,0.18f,1.25f},{0.76f,0.28f,0.22f,0.85f},
            {1.00f,0.99f,0.96f},{0.92f,0.96f,1.00f},
            {
                glm::vec4(glm::normalize(glm::vec3(0.05f,0.96f,0.28f)),0.38f),
                glm::vec4(glm::normalize(glm::vec3(0.72f,0.42f,0.55f)),0.24f),
                glm::vec4(glm::normalize(glm::vec3(-0.74f,0.36f,0.57f)),0.22f),
                glm::vec4(glm::normalize(glm::vec3(0.00f,0.18f,-0.98f)),0.18f)
            },
            {{1.00f,0.96f,0.90f},{0.94f,0.98f,1.00f},{0.84f,0.91f,1.00f},{0.72f,0.82f,1.00f}},
            0.40f
        };
    }
}

class HairApp {
public:
    void run(int argc, char** argv) {
        exeDir_ = std::filesystem::absolute(argc > 0 ? argv[0] : "vulkan_hair_v47").parent_path();
        // Keep derived data outside build/ when running from a source tree so
        // a clean/reconfigure does not throw away expensive exact caches. A
        // packaged binary naturally falls back to an adjacent cache folder.
        runtimeCacheDir_ = exeDir_ / "cache";
        for(auto probe=exeDir_; !probe.empty(); probe=probe.parent_path()) {
            if(std::filesystem::exists(probe/"CMakeLists.txt") && std::filesystem::exists(probe/"assets"/"grooms")) {
                runtimeCacheDir_=probe/".vulkan_hair_cache"; break;
            }
            if(probe==probe.root_path()) break;
        }
        std::cout << "Derived-data cache: " << runtimeCacheDir_.string() << "\n";
        styleCatalog_ = vh::groom::discoverStyleAssets(exeDir_ / "assets" / "grooms");
        if(styleCatalog_.empty()) {
            const auto sourceFallback=std::filesystem::current_path()/"assets"/"grooms";
            styleCatalog_=vh::groom::discoverStyleAssets(sourceFallback);
        }
        if(styleCatalog_.empty()) throw std::runtime_error("No .json groom assets found in assets/grooms");
        std::string requestedStyle="butterfly-blowout";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--style=", 0) == 0) requestedStyle=arg.substr(8);
            else if (arg.rfind("--character=",0)==0) characterObjOverride_=arg.substr(12);
            else if (arg.rfind("--contact-field=",0)==0) characterFieldOverride_=arg.substr(16);
            else if (arg=="--validate-all-grooms") validateAllGroomsRequested_=true;
            else if (arg=="--quality=reference") referenceQuality_=true;
            else if (arg=="--quality=performance") performanceQuality_=true;
        }
        if(const auto i=vh::groom::findStyle(styleCatalog_,requestedStyle)) styleIndex_=*i;
        else throw std::runtime_error("Unknown hairstyle asset: "+requestedStyle);
        if(validateAllGroomsRequested_) { validateAllGrooms(); return; }
        startScenePreparation();
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window_ = nullptr;
    bool framebufferResized_ = false;
    bool dragging_ = false;
    bool resetRequested_ = false;
    bool volumeResetRequested_ = true;
    int styleIndexRequested_ = -1;
    bool resolveContactsRequested_ = false;
    bool validateAllGroomsRequested_ = false;
    bool referenceQuality_ = false;
    bool performanceQuality_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;

    float yaw_ = -0.38f;
    float pitch_ = 0.06f;
    float distance_ = 0.66f;
    vh::wind::Field wind_{};
    vh::physics::HairMaterial hairMaterial_{};
    vh::editor::State editorState_{};
    vh::editor::StudioLighting studioLighting_{};
    bool imguiInitialized_ = false;
    bool editedGroomRebuildRequested_ = false;
    vh::groom::ValidationReport validation_{};
    std::string assetError_{};
    glm::mat4 lastViewProj_{1.0f};
    double lastFrameTime_ = 0.0;
    float elapsed_ = 0.0f;
    float simAccumulator_ = 0.0f;
    uint32_t simStepsThisFrame_ = 0u;
    static constexpr float kFixedSimDt_ = 1.0f / 120.0f;

    std::filesystem::path exeDir_;
    std::filesystem::path runtimeCacheDir_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = 0;
    uint32_t presentFamily_ = 0;
    VkPhysicalDeviceProperties deviceProps_{};
    VkPhysicalDeviceMeshShaderPropertiesEXT meshProps_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    PFN_vkCmdDrawMeshTasksEXT cmdDrawMeshTasksEXT_ = nullptr;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapExtent_{};
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<bool> swapInitialized_;
    bool depthInitialized_ = false;
    bool msaaColorInitialized_ = false;
    bool sceneColorInitialized_ = false;
    bool compositeColorInitialized_ = false;
    bool hairGeometryInitialized_ = false;
    VkSampleCountFlagBits msaaSamples_ = VK_SAMPLE_COUNT_1_BIT;

    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat hdrFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkImage sceneColorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory sceneColorMemory_ = VK_NULL_HANDLE;
    VkImageView sceneColorView_ = VK_NULL_HANDLE;
    VkImage msaaColorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory msaaColorMemory_ = VK_NULL_HANDLE;
    VkImageView msaaColorView_ = VK_NULL_HANDLE;
    VkImage compositeColorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory compositeColorMemory_ = VK_NULL_HANDLE;
    VkImageView compositeColorView_ = VK_NULL_HANDLE;
    VkImage hairGeom0Image_ = VK_NULL_HANDLE;
    VkDeviceMemory hairGeom0Memory_ = VK_NULL_HANDLE;
    VkImageView hairGeom0View_ = VK_NULL_HANDLE;
    VkImage hairGeom1Image_ = VK_NULL_HANDLE;
    VkDeviceMemory hairGeom1Memory_ = VK_NULL_HANDLE;
    VkImageView hairGeom1View_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;

    static constexpr uint32_t kFramesInFlight_ = 2u;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::array<VkCommandBuffer,kFramesInFlight_> commandBuffers_{};
    std::array<VkSemaphore,kFramesInFlight_> imageAvailableSemaphores_{};
    std::array<VkSemaphore,kFramesInFlight_> renderFinishedSemaphores_{};
    std::array<VkFence,kFramesInFlight_> inFlightFences_{};
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE; // current-frame alias
    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;    // current-frame alias
    VkSemaphore renderFinished_ = VK_NULL_HANDLE;   // current-frame alias
    VkFence inFlightFence_ = VK_NULL_HANDLE;         // current-frame alias
    uint32_t frameIndex_ = 0u;
    VkQueryPool timestampQueryPool_ = VK_NULL_HANDLE;
    std::array<bool,kFramesInFlight_> timestampRecorded_{};
    VkDeviceSize frameStride_ = 0;
    VkDeviceSize simStride_ = 0;
    VkDeviceSize materialStride_ = 0;
    double cpuFrameMs_ = 0.0;
    double gpuFrameMs_ = 0.0;
    double gpuPhysicsMs_ = 0.0;
    double gpuOpticalMs_ = 0.0;
    double gpuSceneMs_ = 0.0;
    Buffer screenshotReadback_;
    VkDeviceSize screenshotReadbackBytes_ = 0;
    bool screenshotRequested_ = false;
    bool screenshotIncludeUi_ = true;
    vh::editor::ScreenshotFormat screenshotFormat_ = vh::editor::ScreenshotFormat::Png;
    bool screenshotWithUiDeferred_ = false;
    vh::editor::ScreenshotFormat deferredScreenshotFormat_ = vh::editor::ScreenshotFormat::Png;
    std::array<bool,kFramesInFlight_> screenshotReadbackPending_{};
    std::array<uint32_t,kFramesInFlight_> screenshotWidths_{};
    std::array<uint32_t,kFramesInFlight_> screenshotHeights_{};
    std::array<bool,kFramesInFlight_> screenshotSavedWithUi_{};
    std::array<vh::editor::ScreenshotFormat,kFramesInFlight_> screenshotFormats_{};
    std::array<std::filesystem::path,kFramesInFlight_> screenshotPaths_{};

    VkDescriptorSetLayout frameSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout staticSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dynamicsSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout toneMapSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hairComposeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet frameSet_ = VK_NULL_HANDLE;
    VkDescriptorSet staticSet_ = VK_NULL_HANDLE;
    VkDescriptorSet dynamicsSet_ = VK_NULL_HANDLE;
    VkDescriptorSet materialSet_ = VK_NULL_HANDLE;
    VkDescriptorSet toneMapSet_ = VK_NULL_HANDLE;
    VkDescriptorSet hairComposeSet_ = VK_NULL_HANDLE;

    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath_{};
    VkPipelineLayout headPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout hairPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout toneMapPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout hairComposePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline headPipeline_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;
    VkPipeline hairPipeline_ = VK_NULL_HANDLE;
    VkPipeline toneMapPipeline_ = VK_NULL_HANDLE;
    VkPipeline hairComposePipeline_ = VK_NULL_HANDLE;
    VkPipeline simPipeline_ = VK_NULL_HANDLE;
    VkPipeline volumeClearPipeline_ = VK_NULL_HANDLE;
    VkPipeline volumeP2GPipeline_ = VK_NULL_HANDLE;
    VkPipeline volumeSolvePipeline_ = VK_NULL_HANDLE;
    VkPipeline volumeG2PPipeline_ = VK_NULL_HANDLE;
    VkPipeline opticalDepthPipeline_ = VK_NULL_HANDLE;
    VkPipeline opticalSplatPipeline_ = VK_NULL_HANDLE;
    VkPipeline domSplatPipeline_ = VK_NULL_HANDLE;
    VkPipeline domPrefixPipeline_ = VK_NULL_HANDLE;
    VkSampler sceneSampler_ = VK_NULL_HANDLE;

    Buffer frameBuffer_;
    Buffer materialBuffer_;
    Buffer simBuffer_;
    Buffer restGuides_;
    Buffer currentGuides_;
    Buffer guideVelocities_;
    Buffer restGuideFrames_;
    Buffer currentGuideFrames_;
    Buffer bodyField_;
    Buffer opticalDensity_;
    Buffer opticalDepth_;
    Buffer primaryDeepOpacity_;
    Buffer secondaryDeepOpacity_;
    Buffer groomUploadStaging_;
    Buffer guidePointMeta_;
    Buffer volumeAccum_;
    Buffer volumeState_;
    Buffer restStrands_;
    Buffer strandDescs_;
    Buffer clusters_;
    Buffer headVertices_;
    Buffer headIndices_;
    uint32_t headIndexCount_ = 0;

    vh::character::CharacterSurface character_{};
    vh::groom::GroomBinding groomBinding_{};
    vh::groom::GroomBuildParams groomParams_{};
    std::vector<vh::groom::StyleEntry> styleCatalog_;
    size_t styleIndex_ = 0u;
    vh::groom::GroomStyle styleAsset_{};
    std::shared_ptr<vh::groom::GroomAsset> groom_{};
    std::filesystem::path characterObjOverride_{};
    std::filesystem::path characterFieldOverride_{};
    bool groomUploadPending_ = false;
    bool guideResetPending_ = false;
    bool opticalFieldDirty_ = true;
    bool opticalUpdateThisFrame_ = false;
    float opticalUpdateAccumulator_ = 0.0f;
    bool domFieldDirty_ = true;
    bool domUpdateThisFrame_ = false;
    float domUpdateAccumulator_ = 0.0f;
    static constexpr uint32_t kDomMaxDim_ = 384u;
    static constexpr uint32_t kDomLayers_ = 8u;
    struct PreparedSceneCpu {
        vh::character::CharacterSurface character{};
        vh::groom::GroomBinding binding{};
        double characterMs=0.0;
        double bindingMs=0.0;
        std::filesystem::path bindingCache{};
    };
    std::future<PreparedSceneCpu> scenePrepJob_{};

    std::atomic<bool> validationPending_{false};
    std::atomic<uint64_t> styleGeneration_{0u};
    uint64_t validatedGeneration_ = 0u;
    struct AsyncValidationResult { uint64_t generation=0u; vh::groom::ValidationReport report{}; };
    std::future<AsyncValidationResult> validationJob_{};
    std::shared_ptr<vh::groom::GroomAsset> queuedValidationAsset_{};
    uint64_t queuedValidationGeneration_=0u;
    struct AsyncStyleBuildResult {
        size_t index=0u; vh::groom::GroomStyle style{}; std::shared_ptr<vh::groom::GroomAsset> groom{};
        double buildMs=0.0; double cacheIoMs=0.0; bool cacheHit=false; bool styleSwitch=false; std::string error{};
    };
    std::atomic<bool> styleBuildPending_{false};
    std::mutex styleBuildMutex_;
    std::optional<AsyncStyleBuildResult> completedStyleBuild_;
    std::future<void> styleBuildJob_{};
    double lastGroomBuildMs_ = 0.0;
    double lastOpticalBuildMs_ = 0.0;
    glm::vec3 opticalOrigin_{0.0f};
    float opticalCellSize_ = 0.01f;

    static void framebufferResizeCallback(GLFWwindow* w, int, int) {
        auto* app = static_cast<HairApp*>(glfwGetWindowUserPointer(w));
        app->framebufferResized_ = true;
    }

    static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
        auto* app = static_cast<HairApp*>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
            // Always terminate camera orbit even when the cursor is released over a UI panel.
            app->dragging_ = false;
            return;
        }
        if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            app->dragging_ = true;
            glfwGetCursorPos(w, &app->lastMouseX_, &app->lastMouseY_);
        }
    }

    static void cursorCallback(GLFWwindow* w, double x, double y) {
        auto* app = static_cast<HairApp*>(glfwGetWindowUserPointer(w));
        if (app->editorState_.draggingMacroGuide ||
            (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) || !app->dragging_) return;
        const double dx = x - app->lastMouseX_;
        const double dy = y - app->lastMouseY_;
        app->lastMouseX_ = x;
        app->lastMouseY_ = y;
        app->yaw_ -= static_cast<float>(dx) * 0.005f;
        app->pitch_ = std::clamp(app->pitch_ - static_cast<float>(dy) * 0.004f, -0.75f, 0.75f);
    }

    static void scrollCallback(GLFWwindow* w, double, double yoff) {
        auto* app = static_cast<HairApp*>(glfwGetWindowUserPointer(w));
        if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
        app->distance_ = std::clamp(app->distance_ * std::pow(0.90f, static_cast<float>(yoff)), 0.42f, 1.35f);
    }

    static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        auto* app = static_cast<HairApp*>(glfwGetWindowUserPointer(w));
        if (key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(w, GLFW_TRUE); return; }
        if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard) return;
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) app->wind_.enabled = !app->wind_.enabled;
        if (key == GLFW_KEY_UP) app->wind_.speed = std::min(14.0f, app->wind_.speed + 0.5f);
        if (key == GLFW_KEY_DOWN) app->wind_.speed = std::max(0.0f, app->wind_.speed - 0.5f);
        if (key == GLFW_KEY_R && action == GLFW_PRESS) app->resetRequested_ = true;
        if (key == GLFW_KEY_1 && action == GLFW_PRESS) { app->yaw_ = 0.0f; app->pitch_ = 0.04f; app->distance_ = 0.66f; }
        if (key == GLFW_KEY_2 && action == GLFW_PRESS) { app->yaw_ = -0.58f; app->pitch_ = 0.08f; app->distance_ = 0.67f; }
        if (key == GLFW_KEY_3 && action == GLFW_PRESS) { app->yaw_ = -1.35f; app->pitch_ = 0.04f; app->distance_ = 0.68f; }
    }

    void validateAllGrooms() {
        groomParams_.fiberDiameter=0.000070f;
        groomParams_.collisionMargin=0.00035f;

        const std::filesystem::path defaultObj=exeDir_/"assets"/"character"/"ict_facekit"/"generic_neutral_mesh.obj";
        const std::filesystem::path defaultField=exeDir_/"assets"/"character"/"ict_facekit"/"ict_neutral_v37.contactbin";
        const auto objPath=characterObjOverride_.empty()?defaultObj:characterObjOverride_;
        const auto fieldPath=characterFieldOverride_.empty()
            ? (characterObjOverride_.empty()?defaultField:std::filesystem::path(objPath.string()+".v37.contactbin"))
            : characterFieldOverride_;

        auto character=vh::character::CharacterSurface::loadObj(objPath,fieldPath,0.0020f);
        auto binding=vh::groom::GroomRetargeter{}.bind(character,vh::groom::kRenderStrandCount,groomParams_.seed);
        const std::array<vh::groom::HairTexture,4> textures{
            vh::groom::HairTexture::Straight,vh::groom::HairTexture::Wavy,
            vh::groom::HairTexture::Curly,vh::groom::HairTexture::Coily};
        auto setTexture=[](vh::groom::GroomStyle& style,vh::groom::HairTexture type) {
            for(auto& branch:style.graph.branches) {
                auto& t=branch.texture; t.type=type;
                switch(type) {
                    case vh::groom::HairTexture::Straight: t.amplitude=0.0f;t.cycles=1.0f;t.start=.10f;t.irregularity=.04f;t.phaseCoherence=.90f;break;
                    case vh::groom::HairTexture::Wavy: t.amplitude=.0018f;t.cycles=1.65f;t.start=.10f;t.irregularity=.10f;t.phaseCoherence=.78f;break;
                    case vh::groom::HairTexture::Curly: t.amplitude=.0038f;t.cycles=3.0f;t.start=.08f;t.irregularity=.14f;t.phaseCoherence=.68f;break;
                    case vh::groom::HairTexture::Coily: t.amplitude=.0033f;t.cycles=5.0f;t.start=.06f;t.irregularity=.18f;t.phaseCoherence=.58f;break;
                }
            }
        };
        auto textureName=[](vh::groom::HairTexture type) {
            switch(type){case vh::groom::HairTexture::Straight:return "Straight";case vh::groom::HairTexture::Wavy:return "Wavy";case vh::groom::HairTexture::Curly:return "Curly";case vh::groom::HairTexture::Coily:return "Coily";} return "Unknown";
        };

        uint32_t failures=0,cases=0;
        for(const auto& entry:styleCatalog_) for(const auto texture:textures) {
            ++cases;
            try {
                auto style=vh::groom::loadGroomStyle(entry.file); setTexture(style,texture);
                auto groom=vh::groom::GroomBuilder{}.build(binding,groomParams_,style);
                const auto report=vh::groom::validateGroom(character,groom,groomParams_);
                std::cout<<(report.contactClean()?"[PASS] ":"[WARN] ")<<entry.slug<<" / "<<textureName(texture)
                         <<" visual="<<report.visualInsideSamples<<" base="<<report.baseInsideSamples
                         <<" proxy="<<report.guideProxyViolations<<" deepest="<<1000.0f*report.deepestPenetration<<" mm\n";
                
            } catch(const std::exception& e) {
                ++failures;
                std::cout<<"[FAIL] "<<entry.slug<<" / "<<textureName(texture)<<" structural error: "<<e.what()<<"\n";
            }
        }
        std::cout<<"V39 validation suite: "<<(cases-failures)<<"/"<<cases<<" passed.\n";
        if(failures!=0) throw std::runtime_error("Groom structural validation failed for "+std::to_string(failures)+" case(s)");
    }

    void initWindow() {
        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan Hair Studio", nullptr, nullptr);
        if (!window_) throw std::runtime_error("glfwCreateWindow failed");
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, cursorCallback);
        glfwSetScrollCallback(window_, scrollCallback);
        glfwSetKeyCallback(window_, keyCallback);
    }

    QueueFamilies findQueueFamilies(VkPhysicalDevice dev) const {
        QueueFamilies q;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());
        for (uint32_t i = 0; i < count; ++i) {
            if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) q.graphics = i;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
            if (present) q.present = i;
            if (q.complete()) break;
        }
        return q;
    }

    bool hasDeviceExtension(VkPhysicalDevice dev, const char* wanted) const {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> ext(count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, ext.data());
        for (const auto& e : ext) if (std::strcmp(e.extensionName, wanted) == 0) return true;
        return false;
    }

    SwapSupport querySwapSupport(VkPhysicalDevice dev) const {
        SwapSupport s;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface_, &s.caps);
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &n, nullptr);
        s.formats.resize(n);
        if (n) vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &n, s.formats.data());
        n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &n, nullptr);
        s.modes.resize(n);
        if (n) vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &n, s.modes.data());
        return s;
    }

    bool suitableDevice(VkPhysicalDevice dev) {
        VkPhysicalDeviceProperties basic{};
        vkGetPhysicalDeviceProperties(dev, &basic);
        if (VK_API_VERSION_MAJOR(basic.apiVersion) < 1 ||
            (VK_API_VERSION_MAJOR(basic.apiVersion) == 1 && VK_API_VERSION_MINOR(basic.apiVersion) < 3)) return false;
        const auto q = findQueueFamilies(dev);
        if (!q.complete()) return false;
        if (!hasDeviceExtension(dev, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) return false;
        if (!hasDeviceExtension(dev, VK_EXT_MESH_SHADER_EXTENSION_NAME)) return false;
        const auto swap = querySwapSupport(dev);
        if (swap.formats.empty() || swap.modes.empty()) return false;

        VkPhysicalDeviceMeshShaderFeaturesEXT meshFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13.pNext = &meshFeat;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &v13;
        vkGetPhysicalDeviceFeatures2(dev, &f2);
        if (!v13.dynamicRendering || !meshFeat.meshShader || !meshFeat.taskShader || !f2.features.sampleRateShading) return false;

        VkPhysicalDeviceMeshShaderPropertiesEXT mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &mp;
        vkGetPhysicalDeviceProperties2(dev, &p2);
        return mp.maxTaskWorkGroupInvocations >= 32 && mp.maxTaskWorkGroupSize[0] >= 32 &&
               mp.maxMeshWorkGroupInvocations >= 32 && mp.maxMeshWorkGroupSize[0] >= 32 &&
               mp.maxMeshOutputVertices >= 66 &&
               mp.maxMeshOutputPrimitives >= 64 &&
               mp.maxTaskWorkGroupCount[0] >= 1719 && mp.maxTaskWorkGroupTotalCount >= 1719 &&
               mp.maxMeshWorkGroupCount[0] >= vh::groom::kClusterSize &&
               mp.maxMeshWorkGroupTotalCount >= vh::groom::kClusterSize;
    }

    void createInstance() {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "Vulkan Hair Studio";
        ai.applicationVersion = VK_MAKE_VERSION(44, 0, 0);
        ai.pEngineName = "Vulkan Hair Studio";
        ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        ai.apiVersion = VK_API_VERSION_1_3;

        uint32_t extCount = 0;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&extCount);
        if (!glfwExt || extCount == 0) throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
        std::vector<const char*> extensions(glfwExt, glfwExt + extCount);

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &ai;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        vkCheck(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (!count) throw std::runtime_error("No Vulkan physical device found");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        int bestScore = std::numeric_limits<int>::min();
        for (auto dev : devices) {
            if (!suitableDevice(dev)) continue;
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(dev, &p);
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 10000 : 0;
            score += static_cast<int>(p.limits.maxImageDimension2D);
            if (score > bestScore) { bestScore = score; physicalDevice_ = dev; }
        }
        if (!physicalDevice_) {
            throw std::runtime_error(
                "No suitable Vulkan 1.3 GPU with VK_EXT_mesh_shader was found. "
                "Required: task+mesh shaders, dynamic rendering, >=66 mesh vertices and >=64 mesh primitives/workgroup.");
        }

        vkGetPhysicalDeviceProperties(physicalDevice_, &deviceProps_);
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &meshProps_;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &p2);
        const auto q = findQueueFamilies(physicalDevice_);
        graphicsFamily_ = *q.graphics;
        presentFamily_ = *q.present;

        // Select a filterable floating-point scene target; all hair lighting is
        // accumulated here before the display transform.
        bool hdrFound=false;
        for(VkFormat candidate : {VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R32G32B32A32_SFLOAT}) {
            VkFormatProperties fp{}; vkGetPhysicalDeviceFormatProperties(physicalDevice_,candidate,&fp);
            const VkFormatFeatureFlags required=VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
            if((fp.optimalTilingFeatures&required)==required) { hdrFormat_=candidate; hdrFound=true; break; }
        }
        if(!hdrFound) throw std::runtime_error("No floating-point HDR color attachment/sampled format is supported");
        VkImageFormatProperties hdrProps{};
        vkCheck(vkGetPhysicalDeviceImageFormatProperties(physicalDevice_,hdrFormat_,VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,
                                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,0,&hdrProps),
                "vkGetPhysicalDeviceImageFormatProperties(HDR)");
        VkSampleCountFlags supported = deviceProps_.limits.framebufferColorSampleCounts &
                                       deviceProps_.limits.framebufferDepthSampleCounts & hdrProps.sampleCounts;
        // Hair visibility resolves individual MSAA samples. Beauty/Reference use
        // 8x; Performance uses 4x. The G-buffer requires at least 2x MSAA.
        if (!performanceQuality_ && (supported & VK_SAMPLE_COUNT_8_BIT)) msaaSamples_ = VK_SAMPLE_COUNT_8_BIT;
        else if (performanceQuality_ && (supported & VK_SAMPLE_COUNT_4_BIT)) msaaSamples_ = VK_SAMPLE_COUNT_4_BIT;
        else if (supported & VK_SAMPLE_COUNT_4_BIT) msaaSamples_ = VK_SAMPLE_COUNT_4_BIT;
        else if (supported & VK_SAMPLE_COUNT_2_BIT) msaaSamples_ = VK_SAMPLE_COUNT_2_BIT;
        else throw std::runtime_error("Hair visibility requires at least 2x MSAA support for the HDR/depth formats");

        const char* qualityName=referenceQuality_?"Reference":(performanceQuality_?"Performance":"Beauty");
        const int envLobes=performanceQuality_?2:4;
        const int opticalHz=referenceQuality_?60:(performanceQuality_?20:30);
        std::cout << "Selected GPU: " << deviceProps_.deviceName << "\n"
                  << "Mesh limits: vertices=" << meshProps_.maxMeshOutputVertices
                  << " primitives=" << meshProps_.maxMeshOutputPrimitives
                  << " invocations=" << meshProps_.maxMeshWorkGroupInvocations
                  << " | quality=" << qualityName
                  << " | MSAA=" << static_cast<uint32_t>(msaaSamples_) << "x"
                  << " | env lobes=" << envLobes
                  << " | optical=" << opticalHz << " Hz\n";
    }

    void createDevice() {
        std::set<uint32_t> unique{graphicsFamily_, presentFamily_};
        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queues;
        for (uint32_t family : unique) {
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex = family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queues.push_back(qi);
        }

        VkPhysicalDeviceMeshShaderFeaturesEXT meshFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshFeat.taskShader = VK_TRUE;
        meshFeat.meshShader = VK_TRUE;
        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13.dynamicRendering = VK_TRUE;
        v13.pNext = &meshFeat;

        const std::array<const char*, 2> exts{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_MESH_SHADER_EXTENSION_NAME
        };
        VkPhysicalDeviceFeatures coreFeatures{};
        coreFeatures.sampleRateShading = VK_TRUE;
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.pNext = &v13;
        ci.pEnabledFeatures = &coreFeatures;
        ci.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
        ci.pQueueCreateInfos = queues.data();
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        vkCheck(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
        cmdDrawMeshTasksEXT_ = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(device_, "vkCmdDrawMeshTasksEXT"));
        if (!cmdDrawMeshTasksEXT_) throw std::runtime_error("vkCmdDrawMeshTasksEXT entry point is unavailable");
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
        }
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_R8G8B8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
        }
        return formats.front();
    }

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const {
        for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps) const {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) return caps.currentExtent;
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        VkExtent2D e{static_cast<uint32_t>(std::max(w, 1)), static_cast<uint32_t>(std::max(h, 1))};
        e.width = std::clamp(e.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        e.height = std::clamp(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        return e;
    }

    void createSwapchain() {
        const auto s = querySwapSupport(physicalDevice_);
        const auto format = chooseSurfaceFormat(s.formats);
        const auto presentMode = choosePresentMode(s.modes);
        const auto extent = chooseExtent(s.caps);
        uint32_t imageCount = s.caps.minImageCount + 1;
        if (s.caps.maxImageCount > 0) imageCount = std::min(imageCount, s.caps.maxImageCount);

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = imageCount;
        ci.imageFormat = format.format;
        ci.imageColorSpace = format.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        const uint32_t qIndices[] = {graphicsFamily_, presentFamily_};
        if (graphicsFamily_ != presentFamily_) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = qIndices;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        ci.preTransform = s.caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = presentMode;
        ci.clipped = VK_TRUE;
        vkCheck(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapImages_.data());
        swapFormat_ = format.format;
        swapExtent_ = extent;

        swapViews_.resize(swapImages_.size());
        swapInitialized_.assign(swapImages_.size(), false);
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = swapImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFormat_;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            vkCheck(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]), "vkCreateImageView(swap)");
        }
    }

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("No compatible Vulkan memory type");
    }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* initial = nullptr,
                        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCheck(vkCreateBuffer(device_, &bi, nullptr, &b.handle), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, b.handle, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, properties);
        vkCheck(vkAllocateMemory(device_, &ai, nullptr, &b.memory), "vkAllocateMemory(buffer)");
        vkCheck(vkBindBufferMemory(device_, b.handle, b.memory, 0), "vkBindBufferMemory");
        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            vkCheck(vkMapMemory(device_, b.memory, 0, size, 0, &b.mapped), "vkMapMemory");
            if (initial) std::memcpy(b.mapped, initial, static_cast<size_t>(size));
        } else if (initial) {
            throw std::runtime_error("Initial data supplied to non-host-visible buffer without staging");
        }
        return b;
    }

    VkDeviceSize alignUniformStride(VkDeviceSize size) const {
        const VkDeviceSize a=std::max<VkDeviceSize>(1u,deviceProps_.limits.minUniformBufferOffsetAlignment);
        return (size+a-1u)&~(a-1u);
    }
    uint32_t frameDynamicOffset() const { return uint32_t(frameStride_*frameIndex_); }
    uint32_t simDynamicOffset() const { return uint32_t(simStride_*frameIndex_); }
    uint32_t materialDynamicOffset() const { return uint32_t(materialStride_*frameIndex_); }

    void createSceneStaticGpuBuffers() {
        // Startup data is immutable. Upload all large static resources in one
        // transfer submission instead of serializing five vkQueueWaitIdle calls.
        struct Upload { Buffer* dst=nullptr; Buffer staging{}; VkDeviceSize size=0; };
        std::vector<Upload> uploads; uploads.reserve(3);
        auto addCopy=[&](Buffer& dst,VkDeviceSize size,VkBufferUsageFlags usage,const void* data) {
            Buffer staging=createBuffer(size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,data);
            dst=createBuffer(size,usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            uploads.push_back(Upload{&dst,staging,size});
        };

        const auto t0=std::chrono::steady_clock::now();
        const auto& field=character_.contactField().distance;
        addCopy(bodyField_,VkDeviceSize(field.size()*sizeof(float)),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,field.data());
        addCopy(headVertices_,VkDeviceSize(character_.vertices().size()*sizeof(Vertex)),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,character_.vertices().data());
        addCopy(headIndices_,VkDeviceSize(character_.indices().size()*sizeof(uint32_t)),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,character_.indices().data());

        volumeAccum_=createBuffer(VkDeviceSize(vh::physics::kVolumeGridCount)*sizeof(vh::physics::VolumeAccumGpu),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        volumeState_=createBuffer(VkDeviceSize(vh::physics::kVolumeGridCount)*sizeof(vh::physics::VolumeStateGpu),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=commandPool_; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer cmd=VK_NULL_HANDLE;
        vkCheck(vkAllocateCommandBuffers(device_,&ai,&cmd),"vkAllocateCommandBuffers(static upload)");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkCheck(vkBeginCommandBuffer(cmd,&bi),"vkBeginCommandBuffer(static upload)");
        for(const auto& u:uploads) {
            VkBufferCopy copy{0,0,u.size};
            vkCmdCopyBuffer(cmd,u.staging.handle,u.dst->handle,1,&copy);
        }
        vkCmdFillBuffer(cmd,volumeAccum_.handle,0,volumeAccum_.size,0u);
        vkCmdFillBuffer(cmd,volumeState_.handle,0,volumeState_.size,0u);
        vkCheck(vkEndCommandBuffer(cmd),"vkEndCommandBuffer(static upload)");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount=1; submit.pCommandBuffers=&cmd;
        vkCheck(vkQueueSubmit(graphicsQueue_,1,&submit,VK_NULL_HANDLE),"vkQueueSubmit(static upload)");
        vkCheck(vkQueueWaitIdle(graphicsQueue_),"vkQueueWaitIdle(static upload)");
        vkFreeCommandBuffers(device_,commandPool_,1,&cmd);
        for(auto& u:uploads) destroyBuffer(u.staging);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"Static GPU upload: one batch | "<<ms<<" ms\n";
    }

    void destroyBuffer(Buffer& b) {
        if (b.mapped) vkUnmapMemory(device_, b.memory);
        if (b.handle) vkDestroyBuffer(device_, b.handle, nullptr);
        if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
        b = {};
    }

    VkFormat findDepthFormat() const {
        for (VkFormat f : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            VkFormatProperties p{};
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, f, &p);
            if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
        }
        throw std::runtime_error("No supported depth format");
    }

    void createSceneColorResources() {
        sceneColorInitialized_ = false;
        msaaColorInitialized_ = false;
        compositeColorInitialized_ = false;
        hairGeometryInitialized_ = false;
        auto createColorImage=[&](VkSampleCountFlagBits samples,VkImageUsageFlags usage,
                                  VkImage& image,VkDeviceMemory& memory,VkImageView& view,const char* label) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType=VK_IMAGE_TYPE_2D;
            ii.extent={swapExtent_.width,swapExtent_.height,1};
            ii.mipLevels=1; ii.arrayLayers=1; ii.format=hdrFormat_;
            ii.tiling=VK_IMAGE_TILING_OPTIMAL; ii.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            ii.usage=usage; ii.samples=samples; ii.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
            vkCheck(vkCreateImage(device_,&ii,nullptr,&image),label);
            VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device_,image,&req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize=req.size; ai.memoryTypeIndex=findMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vkCheck(vkAllocateMemory(device_,&ai,nullptr,&memory),"vkAllocateMemory(hair image)");
            vkCheck(vkBindImageMemory(device_,image,memory,0),"vkBindImageMemory(hair image)");
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image=image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=hdrFormat_;
            vi.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; vi.subresourceRange.levelCount=1; vi.subresourceRange.layerCount=1;
            vkCheck(vkCreateImageView(device_,&vi,nullptr,&view),"vkCreateImageView(hair image)");
        };

        createColorImage(VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                         sceneColorImage_,sceneColorMemory_,sceneColorView_,"vkCreateImage(scene base HDR)");
        createColorImage(VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                         compositeColorImage_,compositeColorMemory_,compositeColorView_,"vkCreateImage(hair composite HDR)");
        createColorImage(msaaSamples_,VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         msaaColorImage_,msaaColorMemory_,msaaColorView_,"vkCreateImage(scene base MSAA)");
        // Multisample visibility images must remain sampleable after rasterization,
        // so they are not transient attachments.
        const VkImageUsageFlags visUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        createColorImage(msaaSamples_,visUsage,hairGeom0Image_,hairGeom0Memory_,hairGeom0View_,"vkCreateImage(hair geom0 MSAA)");
        createColorImage(msaaSamples_,visUsage,hairGeom1Image_,hairGeom1Memory_,hairGeom1View_,"vkCreateImage(hair geom1 MSAA)");
        if(!sceneSampler_) {
            VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            si.magFilter=VK_FILTER_LINEAR; si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
            si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=0.0f;
            vkCheck(vkCreateSampler(device_,&si,nullptr,&sceneSampler_),"vkCreateSampler(scene HDR)");
        }
    }


    void createDepthResources() {
        depthFormat_ = findDepthFormat();
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.extent = {swapExtent_.width, swapExtent_.height, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.format = depthFormat_;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ii.samples = msaaSamples_;
        ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCheck(vkCreateImage(device_, &ii, nullptr, &depthImage_), "vkCreateImage(depth)");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, depthImage_, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkCheck(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_), "vkAllocateMemory(depth)");
        vkCheck(vkBindImageMemory(device_, depthImage_, depthMemory_, 0), "vkBindImageMemory(depth)");

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = depthImage_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = depthFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCheck(vkCreateImageView(device_, &vi, nullptr, &depthView_), "vkCreateImageView(depth)");
    }

    void destroySwapchainResources() {
        if(headPipeline_) vkDestroyPipeline(device_,headPipeline_,nullptr);
        if(skyPipeline_) vkDestroyPipeline(device_,skyPipeline_,nullptr);
        if(hairPipeline_) vkDestroyPipeline(device_,hairPipeline_,nullptr);
        if(hairComposePipeline_) vkDestroyPipeline(device_,hairComposePipeline_,nullptr);
        if(toneMapPipeline_) vkDestroyPipeline(device_,toneMapPipeline_,nullptr);
        headPipeline_=skyPipeline_=hairPipeline_=hairComposePipeline_=toneMapPipeline_=VK_NULL_HANDLE;
        auto destroyImage=[&](VkImage& image,VkDeviceMemory& memory,VkImageView& view){
            if(view) vkDestroyImageView(device_,view,nullptr);
            if(image) vkDestroyImage(device_,image,nullptr);
            if(memory) vkFreeMemory(device_,memory,nullptr);
            view=VK_NULL_HANDLE; image=VK_NULL_HANDLE; memory=VK_NULL_HANDLE;
        };
        destroyImage(depthImage_,depthMemory_,depthView_); depthInitialized_=false;
        destroyImage(msaaColorImage_,msaaColorMemory_,msaaColorView_); msaaColorInitialized_=false;
        destroyImage(hairGeom0Image_,hairGeom0Memory_,hairGeom0View_);
        destroyImage(hairGeom1Image_,hairGeom1Memory_,hairGeom1View_); hairGeometryInitialized_=false;
        destroyImage(compositeColorImage_,compositeColorMemory_,compositeColorView_); compositeColorInitialized_=false;
        destroyImage(sceneColorImage_,sceneColorMemory_,sceneColorView_); sceneColorInitialized_=false;
        for(auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
        swapViews_.clear(); swapImages_.clear(); swapInitialized_.clear();
        if(swapchain_) vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        swapchain_=VK_NULL_HANDLE;
    }


    void recreateSwapchain() {
        int w = 0, h = 0;
        do {
            glfwGetFramebufferSize(window_, &w, &h);
            if (w == 0 || h == 0) glfwWaitEvents();
        } while (w == 0 || h == 0);
        vkDeviceWaitIdle(device_);
        destroySwapchainResources();
        createSwapchain();
        createSceneColorResources();
        createDepthResources();
        createGraphicsPipelines();
        if(toneMapSet_) updateToneMapDescriptor();
        if (imguiInitialized_) ImGui_ImplVulkan_SetMinImageCount(2);
        framebufferResized_ = false;
    }

    VkShaderModule createShader(const std::string& name) const {
        const auto bytes = readBinary(exeDir_ / "shaders" / name);
        if (bytes.size() % 4 != 0) throw std::runtime_error("SPIR-V byte size is not 4-byte aligned: " + name);
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
        VkShaderModule module = VK_NULL_HANDLE;
        vkCheck(vkCreateShaderModule(device_, &ci, nullptr, &module), "vkCreateShaderModule");
        return module;
    }

    void createDescriptorLayouts() {
        VkDescriptorSetLayoutBinding frame{};
        frame.binding=0; frame.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; frame.descriptorCount=1;
        frame.stageFlags=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT;
        VkDescriptorSetLayoutCreateInfo l0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l0.bindingCount=1; l0.pBindings=&frame;
        vkCheck(vkCreateDescriptorSetLayout(device_,&l0,nullptr,&frameSetLayout_),"frame set layout");

        std::array<VkDescriptorSetLayoutBinding,3> stat{};
        stat[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        stat[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_TASK_BIT_EXT,nullptr};
        stat[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        VkDescriptorSetLayoutCreateInfo l1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l1.bindingCount=uint32_t(stat.size()); l1.pBindings=stat.data();
        vkCheck(vkCreateDescriptorSetLayout(device_,&l1,nullptr,&staticSetLayout_),"static set layout");

        std::array<VkDescriptorSetLayoutBinding,10> dyn{};
        dyn[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        dyn[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        dyn[2]={2,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        dyn[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        dyn[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        dyn[5]={5,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        dyn[6]={6,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        dyn[7]={7,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        dyn[8]={8,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        dyn[9]={9,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_MESH_BIT_EXT,nullptr};
        VkDescriptorSetLayoutCreateInfo l2{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l2.bindingCount=uint32_t(dyn.size()); l2.pBindings=dyn.data();
        vkCheck(vkCreateDescriptorSetLayout(device_,&l2,nullptr,&dynamicsSetLayout_),"dynamics set layout");

        std::array<VkDescriptorSetLayoutBinding,5> material{};
        material[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        material[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        material[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        material[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        material[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo l3{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l3.bindingCount=uint32_t(material.size()); l3.pBindings=material.data();
        vkCheck(vkCreateDescriptorSetLayout(device_,&l3,nullptr,&materialSetLayout_),"material set layout");

        VkDescriptorSetLayoutBinding tone{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo lt{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; lt.bindingCount=1; lt.pBindings=&tone;
        vkCheck(vkCreateDescriptorSetLayout(device_,&lt,nullptr,&toneMapSetLayout_),"tone map set layout");

        std::array<VkDescriptorSetLayoutBinding,3> compose{};
        for(uint32_t i=0;i<3;++i) compose[i]={i,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo lc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; lc.bindingCount=uint32_t(compose.size()); lc.pBindings=compose.data();
        vkCheck(vkCreateDescriptorSetLayout(device_,&lc,nullptr,&hairComposeSetLayout_),"hair compose set layout");

        const std::array<VkDescriptorSetLayout,4> layouts{frameSetLayout_,staticSetLayout_,dynamicsSetLayout_,materialSetLayout_};
        VkPipelineLayoutCreateInfo headInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; headInfo.setLayoutCount=uint32_t(layouts.size()); headInfo.pSetLayouts=layouts.data();
        vkCheck(vkCreatePipelineLayout(device_,&headInfo,nullptr,&headPipelineLayout_),"head pipeline layout");
        VkPipelineLayoutCreateInfo hairInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; hairInfo.setLayoutCount=uint32_t(layouts.size()); hairInfo.pSetLayouts=layouts.data();
        vkCheck(vkCreatePipelineLayout(device_,&hairInfo,nullptr,&hairPipelineLayout_),"hair pipeline layout");

        const std::array<VkDescriptorSetLayout,4> composeLayouts{frameSetLayout_,hairComposeSetLayout_,dynamicsSetLayout_,materialSetLayout_};
        VkPipelineLayoutCreateInfo composeInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; composeInfo.setLayoutCount=uint32_t(composeLayouts.size()); composeInfo.pSetLayouts=composeLayouts.data();
        vkCheck(vkCreatePipelineLayout(device_,&composeInfo,nullptr,&hairComposePipelineLayout_),"hair compose pipeline layout");

        VkPushConstantRange postPush{}; postPush.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT; postPush.offset=0; postPush.size=sizeof(glm::vec4);
        VkPipelineLayoutCreateInfo toneInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; toneInfo.setLayoutCount=1; toneInfo.pSetLayouts=&toneMapSetLayout_; toneInfo.pushConstantRangeCount=1; toneInfo.pPushConstantRanges=&postPush;
        vkCheck(vkCreatePipelineLayout(device_,&toneInfo,nullptr,&toneMapPipelineLayout_),"tone map pipeline layout");
    }


    void createDescriptorPoolAndSets() {
        const std::array<VkDescriptorPoolSize,3> sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4}
        }};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets=6; pi.poolSizeCount=uint32_t(sizes.size()); pi.pPoolSizes=sizes.data();
        vkCheck(vkCreateDescriptorPool(device_,&pi,nullptr,&descriptorPool_),"vkCreateDescriptorPool");
        const std::array<VkDescriptorSetLayout,6> layouts{frameSetLayout_,staticSetLayout_,dynamicsSetLayout_,materialSetLayout_,hairComposeSetLayout_,toneMapSetLayout_};
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPool_; ai.descriptorSetCount=uint32_t(layouts.size()); ai.pSetLayouts=layouts.data();
        std::array<VkDescriptorSet,6> sets{}; vkCheck(vkAllocateDescriptorSets(device_,&ai,sets.data()),"vkAllocateDescriptorSets");
        frameSet_=sets[0]; staticSet_=sets[1]; dynamicsSet_=sets[2]; materialSet_=sets[3]; hairComposeSet_=sets[4]; toneMapSet_=sets[5];

        VkDescriptorBufferInfo frameInfo{frameBuffer_.handle,0,sizeof(FrameGpu)};
        VkDescriptorBufferInfo strandInfo{strandDescs_.handle,0,strandDescs_.size};
        VkDescriptorBufferInfo clusterInfo{clusters_.handle,0,clusters_.size};
        VkDescriptorBufferInfo restStrandInfo{restStrands_.handle,0,restStrands_.size};
        VkDescriptorBufferInfo currentInfo{currentGuides_.handle,0,currentGuides_.size};
        VkDescriptorBufferInfo restInfo{restGuides_.handle,0,restGuides_.size};
        VkDescriptorBufferInfo simInfo{simBuffer_.handle,0,sizeof(SimGpu)};
        VkDescriptorBufferInfo velInfo{guideVelocities_.handle,0,guideVelocities_.size};
        VkDescriptorBufferInfo fieldInfo{bodyField_.handle,0,bodyField_.size};
        VkDescriptorBufferInfo accumInfo{volumeAccum_.handle,0,volumeAccum_.size};
        VkDescriptorBufferInfo stateInfo{volumeState_.handle,0,volumeState_.size};
        VkDescriptorBufferInfo metaInfo{guidePointMeta_.handle,0,guidePointMeta_.size};
        VkDescriptorBufferInfo currentFrameInfo{currentGuideFrames_.handle,0,currentGuideFrames_.size};
        VkDescriptorBufferInfo restFrameInfo{restGuideFrames_.handle,0,restGuideFrames_.size};
        VkDescriptorBufferInfo materialInfo{materialBuffer_.handle,0,sizeof(HairOpticsGpu)};
        VkDescriptorBufferInfo opticalInfo{opticalDensity_.handle,0,opticalDensity_.size};
        VkDescriptorBufferInfo opticalDepthInfo{opticalDepth_.handle,0,opticalDepth_.size};
        VkDescriptorBufferInfo primaryDomInfo{primaryDeepOpacity_.handle,0,primaryDeepOpacity_.size};
        VkDescriptorBufferInfo secondaryDomInfo{secondaryDeepOpacity_.handle,0,secondaryDeepOpacity_.size};
        std::array<VkWriteDescriptorSet,19> writes{};
        auto fill=[&](size_t i,VkDescriptorSet set,uint32_t binding,VkDescriptorType type,VkDescriptorBufferInfo* info){
            writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[i].dstSet=set; writes[i].dstBinding=binding; writes[i].descriptorCount=1; writes[i].descriptorType=type; writes[i].pBufferInfo=info;
        };
        fill(0,frameSet_,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,&frameInfo);
        fill(1,staticSet_,0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&strandInfo); fill(2,staticSet_,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&clusterInfo); fill(3,staticSet_,2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&restStrandInfo);
        fill(4,dynamicsSet_,0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&currentInfo); fill(5,dynamicsSet_,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&restInfo); fill(6,dynamicsSet_,2,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,&simInfo);
        fill(7,dynamicsSet_,3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&velInfo); fill(8,dynamicsSet_,4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&fieldInfo); fill(9,dynamicsSet_,5,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&accumInfo);
        fill(10,dynamicsSet_,6,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&stateInfo); fill(11,dynamicsSet_,7,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&metaInfo); fill(12,dynamicsSet_,8,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&currentFrameInfo); fill(13,dynamicsSet_,9,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&restFrameInfo);
        fill(14,materialSet_,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,&materialInfo); fill(15,materialSet_,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&opticalInfo); fill(16,materialSet_,2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&opticalDepthInfo); fill(17,materialSet_,3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&primaryDomInfo); fill(18,materialSet_,4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&secondaryDomInfo);
        vkUpdateDescriptorSets(device_,uint32_t(writes.size()),writes.data(),0,nullptr);
        updateToneMapDescriptor();
    }


    void updateToneMapDescriptor() {
        if(!toneMapSet_||!hairComposeSet_||!sceneSampler_||!sceneColorView_||!compositeColorView_||!hairGeom0View_||!hairGeom1View_) return;
        VkDescriptorImageInfo toneImage{}; toneImage.sampler=sceneSampler_; toneImage.imageView=compositeColorView_; toneImage.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet toneWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; toneWrite.dstSet=toneMapSet_; toneWrite.dstBinding=0; toneWrite.descriptorCount=1; toneWrite.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; toneWrite.pImageInfo=&toneImage;
        vkUpdateDescriptorSets(device_,1,&toneWrite,0,nullptr);

        std::array<VkDescriptorImageInfo,3> images{};
        images[0]={sceneSampler_,sceneColorView_,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[1]={sceneSampler_,hairGeom0View_,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[2]={sceneSampler_,hairGeom1View_,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkWriteDescriptorSet,3> writes{};
        for(uint32_t i=0;i<3;++i){
            writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[i].dstSet=hairComposeSet_; writes[i].dstBinding=i; writes[i].descriptorCount=1; writes[i].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[i].pImageInfo=&images[i];
        }
        vkUpdateDescriptorSets(device_,uint32_t(writes.size()),writes.data(),0,nullptr);
    }


    void createPipelineCache() {
        // Vulkan pipeline compilation is pure derived data. Persist the driver cache
        // per GPU/driver so repeat launches do not pay the shader->pipeline cost again.
        const std::string key=std::to_string(deviceProps_.vendorID)+"_"+
                              std::to_string(deviceProps_.deviceID)+"_"+
                              std::to_string(deviceProps_.driverVersion);
        pipelineCachePath_=runtimeCacheDir_/"vulkan"/("pipelines_"+key+".bin");
        std::vector<char> initial;
        {
            std::ifstream f(pipelineCachePath_,std::ios::binary|std::ios::ate);
            if(f) {
                const std::streamoff n=static_cast<std::streamoff>(f.tellg());
                if(n>0 && n<std::streamoff(64*1024*1024)) {
                    initial.resize(size_t(n)); f.seekg(0); f.read(initial.data(),std::streamsize(initial.size()));
                    if(!f) initial.clear();
                }
            }
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize=initial.size();
        ci.pInitialData=initial.empty()?nullptr:initial.data();
        VkResult r=vkCreatePipelineCache(device_,&ci,nullptr,&pipelineCache_);
        if(r!=VK_SUCCESS && !initial.empty()) {
            // A stale/corrupt driver cache is never fatal: recreate from scratch.
            ci.initialDataSize=0; ci.pInitialData=nullptr;
            vkCheck(vkCreatePipelineCache(device_,&ci,nullptr,&pipelineCache_),"vkCreatePipelineCache");
        } else vkCheck(r,"vkCreatePipelineCache");
        std::cout<<"Pipeline cache: "<<(initial.empty()?"cold":"warm")<<" | "<<pipelineCachePath_.string()<<"\n";
    }

    void savePipelineCache() {
        if(!device_ || !pipelineCache_ || pipelineCachePath_.empty()) return;
        size_t size=0;
        if(vkGetPipelineCacheData(device_,pipelineCache_,&size,nullptr)!=VK_SUCCESS || size==0) return;
        std::vector<std::byte> data(size);
        if(vkGetPipelineCacheData(device_,pipelineCache_,&size,data.data())!=VK_SUCCESS || size==0) return;
        std::error_code ec; std::filesystem::create_directories(pipelineCachePath_.parent_path(),ec);
        std::ofstream f(pipelineCachePath_,std::ios::binary|std::ios::trunc);
        if(f) f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(size));
    }

    VkPipeline createComputePipelineFrom(const char* shaderName) {
        VkShaderModule comp = createShader(shaderName);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = comp;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = hairPipelineLayout_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCheck(vkCreateComputePipelines(device_, pipelineCache_, 1, &ci, nullptr, &pipeline), "vkCreateComputePipelines");
        vkDestroyShaderModule(device_, comp, nullptr);
        return pipeline;
    }

    void createComputePipelines() {
        simPipeline_ = createComputePipelineFrom("guide_sim.comp.spv");
        volumeClearPipeline_ = createComputePipelineFrom("volume_clear.comp.spv");
        volumeP2GPipeline_ = createComputePipelineFrom("volume_p2g.comp.spv");
        volumeSolvePipeline_ = createComputePipelineFrom("volume_solve.comp.spv");
        volumeG2PPipeline_ = createComputePipelineFrom("volume_g2p.comp.spv");
        opticalSplatPipeline_ = createComputePipelineFrom("optical_splat.comp.spv");
        opticalDepthPipeline_ = createComputePipelineFrom("optical_depth.comp.spv");
        domSplatPipeline_ = createComputePipelineFrom("dom_splat.comp.spv");
        domPrefixPipeline_ = createComputePipelineFrom("dom_prefix.comp.spv");
    }

    void createGraphicsPipelines() {
        const VkFormat sceneFormat=hdrFormat_;
        VkPipelineRenderingCreateInfo sceneRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        sceneRendering.colorAttachmentCount=1; sceneRendering.pColorAttachmentFormats=&sceneFormat; sceneRendering.depthAttachmentFormat=depthFormat_;
        const std::array<VkFormat,2> hairFormats{hdrFormat_,hdrFormat_};
        VkPipelineRenderingCreateInfo hairRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        hairRendering.colorAttachmentCount=uint32_t(hairFormats.size()); hairRendering.pColorAttachmentFormats=hairFormats.data(); hairRendering.depthAttachmentFormat=depthFormat_;

        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=1; viewport.scissorCount=1;
        const std::array<VkDynamicState,2> dynStates{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=uint32_t(dynStates.size()); dynamic.pDynamicStates=dynStates.data();
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1.0f; raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.cullMode=VK_CULL_MODE_BACK_BIT;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples=msaaSamples_;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable=VK_TRUE; depth.depthWriteEnable=VK_TRUE; depth.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState write{}; write.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo oneBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; oneBlend.attachmentCount=1; oneBlend.pAttachments=&write;

        // Base scene: sky + neutral mannequin, no hair color is written here.
        VkShaderModule headVert=createShader("head.vert.spv"),headFrag=createShader("head.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo,2> headStages{};
        headStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,headVert,"main",nullptr};
        headStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,headFrag,"main",nullptr};
        VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription,2> attrs{{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,uint32_t(offsetof(Vertex,position))},
            {1,0,VK_FORMAT_R32G32B32_SFLOAT,uint32_t(offsetof(Vertex,normal))}}};
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount=1; vertexInput.pVertexBindingDescriptions=&binding; vertexInput.vertexAttributeDescriptionCount=uint32_t(attrs.size()); vertexInput.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkGraphicsPipelineCreateInfo headInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        headInfo.pNext=&sceneRendering; headInfo.stageCount=uint32_t(headStages.size()); headInfo.pStages=headStages.data(); headInfo.pVertexInputState=&vertexInput; headInfo.pInputAssemblyState=&ia;
        headInfo.pViewportState=&viewport; headInfo.pRasterizationState=&raster; headInfo.pMultisampleState=&ms; headInfo.pDepthStencilState=&depth; headInfo.pColorBlendState=&oneBlend; headInfo.pDynamicState=&dynamic; headInfo.layout=headPipelineLayout_;
        vkCheck(vkCreateGraphicsPipelines(device_,pipelineCache_,1,&headInfo,nullptr,&headPipeline_),"head graphics pipeline");
        vkDestroyShaderModule(device_,headVert,nullptr); vkDestroyShaderModule(device_,headFrag,nullptr);

        VkShaderModule skyVert=createShader("sky.vert.spv"),skyFrag=createShader("sky.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo,2> skyStages{};
        skyStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,skyVert,"main",nullptr};
        skyStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,skyFrag,"main",nullptr};
        VkPipelineInputAssemblyStateCreateInfo skyIa{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; skyIa.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo skyRaster=raster; skyRaster.cullMode=VK_CULL_MODE_NONE;
        VkPipelineDepthStencilStateCreateInfo skyDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; skyDepth.depthTestEnable=VK_FALSE; skyDepth.depthWriteEnable=VK_FALSE;
        VkGraphicsPipelineCreateInfo skyInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        skyInfo.pNext=&sceneRendering; skyInfo.stageCount=uint32_t(skyStages.size()); skyInfo.pStages=skyStages.data(); skyInfo.pInputAssemblyState=&skyIa; skyInfo.pViewportState=&viewport; skyInfo.pRasterizationState=&skyRaster; skyInfo.pMultisampleState=&ms; skyInfo.pDepthStencilState=&skyDepth; skyInfo.pColorBlendState=&oneBlend; skyInfo.pDynamicState=&dynamic; skyInfo.layout=headPipelineLayout_;
        vkCheck(vkCreateGraphicsPipelines(device_,pipelineCache_,1,&skyInfo,nullptr,&skyPipeline_),"sky graphics pipeline");
        vkDestroyShaderModule(device_,skyVert,nullptr); vkDestroyShaderModule(device_,skyFrag,nullptr);

        // Primary visibility: physical-width fiber ribbons write geometry only.
        VkShaderModule task=createShader("hair.task.spv"),mesh=createShader("hair.mesh.spv"),frag=createShader("hair.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo,3> hairStages{};
        hairStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_TASK_BIT_EXT,task,"main",nullptr};
        hairStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_MESH_BIT_EXT,mesh,"main",nullptr};
        hairStages[2]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr};
        VkPipelineRasterizationStateCreateInfo hairRaster=raster; hairRaster.cullMode=VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo hairMs=ms; hairMs.sampleShadingEnable=VK_TRUE; hairMs.minSampleShading=1.0f;
        VkPipelineDepthStencilStateCreateInfo hairDepth=depth; hairDepth.depthWriteEnable=VK_TRUE;
        const std::array<VkPipelineColorBlendAttachmentState,2> visWrites{write,write};
        VkPipelineColorBlendStateCreateInfo visBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; visBlend.attachmentCount=uint32_t(visWrites.size()); visBlend.pAttachments=visWrites.data();
        VkGraphicsPipelineCreateInfo hairInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        hairInfo.pNext=&hairRendering; hairInfo.stageCount=uint32_t(hairStages.size()); hairInfo.pStages=hairStages.data(); hairInfo.pViewportState=&viewport; hairInfo.pRasterizationState=&hairRaster; hairInfo.pMultisampleState=&hairMs; hairInfo.pDepthStencilState=&hairDepth; hairInfo.pColorBlendState=&visBlend; hairInfo.pDynamicState=&dynamic; hairInfo.layout=hairPipelineLayout_;
        vkCheck(vkCreateGraphicsPipelines(device_,pipelineCache_,1,&hairInfo,nullptr,&hairPipeline_),"hair visibility pipeline");
        vkDestroyShaderModule(device_,task,nullptr); vkDestroyShaderModule(device_,mesh,nullptr); vkDestroyShaderModule(device_,frag,nullptr);

        // Sample-resolved physical shading/composition. This is the only pass that
        // evaluates the hair BCSDF and combines covered subpixel samples with scene radiance.
        VkPipelineRenderingCreateInfo composeRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; composeRendering.colorAttachmentCount=1; composeRendering.pColorAttachmentFormats=&sceneFormat;
        VkShaderModule fsVert=createShader("tonemap.vert.spv"),composeFrag=createShader("hair_compose.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo,2> composeStages{};
        composeStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,fsVert,"main",nullptr};
        composeStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,composeFrag,"main",nullptr};
        VkPipelineInputAssemblyStateCreateInfo fsIa{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; fsIa.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo fsRaster=raster; fsRaster.cullMode=VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo oneSample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; oneSample.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo noDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; noDepth.depthTestEnable=VK_FALSE; noDepth.depthWriteEnable=VK_FALSE;
        VkGraphicsPipelineCreateInfo composeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        composeInfo.pNext=&composeRendering; composeInfo.stageCount=uint32_t(composeStages.size()); composeInfo.pStages=composeStages.data(); composeInfo.pInputAssemblyState=&fsIa; composeInfo.pViewportState=&viewport; composeInfo.pRasterizationState=&fsRaster; composeInfo.pMultisampleState=&oneSample; composeInfo.pDepthStencilState=&noDepth; composeInfo.pColorBlendState=&oneBlend; composeInfo.pDynamicState=&dynamic; composeInfo.layout=hairComposePipelineLayout_;
        vkCheck(vkCreateGraphicsPipelines(device_,pipelineCache_,1,&composeInfo,nullptr,&hairComposePipeline_),"hair compose pipeline");
        vkDestroyShaderModule(device_,composeFrag,nullptr);

        // Display transform only: no hair-aware spatial blur in the post path.
        VkPipelineRenderingCreateInfo postRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; postRendering.colorAttachmentCount=1; postRendering.pColorAttachmentFormats=&swapFormat_;
        VkShaderModule postFrag=createShader("tonemap.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo,2> postStages{};
        postStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,fsVert,"main",nullptr};
        postStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,postFrag,"main",nullptr};
        VkGraphicsPipelineCreateInfo postInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        postInfo.pNext=&postRendering; postInfo.stageCount=uint32_t(postStages.size()); postInfo.pStages=postStages.data(); postInfo.pInputAssemblyState=&fsIa; postInfo.pViewportState=&viewport; postInfo.pRasterizationState=&fsRaster; postInfo.pMultisampleState=&oneSample; postInfo.pDepthStencilState=&noDepth; postInfo.pColorBlendState=&oneBlend; postInfo.pDynamicState=&dynamic; postInfo.layout=toneMapPipelineLayout_;
        vkCheck(vkCreateGraphicsPipelines(device_,pipelineCache_,1,&postInfo,nullptr,&toneMapPipeline_),"tone-map graphics pipeline");
        vkDestroyShaderModule(device_,postFrag,nullptr); vkDestroyShaderModule(device_,fsVert,nullptr);
    }


    void createCommandResources() {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = graphicsFamily_;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCheck(vkCreateCommandPool(device_, &pi, nullptr, &commandPool_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kFramesInFlight_;
        vkCheck(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()), "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for(uint32_t i=0;i<kFramesInFlight_;++i){
            vkCheck(vkCreateSemaphore(device_, &si, nullptr, &imageAvailableSemaphores_[i]), "create imageAvailable semaphore");
            vkCheck(vkCreateSemaphore(device_, &si, nullptr, &renderFinishedSemaphores_[i]), "create renderFinished semaphore");
            vkCheck(vkCreateFence(device_, &fi, nullptr, &inFlightFences_[i]), "create inFlight fence");
        }
        commandBuffer_=commandBuffers_[0];
        imageAvailable_=imageAvailableSemaphores_[0];
        renderFinished_=renderFinishedSemaphores_[0];
        inFlightFence_=inFlightFences_[0];

        if (deviceProps_.limits.timestampComputeAndGraphics) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = 5u*kFramesInFlight_;
            vkCheck(vkCreateQueryPool(device_, &qi, nullptr, &timestampQueryPool_), "create timestamp query pool");
        }
    }

    void readGpuTimestamps() {
        if (!timestampQueryPool_ || !timestampRecorded_[frameIndex_]) return;
        std::array<uint64_t,5> q{};
        const uint32_t base=5u*frameIndex_;
        const VkResult r=vkGetQueryPoolResults(device_,timestampQueryPool_,base,5,sizeof(q),q.data(),sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
        if(r!=VK_SUCCESS) return;
        const double msPerTick=double(deviceProps_.limits.timestampPeriod)*1e-6;
        gpuFrameMs_=double(q[4]-q[0])*msPerTick;
        gpuPhysicsMs_=double(q[1]-q[0])*msPerTick;
        gpuOpticalMs_=double(q[2]-q[1])*msPerTick;
        gpuSceneMs_=double(q[3]-q[2])*msPerTick;
        timestampRecorded_[frameIndex_]=false;
    }


    struct GroomUploadLayout {
        VkDeviceSize restGuides=0, restFrames=0, meta=0, restStrands=0, desc=0, clusters=0, total=0;
    };
    static VkDeviceSize alignUpload(VkDeviceSize x) { return (x+255u)&~VkDeviceSize(255u); }
    GroomUploadLayout groomUploadLayout() const {
        GroomUploadLayout l{}; VkDeviceSize o=0;
        auto take=[&](VkDeviceSize n){ VkDeviceSize at=o; o=alignUpload(o+n); return at; };
        l.restGuides=take(VkDeviceSize(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::StrandPointGpu));
        l.restFrames=take(VkDeviceSize(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::GuideFrameGpu));
        l.meta=take(VkDeviceSize(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::GuidePointMetaGpu));
        l.restStrands=take(VkDeviceSize(vh::groom::kRenderStrandCount)*vh::groom::kGuidePoints*sizeof(vh::groom::StrandPointGpu));
        l.desc=take(VkDeviceSize(vh::groom::kRenderStrandCount)*sizeof(vh::groom::StrandDescGpu));
        const uint32_t clusterCount=(vh::groom::kRenderStrandCount+vh::groom::kClusterSize-1u)/vh::groom::kClusterSize;
        l.clusters=take(VkDeviceSize(clusterCount)*sizeof(vh::groom::HairClusterGpu)); l.total=o; return l;
    }

    void updateOpticalGridFromGroom(const vh::groom::GroomAsset& groom) {
        const auto t0=std::chrono::steady_clock::now();
        constexpr uint32_t dim=vh::groom::kOpticalGridDim;
        if(groom.simulationGuides.empty() || groom.guidePointMeta.empty())
            throw std::runtime_error("Cannot size optical field from empty reduced groom");
        glm::vec3 lo(std::numeric_limits<float>::max()),hi(-std::numeric_limits<float>::max());
        for(uint32_t gid=0;gid<groom.simulationGuides.size();++gid) for(uint32_t p=0;p<vh::groom::kGuidePoints;++p) {
            const glm::vec3 x=groom.simulationGuides[gid].p[p];
            const float r=std::max(groom.guidePointMeta[size_t(gid)*vh::groom::kGuidePoints+p].supportVolume.x,0.003f);
            lo=glm::min(lo,x-glm::vec3(r)); hi=glm::max(hi,x+glm::vec3(r));
        }
        const glm::vec3 center=0.5f*(lo+hi);
        const float extent=std::max({hi.x-lo.x,hi.y-lo.y,hi.z-lo.z,0.20f})+0.10f;
        opticalCellSize_=extent/float(dim-3u);
        opticalOrigin_=center-glm::vec3(0.5f*opticalCellSize_*float(dim-1u));
        lastOpticalBuildMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"Optical bounds: "<<dim<<"^3, cell="<<1000.0f*opticalCellSize_
                 <<" mm | density generated dynamically on GPU\n";
        opticalFieldDirty_=true;
        domFieldDirty_=true;
    }

    void createGroomBuffers() {
        const GroomUploadLayout l=groomUploadLayout();
        const VkDeviceSize opticalVoxelBytes=VkDeviceSize(vh::groom::kOpticalGridDim)*vh::groom::kOpticalGridDim*vh::groom::kOpticalGridDim*sizeof(uint32_t);
        opticalDensity_=createBuffer(opticalVoxelBytes,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        opticalDepth_=createBuffer(VkDeviceSize(vh::groom::kOpticalGridDim)*vh::groom::kOpticalGridDim*vh::groom::kOpticalGridDim*5u*sizeof(float),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const VkDeviceSize domBytes=VkDeviceSize(kDomMaxDim_)*kDomMaxDim_*kDomLayers_*sizeof(uint32_t);
        primaryDeepOpacity_=createBuffer(domBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        secondaryDeepOpacity_=createBuffer(domBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        restGuides_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::StrandPointGpu),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        restGuideFrames_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::GuideFrameGpu),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        guidePointMeta_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::GuidePointMetaGpu),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        restStrands_=createBuffer(size_t(vh::groom::kRenderStrandCount)*vh::groom::kGuidePoints*sizeof(vh::groom::StrandPointGpu),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        strandDescs_=createBuffer(size_t(vh::groom::kRenderStrandCount)*sizeof(vh::groom::StrandDescGpu),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const uint32_t clusterCount=(vh::groom::kRenderStrandCount+vh::groom::kClusterSize-1u)/vh::groom::kClusterSize;
        clusters_=createBuffer(size_t(clusterCount)*sizeof(vh::groom::HairClusterGpu),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        currentGuides_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::StrandPointGpu),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        guideVelocities_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(glm::vec4),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        currentGuideFrames_=createBuffer(size_t(vh::groom::kSimulationGuideCount)*vh::groom::kGuidePoints*sizeof(vh::groom::GuideFrameGpu),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        groomUploadStaging_=createBuffer(l.total,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        stageGroomUpload();
        requestHairReset();
    }

    void stageGroomUpload() {
        if(!groom_) return;
        const auto& g=*groom_; const GroomUploadLayout l=groomUploadLayout();
        auto cp=[&](VkDeviceSize off,const void* data,size_t bytes){ std::memcpy(static_cast<std::byte*>(groomUploadStaging_.mapped)+off,data,bytes); };
        cp(l.restGuides,g.restGuidePoints.data(),g.restGuidePoints.size()*sizeof(vh::groom::StrandPointGpu));
        cp(l.restFrames,g.restGuideFrames.data(),g.restGuideFrames.size()*sizeof(vh::groom::GuideFrameGpu));
        cp(l.meta,g.guidePointMeta.data(),g.guidePointMeta.size()*sizeof(vh::groom::GuidePointMetaGpu));
        cp(l.restStrands,g.restStrandPoints.data(),g.restStrandPoints.size()*sizeof(vh::groom::StrandPointGpu));
        cp(l.desc,g.strands.data(),g.strands.size()*sizeof(vh::groom::StrandDescGpu));
        cp(l.clusters,g.clusters.data(),g.clusters.size()*sizeof(vh::groom::HairClusterGpu));
        groomUploadPending_=true; opticalFieldDirty_=true; domFieldDirty_=true;
    }

    void recordPendingGroomUpload(VkCommandBuffer cmd) {
        if(!groomUploadPending_) return;
        const GroomUploadLayout l=groomUploadLayout(); const auto& g=*groom_;
        auto copy=[&](VkDeviceSize src,Buffer& dst,VkDeviceSize bytes){ VkBufferCopy c{src,0,bytes}; vkCmdCopyBuffer(cmd,groomUploadStaging_.handle,dst.handle,1,&c); };
        copy(l.restGuides,restGuides_,g.restGuidePoints.size()*sizeof(vh::groom::StrandPointGpu));
        copy(l.restFrames,restGuideFrames_,g.restGuideFrames.size()*sizeof(vh::groom::GuideFrameGpu));
        copy(l.meta,guidePointMeta_,g.guidePointMeta.size()*sizeof(vh::groom::GuidePointMetaGpu));
        copy(l.restStrands,restStrands_,g.restStrandPoints.size()*sizeof(vh::groom::StrandPointGpu));
        copy(l.desc,strandDescs_,g.strands.size()*sizeof(vh::groom::StrandDescGpu));
        copy(l.clusters,clusters_,g.clusters.size()*sizeof(vh::groom::HairClusterGpu));
        std::array<VkBufferMemoryBarrier,6> bs{}; Buffer* targets[6]={&restGuides_,&restGuideFrames_,&guidePointMeta_,&restStrands_,&strandDescs_,&clusters_};
        for(size_t i=0;i<bs.size();++i){ bs[i]={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; bs[i].srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; bs[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT; bs[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bs[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bs[i].buffer=targets[i]->handle; bs[i].offset=0; bs[i].size=VK_WHOLE_SIZE; }
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT|VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,uint32_t(bs.size()),bs.data(),0,nullptr);
        groomUploadPending_=false;
    }

    void destroyGroomBuffers() {
        destroyBuffer(groomUploadStaging_); destroyBuffer(secondaryDeepOpacity_); destroyBuffer(primaryDeepOpacity_); destroyBuffer(opticalDepth_); destroyBuffer(opticalDensity_);
        destroyBuffer(clusters_); destroyBuffer(strandDescs_); destroyBuffer(restStrands_); destroyBuffer(guidePointMeta_);
        destroyBuffer(currentGuideFrames_); destroyBuffer(restGuideFrames_); destroyBuffer(guideVelocities_); destroyBuffer(currentGuides_); destroyBuffer(restGuides_);
    }

    void launchValidation(const std::shared_ptr<vh::groom::GroomAsset>& asset,uint64_t generation) {
        const auto params=groomParams_;
        validationPending_.store(true,std::memory_order_release);
        validationJob_=std::async(std::launch::async,[this,asset,generation,params]{
            return AsyncValidationResult{generation,vh::groom::validateGroom(character_,*asset,params)};
        });
    }

    void scheduleValidation(const std::shared_ptr<vh::groom::GroomAsset>& asset,uint64_t generation) {
        // Exact validation is intentionally single-flight. If the user edits
        // again while it is running, only the newest generation is retained.
        // Obsolete validation work never multiplies across CPU cores.
        if(validationJob_.valid()) {
            if(validationJob_.wait_for(std::chrono::seconds(0))!=std::future_status::ready) {
                queuedValidationAsset_=asset;
                queuedValidationGeneration_=generation;
                validationPending_.store(true,std::memory_order_release);
                return;
            }
            // A completed result belongs to an older generation once a new
            // commit arrives here; consume it without publishing stale state.
            try { (void)validationJob_.get(); } catch(...) {}
        }
        queuedValidationAsset_.reset(); queuedValidationGeneration_=0u;
        launchValidation(asset,generation);
    }

    void pollValidation() {
        if(!validationJob_.valid() || validationJob_.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return;
        std::optional<AsyncValidationResult> ready;
        try { ready=validationJob_.get(); }
        catch(const std::exception& e) {
            validationPending_.store(false,std::memory_order_release);
            assetError_=std::string("Validation failed: ")+e.what();
        }
        const uint64_t current=styleGeneration_.load(std::memory_order_acquire);
        if(ready && ready->generation==current) {
            validation_=ready->report;
            validatedGeneration_=ready->generation;
        }
        if(queuedValidationAsset_) {
            auto next=std::move(queuedValidationAsset_);
            const uint64_t gen=queuedValidationGeneration_;
            queuedValidationGeneration_=0u;
            launchValidation(next,gen);
        } else {
            validationPending_.store(false,std::memory_order_release);
        }
    }

    bool physicsAvailableNow() const {
        // Simulation availability depends only on a committed dynamical state.
        // Contact penetration is a solver/audit concern, never an admission gate.
        return static_cast<bool>(groom_);
    }

    void waitForFrameRingIdle() {
        // Groom staging is one large persistently mapped transfer buffer.  Style
        // commits are infrequent, so wait for both frame slots before overwriting
        // it rather than duplicating tens of MB of staging memory.  Normal camera/
        // physics frames never take this path.
        if(device_ && inFlightFences_[0] && inFlightFences_[1])
            vkCheck(vkWaitForFences(device_,kFramesInFlight_,inFlightFences_.data(),VK_TRUE,UINT64_MAX),"wait frame ring for groom commit");
    }

    void commitGroom(const std::shared_ptr<vh::groom::GroomAsset>& next) {
        groom_=next;
        updateOpticalGridFromGroom(*groom_);
        if(groomUploadStaging_.mapped) { waitForFrameRingIdle(); stageGroomUpload(); }
        updateMaterialFromStyle();
        if(currentGuides_.handle) requestHairReset();
        const uint64_t gen=styleGeneration_.fetch_add(1u,std::memory_order_acq_rel)+1u;
        validatedGeneration_=0u;
        scheduleValidation(groom_,gen);
    }

    static glm::vec3 srgbToLinear(glm::vec3 c) {
        auto f=[](float x){ x=std::clamp(x,0.0f,1.0f); return x<=0.04045f?x/12.92f:std::pow((x+0.055f)/1.055f,2.4f); };
        return {f(c.r),f(c.g),f(c.b)};
    }

    HairOpticsGpu materialForStyle(const vh::groom::GroomStyle& style) const {
        HairOpticsGpu mat{};
        const auto& o=style.optics;
        const SkyPresetDef preset=skyPreset(studioLighting_.environmentPreset);
        mat.colorMelanin=glm::vec4(srgbToLinear(o.directColor),o.melanin);
        mat.pigmentRoughness=glm::vec4(o.melaninRedness,0.0f,o.longitudinalRoughness,o.azimuthalRoughness);
        mat.fiberOptics=glm::vec4(o.ior,o.cuticleTilt,0.0f,1.0f);
        const int envLobes=performanceQuality_?2:4;
        const float envStrength=studioLighting_.useSkyboxLighting?studioLighting_.environmentIrradiance:0.0f;
        mat.ambientParams=glm::vec4(envStrength,studioLighting_.exposureEV,float(envLobes),float(editorState_.hairDebugView));
        mat.opticalGrid=glm::vec4(opticalOrigin_,opticalCellSize_);
        mat.renderParams=glm::vec4(float(vh::groom::kOpticalGridDim),float(o.colorModel==vh::groom::HairColorModel::NaturalMelanin),float(1+envLobes),4096.0f);
        mat.fiberShape=glm::vec4(groomParams_.fiberDiameter*o.rootDiameterScale,std::clamp(o.tipDiameterScale,0.02f,1.0f),1.0f,0.0f);
        const glm::vec3 primaryDir=azimuthElevationDirection(studioLighting_.primaryAzimuthDeg,studioLighting_.primaryElevationDeg);
        const glm::vec3 secondaryDir=azimuthElevationDirection(studioLighting_.secondaryAzimuthDeg,studioLighting_.secondaryElevationDeg);
        mat.primaryDirI=glm::vec4(primaryDir,studioLighting_.primaryIntensity);
        mat.primaryColorShadow=glm::vec4(1.0f,1.0f,1.0f,1.0f);
        mat.secondaryDirI=glm::vec4(secondaryDir,studioLighting_.secondaryIntensity);
        mat.secondaryColorShadow=glm::vec4(1.0f,1.0f,1.0f,1.0f);
        const uint32_t domDim=referenceQuality_?384u:(performanceQuality_?192u:256u);
        mat.domParams=glm::vec4(float(domDim),float(kDomLayers_),4096.0f,0.0f);
        const float areaSamples=referenceQuality_?20.0f:(performanceQuality_?6.0f:12.0f);
        mat.lightingParams=glm::vec4(studioLighting_.primaryAngularRadius,studioLighting_.secondaryAngularRadius,areaSamples,std::clamp(o.volumeScatter,0.0f,0.35f));
        mat.fiberVariation=glm::vec4(std::clamp(o.tipLightening,0.0f,0.95f),0.0f,0.0f,0.0f);
        mat.headParams=glm::vec4(studioLighting_.neutralHeadPresentation?1.0f:0.0f,
                                 std::max(studioLighting_.neutralHeadDomeStrength,0.0f),0.0f,0.0f);

        for(int i=0;i<4;++i) {
            const glm::vec3 dir=rotateY(glm::vec3(preset.envDir[i]),studioLighting_.skyboxRotationDeg);
            const glm::vec4 packed=glm::vec4(glm::normalize(dir), preset.envDir[i].w);
            switch(i) {
                case 0: mat.envDir0I=packed; mat.envColor0=glm::vec4(preset.envColor[i],0.0f); break;
                case 1: mat.envDir1I=packed; mat.envColor1=glm::vec4(preset.envColor[i],0.0f); break;
                case 2: mat.envDir2I=packed; mat.envColor2=glm::vec4(preset.envColor[i],0.0f); break;
                default: mat.envDir3I=packed; mat.envColor3=glm::vec4(preset.envColor[i],0.0f); break;
            }
        }

        if(studioLighting_.useSkyboxBackground) {
            const float bgStrength=std::max(studioLighting_.skyboxBackgroundStrength,0.0f);
            mat.skyTopColor=glm::vec4(preset.topColor*bgStrength,1.0f);
            mat.skyHorizonColor=glm::vec4(preset.horizonColor*bgStrength,1.0f);
            mat.skyBottomColor=glm::vec4(preset.bottomColor*bgStrength,1.0f);
            mat.skyBlob0=glm::vec4(preset.blob0.x + studioLighting_.skyboxRotationDeg/360.0f, preset.blob0.y, preset.blob0.z, preset.blob0.w*bgStrength);
            mat.skyBlob1=glm::vec4(preset.blob1.x + studioLighting_.skyboxRotationDeg/360.0f, preset.blob1.y, preset.blob1.z, preset.blob1.w*bgStrength);
            mat.skyBlob0Color=glm::vec4(preset.blob0Color,1.0f);
            mat.skyBlob1Color=glm::vec4(preset.blob1Color,1.0f);
            mat.skyParams=glm::vec4(preset.vignette,1.0f,0.0f,0.0f);
        } else {
            const float bg=std::clamp(studioLighting_.backgroundGray,0.0f,1.0f);
            mat.skyTopColor=glm::vec4(bg,bg,bg,1.0f);
            mat.skyHorizonColor=glm::vec4(bg,bg,bg,1.0f);
            mat.skyBottomColor=glm::vec4(bg,bg,bg,1.0f);
            mat.skyBlob0=glm::vec4(0.0f);
            mat.skyBlob1=glm::vec4(0.0f);
            mat.skyBlob0Color=glm::vec4(0.0f);
            mat.skyBlob1Color=glm::vec4(0.0f);
            mat.skyParams=glm::vec4(0.0f,0.0f,0.0f,0.0f);
        }
        return mat;
    }

    void updateMaterialFromStyle() {
        if(!materialBuffer_.mapped) return;
        const HairOpticsGpu mat=materialForStyle(styleAsset_);
        std::memcpy(static_cast<std::byte*>(materialBuffer_.mapped)+materialStride_*frameIndex_,&mat,sizeof(mat));
        // Per-frame ring storage prevents CPU material edits from racing the
        // previous GPU frame. Geometry-driven extinction is rebuilt only when
        // the guides move on the independent optical cadence.
    }

    void startScenePreparation() {
        groomParams_.fiberDiameter=0.000070f;
        groomParams_.collisionMargin=0.00035f;
        const std::filesystem::path defaultObj=exeDir_/"assets"/"character"/"ict_facekit"/"generic_neutral_mesh.obj";
        const std::filesystem::path defaultField=runtimeCacheDir_/"character"/"ict_neutral.contactbin";
        const auto objPath=characterObjOverride_.empty()?defaultObj:characterObjOverride_;
        const auto fieldPath=characterFieldOverride_.empty()?
            (characterObjOverride_.empty()?defaultField:std::filesystem::path(objPath.string()+".v39.contactbin")):
            characterFieldOverride_;
        const std::filesystem::path bindingCache=characterObjOverride_.empty()?
            (runtimeCacheDir_/"character"/"ict_neutral.bindingbin"):
            std::filesystem::path(objPath.string()+".v39.bindingbin");
        const auto params=groomParams_;
        scenePrepJob_=std::async(std::launch::async,[objPath,fieldPath,bindingCache,params]{
            PreparedSceneCpu r; r.bindingCache=bindingCache;
            const auto t0=std::chrono::steady_clock::now();
            r.character=vh::character::CharacterSurface::loadObj(objPath,fieldPath,0.0020f);
            r.characterMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            const auto t1=std::chrono::steady_clock::now();
            r.binding=vh::groom::GroomRetargeter{}.loadOrBind(r.character,bindingCache,vh::groom::kRenderStrandCount,params.seed);
            r.bindingMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count();
            return r;
        });
    }

    void createSceneBuffers() {
        if(!scenePrepJob_.valid()) startScenePreparation();
        PreparedSceneCpu prepared=scenePrepJob_.get();
        character_=std::move(prepared.character);
        groomBinding_=std::move(prepared.binding);
        std::cout<<"ICT character surface: "<<character_.vertices().size()<<" vertices, "<<character_.triangles().size()
                 <<" triangles, ContactField "<<character_.contactField().dim<<"^3 | CPU prep="<<prepared.characterMs<<" ms (overlapped)\n";
        std::cout<<"Retarget binding: "<<groomBinding_.audit.rootCount<<" roots, scalpArea="<<groomBinding_.audit.scalpArea
                 <<" m^2, load/build="<<prepared.bindingMs<<" ms (overlapped) | cache="<<prepared.bindingCache.string()<<"\n";

        // Boot does not synchronously materialize 110k strands. Load only the
        // tiny authoritative style asset; the dense derived groom is built in
        // parallel while Vulkan finishes initialization and the head can render.
        styleAsset_=vh::groom::loadGroomStyle(styleCatalog_.at(styleIndex_).file);
        frameStride_=alignUniformStride(sizeof(FrameGpu));
        simStride_=alignUniformStride(sizeof(SimGpu));
        materialStride_=alignUniformStride(sizeof(HairOpticsGpu));
        frameBuffer_=createBuffer(frameStride_*kFramesInFlight_,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        simBuffer_=createBuffer(simStride_*kFramesInFlight_,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        materialBuffer_=createBuffer(materialStride_*kFramesInFlight_,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const HairOpticsGpu mat=materialForStyle(styleAsset_);
        for(uint32_t i=0;i<kFramesInFlight_;++i)
            std::memcpy(static_cast<std::byte*>(materialBuffer_.mapped)+materialStride_*i,&mat,sizeof(mat));
        createSceneStaticGpuBuffers();
        createGroomBuffers();
        headIndexCount_=static_cast<uint32_t>(character_.indices().size());
        scheduleStyleBuildIndex(styleIndex_,styleAsset_,{},true,false);
    }

    void requestHairReset() {
        if(!groom_) return;
        // Hot dynamics buffers stay device-local. Reset is expressed as GPU
        // dataflow in the normal frame command buffer, never as host writes.
        guideResetPending_=true;
        opticalFieldDirty_=true;
        opticalUpdateAccumulator_=0.0f;
        domFieldDirty_=true;
        domUpdateAccumulator_=0.0f;
        simAccumulator_=0.0f; volumeResetRequested_=true; resetRequested_=false;
    }

    void scheduleStyleBuildIndex(size_t nextIndex,const vh::groom::GroomStyle& style,
                                 std::shared_ptr<vh::groom::GroomAsset> reusableBasis={},
                                 bool styleSwitch=false,bool resolveAfterBuild=false) {
        if(styleBuildPending_.load(std::memory_order_acquire)) return;
        const auto params=groomParams_;
        styleBuildPending_.store(true,std::memory_order_release);
        const auto styleFile=styleCatalog_.at(nextIndex).file;
        const auto cacheFile=runtimeCacheDir_/"grooms"/(style.slug+".v39groom");
        styleBuildJob_=std::async(std::launch::async,[this,nextIndex,style,params,reusableBasis,styleSwitch,resolveAfterBuild,styleFile,cacheFile]{
            AsyncStyleBuildResult r; r.index=nextIndex; r.style=style; r.styleSwitch=styleSwitch;
            try {
                const bool canUseDerivedCache=styleSwitch && !reusableBasis && !resolveAfterBuild;
                vh::groom::DerivedGroomCacheKey cacheKey{};
                if(canUseDerivedCache) {
                    cacheKey.styleFingerprint=vh::groom::fingerprintFile(styleFile);
                    cacheKey.meshFingerprint=character_.meshFingerprint();
                    cacheKey.params=params;
                    vh::groom::GroomAsset cached;
                    double loadMs=0.0;
                    if(vh::groom::loadDerivedGroom(cacheFile,cacheKey,cached,&loadMs)) {
                        r.groom=std::make_shared<vh::groom::GroomAsset>(std::move(cached));
                        r.cacheHit=true; r.cacheIoMs=loadMs; r.buildMs=loadMs;
                    }
                }
                if(!r.groom) {
                    const auto t0=std::chrono::steady_clock::now();
                    r.groom=std::make_shared<vh::groom::GroomAsset>(
                        vh::groom::GroomBuilder{}.build(groomBinding_,params,r.style,reusableBasis.get()));
                    if(resolveAfterBuild) vh::groom::resolveContacts(character_,params,*r.groom);
                    r.buildMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
                    if(canUseDerivedCache) vh::groom::saveDerivedGroom(cacheFile,cacheKey,*r.groom,&r.cacheIoMs);
                }
            } catch(const std::exception& e) { r.error=e.what(); }
            std::scoped_lock lock(styleBuildMutex_);
            completedStyleBuild_=std::move(r);
        });
    }

    void rebuildGroom(size_t nextIndex) {
        styleIndexRequested_=-1;
        if(styleCatalog_.empty() || nextIndex>=styleCatalog_.size() || nextIndex==styleIndex_) return;
        if(styleBuildPending_.load(std::memory_order_acquire)) return;
        try {
            auto nextStyle=vh::groom::loadGroomStyle(styleCatalog_.at(nextIndex).file);
            scheduleStyleBuildIndex(nextIndex,nextStyle,{},true,false);
        } catch(const std::exception& e) {
            assetError_=std::string("Style asset invalid: ")+e.what();
            std::cerr<<"[ASSET] "<<assetError_<<"\n";
        }
    }

    void pollStyleBuild() {
        if(!styleBuildPending_.load(std::memory_order_acquire)) return;
        std::optional<AsyncStyleBuildResult> ready;
        {
            std::scoped_lock lock(styleBuildMutex_);
            if(completedStyleBuild_) { ready=std::move(completedStyleBuild_); completedStyleBuild_.reset(); }
        }
        if(!ready) return;
        if(styleBuildJob_.valid()) { try { styleBuildJob_.get(); } catch(...) {} }
        styleBuildPending_.store(false,std::memory_order_release);
        if(!ready->error.empty() || !ready->groom) {
            assetError_=std::string("Style asset invalid: ")+ready->error;
            std::cerr<<"[ASSET] "<<assetError_<<"\n";
            return;
        }
        styleIndex_=ready->index; styleAsset_=std::move(ready->style); lastGroomBuildMs_=ready->buildMs;
        commitGroom(ready->groom);
        editedGroomRebuildRequested_=false; assetError_.clear();
        std::cout<<(ready->styleSwitch?"Switched hairstyle -> ":"Updated hairstyle -> ")<<styleAsset_.name
                 <<" | "<<(ready->cacheHit?"derived-cache hit":"async build")<<"="<<lastGroomBuildMs_<<" ms"
                 <<(ready->cacheHit?"":" | cache-write="+std::to_string(ready->cacheIoMs)+" ms")
                 <<" | validation=pending\n";
    }

    void rebuildEditedGroom() {
        if(styleBuildPending_.load(std::memory_order_acquire)) return;
        editedGroomRebuildRequested_=false;
        // The authoritative Style is tiny; materialize its dense derived geometry
        // off the UI thread. Ordinary edits reuse the stable material-space basis
        // and follower weights from the currently committed groom.
        scheduleStyleBuildIndex(styleIndex_,styleAsset_,groom_,false,editorState_.keepOutsideSkin);
    }

    void resolveCurrentContacts() {
        resolveContactsRequested_=false; if(!groom_) return;
        auto candidate=std::make_shared<vh::groom::GroomAsset>(*groom_);
        try {
            const auto report=vh::groom::resolveContacts(character_,groomParams_,*candidate);
            commitGroom(candidate); assetError_.clear();
            std::cout<<"Resolve Contacts: groups="<<report.correctedGroups<<" constraints="<<report.projectedConstraints
                     <<" iterations="<<report.iterations<<" max="<<report.maxCorrection<<" m | validation=pending\n";
        } catch(const std::exception& e) {
            assetError_=std::string("Contact resolve failed: ")+e.what(); std::cerr<<"[ASSET] "<<assetError_<<"\n";
        }
    }

    glm::vec3 cameraPosition() const {
        const glm::vec3 target(0.0f, -0.005f, 0.0f);
        const float cp = std::cos(pitch_);
        glm::vec3 dir{
            cp * std::sin(yaw_),
            std::sin(pitch_),
            cp * std::cos(yaw_)
        };
        return target + distance_ * dir;
    }

    std::array<glm::vec4, 6> extractFrustumPlanes(const glm::mat4& m) const {
        const glm::vec4 r0(m[0][0], m[1][0], m[2][0], m[3][0]);
        const glm::vec4 r1(m[0][1], m[1][1], m[2][1], m[3][1]);
        const glm::vec4 r2(m[0][2], m[1][2], m[2][2], m[3][2]);
        const glm::vec4 r3(m[0][3], m[1][3], m[2][3], m[3][3]);
        std::array<glm::vec4, 6> p{r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2};
        for (auto& plane : p) {
            const float l = glm::length(glm::vec3(plane));
            if (l > 1e-8f) plane /= l;
        }
        return p;
    }

    void updateUniforms(float dt) {
        const glm::vec3 eye = cameraPosition();
        const glm::vec3 target(0.0f, -0.025f, 0.0f);
        glm::mat4 view = glm::lookAtRH(eye, target, glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(42.0f),
            float(swapExtent_.width) / float(std::max(1u, swapExtent_.height)), 0.04f, 4.0f);
        proj[1][1] *= -1.0f;
        FrameGpu frame{};
        frame.viewProj = proj * view;
        lastViewProj_ = frame.viewProj;
        frame.cameraPos = glm::vec4(eye, 1.0f);
        auto planes = extractFrustumPlanes(frame.viewProj);
        for (size_t i = 0; i < planes.size(); ++i) frame.frustumPlanes[i] = planes[i];
        frame.screenParams = glm::vec4(float(swapExtent_.width), float(swapExtent_.height),
                                       std::tan(glm::radians(21.0f)), float(static_cast<uint32_t>(msaaSamples_)));
        std::memcpy(static_cast<std::byte*>(frameBuffer_.mapped)+frameStride_*frameIndex_, &frame, sizeof(frame));

        // Intuitive [0,1] material sliders are mapped to solver compliance here.
        // The groom asset never stores low-level numerical solver constants.
        const float stretchSoft = 1.0f - std::clamp(hairMaterial_.stretchStiffness, 0.0f, 1.0f);
        const float bendSoft = 1.0f - std::clamp(hairMaterial_.bendStiffness, 0.0f, 1.0f);
        const float styleSoft = 1.0f - std::clamp(hairMaterial_.styleMemory, 0.0f, 1.0f);
        const float stretchCompliance = 2.0e-9f + stretchSoft * stretchSoft * 2.0e-6f;
        const float bendCompliance = 1.0e-7f + bendSoft * bendSoft * 2.0e-4f;
        const float styleCompliance = 8.0e-5f + styleSoft * styleSoft * 2.6e-2f;

        SimGpu sim{};
        (void)dt;
        sim.restFromHead = glm::inverse(sim.headFromRest);
        sim.dtSolver = glm::vec4(kFixedSimDt_, stretchCompliance, bendCompliance, elapsed_);
        glm::vec3 windDir = wind_.direction;
        if (glm::dot(windDir, windDir) < 1e-8f) windDir = glm::vec3(1,0,0);
        windDir = glm::normalize(windDir);
        sim.wind = glm::vec4(windDir, wind_.enabled ? std::max(wind_.speed,0.0f) : 0.0f);
        sim.windPositionType = glm::vec4(wind_.position, float(static_cast<uint32_t>(wind_.type)));
        sim.windShape = glm::vec4(std::max(wind_.range,0.05f),
                                  std::cos(glm::radians(std::clamp(wind_.coneDegrees,1.0f,89.0f))),
                                  std::clamp(wind_.gust,0.0f,0.95f),
                                  std::clamp(wind_.turbulence,0.0f,1.0f));
        sim.forces = glm::vec4(-9.81f,
                               0.72f + 1.9f * std::clamp(hairMaterial_.damping,0.0f,1.0f),
                               0.64f, groomParams_.collisionMargin);
        sim.contact = glm::vec4(std::max(hairMaterial_.staticFriction,0.0f),
                                std::max(hairMaterial_.dynamicFriction,0.0f),
                                0.0014f, 1.0f);
        sim.style = glm::vec4(styleCompliance, 4.5f, 9.0f, 1.2f);
        sim.aero = glm::vec4(std::max(hairMaterial_.airDrag,0.0f), 0.12f, 16.0f, 1.8f);
        const float fullness = std::clamp(hairMaterial_.bulkFullness,0.55f,1.45f);
        sim.volume = glm::vec4(0.58f / fullness, 0.24f * fullness, 0.55f, 0.85f);
        sim.grid = glm::vec4(-0.42f,-0.56f,-0.42f,0.018f);
        sim.bodyField = glm::vec4(character_.contactField().origin, character_.contactField().cellSize);
        sim.bodyInfo = glm::vec4(float(character_.contactField().dim), 1.0f, 0.0f, 0.0f);
        std::memcpy(static_cast<std::byte*>(simBuffer_.mapped)+simStride_*frameIndex_, &sim, sizeof(sim));
        const HairOpticsGpu mat=materialForStyle(styleAsset_);
        std::memcpy(static_cast<std::byte*>(materialBuffer_.mapped)+materialStride_*frameIndex_,&mat,sizeof(mat));

        std::string title = std::string("Vulkan Hair Studio | ") + styleAsset_.name + (styleBuildPending_.load(std::memory_order_acquire)?" [building]":"") + " | " + deviceProps_.deviceName +
            " | mesh-native scalp | 110k -> 1600 rod guides + 48^3 volume | " + std::to_string(static_cast<uint32_t>(msaaSamples_)) +
            "x MSAA | mode " + std::string(editorState_.mode==vh::editor::StudioMode::Display ? "DISPLAY " :
                                      editorState_.mode==vh::editor::StudioMode::Edit ? "EDIT " : "PHYSICS ") +
            "| wind " + (wind_.enabled ? "ON " : "OFF ") +
            std::to_string(int(wind_.speed * 10.0f) / 10.0f) + " m/s";
        glfwSetWindowTitle(window_, title.c_str());
    }

    void initImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename = nullptr; // layout is intentionally deterministic
        ImGui_ImplGlfw_InitForVulkan(window_, true);

        ImGui_ImplVulkan_InitInfo info{};
        info.Instance = instance_;
        info.PhysicalDevice = physicalDevice_;
        info.Device = device_;
        info.QueueFamily = graphicsFamily_;
        info.Queue = graphicsQueue_;
        info.DescriptorPool = VK_NULL_HANDLE;
        info.DescriptorPoolSize = 64;
        info.RenderPass = VK_NULL_HANDLE;
        info.MinImageCount = 2;
        info.ImageCount = static_cast<uint32_t>(swapImages_.size());
        info.MSAASamples = VK_SAMPLE_COUNT_1_BIT; // UI is composited after hair MSAA resolve
        info.UseDynamicRendering = true;
        info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
        info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapFormat_;
        info.MinAllocationSize = 1024u * 1024u;
        if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("ImGui Vulkan backend initialization failed");
        vh::editor::applyStudioTheme();
        imguiInitialized_ = true;
    }

    void ensureScreenshotReadbackBuffer() {
        const VkDeviceSize needed=VkDeviceSize(swapExtent_.width)*VkDeviceSize(swapExtent_.height)*4u;
        if(screenshotReadback_.handle && screenshotReadbackBytes_==needed) return;
        destroyBuffer(screenshotReadback_);
        screenshotReadback_=createBuffer(needed,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        screenshotReadbackBytes_=needed;
    }

    const char* screenshotExtension(vh::editor::ScreenshotFormat fmt) const {
        switch(fmt) {
            case vh::editor::ScreenshotFormat::Bmp: return ".bmp";
            case vh::editor::ScreenshotFormat::Tga: return ".tga";
            default: return ".png";
        }
    }

    std::filesystem::path makeScreenshotPath(bool withUi,vh::editor::ScreenshotFormat fmt) const {
        const auto dir=exeDir_/"screenshots";
        std::filesystem::create_directories(dir);
        const auto now=std::chrono::system_clock::now().time_since_epoch();
        const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        return dir/(std::string("screenshot_")+std::to_string(ms)+(withUi?"_ui":"_clean")+screenshotExtension(fmt));
    }

    void screenshotRgb(std::vector<uint8_t>& out,const void* srcData,uint32_t width,uint32_t height,bool flipY) const {
        const auto* src=static_cast<const std::byte*>(srcData);
        const bool bgra=(swapFormat_==VK_FORMAT_B8G8R8A8_UNORM||swapFormat_==VK_FORMAT_B8G8R8A8_SRGB);
        out.resize(size_t(width)*size_t(height)*3u);
        for(uint32_t y=0;y<height;++y) {
            const uint32_t srcY=flipY?(height-1u-y):y;
            const std::byte* srcRow=src+size_t(srcY)*size_t(width)*4u;
            uint8_t* dst=out.data()+size_t(y)*size_t(width)*3u;
            for(uint32_t x=0;x<width;++x) {
                const uint8_t c0=std::to_integer<uint8_t>(srcRow[x*4u+0u]);
                const uint8_t c1=std::to_integer<uint8_t>(srcRow[x*4u+1u]);
                const uint8_t c2=std::to_integer<uint8_t>(srcRow[x*4u+2u]);
                dst[x*3u+0u]=bgra?c2:c0;
                dst[x*3u+1u]=c1;
                dst[x*3u+2u]=bgra?c0:c2;
            }
        }
    }

    uint32_t pngCrc32(const uint8_t* data,size_t size) const {
        uint32_t crc=0xffffffffu;
        for(size_t i=0;i<size;++i) {
            crc^=data[i];
            for(int b=0;b<8;++b) crc=(crc>>1u)^(0xedb88320u&uint32_t(-int32_t(crc&1u)));
        }
        return crc^0xffffffffu;
    }

    uint32_t pngAdler32(const uint8_t* data,size_t size) const {
        constexpr uint32_t mod=65521u;
        uint32_t a=1u,b=0u;
        for(size_t i=0;i<size;++i) { a=(a+data[i])%mod; b=(b+a)%mod; }
        return (b<<16u)|a;
    }

    void writeBe32(std::ostream& out,uint32_t v) const {
        out.put(char((v>>24u)&0xffu)); out.put(char((v>>16u)&0xffu));
        out.put(char((v>>8u)&0xffu)); out.put(char(v&0xffu));
    }

    void writePngChunk(std::ostream& out,const char type[4],const std::vector<uint8_t>& payload) const {
        writeBe32(out,uint32_t(payload.size()));
        std::vector<uint8_t> crcData(4u+payload.size());
        std::memcpy(crcData.data(),type,4u);
        if(!payload.empty()) std::memcpy(crcData.data()+4u,payload.data(),payload.size());
        out.write(type,4);
        if(!payload.empty()) out.write(reinterpret_cast<const char*>(payload.data()),std::streamsize(payload.size()));
        writeBe32(out,pngCrc32(crcData.data(),crcData.size()));
    }

    void saveScreenshotPng(const std::filesystem::path& path,const void* srcData,uint32_t width,uint32_t height) const {
        std::vector<uint8_t> rgb; screenshotRgb(rgb,srcData,width,height,false);
        std::vector<uint8_t> raw((size_t(width)*3u+1u)*size_t(height));
        for(uint32_t y=0;y<height;++y) {
            uint8_t* row=raw.data()+size_t(y)*(size_t(width)*3u+1u);
            row[0]=0u;
            std::memcpy(row+1u,rgb.data()+size_t(y)*size_t(width)*3u,size_t(width)*3u);
        }
        std::vector<uint8_t> z;
        z.reserve(raw.size()+raw.size()/65535u*5u+16u);
        z.push_back(0x78u); z.push_back(0x01u); // zlib, fastest/no-compression deflate blocks
        size_t pos=0;
        while(pos<raw.size()) {
            const uint16_t len=uint16_t(std::min<size_t>(65535u,raw.size()-pos));
            const bool final=pos+len==raw.size();
            z.push_back(final?1u:0u);
            z.push_back(uint8_t(len&0xffu)); z.push_back(uint8_t((len>>8u)&0xffu));
            const uint16_t nlen=uint16_t(~len);
            z.push_back(uint8_t(nlen&0xffu)); z.push_back(uint8_t((nlen>>8u)&0xffu));
            z.insert(z.end(),raw.begin()+ptrdiff_t(pos),raw.begin()+ptrdiff_t(pos+len));
            pos+=len;
        }
        const uint32_t ad=pngAdler32(raw.data(),raw.size());
        z.push_back(uint8_t((ad>>24u)&0xffu)); z.push_back(uint8_t((ad>>16u)&0xffu));
        z.push_back(uint8_t((ad>>8u)&0xffu)); z.push_back(uint8_t(ad&0xffu));
        std::ofstream out(path,std::ios::binary);
        if(!out) throw std::runtime_error("Failed to open screenshot output: "+path.string());
        const uint8_t sig[8]={137u,80u,78u,71u,13u,10u,26u,10u};
        out.write(reinterpret_cast<const char*>(sig),8);
        std::vector<uint8_t> ihdr(13u,0u);
        ihdr[0]=uint8_t((width>>24u)&0xffu); ihdr[1]=uint8_t((width>>16u)&0xffu); ihdr[2]=uint8_t((width>>8u)&0xffu); ihdr[3]=uint8_t(width&0xffu);
        ihdr[4]=uint8_t((height>>24u)&0xffu); ihdr[5]=uint8_t((height>>16u)&0xffu); ihdr[6]=uint8_t((height>>8u)&0xffu); ihdr[7]=uint8_t(height&0xffu);
        ihdr[8]=8u; ihdr[9]=2u;
        writePngChunk(out,"IHDR",ihdr); writePngChunk(out,"IDAT",z); writePngChunk(out,"IEND",{});
    }

    void saveScreenshotBmp(const std::filesystem::path& path,const void* srcData,uint32_t width,uint32_t height) const {
        std::vector<uint8_t> rgb; screenshotRgb(rgb,srcData,width,height,false);
        const uint32_t rowStride=((width*3u)+3u)&~3u;
        const uint32_t pixelBytes=rowStride*height;
        const uint32_t fileSize=14u+40u+pixelBytes;
        std::ofstream out(path,std::ios::binary);
        if(!out) throw std::runtime_error("Failed to open screenshot output: "+path.string());
        auto put16=[&](uint16_t v){ out.put(char(v&0xff)); out.put(char((v>>8)&0xff)); };
        auto put32=[&](uint32_t v){ out.put(char(v&0xff)); out.put(char((v>>8)&0xff)); out.put(char((v>>16)&0xff)); out.put(char((v>>24)&0xff)); };
        out.put('B'); out.put('M'); put32(fileSize); put16(0); put16(0); put32(54u);
        put32(40u); put32(width); put32(height); put16(1u); put16(24u); put32(0u); put32(pixelBytes);
        put32(2835u); put32(2835u); put32(0u); put32(0u);
        std::vector<char> row(rowStride,0);
        for(uint32_t y=0;y<height;++y) {
            const uint8_t* srcRow=rgb.data()+size_t(height-1u-y)*size_t(width)*3u;
            for(uint32_t x=0;x<width;++x) {
                row[x*3u+0u]=char(srcRow[x*3u+2u]);
                row[x*3u+1u]=char(srcRow[x*3u+1u]);
                row[x*3u+2u]=char(srcRow[x*3u+0u]);
            }
            out.write(row.data(),std::streamsize(row.size()));
        }
    }

    void saveScreenshotTga(const std::filesystem::path& path,const void* srcData,uint32_t width,uint32_t height) const {
        std::vector<uint8_t> rgb; screenshotRgb(rgb,srcData,width,height,false);
        std::ofstream out(path,std::ios::binary);
        if(!out) throw std::runtime_error("Failed to open screenshot output: "+path.string());
        std::array<uint8_t,18> h{}; h[2]=2u; h[12]=uint8_t(width&0xffu); h[13]=uint8_t((width>>8u)&0xffu); h[14]=uint8_t(height&0xffu); h[15]=uint8_t((height>>8u)&0xffu); h[16]=24u; h[17]=0x20u;
        out.write(reinterpret_cast<const char*>(h.data()),std::streamsize(h.size()));
        for(size_t i=0;i<rgb.size();i+=3u) { out.put(char(rgb[i+2u])); out.put(char(rgb[i+1u])); out.put(char(rgb[i])); }
    }

    void saveScreenshot(const std::filesystem::path& path,vh::editor::ScreenshotFormat fmt,const void* srcData,uint32_t width,uint32_t height) const {
        switch(fmt) {
            case vh::editor::ScreenshotFormat::Bmp: saveScreenshotBmp(path,srcData,width,height); break;
            case vh::editor::ScreenshotFormat::Tga: saveScreenshotTga(path,srcData,width,height); break;
            default: saveScreenshotPng(path,srcData,width,height); break;
        }
    }

    void readBackCompletedScreenshot(uint32_t slot) {
        if(!screenshotReadbackPending_[slot] || !screenshotReadback_.mapped) return;
        saveScreenshot(screenshotPaths_[slot],screenshotFormats_[slot],screenshotReadback_.mapped,screenshotWidths_[slot],screenshotHeights_[slot]);
        std::cout << "Saved screenshot: " << screenshotPaths_[slot].string() << "\n";
        screenshotReadbackPending_[slot]=false;
    }

    void captureSwapchainForScreenshot(VkCommandBuffer cmd,uint32_t imageIndex,VkImageLayout returnLayout) {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount=1;
        region.imageExtent={swapExtent_.width,swapExtent_.height,1};
        transitionImage(cmd,swapImages_[imageIndex],VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(cmd,swapImages_[imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,screenshotReadback_.handle,1,&region);
        VkBufferMemoryBarrier copyBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        copyBarrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        copyBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.buffer=screenshotReadback_.handle;
        copyBarrier.offset=0;
        copyBarrier.size=screenshotReadbackBytes_;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&copyBarrier,0,nullptr);
        transitionImage(cmd,swapImages_[imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,returnLayout,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        returnLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR?VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT:VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        returnLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR?0:VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        screenshotReadbackPending_[frameIndex_]=true;
        screenshotWidths_[frameIndex_]=swapExtent_.width;
        screenshotHeights_[frameIndex_]=swapExtent_.height;
        screenshotSavedWithUi_[frameIndex_]=screenshotIncludeUi_;
        screenshotFormats_[frameIndex_]=screenshotFormat_;
        screenshotPaths_[frameIndex_]=makeScreenshotPath(screenshotIncludeUi_,screenshotFormat_);
        screenshotRequested_=false;
    }

    void beginEditorFrame() {
        // A screenshot that includes the editor is deliberately armed one frame
        // after the modal button press. ImGui closes popups at frame boundaries,
        // so this preserves the editor UI without baking the screenshot modal itself.
        if(screenshotWithUiDeferred_) {
            screenshotRequested_=true;
            screenshotIncludeUi_=true;
            screenshotFormat_=deferredScreenshotFormat_;
            screenshotWithUiDeferred_=false;
        }
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const vh::editor::PerformanceStats perf{cpuFrameMs_,gpuFrameMs_,gpuPhysicsMs_,gpuOpticalMs_,gpuSceneMs_,lastGroomBuildMs_,lastOpticalBuildMs_,uint32_t(msaaSamples_)};
        const vh::editor::StudioLighting lightingBefore = studioLighting_;
        auto actions = vh::editor::draw(editorState_, character_, groomBinding_, styleCatalog_, styleIndex_, styleAsset_, groomParams_,
                                        hairMaterial_, wind_, studioLighting_, validation_, validationPending_.load(std::memory_order_acquire), styleBuildPending_.load(std::memory_order_acquire), perf, assetError_, lastViewProj_,
                                        swapExtent_.width, swapExtent_.height);
        if(actions.requestedStyleIndex>=0) styleIndexRequested_=actions.requestedStyleIndex;
        if(actions.requestScreenshot) {
            if(actions.screenshotIncludeUi) {
                screenshotWithUiDeferred_=true;
                deferredScreenshotFormat_=actions.screenshotFormat;
            } else {
                screenshotRequested_=true;
                screenshotIncludeUi_=false;
                screenshotFormat_=actions.screenshotFormat;
            }
        }
        if(actions.groomDirty) editedGroomRebuildRequested_=true;
        if(actions.lookDirty) {
            if(environmentTransportChanged(lightingBefore, studioLighting_)) {
                opticalFieldDirty_=true;
                opticalUpdateThisFrame_=true;
                opticalUpdateAccumulator_=0.0f;
            }
            if(directionalShadowChanged(lightingBefore, studioLighting_)) {
                domFieldDirty_=true;
                domUpdateThisFrame_=true;
                domUpdateAccumulator_=0.0f;
            }
            updateMaterialFromStyle();
        }
        if(actions.resolveContacts) resolveContactsRequested_=true;
        if(actions.resetPhysics || actions.physicsDirty) resetRequested_=true;
        if(actions.modeChangeRequested) {
            editorState_.mode=actions.requestedMode;
            simAccumulator_=0.0f;
            if(actions.requestedMode==vh::editor::StudioMode::Physics) {
                resetRequested_=true;
                assetError_.clear();
            }
        }
        ImGui::Render();
    }

    void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                         VkImageAspectFlags aspect, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                         VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = oldLayout;
        b.newLayout = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange.aspectMask = aspect;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        vkCheck(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkCheck(vkBeginCommandBuffer(commandBuffer_, &bi), "vkBeginCommandBuffer");
        const uint32_t timestampBase=5u*frameIndex_;
        if(timestampQueryPool_) {
            vkCmdResetQueryPool(commandBuffer_,timestampQueryPool_,timestampBase,5);
            vkCmdWriteTimestamp(commandBuffer_,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,timestampQueryPool_,timestampBase+0u);
        }

        // Style commits never stop the device. The persistent buffers are updated
        // here, in queue order, before the same frame consumes them.
        recordPendingGroomUpload(commandBuffer_);
        if(guideResetPending_) {
            VkBufferCopy cg{0,0,restGuides_.size};
            vkCmdCopyBuffer(commandBuffer_,restGuides_.handle,currentGuides_.handle,1,&cg);
            VkBufferCopy cf{0,0,restGuideFrames_.size};
            vkCmdCopyBuffer(commandBuffer_,restGuideFrames_.handle,currentGuideFrames_.handle,1,&cf);
            vkCmdFillBuffer(commandBuffer_,guideVelocities_.handle,0,VK_WHOLE_SIZE,0u);
            std::array<VkBufferMemoryBarrier,3> rb{};
            Buffer* hot[3]={&currentGuides_,&currentGuideFrames_,&guideVelocities_};
            for(size_t i=0;i<rb.size();++i){ rb[i]={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; rb[i].srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; rb[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; rb[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; rb[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; rb[i].buffer=hot[i]->handle; rb[i].offset=0; rb[i].size=VK_WHOLE_SIZE; }
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,0,0,nullptr,uint32_t(rb.size()),rb.data(),0,nullptr);
            guideResetPending_=false;
        }


        // Runtime physics: reduced rods -> P2G -> collective pressure/momentum -> G2P.
        // Runtime contacts are discovered from current positions on the grid;
        // there is no fixed rest-neighbour list and no render-time body projection.
        if (simStepsThisFrame_ > 0u) {
            const uint32_t simOffset=simDynamicOffset();
            vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, hairPipelineLayout_,
                                    2, 1, &dynamicsSet_, 1, &simOffset);
            auto barrierFor = [&](VkBuffer buffer, VkAccessFlags src, VkAccessFlags dst,
                                  VkPipelineStageFlags dstStage) {
                VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                b.srcAccessMask = src;
                b.dstAccessMask = dst;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.buffer = buffer;
                b.offset = 0;
                b.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
            };
            if (volumeResetRequested_) {
                // Reset the persistent previous-step volume state together with the
                // guide state.  This prevents wind shielding from sampling a stale
                // hairstyle for one substep after R or a preset switch.
                vkCmdFillBuffer(commandBuffer_, volumeAccum_.handle, 0, VK_WHOLE_SIZE, 0u);
                vkCmdFillBuffer(commandBuffer_, volumeState_.handle, 0, VK_WHOLE_SIZE, 0u);
                std::array<VkBufferMemoryBarrier, 2> resetBarriers{};
                for (auto& b : resetBarriers) {
                    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.offset = 0;
                    b.size = VK_WHOLE_SIZE;
                }
                resetBarriers[0].buffer = volumeAccum_.handle;
                resetBarriers[1].buffer = volumeState_.handle;
                vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     0, nullptr, static_cast<uint32_t>(resetBarriers.size()),
                                     resetBarriers.data(), 0, nullptr);
                volumeResetRequested_ = false;
            }

            const uint32_t gridGroups=(vh::physics::kVolumeGridCount+255u)/256u;

            for (uint32_t step = 0; step < simStepsThisFrame_; ++step) {
                vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, simPipeline_);
                vkCmdDispatch(commandBuffer_, vh::groom::kSimulationGuideCount, 1, 1);
                barrierFor(currentGuides_.handle, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                barrierFor(guideVelocities_.handle, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

                vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, volumeClearPipeline_);
                vkCmdDispatch(commandBuffer_, gridGroups, 1, 1);
                barrierFor(volumeAccum_.handle, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                barrierFor(volumeState_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

                vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, volumeP2GPipeline_);
                vkCmdDispatch(commandBuffer_, vh::groom::kSimulationGuideCount, 1, 1);
                barrierFor(volumeAccum_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

                vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, volumeSolvePipeline_);
                vkCmdDispatch(commandBuffer_, gridGroups, 1, 1);
                barrierFor(volumeState_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

                vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, volumeG2PPipeline_);
                vkCmdDispatch(commandBuffer_, vh::groom::kSimulationGuideCount, 1, 1);
                barrierFor(guideVelocities_.handle, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
            barrierFor(currentGuides_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
            barrierFor(currentGuideFrames_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
            barrierFor(volumeState_.handle, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }

        if(timestampQueryPool_) vkCmdWriteTimestamp(commandBuffer_,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestampQueryPool_,timestampBase+1u);

        if(groom_ && (opticalFieldDirty_ || opticalUpdateThisFrame_)) {
            // Update optical extinction on its own cadence from the deformed guides.
            vkCmdFillBuffer(commandBuffer_,opticalDensity_.handle,0,VK_WHOLE_SIZE,0u);
            VkBufferMemoryBarrier clearBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            clearBarrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            clearBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            clearBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; clearBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            clearBarrier.buffer=opticalDensity_.handle; clearBarrier.offset=0; clearBarrier.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,0,nullptr,1,&clearBarrier,0,nullptr);

            const std::array<VkDescriptorSet,2> opticalSets{dynamicsSet_,materialSet_};
            const std::array<uint32_t,2> opticalDynamic{simDynamicOffset(),materialDynamicOffset()};
            vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,hairPipelineLayout_,
                                    2,uint32_t(opticalSets.size()),opticalSets.data(),uint32_t(opticalDynamic.size()),opticalDynamic.data());
            vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,opticalSplatPipeline_);
            constexpr uint32_t kOpticalSegments=vh::groom::kSimulationGuideCount*(vh::groom::kGuidePoints-1u);
            vkCmdDispatch(commandBuffer_,kOpticalSegments,1,1);
            VkBufferMemoryBarrier densityBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            densityBarrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; densityBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            densityBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; densityBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            densityBarrier.buffer=opticalDensity_.handle; densityBarrier.offset=0; densityBarrier.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,0,nullptr,1,&densityBarrier,0,nullptr);

            vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,opticalDepthPipeline_);
            const uint32_t materialOffset=materialDynamicOffset();
            vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,hairPipelineLayout_,
                                    3,1,&materialSet_,1,&materialOffset);
            const uint32_t envLobes=performanceQuality_?2u:4u;
            const uint32_t dirCount=1u+envLobes;
            const uint32_t scanlines=dirCount*vh::groom::kOpticalGridDim*vh::groom::kOpticalGridDim;
            vkCmdDispatch(commandBuffer_,(scanlines+63u)/64u,1,1);
            VkBufferMemoryBarrier depthBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            depthBarrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; depthBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            depthBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; depthBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.buffer=opticalDepth_.handle; depthBarrier.offset=0; depthBarrier.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0,0,nullptr,1,&depthBarrier,0,nullptr);
            opticalFieldDirty_=false;
            opticalUpdateThisFrame_=false;
        }

        if(groom_ && (domFieldDirty_ || domUpdateThisFrame_)) {
            vkCmdFillBuffer(commandBuffer_,primaryDeepOpacity_.handle,0,VK_WHOLE_SIZE,0u);
            vkCmdFillBuffer(commandBuffer_,secondaryDeepOpacity_.handle,0,VK_WHOLE_SIZE,0u);
            std::array<VkBufferMemoryBarrier,2> clears{}; Buffer* domBufs[2]={&primaryDeepOpacity_,&secondaryDeepOpacity_};
            for(size_t i=0;i<2;++i){clears[i]={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};clears[i].srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;clears[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;clears[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;clears[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;clears[i].buffer=domBufs[i]->handle;clears[i].offset=0;clears[i].size=VK_WHOLE_SIZE;}
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,uint32_t(clears.size()),clears.data(),0,nullptr);
            const std::array<VkDescriptorSet,2> domSets{dynamicsSet_,materialSet_};
            const std::array<uint32_t,2> domDynamic{simDynamicOffset(),materialDynamicOffset()};
            vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,hairPipelineLayout_,2,uint32_t(domSets.size()),domSets.data(),uint32_t(domDynamic.size()),domDynamic.data());
            vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,domSplatPipeline_);
            constexpr uint32_t kDomSegments=vh::groom::kSimulationGuideCount*(vh::groom::kGuidePoints-1u);
            vkCmdDispatch(commandBuffer_,kDomSegments,1,1);
            std::array<VkBufferMemoryBarrier,2> splatBars{};
            for(size_t i=0;i<2;++i){splatBars[i]={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};splatBars[i].srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;splatBars[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;splatBars[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;splatBars[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;splatBars[i].buffer=domBufs[i]->handle;splatBars[i].offset=0;splatBars[i].size=VK_WHOLE_SIZE;}
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,uint32_t(splatBars.size()),splatBars.data(),0,nullptr);
            vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_COMPUTE,domPrefixPipeline_);
            const uint32_t domDim=referenceQuality_?384u:(performanceQuality_?192u:256u);
            vkCmdDispatch(commandBuffer_,(domDim*domDim+127u)/128u,1,1);
            std::array<VkBufferMemoryBarrier,2> domRead{};
            for(size_t i=0;i<2;++i){domRead[i]={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};domRead[i].srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;domRead[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;domRead[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;domRead[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;domRead[i].buffer=domBufs[i]->handle;domRead[i].offset=0;domRead[i].size=VK_WHOLE_SIZE;}
            vkCmdPipelineBarrier(commandBuffer_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,uint32_t(domRead.size()),domRead.data(),0,nullptr);
            domFieldDirty_=false; domUpdateThisFrame_=false;
        }

        if(timestampQueryPool_) vkCmdWriteTimestamp(commandBuffer_,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestampQueryPool_,timestampBase+2u);

        // Render scene HDR, rasterize sample-resolved hair geometry, compose
        // strand radiance, then apply the display transform.
        const VkImageLayout sceneOld=sceneColorInitialized_?VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,sceneColorImage_,sceneOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        sceneColorInitialized_?VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,sceneColorInitialized_?VK_ACCESS_SHADER_READ_BIT:0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        const VkImageLayout msaaOld=msaaColorInitialized_?VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,msaaColorImage_,msaaOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        msaaColorInitialized_?VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,msaaColorInitialized_?VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT:0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        const VkImageLayout visOld=hairGeometryInitialized_?VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,hairGeom0Image_,visOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        hairGeometryInitialized_?VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,hairGeometryInitialized_?VK_ACCESS_SHADER_READ_BIT:0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        transitionImage(commandBuffer_,hairGeom1Image_,visOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        hairGeometryInitialized_?VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,hairGeometryInitialized_?VK_ACCESS_SHADER_READ_BIT:0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        const VkImageLayout depthOld=depthInitialized_?VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,depthImage_,depthOld,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_DEPTH_BIT,
                        depthInitialized_?VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                        depthInitialized_?VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:0,VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkViewport vp{0.0f,0.0f,float(swapExtent_.width),float(swapExtent_.height),0.0f,1.0f};
        VkRect2D sc{{0,0},swapExtent_};
        VkClearValue clearColor{}; const float bg=std::clamp(studioLighting_.backgroundGray,0.0f,1.0f); clearColor.color={{bg,bg,bg,1.0f}};
        VkRenderingAttachmentInfo baseColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        baseColor.imageView=msaaColorView_; baseColor.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; baseColor.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; baseColor.storeOp=VK_ATTACHMENT_STORE_OP_STORE; baseColor.clearValue=clearColor;
        baseColor.resolveMode=VK_RESOLVE_MODE_AVERAGE_BIT; baseColor.resolveImageView=sceneColorView_; baseColor.resolveImageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkClearValue clearDepth{}; clearDepth.depthStencil={1.0f,0};
        VkRenderingAttachmentInfo sceneDepth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        sceneDepth.imageView=depthView_; sceneDepth.imageLayout=VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; sceneDepth.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; sceneDepth.storeOp=VK_ATTACHMENT_STORE_OP_STORE; sceneDepth.clearValue=clearDepth;
        VkRenderingInfo baseInfo{VK_STRUCTURE_TYPE_RENDERING_INFO}; baseInfo.renderArea.extent=swapExtent_; baseInfo.layerCount=1; baseInfo.colorAttachmentCount=1; baseInfo.pColorAttachments=&baseColor; baseInfo.pDepthAttachment=&sceneDepth;
        vkCmdBeginRendering(commandBuffer_,&baseInfo);
        vkCmdSetViewport(commandBuffer_,0,1,&vp); vkCmdSetScissor(commandBuffer_,0,1,&sc);
        const std::array<VkDescriptorSet,4> headSets{frameSet_,staticSet_,dynamicsSet_,materialSet_};
        const std::array<uint32_t,3> headDynamic{frameDynamicOffset(),simDynamicOffset(),materialDynamicOffset()};
        vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,skyPipeline_);
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,headPipelineLayout_,0,uint32_t(headSets.size()),headSets.data(),uint32_t(headDynamic.size()),headDynamic.data());
        vkCmdDraw(commandBuffer_,3,1,0,0);
        vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,headPipeline_);
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,headPipelineLayout_,0,uint32_t(headSets.size()),headSets.data(),uint32_t(headDynamic.size()),headDynamic.data());
        VkDeviceSize offset=0; vkCmdBindVertexBuffers(commandBuffer_,0,1,&headVertices_.handle,&offset); vkCmdBindIndexBuffer(commandBuffer_,headIndices_.handle,0,VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(commandBuffer_,headIndexCount_,1,0,0,0);
        vkCmdEndRendering(commandBuffer_);

        // Primary visibility pass. Covered MSAA samples carry actual strand geometry;
        // uncovered samples stay zero and later resolve to the base scene.
        VkClearValue clearVis{}; clearVis.color={{0,0,0,0}};
        std::array<VkRenderingAttachmentInfo,2> vis{};
        vis[0]={VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; vis[0].imageView=hairGeom0View_; vis[0].imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; vis[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; vis[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE; vis[0].clearValue=clearVis;
        vis[1]={VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; vis[1].imageView=hairGeom1View_; vis[1].imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; vis[1].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; vis[1].storeOp=VK_ATTACHMENT_STORE_OP_STORE; vis[1].clearValue=clearVis;
        VkRenderingAttachmentInfo hairDepth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; hairDepth.imageView=depthView_; hairDepth.imageLayout=VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; hairDepth.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; hairDepth.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingInfo visInfo{VK_STRUCTURE_TYPE_RENDERING_INFO}; visInfo.renderArea.extent=swapExtent_; visInfo.layerCount=1; visInfo.colorAttachmentCount=uint32_t(vis.size()); visInfo.pColorAttachments=vis.data(); visInfo.pDepthAttachment=&hairDepth;
        vkCmdBeginRendering(commandBuffer_,&visInfo); vkCmdSetViewport(commandBuffer_,0,1,&vp); vkCmdSetScissor(commandBuffer_,0,1,&sc);
        vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairPipeline_);
        const std::array<VkDescriptorSet,4> hairSets{frameSet_,staticSet_,dynamicsSet_,materialSet_};
        const std::array<uint32_t,3> hairDynamic{frameDynamicOffset(),simDynamicOffset(),materialDynamicOffset()};
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairPipelineLayout_,0,uint32_t(hairSets.size()),hairSets.data(),uint32_t(hairDynamic.size()),hairDynamic.data());
        cmdDrawMeshTasksEXT_(commandBuffer_,static_cast<uint32_t>(groom_?groom_->clusters.size():0u),1,1);
        vkCmdEndRendering(commandBuffer_);

        transitionImage(commandBuffer_,sceneColorImage_,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        transitionImage(commandBuffer_,hairGeom0Image_,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        transitionImage(commandBuffer_,hairGeom1Image_,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        sceneColorInitialized_=true; hairGeometryInitialized_=true; msaaColorInitialized_=true; depthInitialized_=true;

        const VkImageLayout compOld=compositeColorInitialized_?VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,compositeColorImage_,compOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        compositeColorInitialized_?VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        compositeColorInitialized_?VK_ACCESS_SHADER_READ_BIT:0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo compColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; compColor.imageView=compositeColorView_; compColor.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; compColor.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; compColor.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo compInfo{VK_STRUCTURE_TYPE_RENDERING_INFO}; compInfo.renderArea.extent=swapExtent_; compInfo.layerCount=1; compInfo.colorAttachmentCount=1; compInfo.pColorAttachments=&compColor;
        vkCmdBeginRendering(commandBuffer_,&compInfo); vkCmdSetViewport(commandBuffer_,0,1,&vp); vkCmdSetScissor(commandBuffer_,0,1,&sc);
        vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairComposePipeline_);
        const uint32_t frameOff=frameDynamicOffset(),matOff=materialDynamicOffset();
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairComposePipelineLayout_,0,1,&frameSet_,1,&frameOff);
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairComposePipelineLayout_,1,1,&hairComposeSet_,0,nullptr);
        vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,hairComposePipelineLayout_,3,1,&materialSet_,1,&matOff);
        vkCmdDraw(commandBuffer_,3,1,0,0); vkCmdEndRendering(commandBuffer_);
        transitionImage(commandBuffer_,compositeColorImage_,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        compositeColorInitialized_=true;
        if(timestampQueryPool_) vkCmdWriteTimestamp(commandBuffer_,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestampQueryPool_,timestampBase+3u);

        const VkImageLayout swapOld=swapInitialized_[imageIndex]?VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:VK_IMAGE_LAYOUT_UNDEFINED;
        transitionImage(commandBuffer_,swapImages_[imageIndex],swapOld,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_ASPECT_COLOR_BIT,
                        swapOld==VK_IMAGE_LAYOUT_UNDEFINED?VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo postColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; postColor.imageView=swapViews_[imageIndex]; postColor.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; postColor.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; postColor.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo postInfo{VK_STRUCTURE_TYPE_RENDERING_INFO}; postInfo.renderArea.extent=swapExtent_; postInfo.layerCount=1; postInfo.colorAttachmentCount=1; postInfo.pColorAttachments=&postColor;
        vkCmdBeginRendering(commandBuffer_,&postInfo); vkCmdSetViewport(commandBuffer_,0,1,&vp); vkCmdSetScissor(commandBuffer_,0,1,&sc);
        vkCmdBindPipeline(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,toneMapPipeline_); vkCmdBindDescriptorSets(commandBuffer_,VK_PIPELINE_BIND_POINT_GRAPHICS,toneMapPipelineLayout_,0,1,&toneMapSet_,0,nullptr);
        const glm::vec4 postParams(studioLighting_.exposureEV,0.0f,0.0f,0.0f); vkCmdPushConstants(commandBuffer_,toneMapPipelineLayout_,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(postParams),&postParams); vkCmdDraw(commandBuffer_,3,1,0,0); vkCmdEndRendering(commandBuffer_);
        if(screenshotRequested_&&!screenshotIncludeUi_) captureSwapchainForScreenshot(commandBuffer_,imageIndex,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if(imguiInitialized_){
            VkRenderingAttachmentInfo uiColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; uiColor.imageView=swapViews_[imageIndex]; uiColor.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; uiColor.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; uiColor.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo uiInfo{VK_STRUCTURE_TYPE_RENDERING_INFO}; uiInfo.renderArea.extent=swapExtent_; uiInfo.layerCount=1; uiInfo.colorAttachmentCount=1; uiInfo.pColorAttachments=&uiColor;
            vkCmdBeginRendering(commandBuffer_,&uiInfo); ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),commandBuffer_); vkCmdEndRendering(commandBuffer_);
        }
        if(screenshotRequested_&&screenshotIncludeUi_) captureSwapchainForScreenshot(commandBuffer_,imageIndex,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if(timestampQueryPool_) vkCmdWriteTimestamp(commandBuffer_,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestampQueryPool_,timestampBase+4u);
        transitionImage(commandBuffer_,swapImages_[imageIndex],VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,0);
        swapInitialized_[imageIndex]=true;
        vkCheck(vkEndCommandBuffer(commandBuffer_),"vkEndCommandBuffer");
    }

    void drawFrame() {
        const auto cpuFrameStart=std::chrono::steady_clock::now();
        // Two frames in flight: CPU command/uniform preparation for frame N+1
        // can overlap GPU execution of frame N.  Each frame owns its command
        // buffer, acquire/present semaphores, fence, and ring-buffer UBO slice.
        commandBuffer_=commandBuffers_[frameIndex_];
        imageAvailable_=imageAvailableSemaphores_[frameIndex_];
        renderFinished_=renderFinishedSemaphores_[frameIndex_];
        inFlightFence_=inFlightFences_[frameIndex_];
        vkCheck(vkWaitForFences(device_, 1, &inFlightFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        readBackCompletedScreenshot(frameIndex_);
        readGpuTimestamps();
        pollValidation();
        pollStyleBuild();
        if(styleIndexRequested_>=0) rebuildGroom(size_t(styleIndexRequested_));
        else if(editedGroomRebuildRequested_ && !(ImGui::GetCurrentContext() && ImGui::IsMouseDown(ImGuiMouseButton_Left))) rebuildEditedGroom();
        else if(resolveContactsRequested_) resolveCurrentContacts();

        uint32_t imageIndex = 0;
        VkResult acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_, VK_NULL_HANDLE, &imageIndex);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) vkCheck(acquire, "vkAcquireNextImageKHR");

        if (resetRequested_) requestHairReset();
        const double now = glfwGetTime();
        float dt = lastFrameTime_ > 0.0 ? static_cast<float>(now - lastFrameTime_) : 1.0f / 60.0f;
        lastFrameTime_ = now;
        dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 30.0f);
        elapsed_ += dt;
        simStepsThisFrame_ = 0u;
        if (editorState_.mode==vh::editor::StudioMode::Physics && physicsAvailableNow()) {
            simAccumulator_ = std::min(simAccumulator_ + dt, 4.0f * kFixedSimDt_);
            while (simAccumulator_ >= kFixedSimDt_ && simStepsThisFrame_ < 4u) {
                simAccumulator_ -= kFixedSimDt_;
                ++simStepsThisFrame_;
            }
        } else {
            simAccumulator_ = 0.0f;
        }

        // Optical density is a spatially integrated low-frequency field, so it
        // does not need the 120 Hz mechanics cadence.  Beauty updates at 30 Hz,
        // Performance at 20 Hz, Reference at 60 Hz; authored/style changes still
        // force an immediate rebuild.
        opticalUpdateThisFrame_=false;
        if(groom_) {
            const float opticalHz=referenceQuality_?60.0f:(performanceQuality_?20.0f:30.0f);
            const float opticalPeriod=1.0f/opticalHz;
            if(opticalFieldDirty_) {
                opticalUpdateThisFrame_=true;
                opticalUpdateAccumulator_=0.0f;
            } else if(simStepsThisFrame_>0u) {
                opticalUpdateAccumulator_+=dt;
                if(opticalUpdateAccumulator_>=opticalPeriod) {
                    opticalUpdateThisFrame_=true;
                    opticalUpdateAccumulator_=std::fmod(opticalUpdateAccumulator_,opticalPeriod);
                }
            } else {
                opticalUpdateAccumulator_=0.0f;
            }
        }

        domUpdateThisFrame_=false;
        if(groom_) {
            const float domHz=referenceQuality_?120.0f:(performanceQuality_?30.0f:60.0f);
            const float domPeriod=1.0f/domHz;
            if(domFieldDirty_){domUpdateThisFrame_=true;domUpdateAccumulator_=0.0f;}
            else if(simStepsThisFrame_>0u){domUpdateAccumulator_+=dt;if(domUpdateAccumulator_>=domPeriod){domUpdateThisFrame_=true;domUpdateAccumulator_=std::fmod(domUpdateAccumulator_,domPeriod);}}
            else domUpdateAccumulator_=0.0f;
        }

        updateUniforms(dt);
        beginEditorFrame();
        if(screenshotRequested_) ensureScreenshotReadbackBuffer();
        recordCommandBuffer(imageIndex);

        vkCheck(vkResetFences(device_, 1, &inFlightFence_), "vkResetFences");
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable_;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_;
        vkCheck(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlightFence_), "vkQueueSubmit");
        if(timestampQueryPool_) timestampRecorded_[frameIndex_]=true;

        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished_;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &imageIndex;
        VkResult pr = vkQueuePresentKHR(presentQueue_, &present);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || framebufferResized_) {
            recreateSwapchain();
        } else if (pr != VK_SUCCESS) {
            vkCheck(pr, "vkQueuePresentKHR");
        }
        cpuFrameMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cpuFrameStart).count();
        frameIndex_=(frameIndex_+1u)%kFramesInFlight_;
    }

    void initVulkan() {
        const auto startupT0=std::chrono::steady_clock::now();
        auto previous=startupT0;
        auto phase=[&](const char* name) {
            const auto now=std::chrono::steady_clock::now();
            const double delta=std::chrono::duration<double,std::milli>(now-previous).count();
            const double total=std::chrono::duration<double,std::milli>(now-startupT0).count();
            previous=now;
            std::cout << "Startup: " << name << " | +" << delta << " ms | total=" << total << " ms\n" << std::flush;
        };
        createInstance();
        vkCheck(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_), "glfwCreateWindowSurface");
        pickPhysicalDevice();
        createDevice();
        createPipelineCache(); phase("device ready");
        createSwapchain();
        createSceneColorResources();
        createDepthResources();
        createCommandResources(); phase("swapchain/resources ready");
        // Pipeline compilation and CPU character/contact preparation are independent.
        // Build the driver objects while the background CPU job finishes.
        createDescriptorLayouts();
        createComputePipelines();
        createGraphicsPipelines(); phase("pipelines ready");
        createSceneBuffers(); phase("character + groom runtime ready");
        createDescriptorPoolAndSets();
        initImGui(); phase("UI ready");
        std::cout << "Controls: Display/Edit/Physics | Hairstyle Library in UI | LMB orbit outside panels | wheel zoom | Space wind | R reset\n";
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void cleanup() {
        if(styleBuildJob_.valid()) { try { styleBuildJob_.wait(); styleBuildJob_.get(); } catch(...) {} }
        if(validationJob_.valid()) { try { validationJob_.wait(); (void)validationJob_.get(); } catch(...) {} }
        queuedValidationAsset_.reset();
        if (device_) vkDeviceWaitIdle(device_);
        if (imguiInitialized_) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            imguiInitialized_ = false;
        }
        if (domPrefixPipeline_) vkDestroyPipeline(device_, domPrefixPipeline_, nullptr);
        if (domSplatPipeline_) vkDestroyPipeline(device_, domSplatPipeline_, nullptr);
        if (opticalDepthPipeline_) vkDestroyPipeline(device_, opticalDepthPipeline_, nullptr);
        if (opticalSplatPipeline_) vkDestroyPipeline(device_, opticalSplatPipeline_, nullptr);
        if (volumeG2PPipeline_) vkDestroyPipeline(device_, volumeG2PPipeline_, nullptr);
        if (volumeSolvePipeline_) vkDestroyPipeline(device_, volumeSolvePipeline_, nullptr);
        if (volumeP2GPipeline_) vkDestroyPipeline(device_, volumeP2GPipeline_, nullptr);
        if (volumeClearPipeline_) vkDestroyPipeline(device_, volumeClearPipeline_, nullptr);
        if (simPipeline_) vkDestroyPipeline(device_, simPipeline_, nullptr);
        destroySwapchainResources();
        destroyBuffer(headIndices_);
        destroyBuffer(headVertices_);
        destroyGroomBuffers();
        destroyBuffer(volumeState_);
        destroyBuffer(volumeAccum_);
        destroyBuffer(bodyField_);
        destroyBuffer(simBuffer_);
        destroyBuffer(materialBuffer_);
        destroyBuffer(frameBuffer_);
        destroyBuffer(screenshotReadback_);
        if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        if (toneMapPipelineLayout_) vkDestroyPipelineLayout(device_, toneMapPipelineLayout_, nullptr);
        if (hairComposePipelineLayout_) vkDestroyPipelineLayout(device_, hairComposePipelineLayout_, nullptr);
        if (hairPipelineLayout_) vkDestroyPipelineLayout(device_, hairPipelineLayout_, nullptr);
        if (headPipelineLayout_) vkDestroyPipelineLayout(device_, headPipelineLayout_, nullptr);
        if (toneMapSetLayout_) vkDestroyDescriptorSetLayout(device_, toneMapSetLayout_, nullptr);
        if (hairComposeSetLayout_) vkDestroyDescriptorSetLayout(device_, hairComposeSetLayout_, nullptr);
        if (materialSetLayout_) vkDestroyDescriptorSetLayout(device_, materialSetLayout_, nullptr);
        if (dynamicsSetLayout_) vkDestroyDescriptorSetLayout(device_, dynamicsSetLayout_, nullptr);
        if (staticSetLayout_) vkDestroyDescriptorSetLayout(device_, staticSetLayout_, nullptr);
        if (frameSetLayout_) vkDestroyDescriptorSetLayout(device_, frameSetLayout_, nullptr);
        if (timestampQueryPool_) vkDestroyQueryPool(device_, timestampQueryPool_, nullptr);
        for(uint32_t i=0;i<kFramesInFlight_;++i){
            if (inFlightFences_[i]) vkDestroyFence(device_, inFlightFences_[i], nullptr);
            if (renderFinishedSemaphores_[i]) vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
            if (imageAvailableSemaphores_[i]) vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
        }
        if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
        if (sceneSampler_) vkDestroySampler(device_, sceneSampler_, nullptr);
        savePipelineCache();
        if (pipelineCache_) vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        if (device_) vkDestroyDevice(device_, nullptr);
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

} // namespace

int main(int argc, char** argv) {
    auto writeFatalRecord = [](const std::string& message) {
        try {
            std::filesystem::create_directories("logs");
            std::ofstream out(std::filesystem::path("logs") / "last_fatal.txt", std::ios::trunc);
            if (out) out << "Vulkan Hair Studio fatal error\n" << message << "\n";
        } catch (...) {
            // Diagnostics must never mask the original failure.
        }
    };

    try {
        auto app=std::make_unique<HairApp>();
        app->run(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        writeFatalRecord(e.what());
        std::cerr << "\n[FATAL] " << e.what() << "\n";
#ifdef _WIN32
        std::cerr << "The error was saved to logs\\last_fatal.txt.\n";
#endif
        return 1;
    } catch (...) {
        writeFatalRecord("Unknown non-std exception");
        std::cerr << "\n[FATAL] Unknown non-std exception\n";
        return 1;
    }
}
