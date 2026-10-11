#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace vh::character {

struct Vertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f,1.0f,0.0f};
};

struct Triangle {
    glm::uvec3 index{0u};
    bool collision = true;
    bool visible = true;
    bool scalp = false;
    bool solidBoundary = false; // external Face/BackHead shell only
};

struct RootBinding {
    uint32_t triangle = 0u;
    glm::vec3 barycentric{1.0f,0.0f,0.0f};
};

struct CharacterFrame {
    glm::vec3 halfExtent{0.090f,0.105f,0.095f};
    glm::mat3 worldFromCanonical{1.0f};
};

struct SurfaceContact {
    float distance = 1.0e3f;      // negative only inside the external body solid
    glm::vec3 normal{0.0f,1.0f,0.0f}; // direction that increases clearance
    uint32_t triangle = 0u;
    bool insideSolid = false;
    bool nearestIsObstacle = false;
};

// Runtime acceleration only. Authoring and validation use CharacterSurface::contact().
struct ContactField {
    uint32_t dim = 0u;
    glm::vec3 origin{0.0f};
    float cellSize = 0.0f;
    std::vector<float> distance;

    bool valid() const { return dim >= 4u && distance.size() == size_t(dim)*dim*dim && cellSize > 0.0f; }
    bool contains(glm::vec3 p) const;
    float sample(glm::vec3 p) const;
    glm::vec3 gradient(glm::vec3 p) const;
};

struct RayBindingResult {
    RootBinding binding{};
    bool exactRayHit = false;
};

class CharacterSurface {
public:
    // The complete ICT OBJ is preserved for rendering and collision. Only solid
    // sign authority is restricted to the external M_Face/M_BackHead shell;
    // all other components remain two-sided collision obstacles.
    static CharacterSurface loadObj(const std::filesystem::path& objPath,
                                    const std::filesystem::path& contactFieldCachePath,
                                    float targetCellSize = 0.0020f);

    const std::vector<Vertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return renderIndices_; }
    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<uint32_t>& scalpTriangles() const { return scalpTriangles_; }
    const CharacterFrame& frame() const { return frame_; }
    const ContactField& contactField() const { return contactField_; }
    // Transitional alias for renderer/descriptor code; it is not an authoring oracle.
    const ContactField& sdf() const { return contactField_; }

    glm::vec3 position(const RootBinding& binding) const;
    glm::vec3 normal(const RootBinding& binding) const;

    SurfaceContact contact(glm::vec3 p) const;
    uint64_t meshFingerprint() const { return meshFingerprint_; }

    RayBindingResult bindScalpDirection(glm::vec3 canonicalDirection) const;

private:
    struct BvhNode {
        glm::vec3 bmin{0.0f};
        glm::vec3 bmax{0.0f};
        uint32_t left = 0u;
        uint32_t right = 0u;
        uint32_t first = 0u;
        uint32_t count = 0u;
        bool leaf() const { return count != 0u; }
    };

    std::vector<Vertex> vertices_;
    std::vector<Triangle> triangles_;
    std::vector<uint32_t> renderIndices_;
    std::vector<uint32_t> scalpTriangles_;
    std::vector<uint32_t> collisionTriangles_;
    std::vector<uint32_t> solidTriangles_;

    std::vector<uint32_t> bvhOrder_;
    std::vector<BvhNode> bvh_;
    std::vector<uint32_t> solidBvhOrder_;
    std::vector<BvhNode> solidBvh_;

    CharacterFrame frame_{};
    ContactField contactField_{};
    uint64_t meshFingerprint_ = 0u;

    void finalize();
    void buildBvhs();
    uint32_t buildBvhNode(std::vector<uint32_t>& order, std::vector<BvhNode>& nodes,
                          uint32_t first, uint32_t count);
    float nearestTriangle(glm::vec3 p, const std::vector<uint32_t>& order,
                          const std::vector<BvhNode>& nodes,
                          uint32_t& tri, glm::vec3& closest, glm::vec3& normal) const;
    bool raycast(glm::vec3 origin, glm::vec3 direction, uint32_t& tri, glm::vec3& bary, float& t) const;
    uint32_t countSolidRayCrossings(glm::vec3 origin, glm::vec3 direction) const;
    bool insideSolid(glm::vec3 p) const;
    void loadOrBuildContactField(const std::filesystem::path& cachePath, float targetCellSize);
    uint64_t computeMeshFingerprint() const;
};

} // namespace vh::character
