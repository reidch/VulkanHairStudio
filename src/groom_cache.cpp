#include "groom_cache.hpp"

#include <chrono>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace vh::groom {
namespace {

constexpr uint32_t kCacheVersion=2u;
constexpr char kMagic[8]={'V','3','9','G','R','M','0','1'};

struct CacheHeader {
    char magic[8]{};
    uint32_t version=0u;
    uint32_t renderStrands=0u;
    uint32_t guidePoints=0u;
    uint32_t simulationGuides=0u;
    uint32_t simulationGuideBytes=0u;
    uint64_t styleFingerprint=0u;
    uint64_t meshFingerprint=0u;
    float follicleClearance=0.0f;
    float minStrandClearance=0.0f;
    float fiberDiameter=0.0f;
    float collisionMargin=0.0f;
    float globalLengthScale=0.0f;
    float globalVolumeScale=0.0f;
    float microDetailScale=0.0f;
    uint32_t seed=0u;
    uint32_t reserved=0u;
};

bool sameFloat(float a,float b) {
    uint32_t x=0u,y=0u; std::memcpy(&x,&a,sizeof(x)); std::memcpy(&y,&b,sizeof(y)); return x==y;
}

bool headerMatches(const CacheHeader& h,const DerivedGroomCacheKey& key) {
    return std::memcmp(h.magic,kMagic,sizeof(kMagic))==0 && h.version==kCacheVersion &&
           h.renderStrands==kRenderStrandCount && h.guidePoints==kGuidePoints &&
           h.simulationGuides==kSimulationGuideCount && h.simulationGuideBytes==sizeof(SimulationGuideCpu) &&
           h.styleFingerprint==key.styleFingerprint && h.meshFingerprint==key.meshFingerprint &&
           sameFloat(h.follicleClearance,key.params.follicleClearance) &&
           sameFloat(h.minStrandClearance,key.params.minStrandClearance) &&
           sameFloat(h.fiberDiameter,key.params.fiberDiameter) &&
           sameFloat(h.collisionMargin,key.params.collisionMargin) &&
           sameFloat(h.globalLengthScale,key.params.globalLengthScale) &&
           sameFloat(h.globalVolumeScale,key.params.globalVolumeScale) &&
           sameFloat(h.microDetailScale,key.params.microDetailScale) && h.seed==key.params.seed;
}

CacheHeader makeHeader(const DerivedGroomCacheKey& key) {
    CacheHeader h{}; std::memcpy(h.magic,kMagic,sizeof(kMagic));
    h.version=kCacheVersion; h.renderStrands=kRenderStrandCount; h.guidePoints=kGuidePoints;
    h.simulationGuides=kSimulationGuideCount; h.simulationGuideBytes=uint32_t(sizeof(SimulationGuideCpu));
    h.styleFingerprint=key.styleFingerprint; h.meshFingerprint=key.meshFingerprint;
    h.follicleClearance=key.params.follicleClearance; h.minStrandClearance=key.params.minStrandClearance;
    h.fiberDiameter=key.params.fiberDiameter; h.collisionMargin=key.params.collisionMargin;
    h.globalLengthScale=key.params.globalLengthScale; h.globalVolumeScale=key.params.globalVolumeScale;
    h.microDetailScale=key.params.microDetailScale; h.seed=key.params.seed;
    return h;
}

template<class T>
void writeVector(std::ofstream& f,const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const uint64_t n=uint64_t(v.size()); f.write(reinterpret_cast<const char*>(&n),sizeof(n));
    if(n) f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(n*sizeof(T)));
    if(!f) throw std::runtime_error("Derived groom cache write failed");
}

template<class T>
bool readVector(std::ifstream& f,std::vector<T>& v,uint64_t expected) {
    static_assert(std::is_trivially_copyable_v<T>);
    uint64_t n=0u; f.read(reinterpret_cast<char*>(&n),sizeof(n));
    if(!f || n!=expected) return false;
    v.resize(size_t(n));
    if(n) f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T)));
    return bool(f);
}

bool readAsset(std::ifstream& f,GroomAsset& g) {
    constexpr uint64_t dense=uint64_t(kRenderStrandCount)*kGuidePoints;
    constexpr uint64_t guidePoints=uint64_t(kSimulationGuideCount)*kGuidePoints;
    constexpr uint64_t clusters=(kRenderStrandCount+kClusterSize-1u)/kClusterSize;
    if(!readVector(f,g.baseStrandPoints,dense)) return false;
    if(!readVector(f,g.restStrandPoints,dense)) return false;
    if(!readVector(f,g.strandPathU,dense)) return false;
    if(!readVector(f,g.strandContactGroupTags,kRenderStrandCount)) return false;
    if(!readVector(f,g.strandTopology,kRenderStrandCount)) return false;
    if(!readVector(f,g.guideSourceStrands,kSimulationGuideCount)) return false;
    if(!readVector(f,g.simulationGuides,kSimulationGuideCount)) return false;
    if(!readVector(f,g.restGuidePoints,guidePoints)) return false;
    if(!readVector(f,g.restGuideFrames,guidePoints)) return false;
    if(!readVector(f,g.guidePointMeta,guidePoints)) return false;
    if(!readVector(f,g.strands,kRenderStrandCount)) return false;
    if(!readVector(f,g.clusters,clusters)) return false;
    f.read(reinterpret_cast<char*>(&g.audit),sizeof(g.audit));
    return bool(f);
}

void writeAsset(std::ofstream& f,const GroomAsset& g) {
    writeVector(f,g.baseStrandPoints); writeVector(f,g.restStrandPoints); writeVector(f,g.strandPathU);
    writeVector(f,g.strandContactGroupTags); writeVector(f,g.strandTopology); writeVector(f,g.guideSourceStrands);
    writeVector(f,g.simulationGuides); writeVector(f,g.restGuidePoints); writeVector(f,g.restGuideFrames);
    writeVector(f,g.guidePointMeta); writeVector(f,g.strands); writeVector(f,g.clusters);
    static_assert(std::is_trivially_copyable_v<GroomAudit>);
    f.write(reinterpret_cast<const char*>(&g.audit),sizeof(g.audit));
    if(!f) throw std::runtime_error("Derived groom cache write failed");
}

} // namespace

uint64_t fingerprintFile(const std::filesystem::path& file) {
    std::ifstream f(file,std::ios::binary);
    if(!f) throw std::runtime_error("Cannot fingerprint groom asset: "+file.string());
    uint64_t h=1469598103934665603ull;
    std::array<char,64*1024> block{};
    while(f) {
        f.read(block.data(),std::streamsize(block.size()));
        const auto n=f.gcount();
        for(std::streamsize i=0;i<n;++i) { h^=uint8_t(block[size_t(i)]); h*=1099511628211ull; }
    }
    return h;
}

bool loadDerivedGroom(const std::filesystem::path& file,const DerivedGroomCacheKey& key,GroomAsset& out,double* loadMs) {
    const auto t0=std::chrono::steady_clock::now();
    std::ifstream f(file,std::ios::binary);
    if(!f) return false;
    CacheHeader h{}; f.read(reinterpret_cast<char*>(&h),sizeof(h));
    if(!f || !headerMatches(h,key)) return false;
    GroomAsset candidate;
    if(!readAsset(f,candidate)) return false;
    out=std::move(candidate);
    if(loadMs) *loadMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    return true;
}

void saveDerivedGroom(const std::filesystem::path& file,const DerivedGroomCacheKey& key,const GroomAsset& groom,double* saveMs) {
    const auto t0=std::chrono::steady_clock::now();
    std::error_code ec; std::filesystem::create_directories(file.parent_path(),ec);
    const auto tmp=std::filesystem::path(file.string()+".tmp");
    {
        std::ofstream f(tmp,std::ios::binary|std::ios::trunc);
        if(!f) return; // cache is an optimization, never a runtime dependency
        const CacheHeader h=makeHeader(key); f.write(reinterpret_cast<const char*>(&h),sizeof(h));
        if(!f) return;
        try { writeAsset(f,groom); } catch(...) { f.close(); std::filesystem::remove(tmp,ec); return; }
    }
    std::filesystem::remove(file,ec); ec.clear();
    std::filesystem::rename(tmp,file,ec);
    if(ec) std::filesystem::remove(tmp,ec);
    if(saveMs) *saveMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
}

} // namespace vh::groom
