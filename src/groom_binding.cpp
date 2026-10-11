#include "groom_binding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vh::groom {
namespace {

constexpr uint32_t kBindingCacheVersion=1u;

template<class F>
void parallelFor(uint32_t count,F&& fn) {
    const uint32_t hw=std::max(1u,std::thread::hardware_concurrency());
    const uint32_t budget=hw>2u?hw-2u:1u;
    const uint32_t workers=std::min(budget,std::max(1u,(count+8191u)/8192u));
    if(workers<=1u) { for(uint32_t i=0;i<count;++i) fn(i); return; }
    std::vector<std::thread> pool; pool.reserve(workers);
    for(uint32_t w=0;w<workers;++w) {
        const uint32_t begin=uint32_t((uint64_t(count)*w)/workers);
        const uint32_t end=uint32_t((uint64_t(count)*(w+1u))/workers);
        pool.emplace_back([=,&fn]{ for(uint32_t i=begin;i<end;++i) fn(i); });
    }
    for(auto& t:pool) t.join();
}
struct BindingCacheHeader {
    char magic[8]{};
    uint32_t version=0u;
    uint32_t rootCount=0u;
    uint32_t seed=0u;
    uint32_t fallbackBindings=0u;
    uint64_t meshFingerprint=0u;
};
struct BindingCacheRecord { uint32_t triangle=0u; float bary[3]{1.0f,0.0f,0.0f}; };

GroomPlacement placementFor(const vh::character::CharacterSurface& character) {
    GroomPlacement out;
    const auto& cf=character.frame();
    const float sx=cf.halfExtent.x/kCanonicalHeadRadii.x;
    const float sz=cf.halfExtent.z/kCanonicalHeadRadii.z;
    out.origin=glm::vec3(0.0f);
    out.worldFromCanonical=cf.worldFromCanonical;
    out.uniformScale=std::clamp(std::sqrt(std::max(1e-6f,sx*sz)),0.35f,1.50f);
    return out;
}

float scalpArea(const vh::character::CharacterSurface& character) {
    double area=0.0;
    const auto& v=character.vertices();
    const auto& t=character.triangles();
    for(uint32_t ti:character.scalpTriangles()) {
        if(ti>=t.size()) continue;
        const auto& tri=t[ti];
        const glm::vec3 a=v[tri.index.x].position;
        const glm::vec3 b=v[tri.index.y].position;
        const glm::vec3 c=v[tri.index.z].position;
        area += 0.5*double(glm::length(glm::cross(b-a,c-a)));
    }
    return float(area);
}

uint32_t part1By1(uint32_t x) {
    x&=0x0000ffffu;
    x=(x|(x<<8u))&0x00ff00ffu;
    x=(x|(x<<4u))&0x0f0f0f0fu;
    x=(x|(x<<2u))&0x33333333u;
    x=(x|(x<<1u))&0x55555555u;
    return x;
}
uint32_t morton2(float u,float v) {
    const uint32_t x=std::min(65535u,uint32_t(std::clamp(u,0.0f,0.999999f)*65536.0f));
    const uint32_t y=std::min(65535u,uint32_t(std::clamp(v,0.0f,0.999999f)*65536.0f));
    return part1By1(x)|(part1By1(y)<<1u);
}

void finalizeDerived(const vh::character::CharacterSurface& character,uint32_t seed,GroomBinding& out) {
    out.placement=placementFor(character);
    out.audit.rootCount=static_cast<uint32_t>(out.roots.size());
    out.audit.placementScale=out.placement.uniformScale;
    out.audit.scalpArea=scalpArea(character);
    out.renderOrder.resize(out.roots.size());
    std::iota(out.renderOrder.begin(),out.renderOrder.end(),0u);
    std::sort(out.renderOrder.begin(),out.renderOrder.end(),[&](uint32_t a,uint32_t b){
        const uint32_t ma=morton2(out.roots[a].canonical.uv.x,out.roots[a].canonical.uv.y);
        const uint32_t mb=morton2(out.roots[b].canonical.uv.x,out.roots[b].canonical.uv.y);
        return ma==mb?a<b:ma<mb;
    });

    // Cheap deterministic spacing audit only. It never changes the binding.
    const uint32_t stride=64u;
    std::vector<uint32_t> ids; ids.reserve((out.roots.size()+stride-1u)/stride);
    for(uint32_t i=0;i<out.roots.size();i+=stride) ids.push_back(i);
    float minD=std::numeric_limits<float>::max(),maxNearest=0.0f;
    for(size_t a=0;a<ids.size();++a) {
        float nearest=std::numeric_limits<float>::max();
        const auto& ra=out.roots[ids[a]];
        const size_t begin=a>24?a-24:0, end=std::min(ids.size(),a+25);
        for(size_t b=begin;b<end;++b) if(a!=b)
            nearest=std::min(nearest,glm::length(ra.position-out.roots[ids[b]].position));
        if(std::isfinite(nearest)) { minD=std::min(minD,nearest); maxNearest=std::max(maxNearest,nearest); }
    }
    out.audit.minSampledRootSpacing=std::isfinite(minD)?minD:0.0f;
    out.audit.maxSampledRootSpacing=maxNearest;
    (void)seed;
}

bool loadCache(const vh::character::CharacterSurface& character,const std::filesystem::path& path,
               uint32_t rootCount,uint32_t seed,GroomBinding& out) {
    std::ifstream f(path,std::ios::binary);
    if(!f) return false;
    BindingCacheHeader h{}; f.read(reinterpret_cast<char*>(&h),sizeof(h));
    if(!f || std::memcmp(h.magic,"V37BIND",7)!=0 || h.version!=kBindingCacheVersion ||
       h.rootCount!=rootCount || h.seed!=seed || h.meshFingerprint!=character.meshFingerprint()) return false;
    std::vector<BindingCacheRecord> rec(rootCount);
    f.read(reinterpret_cast<char*>(rec.data()),std::streamsize(rec.size()*sizeof(BindingCacheRecord)));
    if(!f) return false;
    out.roots.resize(rootCount);
    out.audit.fallbackBindings=h.fallbackBindings;
    out.audit.fallbackFraction=rootCount?float(h.fallbackBindings)/float(rootCount):0.0f;
    for(uint32_t i=0;i<rootCount;++i) {
        BoundRoot r;
        r.canonical=vh::scalp::sample(i,seed);
        r.binding.triangle=rec[i].triangle;
        r.binding.barycentric=glm::vec3(rec[i].bary[0],rec[i].bary[1],rec[i].bary[2]);
        if(r.binding.triangle>=character.triangles().size()) return false;
        r.position=character.position(r.binding);
        r.normal=character.normal(r.binding);
        out.roots[i]=r;
    }
    finalizeDerived(character,seed,out);
    return true;
}

void saveCache(const vh::character::CharacterSurface& character,const std::filesystem::path& path,
               uint32_t seed,const GroomBinding& binding) {
    if(path.empty()) return;
    std::error_code ec; std::filesystem::create_directories(path.parent_path(),ec);
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    if(!f) return;
    BindingCacheHeader h{};
    std::memcpy(h.magic,"V37BIND",7); h.version=kBindingCacheVersion;
    h.rootCount=static_cast<uint32_t>(binding.roots.size()); h.seed=seed;
    h.fallbackBindings=binding.audit.fallbackBindings; h.meshFingerprint=character.meshFingerprint();
    f.write(reinterpret_cast<const char*>(&h),sizeof(h));
    for(const auto& r:binding.roots) {
        BindingCacheRecord q{}; q.triangle=r.binding.triangle;
        q.bary[0]=r.binding.barycentric.x; q.bary[1]=r.binding.barycentric.y; q.bary[2]=r.binding.barycentric.z;
        f.write(reinterpret_cast<const char*>(&q),sizeof(q));
    }
}

} // namespace

GroomBinding GroomRetargeter::bind(const vh::character::CharacterSurface& character,
                                   uint32_t rootCount,
                                   uint32_t seed) const {
    if(rootCount==0u) throw std::runtime_error("GroomRetargeter requires at least one root");
    GroomBinding out;
    out.roots.resize(rootCount);
    out.placement=placementFor(character);
    out.audit.rootCount=rootCount;
    out.audit.placementScale=out.placement.uniformScale;

    std::vector<uint8_t> fallback(rootCount,0u),invalid(rootCount,0u);
    parallelFor(rootCount,[&](uint32_t i) {
        BoundRoot r;
        r.canonical=vh::scalp::sample(i,seed);
        const auto hit=character.bindScalpDirection(r.canonical.direction);
        r.binding=hit.binding;
        fallback[i]=hit.exactRayHit?0u:1u;
        if(r.binding.triangle>=character.triangles().size()) { invalid[i]=1u; return; }
        r.position=character.position(r.binding);
        r.normal=character.normal(r.binding);
        out.roots[i]=r;
    });
    out.audit.fallbackBindings=uint32_t(std::count(fallback.begin(),fallback.end(),uint8_t(1u)));
    out.audit.invalidBindings=uint32_t(std::count(invalid.begin(),invalid.end(),uint8_t(1u)));
    out.audit.fallbackFraction=float(out.audit.fallbackBindings)/float(rootCount);
    if(out.audit.invalidBindings!=0u) throw std::runtime_error("Character retarget produced invalid root bindings");
    if(out.audit.fallbackFraction>0.01f)
        throw std::runtime_error("Character scalp correspondence is incomplete (>1% fallback roots). Check head orientation/topology.");
    finalizeDerived(character,seed,out);
    return out;
}

GroomBinding GroomRetargeter::loadOrBind(const vh::character::CharacterSurface& character,
                                         const std::filesystem::path& cachePath,
                                         uint32_t rootCount,
                                         uint32_t seed) const {
    GroomBinding out;
    if(loadCache(character,cachePath,rootCount,seed,out)) return out;
    out=bind(character,rootCount,seed);
    saveCache(character,cachePath,seed,out);
    return out;
}

} // namespace vh::groom
