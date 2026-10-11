#include "groom_builder.hpp"
#include "groom_space.hpp"
#include "groom_style.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <thread>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace vh::groom {
namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t kGridPhi = 64u;
constexpr uint32_t kGridTheta = 32u;


template<class F>
void parallelFor(uint32_t count,F&& fn) {
    const uint32_t hw=std::max(1u,std::thread::hardware_concurrency());
    // Authoring runs beside the interactive main/GPU-feed thread and possibly
    // one exact-validation worker. Reserve CPU capacity instead of saturating
    // every hardware thread with a memory-bandwidth-bound dense build.
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

float saturate(float x) { return std::clamp(x, 0.0f, 1.0f); }
float smooth01(float x) { x = saturate(x); return x*x*(3.0f-2.0f*x); }

uint32_t hash32(uint32_t x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}
float hash01(uint32_t x) { return float(hash32(x) & 0x00ffffffu) / float(0x01000000u); }

glm::vec3 safeNorm(glm::vec3 v,glm::vec3 fallback) {
    const float l2=glm::dot(v,v);
    return l2>1e-12f ? v/std::sqrt(l2) : fallback;
}

glm::vec3 rotateFromTo(glm::vec3 x,glm::vec3 from,glm::vec3 to) {
    const glm::vec3 a=safeNorm(from,{0,1,0});
    const glm::vec3 b=safeNorm(to,{0,1,0});
    const float c=std::clamp(glm::dot(a,b),-1.0f,1.0f);
    if(c>0.9999f) return x;
    if(c<-0.9999f) {
        const glm::vec3 ref=std::abs(a.y)<0.8f?glm::vec3(0,1,0):glm::vec3(1,0,0);
        const glm::vec3 axis=safeNorm(glm::cross(a,ref),{0,0,1});
        return 2.0f*axis*glm::dot(axis,x)-x;
    }
    const glm::vec3 v=glm::cross(a,b);
    const float inv=1.0f/(1.0f+c);
    return x+glm::cross(v,x)+glm::cross(v,glm::cross(v,x))*inv;
}

glm::vec3 tangentProject(glm::vec3 v,glm::vec3 n,glm::vec3 fallback) {
    return safeNorm(v-n*glm::dot(v,n),fallback);
}

void validateGraph(const GroomStyle& style) {
    const auto& g=style.graph;
    if(g.sweepKeys.empty() || g.curves.empty() || g.branches.empty())
        throw std::runtime_error("Groom asset must contain sweep keys, curves and root branches");
    for(const auto& key:g.sweepKeys) {
        if(glm::length(key.axisUD)<1e-5f || glm::length(key.axisVD)<1e-5f)
            throw std::runtime_error("SweepKey cross-section axis is degenerate");
        if(key.attachment<0.0f || key.attachment>1.0f)
            throw std::runtime_error("SweepKey attachment must be in [0,1]");
    }
    for(const auto& c:g.curves) {
        if(c.keys.size()<2u) throw std::runtime_error("Every StyleCurve needs at least two sweep keys");
        if(c.keys.size()+2u>16u) throw std::runtime_error("StyleCurve exceeds the fixed authoring control budget (14 sweep keys)");
        for(SweepKeyId kid:c.keys) if(kid>=g.sweepKeys.size())
            throw std::runtime_error("StyleCurve references missing SweepKey");
    }
    for(const auto& b:g.branches) {
        if(b.curve>=g.curves.size()) throw std::runtime_error("GroomBranch references missing StyleCurve");
        if(g.curves[b.curve].topology!=b.topology) throw std::runtime_error("Branch topology must match its curve topology");
        auto validateLevel=[](const ClumpLevel& c) {
            return c.cellsU>=1.0f && c.cellsV>=1.0f && c.siteJitter>=0.0f && c.siteJitter<0.5f &&
                   c.rootTightness>=0.0f && c.rootTightness<1.0f && c.midTightness>=0.0f && c.midTightness<1.0f &&
                   c.tipTightness>=0.0f && c.tipTightness<1.0f && c.crossover>=0.0f && c.crossover<0.5f &&
                   c.drift>=0.0f && c.waveAmplitude>=0.0f && c.waveCycles>=0.0f;
        };
        if(!validateLevel(b.clumps.primary) || !validateLevel(b.clumps.secondary))
            throw std::runtime_error("ClumpHierarchy is invalid");
        if(b.cut.minLength<=0.1f || b.cut.minLength>1.0f)
            throw std::runtime_error("CutSurface is invalid");
        for(float v:b.cut.controls) if(v<=0.1f || v>1.20f)
            throw std::runtime_error("CutSurface control is outside (0.1, 1.2]");
        for(const auto& h:b.styleWarps) {
            if(glm::any(glm::lessThanEqual(h.radius,glm::vec2(0.0f))) || h.weight<0.0f || h.lengthScale<=0.0f)
                throw std::runtime_error("StyleWarp is invalid");
        }
        if(b.texture.amplitude<0.0f || b.texture.cycles<0.0f || b.texture.start<0.0f || b.texture.start>0.95f)
            throw std::runtime_error("TextureProfile is invalid");
        if(glm::any(glm::lessThanEqual(b.root.materialExtent,glm::vec2(0.0f))))
            throw std::runtime_error("RootPatch material extent must be positive");
    }
}

const GroomBranch& assignBranch(const GroomStyle& style,const vh::scalp::Sample& canonicalRoot) {
    const GroomBranch* best=nullptr;
    float bestScore=-std::numeric_limits<float>::infinity();
    for(const auto& branch:style.graph.branches) {
        const float score=rootPatchScore(branch.root,canonicalRoot);
        if(score>bestScore) { bestScore=score; best=&branch; }
    }
    if(!best) throw std::runtime_error("No root branch available for follicle");
    return *best;
}

glm::vec2 materialCoordinate(const GroomBranch& branch,const vh::scalp::Sample& canonicalRoot) {
    return branch.materialOffset+branch.materialScale*rootPatchCoordinate(branch.root,canonicalRoot);
}

struct SiteHit {
    glm::vec2 center{0.0f};
    glm::vec2 secondCenter{0.0f};
    uint32_t key=0u;
    uint32_t secondKey=0u;
    float d2=1e30f;
    float secondD2=1e30f;
};

SiteHit nearestClumpSites(const ClumpLevel& level,glm::vec2 eta,uint32_t branchIndex,uint32_t seed,uint32_t levelSalt) {
    const glm::vec2 cell{2.70f/std::max(1.0f,level.cellsU),2.70f/std::max(1.0f,level.cellsV)};
    const glm::vec2 shifted=(eta+glm::vec2(1.35f))/cell;
    const int baseX=int(std::floor(shifted.x));
    const int baseY=int(std::floor(shifted.y));
    SiteHit hit;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int cy=baseY+dy,cx=baseX+dx;
        const float rowShift=(std::abs(cy)%2)?0.5f:0.0f;
        const uint32_t ux=static_cast<uint32_t>(cx),uy=static_cast<uint32_t>(cy);
        const uint32_t key=hash32(seed^levelSalt^(branchIndex*0x9e3779b9u)^(ux*0x85ebca6bu)^(uy*0xc2b2ae35u));
        glm::vec2 center{(float(cx)+0.5f+rowShift)*cell.x-1.35f,(float(cy)+0.5f)*cell.y-1.35f};
        const glm::vec2 jitter{2.0f*hash01(key^0x68bc21ebu)-1.0f,2.0f*hash01(key^0x02e5be93u)-1.0f};
        center+=jitter*(level.siteJitter*cell);
        const float d2=glm::dot(eta-center,eta-center);
        if(d2<hit.d2) {
            hit.secondD2=hit.d2; hit.secondCenter=hit.center; hit.secondKey=hit.key;
            hit.d2=d2; hit.center=center; hit.key=key;
        } else if(d2<hit.secondD2) {
            hit.secondD2=d2; hit.secondCenter=center; hit.secondKey=key;
        }
    }
    if(hit.secondD2>=1e29f) { hit.secondD2=hit.d2; hit.secondCenter=hit.center; hit.secondKey=hit.key; }
    const float ratio=std::sqrt(std::max(hit.d2,0.0f)/std::max(hit.secondD2,1e-8f));
    const float boundary=smooth01((ratio-0.55f)/0.45f);
    hit.center=glm::mix(hit.center,hit.secondCenter,level.crossover*boundary);
    return hit;
}

struct ClumpCoordinate {
    SiteHit primary{};
    SiteHit secondary{};
    glm::vec2 primaryDrift{0.0f};
    glm::vec2 secondaryDrift{0.0f};
};

ClumpCoordinate clumpCoordinate(const GroomBranch& branch,glm::vec2 eta,uint32_t branchIndex,uint32_t seed) {
    ClumpCoordinate c;
    c.primary=nearestClumpSites(branch.clumps.primary,eta,branchIndex,seed,0x243f6a88u);
    // Secondary sites are deterministic inside the primary domain. Including
    // the primary key creates a true hierarchy instead of two unrelated grids.
    c.secondary=nearestClumpSites(branch.clumps.secondary,eta,branchIndex,seed^c.primary.key,0x85a308d3u);
    c.primaryDrift={2.0f*hash01(c.primary.key^0x51633e2du)-1.0f,2.0f*hash01(c.primary.key^0x9b05688cu)-1.0f};
    c.secondaryDrift={2.0f*hash01(c.secondary.key^0xa4093822u)-1.0f,2.0f*hash01(c.secondary.key^0x299f31d0u)-1.0f};
    return c;
}

float clumpTightnessAt(const ClumpLevel& c,float u) {
    if(u<=0.5f) return glm::mix(c.rootTightness,c.midTightness,smooth01(u*2.0f));
    return glm::mix(c.midTightness,c.tipTightness,smooth01((u-0.5f)*2.0f));
}

glm::vec2 clumpWave(const ClumpLevel& level,uint32_t key,float u) {
    if(level.waveAmplitude<=0.0f || level.waveCycles<=0.0f) return glm::vec2(0.0f);
    const float phase=2.0f*PI*hash01(key^0x13198a2eu);
    const float angle=2.0f*PI*hash01(key^0x03707344u);
    const glm::vec2 axis{std::cos(angle),std::sin(angle)};
    const glm::vec2 side{-axis.y,axis.x};
    const float fade=smooth01((u-0.08f)/0.22f);
    const float theta=2.0f*PI*level.waveCycles*u+phase;
    return fade*level.waveAmplitude*(std::sin(theta)*axis+0.28f*std::sin(0.52f*theta+1.1f)*side);
}

float ellipseWeight(glm::vec2 p,glm::vec2 center,glm::vec2 radius) {
    const glm::vec2 q=(p-center)/glm::max(radius,glm::vec2(1e-4f));
    return 1.0f-smooth01((glm::length(q)-0.35f)/0.65f);
}

struct WarpBlend {
    glm::vec2 midDrift{0.0f};
    glm::vec2 tipDrift{0.0f};
    float lift=0.0f;
    float rollTurns=0.0f;
    float lengthScale=1.0f;
};

WarpBlend sampleStyleWarps(const GroomBranch& branch,glm::vec2 p) {
    WarpBlend out; float sum=0.0f,lengthAccum=0.0f;
    for(const auto& warp:branch.styleWarps) {
        const float w=std::max(0.0f,warp.weight)*ellipseWeight(p,warp.center,warp.radius);
        if(w<=1e-6f) continue;
        out.midDrift+=w*warp.midDrift; out.tipDrift+=w*warp.tipDrift;
        out.lift+=w*warp.lift; out.rollTurns+=w*warp.rollTurns;
        lengthAccum+=w*warp.lengthScale; sum+=w;
    }
    if(sum>1e-6f) {
        out.midDrift/=sum; out.tipDrift/=sum; out.lift/=sum; out.rollTurns/=sum;
        out.lengthScale=lengthAccum/sum;
    }
    return out;
}

uint64_t contactGroupTag(uint32_t branchIndex,uint32_t primaryClumpKey) {
    return (uint64_t(branchIndex+1u)<<32u)|uint64_t(primaryClumpKey);
}

glm::vec2 rotate2(glm::vec2 v,float radians) {
    const float c=std::cos(radians),sn=std::sin(radians);
    return {c*v.x-sn*v.y,sn*v.x+c*v.y};
}

float sampleCutSurface(const CutSurface& cut,glm::vec2 eta) {
    const float u=saturate(0.5f*(eta.x/1.35f+1.0f));
    const float v=saturate(0.5f*(eta.y/1.35f+1.0f));
    const float iu=1.0f-u,iv=1.0f-v;
    const std::array<float,4> bu{iu*iu*iu,3.0f*u*iu*iu,3.0f*u*u*iu,u*u*u};
    const std::array<float,4> bv{iv*iv*iv,3.0f*v*iv*iv,3.0f*v*v*iv,v*v*v};
    float value=0.0f;
    for(uint32_t y=0;y<4u;++y) for(uint32_t x=0;x<4u;++x)
        value+=cut.controls[y*4u+x]*bu[x]*bv[y];
    return value;
}

float authoredEndU(const GroomBranch& branch,glm::vec2 eta,const WarpBlend& warp,
                   const FieldSample& field,uint32_t index,uint32_t seed,float globalLengthScale) {
    float u=sampleCutSurface(branch.cut,eta);
    u+=(2.0f*hash01(seed^index*0x27d4eb2du)-1.0f)*branch.cut.breakup;
    u*=std::max(0.55f,warp.lengthScale)*field.length*globalLengthScale;
    return std::clamp(u,branch.cut.minLength,1.0f);
}

glm::vec3 keyPointWorld(const GroomPlacement& placement,const SweepKey& key,glm::vec2 eta,float radialLiftD=0.0f) {
    const glm::vec3 radial=safeNorm(key.centerD,{0,1,0});
    return placement.point(key.centerD+key.axisUD*eta.x+key.axisVD*eta.y+radial*radialLiftD);
}

float catmullKnot(glm::vec3 a,glm::vec3 b) {
    // Centripetal Catmull-Rom (alpha=1/2) limits overshoot for uneven
    // spacing between root, attachment, and scaffold controls.
    return std::sqrt(std::max(glm::length(b-a),1e-6f));
}

glm::vec3 sampleCentripetalSegment(glm::vec3 p0,glm::vec3 p1,glm::vec3 p2,glm::vec3 p3,float u) {
    const float t0=0.0f;
    const float t1=t0+catmullKnot(p0,p1);
    const float t2=t1+catmullKnot(p1,p2);
    const float t3=t2+catmullKnot(p2,p3);
    const float t=glm::mix(t1,t2,saturate(u));
    auto lerpK=[](glm::vec3 a,glm::vec3 b,float ta,float tb,float x) {
        const float den=std::max(tb-ta,1e-6f);
        return ((tb-x)/den)*a+((x-ta)/den)*b;
    };
    const glm::vec3 A1=lerpK(p0,p1,t0,t1,t);
    const glm::vec3 A2=lerpK(p1,p2,t1,t2,t);
    const glm::vec3 A3=lerpK(p2,p3,t2,t3,t);
    const glm::vec3 B1=lerpK(A1,A2,t0,t2,t);
    const glm::vec3 B2=lerpK(A2,A3,t1,t3,t);
    return lerpK(B1,B2,t1,t2,t);
}

glm::vec3 sampleSmoothPath(const glm::vec3* control,size_t count,float u) {
    if(count==0u) return glm::vec3(0.0f);
    if(count==1u) return control[0];
    const float f=saturate(u)*float(count-1u);
    const size_t i=std::min<size_t>(static_cast<size_t>(f),count-2u);
    const float local=f-float(i);
    const glm::vec3 p1=control[i],p2=control[i+1u];
    const glm::vec3 p0=i>0u?control[i-1u]:(2.0f*p1-p2);
    const glm::vec3 p3=i+2u<count?control[i+2u]:(2.0f*p2-p1);
    return sampleCentripetalSegment(p0,p1,p2,p3,local);
}

struct RestStrandCpu {
    TopologyId topology=0u;
    glm::vec3 surfaceRoot{0.0f};
    glm::vec3 root{0.0f};
    glm::vec3 rootNormal{0,1,0};
    glm::vec3 designRoot{0,1,0};
    vh::character::RootBinding rootBinding{};
    float density=1.0f;
    uint64_t contactGroupTag=0u;
    float minVolumeScale=1.0f;
    float maxVolumeScale=1.0f;
    std::array<glm::vec3,kGuidePoints> base{};
    std::array<glm::vec3,kGuidePoints> p{};
    std::array<float,kGuidePoints> attachment{};
    std::array<float,kGuidePoints> pathU{};
    std::array<float,kGuidePoints> crossSectionScale{};
};

RestStrandCpu generateRestStrand(const GroomPlacement& placement,
                                 const GroomBuildParams& params,
                                 const GroomStyle& style,
                                 const BoundRoot& boundRoot,
                                 uint32_t index) {
    RestStrandCpu s;
    s.rootBinding=boundRoot.binding;
    s.surfaceRoot=boundRoot.position;
    s.rootNormal=boundRoot.normal;
    s.designRoot=safeNorm(boundRoot.canonical.direction,{0,1,0});
    s.root=s.surfaceRoot+s.rootNormal*params.follicleClearance;

    const GroomBranch& branch=assignBranch(style,boundRoot.canonical);
    const uint32_t branchIndex=static_cast<uint32_t>(&branch-style.graph.branches.data());
    const StyleCurve& curve=style.graph.curves[branch.curve];
    if(curve.keys.size()+2u>16u) throw std::runtime_error("StyleCurve has too many keys for the compact authoring path");
    s.topology=branch.topology;
    const glm::vec2 eta=materialCoordinate(branch,boundRoot.canonical);
    const FieldSample field=sampleFields(style.fields,boundRoot.canonical.direction);
    const PartSample part=samplePart(style.part,boundRoot.canonical.direction);
    const ClumpCoordinate clump=clumpCoordinate(branch,eta,branchIndex,params.seed);
    const WarpBlend warp=sampleStyleWarps(branch,eta);
    s.density=std::clamp(field.density*part.densityScale,0.0f,1.0f);
    s.contactGroupTag=contactGroupTag(branchIndex,clump.primary.key);

    // Fixed local storage avoids 110k tiny heap allocations during every groom
    // compilation. The authoritative representation is small and curves are
    // intentionally bounded; dense fibers should be a streaming computation.
    std::array<glm::vec3,16> control{};
    size_t controlCount=0u;
    control[controlCount++]=s.root;
    const glm::vec3 firstTarget=keyPointWorld(placement,style.graph.sweepKeys[curve.keys.front()],eta);
    const glm::vec3 fallback=tangentProject({0,-1,0},s.rootNormal,{1,0,0});
    glm::vec3 rootTangent=tangentProject(firstTarget-s.root,s.rootNormal,fallback);
    if(part.flowWeight>1e-5f && glm::dot(part.flowD,part.flowD)>1e-10f) {
        const glm::vec3 worldFlow=tangentProject(placement.direction(part.flowD),s.rootNormal,rootTangent);
        rootTangent=safeNorm(glm::mix(rootTangent,worldFlow,saturate(part.flowWeight)),rootTangent);
    }
    const float firstDistance=glm::length(firstTarget-s.root);
    const float attachLength=std::clamp(0.28f*firstDistance,0.012f,0.038f);
    control[controlCount++]=s.root+rootTangent*attachLength+s.rootNormal*std::min(0.008f,0.20f*attachLength);

    s.minVolumeScale=std::numeric_limits<float>::max();
    s.maxVolumeScale=0.0f;
    for(size_t j=0;j<curve.keys.size();++j) {
        const auto& key=style.graph.sweepKeys[curve.keys[j]];
        const float t=float(j+1u)/float(curve.keys.size());
        const float midEnvelope=4.0f*t*(1.0f-t);
        const float primaryT=clumpTightnessAt(branch.clumps.primary,t);
        const float secondaryT=clumpTightnessAt(branch.clumps.secondary,t);

        glm::vec2 primaryTarget=clump.primary.center
            +branch.clumps.primary.drift*clump.primaryDrift*midEnvelope
            +clumpWave(branch.clumps.primary,clump.primary.key,t);
        glm::vec2 secondaryTarget=clump.secondary.center
            +branch.clumps.secondary.drift*clump.secondaryDrift*midEnvelope
            +clumpWave(branch.clumps.secondary,clump.secondary.key,t);

        // Low frequency first, then the finer clump. Both attractions vanish at
        // the root according to their authored profiles, preserving follicles.
        glm::vec2 strandEta=glm::mix(eta,primaryTarget,primaryT);
        strandEta=glm::mix(strandEta,secondaryTarget,secondaryT);
        const glm::vec2 radialDir=glm::length(eta)>1e-5f?glm::normalize(eta):glm::vec2(1,0);
        strandEta+=branch.clumps.tipSpread*radialDir*t*t;
        strandEta+=warp.midDrift*midEnvelope+warp.tipDrift*t*t;
        const float angle=2.0f*PI*(branch.clumps.twistTurns+warp.rollTurns)*t;
        strandEta=rotate2(strandEta,angle);

        const float volumeScale=std::max(0.20f,field.volume*params.globalVolumeScale);
        strandEta*=volumeScale;
        strandEta=glm::clamp(strandEta,glm::vec2(-1.58f),glm::vec2(1.58f));
        const float supportScale=std::max(0.18f,volumeScale*(1.0f-0.35f*primaryT-0.20f*secondaryT));
        s.minVolumeScale=std::min(s.minVolumeScale,supportScale);
        s.maxVolumeScale=std::max(s.maxVolumeScale,supportScale);
        control[controlCount++]=keyPointWorld(placement,key,strandEta,warp.lift*midEnvelope);
    }

    const float sampleSpan=float(kGuidePoints-1u),controlSpan=float(controlCount-1u);
    for(size_t j=0;j<curve.keys.size();++j) {
        const float pin=style.graph.sweepKeys[curve.keys[j]].attachment;
        if(pin<=0.0f) continue;
        const float keyU=float(j+2u)/controlSpan;
        for(uint32_t p=1;p<kGuidePoints;++p) {
            const float u=float(p)/sampleSpan;
            const float footprint=1.0f-smooth01((std::abs(u-keyU)*sampleSpan)/1.35f);
            s.attachment[p]=std::max(s.attachment[p],pin*footprint);
        }
    }

    const float endU=authoredEndU(branch,eta,warp,field,index,params.seed,params.globalLengthScale);
    const float microScale=std::clamp(params.microDetailScale,0.0f,1.5f);
    const float detailPopulationScale=std::sqrt(microScale);
    const bool hasFrizz=hash01(params.seed^index*0x91e10da5u)<branch.detail.frizzFraction*field.detail*detailPopulationScale;
    const bool hasFlyaway=hash01(params.seed^index*0xa24baed5u)<branch.detail.flyawayFraction*field.detail*detailPopulationScale;
    for(uint32_t p=0;p<kGuidePoints;++p) {
        const float u=float(p)/float(kGuidePoints-1u),su=u*endU;
        s.pathU[p]=su;
        s.base[p]=sampleSmoothPath(control.data(),controlCount,su);
        const float primaryT=clumpTightnessAt(branch.clumps.primary,su);
        const float secondaryT=clumpTightnessAt(branch.clumps.secondary,su);
        s.crossSectionScale[p]=std::max(0.18f,field.volume*params.globalVolumeScale*(1.0f-0.35f*primaryT-0.20f*secondaryT));
    }
    s.base[0]=s.root;

    glm::vec3 prevT=safeNorm(s.base[1]-s.base[0],{0,-1,0});
    glm::vec3 N=safeNorm(s.rootNormal-prevT*glm::dot(s.rootNormal,prevT),glm::cross(prevT,glm::vec3(1,0,0)));
    glm::vec3 B=safeNorm(glm::cross(prevT,N),{0,0,1});
    N=safeNorm(glm::cross(B,prevT),N);
    const float clumpPhase=2.0f*PI*hash01(clump.secondary.key^0x7f4a7c15u);
    const float strandPhase=2.0f*PI*hash01(params.seed^index*0x165667b1u);
    const float coherence=saturate(branch.texture.phaseCoherence);
    const glm::vec2 phaseVec=coherence*glm::vec2(std::cos(clumpPhase),std::sin(clumpPhase))
                            +(1.0f-coherence)*glm::vec2(std::cos(strandPhase),std::sin(strandPhase));
    const float phase=std::atan2(phaseVec.y,phaseVec.x);
    const float cycleJitter=1.0f+branch.texture.irregularity*(2.0f*hash01(index^clump.secondary.key)-1.0f);
    const float cycles=std::max(0.0f,branch.texture.cycles*field.textureFrequency*cycleJitter);
    const float amplitude=branch.texture.amplitude*field.textureAmplitude;

    for(uint32_t p=0;p<kGuidePoints;++p) {
        const uint32_t a=p>0u?p-1u:p,b=p+1u<kGuidePoints?p+1u:p;
        const glm::vec3 T=safeNorm(s.base[b]-s.base[a],prevT);
        if(p>0u) N=rotateFromTo(N,prevT,T);
        N=safeNorm(N-T*glm::dot(N,T),N); B=safeNorm(glm::cross(T,N),B); N=safeNorm(glm::cross(B,T),N); prevT=T;
        const float u=float(p)/float(kGuidePoints-1u);
        const float texFade=smooth01((u-branch.texture.start)/std::max(0.08f,0.24f*(1.0f-branch.texture.start)));
        const float theta=2.0f*PI*cycles*u+phase;
        glm::vec3 q=s.base[p];
        switch(branch.texture.type) {
            case HairTexture::Straight: break;
            case HairTexture::Wavy: q+=texFade*amplitude*(std::sin(theta)*N+branch.texture.secondary*std::sin(0.53f*theta+1.1f)*B); break;
            case HairTexture::Curly: q+=texFade*amplitude*(std::cos(theta)*N+std::sin(theta)*B); break;
            case HairTexture::Coily: q+=texFade*amplitude*(std::cos(theta)*N+0.76f*std::sin(theta)*B+0.18f*std::sin(2.17f*theta+0.8f)*N); break;
        }
        if(hasFrizz && p>0u) {
            const float f=2.0f*PI*(branch.detail.frizzFrequency*u)+strandPhase;
            q+=smooth01((u-0.12f)/0.25f)*branch.detail.frizzAmplitude*field.detail*microScale*(std::sin(f)*N+0.72f*std::sin(1.73f*f+1.3f)*B);
        }
        if(hasFlyaway && p>0u) {
            const float ramp=u*u*smooth01((u-0.18f)/0.55f);
            q+=ramp*branch.detail.flyawayAmplitude*field.detail*microScale*safeNorm(0.72f*N+0.28f*std::sin(strandPhase)*B,N);
        }
        s.p[p]=q;
    }
    s.p[0]=s.root;
    return s;
}

// Simulation guides form a reduced basis of material space, but every primary
// clump must be represented. This keeps simulation interpolation inside the
// hairstyle's mesoscopic mass hierarchy instead of averaging adjacent locks.
std::vector<uint32_t> selectSimulationSources(const std::vector<RestStrandCpu>& rest) {
    std::map<uint64_t,std::vector<uint32_t>> groups;
    for(uint32_t i=0;i<rest.size();++i) groups[rest[i].contactGroupTag].push_back(i);
    if(groups.empty()) throw std::runtime_error("No primary clumps available for guide selection");
    if(groups.size()>kSimulationGuideCount)
        throw std::runtime_error("Primary clump count exceeds the simulation-guide budget");

    std::vector<uint64_t> keys; keys.reserve(groups.size());
    for(const auto& kv:groups) keys.push_back(kv.first);
    std::vector<uint32_t> quota(keys.size(),1u);
    uint32_t sum=static_cast<uint32_t>(keys.size());
    while(sum<kSimulationGuideCount) {
        size_t best=keys.size(); double bestNeed=-1e30;
        for(size_t g=0;g<keys.size();++g) {
            const auto& members=groups.at(keys[g]);
            if(quota[g]>=members.size()) continue;
            const double ideal=double(kSimulationGuideCount)*double(members.size())/double(rest.size());
            const double need=ideal-double(quota[g]);
            if(need>bestNeed) { bestNeed=need; best=g; }
        }
        if(best==keys.size()) break;
        ++quota[best]; ++sum;
    }
    if(sum!=kSimulationGuideCount)
        throw std::runtime_error("Unable to allocate the full simulation-guide budget across primary clumps");

    std::vector<uint32_t> result; result.reserve(kSimulationGuideCount);
    for(size_t g=0;g<keys.size();++g) {
        const auto& members=groups.at(keys[g]); const uint32_t q=quota[g];
        for(uint32_t j=0;j<q;++j) {
            const size_t k=std::min(members.size()-1u,size_t((uint64_t(2u*j+1u)*members.size())/(2u*q)));
            result.push_back(members[k]);
        }
    }
    if(result.size()!=kSimulationGuideCount) throw std::runtime_error("Simulation-guide selection returned the wrong count");
    return result;
}

struct GridCoord { int x=0,y=0; };
GridCoord gridCoord(glm::vec3 unitRoot) {
    const glm::vec3 q=safeNorm(unitRoot,{0,1,0});
    float phi=std::atan2(q.z,q.x); if(phi<0) phi+=2.0f*PI;
    const float theta=std::acos(std::clamp(q.y,-1.0f,1.0f));
    return {std::clamp(int(phi/(2.0f*PI)*float(kGridPhi)),0,int(kGridPhi)-1),
            std::clamp(int(theta/1.48f*float(kGridTheta)),0,int(kGridTheta)-1)};
}

struct Nearest4 {
    std::array<uint32_t,4> id{0,0,0,0};
    std::array<float,4> score{1e9f,1e9f,1e9f,1e9f};
    std::array<float,4> rootDistance{1e9f,1e9f,1e9f,1e9f};
};

float materialBindingScore(const RestStrandCpu& s,const SimulationGuideCpu& g) {
    const float domain=glm::length(s.designRoot-g.canonicalDirection);
    const glm::vec3 st0=safeNorm(s.base[3]-s.base[0],s.designRoot);
    const glm::vec3 gt0=safeNorm(g.p[3]-g.p[0],g.canonicalDirection);
    const glm::vec3 stm=safeNorm(s.base[14]-s.base[8],st0);
    const glm::vec3 gtm=safeNorm(g.p[14]-g.p[8],gt0);
    const float tangentMismatch=1.0f-std::clamp(glm::dot(st0,gt0),-1.0f,1.0f);
    const float shapeMismatch=1.0f-std::clamp(glm::dot(stm,gtm),-1.0f,1.0f);
    return domain+0.040f*tangentMismatch+0.025f*shapeMismatch;
}

Nearest4 findNearestGuides(const RestStrandCpu& s,
                           const std::vector<SimulationGuideCpu>& guides,
                           const std::vector<std::vector<uint32_t>>& grid) {
    const GridCoord c=gridCoord(s.designRoot);
    Nearest4 out;
    uint32_t compatible=0u;
    auto consider=[&](uint32_t gid){
        const auto& g=guides[gid];
        if(s.topology!=g.topology || s.contactGroupTag!=g.contactGroupTag) return;
        ++compatible;
        const float score=materialBindingScore(s,g);
        const float rootDist=glm::length(s.root-g.root);
        for(int k=0;k<4;++k) if(score<out.score[k]) {
            for(int q=3;q>k;--q) { out.score[q]=out.score[q-1]; out.id[q]=out.id[q-1]; out.rootDistance[q]=out.rootDistance[q-1]; }
            out.score[k]=score; out.id[k]=gid; out.rootDistance[k]=rootDist; break;
        }
    };

    // Scan annuli instead of building a temporary candidate vector for every
    // one of the 110k fibers. This removes another dense-build allocation hot
    // spot while preserving the same material-space locality.
    for(int r=0;r<=6 && out.score[3]>=1e8f;++r) {
        for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
            if(r>0 && std::abs(dx)!=r && std::abs(dy)!=r) continue;
            const int yy=c.y+dy; if(yy<0||yy>=int(kGridTheta)) continue;
            int xx=(c.x+dx)%int(kGridPhi); if(xx<0) xx+=int(kGridPhi);
            const auto& cell=grid[uint32_t(yy)*kGridPhi+uint32_t(xx)];
            for(uint32_t gid:cell) consider(gid);
        }
    }
    if(out.score[3]>=1e8f) for(uint32_t gid=0;gid<guides.size();++gid) consider(gid);
    if(out.score[0]>=1e8f) throw std::runtime_error("No primary-clump-compatible Simulation Guide found");
    for(int k=1;k<4;++k) if(out.score[k]>=1e8f) { out.id[k]=out.id[0]; out.score[k]=out.score[0]; out.rootDistance[k]=out.rootDistance[0]; }
    (void)compatible;
    return out;
}

std::array<float,4> guideWeights(const Nearest4& n) {
    std::array<float,4> w{};
    const float sigma=std::max(0.010f,0.72f*n.score[3]+0.004f);
    float sum=0.0f;
    for(int k=0;k<4;++k) { const float x=n.score[k]/sigma; w[k]=std::exp(-0.5f*x*x); sum+=w[k]; }
    if(sum<1e-8f) return {1,0,0,0};
    for(float& x:w) x/=sum;
    return w;
}

std::vector<GuidePointMetaGpu> buildGuidePointMeta(const std::vector<SimulationGuideCpu>& guides,
                                                    const std::vector<double>& representedRoots,
                                                    const std::vector<double>& representedFibers,
                                                    float scalpArea,
                                                    GroomAudit& audit) {
    const size_t n=guides.size();
    std::vector<GuidePointMetaGpu> out(n*kGuidePoints);
    audit.minGuideSupportRadius=std::numeric_limits<float>::max();
    audit.maxGuideSupportRadius=0.0f;
    const float safeScalpArea=std::max(scalpArea,0.01f);
    for(size_t i=0;i<n;++i) {
        const float rootArea=safeScalpArea*float(std::max(0.0,representedRoots[i]))/float(kRenderStrandCount);
        const float footprintRadius=std::clamp(std::sqrt(std::max(rootArea,1e-8f)/PI),0.00115f,0.0075f);
        for(uint32_t p=0;p<kGuidePoints;++p) {
            const float u=float(p)/float(kGuidePoints-1u);
            const float rootFade=smooth01(u/0.18f);
            const float radius=std::clamp(footprintRadius*guides[i].crossSectionScale[p],0.00115f,0.0080f)*rootFade;
            const uint32_t a=p>0u?p-1u:p, b=p+1u<kGuidePoints?p+1u:p;
            const float segment=std::max(0.001f,0.5f*glm::length(guides[i].p[b]-guides[i].p[a]));
            const float volume=PI*radius*radius*segment;
            // z = effective render-fiber multiplicity carried by this reduced guide.
            out[i*kGuidePoints+p].supportVolume=glm::vec4(radius,volume,float(std::max(0.0,representedFibers[i])),0.0f);
            if(p>0u) audit.minGuideSupportRadius=std::min(audit.minGuideSupportRadius,radius);
            audit.maxGuideSupportRadius=std::max(audit.maxGuideSupportRadius,radius);
        }
    }
    return out;
}

std::vector<GuideFrameGpu> buildGuideFrames(const std::vector<SimulationGuideCpu>& guides) {
    std::vector<GuideFrameGpu> out;
    out.reserve(guides.size()*kGuidePoints);
    for(const auto& g:guides) {
        glm::vec3 prevT=safeNorm(g.p[1]-g.p[0],{0,-1,0});
        glm::vec3 N=safeNorm(g.rootNormal-prevT*glm::dot(g.rootNormal,prevT),glm::cross(prevT,glm::vec3(1,0,0)));
        glm::vec3 B=safeNorm(glm::cross(prevT,N),{0,0,1});
        N=safeNorm(glm::cross(B,prevT),N);
        for(uint32_t p=0;p<kGuidePoints;++p) {
            const uint32_t a=p>0u?p-1u:p;
            const uint32_t b=p+1u<kGuidePoints?p+1u:p;
            const glm::vec3 T=safeNorm(g.p[b]-g.p[a],prevT);
            if(p>0u) N=rotateFromTo(N,prevT,T);
            N=safeNorm(N-T*glm::dot(N,T),N);
            B=safeNorm(glm::cross(T,N),B);
            N=safeNorm(glm::cross(B,T),N);
            GuideFrameGpu f; f.normal=glm::vec4(N,0.0f); out.push_back(f);
            prevT=T;
        }
    }
    return out;
}

} // namespace

GroomAsset GroomBuilder::build(const GroomBinding& binding,
                             const GroomBuildParams& params,
                             const GroomStyle& style,
                             const GroomAsset* reusableBasis) const {
    const auto buildStart=std::chrono::steady_clock::now();
    validateGraph(style);
    if(binding.roots.size()!=kRenderStrandCount)
        throw std::runtime_error("GroomBinding root count must equal the dense render-strand count");
    if(!binding.renderOrder.empty() && binding.renderOrder.size()!=kRenderStrandCount)
        throw std::runtime_error("GroomBinding render order is incomplete");

    GroomAsset out;
    out.audit.retargetFallbackRoots=binding.audit.fallbackBindings;
    out.audit.retargetFallbackFraction=binding.audit.fallbackFraction;

    TopologyId maxTopology=0u;
    for(const auto& branch:style.graph.branches) maxTopology=std::max(maxTopology,branch.topology);
    const uint32_t topologyCount=uint32_t(maxTopology)+1u;

    // 1) Build dense rest geometry in parallel, O(N*P).
    std::vector<RestStrandCpu> rest(kRenderStrandCount);
    parallelFor(kRenderStrandCount,[&](uint32_t dst){
        const uint32_t rootId=binding.renderOrder.empty()?dst:binding.renderOrder[dst];
        rest[dst]=generateRestStrand(binding.placement,params,style,binding.roots[rootId],rootId);
    });
    const auto denseDone=std::chrono::steady_clock::now();

    out.audit.minCrossSectionScale=std::numeric_limits<float>::max();
    std::vector<uint32_t> topologyPopulation(topologyCount,0u);
    std::unordered_set<uint64_t> activeContactGroups; activeContactGroups.reserve(512u);
    for(const auto& r:rest) {
        activeContactGroups.insert(r.contactGroupTag);
        out.audit.minCrossSectionScale=std::min(out.audit.minCrossSectionScale,r.minVolumeScale);
        out.audit.maxCrossSectionScale=std::max(out.audit.maxCrossSectionScale,r.maxVolumeScale);
        if(r.topology>=topologyCount) { ++out.audit.unassignedFollicles; continue; }
        ++topologyPopulation[r.topology];
    }
    out.audit.activeContactGroups=static_cast<uint32_t>(activeContactGroups.size());
    out.audit.smallestTopologyPopulation=kRenderStrandCount;
    for(uint32_t n:topologyPopulation) if(n>0u) out.audit.smallestTopologyPopulation=std::min(out.audit.smallestTopologyPopulation,n);
    if(out.audit.unassignedFollicles!=0u) throw std::runtime_error("A follicle was not assigned to a valid groom topology");

    // 2) Reuse guide identities and follower bindings when the groom
    // topology is unchanged.
    bool reuseBasis = reusableBasis &&
                      reusableBasis->guideSourceStrands.size()==kSimulationGuideCount &&
                      reusableBasis->strands.size()==kRenderStrandCount &&
                      reusableBasis->strandTopology.size()==kRenderStrandCount;
    if(reuseBasis) {
        for(uint32_t i=0;i<kRenderStrandCount;++i) {
            if(reusableBasis->strandTopology[i]!=rest[i].topology ||
               reusableBasis->strandContactGroupTags[i]!=rest[i].contactGroupTag) { reuseBasis=false; break; }
        }
    }
    std::vector<uint32_t> guideSources = reuseBasis
        ? reusableBasis->guideSourceStrands
        : selectSimulationSources(rest);
    for(uint32_t source:guideSources) if(source>=rest.size()) { reuseBasis=false; break; }
    if(!reuseBasis && guideSources.size()!=kSimulationGuideCount)
        guideSources=selectSimulationSources(rest);
    out.guideSourceStrands=guideSources;
    out.simulationGuides.reserve(kSimulationGuideCount);
    out.restGuidePoints.reserve(kSimulationGuideCount*kGuidePoints);
    for(uint32_t source:guideSources) {
        const auto& rs=rest[source];
        SimulationGuideCpu g;
        g.topology=rs.topology; g.contactGroupTag=rs.contactGroupTag; g.rootBinding=rs.rootBinding;
        g.root=rs.root; g.rootNormal=rs.rootNormal; g.canonicalDirection=rs.designRoot;
        g.authoredDensity=rs.density;
        for(uint32_t p=0;p<kGuidePoints;++p) {
            g.p[p]=rs.base[p]; g.pathU[p]=rs.pathU[p]; g.crossSectionScale[p]=rs.crossSectionScale[p];
            const float attachment=rs.attachment[p];
            if(attachment>0.01f) ++out.audit.constrainedGuideSamples;
            StrandPointGpu gp; gp.position=glm::vec4(g.p[p],attachment);
            out.restGuidePoints.push_back(gp);
        }
        out.simulationGuides.push_back(g);
    }
    out.restGuideFrames=buildGuideFrames(out.simulationGuides);
    const auto basisDone=std::chrono::steady_clock::now();

    // 3) Rebuild guide-neighborhood acceleration when topology changes.
    std::vector<std::vector<uint32_t>> grid;
    if(!reuseBasis) {
        grid.resize(kGridPhi*kGridTheta);
        for(uint32_t gid=0;gid<out.simulationGuides.size();++gid) {
            const GridCoord c=gridCoord(out.simulationGuides[gid].canonicalDirection);
            grid[uint32_t(c.y)*kGridPhi+uint32_t(c.x)].push_back(gid);
        }
    }

    // 4) Bind render fibers in canonical spatial order.
    out.strands.resize(kRenderStrandCount);
    out.baseStrandPoints.resize(size_t(kRenderStrandCount)*kGuidePoints);
    out.restStrandPoints.resize(size_t(kRenderStrandCount)*kGuidePoints);
    out.strandPathU.resize(size_t(kRenderStrandCount)*kGuidePoints);
    out.strandContactGroupTags.resize(kRenderStrandCount);
    out.strandTopology.resize(kRenderStrandCount);
    std::vector<double> representedRoots(out.simulationGuides.size(),0.0);
    std::vector<double> representedFibers(out.simulationGuides.size(),0.0);

    // Descriptor/binding search is independent for every strand. Keep the
    // output arrays indexed so workers never contend.
    parallelFor(kRenderStrandCount,[&](uint32_t i){
        const auto& rs=rest[i];
        StrandDescGpu d{};
        if(reuseBasis) {
            d.guides=reusableBasis->strands[i].guides;
            d.weights=reusableBasis->strands[i].weights;
        } else {
            const Nearest4 n=findNearestGuides(rs,out.simulationGuides,grid);
            const auto w=guideWeights(n);
            d.guides={n.id[0],n.id[1],n.id[2],n.id[3]};
            d.weights={w[0],w[1],w[2],w[3]};
        }
        const uint32_t rootId=binding.renderOrder.empty()?i:binding.renderOrder[i];
        const float radiusNoise=2.0f*hash01(params.seed+rootId*31u)-1.0f;
        const float diameter=params.fiberDiameter*style.optics.rootDiameterScale*(1.0f+style.optics.diameterVariation*radiusNoise);
        d.rootNormalDiameter=glm::vec4(rs.rootNormal,std::max(1.0e-5f,diameter));
        d.fiber=glm::vec4(std::clamp(style.optics.tipDiameterScale,0.02f,1.0f),rs.density,0.0f,0.0f);
        out.strands[i]=d; out.strandContactGroupTags[i]=rs.contactGroupTag; out.strandTopology[i]=rs.topology;
        for(uint32_t p=0;p<kGuidePoints;++p) {
            out.restStrandPoints[size_t(i)*kGuidePoints+p].position=glm::vec4(rs.p[p],0.0f);
            out.baseStrandPoints[size_t(i)*kGuidePoints+p].position=glm::vec4(rs.base[p],0.0f);
            out.strandPathU[size_t(i)*kGuidePoints+p]=rs.pathU[p];
        }
    });

    // Reduce represented material mass and binding diagnostics.
    out.audit.minDensity=1.0f; out.audit.maxDensity=0.0f;
    for(uint32_t i=0;i<kRenderStrandCount;++i) {
        const auto& rs=rest[i]; const auto& d=out.strands[i];
        out.audit.minDensity=std::min(out.audit.minDensity,rs.density);
        out.audit.maxDensity=std::max(out.audit.maxDensity,rs.density);
        const std::array<uint32_t,4> ids{d.guides.x,d.guides.y,d.guides.z,d.guides.w};
        const std::array<float,4> ww{d.weights.x,d.weights.y,d.weights.z,d.weights.w};
        for(int k=0;k<4;++k) {
            const uint32_t gid=ids[k];
            if(rs.topology!=out.simulationGuides[gid].topology) ++out.audit.crossTopologyBindings;
            if(rs.contactGroupTag!=out.simulationGuides[gid].contactGroupTag) ++out.audit.crossPrimaryClumpBindings;
            representedRoots[gid]+=double(ww[k]);
            representedFibers[gid]+=double(ww[k])*double(rs.density);
        }
        out.audit.maxRootGuideDistance=std::max(out.audit.maxRootGuideDistance,
            glm::length(rs.root-out.simulationGuides[d.guides.x].root));
    }
    out.guidePointMeta=buildGuidePointMeta(out.simulationGuides,representedRoots,representedFibers,
                                           binding.audit.scalpArea,out.audit);
    const auto bindDone=std::chrono::steady_clock::now();

    // 5) Rest deformation transfer identity audit.
    for(uint32_t sid=0;sid<kRenderStrandCount;sid+=997u) {
        const auto& d=out.strands[sid];
        for(uint32_t p=0;p<kGuidePoints;p+=4u) {
            const glm::vec3 restP=glm::vec3(out.restStrandPoints[size_t(sid)*kGuidePoints+p].position);
            glm::vec3 reconstructed(0.0f);
            const std::array<uint32_t,4> ids{d.guides.x,d.guides.y,d.guides.z,d.guides.w};
            const std::array<float,4> ww{d.weights.x,d.weights.y,d.weights.z,d.weights.w};
            for(int k=0;k<4;++k) {
                const glm::vec3 gr=out.simulationGuides[ids[k]].p[p];
                reconstructed+=ww[k]*(gr+(restP-gr));
            }
            out.audit.restReconstructionError=std::max(out.audit.restReconstructionError,glm::length(reconstructed-restP));
        }
    }

    // 6) Cluster bounds in the immutable canonical render order. Membership is
    // stable; only bounds are recomputed when geometry changes.
    const uint32_t clusterCount=(kRenderStrandCount+kClusterSize-1u)/kClusterSize;
    out.clusters.resize(clusterCount);
    parallelFor(clusterCount,[&](uint32_t c){
        const uint32_t first=c*kClusterSize, count=std::min(kClusterSize,kRenderStrandCount-first);
        glm::vec3 center(0.0f); uint32_t n=0u;
        for(uint32_t i=0;i<count;++i) for(uint32_t p=0;p<kGuidePoints;p+=2u) {
            center+=glm::vec3(out.restStrandPoints[size_t(first+i)*kGuidePoints+p].position); ++n;
        }
        center/=float(std::max(1u,n));
        float radius=0.0f;
        for(uint32_t i=0;i<count;++i) for(uint32_t p=0;p<kGuidePoints;++p)
            radius=std::max(radius,glm::length(glm::vec3(out.restStrandPoints[size_t(first+i)*kGuidePoints+p].position)-center));
        out.clusters[c].range={first,count,0u,0u}; out.clusters[c].sphere=glm::vec4(center,radius+0.004f);
    });

    const auto clusterDone=std::chrono::steady_clock::now();
    auto ms=[](auto a,auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
    std::cout << "V39 groom [" << style.name << "]:\n"
              << "  strands=" << kRenderStrandCount << " guides=" << out.simulationGuides.size()
              << " activeContactGroups=" << out.audit.activeContactGroups << "\n"
              << "  material-space basis=" << (reuseBasis?"reused":"rebuilt")
              << " | scalpArea=" << binding.audit.scalpArea << " m^2\n"
              << "  maxRootGuideDistance=" << out.audit.maxRootGuideDistance
              << " restIdentityError=" << out.audit.restReconstructionError << "\n"
              << "  density=[" << out.audit.minDensity << ", " << out.audit.maxDensity << "]"
              << " effectiveGuideSupport=[" << out.audit.minGuideSupportRadius << ", "
              << out.audit.maxGuideSupportRadius << "] m\n"
              << "  compile phases: dense=" << ms(buildStart,denseDone)
              << " ms basis=" << ms(denseDone,basisDone)
              << " ms bind=" << ms(basisDone,bindDone)
              << " ms cluster=" << ms(bindDone,clusterDone)
              << " ms total=" << ms(buildStart,clusterDone) << " ms\n";

    if(out.audit.crossTopologyBindings!=0u)
        throw std::runtime_error("V39 structural invariant failed: guide binding crossed groom topology");
    if(out.audit.crossPrimaryClumpBindings!=0u)
        throw std::runtime_error("V39 structural invariant failed: guide binding crossed primary clump");
    if(out.audit.restReconstructionError>2e-6f)
        throw std::runtime_error("V39 structural invariant failed: simulation basis changed the rest hairstyle");
    return out;
}

} // namespace vh::groom
