#include "groom_contact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vh::groom {
namespace {

constexpr float PI=3.14159265358979323846f;
constexpr uint32_t kCorrectionNodes=32u;
constexpr uint32_t kMaxIterations=8u;
constexpr float kViolationEps=1.0e-5f;

float saturate(float x){ return std::clamp(x,0.0f,1.0f); }

glm::vec3 safeNorm(glm::vec3 v,glm::vec3 fallback={0,1,0}) {
    const float l2=glm::dot(v,v); return l2>1e-12f?v/std::sqrt(l2):fallback;
}

glm::vec3 rotateFromTo(glm::vec3 x,glm::vec3 from,glm::vec3 to) {
    const glm::vec3 a=safeNorm(from),b=safeNorm(to);
    const float c=std::clamp(glm::dot(a,b),-1.0f,1.0f);
    if(c>0.9999f) return x;
    if(c<-0.9999f) {
        const glm::vec3 ref=std::abs(a.y)<0.8f?glm::vec3(0,1,0):glm::vec3(1,0,0);
        const glm::vec3 axis=safeNorm(glm::cross(a,ref),{0,0,1});
        return 2.0f*axis*glm::dot(axis,x)-x;
    }
    const glm::vec3 v=glm::cross(a,b); const float inv=1.0f/(1.0f+c);
    return x+glm::cross(v,x)+glm::cross(v,glm::cross(v,x))*inv;
}

float fieldLowerBound(const vh::character::CharacterSurface& character,glm::vec3 p) {
    const auto& field=character.contactField();
    if(!field.valid() || !field.contains(p)) return std::numeric_limits<float>::infinity();
    const glm::vec3 g=(p-field.origin)/field.cellSize;
    glm::ivec3 b=glm::ivec3(glm::floor(g));
    b=glm::clamp(b,glm::ivec3(0),glm::ivec3(int(field.dim)-2));
    // Exact grid-vertex distances + 1-Lipschitz distance geometry give a
    // conservative broad-phase bound: d(p) >= max_i(d(c_i)-|p-c_i|).
    float lower=-std::numeric_limits<float>::infinity();
    for(int dz=0;dz<=1;++dz) for(int dy=0;dy<=1;++dy) for(int dx=0;dx<=1;++dx) {
        const glm::ivec3 c=b+glm::ivec3(dx,dy,dz);
        const size_t idx=(size_t(c.z)*field.dim+size_t(c.y))*field.dim+size_t(c.x);
        const glm::vec3 cp=field.origin+glm::vec3(c)*field.cellSize;
        lower=std::max(lower,field.distance[idx]-glm::length(p-cp));
    }
    return lower;
}

float exactOrSafeDistance(const vh::character::CharacterSurface& character,glm::vec3 p,float threshold) {
    const auto& field=character.contactField();
    const float approx=field.sample(p);
    if(!std::isfinite(approx)) return approx;
    const float lower=fieldLowerBound(character,p);
    if(lower>threshold) return lower; // conservative and already sufficient for classification
    return character.contact(p).distance;
}

std::vector<GuideFrameGpu> buildGuideFrames(const std::vector<SimulationGuideCpu>& guides) {
    std::vector<GuideFrameGpu> out; out.reserve(guides.size()*kGuidePoints);
    for(const auto& g:guides) {
        glm::vec3 prevT=safeNorm(g.p[1]-g.p[0],{0,-1,0});
        glm::vec3 N=safeNorm(g.rootNormal-prevT*glm::dot(g.rootNormal,prevT),glm::cross(prevT,glm::vec3(1,0,0)));
        glm::vec3 B=safeNorm(glm::cross(prevT,N),{0,0,1}); N=safeNorm(glm::cross(B,prevT),N);
        for(uint32_t p=0;p<kGuidePoints;++p) {
            const uint32_t a=p>0u?p-1u:p,b=p+1u<kGuidePoints?p+1u:p;
            const glm::vec3 T=safeNorm(g.p[b]-g.p[a],prevT);
            if(p>0u) N=rotateFromTo(N,prevT,T);
            N=safeNorm(N-T*glm::dot(N,T),N); B=safeNorm(glm::cross(T,N),B); N=safeNorm(glm::cross(B,T),N);
            GuideFrameGpu f; f.normal=glm::vec4(N,0.0f); out.push_back(f); prevT=T;
        }
    }
    return out;
}

void rebuildClusters(GroomAsset& groom) {
    const uint32_t strandCount=static_cast<uint32_t>(groom.strands.size());
    const uint32_t clusterCount=(strandCount+kClusterSize-1u)/kClusterSize;
    groom.clusters.resize(clusterCount);
    for(uint32_t c=0;c<clusterCount;++c) {
        const uint32_t first=c*kClusterSize,count=std::min(kClusterSize,strandCount-first);
        glm::vec3 center(0.0f); uint32_t n=0u;
        for(uint32_t i=0;i<count;++i) for(uint32_t p=0;p<kGuidePoints;p+=2u) {
            center+=glm::vec3(groom.restStrandPoints[(first+i)*kGuidePoints+p].position); ++n;
        }
        center/=float(std::max(1u,n)); float radius=0.0f;
        for(uint32_t i=0;i<count;++i) for(uint32_t p=0;p<kGuidePoints;++p)
            radius=std::max(radius,glm::length(glm::vec3(groom.restStrandPoints[(first+i)*kGuidePoints+p].position)-center));
        groom.clusters[c].range={first,count,0u,0u}; groom.clusters[c].sphere=glm::vec4(center,radius+0.004f);
    }
}

struct ConstraintSample {
    glm::vec3 x{0.0f};
    float s=0.0f;
    float clearance=0.0f;
};

struct NodeField { std::array<glm::vec3,kCorrectionNodes> d{}; };

struct Interp {
    uint32_t i0=0u,i1=0u; float w0=1.0f,w1=0.0f;
};

Interp nodeInterp(float s) {
    const float f=saturate(s)*float(kCorrectionNodes-1u);
    const uint32_t i0=std::min<uint32_t>(uint32_t(f),kCorrectionNodes-2u);
    return {i0,i0+1u,1.0f-(f-float(i0)),f-float(i0)};
}

glm::vec3 sampleField(const NodeField& field,float s) {
    const auto q=nodeInterp(s); return q.w0*field.d[q.i0]+q.w1*field.d[q.i1];
}

bool projectConstraint(const vh::character::CharacterSurface& character,const ConstraintSample& c,
                       NodeField& field,ContactResolveReport& stats) {
    const auto q=nodeInterp(c.s);
    const glm::vec3 correction=q.w0*field.d[q.i0]+q.w1*field.d[q.i1];
    const glm::vec3 x=c.x+correction;
    const auto hit=character.contact(x);
    const float deficit=c.clearance-hit.distance;
    if(deficit<=kViolationEps) return false;

    float denom=0.0f;
    if(q.i0!=0u) denom+=q.w0*q.w0;
    if(q.i1!=0u) denom+=q.w1*q.w1;
    if(denom<1e-8f) return false;
    const glm::vec3 delta=hit.normal*(std::min(deficit,0.020f)/denom);
    if(q.i0!=0u) field.d[q.i0]+=q.w0*delta;
    if(q.i1!=0u) field.d[q.i1]+=q.w1*delta;
    field.d[0]=glm::vec3(0.0f);
    ++stats.projectedConstraints;
    return true;
}

void smoothField(NodeField& field) {
    const auto old=field.d;
    field.d[0]=glm::vec3(0.0f);
    for(uint32_t i=1u;i+1u<kCorrectionNodes;++i)
        field.d[i]=0.14f*old[i-1u]+0.72f*old[i]+0.14f*old[i+1u];
    field.d[kCorrectionNodes-1u]=0.14f*old[kCorrectionNodes-2u]+0.86f*old[kCorrectionNodes-1u];
}

} // namespace

ValidationReport validateGroom(const vh::character::CharacterSurface& character,
                               const GroomAsset& groom,const GroomBuildParams& params) {
    ValidationReport r;
    const size_t denseCount=size_t(groom.strands.size())*kGuidePoints;
    if(groom.restStrandPoints.size()!=denseCount || groom.baseStrandPoints.size()!=denseCount)
        throw std::runtime_error("GroomAsset dense authoring arrays are inconsistent");

    for(size_t i=0;i<denseCount;++i) {
        if((i%kGuidePoints)==0u) continue; // follicle root is allowed to live at authored clearance
        const glm::vec3 visual=glm::vec3(groom.restStrandPoints[i].position);
        const glm::vec3 base=glm::vec3(groom.baseStrandPoints[i].position);
        const float dv=exactOrSafeDistance(character,visual,0.0f);
        const float db=exactOrSafeDistance(character,base,0.0f);
        r.minVisualDistance=std::min(r.minVisualDistance,dv);
        r.minBaseDistance=std::min(r.minBaseDistance,db);
        if(dv<-kViolationEps) ++r.visualInsideSamples;
        if(db<-kViolationEps) ++r.baseInsideSamples;
    }

    if(groom.guidePointMeta.size()!=groom.simulationGuides.size()*kGuidePoints)
        throw std::runtime_error("GroomAsset guide support metadata is inconsistent");
    for(uint32_t gid=0;gid<groom.simulationGuides.size();++gid) for(uint32_t p=0;p<kGuidePoints;++p) {
        const size_t idx=size_t(gid)*kGuidePoints+p;
        const float radius=std::max(0.0f,groom.guidePointMeta[idx].supportVolume.x);
        const float threshold=radius+params.collisionMargin;
        const float d=exactOrSafeDistance(character,groom.simulationGuides[gid].p[p],threshold);
        const float clearance=d-threshold;
        r.minGuideProxyClearance=std::min(r.minGuideProxyClearance,clearance);
        if(clearance<-kViolationEps) ++r.guideProxyViolations;
    }
    r.deepestPenetration=std::max({0.0f,-r.minVisualDistance,-r.minBaseDistance,-r.minGuideProxyClearance});
    return r;
}

ContactResolveReport resolveContacts(const vh::character::CharacterSurface& character,
                                     const GroomBuildParams& params,GroomAsset& groom) {
    const size_t strandCount=groom.strands.size();
    const size_t denseCount=strandCount*kGuidePoints;
    if(groom.restStrandPoints.size()!=denseCount || groom.baseStrandPoints.size()!=denseCount ||
       groom.strandPathU.size()!=denseCount || groom.strandContactGroupTags.size()!=strandCount)
        throw std::runtime_error("GroomAsset is missing path-parameter authoring data");

    std::unordered_map<uint64_t,std::vector<uint32_t>> members;
    members.reserve(groom.audit.activeContactGroups*2u+1u);
    for(uint32_t sid=0;sid<strandCount;++sid) members[groom.strandContactGroupTags[sid]].push_back(sid);

    std::unordered_map<uint64_t,std::vector<uint32_t>> guides;
    guides.reserve(members.size());
    for(uint32_t gid=0;gid<groom.simulationGuides.size();++gid) guides[groom.simulationGuides[gid].contactGroupTag].push_back(gid);

    ContactResolveReport stats;
    std::unordered_map<uint64_t,NodeField> solved;
    solved.reserve(members.size());
    const float diag=std::sqrt(3.0f)*character.contactField().cellSize;

    for(const auto& [tag,sids]:members) {
        std::vector<ConstraintSample> constraints;
        constraints.reserve(sids.size()*kGuidePoints/2u);
        const float denseTarget=std::max(0.0f,params.minStrandClearance);
        for(uint32_t sid:sids) for(uint32_t p=1;p<kGuidePoints;++p) {
            const size_t idx=size_t(sid)*kGuidePoints+p;
            const float s=groom.strandPathU[idx];
            const glm::vec3 base=glm::vec3(groom.baseStrandPoints[idx].position);
            const glm::vec3 visual=glm::vec3(groom.restStrandPoints[idx].position);
            const float fb=character.contactField().sample(base);
            const float fv=character.contactField().sample(visual);
            if(std::isfinite(fb) && fb<=denseTarget+diag) constraints.push_back({base,s,denseTarget});
            if(std::isfinite(fv) && fv<=denseTarget+diag) constraints.push_back({visual,s,denseTarget});
        }
        if(const auto it=guides.find(tag);it!=guides.end()) for(uint32_t gid:it->second) for(uint32_t p=1;p<kGuidePoints;++p) {
            const size_t idx=size_t(gid)*kGuidePoints+p;
            const float clearance=std::max(0.0f,groom.guidePointMeta[idx].supportVolume.x)+params.collisionMargin;
            const glm::vec3 x=groom.simulationGuides[gid].p[p];
            const float f=character.contactField().sample(x);
            if(std::isfinite(f) && f<=clearance+diag) constraints.push_back({x,groom.simulationGuides[gid].pathU[p],clearance});
        }
        if(constraints.empty()) continue;

        NodeField field{}; bool everChanged=false;
        for(uint32_t iter=0;iter<kMaxIterations;++iter) {
            bool changed=false;
            for(const auto& c:constraints) changed|=projectConstraint(character,c,field,stats);
            smoothField(field);
            // Re-project after smoothing so regularity never wins over contact.
            for(const auto& c:constraints) changed|=projectConstraint(character,c,field,stats);
            stats.iterations=std::max(stats.iterations,iter+1u);
            everChanged|=changed;
            if(!changed) break;
        }
        if(!everChanged) continue;
        ++stats.correctedGroups;
        for(const auto& d:field.d) stats.maxCorrection=std::max(stats.maxCorrection,glm::length(d));
        solved.emplace(tag,field);
    }

    if(solved.empty()) return stats;

    for(uint32_t sid=0;sid<strandCount;++sid) {
        const auto it=solved.find(groom.strandContactGroupTags[sid]); if(it==solved.end()) continue;
        for(uint32_t p=1;p<kGuidePoints;++p) {
            const size_t idx=size_t(sid)*kGuidePoints+p; const glm::vec3 d=sampleField(it->second,groom.strandPathU[idx]);
            groom.baseStrandPoints[idx].position+=glm::vec4(d,0.0f);
            groom.restStrandPoints[idx].position+=glm::vec4(d,0.0f);
        }
    }

    for(uint32_t gid=0;gid<groom.simulationGuides.size();++gid) {
        auto& g=groom.simulationGuides[gid]; const auto it=solved.find(g.contactGroupTag); if(it==solved.end()) continue;
        for(uint32_t p=1;p<kGuidePoints;++p) g.p[p]+=sampleField(it->second,g.pathU[p]);
    }

    std::vector<float> attachments(groom.simulationGuides.size()*kGuidePoints,0.0f);
    if(groom.restGuidePoints.size()==attachments.size())
        for(size_t i=0;i<attachments.size();++i) attachments[i]=groom.restGuidePoints[i].position.w;

    groom.restGuidePoints.clear();
    groom.restGuidePoints.reserve(groom.simulationGuides.size()*kGuidePoints);
    for(uint32_t gid=0;gid<groom.simulationGuides.size();++gid) {
        const auto& g=groom.simulationGuides[gid];
        for(uint32_t p=0;p<kGuidePoints;++p) {
            const size_t idx=size_t(gid)*kGuidePoints+p;
            StrandPointGpu gp; gp.position=glm::vec4(g.p[p],attachments[idx]);
            groom.restGuidePoints.push_back(gp);
            const uint32_t a=p>0u?p-1u:p,b=p+1u<kGuidePoints?p+1u:p;
            const float radius=groom.guidePointMeta[idx].supportVolume.x;
            const float segment=std::max(0.001f,0.5f*glm::length(g.p[b]-g.p[a]));
            groom.guidePointMeta[idx].supportVolume.y=PI*radius*radius*segment;
        }
    }
    groom.restGuideFrames=buildGuideFrames(groom.simulationGuides);
    rebuildClusters(groom);
    return stats;
}

} // namespace vh::groom
