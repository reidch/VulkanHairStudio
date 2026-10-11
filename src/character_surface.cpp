#include "character_surface.hpp"
#include "scalp_domain.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace vh::character {
namespace {

constexpr char kFieldMagic[8]={'V','3','6','C','F','0','1','\0'};
constexpr uint32_t kFieldVersion=1u;

float sqr(float x){ return x*x; }

glm::vec3 safeNorm(glm::vec3 v,glm::vec3 fallback={0,1,0}) {
    const float l2=glm::dot(v,v);
    return l2>1e-12f?v/std::sqrt(l2):fallback;
}

uint32_t parseIndex(const std::string& token,size_t vertexCount) {
    const auto slash=token.find('/');
    const std::string head=slash==std::string::npos?token:token.substr(0,slash);
    const long v=std::stol(head);
    if(v==0) throw std::runtime_error("OBJ index 0 is invalid");
    const long idx=v>0?v-1:long(vertexCount)+v;
    if(idx<0 || idx>=long(vertexCount)) throw std::runtime_error("OBJ face references invalid vertex");
    return static_cast<uint32_t>(idx);
}

bool ictExternalSolidMaterial(const std::string& material) {
    // ICT FaceKit's external human shell is authored as exactly these two
    // components. Eyes, teeth, gums/tongue, lacrimal fluid, eyelashes, etc.
    // remain collision obstacles but do not define human-body inside/outside.
    return material=="M_Face" || material=="M_BackHead";
}

glm::vec3 closestPointTriangle(glm::vec3 p,glm::vec3 a,glm::vec3 b,glm::vec3 c) {
    const glm::vec3 ab=b-a, ac=c-a, ap=p-a;
    const float d1=glm::dot(ab,ap), d2=glm::dot(ac,ap);
    if(d1<=0.0f && d2<=0.0f) return a;
    const glm::vec3 bp=p-b;
    const float d3=glm::dot(ab,bp), d4=glm::dot(ac,bp);
    if(d3>=0.0f && d4<=d3) return b;
    const float vc=d1*d4-d3*d2;
    if(vc<=0.0f && d1>=0.0f && d3<=0.0f) return a+(d1/(d1-d3))*ab;
    const glm::vec3 cp=p-c;
    const float d5=glm::dot(ab,cp), d6=glm::dot(ac,cp);
    if(d6>=0.0f && d5<=d6) return c;
    const float vb=d5*d2-d1*d6;
    if(vb<=0.0f && d2>=0.0f && d6<=0.0f) return a+(d2/(d2-d6))*ac;
    const float va=d3*d6-d5*d4;
    if(va<=0.0f && (d4-d3)>=0.0f && (d5-d6)>=0.0f) {
        const float w=(d4-d3)/((d4-d3)+(d5-d6)); return b+w*(c-b);
    }
    const float inv=1.0f/(va+vb+vc);
    return a+ab*(vb*inv)+ac*(vc*inv);
}

float pointAabbDistance2(glm::vec3 p,glm::vec3 mn,glm::vec3 mx) {
    float d2=0.0f;
    for(int k=0;k<3;++k) {
        if(p[k]<mn[k]) d2+=sqr(mn[k]-p[k]);
        else if(p[k]>mx[k]) d2+=sqr(p[k]-mx[k]);
    }
    return d2;
}

bool rayAabb(glm::vec3 o,glm::vec3 d,glm::vec3 mn,glm::vec3 mx,float maxT) {
    float tmin=0.0f,tmax=maxT;
    for(int k=0;k<3;++k) {
        if(std::abs(d[k])<1e-8f) {
            if(o[k]<mn[k] || o[k]>mx[k]) return false;
            continue;
        }
        const float inv=1.0f/d[k];
        float t0=(mn[k]-o[k])*inv,t1=(mx[k]-o[k])*inv;
        if(t0>t1) std::swap(t0,t1);
        tmin=std::max(tmin,t0); tmax=std::min(tmax,t1);
        if(tmin>tmax) return false;
    }
    return true;
}

bool rayTriangle(glm::vec3 o,glm::vec3 d,glm::vec3 a,glm::vec3 b,glm::vec3 c,
                 float& t,glm::vec3& bary) {
    const glm::vec3 e1=b-a,e2=c-a;
    const glm::vec3 p=glm::cross(d,e2);
    const float det=glm::dot(e1,p);
    if(std::abs(det)<1e-9f) return false;
    const float inv=1.0f/det;
    const glm::vec3 s=o-a;
    const float u=glm::dot(s,p)*inv;
    if(u<0.0f || u>1.0f) return false;
    const glm::vec3 q=glm::cross(s,e1);
    const float v=glm::dot(d,q)*inv;
    if(v<0.0f || u+v>1.0f) return false;
    const float hit=glm::dot(e2,q)*inv;
    if(hit<=1e-6f) return false;
    t=hit; bary={1.0f-u-v,u,v}; return true;
}

void fnvBytes(uint64_t& h,const void* ptr,size_t n) {
    const auto* b=static_cast<const unsigned char*>(ptr);
    for(size_t i=0;i<n;++i) { h^=uint64_t(b[i]); h*=1099511628211ull; }
}

} // namespace

bool ContactField::contains(glm::vec3 p) const {
    if(!valid()) return false;
    const glm::vec3 g=(p-origin)/cellSize;
    return g.x>=0.0f&&g.y>=0.0f&&g.z>=0.0f&&
           g.x<=float(dim-1u)&&g.y<=float(dim-1u)&&g.z<=float(dim-1u);
}

float ContactField::sample(glm::vec3 p) const {
    if(!contains(p)) return std::numeric_limits<float>::infinity();
    const glm::vec3 g=(p-origin)/cellSize;
    glm::ivec3 b=glm::min(glm::ivec3(glm::floor(g)),glm::ivec3(int(dim)-2));
    b=glm::max(b,glm::ivec3(0));
    const glm::vec3 f=glm::clamp(g-glm::vec3(b),glm::vec3(0.0f),glm::vec3(1.0f));
    auto at=[&](int x,int y,int z){ return distance[size_t(x)+size_t(dim)*(size_t(y)+size_t(dim)*size_t(z))]; };
    float out=0.0f;
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        const float wx=x?f.x:1.0f-f.x, wy=y?f.y:1.0f-f.y, wz=z?f.z:1.0f-f.z;
        out+=wx*wy*wz*at(b.x+x,b.y+y,b.z+z);
    }
    return out;
}

glm::vec3 ContactField::gradient(glm::vec3 p) const {
    if(!contains(p)) return {0,1,0};
    const float h=std::max(cellSize,1e-4f);
    const auto finiteSample=[&](glm::vec3 q,float fallback){
        const float d=sample(q); return std::isfinite(d)?d:fallback;
    };
    const float c=sample(p);
    return safeNorm(glm::vec3(finiteSample(p+glm::vec3(h,0,0),c)-finiteSample(p-glm::vec3(h,0,0),c),
                              finiteSample(p+glm::vec3(0,h,0),c)-finiteSample(p-glm::vec3(0,h,0),c),
                              finiteSample(p+glm::vec3(0,0,h),c)-finiteSample(p-glm::vec3(0,0,h),c)));
}

CharacterSurface CharacterSurface::loadObj(const std::filesystem::path& objPath,
                                            const std::filesystem::path& contactFieldCachePath,
                                            float targetCellSize) {
    CharacterSurface out;
    std::ifstream f(objPath);
    if(!f) throw std::runtime_error("Cannot open character OBJ: "+objPath.string());

    std::vector<glm::vec3> positions;
    struct PendingTri { glm::uvec3 index; std::string material; };
    std::vector<PendingTri> pending;
    std::string material,line;
    while(std::getline(f,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream ss(line); std::string tag; ss>>tag;
        if(tag=="v") {
            glm::vec3 p; ss>>p.x>>p.y>>p.z; positions.push_back(p);
        } else if(tag=="usemtl") {
            ss>>material;
        } else if(tag=="f") {
            std::vector<uint32_t> ids; std::string tok;
            while(ss>>tok) ids.push_back(parseIndex(tok,positions.size()));
            for(size_t k=1;k+1<ids.size();++k) pending.push_back({{ids[0],ids[k],ids[k+1]},material});
        }
    }
    if(positions.empty() || pending.empty()) throw std::runtime_error("Character OBJ has no geometry");

    glm::vec3 rawMin(std::numeric_limits<float>::max()),rawMax(-std::numeric_limits<float>::max());
    for(const auto& p:positions){ rawMin=glm::min(rawMin,p); rawMax=glm::max(rawMax,p); }
    const glm::vec3 rawExtent=rawMax-rawMin;
    const float largest=std::max(rawExtent.x,std::max(rawExtent.y,rawExtent.z));
    const float unitScale=largest>2.0f?0.01f:1.0f;
    for(auto& p:positions) p*=unitScale;

    float topY=-std::numeric_limits<float>::max();
    for(const auto& q:pending) for(uint32_t id:{q.index.x,q.index.y,q.index.z}) topY=std::max(topY,positions[id].y);
    glm::vec3 headMin(std::numeric_limits<float>::max()),headMax(-std::numeric_limits<float>::max());
    for(const auto& q:pending) for(uint32_t id:{q.index.x,q.index.y,q.index.z}) {
        const auto p=positions[id]; if(p.y<topY-0.205f) continue;
        headMin=glm::min(headMin,p); headMax=glm::max(headMax,p);
    }
    if(!std::isfinite(headMin.x)) throw std::runtime_error("Unable to identify head surface");
    const glm::vec3 cranialOrigin{0.5f*(headMin.x+headMax.x),topY-0.095f,0.5f*(headMin.z+headMax.z)-0.002f};
    for(auto& p:positions) p-=cranialOrigin;

    out.vertices_.resize(positions.size());
    for(size_t i=0;i<positions.size();++i) out.vertices_[i].position=positions[i];

    bool hasExplicitIctShell=false;
    for(const auto& q:pending) hasExplicitIctShell|=ictExternalSolidMaterial(q.material);
    out.triangles_.reserve(pending.size());
    for(const auto& q:pending) {
        Triangle t; t.index=q.index; t.visible=true; t.collision=true;
        t.solidBoundary=hasExplicitIctShell?ictExternalSolidMaterial(q.material):true;
        out.triangles_.push_back(t);
    }

    out.finalize();
    out.buildBvhs();
    out.meshFingerprint_=out.computeMeshFingerprint();
    out.loadOrBuildContactField(contactFieldCachePath,targetCellSize);
    return out;
}

void CharacterSurface::finalize() {
    renderIndices_.clear(); scalpTriangles_.clear(); collisionTriangles_.clear(); solidTriangles_.clear();
    for(auto& v:vertices_) v.normal=glm::vec3(0.0f);
    std::vector<glm::vec3> seamSourceNormal(vertices_.size(),glm::vec3(0.0f));
    std::vector<uint8_t> usedVertex(vertices_.size(),uint8_t(0));
    glm::vec3 scalpMin(std::numeric_limits<float>::max()),scalpMax(-std::numeric_limits<float>::max());

    for(uint32_t ti=0;ti<triangles_.size();++ti) {
        auto& t=triangles_[ti];
        const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
        const glm::vec3 n=glm::cross(b-a,c-a);
        if(t.visible) {
            renderIndices_.insert(renderIndices_.end(),{t.index.x,t.index.y,t.index.z});
            vertices_[t.index.x].normal+=n; vertices_[t.index.y].normal+=n; vertices_[t.index.z].normal+=n;
        }
        if(!t.collision) continue;
        collisionTriangles_.push_back(ti);
        if(t.solidBoundary) solidTriangles_.push_back(ti);
        for(uint32_t id:{t.index.x,t.index.y,t.index.z}) { seamSourceNormal[id]+=n; usedVertex[id]=1u; }
        const glm::vec3 center=(a+b+c)/3.0f;
        t.scalp=t.solidBoundary && vh::scalp::containsDirection(center,0.045f);
        if(t.scalp) {
            scalpTriangles_.push_back(ti);
            scalpMin=glm::min(scalpMin,glm::min(a,glm::min(b,c)));
            scalpMax=glm::max(scalpMax,glm::max(a,glm::max(b,c)));
        }
    }

    using SeamKey=std::array<long long,3>;
    constexpr double q=1.0e6;
    std::map<SeamKey,glm::vec3> seamNormal;
    for(size_t i=0;i<vertices_.size();++i) if(usedVertex[i]) {
        const auto p=vertices_[i].position;
        seamNormal[{std::llround(double(p.x)*q),std::llround(double(p.y)*q),std::llround(double(p.z)*q)}]+=seamSourceNormal[i];
    }
    for(size_t i=0;i<vertices_.size();++i) {
        if(usedVertex[i]) {
            const auto p=vertices_[i].position;
            vertices_[i].normal=safeNorm(seamNormal[{std::llround(double(p.x)*q),std::llround(double(p.y)*q),std::llround(double(p.z)*q)}]);
        } else vertices_[i].normal=safeNorm(vertices_[i].normal);
    }

    if(collisionTriangles_.empty()) throw std::runtime_error("Character has no collision surface");
    if(solidTriangles_.empty()) throw std::runtime_error("Character has no external solid boundary");
    if(scalpTriangles_.empty()) throw std::runtime_error("Character scalp domain did not intersect external head shell");
    frame_.halfExtent=glm::max(0.5f*(scalpMax-scalpMin),glm::vec3(0.025f));
}

uint32_t CharacterSurface::buildBvhNode(std::vector<uint32_t>& order,std::vector<BvhNode>& nodes,
                                        uint32_t first,uint32_t count) {
    BvhNode node;
    node.bmin=glm::vec3(std::numeric_limits<float>::max());
    node.bmax=glm::vec3(-std::numeric_limits<float>::max());
    glm::vec3 cmin=node.bmin,cmax=node.bmax;
    for(uint32_t i=0;i<count;++i) {
        const auto& t=triangles_[order[first+i]];
        const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
        node.bmin=glm::min(node.bmin,glm::min(a,glm::min(b,c)));
        node.bmax=glm::max(node.bmax,glm::max(a,glm::max(b,c)));
        const glm::vec3 cc=(a+b+c)/3.0f; cmin=glm::min(cmin,cc); cmax=glm::max(cmax,cc);
    }
    const uint32_t index=static_cast<uint32_t>(nodes.size()); nodes.push_back(node);
    if(count<=8u) { nodes[index].first=first; nodes[index].count=count; return index; }
    const glm::vec3 e=cmax-cmin; int axis=0; if(e.y>e.x) axis=1; if(e.z>e[axis]) axis=2;
    const uint32_t mid=first+count/2u;
    std::nth_element(order.begin()+first,order.begin()+mid,order.begin()+first+count,[&](uint32_t ta,uint32_t tb){
        const auto& a=triangles_[ta]; const auto& b=triangles_[tb];
        const glm::vec3 ca=(vertices_[a.index.x].position+vertices_[a.index.y].position+vertices_[a.index.z].position)/3.0f;
        const glm::vec3 cb=(vertices_[b.index.x].position+vertices_[b.index.y].position+vertices_[b.index.z].position)/3.0f;
        return ca[axis]<cb[axis];
    });
    const uint32_t left=buildBvhNode(order,nodes,first,mid-first);
    const uint32_t right=buildBvhNode(order,nodes,mid,first+count-mid);
    nodes[index].left=left; nodes[index].right=right; nodes[index].count=0u;
    return index;
}

void CharacterSurface::buildBvhs() {
    bvhOrder_=collisionTriangles_; bvh_.clear(); bvh_.reserve(bvhOrder_.size()*2u);
    buildBvhNode(bvhOrder_,bvh_,0u,static_cast<uint32_t>(bvhOrder_.size()));
    solidBvhOrder_=solidTriangles_; solidBvh_.clear(); solidBvh_.reserve(solidBvhOrder_.size()*2u);
    buildBvhNode(solidBvhOrder_,solidBvh_,0u,static_cast<uint32_t>(solidBvhOrder_.size()));
}

float CharacterSurface::nearestTriangle(glm::vec3 p,const std::vector<uint32_t>& order,
                                        const std::vector<BvhNode>& nodes,
                                        uint32_t& tri,glm::vec3& closest,glm::vec3& normalOut) const {
    float best2=std::numeric_limits<float>::max(); tri=0u;
    std::vector<uint32_t> stack; stack.reserve(64); stack.push_back(0u);
    while(!stack.empty()) {
        const uint32_t ni=stack.back(); stack.pop_back(); const auto& n=nodes[ni];
        if(pointAabbDistance2(p,n.bmin,n.bmax)>=best2) continue;
        if(n.leaf()) {
            for(uint32_t k=0;k<n.count;++k) {
                const uint32_t ti=order[n.first+k]; const auto& t=triangles_[ti];
                const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
                const glm::vec3 q=closestPointTriangle(p,a,b,c); const float d2=glm::dot(p-q,p-q);
                if(d2<best2) { best2=d2; tri=ti; closest=q; normalOut=safeNorm(glm::cross(b-a,c-a)); }
            }
        } else { stack.push_back(n.left); stack.push_back(n.right); }
    }
    return std::sqrt(best2);
}

bool CharacterSurface::raycast(glm::vec3 origin,glm::vec3 direction,uint32_t& tri,glm::vec3& bary,float& hitT) const {
    direction=safeNorm(direction,{0,1,0}); hitT=std::numeric_limits<float>::max(); bool hit=false;
    std::vector<uint32_t> stack; stack.reserve(64); stack.push_back(0u);
    while(!stack.empty()) {
        const uint32_t ni=stack.back(); stack.pop_back(); const auto& n=solidBvh_[ni];
        if(!rayAabb(origin,direction,n.bmin,n.bmax,hitT)) continue;
        if(n.leaf()) {
            for(uint32_t k=0;k<n.count;++k) {
                const uint32_t ti=solidBvhOrder_[n.first+k]; const auto& t=triangles_[ti];
                const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
                float tt=0.0f; glm::vec3 bb;
                if(rayTriangle(origin,direction,a,b,c,tt,bb) && tt<hitT) { hit=true; hitT=tt; tri=ti; bary=bb; }
            }
        } else { stack.push_back(n.left); stack.push_back(n.right); }
    }
    return hit;
}

uint32_t CharacterSurface::countSolidRayCrossings(glm::vec3 origin,glm::vec3 direction) const {
    direction=safeNorm(direction,{1,0,0});
    std::vector<float> hits; hits.reserve(32u);
    std::vector<uint32_t> stack; stack.reserve(64u); stack.push_back(0u);
    const float inf=std::numeric_limits<float>::max();
    while(!stack.empty()) {
        const uint32_t ni=stack.back(); stack.pop_back(); const auto& node=solidBvh_[ni];
        if(!rayAabb(origin,direction,node.bmin,node.bmax,inf)) continue;
        if(node.leaf()) {
            for(uint32_t k=0;k<node.count;++k) {
                const uint32_t ti=solidBvhOrder_[node.first+k]; const auto& t=triangles_[ti];
                const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
                float tt=0.0f; glm::vec3 bary; if(rayTriangle(origin,direction,a,b,c,tt,bary)) hits.push_back(tt);
            }
        } else { stack.push_back(node.left); stack.push_back(node.right); }
    }
    if(hits.empty()) return 0u;
    std::sort(hits.begin(),hits.end());
    constexpr float kMergeT=1.0e-5f;
    uint32_t unique=1u; float previous=hits.front();
    for(size_t i=1;i<hits.size();++i) if(std::abs(hits[i]-previous)>kMergeT) { ++unique; previous=hits[i]; }
    return unique;
}

bool CharacterSurface::insideSolid(glm::vec3 p) const {
    static const std::array<glm::vec3,3> directions{
        safeNorm(glm::vec3(1.0f,0.371f,0.219f)),
        safeNorm(glm::vec3(-0.283f,1.0f,0.417f)),
        safeNorm(glm::vec3(0.193f,-0.337f,1.0f))
    };
    uint32_t votes=0u; for(const auto d:directions) votes+=(countSolidRayCrossings(p,d)&1u);
    return votes>=2u;
}

SurfaceContact CharacterSurface::contact(glm::vec3 p) const {
    SurfaceContact out;
    uint32_t solidTri=0u, allTri=0u; glm::vec3 qSolid,nSolid,qAll,nAll;
    const float dSolid=nearestTriangle(p,solidBvhOrder_,solidBvh_,solidTri,qSolid,nSolid);
    const float dAll=nearestTriangle(p,bvhOrder_,bvh_,allTri,qAll,nAll);
    out.insideSolid=insideSolid(p);
    if(out.insideSolid) {
        out.distance=-dSolid; out.triangle=solidTri; out.nearestIsObstacle=false;
        out.normal=safeNorm(qSolid-p,nSolid);
    } else if(dAll+1e-7f<dSolid) {
        out.distance=dAll; out.triangle=allTri; out.nearestIsObstacle=!triangles_[allTri].solidBoundary;
        out.normal=safeNorm(p-qAll,nAll);
    } else {
        out.distance=dSolid; out.triangle=solidTri; out.nearestIsObstacle=false;
        out.normal=safeNorm(p-qSolid,nSolid);
    }
    return out;
}

RayBindingResult CharacterSurface::bindScalpDirection(glm::vec3 canonicalDirection) const {
    RayBindingResult out; canonicalDirection=safeNorm(canonicalDirection,{0,1,0});
    uint32_t tri=0u; glm::vec3 bary; float t=0.0f;
    if(raycast(glm::vec3(0.0f),canonicalDirection,tri,bary,t)) {
        out.binding.triangle=tri; out.binding.barycentric=bary; out.exactRayHit=true; return out;
    }
    uint32_t nearest=0u; glm::vec3 q,n;
    nearestTriangle(canonicalDirection*0.11f,solidBvhOrder_,solidBvh_,nearest,q,n);
    const auto& tr=triangles_[nearest];
    const glm::vec3 a=vertices_[tr.index.x].position,b=vertices_[tr.index.y].position,c=vertices_[tr.index.z].position;
    const glm::vec3 v0=b-a,v1=c-a,v2=q-a;
    const float d00=glm::dot(v0,v0),d01=glm::dot(v0,v1),d11=glm::dot(v1,v1),d20=glm::dot(v2,v0),d21=glm::dot(v2,v1);
    const float denom=std::max(1e-12f,d00*d11-d01*d01);
    const float v=(d11*d20-d01*d21)/denom,w=(d00*d21-d01*d20)/denom;
    out.binding.triangle=nearest; out.binding.barycentric={1.0f-v-w,v,w}; out.exactRayHit=false; return out;
}

glm::vec3 CharacterSurface::position(const RootBinding& b) const {
    const auto& t=triangles_.at(b.triangle);
    return b.barycentric.x*vertices_[t.index.x].position+b.barycentric.y*vertices_[t.index.y].position+b.barycentric.z*vertices_[t.index.z].position;
}

glm::vec3 CharacterSurface::normal(const RootBinding& b) const {
    const auto& t=triangles_.at(b.triangle);
    return safeNorm(b.barycentric.x*vertices_[t.index.x].normal+b.barycentric.y*vertices_[t.index.y].normal+b.barycentric.z*vertices_[t.index.z].normal);
}

uint64_t CharacterSurface::computeMeshFingerprint() const {
    uint64_t h=1469598103934665603ull;
    for(const auto& v:vertices_) fnvBytes(h,&v.position,sizeof(v.position));
    for(const auto& t:triangles_) {
        fnvBytes(h,&t.index,sizeof(t.index));
        const uint8_t semantics=uint8_t(t.collision?1u:0u)|uint8_t(t.solidBoundary?2u:0u);
        fnvBytes(h,&semantics,sizeof(semantics));
    }
    return h;
}

void CharacterSurface::loadOrBuildContactField(const std::filesystem::path& cachePath,float targetCellSize) {
    const auto fieldT0=std::chrono::steady_clock::now();
    targetCellSize=std::clamp(targetCellSize,0.0015f,0.0020f);
    glm::vec3 mn(std::numeric_limits<float>::max()),mx(-std::numeric_limits<float>::max());
    for(uint32_t ti:collisionTriangles_) {
        const auto& t=triangles_[ti];
        for(uint32_t id:{t.index.x,t.index.y,t.index.z}) { mn=glm::min(mn,vertices_[id].position); mx=glm::max(mx,vertices_[id].position); }
    }
    const glm::vec3 center=0.5f*(mn+mx); const glm::vec3 ext=mx-mn;
    const float side=std::max(ext.x,std::max(ext.y,ext.z))+0.060f;
    const uint32_t desiredDim=std::clamp(uint32_t(std::ceil(side/targetCellSize))+1u,96u,256u);
    const float desiredCell=side/float(desiredDim-1u);
    const glm::vec3 desiredOrigin=center-glm::vec3(0.5f*side);

    {
        std::ifstream s(cachePath,std::ios::binary);
        if(s) {
            char magic[8]{}; uint32_t version=0u,dim=0u; uint64_t fingerprint=0u,count=0u; glm::vec3 origin{}; float cell=0.0f;
            s.read(magic,8); s.read(reinterpret_cast<char*>(&version),sizeof(version));
            s.read(reinterpret_cast<char*>(&fingerprint),sizeof(fingerprint)); s.read(reinterpret_cast<char*>(&dim),sizeof(dim));
            s.read(reinterpret_cast<char*>(&origin),sizeof(origin)); s.read(reinterpret_cast<char*>(&cell),sizeof(cell));
            s.read(reinterpret_cast<char*>(&count),sizeof(count));
            const bool layoutOk=dim==desiredDim && glm::length(origin-desiredOrigin)<1e-6f && std::abs(cell-desiredCell)<1e-7f;
            if(std::memcmp(magic,kFieldMagic,8)==0 && version==kFieldVersion && fingerprint==meshFingerprint_ && layoutOk && count==uint64_t(dim)*dim*dim) {
                contactField_.dim=dim; contactField_.origin=origin; contactField_.cellSize=cell; contactField_.distance.resize(size_t(count));
                s.read(reinterpret_cast<char*>(contactField_.distance.data()),std::streamsize(count*sizeof(float)));
                if(s && contactField_.valid()) {
                    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fieldT0).count();
                    std::cout << "ContactField cache hit: " << cachePath.string() << " | " << ms << " ms\n";
                    return;
                }
                contactField_={};
            }
        }
    }

    contactField_.dim=desiredDim; contactField_.origin=desiredOrigin; contactField_.cellSize=desiredCell;
    const size_t voxelCount=size_t(desiredDim)*desiredDim*desiredDim;
    contactField_.distance.resize(voxelCount);
    std::cout << "Building ContactField " << desiredDim << "^3 (" << 1000.0f*desiredCell
              << " mm cells; three-axis scanline sign + exact nearest-surface distances)...\n";

    // First-principles acceleration: inside/outside is a topological parity
    // property, so do not launch three independent BVH rays from every voxel.
    // One ray traversal gives all crossings for an entire grid line. Repeat on
    // the three principal axes and retain the old majority-vote robustness.
    // This reduces sign work from O(3*N^3 ray queries) to O(3*N^2 scanlines).
    std::vector<uint8_t> insideVotes(voxelCount,uint8_t(0));
    const uint32_t hw=std::max(1u,std::thread::hardware_concurrency());
    const uint32_t workers=std::max(1u,std::min(desiredDim,hw));
    auto voxelIndex=[desiredDim](uint32_t x,uint32_t y,uint32_t z) {
        return size_t(x)+size_t(desiredDim)*(size_t(y)+size_t(desiredDim)*size_t(z));
    };
    const auto signT0=std::chrono::steady_clock::now();
    for(int axis=0;axis<3;++axis) {
        const int uAxis=(axis+1)%3, vAxis=(axis+2)%3;
        std::atomic<uint32_t> nextLine{0u};
        std::vector<std::thread> signPool; signPool.reserve(workers);
        for(uint32_t worker=0;worker<workers;++worker) signPool.emplace_back([&,axis,uAxis,vAxis]{
            std::vector<float> hits; hits.reserve(64u);
            for(;;) {
                const uint32_t line=nextLine.fetch_add(1u,std::memory_order_relaxed);
                if(line>=desiredDim*desiredDim) break;
                const uint32_t u=line%desiredDim, v=line/desiredDim;
                glm::vec3 origin=desiredOrigin;
                origin[axis]-=2.0f*desiredCell;
                origin[uAxis]+=float(u)*desiredCell;
                origin[vAxis]+=float(v)*desiredCell;
                // Tiny deterministic sub-voxel offset avoids rays exactly on
                // shared triangle edges without changing voxel topology.
                const float eps=1.0e-4f*desiredCell;
                origin[uAxis]+=eps*float(int((line*17u+uint32_t(axis)*13u)%7u)-3);
                origin[vAxis]+=eps*float(int((line*29u+uint32_t(axis)*11u)%7u)-3);
                glm::vec3 direction(0.0f); direction[axis]=1.0f;
                hits.clear();
                std::vector<uint32_t> stack; stack.reserve(64u); stack.push_back(0u);
                const float inf=std::numeric_limits<float>::max();
                while(!stack.empty()) {
                    const uint32_t ni=stack.back(); stack.pop_back(); const auto& node=solidBvh_[ni];
                    if(!rayAabb(origin,direction,node.bmin,node.bmax,inf)) continue;
                    if(node.leaf()) {
                        for(uint32_t k=0;k<node.count;++k) {
                            const uint32_t ti=solidBvhOrder_[node.first+k]; const auto& t=triangles_[ti];
                            const glm::vec3 a=vertices_[t.index.x].position,b=vertices_[t.index.y].position,c=vertices_[t.index.z].position;
                            float tt=0.0f; glm::vec3 bary;
                            if(rayTriangle(origin,direction,a,b,c,tt,bary)) hits.push_back(tt);
                        }
                    } else { stack.push_back(node.left); stack.push_back(node.right); }
                }
                std::sort(hits.begin(),hits.end());
                constexpr float kMergeT=1.0e-5f;
                size_t write=0u;
                for(float h:hits) if(write==0u || std::abs(h-hits[write-1u])>kMergeT) hits[write++]=h;
                hits.resize(write);
                size_t crossings=0u;
                for(uint32_t i=0;i<desiredDim;++i) {
                    const float sampleT=2.0f*desiredCell+float(i)*desiredCell;
                    while(crossings<hits.size() && hits[crossings]<sampleT) ++crossings;
                    if((crossings&1u)==0u) continue;
                    uint32_t xyz[3]{}; xyz[axis]=i; xyz[uAxis]=u; xyz[vAxis]=v;
                    ++insideVotes[voxelIndex(xyz[0],xyz[1],xyz[2])];
                }
            }
        });
        for(auto& t:signPool) t.join();
    }
    const double signMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-signT0).count();

    // Distances are geometric rather than topological, so retain exact BVH
    // nearest-triangle queries for the external shell and all collision
    // obstacles. Only the redundant per-voxel ray casting has been removed.
    const auto distanceT0=std::chrono::steady_clock::now();
    std::atomic<uint32_t> nextZ{0u};
    std::vector<std::thread> pool; pool.reserve(workers);
    for(uint32_t worker=0;worker<workers;++worker) pool.emplace_back([&]{
        for(;;) {
            const uint32_t z=nextZ.fetch_add(1u,std::memory_order_relaxed);
            if(z>=desiredDim) break;
            for(uint32_t y=0;y<desiredDim;++y) for(uint32_t x=0;x<desiredDim;++x) {
                const glm::vec3 p=desiredOrigin+desiredCell*glm::vec3(float(x),float(y),float(z));
                uint32_t solidTri=0u,allTri=0u; glm::vec3 qSolid,nSolid,qAll,nAll;
                const float dSolid=nearestTriangle(p,solidBvhOrder_,solidBvh_,solidTri,qSolid,nSolid);
                const float dAll=nearestTriangle(p,bvhOrder_,bvh_,allTri,qAll,nAll);
                const size_t idx=voxelIndex(x,y,z);
                if(insideVotes[idx]>=2u) contactField_.distance[idx]=-dSolid;
                else contactField_.distance[idx]=(dAll+1.0e-7f<dSolid)?dAll:dSolid;
            }
        }
    });
    for(auto& t:pool) t.join();
    const double distanceMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-distanceT0).count();
    std::cout << "ContactField phases: sign=" << signMs << " ms, distance=" << distanceMs << " ms\n";

    if(!cachePath.empty()) {
        std::error_code ec; std::filesystem::create_directories(cachePath.parent_path(),ec);
        std::ofstream s(cachePath,std::ios::binary);
        if(s) {
            const uint64_t count=uint64_t(contactField_.distance.size());
            s.write(kFieldMagic,8); s.write(reinterpret_cast<const char*>(&kFieldVersion),sizeof(kFieldVersion));
            s.write(reinterpret_cast<const char*>(&meshFingerprint_),sizeof(meshFingerprint_));
            s.write(reinterpret_cast<const char*>(&contactField_.dim),sizeof(contactField_.dim));
            s.write(reinterpret_cast<const char*>(&contactField_.origin),sizeof(contactField_.origin));
            s.write(reinterpret_cast<const char*>(&contactField_.cellSize),sizeof(contactField_.cellSize));
            s.write(reinterpret_cast<const char*>(&count),sizeof(count));
            s.write(reinterpret_cast<const char*>(contactField_.distance.data()),std::streamsize(count*sizeof(float)));
        }
    }
    const double fieldMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fieldT0).count();
    std::cout << "ContactField built and cached: " << fieldMs << " ms | " << cachePath.string() << "\n";
}

} // namespace vh::character
