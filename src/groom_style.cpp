#include "groom_style.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace vh::groom {
namespace {

using json = nlohmann::json;

float saturate(float x) { return std::clamp(x,0.0f,1.0f); }
float smooth01(float x) { x=saturate(x); return x*x*(3.0f-2.0f*x); }

glm::vec3 safeNorm(glm::vec3 v, glm::vec3 fallback = glm::vec3(0.0f, 1.0f, 0.0f)) {
    const float l2=glm::dot(v,v);
    return l2>1e-12f?v/std::sqrt(l2):fallback;
}

glm::vec3 jvec3(const json& j, glm::vec3 fallback = glm::vec3(0.0f)) {
    if(!j.is_array() || j.size()!=3u) return fallback;
    return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}

glm::vec2 jvec2(const json& j, glm::vec2 fallback = glm::vec2(0.0f)) {
    if(!j.is_array() || j.size()!=2u) return fallback;
    return glm::vec2(j[0].get<float>(), j[1].get<float>());
}

ScalarField parseField(const json& j, ScalarField defaults) {
    if(j.is_null()) return defaults;
    defaults.base=j.value("base",defaults.base);
    defaults.minValue=j.value("min",defaults.minValue);
    defaults.maxValue=j.value("max",defaults.maxValue);
    if(defaults.minValue>defaults.maxValue) std::swap(defaults.minValue,defaults.maxValue);
    defaults.controls.clear();
    if(j.contains("controls")) for(const auto& c:j.at("controls")) {
        FieldControl fc;
        fc.centerD=safeNorm(jvec3(c.value("center",json::array({0.0f,1.0f,0.0f}))));
        fc.angularRadius=std::max(0.025f,c.value("radius",0.50f));
        fc.delta=c.value("delta",0.0f);
        defaults.controls.push_back(fc);
    }
    return defaults;
}

ClumpLevel parseClumpLevel(const json& j, ClumpLevel d) {
    if(j.is_null()) return d;
    d.cellsU=std::max(1.0f,j.value("cells_u",d.cellsU));
    d.cellsV=std::max(1.0f,j.value("cells_v",d.cellsV));
    d.siteJitter=std::clamp(j.value("site_jitter",d.siteJitter),0.0f,0.48f);
    d.rootTightness=std::clamp(j.value("root_tightness",d.rootTightness),0.0f,0.95f);
    d.midTightness=std::clamp(j.value("mid_tightness",d.midTightness),0.0f,0.95f);
    d.tipTightness=std::clamp(j.value("tip_tightness",d.tipTightness),0.0f,0.95f);
    d.crossover=std::clamp(j.value("crossover",d.crossover),0.0f,0.49f);
    d.drift=std::max(0.0f,j.value("drift",d.drift));
    d.waveAmplitude=std::max(0.0f,j.value("wave_amplitude",d.waveAmplitude));
    d.waveCycles=std::max(0.0f,j.value("wave_cycles",d.waveCycles));
    return d;
}

ClumpHierarchy parseClumps(const json& j) {
    ClumpHierarchy c;
    if(j.is_null()) return c;
    c.primary=parseClumpLevel(j.value("primary",json::object()),c.primary);
    c.secondary=parseClumpLevel(j.value("secondary",json::object()),c.secondary);
    c.tipSpread=std::max(0.0f,j.value("tip_spread",c.tipSpread));
    c.twistTurns=j.value("twist_turns",c.twistTurns);
    return c;
}

CutSurface parseCutSurface(const json& j) {
    CutSurface c;
    if(j.is_null()) return c;
    c.breakup=std::max(0.0f,j.value("breakup",c.breakup));
    c.minLength=std::clamp(j.value("min_length",c.minLength),0.10f,1.0f);
    if(j.contains("controls")) {
        const auto& a=j.at("controls");
        if(a.is_array() && a.size()==kCutSurfaceResolution) {
            for(uint32_t v=0;v<kCutSurfaceResolution;++v) {
                if(!a[v].is_array() || a[v].size()!=kCutSurfaceResolution)
                    throw std::runtime_error("CutSurface controls must be a 4x4 array");
                for(uint32_t u=0;u<kCutSurfaceResolution;++u)
                    c.controls[v*kCutSurfaceResolution+u]=std::clamp(a[v][u].get<float>(),0.10f,1.20f);
            }
        } else if(a.is_array() && a.size()==kCutSurfaceResolution*kCutSurfaceResolution) {
            for(uint32_t i=0;i<c.controls.size();++i) c.controls[i]=std::clamp(a[i].get<float>(),0.10f,1.20f);
        } else throw std::runtime_error("CutSurface controls must contain 16 values");
    }
    return c;
}

StyleWarp parseStyleWarp(const json& j) {
    StyleWarp h;
    h.center=jvec2(j.value("center",json::array({0.0f,0.0f})),glm::vec2(0.0f));
    h.radius=glm::max(jvec2(j.value("radius",json::array({0.45f,0.45f})),glm::vec2(0.45f)),glm::vec2(0.04f));
    h.midDrift=jvec2(j.value("mid_drift",json::array({0.0f,0.0f})),glm::vec2(0.0f));
    h.tipDrift=jvec2(j.value("tip_drift",json::array({0.0f,0.0f})),glm::vec2(0.0f));
    h.lift=j.value("lift",h.lift);
    h.rollTurns=j.value("roll_turns",h.rollTurns);
    h.lengthScale=std::max(0.20f,j.value("length_scale",h.lengthScale));
    h.weight=std::max(0.0f,j.value("weight",h.weight));
    return h;
}

HairTexture parseTextureType(std::string s) {
    std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return char(std::tolower(c));});
    if(s=="straight") return HairTexture::Straight;
    if(s=="wavy") return HairTexture::Wavy;
    if(s=="curly") return HairTexture::Curly;
    if(s=="coily") return HairTexture::Coily;
    throw std::runtime_error("Unknown hair texture type: "+s);
}

TextureProfile parseTexture(const json& j) {
    TextureProfile t;
    if(j.is_null()) return t;
    t.type=parseTextureType(j.value("type",std::string("straight")));
    t.amplitude=std::max(0.0f,j.value("amplitude",t.amplitude));
    t.cycles=std::max(0.0f,j.value("cycles",t.cycles));
    t.start=std::clamp(j.value("start",t.start),0.0f,0.95f);
    t.irregularity=std::max(0.0f,j.value("irregularity",t.irregularity));
    t.phaseCoherence=std::clamp(j.value("phase_coherence",t.phaseCoherence),0.0f,1.0f);
    t.secondary=j.value("secondary",t.secondary);
    return t;
}

DetailProfile parseDetail(const json& j) {
    DetailProfile d;
    if(j.is_null()) return d;
    d.frizzAmplitude=std::max(0.0f,j.value("frizz_amplitude",d.frizzAmplitude));
    d.frizzFrequency=std::max(0.0f,j.value("frizz_frequency",d.frizzFrequency));
    d.frizzFraction=std::clamp(j.value("frizz_fraction",d.frizzFraction),0.0f,1.0f);
    d.flyawayAmplitude=std::max(0.0f,j.value("flyaway_amplitude",d.flyawayAmplitude));
    d.flyawayFraction=std::clamp(j.value("flyaway_fraction",d.flyawayFraction),0.0f,1.0f);
    return d;
}

uint32_t stableStyleHash(const std::string& text) {
    uint32_t h=2166136261u;
    for(unsigned char c:text) { h^=uint32_t(c); h*=16777619u; }
    return h;
}

void applyDefaultPalette(HairOptics& o,const std::string& slug,const json& material) {
    std::string palette=material.value("default_palette",std::string());
    const bool hasExplicitPigment=material.contains("color") || material.contains("melanin");
    if(palette.empty() && hasExplicitPigment) return;
    if(palette.empty() || palette=="auto") palette=(stableStyleHash(slug)&1u)?"golden-blonde":"black";
    if(palette=="black") {
        o.colorModel=HairColorModel::NaturalMelanin;
        o.directColor=glm::vec3(0.028f,0.020f,0.017f);
        o.melanin=5.0f; o.melaninRedness=0.06f;
        o.longitudinalRoughness=0.10f; o.azimuthalRoughness=0.23f;
        o.tipLightening=0.025f;
    } else if(palette=="golden-blonde" || palette=="blonde" || palette=="gold") {
        o.colorModel=HairColorModel::NaturalMelanin;
        o.directColor=glm::vec3(0.62f,0.34f,0.095f);
        o.melanin=0.42f; o.melaninRedness=0.48f;
        o.longitudinalRoughness=0.12f; o.azimuthalRoughness=0.26f;
        o.tipLightening=0.045f;
    } else if(palette!="explicit") {
        throw std::runtime_error("Unknown default hair palette: "+palette);
    }
}

HairOptics parseOptics(const json& j) {
    HairOptics o;
    if(j.is_null()) return o;
    std::string model=j.value("color_model",std::string("melanin"));
    std::transform(model.begin(),model.end(),model.begin(),[](unsigned char c){return char(std::tolower(c));});
    if(model=="melanin" || model=="natural") o.colorModel=HairColorModel::NaturalMelanin;
    else if(model=="direct" || model=="rgb") o.colorModel=HairColorModel::DirectColor;
    else throw std::runtime_error("Unknown hair color model: "+model);
    if(j.contains("color")) o.directColor=jvec3(j.at("color"),o.directColor);
    o.melanin=j.value("melanin",o.melanin);
    o.melaninRedness=j.value("melanin_redness",o.melaninRedness);
    o.longitudinalRoughness=j.value("longitudinal_roughness",o.longitudinalRoughness);
    o.azimuthalRoughness=j.value("azimuthal_roughness",o.azimuthalRoughness);
    o.ior=j.value("ior",o.ior); o.cuticleTilt=j.value("cuticle_tilt",o.cuticleTilt);
    o.rootDiameterScale=j.value("root_diameter_scale",o.rootDiameterScale);
    o.tipDiameterScale=j.value("tip_diameter_scale",o.tipDiameterScale);
    o.diameterVariation=j.value("diameter_variation",o.diameterVariation);
    o.tipLightening=j.value("tip_lightening",o.tipLightening);
    o.volumeScatter=j.value("volume_scatter",o.volumeScatter);
    return o;
}

RootPatch parseRootPatch(const json& j) {
    RootPatch r;
    r.name=j.value("name",std::string("root"));
    r.centerD=safeNorm(jvec3(j.at("center"), glm::vec3(0.0f, 1.0f, 0.0f)));
    r.invRadius=jvec3(j.value("inv_radius",json::array({2.0f,2.0f,2.0f})), glm::vec3(2.0f));
    r.materialExtent=jvec2(j.value("material_extent",json::array({0.60f,0.60f})), glm::vec2(0.60f));
    r.materialExtent=glm::max(r.materialExtent,glm::vec2(0.05f));
    r.bias=j.value("bias",0.0f);
    return r;
}

void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}

} // namespace

float sampleField(const ScalarField& field, glm::vec3 canonicalDirection) {
    const glm::vec3 d=safeNorm(canonicalDirection);
    float v=field.base;
    for(const auto& c:field.controls) {
        const glm::vec3 center=safeNorm(c.centerD);
        const float angle=std::acos(std::clamp(glm::dot(d,center),-1.0f,1.0f));
        const float x=angle/std::max(c.angularRadius,0.025f);
        if(x>=1.0f) continue;
        const float q=1.0f-x;
        const float w=q*q*q*(1.0f+3.0f*x); // compact C2 kernel
        v+=c.delta*w;
    }
    return std::clamp(v,field.minValue,field.maxValue);
}

FieldSample sampleFields(const ScalpFieldAtlas& f, glm::vec3 d) {
    FieldSample s;
    s.length=sampleField(f.length,d); s.density=sampleField(f.density,d);
    s.volume=sampleField(f.volume,d); s.textureAmplitude=sampleField(f.textureAmplitude,d);
    s.textureFrequency=sampleField(f.textureFrequency,d); s.detail=sampleField(f.detail,d);
    return s;
}

PartSample samplePart(const PartCurve& part, glm::vec3 canonicalDirection) {
    PartSample out;
    if(part.pointsD.size()<2u || part.flowStrength<=0.0f) return out;
    const glm::vec3 d=safeNorm(canonicalDirection);
    float best2=1e30f;
    glm::vec3 nearest=d;
    for(size_t i=0;i+1u<part.pointsD.size();++i) {
        const glm::vec3 a=safeNorm(part.pointsD[i]);
        const glm::vec3 b=safeNorm(part.pointsD[i+1u]);
        const glm::vec3 ab=b-a;
        const float denom=std::max(glm::dot(ab,ab),1e-8f);
        const float t=std::clamp(glm::dot(d-a,ab)/denom,0.0f,1.0f);
        const glm::vec3 q=safeNorm(glm::mix(a,b,t),a);
        const float d2=glm::dot(d-q,d-q);
        if(d2<best2) { best2=d2; nearest=q; }
    }
    const float angle=std::acos(std::clamp(glm::dot(d,nearest),-1.0f,1.0f));
    if(angle<part.influenceRadius) {
        const glm::vec3 tangent=d-nearest*glm::dot(d,nearest);
        out.flowD=safeNorm(tangent, glm::vec3(0.0f));
        const float x=angle/std::max(part.influenceRadius,0.025f);
        out.flowWeight=part.flowStrength*(1.0f-smooth01(x));
    }
    if(part.gapRadius>0.0f) {
        const float inner=0.55f*part.gapRadius;
        const float outer=1.45f*part.gapRadius;
        out.densityScale=smooth01((angle-inner)/std::max(outer-inner,1e-5f));
    }
    return out;
}

float rootPatchScore(const RootPatch& patch,const vh::scalp::Sample& root) {
    const glm::vec3 d=(safeNorm(root.direction)-safeNorm(patch.centerD))*patch.invRadius;
    return patch.bias-glm::dot(d,d);
}

glm::vec2 rootPatchCoordinate(const RootPatch& patch,const vh::scalp::Sample& root) {
    const glm::vec3 center=safeNorm(patch.centerD);
    const glm::vec3 ref=std::abs(center.y)>0.88f?glm::vec3(0,0,1):glm::vec3(0,1,0);
    const glm::vec3 axisU=safeNorm(glm::cross(ref,center), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::vec3 axisV=safeNorm(glm::cross(center,axisU), glm::vec3(0.0f, 0.0f, 1.0f));
    const glm::vec3 rel=safeNorm(root.direction)-center;
    glm::vec2 xi{glm::dot(rel,axisU)/patch.materialExtent.x,
                 glm::dot(rel,axisV)/patch.materialExtent.y};
    return glm::clamp(xi,glm::vec2(-1.35f),glm::vec2(1.35f));
}

GroomStyle loadGroomStyle(const std::filesystem::path& file) {
    std::ifstream in(file);
    if(!in) throw std::runtime_error("Cannot open groom style: "+file.string());
    json j; in>>j;
    const int schema=j.value("schema",0);
    require(schema==4,"Unsupported groom schema (expected schema 4) in "+file.string());

    GroomStyle s;
    s.slug=j.value("slug",file.stem().string());
    s.name=j.value("name",s.slug);
    const json material=j.value("material",json::object());
    s.optics=parseOptics(material);
    applyDefaultPalette(s.optics,s.slug,material);

    if(j.contains("part")) {
        const auto& p=j.at("part");
        s.part.flowStrength=p.value("flow_strength",0.0f);
        s.part.influenceRadius=p.value("influence_radius",0.32f);
        s.part.gapRadius=p.value("gap_radius",0.010f);
        if(p.contains("points")) for(const auto& q:p.at("points")) s.part.pointsD.push_back(safeNorm(jvec3(q)));
    }

    if(j.contains("fields")) {
        const auto& f=j.at("fields");
        auto get=[&](const char* n)->json{return f.contains(n)?f.at(n):json();};
        s.fields.length=parseField(get("length"),s.fields.length);
        s.fields.density=parseField(get("density"),s.fields.density);
        s.fields.volume=parseField(get("volume"),s.fields.volume);
        s.fields.textureAmplitude=parseField(get("texture_amplitude"),s.fields.textureAmplitude);
        s.fields.textureFrequency=parseField(get("texture_frequency"),s.fields.textureFrequency);
        s.fields.detail=parseField(get("detail"),s.fields.detail);
    }

    const auto& g=j.at("graph");
    for(const auto& k:g.at("keys")) {
        SweepKey key;
        key.centerD=jvec3(k.at("center")); key.axisUD=jvec3(k.at("axis_u")); key.axisVD=jvec3(k.at("axis_v"));
        key.attachment=k.value("attachment",0.0f);
        s.graph.sweepKeys.push_back(key);
    }
    for(const auto& c:g.at("curves")) {
        StyleCurve curve; curve.topology=TopologyId(c.value("topology",0));
        for(const auto& id:c.at("keys")) curve.keys.push_back(SweepKeyId(id.get<uint32_t>()));
        s.graph.curves.push_back(std::move(curve));
    }
    for(const auto& b:g.at("branches")) {
        GroomBranch branch;
        branch.curve=StyleCurveId(b.at("curve").get<uint32_t>());
        branch.topology=TopologyId(b.value("topology",0));
        branch.root=parseRootPatch(b.at("root"));
        branch.materialScale=jvec2(b.value("material_scale",json::array({0.72f,0.72f})), glm::vec2(0.72f));
        branch.materialOffset=jvec2(b.value("material_offset",json::array({0.0f,0.0f})), glm::vec2(0.0f));
        branch.clumps=parseClumps(b.value("clumps",json::object()));
        branch.cut=parseCutSurface(b.value("cut_surface",json::object()));
        if(b.contains("style_warps")) for(const auto& h:b.at("style_warps")) branch.styleWarps.push_back(parseStyleWarp(h));
        branch.texture=parseTexture(b.value("texture",json::object()));
        branch.detail=parseDetail(b.value("detail",json::object()));
        s.graph.branches.push_back(std::move(branch));
    }

    require(!s.graph.sweepKeys.empty(),"Groom has no style-volume keys: "+file.string());
    require(!s.graph.curves.empty(),"Groom has no style-volume paths: "+file.string());
    require(!s.graph.branches.empty(),"Groom has no root branches: "+file.string());
    return s;
}

std::vector<StyleEntry> discoverStyleAssets(const std::filesystem::path& directory) {
    std::vector<StyleEntry> out;
    if(!std::filesystem::exists(directory)) return out;
    for(const auto& e:std::filesystem::directory_iterator(directory)) {
        if(!e.is_regular_file() || e.path().extension()!=".json") continue;
        try {
            std::ifstream in(e.path()); json j; in>>j;
            const int schema=j.value("schema",0); if(schema!=4) continue;
            out.push_back({j.value("slug",e.path().stem().string()),j.value("name",e.path().stem().string()),e.path()});
        } catch(...) { /* invalid assets are reported when explicitly loaded */ }
    }
    std::sort(out.begin(),out.end(),[](const StyleEntry& a,const StyleEntry& b){return a.slug<b.slug;});
    return out;
}

std::optional<size_t> findStyle(const std::vector<StyleEntry>& entries,std::string_view key) {
    for(size_t i=0;i<entries.size();++i) if(entries[i].slug==key || entries[i].name==key) return i;
    return std::nullopt;
}

size_t wrapStyleIndex(int index,size_t count) {
    if(count==0u) return 0u;
    const int n=int(count); int i=index%n; if(i<0) i+=n; return size_t(i);
}

} // namespace vh::groom
