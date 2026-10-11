#include "editor_ui.hpp"
#include <imgui.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <glm/gtc/matrix_inverse.hpp>

namespace vh::editor {
namespace {

constexpr float kSideWidth=300.0f;
constexpr float kTopHeight=54.0f;
constexpr float kBottomHeight=36.0f;

const char* modeName(StudioMode m) {
    switch(m) { case StudioMode::Display:return "DISPLAY"; case StudioMode::Edit:return "EDIT"; case StudioMode::Physics:return "PHYSICS"; }
    return "DISPLAY";
}
const char* environmentPresetName(EnvironmentPreset p) {
    switch(p) {
        case EnvironmentPreset::StudioWarmCool: return "Brown Photostudio 02";
        case EnvironmentPreset::OutdoorHighContrast: return "Kloofendal 48d";
        case EnvironmentPreset::CinematicDawn: return "Kiara 1 Dawn";
        case EnvironmentPreset::SoftMoody: return "Misty Pines";
        default: return "Studio Small 09";
    }
}
void sectionHeader(const char* title) { ImGui::Spacing(); ImGui::TextColored(ImVec4(.76f,.67f,1,1),"%s",title); ImGui::Separator(); }
bool vec3Drag(const char* label,glm::vec3& v,float speed,float lo,float hi) { return ImGui::DragFloat3(label,&v.x,speed,lo,hi,"%.3f",ImGuiSliderFlags_AlwaysClamp); }

void applyPortraitSoftLighting(StudioLighting& l) {
    l.primaryIntensity=2.00f;
    l.primaryAzimuthDeg=-35.0f;
    l.primaryElevationDeg=30.0f;
    l.primaryAngularRadius=0.16f;
    l.secondaryIntensity=0.24f;
    l.secondaryAzimuthDeg=135.0f;
    l.secondaryElevationDeg=28.0f;
    l.secondaryAngularRadius=0.20f;
    l.environmentIrradiance=0.42f;
    l.exposureEV=-0.20f;
    l.neutralHeadPresentation=true;
    l.neutralHeadDomeStrength=1.35f;
}

void applyHairBeautyLighting(StudioLighting& l) {
    l.primaryIntensity=2.15f;
    l.primaryAzimuthDeg=-40.0f;
    l.primaryElevationDeg=32.0f;
    l.primaryAngularRadius=0.17f;
    l.secondaryIntensity=0.28f;
    l.secondaryAzimuthDeg=135.0f;
    l.secondaryElevationDeg=28.0f;
    l.secondaryAngularRadius=0.20f;
    l.environmentIrradiance=0.38f;
    l.exposureEV=-0.15f;
    l.neutralHeadPresentation=true;
    l.neutralHeadDomeStrength=1.30f;
}

void applyNeutralLookdevLighting(StudioLighting& l) {
    l.primaryIntensity=1.00f;
    l.primaryAzimuthDeg=-30.0f;
    l.primaryElevationDeg=25.0f;
    l.primaryAngularRadius=0.20f;
    l.secondaryIntensity=0.0f;
    l.secondaryAzimuthDeg=135.0f;
    l.secondaryElevationDeg=28.0f;
    l.secondaryAngularRadius=0.20f;
    l.environmentIrradiance=0.85f;
    l.exposureEV=-0.10f;
    l.neutralHeadPresentation=true;
    l.neutralHeadDomeStrength=1.45f;
}

bool project(const glm::mat4& vp,glm::vec3 p,ImVec2 origin,ImVec2 size,ImVec2& out) {
    glm::vec4 c=vp*glm::vec4(p,1.0f); if(c.w<=1e-5f) return false;
    const glm::vec3 n=glm::vec3(c)/c.w; if(n.z<0.0f||n.z>1.0f) return false;
    out.x=origin.x+(n.x*.5f+.5f)*size.x; out.y=origin.y+(1.0f-(n.y*.5f+.5f))*size.y; return true;
}

void applyTexturePreset(vh::groom::TextureProfile& t,vh::groom::HairTexture type) {
    t.type=type;
    switch(type) {
        case vh::groom::HairTexture::Straight: t.amplitude=0;t.cycles=1;t.start=.10f;t.irregularity=.04f;t.phaseCoherence=.90f;break;
        case vh::groom::HairTexture::Wavy: t.amplitude=.010f;t.cycles=1.30f;t.start=.08f;t.irregularity=.10f;t.phaseCoherence=.72f;break;
        case vh::groom::HairTexture::Curly: t.amplitude=.0085f;t.cycles=2.8f;t.start=.06f;t.irregularity=.14f;t.phaseCoherence=.64f;break;
        case vh::groom::HairTexture::Coily: t.amplitude=.0065f;t.cycles=5.0f;t.start=.04f;t.irregularity=.18f;t.phaseCoherence=.56f;break;
    }
}

void drawValidation(const vh::groom::ValidationReport& v,bool pending) {
    ImGui::TextColored(ImVec4(.45f,.95f,.62f,1),"Simulation: AVAILABLE");
    if(pending) { ImGui::TextColored(ImVec4(.72f,.70f,1.0f,1),"Contact audit: running..."); return; }
    if(v.contactClean()) ImGui::TextColored(ImVec4(.45f,.95f,.62f,1),"Contact audit: CLEAN");
    else {
        ImGui::TextColored(ImVec4(1,.64f,.28f,1),"Contact audit: WARNING (simulation remains enabled)");
        ImGui::Text("Visual inside  %u",v.visualInsideSamples); ImGui::Text("Base inside    %u",v.baseInsideSamples);
        ImGui::Text("Proxy failures %u",v.guideProxyViolations); ImGui::Text("Deepest %.2f mm",1000.0f*v.deepestPenetration);
    }
}

bool sliderField(const char* label,vh::groom::ScalarField& f,float lo,float hi) {
    return ImGui::SliderFloat(label,&f.base,lo,hi,"%.3f");
}

enum class HairstyleCategory : int { All=0, LongWavy, BobLayered, Ponytails, Updos, ShortEditorial };

const char* hairstyleCategoryName(HairstyleCategory c) {
    switch(c) {
        case HairstyleCategory::LongWavy: return "Long & Wavy";
        case HairstyleCategory::BobLayered: return "Bob & Layered";
        case HairstyleCategory::Ponytails: return "Ponytails";
        case HairstyleCategory::Updos: return "Updos";
        case HairstyleCategory::ShortEditorial: return "Short / Editorial";
        default: return "All";
    }
}

bool containsToken(std::string_view text,std::string_view token) {
    return text.find(token)!=std::string_view::npos;
}

HairstyleCategory classifyHairstyle(const vh::groom::StyleEntry& e) {
    const std::string& s=e.slug;
    if(containsToken(s,"pony") || containsToken(s,"twin-tail")) return HairstyleCategory::Ponytails;
    if(containsToken(s,"bun") || containsToken(s,"half-up") || containsToken(s,"topknot")) return HairstyleCategory::Updos;
    if(containsToken(s,"crop") || containsToken(s,"quiff") || containsToken(s,"undercut")) return HairstyleCategory::ShortEditorial;
    if(containsToken(s,"bob") || containsToken(s,"shag")) return HairstyleCategory::BobLayered;
    if(containsToken(s,"long") || containsToken(s,"wavy") || containsToken(s,"butterfly") || containsToken(s,"face-framing") || containsToken(s,"loose")) return HairstyleCategory::LongWavy;
    if(containsToken(s,"layered")) return HairstyleCategory::BobLayered;
    return HairstyleCategory::LongWavy;
}

bool containsCaseInsensitive(std::string_view haystack,std::string_view needle) {
    if(needle.empty()) return true;
    std::string h(haystack), n(needle);
    std::transform(h.begin(),h.end(),h.begin(),[](unsigned char c){return char(std::tolower(c));});
    std::transform(n.begin(),n.end(),n.begin(),[](unsigned char c){return char(std::tolower(c));});
    return h.find(n)!=std::string::npos;
}

bool drawScreenshotMenu(State& state,Actions& actions,const ImGuiViewport* iv) {
    bool open=false;
    if(state.openScreenshotMenu) {
        ImGui::OpenPopup("Screenshot Menu##menu");
        state.openScreenshotMenu=false;
    }

    constexpr ImVec2 panelSize(430.0f,232.0f);
    ImGui::SetNextWindowPos(iv->GetCenter(),ImGuiCond_Always,ImVec2(0.5f,0.5f));
    ImGui::SetNextWindowSize(panelSize,ImGuiCond_Always);
    if(!ImGui::BeginPopupModal("Screenshot Menu##menu",nullptr,
        ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoTitleBar)) return false;
    open=true;

    ImGui::TextColored(ImVec4(.84f,.73f,1.0f,1.0f),"SAVE SCREENSHOT");
    ImGui::TextDisabled("Program directory / screenshots");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const char* formats[]={"PNG","BMP","TGA"};
    int fmt=int(state.screenshotFormat);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Format");
    ImGui::SameLine(92.0f);
    ImGui::SetNextItemWidth(150.0f);
    if(ImGui::Combo("##screenshot-format",&fmt,formats,IM_ARRAYSIZE(formats))) state.screenshotFormat=ScreenshotFormat(fmt);

    ImGui::Spacing();
    const float gap=8.0f;
    const float buttonWidth=(ImGui::GetContentRegionAvail().x-gap)*0.5f;
    if(ImGui::Button("Save with UI",ImVec2(buttonWidth,38.0f))) {
        actions.requestScreenshot=true;
        actions.screenshotIncludeUi=true;
        actions.screenshotFormat=state.screenshotFormat;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine(0.0f,gap);
    if(ImGui::Button("Save clean",ImVec2(buttonWidth,38.0f))) {
        actions.requestScreenshot=true;
        actions.screenshotIncludeUi=false;
        actions.screenshotFormat=state.screenshotFormat;
        ImGui::CloseCurrentPopup();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Clean capture excludes every editor panel and overlay.");
    ImGui::Spacing();
    const float closeWidth=92.0f;
    ImGui::SetCursorPosX((ImGui::GetWindowContentRegionMax().x+ImGui::GetWindowContentRegionMin().x-closeWidth)*0.5f);
    if(ImGui::Button("Close",ImVec2(closeWidth,28.0f))) ImGui::CloseCurrentPopup();

    ImGui::EndPopup();
    return open;
}

bool drawHairstyleLibrary(State& state,const std::vector<vh::groom::StyleEntry>& catalog,size_t styleIndex,bool styleBuildPending,Actions& actions,const ImGuiViewport* iv) {
    bool open=false;
    if(state.openHairstyleMenu) {
        ImGui::OpenPopup("Hairstyle Library##menu");
        state.openHairstyleMenu=false;
    }
    ImGui::SetNextWindowPos(iv->GetCenter(),ImGuiCond_Appearing,ImVec2(0.5f,0.5f));
    ImGui::SetNextWindowSize(ImVec2(720.0f,540.0f),ImGuiCond_Appearing);
    if(!ImGui::BeginPopup("Hairstyle Library##menu",ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove)) return false;
    open=true;

    ImGui::TextColored(ImVec4(.84f,.73f,1.0f,1.0f),"HAIRSTYLE LIBRARY");
    ImGui::SameLine();
    ImGui::TextDisabled("%zu styles",catalog.size());
    if(styleIndex<catalog.size()) {
        ImGui::Text("Current: %s",catalog[styleIndex].name.c_str());
        ImGui::SameLine(); ImGui::TextDisabled("%s",catalog[styleIndex].slug.c_str());
    }
    if(styleBuildPending) ImGui::TextColored(ImVec4(.82f,.66f,1.0f,1.0f),"Building selected groom asynchronously...");

    ImGui::Spacing();
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##hairstyle-search","Search hairstyle...",state.hairstyleSearch.data(),state.hairstyleSearch.size());
    ImGui::Spacing();

    const HairstyleCategory filters[]={HairstyleCategory::All,HairstyleCategory::LongWavy,HairstyleCategory::BobLayered,HairstyleCategory::Ponytails,HairstyleCategory::Updos,HairstyleCategory::ShortEditorial};
    for(size_t i=0;i<std::size(filters);++i) {
        const bool active=state.hairstyleCategory==int(filters[i]);
        if(active) ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(.38f,.25f,.58f,1.0f));
        if(ImGui::Button(hairstyleCategoryName(filters[i]))) state.hairstyleCategory=int(filters[i]);
        if(active) ImGui::PopStyleColor();
        if(i+1<std::size(filters)) ImGui::SameLine();
    }
    ImGui::Separator();

    const std::string_view search(state.hairstyleSearch.data());
    const auto selectedCategory=HairstyleCategory(state.hairstyleCategory);
    int visible=0;
    if(ImGui::BeginChild("##hairstyle-grid",ImVec2(0,0),false,ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        if(ImGui::BeginTable("##hairstyle-table",2,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_PadOuterX)) {
            for(size_t i=0;i<catalog.size();++i) {
                const auto& e=catalog[i];
                const auto category=classifyHairstyle(e);
                if(selectedCategory!=HairstyleCategory::All && category!=selectedCategory) continue;
                if(!containsCaseInsensitive(e.name,search) && !containsCaseInsensitive(e.slug,search)) continue;
                ++visible;
                ImGui::TableNextColumn();
                ImGui::PushID(int(i));
                const bool active=i==styleIndex;
                if(active) {
                    ImGui::PushStyleColor(ImGuiCol_Header,ImVec4(.38f,.25f,.58f,.95f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,ImVec4(.43f,.29f,.64f,1.0f));
                }
                if(styleBuildPending) ImGui::BeginDisabled();
                const ImVec2 cardSize(ImGui::GetContentRegionAvail().x,38.0f);
                if(ImGui::Selectable(e.name.c_str(),active,ImGuiSelectableFlags_None,cardSize)) {
                    if(i!=styleIndex) actions.requestedStyleIndex=int(i);
                    ImGui::CloseCurrentPopup();
                }
                if(styleBuildPending) ImGui::EndDisabled();
                if(active) ImGui::PopStyleColor(2);
                ImGui::TextDisabled("%s  |  %s",hairstyleCategoryName(category),e.slug.c_str());
                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if(visible==0) {
            ImGui::Spacing(); ImGui::TextDisabled("No hairstyles match this filter.");
        }
    }
    ImGui::EndChild();
    ImGui::EndPopup();
    return open;
}

} // namespace

void applyStudioTheme() {
    ImGuiStyle& st=ImGui::GetStyle(); st.WindowRounding=10; st.ChildRounding=8; st.FrameRounding=6; st.PopupRounding=8; st.GrabRounding=6; st.TabRounding=6;
    st.WindowPadding={14,12}; st.FramePadding={9,6}; st.ItemSpacing={8,8}; st.ScrollbarSize=12;
    auto& c=st.Colors; c[ImGuiCol_WindowBg]=ImVec4(.055f,.060f,.075f,.98f); c[ImGuiCol_ChildBg]=ImVec4(.070f,.075f,.092f,.96f);
    c[ImGuiCol_PopupBg]=ImVec4(.065f,.068f,.085f,.99f); c[ImGuiCol_Border]=ImVec4(.18f,.18f,.24f,.60f); c[ImGuiCol_FrameBg]=ImVec4(.105f,.105f,.135f,1);
    c[ImGuiCol_FrameBgHovered]=ImVec4(.16f,.14f,.22f,1); c[ImGuiCol_FrameBgActive]=ImVec4(.23f,.18f,.34f,1); c[ImGuiCol_Button]=ImVec4(.15f,.12f,.22f,1);
    c[ImGuiCol_ButtonHovered]=ImVec4(.31f,.21f,.48f,1); c[ImGuiCol_ButtonActive]=ImVec4(.42f,.27f,.65f,1); c[ImGuiCol_Header]=ImVec4(.18f,.14f,.27f,1);
    c[ImGuiCol_HeaderHovered]=ImVec4(.31f,.22f,.45f,1); c[ImGuiCol_HeaderActive]=ImVec4(.40f,.26f,.60f,1); c[ImGuiCol_CheckMark]=ImVec4(.72f,.54f,1,1);
    c[ImGuiCol_SliderGrab]=ImVec4(.57f,.39f,.88f,1); c[ImGuiCol_SliderGrabActive]=ImVec4(.76f,.58f,1,1); c[ImGuiCol_Text]=ImVec4(.91f,.91f,.95f,1); c[ImGuiCol_TextDisabled]=ImVec4(.52f,.52f,.60f,1);
}

Actions draw(State& state,const vh::character::CharacterSurface& character,const vh::groom::GroomBinding& binding,
             const std::vector<vh::groom::StyleEntry>& catalog,size_t styleIndex,vh::groom::GroomStyle& style,
             vh::groom::GroomBuildParams& build,vh::physics::HairMaterial& material,vh::wind::Field& wind,StudioLighting& lighting,
             const vh::groom::ValidationReport& validation,bool validationPending,bool styleBuildPending,const PerformanceStats& perf,const std::string& assetError,const glm::mat4& viewProj,
             uint32_t viewportWidth,uint32_t viewportHeight) {
    Actions a; (void)viewportWidth;(void)viewportHeight;
    const ImGuiViewport* iv=ImGui::GetMainViewport(); const ImVec2 full=iv->Size;
    const ImGuiWindowFlags fixed=ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(iv->Pos); ImGui::SetNextWindowSize({full.x,kTopHeight}); ImGui::Begin("##top",nullptr,fixed);
    ImGui::TextColored(ImVec4(.80f,.68f,1,1),"V39 GROOM  /  V44 VISIBILITY-FIRST"); ImGui::SameLine(330);
    for(StudioMode m:{StudioMode::Display,StudioMode::Edit,StudioMode::Physics}) { const bool active=state.mode==m; if(active) ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(.36f,.23f,.56f,1));
        if(ImGui::Button(modeName(m),{98,30})){a.modeChangeRequested=true;a.requestedMode=m;} if(active) ImGui::PopStyleColor(); ImGui::SameLine(); }
    if(ImGui::Button("HAIRSTYLES",{124,30})) state.openHairstyleMenu=true;
    ImGui::SameLine();
    if(ImGui::Button("SCREENSHOT",{124,30})) state.openScreenshotMenu=true;
    drawHairstyleLibrary(state,catalog,styleIndex,styleBuildPending,a,iv);
    drawScreenshotMenu(state,a,iv);
    ImGui::SameLine(full.x-110); if(state.mode==StudioMode::Physics&&ImGui::Button("Reset",{82,30}))a.resetPhysics=true; ImGui::End();

    ImGui::SetNextWindowPos({iv->Pos.x,iv->Pos.y+kTopHeight}); ImGui::SetNextWindowSize({kSideWidth,full.y-kTopHeight-kBottomHeight}); ImGui::Begin("##left",nullptr,fixed);
    sectionHeader("HAIRSTYLE");
    ImGui::TextColored(ImVec4(.93f,.91f,1.0f,1.0f),"%s",style.name.c_str());
    ImGui::TextDisabled("%s",style.slug.c_str());
    ImGui::TextDisabled("%zu styles available",catalog.size());
    if(ImGui::Button("Browse Hairstyle Library",{-1,30})) state.openHairstyleMenu=true;
    if(styleBuildPending) ImGui::TextColored(ImVec4(.82f,.66f,1,1),"Building derived groom asynchronously...");
    if(state.mode!=StudioMode::Physics){ ImGui::Checkbox("Keep Outside Skin",&state.keepOutsideSkin); if(!validationPending&&!styleBuildPending&&validation.hasContactIssue()&&ImGui::Button("Resolve Contacts",{-1,28}))a.resolveContacts=true; }

    if(state.mode==StudioMode::Display){ sectionHeader("CONTACT STATUS"); drawValidation(validation,validationPending||styleBuildPending); sectionHeader("CHARACTER"); ImGui::Text("Triangles %zu",character.triangles().size()); ImGui::Text("Scalp %zu",character.scalpTriangles().size()); ImGui::Text("Field %u^3",character.contactField().dim); }
    else if(state.mode==StudioMode::Edit){
        sectionHeader("TOPOLOGY / SCAFFOLD"); state.selectedCurve=std::clamp(state.selectedCurve,0,std::max(0,int(style.graph.curves.size())-1));
        for(int r=0;r<int(style.graph.curves.size());++r){std::string label="Curve "+std::to_string(r);if(ImGui::Selectable(label.c_str(),state.selectedCurve==r)){state.selectedCurve=r;state.selectedKey=0;}}
        sectionHeader("ROOT BRANCHES"); state.selectedBranch=std::clamp(state.selectedBranch,0,std::max(0,int(style.graph.branches.size())-1));
        for(int i=0;i<int(style.graph.branches.size());++i){const auto& b=style.graph.branches[size_t(i)];std::string label=b.root.name+"##"+std::to_string(i);if(ImGui::Selectable(label.c_str(),i==state.selectedBranch))state.selectedBranch=i;}
    } else {
        sectionHeader("PHYSICS STATUS");drawValidation(validation,validationPending||styleBuildPending);sectionHeader("HAIR MATERIAL");
        if(ImGui::SliderFloat("Stretch",&material.stretchStiffness,.2f,1,"%.2f"))a.physicsDirty=true; if(ImGui::SliderFloat("Bend",&material.bendStiffness,.05f,1,"%.2f"))a.physicsDirty=true;
        if(ImGui::SliderFloat("Damping",&material.damping,0,1,"%.2f"))a.physicsDirty=true; if(ImGui::SliderFloat("Static friction",&material.staticFriction,0,1.2f,"%.2f"))a.physicsDirty=true;
        if(ImGui::SliderFloat("Dynamic friction",&material.dynamicFriction,0,1,"%.2f"))a.physicsDirty=true; if(ImGui::SliderFloat("Style memory",&material.styleMemory,0,1,"%.2f"))a.physicsDirty=true;
        if(ImGui::SliderFloat("Air drag",&material.airDrag,0,1.2f,"%.2f"))a.physicsDirty=true; if(ImGui::SliderFloat("Bulk fullness",&material.bulkFullness,.55f,1.45f,"%.2f"))a.physicsDirty=true;
        sectionHeader("WIND"); int type=int(wind.type); const char* types[]={"Directional","Fan / Cone","Vortex"}; if(ImGui::Combo("Type",&type,types,3))wind.type=vh::wind::Type(type);
        ImGui::Checkbox("Enabled",&wind.enabled);ImGui::SliderFloat("Speed",&wind.speed,0,14,"%.1f m/s");vec3Drag("Direction",wind.direction,.01f,-1,1);
    }
    ImGui::End();

    ImGui::SetNextWindowPos({iv->Pos.x+full.x-kSideWidth,iv->Pos.y+kTopHeight}); ImGui::SetNextWindowSize({kSideWidth,full.y-kTopHeight-kBottomHeight}); ImGui::Begin("##right",nullptr,fixed);
    if(!assetError.empty()){sectionHeader("ASSET ERROR");ImGui::TextColored(ImVec4(1,.42f,.36f,1),"%s",assetError.c_str());}
    if(state.mode==StudioMode::Edit&&!style.graph.branches.empty()) {
        if(styleBuildPending) ImGui::BeginDisabled();
        sectionHeader("CONTINUOUS SCALP FIELDS");
        if(sliderField("Length",style.fields.length,.45f,1.10f))a.groomDirty=true; if(sliderField("Density",style.fields.density,.25f,1.0f))a.groomDirty=true;
        if(sliderField("Volume",style.fields.volume,.55f,1.55f))a.groomDirty=true;
        if(sliderField("Texture amplitude",style.fields.textureAmplitude,0,1.8f))a.groomDirty=true; if(sliderField("Texture frequency",style.fields.textureFrequency,.45f,1.8f))a.groomDirty=true;
        if(sliderField("Micro detail",style.fields.detail,0,1.8f))a.groomDirty=true;
        if(style.part.pointsD.size()>=2){ImGui::Spacing();ImGui::TextDisabled("Part / root flow"); if(ImGui::SliderFloat("Flow strength",&style.part.flowStrength,0,1,"%.2f"))a.groomDirty=true; if(ImGui::SliderFloat("Part gap",&style.part.gapRadius,0,.05f,"%.3f rad"))a.groomDirty=true;}

        sectionHeader("MACRO SCAFFOLD"); if(!style.graph.curves.empty()) { auto& curve=style.graph.curves[size_t(state.selectedCurve)]; state.selectedKey=std::clamp(state.selectedKey,0,std::max(0,int(curve.keys.size())-1)); auto& gate=style.graph.sweepKeys[curve.keys[size_t(state.selectedKey)]];
            if(vec3Drag("Position",gate.centerD,.005f,-2,2))a.groomDirty=true; if(vec3Drag("Width axis",gate.axisUD,.003f,-.6f,.6f))a.groomDirty=true; if(vec3Drag("Depth axis",gate.axisVD,.003f,-.6f,.6f))a.groomDirty=true; if(ImGui::SliderFloat("Attachment",&gate.attachment,0,1,"%.2f"))a.groomDirty=true; }

        sectionHeader("BRANCH / CLUMP HIERARCHY"); auto& b=style.graph.branches[size_t(state.selectedBranch)]; ImGui::TextDisabled("%s",b.root.name.c_str());
        ImGui::TextDisabled("Primary locks");
        if(ImGui::SliderFloat("Primary mid tightness",&b.clumps.primary.midTightness,0,.85f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Primary tip tightness",&b.clumps.primary.tipTightness,0,.75f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Primary crossover",&b.clumps.primary.crossover,0,.45f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Primary drift",&b.clumps.primary.drift,0,.25f,"%.3f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Primary wave",&b.clumps.primary.waveAmplitude,0,.28f,"%.3f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Primary wave cycles",&b.clumps.primary.waveCycles,0,4,"%.2f"))a.groomDirty=true;
        ImGui::TextDisabled("Secondary clumps");
        if(ImGui::SliderFloat("Secondary mid tightness",&b.clumps.secondary.midTightness,0,.65f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Secondary tip tightness",&b.clumps.secondary.tipTightness,0,.55f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Secondary crossover",&b.clumps.secondary.crossover,0,.45f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Secondary wave",&b.clumps.secondary.waveAmplitude,0,.14f,"%.3f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Tip spread",&b.clumps.tipSpread,0,.35f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Mass twist",&b.clumps.twistTurns,-.5f,.5f,"%.2f turns"))a.groomDirty=true;

        sectionHeader("CUT SURFACE");
        ImGui::TextDisabled("4x4 smooth termination lattice (U across, V rows)");
        for(uint32_t row=0;row<vh::groom::kCutSurfaceResolution;++row) {
            std::string label="Layer row "+std::to_string(row+1);
            float* r=&b.cut.controls[row*vh::groom::kCutSurfaceResolution];
            if(ImGui::DragFloat4(label.c_str(),r,.005f,.15f,1.15f,"%.2f",ImGuiSliderFlags_AlwaysClamp))a.groomDirty=true;
        }
        if(ImGui::SliderFloat("Cut breakup",&b.cut.breakup,0,.08f,"%.3f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Minimum length",&b.cut.minLength,.20f,.90f,"%.2f"))a.groomDirty=true;

        ImGui::Spacing(); ImGui::TextDisabled("Optional style warps (editorial only)");
        state.selectedStyleWarp=std::clamp(state.selectedStyleWarp,0,std::max(0,int(b.styleWarps.size())-1));
        for(int i=0;i<int(b.styleWarps.size());++i){std::string label="Style warp "+std::to_string(i+1)+"##warp"+std::to_string(i); if(ImGui::Selectable(label.c_str(),i==state.selectedStyleWarp)) state.selectedStyleWarp=i;}
        if(ImGui::Button("Add style warp",{118,24})) { b.styleWarps.push_back(vh::groom::StyleWarp{}); state.selectedStyleWarp=int(b.styleWarps.size())-1; a.groomDirty=true; }
        ImGui::SameLine();
        if(!b.styleWarps.empty() && ImGui::Button("Remove warp",{104,24})) { b.styleWarps.erase(b.styleWarps.begin()+state.selectedStyleWarp); state.selectedStyleWarp=std::max(0,state.selectedStyleWarp-1); a.groomDirty=true; }
        if(!b.styleWarps.empty()) { auto& h=b.styleWarps[size_t(state.selectedStyleWarp)];
            if(ImGui::DragFloat2("Warp center",&h.center.x,.01f,-1.3f,1.3f,"%.2f"))a.groomDirty=true; if(ImGui::DragFloat2("Warp radius",&h.radius.x,.01f,.04f,1.3f,"%.2f"))a.groomDirty=true;
            if(ImGui::DragFloat2("Mid drift",&h.midDrift.x,.01f,-.8f,.8f,"%.2f"))a.groomDirty=true; if(ImGui::DragFloat2("Tip drift",&h.tipDrift.x,.01f,-1.0f,1.0f,"%.2f"))a.groomDirty=true;
            if(ImGui::SliderFloat("Lift",&h.lift,-.10f,.16f,"%.3f"))a.groomDirty=true; if(ImGui::SliderFloat("Roll",&h.rollTurns,-.5f,.5f,"%.2f turns"))a.groomDirty=true; if(ImGui::SliderFloat("Length scale",&h.lengthScale,.55f,1.25f,"%.2f"))a.groomDirty=true; if(ImGui::SliderFloat("Weight",&h.weight,0,2,"%.2f"))a.groomDirty=true;
        }

        ImGui::Spacing(); ImGui::TextDisabled("Intrinsic strand texture");
        int tex=int(b.texture.type); const char* tn[]={"Straight","Wavy","Curly","Coily"}; if(ImGui::Combo("Texture",&tex,tn,4)){applyTexturePreset(b.texture,vh::groom::HairTexture(tex));a.groomDirty=true;}
        if(ImGui::SliderFloat("Amplitude",&b.texture.amplitude,0,.035f,"%.4f m"))a.groomDirty=true; if(ImGui::SliderFloat("Cycles",&b.texture.cycles,0,7,"%.2f"))a.groomDirty=true;

        sectionHeader("GLOBAL"); if(ImGui::SliderFloat("All hair length",&build.globalLengthScale,.60f,1.10f,"%.2f"))a.groomDirty=true; if(ImGui::SliderFloat("Overall volume",&build.globalVolumeScale,.65f,1.45f,"%.2f"))a.groomDirty=true;
        if(ImGui::SliderFloat("Residual micro detail",&build.microDetailScale,0.0f,1.25f,"%.2f"))a.groomDirty=true;
        ImGui::TextDisabled("Scales only frizz/flyaway residuals; clumps and authored silhouette stay intact.");
        ImGui::Checkbox("Show style guides",&state.showGuides); float micron=build.fiberDiameter*1e6f; if(ImGui::SliderFloat("Fiber diameter",&micron,35,120,"%.0f um")){build.fiberDiameter=micron*1e-6f;a.groomDirty=true;}

        sectionHeader("HAIR COLOR");
        ImGui::TextDisabled("Black / warm blonde are defaults only. Choose any color here.");
        const bool customColor=style.optics.colorModel==vh::groom::HairColorModel::DirectColor;
        if(ImGui::RadioButton("Custom color",customColor)) {
            style.optics.colorModel=vh::groom::HairColorModel::DirectColor;
            a.lookDirty=true;
        }
        ImGui::SameLine();
        if(ImGui::RadioButton("Natural pigment",!customColor)) {
            style.optics.colorModel=vh::groom::HairColorModel::NaturalMelanin;
            a.lookDirty=true;
        }

        if(style.optics.colorModel==vh::groom::HairColorModel::DirectColor) {
            ImGui::TextDisabled("Base Color is the reflected surface body color under ordinary lighting.");
            ImGui::TextDisabled("Internal transmission is a separate controllable overlay; dielectric specular stays neutral.");
            constexpr ImGuiColorEditFlags pickerFlags = ImGuiColorEditFlags_PickerHueWheel |
                                                        ImGuiColorEditFlags_DisplayRGB |
                                                        ImGuiColorEditFlags_Float;
            if(ImGui::ColorPicker3("##hair-color-picker",&style.optics.directColor.x,pickerFlags)) a.lookDirty=true;
            if(ImGui::ColorEdit3("RGB",&style.optics.directColor.x,
                                 ImGuiColorEditFlags_DisplayRGB|ImGuiColorEditFlags_Float)) a.lookDirty=true;
            float h=0.0f,s=0.0f,v=0.0f;
            ImGui::ColorConvertRGBtoHSV(style.optics.directColor.r,style.optics.directColor.g,style.optics.directColor.b,h,s,v);
            float hsv[3]={h,s,v};
            if(ImGui::SliderFloat3("HSV",hsv,0.0f,1.0f,"%.3f",ImGuiSliderFlags_AlwaysClamp)) {
                ImGui::ColorConvertHSVtoRGB(hsv[0],hsv[1],hsv[2],style.optics.directColor.r,style.optics.directColor.g,style.optics.directColor.b);
                a.lookDirty=true;
            }
        } else {
            ImGui::TextDisabled("Natural eumelanin/pheomelanin absorption for blonde / brown / red / black hair.");
            if(ImGui::Button("Natural black")) {
                style.optics.melanin=5.0f;
                style.optics.melaninRedness=0.06f;
                style.optics.directColor=glm::vec3(0.028f,0.020f,0.017f);
                a.lookDirty=true;
            }
            ImGui::SameLine();
            if(ImGui::Button("Warm blonde")) {
                style.optics.melanin=0.42f;
                style.optics.melaninRedness=0.48f;
                style.optics.directColor=glm::vec3(0.62f,0.34f,0.095f);
                a.lookDirty=true;
            }
            if(ImGui::SliderFloat("Melanin",&style.optics.melanin,0.0f,8.0f,"%.2f"))a.lookDirty=true;
            ImGui::TextDisabled("~0.3 blonde / ~1.3 brown / ~5 black");
            if(ImGui::SliderFloat("Redness",&style.optics.melaninRedness,0.0f,1.0f,"%.2f"))a.lookDirty=true;
        }
        sectionHeader("HAIR OPTICS");
        ImGui::TextDisabled("Directional coated-pigment BRDF + anisotropic GGX importance sampling.");
        if(ImGui::SliderFloat("Longitudinal roughness",&style.optics.longitudinalRoughness,0.03f,0.45f,"%.3f")) a.lookDirty=true;
        if(ImGui::SliderFloat("Azimuthal roughness",&style.optics.azimuthalRoughness,0.05f,0.60f,"%.3f")) a.lookDirty=true;
        if(ImGui::SliderFloat("IOR",&style.optics.ior,1.40f,1.65f,"%.3f")) a.lookDirty=true;
        float cuticleDeg=glm::degrees(style.optics.cuticleTilt);
        if(ImGui::SliderFloat("Cuticle tilt",&cuticleDeg,0.0f,8.0f,"%.2f deg")){ style.optics.cuticleTilt=glm::radians(cuticleDeg); a.lookDirty=true; }
        if(ImGui::SliderFloat("Internal scattering share",&style.optics.volumeScatter,0.0f,0.35f,"%.2f")) a.lookDirty=true;
        ImGui::TextDisabled("0 = all surface reflection; up to 0.35 replaces that share with TT/TRT.");
        ImGui::TextDisabled("Direct-light DOM is directional; coarse volume does not mask surface RGB.");
        const char* hairDebugModes[]={"Final appearance","Material reflectance (unlit)",
            "All pigmented reflection","All dielectric specular","Primary DOM transmittance","Internal TT/TRT only",
            "Environment body reflection","Direct body reflection","Direct specular (key + fill)","Environment specular"};
        if(ImGui::Combo("Hair shading diagnostic",&state.hairDebugView,hairDebugModes,IM_ARRAYSIZE(hairDebugModes))) a.lookDirty=true;

        sectionHeader("ENVIRONMENT");
        const char* envNames[]={"Neutral Softbox","Warm Studio","Outdoor Contrast","Dawn","Soft Moody"};
        int envPreset=int(lighting.environmentPreset);
        if(ImGui::Combo("Skybox preset",&envPreset,envNames,IM_ARRAYSIZE(envNames))) {
            lighting.environmentPreset=EnvironmentPreset(envPreset);
            a.lookDirty=true;
        }
        ImGui::TextDisabled("Directional softbox radiance lobes; no uniform-color dome.");
        if(ImGui::Checkbox("Visible skybox background",&lighting.useSkyboxBackground)) a.lookDirty=true;
        if(ImGui::Checkbox("Use preset for environment lighting",&lighting.useSkyboxLighting)) a.lookDirty=true;
        if(ImGui::SliderFloat("Skybox rotation",&lighting.skyboxRotationDeg,-180.0f,180.0f,"%.0f deg")) a.lookDirty=true;
        if(ImGui::SliderFloat("Background strength",&lighting.skyboxBackgroundStrength,0.0f,1.5f,"%.2f")) a.lookDirty=true;
        if(!lighting.useSkyboxBackground) {
            if(ImGui::SliderFloat("Fallback gray",&lighting.backgroundGray,0.0f,1.0f,"%.2f"))a.lookDirty=true;
            if(ImGui::Button("Black")) { lighting.backgroundGray=0.0f; a.lookDirty=true; }
            ImGui::SameLine();
            if(ImGui::Button("Dark gray")) { lighting.backgroundGray=0.18f; a.lookDirty=true; }
            ImGui::SameLine();
            if(ImGui::Button("Mid gray")) { lighting.backgroundGray=0.35f; a.lookDirty=true; }
            ImGui::SameLine();
            if(ImGui::Button("Light gray")) { lighting.backgroundGray=0.65f; a.lookDirty=true; }
        }

        sectionHeader("HEAD DISPLAY");
        ImGui::TextDisabled("Mannequin-only presentation control. Hair shading remains physically coupled.");
        if(ImGui::Checkbox("Neutral shadowless head",&lighting.neutralHeadPresentation)) a.lookDirty=true;
        if(lighting.neutralHeadPresentation) {
            if(ImGui::SliderFloat("Head light-tent strength",&lighting.neutralHeadDomeStrength,0.25f,2.5f,"%.2f")) a.lookDirty=true;
            ImGui::TextDisabled("Disables hair-cast DOM/volume shadows on the mannequin and adds isotropic studio irradiance.");
        } else {
            ImGui::TextDisabled("Physical scene mode: hair may cast deep-opacity and volume shadows onto the head.");
        }

        sectionHeader("PHYSICAL LIGHTS");
        ImGui::TextDisabled("Presets only set real light/camera parameters; they do not enable shader effects.");
        if(ImGui::Button("Portrait Soft")) { applyPortraitSoftLighting(lighting); a.lookDirty=true; }
        ImGui::SameLine();
        if(ImGui::Button("Hair Beauty")) { applyHairBeautyLighting(lighting); a.lookDirty=true; }
        if(ImGui::Button("Neutral Lookdev")) { applyNeutralLookdevLighting(lighting); a.lookDirty=true; }
        ImGui::TextDisabled("Default: physical key + subtle secondary + directional studio environment.");
        if(ImGui::SliderFloat("Primary intensity",&lighting.primaryIntensity,0.0f,4.0f,"%.2f"))a.lookDirty=true;
        if(ImGui::SliderFloat("Primary azimuth",&lighting.primaryAzimuthDeg,-180.0f,180.0f,"%.0f deg"))a.lookDirty=true;
        if(ImGui::SliderFloat("Primary elevation",&lighting.primaryElevationDeg,-80.0f,80.0f,"%.0f deg"))a.lookDirty=true;
        if(ImGui::SliderFloat("Secondary intensity",&lighting.secondaryIntensity,0.0f,3.0f,"%.2f"))a.lookDirty=true;
        if(ImGui::SliderFloat("Secondary azimuth",&lighting.secondaryAzimuthDeg,-180.0f,180.0f,"%.0f deg"))a.lookDirty=true;
        if(ImGui::SliderFloat("Secondary elevation",&lighting.secondaryElevationDeg,-80.0f,80.0f,"%.0f deg"))a.lookDirty=true;
        if(ImGui::SliderFloat("Environment radiance",&lighting.environmentIrradiance,0.0f,2.0f,"%.2f"))a.lookDirty=true;
        if(ImGui::SliderFloat("Primary source size",&lighting.primaryAngularRadius,0.015f,0.22f,"%.3f rad"))a.lookDirty=true;
        if(ImGui::SliderFloat("Secondary source size",&lighting.secondaryAngularRadius,0.015f,0.28f,"%.3f rad"))a.lookDirty=true;
        if(ImGui::SliderFloat("Exposure (EV)",&lighting.exposureEV,-3.0f,3.0f,"%+.2f"))a.lookDirty=true;
        if(styleBuildPending) ImGui::EndDisabled();
    } else { sectionHeader("CONTACT CONTRACT"); drawValidation(validation,validationPending||styleBuildPending); ImGui::Text("min visual %.2f mm",1000*validation.minVisualDistance); ImGui::Text("min base %.2f mm",1000*validation.minBaseDistance); ImGui::Text("min proxy %.2f mm",1000*validation.minGuideProxyClearance); }
    ImGui::End();

    ImGui::SetNextWindowPos({iv->Pos.x,iv->Pos.y+full.y-kBottomHeight});ImGui::SetNextWindowSize({full.x,kBottomHeight});ImGui::Begin("##status",nullptr,fixed);
    ImGui::TextDisabled("%s | %s%s | CPU %.1f ms | GPU %.1f ms [physics %.1f / optical+shadow %.1f / scene %.1f] | build %.1f ms | cache %.1f ms | %ux MSAA",
                        style.name.c_str(),modeName(state.mode),styleBuildPending?" / BUILDING":"",perf.cpuFrameMs,perf.gpuFrameMs,perf.gpuPhysicsMs,perf.gpuOpticalMs,perf.gpuSceneMs,perf.groomBuildMs,perf.opticalBuildMs,perf.msaaSamples);ImGui::End();

    if(state.mode==StudioMode::Edit&&!styleBuildPending&&state.showGuides&&!style.graph.curves.empty()) {
        const ImVec2 origin=iv->Pos,size=full,mouse=ImGui::GetIO().MousePos; const bool inside=mouse.x>iv->Pos.x+kSideWidth&&mouse.x<iv->Pos.x+full.x-kSideWidth&&mouse.y>iv->Pos.y+kTopHeight&&mouse.y<iv->Pos.y+full.y-kBottomHeight;
        ImDrawList* dl=ImGui::GetForegroundDrawList(); auto& curve=style.graph.curves[size_t(state.selectedCurve)]; ImVec2 prev{};bool hasPrev=false;int hover=-1;float best=1e9f;std::vector<ImVec2> screen(curve.keys.size());
        for(int i=0;i<int(curve.keys.size());++i){const auto& g=style.graph.sweepKeys[curve.keys[size_t(i)]];ImVec2 sp{};if(!project(viewProj,binding.placement.point(g.centerD),origin,size,sp))continue;screen[size_t(i)]=sp;if(hasPrev)dl->AddLine(prev,sp,IM_COL32(155,110,235,180),2);const bool sel=i==state.selectedKey;dl->AddCircleFilled(sp,sel?7.5f:5.5f,sel?IM_COL32(255,210,110,255):IM_COL32(178,135,255,230));float dx=mouse.x-sp.x,dy=mouse.y-sp.y,dd=dx*dx+dy*dy;if(dd<best&&dd<196){best=dd;hover=i;}prev=sp;hasPrev=true;}
        if(inside&&hover>=0&&ImGui::IsMouseClicked(ImGuiMouseButton_Left)){state.selectedKey=hover;state.draggingMacroGuide=true;} if(!ImGui::IsMouseDown(ImGuiMouseButton_Left))state.draggingMacroGuide=false;
        if(state.draggingMacroGuide&&inside&&state.selectedKey>=0&&state.selectedKey<int(curve.keys.size())) { auto& gate=style.graph.sweepKeys[curve.keys[size_t(state.selectedKey)]]; glm::vec4 clip=viewProj*glm::vec4(binding.placement.point(gate.centerD),1); if(std::abs(clip.w)>1e-6f){float z=clip.z/clip.w,nx=2*(mouse.x-origin.x)/std::max(size.x,1.0f)-1,ny=1-2*(mouse.y-origin.y)/std::max(size.y,1.0f);glm::vec4 wh=glm::inverse(viewProj)*glm::vec4(nx,ny,z,1);if(std::abs(wh.w)>1e-6f){gate.centerD=binding.placement.toCanonicalNormalized(glm::vec3(wh)/wh.w);a.groomDirty=true;}} }
    }
    return a;
}

} // namespace vh::editor
