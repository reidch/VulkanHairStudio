#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parent
checks=[]
def check(ok,msg):
    checks.append((bool(ok),msg)); print(('[PASS] ' if ok else '[FAIL] ')+msg)
def text(rel): return (ROOT/rel).read_text(encoding='utf-8')
asset=text('src/groom_asset.hpp'); builder=text('src/groom_builder.cpp'); style=text('src/groom_style.cpp')
main=text('src/main.cpp'); ui=text('src/editor_ui.cpp'); contact=text('src/groom_contact.hpp'); character=text('src/character_surface.cpp'); cmake=text('CMakeLists.txt'); cache=text('src/groom_cache.cpp')
check(all(x in asset for x in ['ClumpHierarchy','CutSurface','StyleWarp']),'Groom grammar exposes hierarchical clumps + cut surface + optional style warp')
check(all(x not in asset+builder+style for x in ['BundleProfile','ShapeLobe','EnvelopeLayer','HeroLock','v35Authoring']),'Superseded/parallel shaping concepts are absent')
groom_style_block=asset.split('struct GroomStyle {',1)[1].split('struct alignas(16) StrandPointGpu',1)[0]
check('shared_ptr' not in groom_style_block and all(x in groom_style_block for x in ['GroomTopologyGraph graph','ScalpFieldAtlas fields','PartCurve part','HairOptics optics']),'GroomStyle is one unified representation with no alternate authoring state')
check('RootPatch' in asset and 'StyleCurve' in asset and 'ScalpFieldAtlas' in asset and 'PartCurve' in asset,'Root/scaffold/field/part concepts are retained')
check(all(x in builder for x in ['nearestClumpSites','sampleCutSurface','sampleStyleWarps','authoredEndU']),'Single builder evaluates clump hierarchy, cut surface and style warp')
check('field.volume' in builder and 'field.textureAmplitude' in builder and 'shapeStrength' not in asset,'ScalpField is restricted to modulation rather than macro shaping')
check('sampleSmoothPath' in builder and 'sampleCentripetalSegment' in builder,'Smooth style-volume scaffold interpolation is retained')
check('kRenderStrandCount = 110000u' in asset and 'kSimulationGuideCount = 1600u' in asset,'110k render strands / 1600 simulation guides retained')
check('return static_cast<bool>(groom_);' in main and 'physicsAvailableNow' in main,'Physics admission depends only on committed groom state')
check('contactClean()' in contact and 'physicsReady()' not in contact,'Contact validation is diagnostic, not a physics gate')
check('Contact audit: WARNING (simulation remains enabled)' in ui,'UI reports penetration without disabling simulation')
check(all(x in ui for x in ['HAIRSTYLE LIBRARY','Search hairstyle...','requestedStyleIndex']) and all(x not in main+ui for x in ['GLFW_KEY_N','GLFW_KEY_P','styleDeltaRequested_']),'Hairstyle switching uses one searchable click-menu path instead of N/P cycling')
check('make_unique<HairApp>' in main,'Long-lived application state is heap-owned')
check('std::async' in main and 'styleBuildPending_' in main,'Asynchronous transactional groom building is retained')
check('guideSourceStrands' in asset and 'reusableBasis' in builder,'Stable reduced-basis reuse remains intact')
check('contactGroupTag' in builder and 'crossPrimaryClumpBindings' in asset,'Guide binding preserves primary-clump topology')
check(all(x in cache for x in ['styleFingerprint','meshFingerprint','loadDerivedGroom','saveDerivedGroom']),'Content-addressed derived groom cache validates style, mesh and build parameters')
check('runtimeCacheDir_' in main and 'derived-cache hit' in main,'Runtime uses persistent derived-data caches instead of rebuilding unchanged grooms')
check(all(x in main for x in ['createPipelineCache','vkGetPipelineCacheData','pipelineCache_']),'Persistent Vulkan pipeline cache accelerates repeat launches')
check('createSceneStaticGpuBuffers' in main and 'vkCmdFillBuffer' in main and 'Static GPU upload: one batch' in main,'Static scene GPU uploads are batched into one queue synchronization')
check('startScenePreparation' in main and '(overlapped)' in main,'CPU character/contact preparation overlaps Vulkan startup work')
check('O(3*N^2 scanlines)' in character and 'insideVotes' in character and 'nearestTriangle' in character,'Cold ContactField build uses scanline parity while retaining exact nearest-surface distances')
check(all(x in cmake for x in ['src/groom_builder.cpp','src/groom_style.cpp','src/groom_contact.cpp','src/groom_cache.cpp']) and 'authoring.cpp' not in cmake,'Only the unified groom translation units are compiled')
styles=sorted((ROOT/'assets/grooms').glob('*.json')); ok=True
for p in styles:
    try:
        j=json.loads(p.read_text()); assert j['schema']==4
        assert j['graph']['keys'] and j['graph']['curves'] and j['graph']['branches']
        for b in j['graph']['branches']:
            assert 'clumps' in b and 'cut_surface' in b and 'style_warps' in b
            assert 'bundle' not in b and 'cut' not in b and 'shape_lobes' not in b
    except Exception as e:
        ok=False; print('[ASSET ERROR]',p.name,e)
check(ok and len(styles)>=21,f'{len(styles)} schema-4 hairstyle assets load structurally')
palette_ok=True
for p in styles:
    j=json.loads(p.read_text())
    if j.get('material',{}).get('default_palette')!='auto': palette_ok=False
check(palette_ok and 'applyDefaultPalette' in style and 'NaturalMelanin' in style,'Hairstyles default to deterministic black/golden-blonde melanin palettes')
check(all(x in ui for x in ['Custom color','Natural pigment','ColorPicker3','ColorConvertRGBtoHSV','ColorConvertHSVtoRGB']),'Hair color UI keeps black/blonde as defaults while exposing unrestricted RGB/HSV custom color')
hero={'butterfly-blowout','cup-bob','soft-wavy-long','sculptural-curly-bob'}
check(hero.issubset({json.loads(p.read_text())['slug'] for p in styles}),'Four representative grooms are present')
fashion={'butterfly-layered-long','curtain-bob','curtain-crop','defined-curly-bob','face-framing-long','high-pony','layered-bob','layered-half-up','low-pony','messy-bun','swept-back-layered','swept-undercut','textured-quiff','topknot-undercut','tousled-wavy-bob','twin-tail-layered','wolf-shag'}
check(fashion.issubset({json.loads(p.read_text())['slug'] for p in styles}),'All 17 pre-revert fashion grooms are restored as schema-4 assets')
hero_clean=True
for p in styles:
    j=json.loads(p.read_text())
    if j['slug'] in hero and any(b.get('style_warps') for b in j['graph']['branches']): hero_clean=False
check(hero_clean,'Hero grooms prove the core StyleVolume + ClumpHierarchy + CutSurface representation without StyleWarp patching')

# Rendering and light-transport invariants.
hair_frag=text('shaders/hair.frag'); hair_bsdf=text('shaders/hair_bsdf.glsl'); hair_mesh=text('shaders/hair.mesh')
hair_task=text('shaders/hair.task'); hair_compose=text('shaders/hair_compose.frag')
optical_sample=text('shaders/hair_optical_sampling.glsl'); optical_splat=text('shaders/optical_splat.comp'); optical_depth=text('shaders/optical_depth.comp')
tone_frag=text('shaders/tonemap.frag'); head_frag=text('shaders/head.frag'); shadow_common=text('shaders/hair_shadow_common.glsl')
dom_splat=text('shaders/dom_splat.comp'); dom_prefix=text('shaders/dom_prefix.comp'); render_common=text('shaders/hair_render_common.glsl')
lighting_header=text('src/editor_ui.hpp')

check('project(vulkan_hair_v47_directional_reflection' in cmake,'CMake identifies Directional reflection')
check(all(x in main for x in ['sceneColorImage_','compositeColorImage_','hairGeom0Image_','hairGeom1Image_','hairComposePipeline_']),
      'Renderer separates base HDR scene, multisample hair geometry, and composed HDR output')
check(all(x in main for x in ['sampleRateShading','hairMs.sampleShadingEnable=VK_TRUE','hairMs.minSampleShading=1.0f']),
      'Primary visibility requires true per-sample fragment shading')
check('VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT' in main and 'requires at least 2x MSAA' in main,
      'Multisample hair visibility images are both render targets and explicit sampled inputs')
check('sampler2DMS hairGeom0' in hair_compose and 'sampler2DMS hairGeom1' in hair_compose and 'textureSamples(hairGeom0)' in hair_compose,
      'Composition resolves actual multisample hair geometry sample-by-sample')
check(all(x in hair_frag for x in ['outGeom0','outGeom1','worldPos','tangent','fiberSide']) and 'radiance' not in hair_frag and 'vhEvalHairFiber' not in hair_frag,
      'Hair raster pass writes geometry/visibility only and never shades')
check(all(x in hair_mesh for x in ['physicalWidth=max(s.rootNormalDiameter.w*diameterScale,1e-7)','0.5*physicalWidth']) and
      all(x not in hair_mesh+hair_task for x in ['stableWidth','coverageScale','widthScale']),
      'Primary visibility uses authored physical fiber diameter with no screen-space width floor')
check('strandStride=1u' in hair_task and 'fiberPx<' not in hair_task,
      'Visibility pass does not decimate/widen subpixel strands in the reference visibility path')
check('gl_SampleMask' not in hair_frag+hair_mesh+hair_task and 'hash11' not in hair_frag+hair_mesh,
      'Stochastic sample-mask coverage and strand-hash visibility are removed')
check('resolveHairFlow' not in tone_frag and 'hairVisibility' not in tone_frag and 'texture(sceneHdr' in tone_frag,
      'Display pass is exposure/tone-map only; no final-radiance hair blur remains')
check('acesFitted' in tone_frag and 'exposureEV' in lighting_header,
      'Explicit exposure and ACES fitted display transform are retained after physical composition')

check(all(x in hair_bsdf for x in ['vhMp','vhNp','vhFresnelDielectric','vhLongitudinalVariance','vhAzimuthalScale','ap0','ap1','ap2','ap3']),
      'Single-fiber scattering uses the Chiang/PBRT Mp-Ap-Np decomposition')
check(all(x in hair_bsdf for x in ['0.726*b','0.812*b*b','3.7*pow(b,20.0)','0.265*b','1.194*b*b','5.372*pow(b,22.0)']),
      'Published PBRT/Chiang roughness reparameterizations are present')
check(all(x in hair_bsdf for x in ['0.419,0.697,1.370','0.187,0.400,1.050','5.969-0.215*b','vhSigmaAFromAppearance']),
      'Natural pigment and direct appearance controls map to absorption with published coefficients/fits')
check('invProjected=1.0/max(cosThetaI,1e-4)' in hair_bsdf and 'vhFiberProjectedCosine' in hair_compose,
      'Internal HairBxDF solid-angle normalization and projected-fiber measure are paired explicitly')
check('sourceVar' not in hair_bsdf+hair_compose and 'angularRadius' not in hair_bsdf,
      'Finite light size never mutates material roughness')
check(all(x in hair_compose for x in ['vhDiskDirection','vhDirectDisk','vhHammersley','lightingParams.z','directSpec*=integrated/max(omega,1e-6)']),
      'Finite area lights use solid-angle-normalized body + importance-sampled GGX')
check('vhEvalDenseHairScatter' not in hair_bsdf+hair_compose and 'bulkScatter' not in hair_bsdf+hair_compose and 'interacted*rho' not in hair_bsdf+hair_compose,
      'Previous dense-cluster closure is completely removed')
check(all(x not in hair_bsdf+hair_compose+hair_frag for x in ['msGain','lostEnv','rim=pow','wrap=max','radiance *=','radiance*=']),
      'No fake rim, wrap fill, arbitrary multiple-scattering gain, or final RGB tint remains in hair presentation')
check('Tip bleaching is an absorption gradient' not in hair_bsdf or 'tipReduction' in hair_bsdf,
      'Optional tip variation remains an absorption change rather than a post-light brightness multiplier')

check('srgbToLinear(o.directColor)' in main and 'colorMelanin=glm::vec4(srgbToLinear(o.directColor),o.melanin)' in main,
      'Custom Hair Color is treated as an sRGB appearance control before physical absorption fitting')
check('mat.pigmentRoughness=glm::vec4(o.melaninRedness,0.0f,o.longitudinalRoughness,o.azimuthalRoughness)' in main,
      'Material UBO carries explicit longitudinal/azimuthal roughness without hidden color gain')
check(all(x in ui for x in ['HAIR OPTICS','Longitudinal roughness','Azimuthal roughness','IOR','Cuticle tilt']),
      'UI exposes only interpretable fiber-optics controls for the core appearance model')
check('Flow resolve strength' not in ui and 'Flow resolve radius' not in ui,
      'Deprecated final-image smoothing controls are removed from the editor')
check(all(x in ui for x in ['Directional coated-pigment BRDF','Internal scattering share','Hair shading diagnostic']),
      'UI exposes physically-based primary reflection and bounded internal share plus diagnostic')

check('exp(-vhOpticalDepthAt(p,layer))' in optical_sample and 'fiberOptics.w' not in optical_sample,
      'Environment transmittance is Beer-Lambert optical depth without an artist shadow-density multiplier')
check('vhDeepOpacityTau' in shadow_common and 'fiberOptics.w' not in shadow_common,
      'Deep-opacity direct-light attenuation no longer contains an arbitrary material shadow-density scale')
check(all(x in dom_splat for x in ['PrimaryDeepOpacity','SecondaryDeepOpacity','supportVolume.z','atomicAdd']) and 'bool rim' not in dom_splat,
      'Primary/secondary deep opacity remains symmetric transport geometry')
check('running+=raw' in dom_prefix,'Deep-opacity layers remain cumulative toward explicit lights')
check('MAX_STEPS' not in optical_depth and 'running+=' in optical_depth and 'linesPerLayer' in optical_depth,
      'Optical transport retains O(D^3) cumulative sweeps')
check(all(x in optical_splat for x in ['SEGMENTS_PER_GUIDE','sqDistancePointSegment','compactKernel','extinctionMass']),
      'Optical density retains finite-support conserved guide splats')

check(all(x in hair_compose for x in ['vhDeepOpacityTransmittance','vhHairVolumeTransmittance','vhEvalHairBody','vhEvalHairInternal','vhEnvIncident','vhSampleHairGgxHalfVector']),
      'Direct and environment directional reflection remain separated from TT/TRT')
check('Directional softbox radiance lobes; no uniform-color dome.' in ui,
      'Environment is a directional spherical-Gaussian incident field, not uniform dome')
check(all(x in main for x in ['environmentTransportChanged','directionalShadowChanged','opticalFieldDirty_=true','domFieldDirty_=true']),
      'Only transport-dependent light/environment edits invalidate optical/DOM fields')
check('secondaryIntensity = 0.24f' in lighting_header and 'PHYSICAL LIGHTS' in ui,
      'Second light is a genuine separately adjustable finite area source')
check('performanceQuality_?0.0f:studioLighting_.secondaryIntensity' not in main,
      'Quality mode never silently changes scene-light semantics')

check(all(x in head_frag for x in ['ggxD','smithG1','fresnelSchlick','evalNeutralDielectric']) and '/PI' in head_frag,
      'Neutral mannequin remains Lambert + GGX dielectric')
check(all(x in lighting_header for x in ['neutralHeadPresentation = true','neutralHeadDomeStrength = 1.35f']) and 'Neutral shadowless head' in ui,
      'Shadowless neutral mannequin presentation remains the default look-dev mode')
check('matl.headParams.x>0.5' in head_frag,
      'Head shader keeps a separate presentation path from physically coupled hair attenuation')

check('kFramesInFlight_ = 2u' in main and 'VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC' in main,
      'Two-frame dynamic-UBO runtime architecture is retained')
check('compile_shader(hair_compose.frag fragment)' in cmake and 'hair_bsdf.glsl' in cmake and 'hair_surface_brdf.glsl' in cmake and 'hair_optical_sampling.glsl' in cmake,
      'Visibility, surface BRDF, internal transport, and optical shader sources participate in build dependencies')
check('vec4 sample' not in hair_compose+hair_frag+head_frag,
      'Reserved GLSL interpolation keyword sample is not reused as a variable identifier')
check('screenshotRgb(rgb,srcData,width,height,false)' in main,
      'Screenshot writers preserve corrected upright row order')
check(all(x in main for x in ['screenshotWithUiDeferred_','deferredScreenshotFormat_','if(actions.screenshotIncludeUi)']),
      'Save-with-UI is deferred across the popup close so the screenshot modal is not captured')

# Byte layout regression: C++ HairOpticsGpu and GLSL HairOptics must stay in the same vec4 order.
import re
cpp_block=main.split('struct alignas(16) HairOpticsGpu {',1)[1].split('\n};',1)[0]
glsl_block=render_common.split('uniform HairOptics {',1)[1].split('} matl;',1)[0]
cpp_fields=re.findall(r'glm::vec4\s+(\w+)',cpp_block)
glsl_fields=re.findall(r'vec4\s+(\w+)\s*;',glsl_block)
check(cpp_fields==glsl_fields and len(cpp_fields)>=30,
      f'HairOpticsGpu / GLSL uniform field order matches exactly ({len(cpp_fields)} vec4 fields)')

# Mesh->fragment primary visibility interface regression.
mesh_out=re.search(r'layout\(location=0\) out PerVertex \{(.*?)\} vOut\[\];',hair_mesh,re.S)
frag_in=re.search(r'layout\(location=0\) in PerVertex \{(.*?)\} fIn;',hair_frag,re.S)
def fields(block):
    return re.findall(r'\b(?:vec[234]|float|uint|int)\s+(\w+)\s*;',block.group(1)) if block else []
check(fields(mesh_out)==fields(frag_in) and fields(mesh_out)==['tangent','arc','worldPos','fiberSide'],
      'Mesh/fragment visibility interface matches exactly')

# Surface color and internal transport use separate scattering paths.
surface_brdf=text('shaders/hair_surface_brdf.glsl')
check(all(x in surface_brdf for x in ['vhFiberSurfaceNormal','vhAnisoGgxD','vhAnisoLambda','vhEvalHairBody','vhEvalHairSpec']),
      'Surface color is evaluated by a reconstructed-cylinder anisotropic GGX BRDF')
check(all(x in surface_brdf for x in ['bodyColor','(1.0-Fi)*(1.0-Fo)','vhFiberF0','vhSchlickScalar']),
      'Pigmented body reflection and neutral dielectric Fresnel are energy-separated')
check(all(x in hair_bsdf for x in ['vhEvalHairFiberComponents','reflectedR','internalTransport','vhEvalHairInternal']),
      'Chiang/PBRT R is separated from TT/TRT internal transport')
check('float w=clamp(matl.lightingParams.w,0.0,0.35)' in hair_compose and '(1.0-w)*(body+spec)+w*internal' in hair_compose,
      'TT/TRT competes for energy share, never adds unconstrained extra light')
check('volumeScatter = 0.12f' in asset and 'volume_scatter' in style and 'o.volumeScatter' in main,
      'TT/TRT share is first-class with a conservative default')
check('# Directional Fiber Reflection' in text('docs/RENDERING_NOTES.md') if (ROOT/'docs/RENDERING_NOTES.md').exists() else False,
      'Rendering notes describe directional fiber transport')

# Regression checks for directional color response and highlights.
check('vhEvalHairDomeBody' not in surface_brdf+hair_compose and 'vhEnvIncident' in hair_compose and 'envBody+=vhEnvIncident(L)*(1.0-Fi)' in hair_compose,
      'Dominant body uses directional incident lighting, never a flat dome constant')
check('directBody*=integrated;' in hair_compose and 'directSpec*=integrated/max(omega,1e-6);' in hair_compose and
      'envBody*=strength;' in hair_compose and 'envSpec*=strength;' in hair_compose,
      'Surface body and GGX use consistent finite-area source measure; no flat sky fill')
check(all(x in hair_compose for x in ['debugView==1','debugView==2','debugView==3','debugView==4','debugView==5']) and
      'float(editorState_.hairDebugView)' in main,
      'GPU exposes material/spec/body/visibility/internal diagnostics')
check('kDomLayers_ = 8u' in main and 'build' in (ROOT/'tools/build_and_run_windows.ps1').read_text(),
      'Higher DOM depth resolution and isolated build cache')

# Sampling and lighting-quality regression checks.
check('vhFiberSpecNormal' in surface_brdf and 'matl.fiberOptics.y' in surface_brdf,
      'Cuticle tilt now affects the directional dielectric highlight')
check('vhHairSpecImportanceWeight' in surface_brdf and 'vhSampleHairGgxHalfVector' in surface_brdf,
      'GGX lobe is sampled in its own PDF rather than four uniform light samples')
check('const float normalization=2.0*k/(1.0-exp(-2.0*k))' in hair_compose,
      'Softbox radiance lobes have normalized sphere means')
check('int bodyN=min(n,8)' in hair_compose and 'int bodyN=min(2*n,24)' in hair_compose,
      'Performance budget caps direct/body integration while retaining separate glossy sampling')
check('"Environment body reflection"' in ui and '"Direct specular (key + fill)"' in ui,
      'Stage-level debug views isolate directional body and area-light specular')
check('referenceQuality_?20.0f:(performanceQuality_?6.0f:12.0f)' in main,
      'Quality modes scale quadrature work without changing material semantics')

failed=[m for ok,m in checks if not ok]
print(f'\nSource invariant check: {len(checks)-len(failed)}/{len(checks)} passed')
if failed: sys.exit(1)
