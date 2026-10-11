#ifndef VH_HAIR_RENDER_COMMON_GLSL
#define VH_HAIR_RENDER_COMMON_GLSL

// Lighting UBO layout matches HairOpticsGpu in src/main.cpp.
// Radiance and material color are stored in linear space.
layout(set=3,binding=0,std140) uniform HairOptics {
    vec4 colorMelanin;        // rgb dyed appearance target (linear), a eumelanin concentration
    vec4 pigmentRoughness;    // x pheomelanin ratio, y reserved, z beta_m, w beta_n
    vec4 fiberOptics;         // x eta, y cuticle tilt (radians), z/w reserved
    vec4 ambientParams;       // x incident environment radiance, y exposure, z active softboxes, w diagnostic mode
    vec4 opticalGrid;         // xyz origin, w cell size
    vec4 renderParams;        // x optical dim, y natural-pigment flag, z optical directions, w fixed-point scale
    vec4 fiberShape;          // x root diameter, y tip diameter scale, z support kernel scale, w reserved
    vec4 primaryDirI;         // xyz point-to-primary-light direction, w incident radiance scale
    vec4 primaryColorShadow;  // rgb light color, w reserved
    vec4 envDir0I;
    vec4 envDir1I;
    vec4 envDir2I;
    vec4 envDir3I;
    vec4 envColor0;
    vec4 envColor1;
    vec4 envColor2;
    vec4 envColor3;
    vec4 secondaryDirI;       // ordinary second physical light; no special rim behavior
    vec4 secondaryColorShadow;// rgb light color, w reserved
    vec4 domParams;           // x DOM dimension, y layer count, z fixed scale, w reserved
    vec4 lightingParams;      // x/y angular radius, z GGX importance samples / light quadrature, w internal transport share [0,0.35]
    vec4 fiberVariation;      // x tip absorption reduction, y/z/w reserved
    vec4 headParams;          // x neutral head presentation, y isotropic dome strength
    vec4 skyTopColor;
    vec4 skyHorizonColor;
    vec4 skyBottomColor;
    vec4 skyBlob0;
    vec4 skyBlob1;
    vec4 skyBlob0Color;
    vec4 skyBlob1Color;
    vec4 skyParams;
} matl;

vec4 vhEnvDirI(int i) {
    if(i==0) return matl.envDir0I;
    if(i==1) return matl.envDir1I;
    if(i==2) return matl.envDir2I;
    return matl.envDir3I;
}
vec3 vhEnvColor(int i) {
    if(i==0) return matl.envColor0.rgb;
    if(i==1) return matl.envColor1.rgb;
    if(i==2) return matl.envColor2.rgb;
    return matl.envColor3.rgb;
}
vec3 vhOpticalDirection(int layer) {
    if(layer==0) return normalize(matl.primaryDirI.xyz);
    return normalize(vhEnvDirI(layer-1).xyz);
}

#endif
