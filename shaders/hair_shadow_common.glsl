#ifndef VH_HAIR_SHADOW_COMMON_GLSL
#define VH_HAIR_SHADOW_COMMON_GLSL
#include "hair_render_common.glsl"

// Directional deep-opacity fields. Primary and secondary are identical physical
// light types; separate fields only preserve each light's directional transport.
layout(set=3,binding=3,std430) readonly buffer PrimaryDeepOpacity { uint primaryDomFixed[]; };
layout(set=3,binding=4,std430) readonly buffer SecondaryDeepOpacity { uint secondaryDomFixed[]; };

const float VH_SQRT3 = 1.7320508075688772;

void vhLightBasis(vec3 L, out vec3 right, out vec3 up){
    L=normalize(L);
    vec3 ref=abs(L.y)<0.92?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0);
    right=normalize(cross(ref,L));
    up=normalize(cross(L,right));
}

int vhDomIndex(ivec2 p,int layer,int dim){ return layer*dim*dim+p.y*dim+p.x; }
float vhDomRead(bool secondary,ivec2 p,int layer,int dim,float invFixed){
    int i=vhDomIndex(p,layer,dim);
    return secondary?float(secondaryDomFixed[i])*invFixed:float(primaryDomFixed[i])*invFixed;
}

float vhDeepOpacityTau(vec3 worldPos,bool secondary){
    int dim=max(int(matl.domParams.x+0.5),8);
    int layers=clamp(int(matl.domParams.y+0.5),2,8);
    float invFixed=1.0/max(matl.domParams.z,1.0);
    int opticalDim=max(int(matl.renderParams.x+0.5),2);
    float cell=max(matl.opticalGrid.w,1e-5);
    vec3 center=matl.opticalGrid.xyz+vec3(0.5*cell*float(opticalDim));
    float cubeHalf=0.5*cell*float(opticalDim);
    float halfSpan=max(cubeHalf*VH_SQRT3,0.05);
    vec3 L=normalize(secondary?matl.secondaryDirI.xyz:matl.primaryDirI.xyz);
    vec3 R,U; vhLightBasis(L,R,U);
    vec3 q=worldPos-center;
    vec2 uv=0.5+0.5*vec2(dot(q,R),dot(q,U))/halfSpan;
    float z=clamp(0.5+0.5*dot(q,L)/halfSpan,0.0,0.999999);
    if(any(lessThan(uv,vec2(0.0)))||any(greaterThan(uv,vec2(1.0)))) return 0.0;
    vec2 gp=uv*float(dim)-vec2(0.5);
    ivec2 b=clamp(ivec2(floor(gp)),ivec2(0),ivec2(dim-2));
    vec2 f=clamp(gp-vec2(b),vec2(0.0),vec2(1.0));
    float lf=z*float(layers);
    int l=clamp(int(floor(lf)),0,layers-1);
    float t=clamp(lf-float(l),0.0,1.0);
    float tau=0.0;
    for(int dy=0;dy<=1;++dy) for(int dx=0;dx<=1;++dx){
        ivec2 c=b+ivec2(dx,dy);
        float w=(dx!=0?f.x:1.0-f.x)*(dy!=0?f.y:1.0-f.y);
        float a=vhDomRead(secondary,c,l,dim,invFixed);
        float bnd=(l+1<layers)?vhDomRead(secondary,c,l+1,dim,invFixed):0.0;
        tau+=mix(a,bnd,t)*w;
    }
    return max(tau,0.0);
}

float vhDeepOpacityTransmittance(vec3 worldPos,bool secondary,float strength){
    return exp(-max(strength,0.0)*vhDeepOpacityTau(worldPos,secondary));
}
#endif
