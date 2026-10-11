#version 460
#include "hair_shadow_common.glsl"
#include "hair_optical_sampling.glsl"
#include "hair_bsdf.glsl"
#include "hair_surface_brdf.glsl"

layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std140) uniform Frame {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 frustumPlanes[6];
    vec4 screenParams;
} frame;
layout(set=1,binding=0) uniform sampler2D sceneBase;
layout(set=1,binding=1) uniform sampler2DMS hairGeom0;
layout(set=1,binding=2) uniform sampler2DMS hairGeom1;

// Diagnostic output paths: 0 final, 1 unlit pigment, 2 all body, 3 all
// neutral spec, 4 key DOM, 5 internal, 6 environment body, 7 key+fill body,
// 8 direct spec, 9 environment spec. Always run through normal MSAA coverage.
int vhDebugView(){return clamp(int(matl.ambientParams.w+0.5),0,9);}
float vhRadicalInverse(uint i){
    i=(i<<16u)|(i>>16u);
    i=((i&0x55555555u)<<1u)|((i&0xAAAAAAAAu)>>1u);
    i=((i&0x33333333u)<<2u)|((i&0xCCCCCCCCu)>>2u);
    i=((i&0x0F0F0F0Fu)<<4u)|((i&0xF0F0F0F0u)>>4u);
    i=((i&0x00FF00FFu)<<8u)|((i&0xFF00FF00u)>>8u);
    return float(i)*2.3283064365386963e-10;
}
vec2 vhHammersley(int i,int count){return vec2((float(i)+0.5)/float(count),vhRadicalInverse(uint(i)));}
void vhBasis(vec3 N,out vec3 X,out vec3 Y){
    vec3 ref=abs(N.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0);
    X=normalize(cross(ref,N));Y=normalize(cross(N,X));
}

vec3 vhDiskDirection(vec3 center,float radius,int i,int n){
    if(n<=1 || radius<=1e-5)return normalize(center);
    vec2 xi=vhHammersley(i,n);
    float r=sqrt(xi.x)*tan(radius),phi=2.0*VH_PI*xi.y;
    vec3 X,Y;vhBasis(normalize(center),X,Y);
    return normalize(center+r*(cos(phi)*X+sin(phi)*Y));
}

// The four sky presets describe *directional incident radiance*, represented
// as non-negative spherical-Gaussian softboxes. Each lobe is normalized so its
// sphere-mean equals its supplied weight*color, rather than acting as a delta
// quadrature sample or as an isotropic emission term.
vec3 vhEnvIncident(vec3 L){
    float sumW=0.0;
    vec3 v=vec3(0.0);
    const float k=2.7;
    const float normalization=2.0*k/(1.0-exp(-2.0*k));
    int n=clamp(int(matl.ambientParams.z+0.5),1,4);
    for(int i=0;i<n;++i){
        vec4 d=vhEnvDirI(i);
        float w=max(d.w,0.0);
        sumW+=w;
        float g=normalization*exp(k*(dot(L,normalize(d.xyz))-1.0));
        v+=w*g*max(vhEnvColor(i),vec3(0.0));
    }
    return v/max(sumW,1e-6);
}

// Directional explicit source, integrated as a solid-angle disk for broad
// body; anisotropic GGX is evaluated by half-vector importance quadrature.
// Both are multiplied by the SAME direction-specific light visibility, and
// only once. Coarse optical volume cannot darken the entire final material.
void vhDirectDisk(vec3 center,vec3 color,float intensity,float radius,bool secondary,
                  vec3 P,vec3 V,vec3 T,vec3 N,float h,vec3 sigmaA,vec3 bodyColor,
                  out vec3 directBody,out vec3 directSpec,out vec3 directInternal){
    directBody=vec3(0.0);directSpec=vec3(0.0);directInternal=vec3(0.0);
    if(intensity<=0.0)return;
    center=normalize(center);
    int n=clamp(int(matl.lightingParams.z+0.5),4,24);
    int bodyN=min(n,8);
    float Tr=vhDeepOpacityTransmittance(P,secondary,1.0);
    for(int i=0;i<bodyN;++i){
        vec3 L=vhDiskDirection(center,radius,i,bodyN);
        directBody+=vhEvalHairBody(L,V,N,bodyColor);
    }
    directBody/=float(bodyN);
    // The anisotropic GGX half-vector is sampled in its natural measure.
    // Testing the reflected L against the actual angular lamp gives the
    // finite-disk integral, instead of editing roughness to fake source size.
    float capCos=cos(max(radius,0.001));
    vec3 Ns=vhFiberSpecNormal(N,T);
    for(int i=0;i<n;++i){
        vec3 H=vhSampleHairGgxHalfVector(Ns,T,vhHammersley(i,n));
        float VoH=dot(V,H);
        if(VoH<=0.0)continue;
        vec3 L=normalize(2.0*VoH*H-V);
        if(dot(L,center)<capCos || dot(N,L)<=0.0)continue;
        float g=vhHairSpecImportanceWeight(L,V,Ns,T,H);
        directSpec+=vec3(g);
    }
    directSpec/=float(n);
    // TT/TRT is an optional secondary path and uses the same physical source.
    float internalShare=clamp(matl.lightingParams.w,0.0,0.35);
    if(internalShare>1e-5 || vhDebugView()==5){
        const int internalN=4;
        for(int i=0;i<internalN;++i){
            vec3 L=vhDiskDirection(center,radius,i,internalN);
            directInternal+=vhEvalHairInternal(L,V,T,h,sigmaA)*vhFiberProjectedCosine(L,T);
        }
        directInternal/=float(internalN);
    }
    // intensity is the solid-angle integrated source radiance.  A smaller
    // disk has proportionally higher radiance, not weaker energy. Body and
    // GGX now use the SAME normalized finite-area source measure.
    float omega=2.0*VH_PI*(1.0-cos(max(radius,0.001)));
    vec3 integrated=max(color,vec3(0.0))*intensity*Tr;
    directBody*=integrated;
    directSpec*=integrated/max(omega,1e-6);
    directInternal*=integrated;
}

void vhEnvironment(vec3 P,vec3 V,vec3 T,vec3 N,float h,vec3 sigmaA,vec3 bodyColor,
                   out vec3 envBody,out vec3 envSpec,out vec3 envInternal){
    envBody=vec3(0.0);envSpec=vec3(0.0);envInternal=vec3(0.0);
    float strength=max(matl.ambientParams.x,0.0);
    if(strength<=0.0)return;
    int n=clamp(int(matl.lightingParams.z+0.5),4,24);
    int bodyN=min(2*n,24);
    vec3 X,Y;vhBasis(N,X,Y);
    // p(L)=(N.L)/pi, so body integral estimator simplifies exactly to
    // C*(1-Fi)*(1-Fo)*Lenv; it retains orientation against the light field.
    float NoV=max(dot(N,V),1e-5);
    float Fo=vhSchlickScalar(NoV,vhFiberF0());
    for(int i=0;i<bodyN;++i){
        vec2 xi=vhHammersley(i,bodyN);
        float r=sqrt(xi.x),phi=2.0*VH_PI*xi.y;
        vec3 L=normalize(r*cos(phi)*X+r*sin(phi)*Y+sqrt(1.0-xi.x)*N);
        float Fi=vhSchlickScalar(max(dot(N,L),0.0),vhFiberF0());
        envBody+=vhEnvIncident(L)*(1.0-Fi);
    }
    envBody*=bodyColor*((1.0-Fo)/float(bodyN));
    vec3 Ns=vhFiberSpecNormal(N,T);
    // Specular importance quadrature follows the BRDF lobe instead of the
    // sky's four source centers; highlights remain visible at low roughness.
    for(int i=0;i<n;++i){
        vec3 H=vhSampleHairGgxHalfVector(Ns,T,vhHammersley(i,n));
        float VoH=dot(V,H);
        if(VoH<=0.0)continue;
        vec3 L=normalize(2.0*VoH*H-V);
        float weight=(dot(N,L)>0.0)?vhHairSpecImportanceWeight(L,V,Ns,T,H):0.0;
        envSpec+=weight*vhEnvIncident(L);
    }
    envSpec/=float(n);
    // The unvalidated guide-resolution extinction field is not applied to
    // primary surface reflection (V45 black-hair regression). Optional TT/TRT
    // remains volume-attenuated, and never controls the visibility of RGB.
    float internalShare=clamp(matl.lightingParams.w,0.0,0.35);
    if(internalShare>1e-5 || vhDebugView()==5){
        int envCount=clamp(int(matl.ambientParams.z+0.5),1,4);
        float weightSum=0.0;
        for(int i=0;i<envCount;++i){
            vec4 d=vhEnvDirI(i);
            float w=max(d.w,0.0);weightSum+=w;
            vec3 L=normalize(d.xyz);
            float Tr=vhHairVolumeTransmittance(P,i+1);
            envInternal+=w*Tr*max(vhEnvColor(i),vec3(0.0))*
                vhEvalHairInternal(L,V,T,h,sigmaA)*vhFiberProjectedCosine(L,T);
        }
        envInternal/=max(weightSum,1e-6);
    }
    envBody*=strength;
    envSpec*=strength;
    envInternal*=strength;
}

vec3 vhShadeFiber(vec4 g0,vec4 g1){
    vec3 P=g0.xyz;
    float arc=clamp(g0.w-1.0,0.0,1.0);
    vec3 T=normalize(g1.xyz);
    float h=clamp(g1.w,-0.999,0.999);
    vec3 V=normalize(frame.cameraPos.xyz-P);
    vec3 N=vhFiberSurfaceNormal(V,T,h);
    vec3 sigmaA=vhSigmaA(arc);
    vec3 bodyColor=vhAppearanceFromSigmaA(sigmaA,max(matl.pigmentRoughness.w,0.001));
    int debugView=vhDebugView();
    if(debugView==1)return bodyColor;
    if(debugView==4)return vec3(vhDeepOpacityTransmittance(P,false,1.0));
    vec3 b0,s0,i0,b1,s1,i1,be,se,ie;
    vhDirectDisk(matl.primaryDirI.xyz,matl.primaryColorShadow.rgb,max(matl.primaryDirI.w,0.0),
                 max(matl.lightingParams.x,0.0),false,P,V,T,N,h,sigmaA,bodyColor,b0,s0,i0);
    vhDirectDisk(matl.secondaryDirI.xyz,matl.secondaryColorShadow.rgb,max(matl.secondaryDirI.w,0.0),
                 max(matl.lightingParams.y,0.0),true,P,V,T,N,h,sigmaA,bodyColor,b1,s1,i1);
    vhEnvironment(P,V,T,N,h,sigmaA,bodyColor,be,se,ie);
    vec3 body=b0+b1+be,spec=s0+s1+se,internal=i0+i1+ie;
    if(debugView==2)return body;
    if(debugView==3)return spec;
    if(debugView==5)return internal;
    if(debugView==6)return be;
    if(debugView==7)return b0+b1;
    if(debugView==8)return s0+s1;
    if(debugView==9)return se;
    float w=clamp(matl.lightingParams.w,0.0,0.35);
    return max((1.0-w)*(body+spec)+w*internal,vec3(0.0));
}

void main(){
    ivec2 px=ivec2(gl_FragCoord.xy);
    int n=textureSamples(hairGeom0);
    vec3 base=max(texture(sceneBase,vUv).rgb,vec3(0.0));
    vec3 sum=vec3(0.0);
    for(int s=0;s<n;++s){
        vec4 g0=texelFetch(hairGeom0,px,s);
        if(g0.w>0.5){
            vec4 g1=texelFetch(hairGeom1,px,s);
            sum+=vhShadeFiber(g0,g1);
        }else sum+=base;
    }
    outColor=vec4(sum/float(max(n,1)),1.0);
}
