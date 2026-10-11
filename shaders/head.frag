#version 460
#include "hair_shadow_common.glsl"
#include "hair_optical_sampling.glsl"

layout(location=0) in vec3 vWorldPos;
layout(location=1) in vec3 vNormal;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std140) uniform Frame {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 frustumPlanes[6];
    vec4 screenParams;
} frame;

const float PI=3.14159265358979323846;

float saturate(float x){ return clamp(x,0.0,1.0); }
vec3 fresnelSchlick(float cosTheta,vec3 F0){
    float x=pow(1.0-saturate(cosTheta),5.0);
    return F0+(vec3(1.0)-F0)*x;
}
float ggxD(float NoH,float alpha){
    float a2=alpha*alpha;
    float d=NoH*NoH*(a2-1.0)+1.0;
    return a2/max(PI*d*d,1e-6);
}
float smithG1(float NoX,float alpha){
    float a2=alpha*alpha;
    return 2.0*NoX/max(NoX+sqrt(a2+(1.0-a2)*NoX*NoX),1e-6);
}
vec3 neutralBaseColor(){ return vec3(0.68,0.69,0.71); }

vec3 evalNeutralDielectric(vec3 N,vec3 V,vec3 L){
    float NoV=max(dot(N,V),0.0), NoL=max(dot(N,L),0.0);
    if(NoV<=0.0||NoL<=0.0) return vec3(0.0);
    vec3 H=normalize(V+L);
    float NoH=max(dot(N,H),0.0), VoH=max(dot(V,H),0.0);

    // Neutral mannequin material: ordinary dielectric microfacet surface.
    const float eta=1.45;
    const float roughness=0.48;
    float alpha=roughness*roughness;
    float f0=(eta-1.0)/(eta+1.0); f0*=f0;
    vec3 F=fresnelSchlick(VoH,vec3(f0));
    float D=ggxD(NoH,alpha);
    float G=smithG1(NoV,alpha)*smithG1(NoL,alpha);
    vec3 spec=F*(D*G/max(4.0*NoV*NoL,1e-5));

    vec3 diffuse=(vec3(1.0)-F)*neutralBaseColor()/PI;
    return (diffuse+spec)*NoL;
}

vec3 averageEnvironmentRadiance(){
    float totalWeight=0.0;
    vec3 weighted=vec3(0.0);
    for(int i=0;i<4;++i){
        vec4 envSample=vhEnvDirI(i);
        float w=max(envSample.w,0.0);
        weighted += w*max(vhEnvColor(i),vec3(0.0));
        totalWeight += w;
    }
    if(totalWeight<=1e-6) return vec3(0.0);
    return max(matl.ambientParams.x,0.0)*weighted/totalWeight;
}

vec3 shadeNeutralPresentation(vec3 N,vec3 V){
    // Presentation mode is deliberately isolated from the hair transport model.
    // The mannequin is treated as if it sits inside a diffuse light tent:
    // for uniform incident radiance, the Lambertian hemisphere integral is
    // exactly rho * L, so this gives a stable shadow-free baseline without
    // inventing an edge/rim term. Explicit lights remain ordinary BRDF lights,
    // but hair DOM/volume attenuation is not applied to the mannequin.
    vec3 radiance=neutralBaseColor()*averageEnvironmentRadiance()*max(matl.headParams.y,0.0);

    if(matl.primaryDirI.w>0.0){
        vec3 L=normalize(matl.primaryDirI.xyz);
        radiance += matl.primaryDirI.w*max(matl.primaryColorShadow.rgb,vec3(0.0))*evalNeutralDielectric(N,V,L);
    }
    if(matl.secondaryDirI.w>0.0){
        vec3 L=normalize(matl.secondaryDirI.xyz);
        radiance += matl.secondaryDirI.w*max(matl.secondaryColorShadow.rgb,vec3(0.0))*evalNeutralDielectric(N,V,L);
    }
    return radiance;
}

vec3 shadePhysicallyCoupled(vec3 N,vec3 V){
    vec3 radiance=vec3(0.0);

    if(matl.primaryDirI.w>0.0){
        vec3 L=normalize(matl.primaryDirI.xyz);
        float Tr=vhDeepOpacityTransmittance(vWorldPos,false,matl.primaryColorShadow.w);
        radiance += matl.primaryDirI.w*Tr*max(matl.primaryColorShadow.rgb,vec3(0.0))*evalNeutralDielectric(N,V,L);
    }
    if(matl.secondaryDirI.w>0.0){
        vec3 L=normalize(matl.secondaryDirI.xyz);
        float Tr=vhDeepOpacityTransmittance(vWorldPos,true,matl.secondaryColorShadow.w);
        radiance += matl.secondaryDirI.w*Tr*max(matl.secondaryColorShadow.rgb,vec3(0.0))*evalNeutralDielectric(N,V,L);
    }

    float envStrength=max(matl.ambientParams.x,0.0);
    int envCount=clamp(int(matl.ambientParams.z+0.5),1,4);
    float totalWeight=0.0, activeWeight=0.0;
    for(int i=0;i<4;++i) totalWeight+=max(vhEnvDirI(i).w,0.0);
    for(int i=0;i<envCount;++i) activeWeight+=max(vhEnvDirI(i).w,0.0);
    float renorm=totalWeight/max(activeWeight,1e-5);
    for(int i=0;i<envCount;++i){
        vec4 envSample=vhEnvDirI(i);
        float wi=max(envSample.w,0.0)*envStrength*renorm;
        if(wi<=0.0) continue;
        float Tr=vhHairVolumeTransmittance(vWorldPos,i+1);
        radiance += wi*Tr*max(vhEnvColor(i),vec3(0.0))*evalNeutralDielectric(N,V,normalize(envSample.xyz));
    }
    return radiance;
}

void main(){
    vec3 N=normalize(vNormal);
    vec3 V=normalize(frame.cameraPos.xyz-vWorldPos);
    bool neutralPresentation=matl.headParams.x>0.5;
    vec3 radiance=neutralPresentation?shadeNeutralPresentation(N,V):shadePhysicallyCoupled(N,V);
    outColor=vec4(max(radiance,vec3(0.0)),1.0);
}
