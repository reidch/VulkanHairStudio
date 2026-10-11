#ifndef VH_HAIR_BSDF_GLSL
#define VH_HAIR_BSDF_GLSL

// V47 retained reference single-fiber BCSDF for optional TT/TRT.
// Follows the Chiang / PBRT HairBxDF decomposition f = sum_p Mp * Ap * Np.
// This file deliberately contains no dense-cluster closure, view rim, diffuse
// fill, final RGB tint, or light-size-to-roughness shortcut.

const float VH_PI = 3.14159265358979323846;
const float VH_SQRT_PI_OVER_8 = 0.626657069;

float vhSafeSqrt(float x){ return sqrt(max(x,0.0)); }
float vhSqr(float x){ return x*x; }

float vhI0(float x){
    float ax=abs(x);
    if(ax<3.75){
        float y=x/3.75; y*=y;
        return 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.0360768+y*0.0045813)))));
    }
    float y=3.75/max(ax,1e-6);
    float poly=0.39894228+y*(0.01328592+y*(0.00225319+y*(-0.00157565+y*(0.00916281+y*(-0.02057706+y*(0.02635537+y*(-0.01647633+y*0.00392377)))))));
    return exp(ax)*poly/sqrt(ax);
}
float vhLogI0(float x){
    float ax=abs(x);
    if(ax<12.0) return log(max(vhI0(ax),1e-20));
    float inv=1.0/max(ax,1e-6);
    return ax-0.5*log(2.0*VH_PI*ax)+0.125*inv+0.0703125*inv*inv;
}
float vhMp(float cosThetaI,float cosThetaO,float sinThetaI,float sinThetaO,float variance){
    float v=max(variance,1e-5);
    float a=cosThetaI*cosThetaO/v;
    float b=sinThetaI*sinThetaO/v;
    if(v<=0.1) return exp(vhLogI0(a)-b-1.0/v-log(v));
    return exp(-b)*vhI0(a)/(2.0*v*sinh(1.0/v));
}
float vhLogistic(float x,float s){
    s=max(s,1e-5);
    float e=exp(-abs(x)/s);
    return e/(s*vhSqr(1.0+e));
}
float vhLogisticCdf(float x,float s){ return 1.0/(1.0+exp(-x/max(s,1e-5))); }
float vhTrimmedLogistic(float x,float s,float a,float b){
    return vhLogistic(x,s)/max(vhLogisticCdf(b,s)-vhLogisticCdf(a,s),1e-6);
}
float vhWrapPi(float x){ return mod(x+VH_PI,2.0*VH_PI)-VH_PI; }
float vhNp(float phi,int p,float s,float gammaO,float gammaT){
    float phiP=2.0*float(p)*gammaT-2.0*gammaO+float(p)*VH_PI;
    return vhTrimmedLogistic(vhWrapPi(phi-phiP),s,-VH_PI,VH_PI);
}
float vhFresnelDielectric(float cosThetaI,float eta){
    cosThetaI=clamp(cosThetaI,-1.0,1.0);
    float etaI=1.0,etaT=max(eta,1.0001);
    if(cosThetaI<=0.0){ float t=etaI; etaI=etaT; etaT=t; cosThetaI=abs(cosThetaI); }
    float sinThetaI=vhSafeSqrt(1.0-cosThetaI*cosThetaI);
    float sinThetaT=etaI/etaT*sinThetaI;
    if(sinThetaT>=1.0) return 1.0;
    float cosThetaT=vhSafeSqrt(1.0-sinThetaT*sinThetaT);
    float rParl=((etaT*cosThetaI)-(etaI*cosThetaT))/((etaT*cosThetaI)+(etaI*cosThetaT));
    float rPerp=((etaI*cosThetaI)-(etaT*cosThetaT))/((etaI*cosThetaI)+(etaT*cosThetaT));
    return 0.5*(rParl*rParl+rPerp*rPerp);
}
float vhLongitudinalVariance(float betaM){
    float b=clamp(betaM,0.001,1.0);
    float t=0.726*b+0.812*b*b+3.7*pow(b,20.0);
    return t*t;
}
float vhAzimuthalScale(float betaN){
    float b=clamp(betaN,0.001,1.0);
    return VH_SQRT_PI_OVER_8*(0.265*b+1.194*b*b+5.372*pow(b,22.0));
}
float vhAppearanceFitD(float betaN){
    float b=clamp(betaN,0.001,1.0),b2=b*b,b3=b2*b,b4=b3*b,b5=b4*b;
    return max(abs(5.969-0.215*b+2.532*b2-10.73*b3+5.574*b4+0.245*b5),0.25);
}
vec3 vhSigmaAFromAppearance(vec3 appearance,float betaN){
    vec3 c=clamp(appearance,vec3(0.001),vec3(0.999));
    vec3 q=log(c)/vhAppearanceFitD(betaN);
    return q*q;
}

// Material color becomes a wavelength-dependent absorption coefficient.
// DirectColor is an appearance-oriented control; NaturalMelanin uses measured
// eumelanin / pheomelanin absorption coefficients from the PBRT hair model.
vec3 vhSigmaA(float arc){
    float betaN=max(matl.pigmentRoughness.w,0.001);
    vec3 sigmaA;
    if(matl.renderParams.y>0.5){
        float amount=max(matl.colorMelanin.w,0.0);
        float pheoRatio=clamp(matl.pigmentRoughness.x,0.0,1.0);
        sigmaA=amount*(1.0-pheoRatio)*vec3(0.419,0.697,1.370)
              +amount*pheoRatio*vec3(0.187,0.400,1.050);
    }else{
        sigmaA=vhSigmaAFromAppearance(matl.colorMelanin.rgb,betaN);
    }
    // Optional authored tip bleaching is an absorption gradient, never a final-light multiplier.
    float tipReduction=clamp(matl.fiberVariation.x,0.0,0.95);
    return sigmaA*max(0.05,1.0-tipReduction*clamp(arc,0.0,1.0));
}

float vhFiberProjectedCosine(vec3 L,vec3 tangent){
    float s=clamp(dot(normalize(tangent),normalize(L)),-1.0,1.0);
    return vhSafeSqrt(1.0-s*s);
}

void vhEvalHairFiberComponents(vec3 L,vec3 V,vec3 tangent,float h,vec3 sigmaA,
                               out vec3 reflectedR,out vec3 internalTransport){
    vec3 T=normalize(tangent); L=normalize(L); V=normalize(V);
    float sinThetaI=clamp(dot(T,L),-0.999999,0.999999);
    float sinThetaO=clamp(dot(T,V),-0.999999,0.999999);
    float cosThetaI=vhSafeSqrt(1.0-sinThetaI*sinThetaI);
    float cosThetaO=vhSafeSqrt(1.0-sinThetaO*sinThetaO);

    vec3 Li=L-T*sinThetaI,Vo=V-T*sinThetaO;
    Li=dot(Li,Li)>1e-10?normalize(Li):vec3(0.0,1.0,0.0);
    Vo=dot(Vo,Vo)>1e-10?normalize(Vo):vec3(0.0,1.0,0.0);
    float phi=atan(dot(cross(Vo,Li),T),clamp(dot(Vo,Li),-1.0,1.0));

    float eta=max(matl.fiberOptics.x,1.0001);
    h=clamp(h,-0.999,0.999);
    float gammaO=asin(h);
    float sinThetaT=sinThetaO/eta;
    float cosThetaT=vhSafeSqrt(1.0-sinThetaT*sinThetaT);
    float etaP=vhSafeSqrt(eta*eta-sinThetaO*sinThetaO)/max(cosThetaO,1e-5);
    float sinGammaT=clamp(h/max(etaP,1.0001),-0.999,0.999);
    float cosGammaT=vhSafeSqrt(1.0-sinGammaT*sinGammaT);
    float gammaT=asin(sinGammaT);
    vec3 transmittance=exp(-sigmaA*(2.0*cosGammaT/max(cosThetaT,1e-4)));

    float cosGammaO=vhSafeSqrt(1.0-h*h);
    float F=vhFresnelDielectric(cosThetaO*cosGammaO,eta);
    vec3 ap0=vec3(F);
    vec3 ap1=vhSqr(1.0-F)*transmittance;
    vec3 ap2=ap1*transmittance*F;
    vec3 Tf=transmittance*F;
    vec3 ap3=ap2*Tf/max(vec3(1.0)-Tf,vec3(1e-4));

    float betaM=clamp(matl.pigmentRoughness.z,0.001,1.0);
    float betaN=clamp(matl.pigmentRoughness.w,0.001,1.0);
    float v0=vhLongitudinalVariance(betaM);
    float vR=v0,vTT=0.25*v0,vTRT=4.0*v0;
    float s=vhAzimuthalScale(betaN);

    float alpha=matl.fiberOptics.y;
    float sa=sin(alpha),ca=cos(alpha);
    float s2=2.0*sa*ca,c2=ca*ca-sa*sa;
    float s4=2.0*s2*c2,c4=c2*c2-s2*s2;
    float sinO0=sinThetaO*c2-cosThetaO*s2;
    float cosO0=abs(cosThetaO*c2+sinThetaO*s2);
    float sinO1=sinThetaO*ca+cosThetaO*sa;
    float cosO1=abs(cosThetaO*ca-sinThetaO*sa);
    float sinO2=sinThetaO*c4+cosThetaO*s4;
    float cosO2=abs(cosThetaO*c4-sinThetaO*s4);

    float invProjected=1.0/max(cosThetaI,1e-4);
    reflectedR=max(vhMp(cosThetaI,cosO0,sinThetaI,sinO0,vR)*ap0*vhNp(phi,0,s,gammaO,gammaT)*invProjected,vec3(0.0));
    internalTransport=vec3(0.0);
    internalTransport+=vhMp(cosThetaI,cosO1,sinThetaI,sinO1,vTT)*ap1*vhNp(phi,1,s,gammaO,gammaT);
    internalTransport+=vhMp(cosThetaI,cosO2,sinThetaI,sinO2,vTRT)*ap2*vhNp(phi,2,s,gammaO,gammaT);
    internalTransport+=vhMp(cosThetaI,cosThetaO,sinThetaI,sinThetaO,vTRT)*ap3*(1.0/(2.0*VH_PI));
    internalTransport=max(internalTransport*invProjected,vec3(0.0));
}

vec3 vhEvalHairFiber(vec3 L,vec3 V,vec3 tangent,float h,vec3 sigmaA){
    vec3 r,t; vhEvalHairFiberComponents(L,V,tangent,h,sigmaA,r,t); return r+t;
}
vec3 vhEvalHairInternal(vec3 L,vec3 V,vec3 tangent,float h,vec3 sigmaA){
    vec3 r,t; vhEvalHairFiberComponents(L,V,tangent,h,sigmaA,r,t); return t;
}

#endif
