#ifndef VH_HAIR_SURFACE_BRDF_GLSL
#define VH_HAIR_SURFACE_BRDF_GLSL

// Directional fiber response: pigmented body reflection and anisotropic
// dielectric GGX, evaluated on the visible cylindrical surface.
vec3 vhAppearanceFromSigmaA(vec3 sigmaA,float betaN){
    return clamp(exp(-vhAppearanceFitD(betaN)*sqrt(max(sigmaA,vec3(0.0)))),vec3(0.0),vec3(1.0));
}

vec3 vhFiberSurfaceNormal(vec3 V,vec3 T,float h){
    T=normalize(T); V=normalize(V);
    vec3 Vn=V-T*dot(V,T);
    if(dot(Vn,Vn)<1e-10){
        vec3 ref=abs(T.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0);
        Vn=normalize(cross(cross(T,ref),T));
    }else Vn=normalize(Vn);
    vec3 S=normalize(cross(T,Vn));
    return normalize(vhSafeSqrt(1.0-clamp(h*h,0.0,1.0))*Vn+h*S);
}

float vhAnisoGgxD(vec3 N,vec3 X,vec3 Y,vec3 H,float ax,float ay){
    float NoH=max(dot(N,H),0.0);
    if(NoH<=0.0) return 0.0;
    float hx=dot(H,X),hy=dot(H,Y);
    float q=hx*hx/(ax*ax)+hy*hy/(ay*ay)+NoH*NoH;
    return 1.0/(VH_PI*ax*ay*q*q);
}
float vhAnisoLambda(vec3 N,vec3 X,vec3 Y,vec3 W,float ax,float ay){
    float NoW=max(dot(N,W),1e-5);
    float wx=dot(W,X),wy=dot(W,Y);
    float a2=(ax*wx)*(ax*wx)+(ay*wy)*(ay*wy);
    return 0.5*(sqrt(1.0+a2/(NoW*NoW))-1.0);
}
float vhSchlickScalar(float cosTheta,float f0){
    float m=pow(1.0-clamp(cosTheta,0.0,1.0),5.0);
    return f0+(1.0-f0)*m;
}

// Cuticle scales tilt the dielectric highlight distribution along the fiber.
// Body shading and coverage use the original cylinder normal.
vec3 vhFiberSpecNormal(vec3 N,vec3 T){
    float tilt=clamp(matl.fiberOptics.y,-0.14,0.14);
    return normalize(cos(tilt)*N+sin(tilt)*T);
}

float vhFiberF0(){
    float eta=max(matl.fiberOptics.x,1.0001);
    return vhSqr((eta-1.0)/(eta+1.0));
}

// Coated-Lambert approximation for pigmented body reflection.
// Fresnel transmission is evaluated at both interfaces.
vec3 vhEvalHairBody(vec3 L,vec3 V,vec3 N,vec3 bodyColor){
    float NoL=max(dot(N,L),0.0);
    float NoV=max(dot(N,V),1e-5);
    if(NoL<=0.0) return vec3(0.0);
    float f0=vhFiberF0();
    float Fi=vhSchlickScalar(NoL,f0),Fo=vhSchlickScalar(NoV,f0);
    return clamp(bodyColor,vec3(0.0),vec3(1.0))*((1.0-Fi)*(1.0-Fo)*NoL/VH_PI);
}

void vhSpecFrame(vec3 N,vec3 T,out vec3 X,out vec3 Y,out float ax,out float ay){
    X=normalize(T-N*dot(T,N));
    Y=normalize(cross(N,X));
    float rt=clamp(matl.pigmentRoughness.z,0.025,0.95);
    float rb=clamp(matl.pigmentRoughness.w,0.025,0.95);
    ax=max(rt*rt,0.0025);
    ay=max(rb*rb,0.0025);
}

// Material roughness and source angular extent are evaluated separately.
float vhEvalHairSpec(vec3 L,vec3 V,vec3 N,vec3 T){
    float NoL=max(dot(N,L),0.0),NoV=max(dot(N,V),1e-5);
    if(NoL<=0.0) return 0.0;
    vec3 H=L+V;
    if(dot(H,H)<1e-10) return 0.0;
    H=normalize(H);
    vec3 X,Y;float ax,ay;vhSpecFrame(N,T,X,Y,ax,ay);
    float D=vhAnisoGgxD(N,X,Y,H,ax,ay);
    float G=1.0/(1.0+vhAnisoLambda(N,X,Y,L,ax,ay)+vhAnisoLambda(N,X,Y,V,ax,ay));
    float F=vhSchlickScalar(max(dot(V,H),0.0),vhFiberF0());
    return D*G*F/max(4.0*NoV,1e-5); // f_spec * (N.L)
}

void vhEvalHairSurfaceComponents(vec3 L,vec3 V,vec3 T,float h,vec3 bodyColor,
                                 out vec3 body,out vec3 spec){
    vec3 N=vhFiberSurfaceNormal(V,T,h);
    body=vhEvalHairBody(normalize(L),normalize(V),N,bodyColor);
    spec=vec3(vhEvalHairSpec(normalize(L),normalize(V),vhFiberSpecNormal(N,normalize(T)),normalize(T)));
}

// GGX half-vector sampling, with p(L)=D(H)*(N.H)/(4*V.H).
vec3 vhSampleHairGgxHalfVector(vec3 N,vec3 T,vec2 xi){
    vec3 X,Y;float ax,ay;vhSpecFrame(N,T,X,Y,ax,ay);
    float phi=atan(ay*sin(2.0*VH_PI*xi.x),ax*cos(2.0*VH_PI*xi.x));
    float cp=cos(phi),sp=sin(phi);
    float alpha2=1.0/(cp*cp/(ax*ax)+sp*sp/(ay*ay));
    float tan2=alpha2*xi.y/max(1.0-xi.y,1e-6);
    float inv=sqrt(1.0+tan2);
    return normalize((sqrt(tan2)/inv)*(cp*X+sp*Y)+(1.0/inv)*N);
}

// NDF importance weighting cancels D against the half-vector PDF.
float vhHairSpecImportanceWeight(vec3 L,vec3 V,vec3 N,vec3 T,vec3 H){
    float NoL=dot(N,L),NoV=dot(N,V),NoH=dot(N,H),VoH=dot(V,H);
    if(NoL<=0.0 || NoV<=0.0 || NoH<=0.0 || VoH<=0.0) return 0.0;
    vec3 X,Y;float ax,ay;vhSpecFrame(N,T,X,Y,ax,ay);
    float G=1.0/(1.0+vhAnisoLambda(N,X,Y,L,ax,ay)+vhAnisoLambda(N,X,Y,V,ax,ay));
    float F=vhSchlickScalar(VoH,vhFiberF0());
    return F*G*VoH/max(NoV*NoH,1e-5);
}

#endif
