#version 460
#include "hair_render_common.glsl"
layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;

float wrappedDx(float x,float c){
    float d=abs(x-c);
    return min(d,1.0-d);
}

float blob(vec2 uv, vec4 p){
    if(p.w<=0.0 || p.z<=1e-5) return 0.0;
    vec2 d=vec2(wrappedDx(fract(uv.x),fract(p.x)), uv.y-p.y);
    float r2=dot(d,d)/(p.z*p.z);
    return exp(-3.5*r2)*p.w;
}

void main(){
    vec2 uv=vUv;
    float y=clamp(uv.y,0.0,1.0);
    vec3 col=mix(matl.skyBottomColor.rgb,matl.skyHorizonColor.rgb,smoothstep(0.0,0.52,y));
    col=mix(col,matl.skyTopColor.rgb,smoothstep(0.38,1.0,y));
    float b0=blob(uv,matl.skyBlob0);
    float b1=blob(uv,matl.skyBlob1);
    col += b0*matl.skyBlob0Color.rgb + b1*matl.skyBlob1Color.rgb;
    float edge=distance(uv,vec2(0.5,0.46));
    float vignette=1.0-clamp(edge*matl.skyParams.x,0.0,0.55);
    col*=mix(0.92,1.0,vignette);
    outColor=vec4(max(col,vec3(0.0)),1.0);
}
