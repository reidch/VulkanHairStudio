#version 460
layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0) uniform sampler2D sceneHdr;
layout(push_constant) uniform Post { vec4 params; } post; // x exposure EV
vec3 acesFitted(vec3 x){
    const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}
void main(){
    vec3 hdr=max(texture(sceneHdr,vUv).rgb,vec3(0.0))*exp2(post.params.x);
    outColor=vec4(acesFitted(hdr),1.0);
}
