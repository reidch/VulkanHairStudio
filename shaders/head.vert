#version 460
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;

layout(set=0,binding=0,std140) uniform Frame {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 frustumPlanes[6];
    vec4 screenParams;
} frame;

layout(location=0) out vec3 vWorldPos;
layout(location=1) out vec3 vNormal;

void main(){
    vWorldPos=inPosition;
    vNormal=normalize(inNormal);
    gl_Position=frame.viewProj*vec4(inPosition,1.0);
}
