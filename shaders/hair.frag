#version 460
// V44 primary visibility pass. No lighting is evaluated here.
// Hardware MSAA sample coverage is the visibility estimator; every surviving
// sample stores the geometry needed by the later shading/composition pass.
layout(location=0) in PerVertex {
    vec3 tangent;
    float arc;
    vec3 worldPos;
    float fiberSide;
} fIn;
layout(location=0) out vec4 outGeom0; // world position.xyz, arc+1 (0 means empty)
layout(location=1) out vec4 outGeom1; // tangent.xyz, cross-fiber h
void main(){
    outGeom0=vec4(fIn.worldPos,clamp(fIn.arc,0.0,1.0)+1.0);
    outGeom1=vec4(normalize(fIn.tangent),clamp(fIn.fiberSide,-0.999,0.999));
}
