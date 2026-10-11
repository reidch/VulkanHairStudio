#ifndef VH_HAIR_OPTICAL_SAMPLING_GLSL
#define VH_HAIR_OPTICAL_SAMPLING_GLSL

layout(set=3,binding=2,std430) readonly buffer OpticalDepth { float opticalTau[]; };

int vhOpticalIndex(ivec3 c,int dim,int layer){
    return layer*dim*dim*dim+(c.z*dim+c.y)*dim+c.x;
}

float vhOpticalDepthAt(vec3 p,int layer){
    int dim=max(int(matl.renderParams.x+0.5),2);
    int dirCount=clamp(int(matl.renderParams.z+0.5),1,5);
    layer=clamp(layer,0,dirCount-1);
    float cell=max(matl.opticalGrid.w,1e-5);
    vec3 g=(p-matl.opticalGrid.xyz)/cell-vec3(0.5);
    if(any(lessThan(g,vec3(-0.5)))||any(greaterThan(g,vec3(float(dim)-0.5)))) return 0.0;
    ivec3 b=clamp(ivec3(floor(g)),ivec3(0),ivec3(dim-2));
    vec3 f=clamp(g-vec3(b),vec3(0.0),vec3(1.0));
    float v=0.0;
    for(int dz=0;dz<=1;++dz) for(int dy=0;dy<=1;++dy) for(int dx=0;dx<=1;++dx){
        ivec3 c=b+ivec3(dx,dy,dz);
        float wx=dx!=0?f.x:1.0-f.x;
        float wy=dy!=0?f.y:1.0-f.y;
        float wz=dz!=0?f.z:1.0-f.z;
        v+=opticalTau[vhOpticalIndex(c,dim,layer)]*wx*wy*wz;
    }
    return max(v,0.0);
}

float vhHairVolumeTransmittance(vec3 p,int layer){
    // Beer-Lambert attenuation through the strand-density optical field.
    return exp(-vhOpticalDepthAt(p,layer));
}

#endif
