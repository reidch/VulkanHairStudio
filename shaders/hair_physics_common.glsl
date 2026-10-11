#ifndef V38_HAIR_PHYSICS_COMMON_GLSL
#define V38_HAIR_PHYSICS_COMMON_GLSL

const int HP_GRID_DIM = 48;
const uint HP_GRID_COUNT = 110592u;
const float HP_OCC_SCALE = 4096.0;
const float HP_VEL_SCALE = 256.0;

struct VolumeAccumNode {
    uint occupancy;
    int momentumX;
    int momentumY;
    int momentumZ;
};

layout(std430,set=2,binding=4) readonly buffer BodyFieldBuffer {
    float bodyDistance[];
};
layout(std430,set=2,binding=5) buffer VolumeAccumBuffer {
    VolumeAccumNode volumeAccum[];
};
layout(std430,set=2,binding=6) buffer VolumeStateBuffer {
    vec4 volumeState[]; // xyz velocity, w packing fraction
};
layout(std430,set=2,binding=7) readonly buffer GuideMetaBuffer {
    vec4 guideMeta[];   // x support radius, y represented volume
};

layout(set=2,binding=2,std140) uniform SimParams {
    mat4 headFromRest;
    mat4 restFromHead;
    vec4 dtSolver; // x dt, y stretch compliance, z bend compliance, w time
    vec4 wind;     // xyz primary direction, w air speed (m/s)
    vec4 windPositionType; // xyz source position, w 0 directional / 1 fan / 2 vortex
    vec4 windShape; // x range, y cone cosine, z gust fraction, w turbulence
    vec4 forces;   // x gravity, y free-air damping, z root direction stiffness, w global contact margin
    vec4 contact;  // x static friction, y dynamic friction, z proximity band, w normal damping
    vec4 style;    // x shape compliance, y wind release start, z wind release end, w tip freedom
    vec4 aero;     // x quadratic drag coefficient, y gust fraction, z max aero accel, w root fade exponent
    vec4 volume;   // x target packing, y pressure stiffness, z grid velocity coupling, w wind shielding
    vec4 grid;     // xyz hair-volume grid origin, w cell size
    vec4 bodyField;  // xyz rest-pose body ContactField origin, w cell size
    vec4 bodyInfo; // x ContactField dimension, y gravity support load, z/w reserved
} sim;

struct ContactHit {
    float distance;
    vec3 normal;
    vec2 friction;
};

vec3 hpSafeNormalize(vec3 v, vec3 fallback) {
    float l2=dot(v,v);
    return l2>1e-12 ? v*inversesqrt(l2) : fallback;
}

float hpHash31(vec3 p) {
    p=fract(p*0.1031);
    p+=dot(p,p.yzx+33.33);
    return fract((p.x+p.y)*p.z);
}

vec3 hpAirVelocity(vec3 pWorld, uint seed) {
    float speed=max(sim.wind.w,0.0);
    if(speed<=0.0) return vec3(0.0);
    vec3 axis=hpSafeNormalize(sim.wind.xyz,vec3(1.0,0.0,0.0));
    int windType=int(round(sim.windPositionType.w));
    float gain=1.0;
    vec3 dir=axis;
    if(windType==1) {
        vec3 r=pWorld-sim.windPositionType.xyz;
        float dist=max(length(r),1e-5);
        vec3 rd=r/dist;
        float cone=smoothstep(sim.windShape.y, min(1.0,sim.windShape.y+0.10), dot(rd,axis));
        float rangeFade=1.0-smoothstep(max(0.05,sim.windShape.x*0.72),max(0.06,sim.windShape.x),dist);
        gain=cone*rangeFade;
        dir=rd;
    } else if(windType==2) {
        vec3 r=pWorld-sim.windPositionType.xyz;
        vec3 axial=axis*dot(r,axis);
        vec3 radial=r-axial;
        float dist=max(length(radial),1e-5);
        dir=hpSafeNormalize(cross(axis,radial),axis);
        gain=1.0-smoothstep(max(0.05,sim.windShape.x*0.72),max(0.06,sim.windShape.x),dist);
    }
    float gust=1.0+sim.windShape.z*sin(sim.dtSolver.w*1.61+float(seed%127u)*0.073);
    float n=2.0*hpHash31(pWorld*5.7+vec3(sim.dtSolver.w*0.7,float(seed%17u),0.0))-1.0;
    vec3 side=hpSafeNormalize(cross(dir,vec3(0.0,1.0,0.0)),vec3(0.0,0.0,1.0));
    return speed*max(gain,0.0)*gust*hpSafeNormalize(dir+side*(sim.windShape.w*0.18*n),dir);
}

uint hpGridIndex(ivec3 c) {
    return uint(c.x + HP_GRID_DIM*(c.y + HP_GRID_DIM*c.z));
}

bool hpGridInside(ivec3 c) {
    return all(greaterThanEqual(c,ivec3(0))) && all(lessThan(c,ivec3(HP_GRID_DIM)));
}

float hpNodePacking(ivec3 c) {
    if(!hpGridInside(c)) return 0.0;
    return float(volumeAccum[hpGridIndex(c)].occupancy)/HP_OCC_SCALE;
}

float hpStatePacking(ivec3 c) {
    if(!hpGridInside(c)) return 0.0;
    return volumeState[hpGridIndex(c)].w;
}

vec4 hpSampleVolumeState(vec3 pWorld) {
    float h=max(sim.grid.w,1e-5);
    vec3 g=(pWorld-sim.grid.xyz)/h;
    ivec3 b=ivec3(floor(g));
    vec3 f=fract(g);
    if(any(lessThan(b,ivec3(0))) || any(greaterThan(b,ivec3(HP_GRID_DIM-2)))) return vec4(0.0);
    vec4 acc=vec4(0.0);
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        ivec3 o=ivec3(x,y,z);
        vec3 q=mix(vec3(1.0)-f,f,vec3(o));
        float w=q.x*q.y*q.z;
        acc+=w*volumeState[hpGridIndex(b+o)];
    }
    return acc;
}

vec4 hpAccumNodeState(ivec3 c) {
    if(!hpGridInside(c)) return vec4(0.0);
    VolumeAccumNode a=volumeAccum[hpGridIndex(c)];
    float m=float(a.occupancy);
    if(m<0.5) return vec4(0.0);
    vec3 v=vec3(float(a.momentumX),float(a.momentumY),float(a.momentumZ))/(m*HP_VEL_SCALE);
    return vec4(v,m/HP_OCC_SCALE);
}

vec4 hpSampleP2GState(vec3 pWorld) {
    float h=max(sim.grid.w,1e-5);
    vec3 g=(pWorld-sim.grid.xyz)/h;
    ivec3 b=ivec3(floor(g));
    vec3 f=fract(g);
    if(any(lessThan(b,ivec3(0))) || any(greaterThan(b,ivec3(HP_GRID_DIM-2)))) return vec4(0.0);
    vec4 acc=vec4(0.0);
    float weightSum=0.0;
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        ivec3 o=ivec3(x,y,z);
        vec3 q=mix(vec3(1.0)-f,f,vec3(o));
        float w=q.x*q.y*q.z;
        vec4 node=hpAccumNodeState(b+o);
        if(node.w>0.0) { acc+=w*node; weightSum+=w; }
    }
    return weightSum>1e-6 ? acc/weightSum : vec4(0.0);
}

float hpWindExposure(vec3 pWorld, vec3 windDirection) {
    if(sim.volume.w<=0.0 || sim.wind.w<=0.0) return 1.0;
    vec3 d=hpSafeNormalize(windDirection,vec3(1.0,0.0,0.0));
    float h=max(sim.grid.w,1e-5);
    float optical=0.0;
    for(int k=1;k<=4;++k) {
        float distanceCells=1.25+1.35*float(k);
        float rho=hpSampleVolumeState(pWorld-d*h*distanceCells).w;
        optical+=max(rho-0.08,0.0)*(0.31-0.045*float(k));
    }
    return clamp(exp(-sim.volume.w*optical),0.12,1.0);
}

float hpBodyFieldAt(ivec3 c) {
    int dim=max(int(round(sim.bodyInfo.x)),2);
    c=clamp(c,ivec3(0),ivec3(dim-1));
    uint idx=uint(c.x+dim*(c.y+dim*c.z));
    return bodyDistance[idx];
}

bool hpBodyFieldContainsRest(vec3 pRest) {
    float h=max(sim.bodyField.w,1e-5);
    vec3 g=(pRest-sim.bodyField.xyz)/h;
    int dim=max(int(round(sim.bodyInfo.x)),2);
    return all(greaterThanEqual(g,vec3(0.0))) && all(lessThanEqual(g,vec3(float(dim-1))));
}

float hpSampleBodyFieldRest(vec3 pRest) {
    float h=max(sim.bodyField.w,1e-5);
    vec3 g=(pRest-sim.bodyField.xyz)/h;
    int dim=max(int(round(sim.bodyInfo.x)),2);
    if(!hpBodyFieldContainsRest(pRest)) return 1.0e20;
    ivec3 b=min(ivec3(floor(g)),ivec3(dim-2));
    b=max(b,ivec3(0));
    vec3 f=clamp(g-vec3(b),vec3(0.0),vec3(1.0));
    float acc=0.0;
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        ivec3 o=ivec3(x,y,z);
        vec3 q=mix(vec3(1.0)-f,f,vec3(o));
        acc+=q.x*q.y*q.z*hpBodyFieldAt(b+o);
    }
    return acc;
}

vec3 hpBodyNormalRest(vec3 pRest) {
    float h=max(sim.bodyField.w,1e-5);
    vec3 g=vec3(
        hpSampleBodyFieldRest(pRest+vec3(h,0,0))-hpSampleBodyFieldRest(pRest-vec3(h,0,0)),
        hpSampleBodyFieldRest(pRest+vec3(0,h,0))-hpSampleBodyFieldRest(pRest-vec3(0,h,0)),
        hpSampleBodyFieldRest(pRest+vec3(0,0,h))-hpSampleBodyFieldRest(pRest-vec3(0,0,h)));
    return hpSafeNormalize(g,vec3(0.0,1.0,0.0));
}

ContactHit hpQueryBody(vec3 pWorld, float supportRadius) {
    vec3 p=(sim.restFromHead*vec4(pWorld,1.0)).xyz;
    ContactHit h;
    h.friction=max(sim.contact.xy,vec2(0.0));
    if(!hpBodyFieldContainsRest(p)) {
        h.distance=1.0e20;
        h.normal=vec3(0.0,1.0,0.0);
        return h;
    }
    h.distance=hpSampleBodyFieldRest(p)-max(supportRadius,0.0)-sim.forces.w;
    h.normal=hpSafeNormalize(mat3(sim.headFromRest)*hpBodyNormalRest(p),vec3(0.0,1.0,0.0));
    return h;
}

vec3 hpProjectBody(vec3 p, float supportRadius) {
    for(int i=0;i<4;++i) {
        ContactHit h=hpQueryBody(p,supportRadius);
        if(h.distance>=0.0) break;
        p-=h.distance*h.normal;
    }
    return p;
}

vec3 hpContactVelocityCoulomb(vec3 p, vec3 v, float supportRadius, float dt) {
    ContactHit h=hpQueryBody(p,supportRadius);
    if(h.distance>sim.contact.z) return v;

    float vn=dot(v,h.normal);
    float normalSpeed=max(-vn,0.0);
    if(vn<0.0) v-=h.normal*vn*clamp(sim.contact.w,0.0,1.0);

    vec3 vt=v-h.normal*dot(v,h.normal);
    float vtLen=length(vt);
    if(vtLen<1e-8) return v;

    // Velocity-level Coulomb cone.  The normal load comes from actual impact
    // plus gravity-supported contact; it is intentionally not proportional to an
    // arbitrary penetration correction, so face contact does not become glue.
    float gravityLoad=max(0.0,-sim.forces.x*h.normal.y)*dt*max(sim.bodyInfo.y,0.0);
    float normalImpulseSpeed=normalSpeed+gravityLoad;
    float staticLimit=max(h.friction.x,0.0)*normalImpulseSpeed;
    if(vtLen<=staticLimit) return v-vt;

    float dynamicDrop=min(vtLen,max(h.friction.y,0.0)*normalImpulseSpeed);
    return v-vt*(dynamicDrop/vtLen);
}

float hpStyleRelease() {
    float lo=min(sim.style.y,sim.style.z);
    float hi=max(sim.style.y,sim.style.z);
    return smoothstep(lo,hi,sim.wind.w);
}

#endif
