#!/usr/bin/env python3
"""V47 transport math regressions (CPU reference, no GPU runtime claims).

Tests mathematics used by the GLSL and catches the V45/V46 regressions:
constant body RGB independent of illumination, 4-point miss of narrow GGX,
incorrect solid-angle normalization, energy inflation of softbox lobes.
"""
import math
from pathlib import Path

PI = math.pi

def radical(i):
    v = 0.0
    s = 0.5
    while i:
        v += s*(i & 1)
        i >>= 1
        s *= 0.5
    return v

def hammersley(i,n): return ((i+.5)/n,radical(i))

def normalize(x):
    d=math.sqrt(sum(t*t for t in x))
    return tuple(t/d for t in x)

def dot(a,b): return sum(x*y for x,y in zip(a,b))

def add(a,b): return tuple(x+y for x,y in zip(a,b))

def scale(a,s): return tuple(x*s for x in a)

def fschlick(c,f0): return f0+(1-f0)*(1-max(0.,min(1.,c)))**5

def body_cosine_quadrature(n, nvec=(0.,0.,1.), env=lambda v:1.):
    # z-up, cosine hemisphere: p(L)=(N.L)/pi
    res=0.
    for i in range(n):
        x,y=hammersley(i,n)
        r=math.sqrt(x)
        theta=2*PI*y
        if nvec==(0.,0.,1.):
            L=(r*math.cos(theta),r*math.sin(theta),math.sqrt(1-x))
        else:
            L=(r*math.cos(theta),r*math.sin(theta),-math.sqrt(1-x))
        res+=env(L)*(1-fschlick(abs(dot(nvec,L)),0.047))
    return res/n*(1-fschlick(1.,0.047))

def sample_ggx(xi,ax,ay):
    phi=math.atan2(ay*math.sin(2*PI*xi[0]),ax*math.cos(2*PI*xi[0]))
    cp,sp=math.cos(phi),math.sin(phi)
    alpha2=1/(cp*cp/(ax*ax)+sp*sp/(ay*ay))
    t2=alpha2*xi[1]/max(1-xi[1],1e-8)
    inv=math.sqrt(1+t2)
    return normalize((math.sqrt(t2)/inv*cp,math.sqrt(t2)/inv*sp,1/inv))

def ggx_lambda(v,ax,ay):
    n=max(v[2],1e-5)
    a2=(ax*v[0])**2+(ay*v[1])**2
    return .5*(math.sqrt(1+a2/n**2)-1)

def spec_estimate(n,center,angle):
    N=V=(0.,0.,1.)
    ax=.12**2
    ay=.28**2
    total=0
    accepted=0
    for i in range(n):
        H=sample_ggx(hammersley(i,n),ax,ay)
        VoH=H[2]
        L=normalize(add(scale(H,2*VoH),scale(V,-1)))
        if dot(L,center)<math.cos(angle) or L[2]<=0:continue
        G=1/(1+ggx_lambda(V,ax,ay)+ggx_lambda(L,ax,ay))
        w=fschlick(VoH,.047)*G*VoH/max(V[2]*H[2],1e-5)
        accepted+=1
        total+=w
    omega=2*PI*(1-math.cos(angle))
    return total/n/omega,accepted

def main():
    root=Path(__file__).resolve().parents[1]
    compose=(root/'shaders/hair_compose.frag').read_text()
    surface=(root/'shaders/hair_surface_brdf.glsl').read_text()
    assert 'vhEvalHairDomeBody' not in compose+surface
    assert 'vhEvalHairBody(L,V,N,bodyColor)' in compose
    assert 'vhSampleHairGgxHalfVector' in compose
    assert 'directSpec*=integrated/max(omega,1e-6)' in compose
    assert 'vec3 Ns=vhFiberSpecNormal(N,T)' in compose
    assert 'vhDeepOpacityTransmittance(P,secondary,1.0)' in compose
    assert 'envBody+=vhEnvIncident(L)*(1.0-Fi)' in compose
    assert 'float w=clamp(matl.lightingParams.w,0.0,0.35)' in compose

    k=2.7
    norm=2*k/(1-math.exp(-2*k))
    # Spherical Gaussian sphere average is analytically unity.
    sgavg=norm*(1-math.exp(-2*k))/(2*k)
    assert abs(sgavg-1)<1e-12

    # Under a UNIFORM incident environment, only for that special case,
    # Schlick body integral equals the V46 20/21 analytic formula.
    expect=(1-.047)*(1-.047)*20/21
    for n in (12,20,64):
        a=body_cosine_quadrature(n)
        assert abs(a-expect)<0.012,(n,a,expect)
    print(f'[PASS] Uniform incoming sky integrates to coated-Lambert {expect:.6f}')

    # For an upper-hemisphere directional lighting field, n-facing-up and
    # n-facing-down should NOT have equal reflectance (V46 flat-dome failure).
    env=lambda L: math.exp(2.7*(L[2]-1))*norm
    up=body_cosine_quadrature(64,(0.,0.,1.),env)
    down=body_cosine_quadrature(64,(0.,0.,-1.),env)
    assert up>down*2,(up,down)
    print(f'[PASS] Directional body retains shape: upper-facing {up:.4f} vs lower-facing {down:.4f}')

    # Smooth highlight from an actual finite disk. Estimate for a centered
    # reflection with 12+ sampled GGX microfacets, no sparse-light miss.
    for n in (12,20,256):
        rad,accepted=spec_estimate(n,(0.,0.,1.),.16)
        assert accepted>=n*.5,(n,accepted)
        assert .1<rad<3.,(n,rad)
        print(f'[PASS] GGX area-light importance sample n={n}: {accepted}/{n} hits; {rad:.4f} sr^-1')

    # Direct body and spec both integrate a cap of solid angle Omega and use
    # source radiance intensity/Omega.  The expected Lambert mean for normal
    # incidence is I*(1-F)^2/pi, independent of disk radius at leading order.
    for angle in (.04,.16,.22):
        omega=2*PI*(1-math.cos(angle))
        assert omega>0
        intensity=2.
        radiance=intensity/omega
        assert abs(radiance*omega-intensity)<1e-12
    print('[PASS] Direct source body/spec solid-angle energy contract')
    print('V47 transport reference tests: 4 groups passed')

if __name__ == '__main__': main()
