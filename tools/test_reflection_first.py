#!/usr/bin/env python3
"""Numerical checks for the V46 analytic coated-Lambert sky model.

This checks underlying math, not GPU graphics or SPIR-V compilation.
"""
import math


def schlick(mu, f0):
    return f0 + (1.0 - f0) * (1.0 - mu) ** 5


def coated_lambert_hemisphere_integral(f0, resolution=30000):
    # 2*int_0^1 (1-Fi(mu))*mu dmu, midpoint integration
    return sum(2.0 * (1.0 - schlick((i + 0.5)/resolution, f0)) *
               (i + 0.5)/resolution for i in range(resolution)) / resolution


def body_sky(color, radiance, f0, no_v):
    return [c * l * (1.0 - schlick(no_v, f0)) *
            (1.0 - f0) * (20.0 / 21.0)
            for c, l in zip(color, radiance)]


for ior in (1.4, 1.55, 1.65):
    f0 = ((ior - 1.0)/(ior + 1.0))**2
    numeric = coated_lambert_hemisphere_integral(f0)
    analytic = (1.0 - f0) * (20.0 / 21.0)
    assert abs(numeric-analytic) < 1e-8, (ior,numeric,analytic)

# Direct shadow transmittance does not appear in analytic dome body at all:
f0=((1.55 - 1.0)/(1.55 + 1.0))**2
c0=body_sky([1.0,0.0,0.0],[0.75]*3,f0,0.8)
c1=body_sky([0.0,0.8,0.0],[0.75]*3,f0,0.8)
assert c0[0] > 0.0 and c0[1] == 0.0
assert c1[1] > 0.0 and c1[0] == 0.0
assert body_sky([1.0,0.0,0.0],[0.0]*3,f0,0.8)==[0.0]*3
assert all(0.0<=v<=1.0 for v in body_sky([1.0]*3,[1.0]*3,f0,0.8))
print('[PASS] Schlick hemisphere analytic factor 20/21 matches numeric integration')
print('[PASS] Uniform reflected sky responds to red/green material across fibers')
print('[PASS] No sky radiance creates no body emission')
