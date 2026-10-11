#!/usr/bin/env python3
"""Validate hairstyle presets against the hierarchical groom schema."""
from pathlib import Path
import json, math, sys
ROOT=Path(__file__).resolve().parents[1]/'assets'/'grooms'

def norm(v):
    l=math.sqrt(sum(x*x for x in v)); return [x/l for x in v]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def score(root,d):
    c=norm(root['center']); ir=root.get('inv_radius',[2,2,2]); q=[(d[i]-c[i])*ir[i] for i in range(3)]
    return root.get('bias',0)-dot(q,q)
def sphere(n=4096):
    phi=(1+5**.5)/2
    for i in range(n):
        y=1-2*(i+.5)/n; r=math.sqrt(max(0,1-y*y)); a=2*math.pi*i/phi
        yield [r*math.cos(a),y,r*math.sin(a)]

def level_ok(q):
    return (q['cells_u']>=1 and q['cells_v']>=1 and 0<=q['site_jitter']<.5 and
            all(0<=q[k]<1 for k in ['root_tightness','mid_tightness','tip_tightness']) and
            0<=q['crossover']<.5 and q['drift']>=0 and q['wave_amplitude']>=0 and q['wave_cycles']>=0)

bad=0
files=sorted(ROOT.glob('*.json'))
for p in files:
    try:
        j=json.loads(p.read_text()); assert j.get('schema')==4
        g=j['graph']; keys=g['keys']; curves=g['curves']; branches=g['branches']
        assert keys and curves and branches
        for c in curves: assert 2<=len(c['keys'])<=14 and all(0<=k<len(keys) for k in c['keys'])
        warps=0
        for b in branches:
            assert 0<=b['curve']<len(curves) and curves[b['curve']].get('topology',0)==b.get('topology',0)
            q=b['clumps']; assert level_ok(q['primary']) and level_ok(q['secondary'])
            assert q['tip_spread']>=0
            c=b['cut_surface']; controls=c['controls'];
            assert len(controls)==4 and all(len(row)==4 for row in controls)
            flat=[x for row in controls for x in row]
            assert all(.1<x<=1.2 for x in flat) and .1<c['min_length']<=1 and c['breakup']>=0
            for h in b.get('style_warps',[]):
                assert min(h['radius'])>0 and h['weight']>=0 and h['length_scale']>0; warps+=1
            assert 'bundle' not in b and 'cut' not in b and 'shape_lobes' not in b
        fields=j.get('fields',{})
        allowed={'length','density','volume','texture_amplitude','texture_frequency','detail'}
        assert set(fields).issubset(allowed), f'unsupported field keys {set(fields)-allowed}'
        pop=[0]*len(branches)
        for d in sphere(): pop[max(range(len(branches)),key=lambda k:score(branches[k]['root'],d))]+=1
        assert min(pop)>0, f'dead branch {pop}'
        print(f"[PASS] {j['slug']:<24} branches={len(branches):2d} warps={warps:2d} rootPop=[{min(pop)},{max(pop)}]")
    except Exception as e:
        bad+=1; print('[FAIL]',p.name,e)
print(f'Groom asset validation: {len(files)-bad}/{len(files)} passed')
sys.exit(1 if bad else 0)
