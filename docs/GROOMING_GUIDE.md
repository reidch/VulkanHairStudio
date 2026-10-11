# Groom Studio Guide

Author from low frequency to high frequency. A good preset should remain recognizable with fiber texture and flyaways set to zero.

## 1. Style volume

Edit the `StyleCurve` sweep first. `SweepKey.center` is the flow centerline; `axis_u/axis_v` are the finite cross-section and therefore the silhouette/volume. Do not use local warps to compensate for a bad macro volume.

## 2. Primary and secondary clumps

Primary clumps are visible locks and also define simulation neighborhoods. Use moderate mid/tip tightness and low-frequency wave to create blowout or broad curl motion. Secondary clumps add smaller breakup. `crossover` smooths Voronoi boundaries; it is not random frizz.

## 3. CutSurface

The 4x4 control lattice is a continuous normalized terminal-length surface. Use it for face-framing layers, crown/back length transitions, bob outlines and stacked layers. If a hairstyle requires many StyleWarps just to obtain its haircut, fix the CutSurface instead.

## 4. StyleWarp

Use sparsely for intentional asymmetric/editorial deformation. It may change broad local drift/lift/roll/length, but it should not become a patch system.

## 5. Fiber texture and detail

Only after silhouette/clumps/cut are correct, add residual wave/curl texture and sparse frizz/flyaways. Large waves belong to the clump levels so neighboring fibers remain coherent.

## 6. Physics

Physics can be entered whenever a groom is committed. Contact warnings remain visible. `Resolve Contacts` is an optional authoring repair, not admission control.

## 7. Performance

Normal style switching can use the derived groom cache. Interactive edits intentionally rebuild geometry asynchronously because the style has changed, but they reuse the stable guide basis whenever primary-clump topology is unchanged.
## Hairstyle selection

Use the **HAIRSTYLES** button to open the library. Search by name/slug or filter by Long & Wavy, Bob & Layered, Ponytails, Updos, or Short / Editorial. Selecting an item requests that exact catalog entry; loading remains asynchronous and uses the derived groom cache when valid. There is no N/P keyboard cycling path.


## Hair appearance and lighting

Use **Studio Light** for key intensity, environment radiance and exposure. Use **Hair Color** for Natural Pigment or unrestricted Custom Color. Material and lighting adjustments leave the procedural hairstyle geometry intact.
