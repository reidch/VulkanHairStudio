# Directional Fiber Reflection

This note records the optical model and the numerical integration used by the current renderer. The [technical report](TECHNICAL_REPORT.md) presents the full pipeline, from procedural modeling to simulation and display.

## Visibility and the radiance equation

The mesh shader expands deformed strands to their physical ribbon width. Each covered MSAA sample stores world position, tangent, root-to-tip parameter, and a cross-fiber coordinate. `hair_compose.frag` reconstructs the visible cylindrical normal from the stored tangent and cross-section coordinate and evaluates the incident-light response.

For a visible surface point, the reflected radiance is approximated by

$$
L_o(\omega_o)=\int_{\Omega^+}
    f_r(\omega_i,\omega_o) L_i(\omega_i)
    V(\omega_i) (\mathbf n\cdot\omega_i)^+\,d\omega_i.
$$

Visibility and shading are separate passes. This lets physical fiber diameter determine subpixel coverage while the optical model determines the radiance at each covered sample.

## Surface reflection

The current material uses a pigmented body term and a dielectric specular term:

$$
f_r=f_{\mathrm{body}}+f_{\mathrm{GGX}}, \qquad
f_{\mathrm{body}}=\frac{\mathbf C_{\mathrm{body}}}{\pi}
(1-F_i)(1-F_o).
$$

`hair_bsdf.glsl` derives body appearance from either editable custom RGB or the natural-pigment absorption coefficients. `hair_surface_brdf.glsl` provides a reconstructed cylindrical surface frame, Schlick Fresnel factors, anisotropic GGX evaluation, cuticle-tilted highlight orientation, and GGX half-vector sampling. The two roughness controls act along and across the fiber.

The sampled GGX estimator uses the half-vector distribution

$$
p_H(H)=D(H)(N\cdot H),\qquad
p_L(L)=\frac{p_H(H)}{4|V\cdot H|}.
$$

With a matching importance-sampling weight, narrow highlights can be estimated more reliably than with a few uniformly sampled incoming directions.

## Lighting integration

**Area lights.** Primary and secondary lights are finite angular caps. Their broad body reflection is integrated using directions within the cap. Specular reflection is sampled from GGX and accumulated when the corresponding incoming direction falls inside the source. The direct-light intensity corresponds to the source's integrated incident radiance, so the specular evaluation divides by the cap's solid angle, $\Omega=2\pi(1-\cos r)$.

**Studio environment.** Four spherical-Gaussian lobes describe incoming light as a smooth directional distribution:

$$
L_{\mathrm{env}}(\omega)=E\frac{\sum_k w_k\mathbf C_k K
\exp[\kappa(\omega\cdot\mathbf d_k-1)]}{\sum_k w_k},
\qquad K=\frac{2\kappa}{1-\exp(-2\kappa)}.
$$

The current shader uses a fixed $\kappa=2.7$. Cosine-weighted hemisphere samples integrate the broad body term, while GGX importance samples integrate the specular term. The quality settings select 6, 12, or 20 specular samples, with a separately capped body-sampling budget. Directional environment lighting retains the response to local fiber orientation and view direction.

## Shadows and internal paths

Two guide-derived Deep Opacity Maps estimate the optical depth toward the direct lights. Their transmittance is $T=\exp(-\tau)$. A separate 64³ extinction grid, built from guide-segment support and cumulative sweeps, supplies approximate visibility for secondary optical paths.

Internal TT/TRT-style scattering uses the existing fiber absorption coefficients and published longitudinal/azimuthal approximations. The final integration controls its contribution with

$$
L_{\mathrm{fiber}}=(1-w)(L_{\mathrm{body}}+L_{\mathrm{GGX}})
+wL_{\mathrm{internal}},\qquad 0\leq w\leq0.35.
$$

The default share is $w=0.12$. The current coarse optical grid is used for the secondary internal contribution; environment surface reflection is evaluated from the studio illumination distribution. Direct surface reflection uses its corresponding Deep Opacity Map.

## Shader map

| Source | Responsibility |
|---|---|
| `shaders/hair.task`, `hair.mesh`, `hair.frag` | Dense strand reconstruction and per-sample visibility |
| `shaders/hair_surface_brdf.glsl` | Surface frame, Fresnel, pigmented body, GGX, importance sampling |
| `shaders/hair_bsdf.glsl` | Color-to-absorption and internal fiber scattering |
| `shaders/hair_compose.frag` | Direct and environment integration; MSAA composition |
| `shaders/dom_splat.comp`, `dom_prefix.comp` | Directional optical-depth layers |
| `shaders/optical_splat.comp`, `optical_depth.comp` | Coarse volume extinction and sweep |
| `shaders/tonemap.frag` | Exposure and display transform |

## Evaluation

A useful matched scene compares the final image with the **material reflectance**, **direct body**, **environment body**, **direct specular**, **environment specular**, **direct shadow visibility**, and **internal transport** diagnostics. Hold camera, grooming parameters, exposure, and material color fixed while rotating the key light; then repeat for two substantially different grooms and colors. Record GPU stage times at each quality setting.

Source and CPU numerical checks can be run with `python verify_project.py` and `python tools/test_v47_transport.py`. Shader compilation and image-level evaluation require a Vulkan-compatible GPU and the Vulkan SDK. The environment here uses analytic studio lobes, and the current light-transport approximation leaves strand-to-strand multiple scattering as a future extension.
