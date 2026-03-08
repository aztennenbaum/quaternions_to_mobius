# Mobius star tracker MATLAB reference

This repo is a flat MATLAB reference implementation for the Mobius-domain
star tracker work. All functions live at the top level. There are no example
or legacy subfolders. Run `demo.m` to exercise the library end to end,
including the analytic M-TRIAD equivalence check.

## Quick start

```matlab
addpath(pwd);
demo
```

## Main functions

- `m_triad.m` - exact 3-point Mobius fit via `null(A)`
- `m_triad_formula.m` - closed-form 3-point formula from the notes
- `davenport_m_method.m` - weighted least-squares Mobius fit
- `mobest.m` - fast inverse-iteration solver for the same eigenproblem
- `crossratio.m` - ordered 4-point invariant
- `canonicalize_crossratio.m` - canonical representative plus ambiguity branches
- `crossratio_uvf_jacobian.m` - cross-ratio Jacobian wrt pixel coordinates
- `uvf_to_stereo.m` - pixels to stereographic coordinates
- `uvf_jacobian.m` - analytic pixel/focal Jacobian
- `vqs2mobius.m` - quaternion, speed, and axis to Mobius matrix
- `mobius2vqs.m` - inverse bridge back to classical parameters

## Repo layout

- `README.md` - usage and function summary
- `demo.m` - one script that exercises the functions and checks consistency
- `*.m` - library functions

## Conventions

- Quaternion convention is `[qx;qy;qz;qw]`.
- Mobius matrices are `2x2` complex matrices normalized to `det(M)=1` up to sign.
- The cross-ratio is order-dependent. Use `canonicalize_crossratio()` before
  building lookup tables.
- `m_triad.m` uses MATLAB's `null()` on `A = [z 1 -z.*w -w]`.
- `demo.m` checks that the closed-form notes formula spans the same nullspace
  as `m_triad()`; if the Symbolic Math Toolbox is available it also performs a
  symbolic residual check.

## Function list

Quaternion and rotation:

- `a2q`, `q2a`, `quatmult`, `quatmult2`, `quatmat`, `q2mrp`, `qinv`, `qinv2`
- `normalize`, `normalize_q`, `normalize_vec`, `q_max`, `sqrt_q`, `sign_nz`, `skew`

Mobius and Lorentz bridge:

- `q2mobius`, `mobius2q_norm`, `s2mobius`, `mobius_apply`, `mobius_inv`
- `division2mobius`, `lorentz_z`, `lorentz_z_inv`, `lorentz_z_vec`
- `vqs2mobius`, `mobius2vqs`, `svd2b`, `normalize_m`

Stereographic and camera:

- `stereoproj`, `stereoproj2`, `stereoprojinv`
- `stereoproj_nocomplex`, `stereoprojinv_nocomplex`
- `uvf_to_stereo`, `uvf_jacobian`

Cross-ratio and canonicalization:

- `crossratio`, `crossratio_orbit`, `canonicalize_crossratio`
- `crossratio_nocomplex`, `crossratio_jacobian`, `crossratio_jacobian_nocomplex`
- `crossratio_jacobian_det_nocomplex`, `crossratio_uvf_jacobian`
- `constellation_normalized_sensitivity`, `constellation_permute`
- `constellation_permutation_idx`, `crossratio_jacobian_score_nocomplex`
- `crossratio_jacobian_score_nocomplex2`

Estimators:

- `m_triad`, `m_triad_formula`, `davenport_m_method`, `mobest`, `adjlower4`


No-complex helpers use an interleaved real-pair layout:
[x1 y1 x2 y2 ...]. Functions such as csplit, cjoin,
stereoproj_nocomplex, stereoprojinv_nocomplex, and
crossratio_nocomplex preserve that packed row format.
