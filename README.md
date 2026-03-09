# Mobius star tracker reference implementation

This repository accompanies the paper and presentation
'From Quaternions to Mobius Transformations: A Relativistic Framework for
Spacecraft Star Tracking'. It provides compact MATLAB and Python reference
implementations of the main constructions used in the work:

- stereographic projection between unit sight lines and the complex plane
- Mobius representations of rotations and Lorentz boosts
- cross-ratio invariants and canonicalization on the Riemann sphere
- relativistic analogs of TRIAD, Davenport's Q-method, and QUEST
- a bridge between classical quaternion kinematics and a single Mobius map

The code is organized as a readable reference library. The top-level tests run
end-to-end checks in MATLAB and Python using the source trees in `matlab/` and
`python/`.

## Repository layout

- `test.m` - MATLAB self-test run from the repository root
- `test.py` - Python self-test run from the repository root
- `matlab/` - MATLAB reference functions
- `python/` - Python reference library and experiment script

## Quick start

### MATLAB

From the repository root:

```matlab
run('test.m')
```

`test.m` adds `matlab/` to the MATLAB path relative to its own location, so it
can be run directly from the repository root.

### Python

From the repository root:

```bash
python3 test.py
```

`test.py` adds `python/` to `sys.path` relative to its own location before
importing the reference library. The Python code depends on NumPy. The
experiment script may also use SciPy and Matplotlib.

## What the tests do

Both tests run deterministic checks for:

- quaternion and rotation utilities
- Mobius application, inversion, normalization, and factorization
- stereographic projection and pixel-to-stereographic Jacobians
- cross-ratio evaluation, Jacobians, and canonicalization
- M-TRIAD, Davenport M-method, and MOBEST
- the bridge between `(q, v, s)` and a single Mobius transform

The MATLAB test also performs the symbolic nullspace check for the analytic
M-TRIAD formula when the Symbolic Math Toolbox is available.

## Conceptual map

The implementation follows the same classical-to-relativistic correspondence
used in the paper:

- quaternions -> `q2mobius`, `mobius2q_norm`
- interstar angles -> `crossratio`, `canonicalize_crossratio`
- TRIAD -> `m_triad`, `m_triad_formula`
- Davenport Q-method -> `davenport_m_method`
- QUEST -> `mobest`
- quaternion plus boost parameters -> `vqs2mobius`, `mobius2vqs`

## Core conventions

- Quaternion convention is `[qx; qy; qz; qw]` in MATLAB and `[qx, qy, qz, qw]`
  in Python.
- Mobius matrices are `2 x 2` complex matrices used up to projective scale.
  Helper routines normalize representatives to determinant `1`.
- The cross-ratio is order dependent. Canonicalize it before building a lookup
  table or comparing signatures across permutations.
- Packed no-complex helper routines use interleaved real pairs of the form
  `[x1 y1 x2 y2 ...]`.

## Function guide

### Test entry points

- `test.m` - MATLAB end-to-end self-test for the reference implementation.
- `test.py` - Python end-to-end self-test for the reference implementation.

### Quaternion and rotation utilities

- `a2q.m` - converts a direction cosine matrix to a quaternion.
- `q2a.m` - converts a quaternion to a direction cosine matrix.
- `q2mobius.m` - lifts a quaternion rotation to a unitary Mobius matrix.
- `mobius2q_norm.m` - recovers a normalized quaternion from a unitary Mobius matrix.
- `q2mrp.m` - converts a quaternion to modified Rodrigues parameters.
- `qinv.m` - quaternion inverse.
- `qinv2.m` - compatibility alias for `qinv`.
- `quatmult.m` - Hamilton product of two quaternions.
- `quatmult2.m` - compatibility alias for `quatmult`.
- `quatmat.m` - quaternion left-multiplication matrix.
- `q_max.m` - returns the candidate row with the larger score.
- `normalize_q.m` - selects and normalizes the best quaternion candidate.
- `sqrt_q.m` - principal square root of a unit quaternion.
- `skew.m` - 3 x 3 cross-product matrix.

### Generic normalization and small helpers

- `normalize.m` - legacy alias for Euclidean vector normalization.
- `normalize_vec.m` - divides a vector by its Euclidean norm.
- `normalize_m.m` - scales a 2 x 2 complex matrix to determinant 1.
- `sign_nz.m` - sign function with `sign(0) = 1`.

### Mobius, boost, and factorization routines

- `mobius_apply.m` - evaluates `z -> (a z + b)/(c z + d)`.
- `mobius_inv.m` - inverse Mobius matrix up to projective scale.
- `s2mobius.m` - axis-alignment rotation sending the stereographic origin to `s`.
- `division2mobius.m` - Mobius matrix for scalar division in stereographic coordinates.
- `lorentz_z.m` - pure boost along the `+z` axis.
- `lorentz_z_inv.m` - recovers boost speed from a pure `z`-boost matrix.
- `lorentz_z_vec.m` - applies a `z`-boost to a 4-vector.
- `vqs2mobius.m` - builds one Mobius matrix from speed, quaternion, and boost axis.
- `mobius2vqs.m` - factors a Mobius matrix back into speed, quaternion, and boost axis.
- `svd2b.m` - 2 x 2 complex factorization helper used by `mobius2vqs`.

### Stereographic and camera-model routines

- `stereoproj.m` - stereographic projection from a unit vector to the complex plane.
- `stereoprojinv.m` - inverse stereographic projection back to a unit vector.
- `stereoproj2.m` - scalar stereographic map from image-plane radius and focal length.
- `uvf_to_stereo.m` - maps pixel coordinates and focal length to stereographic coordinates.
- `uvf_jacobian.m` - analytic Jacobian of `uvf_to_stereo`.
- `stereoproj_nocomplex.m` - packed real-pair stereographic projection.
- `stereoprojinv_nocomplex.m` - inverse packed real-pair stereographic projection.

### Cross-ratio, ordering, and sensitivity routines

- `crossratio.m` - ordered cross-ratio of four complex points.
- `crossratio_orbit.m` - the six anharmonic values generated by permutations.
- `canonicalize_crossratio.m` - maps a cross-ratio to the canonical lens domain and returns nearby branches.
- `crossratio_jacobian.m` - holomorphic partial derivatives of the cross-ratio.
- `crossratio_uvf_jacobian.m` - cross-ratio Jacobian with respect to pixel coordinates and focal length.
- `crossratio_nocomplex.m` - packed real-pair cross-ratio.
- `crossratio_jacobian_nocomplex.m` - real 2 x 8 Jacobian of packed cross-ratio coordinates.
- `crossratio_jacobian_det_nocomplex.m` - determinant-based packed sensitivity quantity.
- `crossratio_jacobian_score_nocomplex.m` - sensitivity score for all six standard orderings.
- `crossratio_jacobian_score_nocomplex2.m` - sensitivity and cross-ratio size for all six orderings.
- `constellation_permute.m` - applies one of the six standard four-point orderings.
- `constellation_permutation_idx.m` - chooses the preferred canonical permutation index.
- `constellation_normalized_sensitivity.m` - equal-noise cross-ratio sensitivity score.

### Estimators

- `m_triad.m` - exact three-point Mobius fit from the nullspace of a 3 x 4 system.
- `m_triad_formula.m` - closed-form analytic version of the three-point fit.
- `davenport_m_method.m` - weighted least-squares Mobius fit for three or more correspondences.
- `mobest.m` - inverse-iteration solver for the same eigenproblem solved by the M-method.
- `adjlower4.m` - adjugate of a 4 x 4 lower-triangular matrix used by low-level estimator algebra.

### Packed complex helper routines

These functions implement complex arithmetic on interleaved real columns of the
form `[x1 y1 x2 y2 ...]`.

- `csplit.m` - splits interleaved real and imaginary columns.
- `cjoin.m` - interleaves real and imaginary columns.
- `cmult.m` - multiplies packed complex pairs column-wise.
- `cinv.m` - reciprocal of packed complex pairs.

### Python files

- `python/core.py` - Python reference implementation of the main Mobius, stereographic, cross-ratio, estimator, and bridge routines.
- `python/exper_results_mobius_startracker.py` - experiment script for the numerical study and plots described in the paper.
