# exper_results_mobius_startracker.py
# Python 3.10+
# Dependencies: numpy, scipy
# Optional (for plots): matplotlib
# This script now includes:
# - Cross-ratio error propagation under pixel and focal-length perturbations
# - Interstar-angle error propagation for comparison
# - A bijection between (quaternion, velocity, direction) and SL(2,C) Möbius transforms

import numpy as np
from numpy.linalg import norm, svd, inv
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
rng = np.random.RandomState(42)

# ============================
# High-level experiment toggles
# ============================
PLOT = True


# ----------------------------
# Complex helpers (MATLAB ports)
# ----------------------------
def normalize_m(M: np.ndarray) -> np.ndarray:
    # scale-invariant PSL(2,C): det -> 1 (up to complex sign)
    d = np.linalg.det(M)
    if d == 0:
        raise ValueError("Singular M")
    return M / np.sqrt(d)

def mobius(M: np.ndarray, z: np.ndarray) -> np.ndarray:
    # (a z + b) / (c z + d), elementwise on z
    a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
    return (a*z + b) / (c*z + d)

def s2mobius(s: complex) -> np.ndarray:
    # rotate 0 to s (normalize after)
    M = np.array([[1, s], [-np.conjugate(s), 1]], dtype=np.complex128)
    return normalize_m(M)
    
def s2mobius_inv(s: complex) -> np.ndarray:
    # inverse of s2mobius(s) up to scale is s2mobius(-s)
    return s2mobius(-s)

def division2mobius(s: complex) -> np.ndarray:
    # divide by s
    return np.array([[1, 0], [0, s]], dtype=np.complex128)

def lorentz_z(v: float) -> np.ndarray:
    # MATLAB: normalize_m([sqrt(1-v) 0; 0 sqrt(1+v)])
    M = np.array([[np.sqrt(1.0 - v), 0.0],
                  [0.0, np.sqrt(1.0 + v)]], dtype=np.complex128)
    return normalize_m(M)

# ----------------------------
# Quaternion <-> rotation and to Möbius (MATLAB ports)
# ----------------------------
def q2a(q: np.ndarray) -> np.ndarray:
    # rotation matrix from quaternion [qx,qy,qz,qw] (unnormalized ok)
    qx, qy, qz, qw = q
    A = np.array([
        [ qx*qx - qy*qy - qz*qz + qw*qw, 2*(qx*qy + qz*qw),          2*(qx*qz - qy*qw)],
        [ 2*(qx*qy - qz*qw),             -qx*qx + qy*qy - qz*qz + qw*qw, 2*(qy*qz + qx*qw)],
        [ 2*(qx*qz + qy*qw),             2*(qy*qz - qx*qw),             -qx*qx - qy*qy + qz*qz + qw*qw]
    ], dtype=np.float64)
    # Not strictly normalized; callers typically normalize q themselves
    return A

def q2mobius(q: np.ndarray) -> np.ndarray:
    # MATLAB q2mobius.m (note: matches Visual Complex Analysis mapping, up to scaling which we re-normalize)
    qx, qy, qz, qw = q
    a = qw - 1j*qz
    b = -qy + 1j*qx
    M = np.array([[a, b], [-np.conjugate(b), np.conjugate(a)]], dtype=np.complex128)
    return normalize_m(M)

def random_unit_quaternion(rng) -> np.ndarray:
    u1, u2, u3 = rng.rand(3)
    q = np.array([
        np.sqrt(1-u1)*np.sin(2*np.pi*u2),
        np.sqrt(1-u1)*np.cos(2*np.pi*u2),
        np.sqrt(u1)*np.sin(2*np.pi*u3),
        np.sqrt(u1)*np.cos(2*np.pi*u3),
    ])
    return q / norm(q)

# ----------------------------
# Stereographic projection (MATLAB ports)
# ----------------------------
def stereoproj(s: np.ndarray, zsign: int = 1) -> np.ndarray:
    # s: shape (3,N), unit vectors
    # MATLAB: (sx + i zsign sy) / (sz + zsign*||s||). For unit s, ||s||=1
    sx, sy, sz = s
    v = zsign * np.sqrt(sx*sx + sy*sy + sz*sz)
    return (sx + 1j*zsign*sy) / (sz + v)

def stereoprojinv(c: np.ndarray, zsign: int = 1) -> np.ndarray:
    # Return 3xN unit vectors on S^2
    x = 0.5 * zsign * (c + np.conjugate(c))
    y = (c - np.conjugate(c)) / (2j)
    z = zsign * np.ones_like(c)
    s = np.vstack([x, y, z])
    s = 2*s / np.sum(s*s, axis=0) - np.array([[0],[0],[1]])
    return s.real

# ----------------------------
# Cross-ratio and canonicalization (MATLAB + paper Sec. 6)
# ----------------------------

def anharmonic_orbit(z):
    return [
        z,
        1 - z,
        1 / z,
        z / (z - 1),
        1 - 1 / z,
        1 / (1 - z),
    ]

def canonicalize_crossratio(z, tol=1e-8):
    """
    Canonicalize cross-ratio to the fundamental domain F using the corrected
    three-step rule (order matters):
      1) if Re(z) < 1/2, apply z <- 1 - z
      2) if |z| > 1,    apply z <- 1 / z
      3) if Im(z) < 0,  apply z <- 1 - z
    Returns (z_canonical, branches, logs) for compatibility.
    """
    logs = []
    # 1) push to the right half of the lens
    if np.real(z) < 0.5:
        z = 1 - z
        logs.append("1-z")
    # 2) shrink inside the unit circle
    if np.abs(z) > 1:
        z = 1 / z
        logs.append("inv")
    # 3) ensure nonnegative imaginary part
    if np.imag(z) < 0:
        z = 1 - z
        logs.append("1-z")

    # Optional branching near borders (kept for downstream code)
    branches = [z]
    on_unit = abs(abs(z) - 1) < tol
    on_one  = abs(abs(1 - z) - 1) < tol
    if on_unit:
        branches.append(1 / branches[0])
    if on_one:
        branches.append(1 - branches[0])
    return z, branches, logs

# ----------------------------
# Three-point analytic solver (M-TRIAD) per MATLAB mobiusmadness2.m
# ----------------------------
def mtriad_from_pairs(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Inputs:
      z, w: complex arrays of length 3 (non-collinear on the sphere)
    Returns:
      M in SL(2,C) such that M·z_k = w_k exactly (up to scaling -> normalize_m).
    """
    # MATLAB mobiusmadness2: build 4-column matrix for nullspace
    # eq = [b, 1, -b.*r, -r]; M = null(eq'*eq)
    b = z.astype(np.complex128).reshape(-1,1)
    r = w.astype(np.complex128).reshape(-1,1)
    eq = np.hstack([b, np.ones_like(b), -b*r, -r])
    A = eq.conj().T @ eq
    # Smallest eigenvector of A
    vals, vecs = np.linalg.eigh(A)
    v = vecs[:, np.argmin(vals)]
    M = np.array([[v[0], v[1]],[v[2], v[3]]], dtype=np.complex128)
    return normalize_m(M)

# ----------------------------
# 4+ point nonlinear refinement for M (paper residual, MATLAB-consistent)
# ----------------------------
def refine_m_least_squares(z: np.ndarray,
                           w: np.ndarray,
                           M0: np.ndarray,
                           max_nfev: int = 200) -> np.ndarray:
    """
    Minimize sum_i | w_i - (a z_i + b)/(c z_i + d) |^2 over complex a,b,c,d,
    with scale fixed by normalizing det(M)=1 at each evaluation.
    Inputs:
      z, w: complex 1D arrays of length >= 4
      M0: initial 2x2 complex matrix (e.g., from M-TRIAD on any 3 pairs)
    """
    assert z.ndim == 1 and w.ndim == 1 and z.size == w.size and z.size >= 4
    # pack/unpack helpers for complex 2x2 to real vector length 8
    def pack(M):
        return np.array([M[0,0].real, M[0,0].imag,
                         M[0,1].real, M[0,1].imag,
                         M[1,0].real, M[1,0].imag,
                         M[1,1].real, M[1,1].imag], dtype=np.float64)
    def unpack(x):
        a = x[0] + 1j*x[1]
        b = x[2] + 1j*x[3]
        c = x[4] + 1j*x[5]
        d = x[6] + 1j*x[7]
        M = np.array([[a,b],[c,d]], dtype=np.complex128)
        return normalize_m(M)
    x0 = pack(M0)
    def fun(x):
        M = unpack(x)
        azpb = M[0,0]*z + M[0,1]
        czpd = M[1,0]*z + M[1,1]
        f = azpb / czpd
        r = w - f
        # return stacked real and imag parts as real residual vector
        return np.concatenate([r.real, r.imag])
    res = least_squares(fun, x0, method="lm", max_nfev=max_nfev)
    return unpack(res.x)

# ----------------------------
# Compose a boost+rotation Möbius transform (MATLAB vqs2mobius.m)
# ----------------------------
def vqs2mobius(v: float, q: np.ndarray, s: complex = 0+0j) -> np.ndarray:
    # M = normalize( S * L(v) * S^{-1} * Q )
    S  = s2mobius(s)
    L  = lorentz_z(v)
    Q  = q2mobius(q)
    # use explicit inverse form consistent with MATLAB: S^{-1} ~ s2mobius(-s)
    Sinv = s2mobius_inv(s)
    M = S @ L @ Sinv @ Q
    return normalize_m(M)

def vqs2mobius_vec(vqs: np.ndarray) -> np.ndarray:
    """
    MATLAB-compatible wrapper:
      vqs = [v, qx, qy, qz, qw]           (length 5)
      or vqs = [v, qx, qy, qz, qw, sx, sy] (length 7) with s = sx + i*sy
    Returns M in SL(2,C).
    """
    if vqs.size not in (5, 7):
        raise ValueError("vqs must have length 5 or 7")
    v = float(vqs[0])
    q = np.asarray(vqs[1:5], dtype=float)
    s = 0.0 + 0.0j
    if vqs.size == 7:
        s = complex(float(vqs[5]), float(vqs[6]))
    return vqs2mobius(v, q, s)

# ----------------------------
# Random points on the sphere and projection
# ----------------------------
def random_unit_vectors(n, rng):
    v = rng.normal(size=(3,n))
    v /= np.linalg.norm(v, axis=0, keepdims=True) + 1e-15
    return v


# ============================
# Pixel/Focal model and Jacobians
# ============================
# Pinhole: p=[u,v,f], ray^=p/||p||, stereographic: z=(px+i py)/(1+pz)
def uvf_to_z(u: np.ndarray, v: np.ndarray, f: float) -> np.ndarray:
    """
    Map pixel centroids (u,v) and focal length f [pixels] to stereographic z.
    u,v may be vectorized. Returns complex array z.
    """
    p = np.vstack([u, v, np.full_like(u, f, dtype=float)])
    p_hat = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-15)
    z = (p_hat[0] + 1j*p_hat[1]) / (1.0 + p_hat[2])
    return z

def uvf_jacobians(u: float, v: float, f: float):
    """
    Analytic Jacobians dz/du, dz/dv, dz/df derived from the paper's
    Eqs. (stereo_fwd)-(stereo_jac_f). Returns complex partials (scalars).
    """
    # Build once in scalar form
    px, py, pz = u, v, f
    L = np.sqrt(px*px + py*py + pz*pz)
    phx, phy, phz = px/L, py/L, pz/L
    denom = (1.0 + phz)
    # z = (phx + i phy) / (1+phz)
    # We use quotient rule on real variables then join into complex
    # Compute partials of p_hat wrt u, v, f
    def dph_dp(alpha, palpha):
        # d(palpha/L)/d alpha = (L - palpha*(alpha/L)) / L^2  but do carefully
        # More generally: d(p_i/L)/d alpha = (delta_{i,alpha}*L - p_i*(alpha/L))/L^2
        # Implement component-wise
        pass
    # Use compact vector form instead of the symbolic helper above
    p = np.array([px, py, pz], dtype=float)
    I = np.eye(3)
    L = np.linalg.norm(p)
    # d p_hat / d p = (I*L - p p^T / L) / L^2
    J_phat = (I*L - np.outer(p, p)/L) / (L**2 + 1e-15)
    # Chain to u,v,f
    dph_du = J_phat[:,0]
    dph_dv = J_phat[:,1]
    dph_df = J_phat[:,2]
    # z = (phx + i phy)/ (1+phz)
    num = phx + 1j*phy
    dnum_du = dph_du[0] + 1j*dph_du[1]
    dnum_dv = dph_dv[0] + 1j*dph_dv[1]
    dnum_df = dph_df[0] + 1j*dph_df[1]
    dden_du = dph_du[2]
    dden_dv = dph_dv[2]
    dden_df = dph_df[2]
    z = num/denom
    dz_du = (dnum_du*denom - num*dden_du) / (denom**2)
    dz_dv = (dnum_dv*denom - num*dden_dv) / (denom**2)
    dz_df = (dnum_df*denom - num*dden_df) / (denom**2)
    return z, dz_du, dz_dv, dz_df

# ============================
# Cross-ratio and Jacobians
# ============================

def crossratio(a, b, c, d):
    return ((a - c) * (b - d)) / ((b - c) * (a - d))

def _block_from_complex(g: complex) -> np.ndarray:
    """
    Build the 2x2 real block for ∂[Re,Im]/∂[x,y] from complex derivative g:
      [[Re g, -Im g],
       [Im g,  Re g]]
    This matches crossratio_jacobian_nocomplex.m's cjoin/csplit construction.
    """
    return np.array([[np.real(g), -np.imag(g)],
                     [np.imag(g),  np.real(g)]], dtype=float)

def crossratio_jacobian_nocomplex_py(a: complex, b: complex, c: complex, d: complex):
    """
    Real 2x8 Jacobian of [Re CR, Im CR] w.r.t. [ax,ay,bx,by,cx,cy,dx,dy],
    faithfully matching MATLAB crossratio_jacobian_nocomplex.m:
      ∂CR/∂a = CR * (1/(a-c) - 1/(a-d)), etc.,
    then map complex partials to 2x2 real blocks.
    """
    cr = crossratio(a,b,c,d)
    ga = cr * (1.0/(a - c) - 1.0/(a - d))
    gb = cr * (1.0/(b - d) - 1.0/(b - c))
    gc = cr * (1.0/(b - c) - 1.0/(a - c))
    gd = cr * (1.0/(a - d) - 1.0/(b - d))
    Ja = _block_from_complex(ga)
    Jb = _block_from_complex(gb)
    Jc = _block_from_complex(gc)
    Jd = _block_from_complex(gd)
    # assemble 2x8
    J = np.hstack([Ja, Jb, Jc, Jd])
    return cr, J

def crossratio_uvf_jac(u4, v4, f):
    """
    Real Jacobian of [Re(CR), Im(CR)] w.r.t. [u1,v1,u2,v2,u3,v3,u4,v4,f]
    using crossratio_jacobian_nocomplex as the reference.
    """
    # z_k and their real 2x3 Jacobians wrt (u_k, v_k, f)
    z = []
    Jz_blocks = []
    for k in range(4):
        zk, dz_du, dz_dv, dz_df = uvf_jacobians(u4[k], v4[k], f)
        z.append(zk)
        Jz_blocks.append(np.array([[np.real(dz_du), np.real(dz_dv), np.real(dz_df)],
                                   [np.imag(dz_du), np.imag(dz_dv), np.imag(dz_df)]], dtype=float))
    a,b,c,d = z
    cr, J_cr_z = crossratio_jacobian_nocomplex_py(a,b,c,d)  # 2x8 wrt [ax,ay,bx,by,cx,cy,dx,dy]
    # Build 8x9 chain matrix T mapping [u1,v1,u2,v2,u3,v3,u4,v4,f] -> [ax,ay,bx,by,cx,cy,dx,dy]
    T = np.zeros((8,9), dtype=float)
    # For each point k, rows (2k,2k+1) get the 2x3 block into columns (2k,2k+1) for u,v and col 8 for f
    for k in range(4):
        rows = slice(2*k, 2*k+2)
        # u column for this point
        T[rows, 2*k]   = Jz_blocks[k][:,0]
        # v column
        T[rows, 2*k+1] = Jz_blocks[k][:,1]
        # shared f column (last)
        T[rows, 8]     = Jz_blocks[k][:,2]
    # Chain rule
    J = J_cr_z @ T  # 2x9
    return cr, J
# ============================
# Interstar angle pipeline
# ============================
def uvf_to_unit(u: np.ndarray, v: np.ndarray, f: float) -> np.ndarray:
    p = np.vstack([u, v, np.full_like(u, f, dtype=float)])
    p_hat = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-15)
    return p_hat

def pair_angle(u1,v1,u2,v2,f):
    b = uvf_to_unit(np.array([u1,u2]), np.array([v1,v2]), f)
    c = np.clip(np.dot(b[:,0], b[:,1]), -1.0, 1.0)
    return np.arccos(c)

def angles_uvf_jac(u4, v4, f, pairs=((0,1),(0,2))):
    """
    Real Jacobian of two angles [theta_01, theta_02] w.r.t. the 9 parameters.
    Uses complex-step friendly finite differences for robustness.
    Returns angles vector (2,) and J (2x9).
    """
    th = []
    for (i,j) in pairs:
        th.append(pair_angle(u4[i],v4[i],u4[j],v4[j],f))
    th = np.array(th)
    # finite diff (safe for this scalar)
    base = np.array([u4[0],v4[0],u4[1],v4[1],u4[2],v4[2],u4[3],v4[3],f], dtype=float)
    J = np.zeros((len(th), 9))
    eps = 1e-7
    for c in range(9):
        x = base.copy(); x[c] += eps
        u2v2f = (x[0::2][:4], x[1::2][:4], x[-1])  # quick unpack helper
        u2 = np.array([x[0],x[2],x[4],x[6]])
        v2 = np.array([x[1],x[3],x[5],x[7]])
        f2 = x[8]
        th2 = []
        for (i,j) in pairs:
            th2.append(pair_angle(u2[i],v2[i],u2[j],v2[j],f2))
        th2 = np.array(th2)
        J[:,c] = (th2 - th) / eps
    return th, J

# ============================
# Möbius <-> (v,q,s) bijection
# ============================
def svd2b_complex_2x2(X: np.ndarray):
    """
    Closed-form SVD specialization for 2x2 complex with det(X)=1
    Mirrors MATLAB svd2b.m. Returns U,S,V with det(U)=det(V)=1 and S Hermitian pos.def.
    """
    X = normalize_m(X)
    P = X @ X.conj().T
    tr = np.trace(P)
    # Choose sign branch for stable eigen ordering
    sgn = 1.0 if (P[0,0]-P[1,1]).real >= 0 else -1.0
    root = np.sqrt((tr**2 - 4.0))
    Sp_sq = np.array([[ (tr-root)/2.0, 0.0],
                      [ 0.0,            (tr+root)/2.0 ]], dtype=np.complex128)
    # Build U from (P - Sp_sq) as in MATLAB
    U = sgn * (P - Sp_sq) @ np.array([[1,0],[0,-1]], dtype=np.complex128)
    U = normalize_m(U)
    Sp = np.sqrt(Sp_sq)
    S = np.linalg.inv(Sp)
    V = (Sp @ U.conj().T @ X).T
    return U, S, V

def mobius2vqs_py(M: np.ndarray):
    """
    Factor M = U S V^H with U,V unitary det=1, S Hermitian pos.def det=1.
    Then interpret S as a boost along +z, U*V^H as rotation quaternion,
    and U[0,1]/U[1,1] as stereographic location 's' of the boost axis.
    Returns (v, q[4], s complex).
    """
    U, S, V = svd2b_complex_2x2(M)
    # Rotation from U*V^H
    R = U @ V.conj().T
    q = mobius2q_norm_py(R)  # scalar-first in MATLAB alias; we keep [qx,qy,qz,qw]
    # Boost speed from S (cf. lorentz_z_inv)
    # r = |S22/S11| >= 1, then v = (r^2 - 1)/(r^2 + 1)
    r = np.abs(S[1,1]/S[0,0])
    r2 = (r*r).real
    v = (r2 - 1.0) / (r2 + 1.0)
    # Local boost direction parameter s from U entries
    s = U[0,1]/U[1,1]
    return v, q, s

def mobius2q_norm_py(M: np.ndarray) -> np.ndarray:
    """
    Python port of mobius2q_norm.m
    Returns normalized quaternion [qx,qy,qz,qw].
    """
    q = np.array([
        np.imag(M[0,1]),
        np.real(M[1,0]),
        np.imag(M[1,1]),
        np.real(M[0,0]),
    ], dtype=float)
    # normalize with positive scalar part convention
    s = np.sign(q[3]) if q[3] != 0 else 1.0
    q = q / (s*np.linalg.norm(q))
    return q
    

def mobius2vqs(M: np.ndarray) -> np.ndarray:
    """
    MATLAB-compatible wrapper. Returns vqs in the format:
      [v, qx, qy, qz, qw, sx, sy]
    """
    v, q, s = mobius2vqs_py(M)
    return np.array([v, q[0], q[1], q[2], q[3], np.real(s), np.imag(s)], dtype=float)

# ============================
# Monte-Carlo experiments
# ============================
@dataclass
class CRVsAngleConfig:
    f_nom: float = 800.0          # pixels
    sig_uv: float = 0.15          # pixel centroid 1-sigma per axis
    sig_f: float = 1.0            # pixel focal 1-sigma
    N_trials: int = 20000
    seed: int = 7

def simulate_cr_vs_angle(cfg: CRVsAngleConfig):
    """
    Draw random 4-tuples of unit directions, project to pixels with f_nom,
    perturb with N(0,sig_uv^2) on u,v and N(0,sig_f^2) on f, and measure:
      - variance of cross-ratio (Re/Im) and its canonicalized representative
      - variance of two interstar angles
      - implied search-space area in canonical lens by binning CR
    Returns a dictionary of summary metrics and raw aggregates for plotting.
    """
    rng = np.random.RandomState(cfg.seed)
    # Generate a random camera orientation so the pixel pattern varies
    q = random_unit_quaternion(rng)
    R = q2a(q)
    # Random 4 stars uniform on S^2 then rotate into camera frame
    all_stats = {
        "cr": [],
        "cr_canon": [],
        "ang": [],
    }
    # Define a fixed pixelization: use perspective with u=fx*x/z, v=fy*y/z, here f=f_nom
    for _ in range(cfg.N_trials):
        s_ref = random_unit_vectors(4, rng)  # 3x4
        s_cam = R @ s_ref
        # project to pixels using simple pinhole
        u0 = cfg.f_nom * (s_cam[0]/(s_cam[2] + 1e-12))
        v0 = cfg.f_nom * (s_cam[1]/(s_cam[2] + 1e-12))
        f0 = cfg.f_nom
        # perturbations
        du = rng.normal(0, cfg.sig_uv, size=4)
        dv = rng.normal(0, cfg.sig_uv, size=4)
        df = rng.normal(0, cfg.sig_f)
        u = u0 + du
        v = v0 + dv
        f = f0 + df
        # Cross-ratio
        z = uvf_to_z(u, v, f)
        cr = crossratio(z[0], z[1], z[2], z[3])
        cr_can, _, _ = canonicalize_crossratio(cr)
        all_stats["cr"].append(cr)
        all_stats["cr_canon"].append(cr_can)
        # Interstar angles: pick two independent pairs
        ang01 = pair_angle(u[0],v[0],u[1],v[1],f)
        ang02 = pair_angle(u[0],v[0],u[2],v[2],f)
        all_stats["ang"].append(np.array([ang01, ang02]))
    # Convert to arrays
    all_stats["cr"] = np.asarray(all_stats["cr"])
    all_stats["cr_canon"] = np.asarray(all_stats["cr_canon"])
    all_stats["ang"] = np.asarray(all_stats["ang"])
    # Summary
    out = {
        "cr_var": np.var(np.vstack([all_stats["cr"].real, all_stats["cr"].imag]), axis=1),
        "crcanon_var": np.var(np.vstack([all_stats["cr_canon"].real, all_stats["cr_canon"].imag]), axis=1),
        "ang_var": np.var(all_stats["ang"], axis=0),
        "raw": all_stats
    }
    return out

def demo_plots(summary):
    if not PLOT: 
        return
    cr = summary["raw"]["cr"]
    cr_can = summary["raw"]["cr_canon"]
    fig1 = plt.figure(figsize=(5,5))
    plt.scatter(cr.real, cr.imag, s=4, alpha=0.25)
    plt.title("Raw cross-ratio")
    plt.xlabel("Re")
    plt.ylabel("Im")
    fig2 = plt.figure(figsize=(5,5))
    plt.scatter(cr_can.real, cr_can.imag, s=4, alpha=0.25)
    plt.title("Canonicalized cross-ratio")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()

if __name__ == "__main__":
    # Example end-to-end: simulate and print headline numbers
    cfg = CRVsAngleConfig()
    summary = simulate_cr_vs_angle(cfg)
    print("Var[Re(CR)], Var[Im(CR)] =", summary["cr_var"])
    print("Var[Re(CR_canon)], Var[Im(CR_canon)] =", summary["crcanon_var"])
    print("Var[angles] (rad^2) =", summary["ang_var"])
    demo_plots(summary)
