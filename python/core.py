"""
Mobius star tracker Python reference.

This module collects the Python counterparts of the MATLAB reference
functions used by the demos and experiment script.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _is_vector(x: np.ndarray) -> bool:
    return np.asarray(x).ndim == 1


def _as_rows(x) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x)
    was_vector = arr.ndim == 1
    if was_vector:
        arr = arr.reshape(1, -1)
    return arr, was_vector


def _restore_rows(arr: np.ndarray, was_vector: bool):
    return arr.reshape(-1) if was_vector else arr


def sign_nz(x):
    y = np.sign(x)
    if np.isscalar(y):
        return 1 if y == 0 else y
    y = np.asarray(y)
    y[y == 0] = 1
    return y


def normalize_vec(x):
    x = np.asarray(x)
    n = np.linalg.norm(x)
    if n == 0:
        raise ValueError("normalize_vec:zero")
    return x / n


def normalize(x):
    return normalize_vec(x)


def skew(v):
    v = np.asarray(v).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.asarray(v).dtype,
    )


def q2a(q):
    q = np.asarray(q).reshape(4)
    qx, qy, qz, qw = q
    A = np.array(
        [
            [qx * qx - qy * qy - qz * qz + qw * qw, 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)],
            [2 * (qx * qy - qz * qw), -qx * qx + qy * qy - qz * qz + qw * qw, 2 * (qy * qz + qx * qw)],
            [2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), -qx * qx - qy * qy + qz * qz + qw * qw],
        ],
        dtype=float,
    )
    return A / np.sum(q ** 2)


def a2q(A):
    A = np.asarray(A, dtype=float)
    tr = np.trace(A)
    if tr >= 0:
        s = 2 * math.sqrt(1 + tr)
        q = np.array(
            [
                (A[1, 2] - A[2, 1]) / s,
                (A[2, 0] - A[0, 2]) / s,
                (A[0, 1] - A[1, 0]) / s,
                s / 4,
            ],
            dtype=float,
        )
    else:
        k = int(np.argmax(np.diag(A))) + 1
        if k == 1:
            s = 2 * math.sqrt(1 + A[0, 0] - A[1, 1] - A[2, 2])
            q = np.array(
                [
                    s / 4,
                    (A[0, 1] + A[1, 0]) / s,
                    (A[0, 2] + A[2, 0]) / s,
                    (A[1, 2] - A[2, 1]) / s,
                ],
                dtype=float,
            )
        elif k == 2:
            s = 2 * math.sqrt(1 - A[0, 0] + A[1, 1] - A[2, 2])
            q = np.array(
                [
                    (A[0, 1] + A[1, 0]) / s,
                    s / 4,
                    (A[1, 2] + A[2, 1]) / s,
                    (A[2, 0] - A[0, 2]) / s,
                ],
                dtype=float,
            )
        else:
            s = 2 * math.sqrt(1 - A[0, 0] - A[1, 1] + A[2, 2])
            q = np.array(
                [
                    (A[0, 2] + A[2, 0]) / s,
                    (A[1, 2] + A[2, 1]) / s,
                    s / 4,
                    (A[0, 1] - A[1, 0]) / s,
                ],
                dtype=float,
            )
    return q / (sign_nz(q[3]) * np.linalg.norm(q))


def quatmult(q, r):
    q = np.asarray(q).reshape(4)
    r = np.asarray(r).reshape(4)
    return np.concatenate(
        [
            q[3] * r[:3] + r[3] * q[:3] - np.cross(q[:3], r[:3]),
            np.array([q[3] * r[3] - np.dot(q[:3], r[:3])]),
        ]
    )


def quatmult2(q, r):
    return quatmult(q, r)


def quatmat(q):
    q = np.asarray(q).reshape(4)
    top = np.hstack([q[3] * np.eye(3) - skew(q[:3]), q[:3].reshape(3, 1)])
    bot = np.hstack([-q[:3].reshape(1, 3), np.array([[q[3]]])])
    return np.vstack([top, bot])


def qinv(q):
    q = np.asarray(q).reshape(4)
    return np.concatenate([-q[:3], np.array([q[3]])]) / np.sum(q ** 2)


def qinv2(q):
    return qinv(q)


def q2mrp(q):
    q = np.asarray(q).reshape(4)
    return q[:3] / (q[3] + sign_nz(q[3]) * np.linalg.norm(q))


def q_max(q1, q2):
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    return q1 if q1[4] > q2[4] else q2


def normalize_q(Q):
    Q = np.asarray(Q)
    if Q.ndim == 1:
        if Q.size > 4:
            Q = Q[:4]
        q = Q.reshape(4)
    else:
        if Q.shape[1] > 4:
            Q = Q[:, :4]
        i = int(np.argmax(np.sum(np.abs(Q) ** 2, axis=1)))
        q = Q[i, :].reshape(4)
    return q / (sign_nz(q[3]) * np.linalg.norm(q))


def sqrt_q(qin):
    qin = np.asarray(qin).reshape(4)
    qin = qin / np.linalg.norm(qin)
    s = math.sqrt(2 * (qin[3] + sign_nz(qin[3])))
    return np.concatenate([qin[:3] / s, np.array([s / 2])])


def normalize_m(M):
    M = np.asarray(M, dtype=np.complex128)
    d = np.linalg.det(M)
    if d == 0:
        raise ValueError("normalize_m:singular")
    return M / np.sqrt(d)


def q2mobius(q):
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    a = q[3] - 1j * q[2]
    b = -q[1] + 1j * q[0]
    return normalize_m(np.array([[a, b], [-np.conj(b), np.conj(a)]], dtype=np.complex128))


def mobius2q_norm(M):
    M = normalize_m(M)
    q = np.array([np.imag(M[0, 1]), np.real(M[1, 0]), np.imag(M[1, 1]), np.real(M[0, 0])], dtype=float)
    return q / (sign_nz(q[3]) * np.linalg.norm(q))


def s2mobius(s):
    return normalize_m(np.array([[1, s], [-np.conj(s), 1]], dtype=np.complex128))


def division2mobius(s):
    return np.array([[1, 0], [0, s]], dtype=np.complex128)


def lorentz_z(v):
    if abs(v) >= 1:
        raise ValueError("lorentz_z:range")
    return normalize_m(np.array([[math.sqrt(1 - v), 0], [0, math.sqrt(1 + v)]], dtype=np.complex128))


def lorentz_z_inv(M):
    M = normalize_m(M)
    r = abs(M[1, 1] / M[0, 0])
    r2 = r ** 2
    return (r2 - 1) / (r2 + 1)


def lorentz_z_vec(v, x):
    x = np.asarray(x, dtype=float).reshape(4).copy()
    g = 1 / math.sqrt(1 - v ** 2)
    z0 = x[2]
    t0 = x[3]
    x[2] = g * (z0 + v * t0)
    x[3] = g * (t0 + v * z0)
    return x


def mobius_apply(M, z):
    M = np.asarray(M, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


def mobius_inv(M):
    M = np.asarray(M, dtype=np.complex128)
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=np.complex128)


def svd2b(X):
    X = normalize_m(X)
    P = X @ X.conj().T
    sgn = sign_nz(np.real(P[0, 0] - P[1, 1]))
    tr = np.trace(P)
    root = np.sqrt(tr ** 2 - 4)
    Sp2 = np.array([[tr - sgn * root, 0], [0, tr + sgn * root]], dtype=np.complex128) / 2
    U = sgn * (P - Sp2) @ np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if np.linalg.norm(U, ord="fro") < 100 * np.finfo(float).eps:
        U = np.eye(2, dtype=np.complex128)
    else:
        U = normalize_m(U)
    Sp = np.diag(np.sqrt(np.real(np.diag(Sp2))))
    S = np.linalg.inv(Sp)
    V = (Sp @ U.conj().T @ X).T
    V = normalize_m(V)
    return U, S, V


def vqs2mobius(vqs):
    vqs = np.asarray(vqs)
    v = float(vqs[0])
    q = np.asarray(vqs[1:5], dtype=float).reshape(4)
    s = 0.0 + 0.0j
    if vqs.size >= 7:
        s = float(vqs[5]) + 1j * float(vqs[6])
    return normalize_m(s2mobius(s) @ lorentz_z(v) @ s2mobius(-s) @ q2mobius(q))


def mobius2vqs(M):
    U, S, V = svd2b(M)
    R = U @ V.T
    q = mobius2q_norm(R)
    r = abs(S[1, 1] / S[0, 0])
    r2 = r ** 2
    v = (r2 - 1) / (r2 + 1)
    s = U[0, 1] / U[1, 1]
    return np.array([np.real(v), q[0], q[1], q[2], q[3], np.real(s), np.imag(s)], dtype=float)


def stereoproj(s, zsign=1):
    s = np.asarray(s, dtype=float)
    v = zsign * np.sqrt(np.sum(s ** 2, axis=0))
    return (s[0, :] + 1j * zsign * s[1, :]) / (s[2, :] + v)


def stereoprojinv(z, zsign=1):
    z = np.asarray(z, dtype=np.complex128)
    s = np.vstack(
        [
            zsign * (z + np.conj(z)) / 2,
            (z - np.conj(z)) / (2j),
            zsign * np.ones_like(z),
        ]
    )
    s = 2 * s / np.sum(s ** 2, axis=0) - np.array([[0], [0], [1]])
    return np.real(s)


def stereoproj2(s, f):
    s = np.asarray(s, dtype=np.complex128)
    return s / (np.sqrt(s * np.conj(s) + f ** 2) + f)


def csplit(x):
    arr, was_vector = _as_rows(x)
    xr = arr[:, 0::2]
    xi = arr[:, 1::2]
    return _restore_rows(xr, was_vector), _restore_rows(xi, was_vector)


def cjoin(xr, xi):
    xr_arr, was_vector = _as_rows(xr)
    xi_arr, _ = _as_rows(xi)
    out = np.empty((xr_arr.shape[0], xr_arr.shape[1] * 2), dtype=np.asarray(xr_arr + xi_arr).dtype)
    out[:, 0::2] = xr_arr
    out[:, 1::2] = xi_arr
    return _restore_rows(out, was_vector)


def cinv(x):
    xr, xi = csplit(x)
    d = xr ** 2 + xi ** 2
    return cjoin(xr / d, -xi / d)


def cmult(x1, x2):
    x1r, x1i = csplit(x1)
    x2r, x2i = csplit(x2)
    return cjoin(x1r * x2r - x1i * x2i, x1r * x2i + x1i * x2r)


def stereoproj_nocomplex(s, f, zsign=1):
    sx, sy = csplit(s)
    v = zsign * np.sqrt(sx ** 2 + sy ** 2 + f ** 2)
    return cjoin(sx / (f + v), zsign * sy / (f + v))


def stereoprojinv_nocomplex(z, f, zsign=1):
    zx, zy = csplit(z)
    c = 2 / (zx ** 2 + zy ** 2 + 1)
    c = f * c / (c - 1)
    return cjoin(zx * c, zsign * zy * c)


def uvf_to_stereo(u, v, f):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=float)
    L = np.sqrt(u ** 2 + v ** 2 + f ** 2)
    px = u / L
    py = v / L
    pz = f / L
    return (px + 1j * py) / (1 + pz)


def uvf_jacobian(u, v, f):
    p = np.array([u, v, f], dtype=float)
    L = np.linalg.norm(p)
    ph = p / L
    J = (np.eye(3) * L - np.outer(p, p) / L) / (L ** 2)
    num = ph[0] + 1j * ph[1]
    den = 1 + ph[2]
    z = num / den
    dnum = np.array([J[0, 0] + 1j * J[1, 0], J[0, 1] + 1j * J[1, 1], J[0, 2] + 1j * J[1, 2]])
    dden = J[2, :]
    dz = (dnum * den - num * dden) / (den ** 2)
    return z, dz[0], dz[1], dz[2]


def crossratio(a, b, c, d):
    return ((a - c) * (b - d)) / ((b - c) * (a - d))


def crossratio_orbit(z):
    return np.array([z, 1 - z, 1 / z, z / (z - 1), 1 - 1 / z, 1 / (1 - z)], dtype=np.complex128)


def canonicalize_crossratio(z, tol=1e-8):
    ops: List[str] = []
    if np.real(z) < 0.5:
        z = 1 - z
        ops.append("1-z")
    if abs(z) > 1:
        z = 1 / z
        ops.append("inv")
    if np.imag(z) < 0:
        z = 1 - z
        ops.append("1-z")
    branches = [z]
    if abs(abs(z) - 1) < tol:
        branches.append(1 / z)
    if abs(abs(1 - z) - 1) < tol:
        branches.append(1 - z)
    uniq = []
    seen = set()
    for val in branches:
        key = (round(np.real(val) / tol), round(np.imag(val) / tol))
        if key not in seen:
            seen.add(key)
            uniq.append(key[0] * tol + 1j * key[1] * tol)
    return z, np.array(uniq, dtype=np.complex128), ops


def crossratio_nocomplex(x):
    arr, was_vector = _as_rows(x)
    a = arr[:, 0:2]
    b = arr[:, 2:4]
    c = arr[:, 4:6]
    d = arr[:, 6:8]
    z = cmult(cmult(a - c, b - d), cinv(cmult(b - c, a - d)))
    return _restore_rows(np.asarray(z), was_vector)


def crossratio_jacobian(a, b, c, d):
    z = crossratio(a, b, c, d)
    return np.array(
        [
            z * (1 / (a - c) - 1 / (a - d)),
            z * (1 / (b - d) - 1 / (b - c)),
            z * (1 / (b - c) - 1 / (a - c)),
            z * (1 / (a - d) - 1 / (b - d)),
        ],
        dtype=np.complex128,
    ).reshape(1, 4)


def _complex_to_real_block(g):
    return np.array([[np.real(g), -np.imag(g)], [np.imag(g), np.real(g)]], dtype=float)


def crossratio_jacobian_nocomplex(x):
    arr, _ = _as_rows(x)
    if arr.shape[0] != 1:
        raise ValueError("crossratio_jacobian_nocomplex supports one row at a time")
    row = arr[0, :]
    a = row[0] + 1j * row[1]
    b = row[2] + 1j * row[3]
    c = row[4] + 1j * row[5]
    d = row[6] + 1j * row[7]
    z = crossratio(a, b, c, d)
    ga = z * (1 / (a - c) - 1 / (a - d))
    gb = z * (1 / (b - d) - 1 / (b - c))
    gc = z * (1 / (b - c) - 1 / (a - c))
    gd = z * (1 / (a - d) - 1 / (b - d))
    return np.hstack([_complex_to_real_block(ga), _complex_to_real_block(gb), _complex_to_real_block(gc), _complex_to_real_block(gd)])


def constellation_normalized_sensitivity(x):
    arr, was_vector = _as_rows(x)
    a = arr[:, 0:2]
    b = arr[:, 2:4]
    c = arr[:, 4:6]
    d = arr[:, 6:8]
    y = np.sum(
        np.asarray(
            [
                cinv(a - c) - cinv(a - d),
                cinv(b - d) - cinv(b - c),
                cinv(b - c) - cinv(a - c),
                cinv(a - d) - cinv(b - d),
            ]
        )
        ** 2,
        axis=2,
    )
    out = np.sum(y, axis=0)
    return out[0] if was_vector else out


def crossratio_jacobian_det_nocomplex(x):
    arr, was_vector = _as_rows(x)
    z = crossratio_nocomplex(arr)
    z2 = np.sum(np.asarray(z) ** 2, axis=1)
    a = arr[:, 0:2]
    b = arr[:, 2:4]
    c = arr[:, 4:6]
    d = arr[:, 6:8]
    s2 = np.sum(
        np.asarray(
            [
                cinv(a - c) - cinv(a - d),
                cinv(b - d) - cinv(b - c),
                cinv(b - c) - cinv(a - c),
                cinv(a - d) - cinv(b - d),
            ]
        )
        ** 2,
        axis=2,
    )
    s2 = np.sum(s2, axis=0)
    y = (z2 * s2) ** 2
    return y[0] if was_vector else y


def constellation_permute(x, idx):
    arr, was_vector = _as_rows(x)
    idx = np.asarray(idx).reshape(-1)
    if idx.size == 1 and arr.shape[0] > 1:
        idx = np.full(arr.shape[0], idx.item(), dtype=int)
    if idx.size != arr.shape[0]:
        raise ValueError("constellation_permute:size")
    perm = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 4, 5, 2, 3, 6, 7],
            [0, 1, 6, 7, 2, 3, 4, 5],
            [0, 1, 2, 3, 6, 7, 4, 5],
            [0, 1, 4, 5, 6, 7, 2, 3],
            [0, 1, 6, 7, 4, 5, 2, 3],
        ],
        dtype=int,
    )
    out = np.empty_like(arr)
    for r in range(arr.shape[0]):
        out[r, :] = arr[r, perm[int(idx[r]) - 1]]
    return _restore_rows(out, was_vector)


def constellation_permutation_idx(x):
    arr, was_vector = _as_rows(x)
    cr = np.zeros((arr.shape[0], 6), dtype=float)
    for k in range(1, 7):
        y = crossratio_nocomplex(constellation_permute(arr, np.full(arr.shape[0], k)))
        cr[:, k - 1] = (2 * y[:, 0] - 1) ** 2 + y[:, 1] ** 2
    idx = np.argmin(cr, axis=1) + 1
    return int(idx[0]) if was_vector else idx


def crossratio_jacobian_score_nocomplex(x):
    arr, was_vector = _as_rows(x)
    y = np.zeros((arr.shape[0], 6), dtype=float)
    for k in range(1, 7):
        y[:, k - 1] = np.asarray(constellation_normalized_sensitivity(constellation_permute(arr, np.full(arr.shape[0], k))), dtype=float)
    return y[0, :] if was_vector else y


def crossratio_jacobian_score_nocomplex2(x):
    arr, was_vector = _as_rows(x)
    sens = np.zeros((arr.shape[0], 6), dtype=float)
    cr = np.zeros((arr.shape[0], 6), dtype=float)
    for k in range(1, 7):
        y = constellation_permute(arr, np.full(arr.shape[0], k))
        sens[:, k - 1] = np.asarray(constellation_normalized_sensitivity(y), dtype=float)
        cr[:, k - 1] = np.sum(np.asarray(crossratio_nocomplex(y), dtype=float) ** 2, axis=1)
    if was_vector:
        return sens[0, :], cr[0, :]
    return sens, cr


def crossratio_uvf_jacobian(u, v, f):
    u = np.asarray(u, dtype=float).reshape(4)
    v = np.asarray(v, dtype=float).reshape(4)
    zz = np.zeros(4, dtype=np.complex128)
    B = np.zeros((8, 9), dtype=float)
    for k in range(4):
        zz[k], dzdu, dzdv, dzdf = uvf_jacobian(u[k], v[k], f)
        rows = slice(2 * k, 2 * k + 2)
        B[rows, 2 * k - 0 - 1 if False else 2 * k] = [np.real(dzdu), np.imag(dzdu)]
        B[rows, 2 * k + 1] = [np.real(dzdv), np.imag(dzdv)]
        B[rows, 8] = [np.real(dzdf), np.imag(dzdf)]
    z = crossratio(zz[0], zz[1], zz[2], zz[3])
    g = crossratio_jacobian(zz[0], zz[1], zz[2], zz[3]).reshape(4)
    G = np.array(
        [
            [np.real(g[0]), -np.imag(g[0]), np.real(g[1]), -np.imag(g[1]), np.real(g[2]), -np.imag(g[2]), np.real(g[3]), -np.imag(g[3])],
            [np.imag(g[0]), np.real(g[0]), np.imag(g[1]), np.real(g[1]), np.imag(g[2]), np.real(g[2]), np.imag(g[3]), np.real(g[3])],
        ],
        dtype=float,
    )
    J = G @ B
    return z, J


def _null_vector(A):
    U, S, Vh = np.linalg.svd(A)
    return Vh.conj().T[:, -1]


def m_triad(z, w):
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    w = np.asarray(w, dtype=np.complex128).reshape(-1)
    if z.size != 3 or w.size != 3:
        raise ValueError("m_triad:size")
    A = np.column_stack([z, np.ones(3, dtype=np.complex128), -z * w, -w])
    m = _null_vector(A)
    return normalize_m(np.array([[m[0], m[1]], [m[2], m[3]]], dtype=np.complex128))


def m_triad_formula(z, w):
    z = np.asarray(z, dtype=np.complex128).reshape(3)
    w = np.asarray(w, dtype=np.complex128).reshape(3)
    u = np.cross(np.array([1, 1, 1], dtype=np.complex128), z * w)
    v = np.cross(z, w)
    m = np.array([u @ w, v @ (z * w), v @ np.array([1, 1, 1]), u @ z], dtype=np.complex128)
    return normalize_m(np.array([[m[0], m[1]], [m[2], m[3]]], dtype=np.complex128))


def davenport_m_method(z, w, alpha=None):
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    w = np.asarray(w, dtype=np.complex128).reshape(-1)
    n = z.size
    if alpha is None:
        alpha = np.ones(n, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if w.size != n or alpha.size != n:
        raise ValueError("davenport_m_method:size")
    K = np.zeros((4, 4), dtype=np.complex128)
    for k in range(n):
        row = np.array([z[k], 1.0 + 0.0j, -z[k] * w[k], -w[k]], dtype=np.complex128).reshape(1, 4)
        K = K + alpha[k] * (row.conj().T @ row)
    K = (K + K.conj().T) / 2
    evals, V = np.linalg.eigh(K)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    m = V[:, idx[0]]
    M = normalize_m(np.array([[m[0], m[1]], [m[2], m[3]]], dtype=np.complex128))
    return M, K, evals


def mobest(z, wobs, alpha, niter=6, shift=1e-10):
    _, K, _ = davenport_m_method(z, wobs, alpha)
    A = K + shift * np.eye(4)
    w = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    w = w / np.linalg.norm(w)
    for _ in range(niter):
        w = np.linalg.solve(A, w)
        w = w / np.linalg.norm(w)
    lam = np.real((w.conj().T @ K @ w) / (w.conj().T @ w))
    M = normalize_m(np.array([[w[0], w[1]], [w[2], w[3]]], dtype=np.complex128))
    return M, w.reshape(4, 1), lam


def adjlower4(L):
    L = np.asarray(L)
    a1 = L[0, 0]
    b1 = L[1, 0]
    b2 = L[1, 1]
    c1 = L[2, 0]
    c2 = L[2, 1]
    c3 = L[2, 2]
    d1 = L[3, 0]
    d2 = L[3, 1]
    d3 = L[3, 2]
    d4 = L[3, 3]
    return np.array(
        [
            [b2 * c3 * d4, 0, 0, 0],
            [-b1 * c3 * d4, a1 * c3 * d4, 0, 0],
            [b1 * c2 * d4 - b2 * c1 * d4, -a1 * c2 * d4, a1 * b2 * d4, 0],
            [
                b1 * c3 * d2 - b1 * c2 * d3 + b2 * c1 * d3 - b2 * c3 * d1,
                a1 * c2 * d3 - a1 * c3 * d2,
                -a1 * b2 * d3,
                a1 * b2 * c3,
            ],
        ],
        dtype=np.asarray(L).dtype,
    )
