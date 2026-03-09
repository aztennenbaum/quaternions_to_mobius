#!/usr/bin/env python3
"""
TEST  Exercise the Mobius star tracker Python reference.
"""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / 'python'))

import numpy as np

from core import (
    a2q,
    adjlower4,
    canonicalize_crossratio,
    constellation_normalized_sensitivity,
    constellation_permutation_idx,
    constellation_permute,
    crossratio,
    crossratio_jacobian,
    crossratio_jacobian_det_nocomplex,
    crossratio_jacobian_nocomplex,
    crossratio_jacobian_score_nocomplex,
    crossratio_jacobian_score_nocomplex2,
    crossratio_nocomplex,
    crossratio_orbit,
    crossratio_uvf_jacobian,
    davenport_m_method,
    division2mobius,
    lorentz_z,
    lorentz_z_inv,
    lorentz_z_vec,
    m_triad,
    m_triad_formula,
    mobest,
    mobius2q_norm,
    mobius2vqs,
    mobius_apply,
    mobius_inv,
    normalize,
    normalize_m,
    normalize_q,
    normalize_vec,
    q2a,
    q2mobius,
    q2mrp,
    q_max,
    qinv,
    qinv2,
    quatmat,
    quatmult,
    quatmult2,
    s2mobius,
    sign_nz,
    skew,
    sqrt_q,
    stereoproj,
    stereoproj2,
    stereoproj_nocomplex,
    stereoprojinv,
    stereoprojinv_nocomplex,
    uvf_jacobian,
    uvf_to_stereo,
    vqs2mobius,
    svd2b,
)


def main():
    np.random.seed(1)
    print("Mobius star tracker demo")

    x = sign_nz(0)
    assert x == 1
    v = normalize_vec(np.array([3.0, 4.0, 0.0]))
    assert abs(np.linalg.norm(v) - 1) < 1e-12
    assert np.linalg.norm(normalize(np.array([3.0, 4.0, 0.0])) - v) < 1e-12
    S = skew(np.array([1.0, 2.0, 3.0]))
    assert S.shape == (3, 3)
    q = normalize_vec(np.array([0.2, -0.3, 0.1, 0.9]))
    A = q2a(q)
    q_back = a2q(A)
    assert abs(abs(q_back @ q) - 1) < 1e-10
    qp = quatmult(q, qinv2(q))
    assert np.linalg.norm(qp - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-10
    qp2 = quatmult2(q, qinv2(q))
    assert np.linalg.norm(qp2 - qp) < 1e-12
    Phi = quatmat(q)
    assert Phi.shape == (4, 4)
    mrp = q2mrp(q)
    assert mrp.size == 3
    q_conj = qinv(q)
    assert abs(q_conj[3] - q[3]) < 1e-12
    assert np.linalg.norm(q_conj[:3] + q[:3]) < 1e-12
    q_pick = q_max(np.array([1, 0, 0, 0, 1]), np.array([0, 0, 0, 1, 2]))
    assert q_pick[4] == 2
    qn = normalize_q(np.vstack([np.concatenate([q, [0]]), np.concatenate([2 * q, [0]])]))
    assert abs(np.linalg.norm(qn) - 1) < 1e-12
    qs = sqrt_q(np.array([0.0, 0.0, 0.0, 1.0]))
    assert abs(np.linalg.norm(qs) - 1) < 1e-12

    Mq = q2mobius(q)
    q_from_M = mobius2q_norm(Mq)
    assert abs(abs(q_from_M @ q) - 1) < 1e-10
    Ms = s2mobius(0.2 + 0.3j)
    Md = division2mobius(2 - 1j)
    Ml = lorentz_z(0.1)
    assert abs(np.linalg.det(normalize_m(Ms)) - 1) < 1e-10
    assert Md.shape == (2, 2)
    assert abs(lorentz_z_inv(Ml) - 0.1) < 1e-10
    vqs = np.array([0.15, *q, 0.2, -0.1])
    M = vqs2mobius(vqs)
    vqs_back = mobius2vqs(M)
    assert abs(vqs_back[0] - vqs[0]) < 1e-8
    assert abs(abs(vqs_back[1:5] @ q) - 1) < 1e-8
    U, Sb, Vb = svd2b(M)
    assert np.linalg.norm(U @ Sb @ Vb.T - normalize_m(M), ord="fro") < 1e-8
    z0 = 0.1 + 0.2j
    z1 = mobius_apply(M, z0)
    z2 = mobius_apply(mobius_inv(M), z1)
    assert abs(z2 - z0) < 1e-10
    vec4 = np.array([0.0, 0.0, 0.3, 1.0])
    vec4b = lorentz_z_vec(0.1, vec4)
    assert vec4b.size == 4

    s = normalize_vec(np.array([0.3, 0.2, 0.9]))
    zs = stereoproj(s.reshape(3, 1), 1)[0]
    s_back = stereoprojinv(np.array([zs]), 1)[:, 0]
    assert np.linalg.norm(s_back - s) < 1e-10
    zs2 = stereoproj2(0.3 + 0.2j, 1.7)
    assert not np.isnan(zs2)
    uv = np.array([10.0, -7.0, 4.0, 2.0])
    vv = np.array([5.0, 3.0, -9.0, 1.0])
    zuv = uvf_to_stereo(uv, vv, 800.0)
    zuv1, dzdu, dzdv, dzdf = uvf_jacobian(uv[0], vv[0], 800.0)
    assert abs(zuv1 - zuv[0]) < 1e-12
    vals = [np.real(dzdu), np.imag(dzdu), np.real(dzdv), np.imag(dzdv), np.real(dzdf), np.imag(dzdf)]
    assert np.all(np.isfinite(vals))
    sp_in_nc = np.array([1.0, 2.0, 3.0, 4.0])
    sp_nc = stereoproj_nocomplex(sp_in_nc, 5.0, 1)
    sp_inv_nc = stereoprojinv_nocomplex(sp_nc, 5.0, 1)
    assert sp_nc.shape == sp_in_nc.shape
    assert sp_inv_nc.shape == sp_in_nc.shape
    assert np.linalg.norm(sp_inv_nc - sp_in_nc) < 1e-12

    z = np.array([0.1 + 0.2j, 0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.5j])
    cr = crossratio(z[0], z[1], z[2], z[3])
    orb = crossratio_orbit(cr)
    assert orb.size == 6
    cr_can, branches, ops = canonicalize_crossratio(cr)
    assert branches.size > 0
    assert isinstance(ops, list)
    xnc = np.array([np.real(z[0]), np.imag(z[0]), np.real(z[1]), np.imag(z[1]), np.real(z[2]), np.imag(z[2]), np.real(z[3]), np.imag(z[3])])
    cr_nc = crossratio_nocomplex(xnc)
    assert np.linalg.norm(cr_nc - np.array([np.real(cr), np.imag(cr)])) < 1e-10
    Jc = crossratio_jacobian(z[0], z[1], z[2], z[3])
    Jnc = crossratio_jacobian_nocomplex(xnc)
    assert Jc.shape == (1, 4)
    assert Jnc.shape == (2, 8)
    Jdet = crossratio_jacobian_det_nocomplex(xnc)
    assert Jdet >= 0
    sens = constellation_normalized_sensitivity(xnc)
    assert sens >= 0
    perm3 = constellation_permute(xnc, 3)
    assert perm3.size == 8
    pidx = constellation_permutation_idx(xnc)
    assert 1 <= pidx <= 6
    score1 = crossratio_jacobian_score_nocomplex(xnc)
    score2, cr2 = crossratio_jacobian_score_nocomplex2(xnc)
    assert score1.shape == (6,)
    assert score2.shape == (6,)
    assert cr2.shape == (6,)
    cr_uvf, Juvf = crossratio_uvf_jacobian(uv, vv, 800.0)
    assert Juvf.shape == (2, 9)
    assert np.isfinite(np.real(cr_uvf)) and np.isfinite(np.imag(cr_uvf))

    z_ref = np.array([0.1 + 0.2j, -0.4 + 0.3j, 0.6 - 0.2j, -0.2 - 0.5j])
    M_true = vqs2mobius(np.array([0.08, *q, 0.12, -0.15]))
    w_obs = mobius_apply(M_true, z_ref)
    M3 = m_triad(z_ref[:3], w_obs[:3])
    M3f = m_triad_formula(z_ref[:3], w_obs[:3])
    for k in range(3):
        assert abs(mobius_apply(M3, z_ref[k]) - w_obs[k]) < 1e-8
        assert abs(mobius_apply(M3f, z_ref[k]) - w_obs[k]) < 1e-8
    Mls, K, evals = davenport_m_method(z_ref, w_obs, np.ones(4))
    assert K.shape == (4, 4)
    assert evals.size == 4
    Mfast, wvec, lam = mobest(z_ref, w_obs, np.ones(4), 6, 1e-10)
    assert wvec.shape == (4, 1)
    assert np.isfinite(lam)
    for k in range(4):
        assert abs(mobius_apply(Mls, z_ref[k]) - w_obs[k]) < 1e-6
        assert abs(mobius_apply(Mfast, z_ref[k]) - w_obs[k]) < 1e-5

    A3 = np.column_stack([z_ref[:3], np.ones(3), -z_ref[:3] * w_obs[:3], -w_obs[:3]])
    m_formula = np.array([M3f[0, 0], M3f[0, 1], M3f[1, 0], M3f[1, 1]])
    res = A3 @ m_formula
    assert np.linalg.norm(res) < 1e-8
    _, _, Vh = np.linalg.svd(A3)
    m_null = Vh.conj().T[:, -1]
    scale = (m_null.conj().T @ m_formula) / (m_null.conj().T @ m_null)
    assert np.linalg.norm(m_formula - scale * m_null) < 1e-8
    print("Symbolic proof skipped; numeric proof check passed")

    H = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    L = np.linalg.cholesky(H)
    Adj = adjlower4(L)
    assert Adj.shape == (4, 4)

    print("All demo checks passed")


if __name__ == "__main__":
    main()
