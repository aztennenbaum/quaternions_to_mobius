# exper_results_mobius_startracker.py
# Python 3.10+
# Dependencies: numpy
# Optional: matplotlib, scipy
#
# This is a lightly cleaned experiment scaffold for the Mobius star tracker
# Python port. It reuses the reference implementation in core.py.

import numpy as np

from core import canonicalize_crossratio, crossratio, q2a, q2mobius, stereoproj, uvf_jacobian, uvf_to_stereo, vqs2mobius


def random_unit_quaternion(rng):
    u1, u2, u3 = rng.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    return q / np.linalg.norm(q)


def random_unit_vectors(n, rng):
    v = rng.normal(size=(3, n))
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    return v


def simulate_crossratio_variance(n_trials=1000, f_nom=800.0, sig_uv=0.15, sig_f=1.0, seed=7):
    rng = np.random.RandomState(seed)
    q = random_unit_quaternion(rng)
    R = q2a(q)
    raw = []
    canon = []
    for _ in range(n_trials):
        s_ref = random_unit_vectors(4, rng)
        s_cam = R @ s_ref
        u0 = f_nom * (s_cam[0] / (s_cam[2] + 1e-12))
        v0 = f_nom * (s_cam[1] / (s_cam[2] + 1e-12))
        u = u0 + rng.normal(0, sig_uv, size=4)
        v = v0 + rng.normal(0, sig_uv, size=4)
        f = f_nom + rng.normal(0, sig_f)
        z = uvf_to_stereo(u, v, f)
        cr = crossratio(z[0], z[1], z[2], z[3])
        cr_can, _, _ = canonicalize_crossratio(cr)
        raw.append(cr)
        canon.append(cr_can)
    raw = np.asarray(raw)
    canon = np.asarray(canon)
    return {
        "cr_var": np.var(np.vstack([raw.real, raw.imag]), axis=1),
        "crcanon_var": np.var(np.vstack([canon.real, canon.imag]), axis=1),
    }


if __name__ == "__main__":
    out = simulate_crossratio_variance()
    print("Var[Re(CR)], Var[Im(CR)] =", out["cr_var"])
    print("Var[Re(CR_canon)], Var[Im(CR_canon)] =", out["crcanon_var"])
