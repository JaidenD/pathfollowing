import json
from pathlib import Path
import numpy as np

from so3 import exp_so3, log_so3, project_to_so3, orthogonality_error
from path import integrate_path
from controller import Gains, RefProfile, closed_loop_step


def check_so3_roundtrip(samples: int = 4000, seed: int = 10):
    rng = np.random.default_rng(seed)
    max_roundtrip = 0.0
    max_group_err = 0.0
    near_pi_fail = 0

    for _ in range(samples):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0.0, np.pi - 1e-4)
        phi = axis * theta
        R = exp_so3(phi)
        phi_hat = log_so3(R)
        R2 = exp_so3(phi_hat)
        max_roundtrip = max(max_roundtrip, np.linalg.norm(phi - phi_hat))
        max_group_err = max(max_group_err, np.linalg.norm(R - R2, ord='fro'))

    for _ in range(samples // 10):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = np.pi - rng.uniform(1e-7, 1e-5)
        R = exp_so3(axis * theta)
        phi = log_so3(R)
        if not np.all(np.isfinite(phi)):
            near_pi_fail += 1

    return {
        "samples": samples,
        "max_phi_roundtrip_err": float(max_roundtrip),
        "max_R_roundtrip_fro_err": float(max_group_err),
        "near_pi_nonfinite_count": int(near_pi_fail),
    }


def check_projection(samples: int = 1000, seed: int = 11):
    rng = np.random.default_rng(seed)
    max_ortho = 0.0
    det_min = 10.0
    for _ in range(samples):
        A = np.eye(3) + 1e-2 * rng.normal(size=(3, 3))
        Rp = project_to_so3(A)
        max_ortho = max(max_ortho, orthogonality_error(Rp))
        det_min = min(det_min, np.linalg.det(Rp))
    return {
        "samples": samples,
        "max_projection_orthogonality_error": float(max_ortho),
        "min_projection_det": float(det_min),
    }


def check_invariance(dt: float = 0.0025, T: float = 2.0):
    path = integrate_path(L=12.0, N=8000)
    J = np.diag([1.0, 1.2, 0.8])
    gains = Gains(Kp_xi=225.0 * np.eye(2), Kd_xi=30.0 * np.eye(2), kt=50.0)
    ref = RefProfile(kind="eta")

    eta = 2.0
    gamma, omega_g, _, _, _, _, _ = path.eval(eta)
    R = gamma.copy()
    w = 0.7 * omega_g
    xi_prev = np.zeros(2)

    N = int(T / dt)
    max_xi = 0.0
    for k in range(N):
        R, w, eta, xi, *_ = closed_loop_step(k * dt, R, w, path, eta, gains, J, J, ref, dt, xi_prev)
        xi_prev = xi
        max_xi = max(max_xi, float(np.linalg.norm(xi)))

    return {"dt": dt, "T": T, "max_transverse_error_from_on_path_init": float(max_xi)}


def main():
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "so3_checks": check_so3_roundtrip(),
        "projection_checks": check_projection(),
        "invariance_check": check_invariance(),
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
