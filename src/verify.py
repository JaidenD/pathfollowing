import json
from pathlib import Path
import numpy as np

from so3 import exp_so3, log_so3, project_to_so3, orthogonality_error
from path import integrate_path
from controller import Gains, RefProfile, closed_loop_step


def check_so3_roundtrip(samples: int = 5000, seed: int = 10):
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

    # Stress near pi
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


def closed_loop_run(dt: float, T: float = 4.0, kind: str = "eta"):
    path = integrate_path(L=12.0, N=1000)
    J_nom = np.diag([1.0, 1.2, 0.8])
    J_true = J_nom.copy()
    gains = Gains(Kp_xi=225.0 * np.eye(2), Kd_xi=30.0 * np.eye(2), kt=50.0)
    ref = RefProfile(kind=kind)

    N = int(T / dt)
    t = np.linspace(0.0, T, N + 1)
    R = np.eye(3)
    w = np.zeros(3)
    eta = 0.1
    xi_prev = np.array([0.1, -0.1])

    xi_norm = np.zeros(N + 1)
    vel_err = np.zeros(N + 1)
    orth = np.zeros(N + 1)

    for k in range(1, N + 1):
        R, w, eta, xi, etadot, nu, _ = closed_loop_step(
            t[k - 1], R, w, path, eta, gains, J_nom, J_true, ref, dt, xi_prev
        )
        xi_prev = xi
        xi_norm[k] = np.linalg.norm(xi)
        vel_err[k] = etadot - nu
        orth[k] = orthogonality_error(R)

    return {
        "dt": dt,
        "final_xi_norm": float(xi_norm[-1]),
        "rms_vel_err": float(np.sqrt(np.mean(vel_err**2))),
        "max_orthogonality_error": float(np.max(orth)),
    }


def check_invariance(dt: float = 0.01, T: float = 2.0):
    path = integrate_path(L=12.0, N=1000)
    J_nom = np.diag([1.0, 1.2, 0.8])
    gains = Gains(Kp_xi=225.0 * np.eye(2), Kd_xi=30.0 * np.eye(2), kt=50.0)
    ref = RefProfile(kind="eta")

    eta = 2.0
    R = path.gamma_at(eta)
    w = 0.7 * path.omega_at(eta)  # tangent velocity only
    xi_prev = np.zeros(2)

    N = int(T / dt)
    max_xi = 0.0
    for k in range(N):
        R, w, eta, xi, *_ = closed_loop_step(
            k * dt, R, w, path, eta, gains, J_nom, J_nom, ref, dt, xi_prev
        )
        xi_prev = xi
        max_xi = max(max_xi, float(np.linalg.norm(xi)))

    return {
        "dt": dt,
        "T": T,
        "max_transverse_error_from_on_path_init": float(max_xi),
    }


def evaluate_pass_fail(report):
    checks = {}
    checks["so3_roundtrip"] = report["so3_checks"]["max_R_roundtrip_fro_err"] < 1e-6 and report["so3_checks"]["near_pi_nonfinite_count"] == 0
    checks["projection"] = report["projection_checks"]["max_projection_orthogonality_error"] < 1e-10 and report["projection_checks"]["min_projection_det"] > 0.999999
    checks["orthogonality_closed_loop"] = max(x["max_orthogonality_error"] for x in report["closed_loop_dt_sweep"]) < 1e-10
    checks["invariance"] = report["invariance_check"]["max_transverse_error_from_on_path_init"] < 1e-3
    report["pass_fail"] = checks
    report["all_pass"] = all(checks.values())
    return report


def main():
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    so3_checks = check_so3_roundtrip()
    proj_checks = check_projection()
    coarse = closed_loop_run(dt=0.03)
    medium = closed_loop_run(dt=0.015)
    fine = closed_loop_run(dt=0.0075)
    invariance = check_invariance(dt=0.01)

    # empirical convergence ratio for final xi and rms vel err
    conv = {
        "xi_ratio_coarse_to_medium": float(abs(coarse["final_xi_norm"] - medium["final_xi_norm"]) / max(abs(medium["final_xi_norm"] - fine["final_xi_norm"]), 1e-12)),
        "vel_ratio_coarse_to_medium": float(abs(coarse["rms_vel_err"] - medium["rms_vel_err"]) / max(abs(medium["rms_vel_err"] - fine["rms_vel_err"]), 1e-12)),
    }

    report = {
        "so3_checks": so3_checks,
        "projection_checks": proj_checks,
        "closed_loop_dt_sweep": [coarse, medium, fine],
        "invariance_check": invariance,
        "convergence_indicators": conv,
    }
    report = evaluate_pass_fail(report)
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
