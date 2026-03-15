import json
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from so3 import orthogonality_error, project_to_so3, exp_so3
from path import integrate_path
from controller import Gains, RefProfile, closed_loop_step, ControllerOptions


def convergence_check(t, xi_hist, vel_err, settle_t=10.0, tol_xi=2e-4, tol_ev=1e-3):
    mask = t >= settle_t
    if not np.any(mask):
        return {"pass": False, "reason": "settle_time_out_of_range"}
    xi_tail = np.linalg.norm(xi_hist[mask], axis=1)
    ev_tail = np.abs(vel_err[mask])
    xi_max = float(np.max(xi_tail))
    ev_max = float(np.max(ev_tail))
    return {
        "pass": bool((xi_max < tol_xi) and (ev_max < tol_ev)),
        "xi_tail_max": xi_max,
        "ev_tail_max": ev_max,
        "tol_xi": tol_xi,
        "tol_ev": tol_ev,
        "settle_t": settle_t,
    }


def run_case(
    kind: str,
    T: float = 12.0,
    dt: float = 0.005,
    J_true=None,
    eta0: float = 0.1,
    xi0=None,
    path=None,
    options: Optional[ControllerOptions] = None,
):
    if path is None:
        path = integrate_path(L=12.0, N=2500)
    if options is None:
        options = ControllerOptions()  # faithful: no clipping/regularization by default

    J_nom = np.diag([1.0, 1.2, 0.8])
    if J_true is None:
        J_true = J_nom.copy()
    if xi0 is None:
        xi0 = np.array([0.1, -0.1], dtype=float)

    gains = Gains(Kp_xi=225.0 * np.eye(2), Kd_xi=30.0 * np.eye(2), kt=50.0)
    ref = RefProfile(kind=kind)

    N = int(T / dt)
    t = np.linspace(0.0, T, N + 1)

    eta = float(eta0)
    gamma0, _, _, _, N0, _, _ = path.eval(eta)
    R = gamma0 @ exp_so3(N0 @ xi0)
    w = np.zeros(3)

    xi_hist = np.zeros((N + 1, 2))
    eta_hist = np.zeros(N + 1)
    edot_hist = np.zeros(N + 1)
    nu_hist = np.zeros(N + 1)
    tau_hist = np.zeros((N + 1, 3))
    orth_hist = np.zeros(N + 1)

    sim_ok = True
    fail_reason = ""

    for k in range(N + 1):
        if k > 0:
            try:
                R, w, eta, xi, etadot, nu, tau = closed_loop_step(
                    t[k - 1], R, w, path, eta, gains, J_nom, J_true, ref, dt, options
                )
            except Exception as exc:
                sim_ok = False
                fail_reason = str(exc)
                xi = xi_hist[k - 1]
                etadot = edot_hist[k - 1]
                nu = nu_hist[k - 1]
                tau = tau_hist[k - 1]
                xi_hist[k:] = xi
                eta_hist[k:] = eta
                edot_hist[k:] = etadot
                nu_hist[k:] = nu
                tau_hist[k:] = tau
                orth_hist[k:] = orthogonality_error(R)
                break

            if k % 50 == 0:
                R = project_to_so3(R)
        else:
            xi = xi0.copy()
            etadot = 0.0
            nu = ref.nu(0.0, eta)
            tau = np.zeros(3)

        xi_hist[k] = xi
        eta_hist[k] = eta
        edot_hist[k] = etadot
        nu_hist[k] = nu
        tau_hist[k] = tau
        orth_hist[k] = orthogonality_error(R)

    vel_err = edot_hist - nu_hist
    eps = 1e-3
    boundary_hits = int(np.sum((eta_hist < eps) | (eta_hist > path.L - eps)))

    # Additional assumption checks tied to theorem conditions
    tube_radius_limit = 0.25
    injectivity_violations = int(np.sum(np.linalg.norm(xi_hist, axis=1) >= np.pi))
    tube_violations = int(np.sum(np.linalg.norm(xi_hist, axis=1) >= tube_radius_limit))

    metrics = {
        "sim_ok": sim_ok,
        "fail_reason": fail_reason,
        "final_xi_norm": float(np.linalg.norm(xi_hist[-1])),
        "rms_vel_err": float(np.sqrt(np.mean(vel_err**2))),
        "max_tau_norm": float(np.max(np.linalg.norm(tau_hist, axis=1))),
        "max_orthogonality_error": float(np.max(orth_hist)),
        "eta_min": float(np.min(eta_hist)),
        "eta_max": float(np.max(eta_hist)),
        "boundary_hit_fraction": float(boundary_hits / len(eta_hist)),
        "tube_radius_limit": tube_radius_limit,
        "tube_violation_fraction": float(tube_violations / len(eta_hist)),
        "injectivity_violation_fraction": float(injectivity_violations / len(eta_hist)),
    }
    metrics["convergence"] = convergence_check(t, xi_hist, vel_err, settle_t=min(10.0, 0.7 * T), tol_xi=2e-4, tol_ev=1e-3)

    return t, xi_hist, vel_err, eta_hist, metrics


def plot_case(path_png: Path, t, xi_hist, vel_err, title: str):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, xi_hist[:, 0], label=r"$\xi_1$")
    ax[0].plot(t, xi_hist[:, 1], label=r"$\xi_2$")
    ax[0].plot(t, np.linalg.norm(xi_hist, axis=1), "k--", label=r"$\|\xi\|$")
    ax[0].set_ylabel("Transverse coordinates")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(t, vel_err, color="tab:red")
    ax[1].set_ylabel(r"$\dot\eta-\nu_d$")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)


def monte_carlo_traces(path_png: Path, path, options: ControllerOptions, trials: int = 30, seed: int = 7, T: float = 15.0, dt: float = 0.02):
    rng = np.random.default_rng(seed)
    J_nom = np.diag([1.0, 1.2, 0.8])

    t_nom, xi_nom, ve_nom, eta_nom, m_nom = run_case("eta", T=T, dt=dt, J_true=J_nom, path=path, options=options)

    finals, rmses, pass_count = [], [], 0
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for _ in range(trials):
        P = rng.uniform(-0.2, 0.2, size=(3, 3))
        P = 0.5 * (P + P.T)
        J_true = J_nom @ (np.eye(3) + P)
        eigvals, eigvecs = np.linalg.eigh(J_true)
        eigvals = np.clip(eigvals, 0.2, None)
        J_true = eigvecs @ np.diag(eigvals) @ eigvecs.T

        t_mc, xi_mc, ve_mc, eta_mc, metrics = run_case("eta", T=T, dt=dt, J_true=J_true, path=path, options=options)
        finals.append(metrics["final_xi_norm"])
        rmses.append(metrics["rms_vel_err"])
        if metrics["convergence"]["pass"]:
            pass_count += 1

        ax[0].plot(t_mc, np.linalg.norm(xi_mc, axis=1), color="tab:blue", alpha=0.15, linewidth=1.0)
        ax[1].plot(t_mc, ve_mc, color="tab:red", alpha=0.15, linewidth=1.0)

    ax[0].plot(t_nom, np.linalg.norm(xi_nom, axis=1), color="black", linewidth=2.0, label="nominal inertia")
    ax[1].plot(t_nom, ve_nom, color="black", linewidth=2.0, label="nominal inertia")

    ax[0].set_ylabel(r"$\|\xi\|$")
    ax[0].set_title("Monte Carlo (30 runs): slowdown profile, T=15s")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].set_ylabel(r"$\dot\eta-\nu_d$")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)

    return {
        "trials": trials,
        "T": T,
        "pass_count_relaxed_criterion": pass_count,
        "final_xi_norm_mean": float(np.mean(finals)),
        "final_xi_norm_std": float(np.std(finals)),
        "rms_vel_err_mean": float(np.mean(rmses)),
        "rms_vel_err_std": float(np.std(rmses)),
        "nominal": m_nom,
        "nominal_eta_min": float(np.min(eta_nom)),
        "nominal_eta_max": float(np.max(eta_nom)),
    }


def main():
    out_fig = Path("figures")
    out_res = Path("results")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    path = integrate_path(L=12.0, N=2500)
    faithful = ControllerOptions()

    t1, xi1, ve1, eta1, m1 = run_case("time", T=12.0, dt=0.005, eta0=6.0, xi0=np.array([0.1, -0.1]), path=path, options=faithful)
    plot_case(out_fig / "wavy-path.png", t1, xi1, ve1, "Case A: time-varying tangential speed")

    t2, xi2, ve2, eta2, m2 = run_case("eta", T=15.0, dt=0.005, eta0=0.1, xi0=np.array([0.1, -0.1]), path=path, options=faithful)
    plot_case(out_fig / "slowdown.png", t2, xi2, ve2, "Case B: eta-dependent tangential speed")

    mmc = monte_carlo_traces(out_fig / "monte-carlo.png", path=path, options=faithful, trials=30, T=15.0, dt=0.02)

    summary = {
        "controller_mode": "faithful_no_clipping",
        "convergence_criterion": {"tol_xi": 2e-4, "tol_ev": 1e-3},
        "case_time": m1,
        "case_time_eta_min": float(np.min(eta1)),
        "case_time_eta_max": float(np.max(eta1)),
        "case_eta": m2,
        "case_eta_eta_min": float(np.min(eta2)),
        "case_eta_eta_max": float(np.max(eta2)),
        "monte_carlo": mmc,
    }
    (out_res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
