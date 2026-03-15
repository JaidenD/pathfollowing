import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from so3 import orthogonality_error, project_to_so3
from path import integrate_path
from controller import Gains, RefProfile, closed_loop_step


def run_case(kind: str, T: float = 6.0, dt: float = 0.03, J_true=None, seed: int = 0):
    path = integrate_path(L=12.0, N=800)
    J_nom = np.diag([1.0, 1.2, 0.8])
    if J_true is None:
        J_true = J_nom.copy()

    gains = Gains(Kp_xi=225.0 * np.eye(2), Kd_xi=30.0 * np.eye(2), kt=50.0)
    ref = RefProfile(kind=kind)

    N = int(T / dt)
    t = np.linspace(0.0, T, N + 1)

    R = np.eye(3)
    # initialize near eta=0.1, xi=(0.1,-0.1)
    eta = 0.1
    w = np.array([0.0, 0.0, 0.0])
    xi_prev = np.array([0.1, -0.1])

    xi_hist = np.zeros((N + 1, 2))
    eta_hist = np.zeros(N + 1)
    edot_hist = np.zeros(N + 1)
    nu_hist = np.zeros(N + 1)
    tau_hist = np.zeros((N + 1, 3))
    orth_hist = np.zeros(N + 1)

    for k in range(N + 1):
        if k > 0:
            R, w, eta, xi, etadot, nu, tau = closed_loop_step(
                t[k - 1], R, w, path, eta, gains, J_nom, J_true, ref, dt, xi_prev
            )
            if k % 200 == 0:
                R = project_to_so3(R)
            xi_prev = xi
        else:
            xi = xi_prev
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
    metrics = {
        "final_xi_norm": float(np.linalg.norm(xi_hist[-1])),
        "rms_vel_err": float(np.sqrt(np.mean(vel_err**2))),
        "max_tau_norm": float(np.max(np.linalg.norm(tau_hist, axis=1))),
        "max_orthogonality_error": float(np.max(orth_hist)),
    }

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


def monte_carlo(path_png: Path, trials: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    J = np.diag([1.0, 1.2, 0.8])
    finals = []
    rmses = []
    for _ in range(trials):
        P = rng.uniform(-0.2, 0.2, size=(3, 3))
        P = 0.5 * (P + P.T)
        J_true = J @ (np.eye(3) + P)
        # enforce SPD
        eigvals, eigvecs = np.linalg.eigh(J_true)
        eigvals = np.clip(eigvals, 0.2, None)
        J_true = eigvecs @ np.diag(eigvals) @ eigvecs.T

        _, xi_hist, vel_err, _, metrics = run_case("eta", T=5.0, dt=0.03, J_true=J_true)
        finals.append(metrics["final_xi_norm"])
        rmses.append(metrics["rms_vel_err"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(finals, bins=10, color="tab:blue", alpha=0.8)
    ax[0].set_title(r"Final $\|\xi(T)\|$")
    ax[0].grid(True, alpha=0.3)
    ax[1].hist(rmses, bins=10, color="tab:orange", alpha=0.8)
    ax[1].set_title(r"RMS $|\dot\eta-\nu_d|$")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)

    return {
        "trials": trials,
        "final_xi_norm_mean": float(np.mean(finals)),
        "final_xi_norm_std": float(np.std(finals)),
        "rms_vel_err_mean": float(np.mean(rmses)),
        "rms_vel_err_std": float(np.std(rmses)),
    }


def main():
    out_fig = Path("figures")
    out_res = Path("results")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    t1, xi1, ve1, _, m1 = run_case("time", T=6.0, dt=0.03)
    plot_case(out_fig / "wavy-path.png", t1, xi1, ve1, "Case A: time-varying tangential speed")

    t2, xi2, ve2, _, m2 = run_case("eta", T=6.0, dt=0.03)
    plot_case(out_fig / "slowdown.png", t2, xi2, ve2, "Case B: eta-dependent tangential speed")

    mmc = monte_carlo(out_fig / "monte-carlo.png", trials=30)

    summary = {"case_time": m1, "case_eta": m2, "monte_carlo": mmc}
    (out_res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
