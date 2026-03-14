from dataclasses import dataclass
import numpy as np

from so3 import exp_so3
from path import feedback_linearization, Path


@dataclass
class Gains:
    Kp_xi: np.ndarray
    Kd_xi: np.ndarray
    kt: float


@dataclass
class RefProfile:
    kind: str  # 'time' or 'eta'

    def nu(self, t: float, eta: float) -> float:
        if self.kind == "time":
            return 2.75 * np.sin(0.5 * t)
        return 1.5 - np.sin(np.pi * eta / 12.0)

    def dnu_dt(self, t: float, eta: float) -> float:
        if self.kind == "time":
            return 1.375 * np.cos(0.5 * t)
        return 0.0

    def dnu_deta(self, t: float, eta: float) -> float:
        if self.kind == "time":
            return 0.0
        return -(np.pi / 12.0) * np.cos(np.pi * eta / 12.0)


def closed_loop_step(
    t: float,
    R: np.ndarray,
    w: np.ndarray,
    path: Path,
    eta_prev: float,
    gains: Gains,
    J_nom: np.ndarray,
    J_true: np.ndarray,
    ref: RefProfile,
    dt: float,
    xi_prev: np.ndarray,
):
    del xi_prev  # not needed with exact formulas
    data = feedback_linearization(R=R, omega=w, eta_guess=eta_prev, path=path, Ib_inv=np.linalg.inv(J_nom))

    eta = data["eta"]
    xi = data["xi"]
    eta_dot = data["eta_dot"]
    xi_dot = data["xi_dot"]
    A = data["A"]
    f = data["f"]

    v_perp = -(gains.Kp_xi @ xi + gains.Kd_xi @ xi_dot)
    nu = ref.nu(t, eta)
    v_par = ref.dnu_dt(t, eta) + ref.dnu_deta(t, eta) * eta_dot - gains.kt * (eta_dot - nu)
    v = np.hstack([v_perp, v_par])

    rhs = v - f
    if np.linalg.cond(A) < 1e8:
        tau = np.linalg.solve(A, rhs)
    else:
        tau = np.linalg.lstsq(A, rhs, rcond=None)[0]

    J_true_inv = np.linalg.inv(J_true)

    def calc_wdot(wk: np.ndarray) -> np.ndarray:
        return J_true_inv @ (tau - np.cross(wk, J_true @ wk))

    k1 = calc_wdot(w)
    w2 = w + 0.5 * dt * k1
    k2 = calc_wdot(w2)
    w3 = w + 0.5 * dt * k2
    k3 = calc_wdot(w3)
    w4 = w + dt * k3
    k4 = calc_wdot(w4)

    wn = w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    w_avg = (w + 2 * w2 + 2 * w3 + w4) / 6.0
    Rn = R @ exp_so3(w_avg * dt)

    return Rn, wn, eta, xi, eta_dot, nu, tau
