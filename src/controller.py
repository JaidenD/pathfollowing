from dataclasses import dataclass
import numpy as np

from so3 import exp_so3, log_so3
from path import tube_coords, SO3Path


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


def compute_torque(
    t: float,
    R: np.ndarray,
    w: np.ndarray,
    path: SO3Path,
    eta_prev: float,
    gains: Gains,
    J_nom: np.ndarray,
    dt: float,
    xi_prev: np.ndarray,
):
    eta_guess = eta_prev + dt * (w @ path.omega_at(eta_prev))
    eta, xi, _ = tube_coords(path, R, eta_guess)

    # Approximate ydot by projecting angular velocity onto tangent/normal frame
    tvec = path.omega_at(eta)
    n1, n2 = path.frame_at(eta)
    ydot = np.array([tvec @ w, n1 @ w, n2 @ w])
    xidot = ydot[1:]
    etadot = ydot[0]

    nu = 0.0
    dnu_dt = 0.0
    dnu_deta = 0.0
    # filled by caller with profile object

    v_perp = -(gains.Kp_xi @ xi + gains.Kd_xi @ xidot)

    return eta, xi, ydot, v_perp


def closed_loop_step(
    t: float,
    R: np.ndarray,
    w: np.ndarray,
    path: SO3Path,
    eta_prev: float,
    gains: Gains,
    J_nom: np.ndarray,
    J_true: np.ndarray,
    ref: RefProfile,
    dt: float,
    xi_prev: np.ndarray,
):
    eta, xi, ydot, v_perp = compute_torque(t, R, w, path, eta_prev, gains, J_nom, dt, xi_prev)
    etadot = ydot[0]
    nu = ref.nu(t, eta)
    v_par = ref.dnu_dt(t, eta) + ref.dnu_deta(t, eta) * etadot - gains.kt * (etadot - nu)
    v = np.array([v_par, v_perp[0], v_perp[1]])

    # Approximate mapping omega ~= [t n1 n2] ydot, thus yddot ~= [t n1 n2]^T wdot.
    tvec = path.omega_at(eta)
    n1, n2 = path.frame_at(eta)
    B = np.stack([tvec, n1, n2], axis=1)

    # D = M^{-1}J^{-1} with M approx B^T -> M^{-1} approx B
    D = B.T @ np.linalg.inv(J_nom)
    f = np.zeros(3)
    rhs = v - f
    tau = np.linalg.solve(D, rhs)

    # true rigid body dynamics
    wdot = np.linalg.solve(J_true, tau - np.cross(w, J_true @ w))
    w_mid = w + 0.5 * dt * wdot
    Rn = R @ exp_so3(w_mid * dt)
    wn = w + dt * wdot

    return Rn, wn, eta, xi, etadot, nu, tau
