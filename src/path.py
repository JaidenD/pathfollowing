from dataclasses import dataclass
from typing import Callable
import numpy as np

from so3 import exp_so3, log_so3, project_to_so3, left_jacobian_inv_SO3, left_jacobian_inv_dot_SO3


@dataclass
class Path:
    L: float
    ds: float
    s_grid: np.ndarray
    omega: np.ndarray
    omega_p: np.ndarray
    omega_pp: np.ndarray
    gamma: np.ndarray
    N: np.ndarray
    N_p: np.ndarray
    N_pp: np.ndarray

    def _interpolation_index(self, s: float):
        s_clamped = float(np.clip(s, 0.0, self.L - 1e-9))
        i = int(np.floor(s_clamped / self.ds))
        s_i = self.s_grid[i]
        lam = (s_clamped - s_i) / self.ds
        return i, lam

    def eval(self, s: float):
        i, lam = self._interpolation_index(s)
        s_i = self.s_grid[i]
        delta = float(s - s_i)

        gamma_i = self.gamma[i]
        omega_i = self.omega[i]
        gamma_s = gamma_i @ exp_so3(omega_i * delta)

        omega_s = (1 - lam) * self.omega[i] + lam * self.omega[i + 1]
        omega_p_s = (1 - lam) * self.omega_p[i] + lam * self.omega_p[i + 1]
        omega_pp_s = (1 - lam) * self.omega_pp[i] + lam * self.omega_pp[i + 1]

        N_s = (1 - lam) * self.N[i] + lam * self.N[i + 1]
        N_p_s = (1 - lam) * self.N_p[i] + lam * self.N_p[i + 1]
        N_pp_s = (1 - lam) * self.N_pp[i] + lam * self.N_pp[i + 1]
        return gamma_s, omega_s, omega_p_s, omega_pp_s, N_s, N_p_s, N_pp_s


def omega_profile(s: np.ndarray) -> np.ndarray:
    w = np.zeros((len(s), 3))
    freq = 4.0
    w[:, 0] = np.cos(s * freq)
    w[:, 1] = np.sin(s * freq)
    w[:, 2] = 1.0
    return w


def build_path(omega_raw_fn: Callable[[np.ndarray], np.ndarray], L: float = 12.0, N_seg: int = 8000, R0=None):
    if R0 is None:
        R0 = np.eye(3)

    s_grid = np.linspace(0.0, L, N_seg + 1)
    ds = s_grid[1] - s_grid[0]

    omega_raw = omega_raw_fn(s_grid)
    omega = omega_raw / np.linalg.norm(omega_raw, axis=1, keepdims=True)
    omega_p = np.gradient(omega, ds, axis=0, edge_order=2)
    omega_pp = np.gradient(omega_p, ds, axis=0, edge_order=2)

    gamma = np.zeros((N_seg + 1, 3, 3))
    gamma[0] = R0
    for k in range(N_seg):
        omega_mid = 0.5 * (omega[k] + omega[k + 1])
        gamma[k + 1] = gamma[k] @ exp_so3(omega_mid * ds)
    for k in range(0, N_seg + 1, 200):
        gamma[k] = project_to_so3(gamma[k])

    N_frame = np.zeros((N_seg + 1, 3, 2))
    ref1 = np.array([0.0, 0.0, 1.0])
    ref2 = np.array([0.0, 1.0, 0.0])
    n1_prev = None
    for k in range(N_seg + 1):
        w = omega[k]
        a = ref1 if abs(np.dot(w, ref1)) < 0.95 else ref2
        n1 = a - np.dot(a, w) * w
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(w, n1)
        if n1_prev is not None and np.dot(n1, n1_prev) < 0:
            n1 = -n1
            n2 = -n2
        n1_prev = n1
        N_frame[k, :, 0] = n1
        N_frame[k, :, 1] = n2

    N_p = np.gradient(N_frame, ds, axis=0, edge_order=2)
    N_pp = np.gradient(N_p, ds, axis=0, edge_order=2)

    return Path(L=L, ds=ds, s_grid=s_grid, omega=omega, omega_p=omega_p, omega_pp=omega_pp, gamma=gamma, N=N_frame, N_p=N_p, N_pp=N_pp)


def integrate_path(L: float = 12.0, N: int = 8000) -> Path:
    return build_path(omega_profile, L=L, N_seg=N)


def closest_point(R: np.ndarray, eta_init: float, path: Path, max_iter: int = 10, tol: float = 1e-10):
    s = float(np.clip(eta_init, 0.0, path.L))
    for _ in range(max_iter):
        gamma_s, omega_s, omega_p_s, _, _, _, _ = path.eval(s)
        R_err = gamma_s.T @ R
        e = log_so3(R_err)

        g = float(e @ omega_s)
        if abs(g) < tol:
            return s

        J_inv = left_jacobian_inv_SO3(e)
        a = R_err.T @ omega_s
        alpha = float(e @ omega_p_s - omega_s @ (J_inv @ a))

        if abs(alpha) < 1e-8:
            step = -0.1 * np.sign(g)
        else:
            step = float(np.clip(-g / alpha, -0.5, 0.5))

        s_new = float(np.clip(s + step, 0.0, path.L))
        if abs(s_new - s) < 1e-12:
            return s_new
        s = s_new
    return s


def feedback_linearization(R: np.ndarray, omega: np.ndarray, eta_guess: float, path: Path, Ib_inv: np.ndarray):
    eta = closest_point(R, eta_guess, path)
    gamma_s, omega_g, omega_gp, omega_gpp, N, Np, Npp = path.eval(eta)

    R_err = gamma_s.T @ R
    e = log_so3(R_err)
    J_inv = left_jacobian_inv_SO3(e)

    a = R_err.T @ omega_g

    alpha = float(e @ omega_gp - omega_g @ (J_inv @ a))
    eta_dot_num = float(omega_g @ (J_inv.T @ omega))
    eta_dot = -eta_dot_num / alpha

    omega_err = omega - a * eta_dot
    e_dot = J_inv @ omega_err

    xi = N.T @ e
    B = (Np.T @ e) - (N.T @ (J_inv @ a))
    xi_dot = (N.T @ (J_inv @ omega)) + B * eta_dot

    J_dot_inv = left_jacobian_inv_dot_SO3(e, e_dot)
    a_dot = -np.cross(omega_err, a) + (R_err.T @ (omega_gp * eta_dot))

    alpha_dot = (
        float(e_dot @ omega_gp)
        + float(e @ omega_gpp) * eta_dot
        - eta_dot * float(omega_gp @ (J_inv @ a))
        - float(omega_g @ (J_dot_inv @ a))
        - float(omega_g @ (J_inv @ a_dot))
    )

    Ib = np.linalg.inv(Ib_inv)
    drift_val = -Ib_inv @ np.cross(omega, Ib @ omega)
    B_matrix = Ib_inv

    driftN = (
        -float((omega_gp * eta_dot) @ (J_inv.T @ omega))
        - float(omega_g @ (J_dot_inv.T @ omega))
        - float(omega_g @ (J_inv.T @ drift_val))
    )

    f_eta = (driftN / alpha) - (eta_dot * alpha_dot / alpha)
    A_eta = (-1.0 / alpha) * (omega_g @ (J_inv.T @ B_matrix))

    B_dot = (
        eta_dot * (Npp.T @ e)
        + (Np.T @ e_dot)
        - eta_dot * (Np.T @ (J_inv @ a))
        - (N.T @ (J_dot_inv @ a))
        - (N.T @ (J_inv @ a_dot))
    )

    f_xi = (
        eta_dot * (Np.T @ (J_inv @ omega))
        + (N.T @ (J_dot_inv @ omega))
        + (N.T @ (J_inv @ drift_val))
        + (B_dot * eta_dot)
        + (B * f_eta)
    )

    A_xi = (N.T @ (J_inv @ B_matrix)) + np.outer(B, A_eta)

    A = np.vstack([A_xi, A_eta])
    f = np.hstack([f_xi, f_eta])

    return {
        "eta": eta,
        "xi": xi,
        "eta_dot": eta_dot,
        "xi_dot": xi_dot,
        "A": A,
        "f": f,
        "gamma": gamma_s,
    }
