from dataclasses import dataclass
import numpy as np

from so3 import exp_so3, log_so3


@dataclass
class SO3Path:
    s_grid: np.ndarray
    R_grid: np.ndarray
    omega_grid: np.ndarray
    n1_grid: np.ndarray
    n2_grid: np.ndarray
    L: float

    def wrap_s(self, s: float) -> float:
        return float(np.clip(s, self.s_grid[0], self.s_grid[-1]))

    def _idx_alpha(self, s: float):
        s = self.wrap_s(s)
        k = np.searchsorted(self.s_grid, s, side="right") - 1
        k = int(np.clip(k, 0, len(self.s_grid) - 2))
        ds = self.s_grid[k + 1] - self.s_grid[k]
        alpha = (s - self.s_grid[k]) / ds
        return k, alpha

    def omega_at(self, s: float) -> np.ndarray:
        k, a = self._idx_alpha(s)
        v = (1 - a) * self.omega_grid[k] + a * self.omega_grid[k + 1]
        return v / np.linalg.norm(v)

    def frame_at(self, s: float):
        k, a = self._idx_alpha(s)
        n1 = (1 - a) * self.n1_grid[k] + a * self.n1_grid[k + 1]
        n1 = n1 - self.omega_at(s) * (self.omega_at(s) @ n1)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(self.omega_at(s), n1)
        n2 /= np.linalg.norm(n2)
        return n1, n2

    def gamma_at(self, s: float) -> np.ndarray:
        k, _ = self._idx_alpha(s)
        sk = self.s_grid[k]
        ds = s - sk
        return self.R_grid[k] @ exp_so3(self.omega_grid[k] * ds)


def omega_profile(s: np.ndarray) -> np.ndarray:
    w = np.stack([np.cos(4.0 * s), np.sin(4.0 * s), np.ones_like(s)], axis=1) / np.sqrt(2.0)
    return w


def integrate_path(L: float = 12.0, N: int = 2400) -> SO3Path:
    s_grid = np.linspace(0.0, L, N + 1)
    ds = s_grid[1] - s_grid[0]
    omega = omega_profile(s_grid)

    R_grid = np.zeros((N + 1, 3, 3), dtype=float)
    R_grid[0] = np.eye(3)
    for k in range(N):
        w_mid = (omega[k] + omega[k + 1])
        w_mid /= np.linalg.norm(w_mid)
        R_grid[k + 1] = R_grid[k] @ exp_so3(w_mid * ds)

    # transport-like frame construction
    n1 = np.zeros((N + 1, 3), dtype=float)
    n2 = np.zeros((N + 1, 3), dtype=float)
    t0 = omega[0] / np.linalg.norm(omega[0])
    seed = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1[0] = seed - t0 * (seed @ t0)
    n1[0] /= np.linalg.norm(n1[0])
    n2[0] = np.cross(t0, n1[0])

    for k in range(1, N + 1):
        t = omega[k] / np.linalg.norm(omega[k])
        v = n1[k - 1] - t * (n1[k - 1] @ t)
        if np.linalg.norm(v) < 1e-10:
            seed = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = seed - t * (seed @ t)
        v /= np.linalg.norm(v)
        if v @ n1[k - 1] < 0:
            v = -v
        n1[k] = v
        n2k = np.cross(t, n1[k])
        n2k /= np.linalg.norm(n2k)
        if n2k @ n2[k - 1] < 0:
            n2k = -n2k
            n1[k] = -n1[k]
        n2[k] = n2k

    return SO3Path(s_grid=s_grid, R_grid=R_grid, omega_grid=omega, n1_grid=n1, n2_grid=n2, L=L)


def tube_coords(path: SO3Path, R: np.ndarray, s_init: float, max_it: int = 15):
    s = path.wrap_s(s_init)

    def g(s_val: float):
        G = path.gamma_at(s_val)
        e = log_so3(G.T @ R)
        return float(e @ path.omega_at(s_val))

    def gprime_fd(s_val: float):
        h = 1e-4
        s_p = path.wrap_s(s_val + h)
        s_m = path.wrap_s(s_val - h)
        return (g(s_p) - g(s_m)) / max(s_p - s_m, 1e-8)

    for _ in range(max_it):
        gv = g(s)
        if abs(gv) < 1e-10:
            break
        gp = gprime_fd(s)
        step = gv / gp if abs(gp) > 1e-8 else np.sign(gv) * 1e-3
        step = float(np.clip(step, -0.2, 0.2))
        s_new = path.wrap_s(s - step)
        if abs(g(s_new)) > abs(gv):
            s_new = path.wrap_s(s - 0.5 * step)
        s = s_new

    G = path.gamma_at(s)
    e = log_so3(G.T @ R)
    n1, n2 = path.frame_at(s)
    xi = np.array([n1 @ e, n2 @ e])
    return s, xi, e
