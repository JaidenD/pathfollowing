import numpy as np

_EPS = 1e-12


def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def sinc(x: float) -> float:
    ax = abs(x)
    if ax < 1e-6:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return np.sin(x) / x


def one_minus_cos_over_x2(x: float) -> float:
    ax = abs(x)
    if ax < 1e-6:
        x2 = x * x
        return 0.5 - x2 / 24.0 + x2 * x2 / 720.0
    return (1.0 - np.cos(x)) / (x * x)


def exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    K = hat(phi)
    return np.eye(3) + sinc(theta) * K + one_minus_cos_over_x2(theta) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-7:
        return vee(0.5 * (R - R.T))

    if np.pi - theta < 1e-5:
        # Robust axis extraction near pi
        A = (R + np.eye(3)) * 0.5
        axis = np.zeros(3)
        idx = np.argmax(np.diag(A))
        axis[idx] = np.sqrt(max(A[idx, idx], 0.0))
        if axis[idx] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            j = (idx + 1) % 3
            k = (idx + 2) % 3
            axis[j] = A[idx, j] / axis[idx]
            axis[k] = A[idx, k] / axis[idx]
        nrm = np.linalg.norm(axis)
        if nrm < _EPS:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / nrm
        return theta * axis

    return vee((theta / (2.0 * np.sin(theta))) * (R - R.T))


def project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj


def left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    cot_half = np.cos(half) / np.sin(half)
    a = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * K + a * (K @ K)


def orthogonality_error(R: np.ndarray) -> float:
    return np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
