import numpy as np


def hat(v: np.ndarray):
    vx, vy, vz = v
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=float)


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1
        Rproj = U @ Vt
    return Rproj


def exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    Ph = hat(phi)
    I = np.eye(3)
    if theta < 1e-10:
        return I + Ph + 0.5 * (Ph @ Ph)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    return I + A * Ph + B * (Ph @ Ph)


def log_so3(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return 0.5 * vee(R - R.T)

    if np.pi - theta < 1e-6:
        A = 0.5 * (R + np.eye(3))
        idx = int(np.argmax(np.diag(A)))
        axis = np.zeros(3)
        axis[idx] = np.sqrt(max(A[idx, idx], 0.0))
        j = (idx + 1) % 3
        k = (idx + 2) % 3

        if axis[idx] > 1e-8:
            axis[j] = A[j, idx] / axis[idx]
            axis[k] = A[k, idx] / axis[idx]
        else:
            axis = np.array(
                [
                    np.sqrt(max(A[0, 0], 0.0)),
                    np.sqrt(max(A[1, 1], 0.0)),
                    np.sqrt(max(A[2, 2], 0.0)),
                ]
            )
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        skew = 0.5 * (R - R.T)
        if np.dot(vee(skew), axis) < 0:
            axis = -axis
        return theta * axis

    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


def left_jacobian_inv_SO3(e: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(e)
    E = hat(e)
    I = np.eye(3)
    if theta < 1e-6:
        th2 = theta * theta
        A = (1.0 / 12.0) + th2 / 720.0 + (th2 * th2) / 30240.0
        return I + 0.5 * E + A * (E @ E)

    s = np.sin(theta)
    c = np.cos(theta)
    A = (1.0 / (theta * theta)) - (1.0 + c) / (2.0 * theta * s)
    return I + 0.5 * E + A * (E @ E)


def _A_and_Aprime(theta: float):
    if theta < 1e-6:
        th2 = theta * theta
        A = (1.0 / 12.0) + th2 / 720.0 + (th2 * th2) / 30240.0
        Aprime = theta / 360.0 + (theta * th2) / 7560.0 + (theta * th2 * th2) / 201600.0
        return A, Aprime

    s = np.sin(theta)
    c = np.cos(theta)
    A = (1.0 / (theta * theta)) - (1.0 + c) / (2.0 * theta * s)
    num = theta * (s * s) + (1.0 + c) * (s + theta * c)
    Aprime = (-2.0 / (theta**3)) + (num / (2.0 * (theta**2) * (s**2)))
    return A, Aprime


def left_jacobian_inv_dot_SO3(e: np.ndarray, e_dot: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(e)
    E = hat(e)
    E_dot = hat(e_dot)
    A, Aprime = _A_and_Aprime(theta)

    theta_dot = 0.0 if theta < 1e-12 else float(np.dot(e, e_dot) / theta)
    A_dot = Aprime * theta_dot

    E2 = E @ E
    E2_dot = E_dot @ E + E @ E_dot
    return 0.5 * E_dot + A_dot * E2 + A * E2_dot


def orthogonality_error(R: np.ndarray) -> float:
    return np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
