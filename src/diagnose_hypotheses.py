import json
from pathlib import Path
import numpy as np

from path import integrate_path
from simulate import run_case
from controller import ControllerOptions


def test_hypotheses():
    results = {}

    # Base paths for resolution hypothesis
    path_coarse = integrate_path(L=12.0, N=1500)
    path_fine = integrate_path(L=12.0, N=5000)

    faithful = ControllerOptions()
    clipped = ControllerOptions(regularize=True, alpha_epsilon=1e-8, eta_dot_limit=50.0, torque_limit=300.0, omega_limit=1e3)

    # H1: Boundary interactions explain wavy case degradation
    _, _, _, _, m_h1_boundary = run_case("time", T=12.0, dt=0.005, eta0=0.1, path=path_coarse, options=faithful)
    _, _, _, _, m_h1_interior = run_case("time", T=12.0, dt=0.005, eta0=6.0, path=path_coarse, options=faithful)
    results["H1_boundary_effect"] = {
        "boundary_start": m_h1_boundary,
        "interior_start": m_h1_interior,
        "supports_hypothesis": bool(m_h1_boundary["boundary_hit_fraction"] > m_h1_interior["boundary_hit_fraction"]),
    }

    # H2: Safety clipping changes behavior (faithfulness concern)
    _, _, _, _, m_h2_faithful = run_case("eta", T=15.0, dt=0.005, eta0=0.1, path=path_coarse, options=faithful)
    _, _, _, _, m_h2_clipped = run_case("eta", T=15.0, dt=0.005, eta0=0.1, path=path_coarse, options=clipped)
    results["H2_safety_clipping"] = {
        "faithful": m_h2_faithful,
        "clipped": m_h2_clipped,
        "supports_hypothesis": bool(abs(m_h2_faithful["rms_vel_err"] - m_h2_clipped["rms_vel_err"]) > 1e-6),
    }

    # H3: Resolution impacts derivative-heavy FBL terms
    _, _, _, _, m_h3_coarse = run_case("eta", T=15.0, dt=0.005, eta0=0.1, path=path_coarse, options=faithful)
    _, _, _, _, m_h3_fine = run_case("eta", T=15.0, dt=0.005, eta0=0.1, path=path_fine, options=faithful)
    results["H3_path_resolution"] = {
        "N1500": m_h3_coarse,
        "N5000": m_h3_fine,
        "supports_hypothesis": bool(abs(m_h3_coarse["rms_vel_err"] - m_h3_fine["rms_vel_err"]) > 1e-6),
    }

    # H4: Apparent non-convergence can be finite-horizon effect
    _, _, _, _, m_h4_15 = run_case("eta", T=15.0, dt=0.005, eta0=0.1, path=path_coarse, options=faithful)
    _, _, _, _, m_h4_30 = run_case("eta", T=30.0, dt=0.005, eta0=0.1, path=path_coarse, options=faithful)
    results["H4_horizon_length"] = {
        "T15": m_h4_15,
        "T30": m_h4_30,
        "supports_hypothesis": bool(m_h4_30["final_xi_norm"] < m_h4_15["final_xi_norm"]),
    }

    # Overall strict convergence criterion requested by user
    results["pass_fail_tol_xi_2e-4_tol_ev_1e-3"] = {
        "slowdown_T15": bool(m_h2_faithful["convergence"]["pass"]),
        "wavy_T12": bool(m_h1_interior["convergence"]["pass"]),
    }

    return results


def main():
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    report = test_hypotheses()
    (out / "hypothesis_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
