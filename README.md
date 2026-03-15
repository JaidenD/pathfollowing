# SO(3) Path-Following Numerical Implementation

This repository contains a numerically robust Python implementation of the path-following controller drafted in the paper text.

## Run

```bash
python3 src/simulate.py
python3 src/verify.py
python3 src/diagnose_hypotheses.py
```

Generated at runtime (not committed to git):
- `figures/wavy-path.png`
- `figures/slowdown.png`
- `figures/monte-carlo.png`
- `results/summary.json`
- `results/verification.json`
- `results/hypothesis_diagnostics.json`

## Notes

The implementation includes stabilized SO(3) logarithm/exponential operators, safeguarded projection to tube coordinates, Lie-group attitude integration, and Monte Carlo inertia perturbation testing.


Convergence diagnostics use relaxed practical thresholds: `tol_xi=2e-4`, `tol_ev=1e-3` after settling.
