# Implementation Report: Numerically Robust Path Following on $\mathrm{SO}(3)$

## 1. Scope and Objectives

This report specifies a production-grade numerical implementation of the paper draft's controller for path following on $\mathrm{SO}(3)$, with emphasis on numerical conditioning, reproducibility, and robust behavior near geometric singular sets (especially angles near $0$ and $\pi$).

The implementation goals are:
1. Preserve the geometric structure of $\mathrm{SO}(3)$ during simulation.
2. Realize tubular path coordinates $(\eta,\xi)$ and their dynamics reliably.
3. Maintain well-conditioned decoupling and feedback linearization.
4. Reproduce and extend the simulation studies outlined in the manuscript.
5. Explicitly include diagnostics for failure modes (projection ambiguity, Jacobian conditioning, tube boundary violation).

---

## 2. Key Design Changes from the Draft (for Stability)

The draft implementation section is mathematically aligned but under-specifies several numerically sensitive parts. The following changes are introduced.

### 2.1 Path generation and interpolation
- **Draft issue**: A first-order piecewise interpolation in $s$ can degrade smoothness of derivatives needed for $M(y)$ and $\dot M(y)$.
- **Change**: Use a dense arc-length grid and **Lie-group-consistent segment interpolation**
  \[
  \gamma(s)=\gamma(s_k)\exp\!\big((s-s_k)\widehat{\omega}_{\gamma,k}\big)
  \]
  with optional cubic smoothing of $\omega_\gamma(s)$ on $\mathbb{R}^3$ before integration.
- **Benefit**: controlled local truncation error and smooth rotational progression.

### 2.2 Normal-frame construction
- **Draft issue**: raw Gram–Schmidt at each node can create frame flips and high-frequency jitter.
- **Change**: build frame by **projected transport + sign continuity correction**:
  1. project previous normal onto current orthogonal complement of tangent,
  2. renormalize,
  3. define second normal by cross product,
  4. enforce sign continuity against prior step.
- **Benefit**: avoids discontinuities that contaminate finite differences and Newton steps.

### 2.3 Log/Exp and left Jacobian numerics
- **Draft issue**: direct formulas with $\sin\theta/\theta$ and $\theta/(2\sin\theta)$ are unstable for $\theta\to 0$; branch sensitivity near $\pi$.
- **Change**:
  - Use series expansions for small angle ($\theta<10^{-6}$).
  - Clamp trace argument for arccos to $[-1,1]$.
  - Use robust branch handling near $\pi$ via symmetric part / axis extraction.
- **Benefit**: avoids NaN/Inf and large roundoff amplification.

### 2.4 Projection to the path (solve for $\eta$)
- **Draft issue**: pure Newton can diverge if initialized poorly or if $g'(s)$ is small.
- **Change**: **safeguarded Newton with fallback to bisection/line-search** over a local bracket around feedforward prediction.
- **Benefit**: globalized convergence and deterministic behavior.

### 2.5 Decoupling matrix and drift
- **Draft issue**: finite-difference $\dot M^{-1}$ can be noisy.
- **Change**:
  - Compute $M$ columns from differential geometry identities.
  - Obtain $\dot M$ analytically where possible; otherwise central differences in $\eta$ only (not time), and chain with $\dot y$.
  - Use `solve(D, rhs)` rather than explicit `inv(D)`.
- **Benefit**: better conditioning and less noise injection into torque.

### 2.6 Integration scheme
- **Draft issue**: classical RK on matrix entries followed by occasional projection can accumulate bias.
- **Change**: integrate kinematics by **Lie-group exponential update** each step:
  \[
  R_{k+1}=R_k\exp(\widehat{\omega}_{k+1/2}\Delta t)
  \]
  and integrate angular dynamics with RK4 or midpoint in body coordinates.
- **Benefit**: maintains orthogonality by construction; SVD re-projection only as safety.

### 2.7 Diagnostics and fail-safe handling
- Add online checks:
  - tube membership $\|\xi\|<\varepsilon$,
  - decoupling condition number,
  - Newton convergence status,
  - minimum distance to cut-locus threshold ($\theta<\pi-\delta$).
- If violated, clip commands and log event for post-analysis.

---

## 3. Mathematical-Numerical Pipeline

At each simulation time step:
1. **State**: $(R,\omega)$.
2. **Project to tube coordinates**:
   - solve $g(s)=\log(\gamma(s)^TR)^\vee\cdot\omega_\gamma(s)=0$ for $\eta$,
   - compute error algebra element $e=\log(\gamma(\eta)^TR)^\vee$,
   - resolve normal coordinates $\xi=[n_1(\eta)^Te,\,n_2(\eta)^Te]^T$.
3. **Velocity coordinates**:
   - use $\omega=M(y)\dot y$, so $\dot y=M^{-1}\omega$.
4. **Auxiliary inputs**:
   - transverse PD: $v_\perp=-K_P\xi-K_D\dot\xi$,
   - tangential velocity-field tracking:
     \[
     v_\parallel=\partial_t\nu_d+\partial_\eta\nu_d\,\dot\eta-k_t(\dot\eta-\nu_d)
     \]
5. **Feedback linearization torque**:
   - $\ddot y=D\tau+f$, $D=M^{-1}J^{-1}$,
   - solve $\tau=D^{-1}(v-f)$ using linear solve.
6. **Integrate dynamics** with Lie update on $R$.

---

## 4. Simulation Plan

### Scenario A (time-varying speed)
- Path: generated from
  \[
  \omega_\gamma(s)=\frac{1}{\sqrt 2}(\cos 4s,\sin 4s,1)
  \]
  over $s\in[0,12]$.
- Target speed: $\nu_d(t)=2.75\sin(0.5t)$.

### Scenario B (position-dependent speed)
- Target speed: $\nu_d(\eta)=1.5-\sin\left(\frac{\pi}{12}\eta\right)$.

### Scenario C (Monte Carlo robustness, 30 trials)
- True inertia: $J_{\text{true}}=J(I+P)$ with symmetric random perturbation entries in $[-0.2,0.2]$.
- Controller uses nominal $J$.
- Report success metrics: final transverse error norm, velocity error RMS, max torque, and any tube-exit count.

---

## 5. Validation Metrics

Primary metrics:
- $\|\xi(t)\|$ convergence rate and final value.
- Tangential velocity error $|\dot\eta-\nu_d|$.
- Invariance check when initialized exactly on path and tangent velocity.

Secondary diagnostics:
- condition number of decoupling matrix $D$,
- Newton iterations per step,
- orthogonality defect $\|R^TR-I\|_F$.

---

## 6. Reproducibility and Packaging

Implementation will include:
- `src/so3.py`: robust Lie-group operators,
- `src/path.py`: path generation, interpolation, frame construction,
- `src/controller.py`: coordinate extraction + controller,
- `src/simulate.py`: experiments and plotting,
- deterministic random seed for Monte Carlo,
- saved figures and summary CSV/JSON under `results/` and `figures/`.

This structure enables direct migration to journal supplementary material and reproducible benchmark scripts.
