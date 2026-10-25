"""
Step 2 -- Check the analytic reproduction number numerically.

Three independent checks of the formulas in model.py:
  (A) closed-form R_v  ==  spectral radius of F V^{-1}, where F and V are
      built by finite-difference Jacobians of the ODE terms (no algebra reused)
  (B) the limiting cases theta = 1 and theta = 0 agree with textbook results
  (C) simulations started just below / just above the critical coverage
      die out / grow, as the threshold theorem predicts

Run:  python src/02_check_reproduction_number.py
"""

import numpy as np
from model import (force_of_infection, next_generation_matrix, reproduction_number,
                   critical_s2, effective_susceptibility, initial_state, simulate)

rng = np.random.default_rng(2026)


def numerical_ngm_radius(beta, theta, sigma, gamma, N1, N2, s1, s2, h=1e-6):
    """Build F and V for x = (E1, E2, I1, I2) by finite differences."""
    def new_infections(x):                      # the F-terms (enter E only)
        E1, E2, I1, I2 = x
        lam1, lam2 = force_of_infection(I1, I2, N1, N2, beta, theta)
        return np.array([lam1 * s1 * N1, lam2 * s2 * N2, 0.0, 0.0])

    def transitions(x):                         # the V-terms (outflow - inflow)
        E1, E2, I1, I2 = x
        return np.array([sigma * E1, sigma * E2,
                         gamma * I1 - sigma * E1, gamma * I2 - sigma * E2])

    def jac(f):
        J = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4); e[j] = h
            J[:, j] = (f(e) - f(-e)) / (2 * h)
        return J

    F, V = jac(new_infections), jac(transitions)
    return max(abs(np.linalg.eigvals(F @ np.linalg.inv(V))))


# ---------------------------------------------------------------- (A)
print("(A) closed form vs numerical next-generation matrix, 1000 random draws")
worst = 0.0
for _ in range(1000):
    R0 = rng.uniform(5, 20)
    gamma, sigma = 1 / rng.uniform(5, 9), 1 / rng.uniform(8, 12)
    theta, p1 = rng.uniform(0, 1), rng.uniform(0.5, 0.99)
    s1, s2 = rng.uniform(0.02, 0.2), rng.uniform(0.05, 0.8)
    N = 1e5; N1, N2 = p1 * N, (1 - p1) * N
    analytic = reproduction_number(R0, theta, p1, s1, s2)
    numeric = numerical_ngm_radius(R0 * gamma, theta, sigma, gamma, N1, N2, s1, s2)
    worst = max(worst, abs(analytic - numeric) / numeric)
print(f"    largest relative difference = {worst:.2e}")

# ---------------------------------------------------------------- (B)
print("\n(B) limiting cases")
R0, p1, s1, s2 = 12.0, 0.9, 0.05, 0.40
r_full = reproduction_number(R0, 0.37, p1, 1.0, 1.0)
print(f"    no immunity anywhere: R_v = {r_full:.6f}   (should equal R0 = {R0})")
print(f"    theta = 1: R_v = {reproduction_number(R0, 1, p1, s1, s2):.6f}  "
      f"max(R0 s1, R0 s2) = {max(R0 * s1, R0 * s2):.6f}")
print(f"    theta = 0: R_v = {reproduction_number(R0, 0, p1, s1, s2):.6f}  "
      f"R0 (p1 s1 + p2 s2) = {R0 * (p1 * s1 + (1 - p1) * s2):.6f}")
print(f"    theta = 1: critical s2 = {critical_s2(R0, 1, p1, s1):.6f}  (1/R0 = {1 / R0:.6f})")

# ---------------------------------------------------------------- (C)
print("\n(C) threshold behaviour in simulation")
p = dict(beta=R0 / 8, theta=0.7, sigma=1 / 10, gamma=1 / 8,
         eps1=0.93, eps2=0.97, N1=0.9e8, N2=0.1e8)      # large N: no depletion early
s1 = effective_susceptibility(0.03, 0.92, p["eps1"], p["eps2"])
s2_star = critical_s2(R0, p["theta"], 0.9, s1)
c2_star = (1 - s2_star) / p["eps2"]
print(f"    group 1: s1 = {s1:.4f};  critical s2 = {s2_star:.4f}  "
      f"-> critical two-dose coverage in group 2 = {c2_star:.1%}")
for label, c2 in (("just below critical coverage", c2_star - 0.02),
                  ("just above critical coverage", c2_star + 0.02)):
    s2 = effective_susceptibility(0, c2, p["eps1"], p["eps2"])
    Rv = reproduction_number(R0, p["theta"], 0.9, s1, s2)
    y0 = initial_state(p["N1"], p["N2"], (0.03, 0.92), (0.0, c2), I0=(0, 10))
    sol = simulate(p, y0, 400, t_eval=[200, 400])
    inf = sol.y[3] + sol.y[4] + sol.y[10] + sol.y[11]   # E1 + I1 + E2 + I2
    trend = "growing" if inf[1] > inf[0] else "dying out"
    print(f"    {label:30s} coverage={c2:.1%}  R_v={Rv:.3f}  -> {trend}")
