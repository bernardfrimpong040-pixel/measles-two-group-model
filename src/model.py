"""
Two-group SV1V2EIR measles model with preferential mixing.

Groups:  k = 1  the general, well-vaccinated population
         k = 2  the under-vaccinated (high-exemption) community

Compartments in each group k:
  S_k   unvaccinated, susceptible
  V1_k  one MMR dose  (leaky protection, efficacy eps1)
  V2_k  two MMR doses (leaky protection, efficacy eps2)
  E_k   infected, not yet infectious (latent)
  I_k   infectious
  R_k   recovered (immune)
plus C_k, the cumulative number of people who became infectious
(used to compare the model with reported case counts).

Force of infection on group k:
  lambda_k = beta * [ theta * I_k / N_k  +  (1 - theta) * (I_1 + I_2) / N ]
theta = 1 : each group only meets its own members
theta = 0 : everyone mixes at random (homogeneous mixing)

Time unit: days.
"""

import numpy as np
from scipy.integrate import solve_ivp

# order of the state vector
STATE_NAMES = ["S1", "V11", "V21", "E1", "I1", "R1", "C1",
               "S2", "V12", "V22", "E2", "I2", "R2", "C2"]


def force_of_infection(I1, I2, N1, N2, beta, theta):
    N = N1 + N2
    lam1 = beta * (theta * I1 / N1 + (1 - theta) * (I1 + I2) / N)
    lam2 = beta * (theta * I2 / N2 + (1 - theta) * (I1 + I2) / N)
    return lam1, lam2


def rhs(t, y, p, u=(0.0, 0.0)):
    """Right-hand side of the ODE system.

    p : dict with beta, theta, sigma, gamma, eps1, eps2, N1, N2
    u : (u1, u2) catch-up vaccination rates (per day) moving S -> V1.
        Constant here; time-varying controls are added in the
        optimal-control step.
    """
    S1, V11, V21, E1, I1, R1, C1, S2, V12, V22, E2, I2, R2, C2 = y
    lam1, lam2 = force_of_infection(I1, I2, p["N1"], p["N2"], p["beta"], p["theta"])
    e1, e2, sig, gam = p["eps1"], p["eps2"], p["sigma"], p["gamma"]

    out = []
    for S, V1, V2, E, I, lam, uk in ((S1, V11, V21, E1, I1, lam1, u[0]),
                                     (S2, V12, V22, E2, I2, lam2, u[1])):
        new_inf = lam * (S + (1 - e1) * V1 + (1 - e2) * V2)
        dS = -lam * S - uk * S
        dV1 = uk * S - (1 - e1) * lam * V1
        dV2 = -(1 - e2) * lam * V2
        dE = new_inf - sig * E
        dI = sig * E - gam * I
        dR = gam * I
        dC = sig * E                      # flow into I = new infectious cases
        out += [dS, dV1, dV2, dE, dI, dR, dC]
    return out


# ------------------------------------------------- reproduction number
def effective_susceptibility(c1, c2, eps1, eps2):
    """Fraction of a group that is (effectively) susceptible.

    c1 = fraction with exactly one dose, c2 = fraction with two doses.
    s = (1 - c1 - c2) + (1 - eps1) c1 + (1 - eps2) c2 = 1 - eps1 c1 - eps2 c2
    """
    return 1.0 - eps1 * c1 - eps2 * c2


def next_generation_matrix(R0, theta, p1, s1, s2):
    """K[k, j] = expected new infections in group k caused by ONE
    infectious person in group j, at the start of the outbreak.

    K = R0 * diag(s) * (theta * I + (1 - theta) * p 1^T),   R0 = beta / gamma
    """
    p = np.array([p1, 1 - p1])
    s = np.array([s1, s2])
    M = theta * np.eye(2) + (1 - theta) * np.outer(p, np.ones(2))
    return R0 * np.diag(s) @ M


def reproduction_number(R0, theta, p1, s1, s2):
    """Largest eigenvalue of K, written out in closed form for a 2x2 matrix."""
    K = next_generation_matrix(R0, theta, p1, s1, s2)
    a, b, c, d = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
    return 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b * c))


def critical_s2(R0, theta, p1, s1):
    """Largest susceptible fraction in group 2 that still gives R_v < 1
    (given group 1's susceptible fraction s1).  Returns 0 if group 1 alone
    is already above threshold (no value of s2 can save it)."""
    p2 = 1 - p1
    a11 = R0 * s1 * (theta + (1 - theta) * p1)
    if a11 >= 1:
        return 0.0
    denom = R0 * (theta + (1 - theta) * p2) * (1 - a11) \
        + R0 ** 2 * s1 * (1 - theta) ** 2 * p1 * p2
    return min(1.0, (1 - a11) / denom)


def critical_coverage_group2(R0, theta, p1, s1, eps2):
    """Two-dose MMR coverage needed in group 2 for R_v < 1."""
    return (1 - critical_s2(R0, theta, p1, s1)) / eps2


# ------------------------------------------------- simulation helper
def initial_state(N1, N2, cov1, cov2, E0=(0.0, 0.0), I0=(0.0, 1.0)):
    """cov_k = (one-dose fraction, two-dose fraction) in group k."""
    y = []
    for N, (c1, c2), e0, i0 in ((N1, cov1, E0[0], I0[0]), (N2, cov2, E0[1], I0[1])):
        V1, V2 = c1 * N, c2 * N
        S = N - V1 - V2 - e0 - i0
        y += [S, V1, V2, e0, i0, 0.0, i0]
    return np.array(y, dtype=float)


def simulate(p, y0, t_end, u=(0.0, 0.0), t_eval=None):
    return solve_ivp(rhs, (0, t_end), y0, args=(p, u), method="LSODA",
                     t_eval=t_eval, rtol=1e-8, atol=1e-8)
