"""
The two-group model with two controls in the community network, and an
optimal-control solver (Pontryagin's maximum principle, forward-backward sweep).

Controls (both act only in the community network, group 2)
  u1(t)  catch-up vaccination: unvaccinated susceptibles receive a first MMR
         dose at rate u1 per day (S2 -> V12, protection eps1 = 0.93).
  u2(t)  isolation: the fraction of community infectiousness removed by
         isolating cases and quarantining contacts (0 = none, 1 = all).

Doses are counted explicitly.  A campaign cannot tell who is susceptible, so it
reaches every unvaccinated community member who is not currently infectious:
    U = S2 + E2 + R2.
Doses given per day = u1 * U.  Only the doses given to S2 protect anyone; doses
to E2 (already infected) and R2 (already recovered) are wasted.

Objective (minimised over the response period [t0, T]):
    J = integral of [ sigma*(E1 + E2)                    new cases per day
                      + (c_d + eta1/2 * u1) * u1 * U      doses
                      + (c_q + eta2/2 * u2) * u2 * I2     isolation person-days ] dt
c_d and c_q say how many cases one dose / one isolation-day is "worth".
The eta terms make each extra dose (or isolation-day) slightly dearer as the
effort grows - reaching the last people is harder - which makes the optimal
controls smooth so the numerical method converges.  Because they are
multiplied by the number of people reached (U or I2), the optimal control does
not depend on how large the outbreak is.  They are kept small.

The state vector has the same 14 entries, in the same order, as model.py.
"""

import numpy as np

SIGMA, GAMMA = 1 / 10, 1 / 8
EPS1, EPS2 = 0.93, 0.97
iS1, iV11, iV21, iE1, iI1, iR1, iC1, iS2, iV12, iV22, iE2, iI2, iR2, iC2 = range(14)


# ---------------------------------------------------------------- the model
def f(x, u1, u2, p):
    """Right-hand side dx/dt.  Works for real or complex x, and for x of
    shape (14,) or (14, m) (m copies at once), which the Jacobian uses."""
    S1, V11, V21, E1, I1, R1, C1, S2, V12, V22, E2, I2, R2, C2 = x
    N1, N2 = p["N1"], p["N2"]
    N = N1 + N2
    beta, theta, sigma, gamma = p["beta"], p["theta"], SIGMA, GAMMA

    I2e = (1 - u2) * I2                       # infectiousness left after isolation
    lam1 = beta * (theta * I1 / N1 + (1 - theta) * (I1 + I2e) / N)
    lam2 = beta * (theta * I2e / N2 + (1 - theta) * (I1 + I2e) / N)

    inf1 = lam1 * (S1 + (1 - EPS1) * V11 + (1 - EPS2) * V21)
    inf2 = lam2 * (S2 + (1 - EPS1) * V12 + (1 - EPS2) * V22)
    return np.array([
        -lam1 * S1,
        -(1 - EPS1) * lam1 * V11,
        -(1 - EPS2) * lam1 * V21,
        inf1 - sigma * E1,
        sigma * E1 - gamma * I1,
        gamma * I1,
        sigma * E1,
        -lam2 * S2 - u1 * S2,
        u1 * S2 - (1 - EPS1) * lam2 * V12,
        -(1 - EPS2) * lam2 * V22,
        inf2 - sigma * E2,
        sigma * E2 - gamma * I2,
        gamma * I2,
        sigma * E2,
    ])


def jacobian(x, u1, u2, p, h=1e-20):
    """d f / d x by the complex-step method: f(x + i h e_j) = f(x) + i h J e_j + ...
    so J e_j = Im f(x + i h e_j) / h, exact to machine precision (no
    subtraction, so no rounding error)."""
    X = x[:, None] + 1j * h * np.eye(len(x))     # 14 perturbed copies, one per column
    return np.imag(f(X, u1, u2, p)) / h


def unvaccinated_reached(x):
    """U = S2 + E2 + R2: everyone a campaign would vaccinate."""
    return x[iS2] + x[iE2] + x[iR2]


# ---------------------------------------------------------------- integration
def rk4_forward(x0, t, u1, u2, p):
    """Classical 4th-order Runge-Kutta on the grid t; controls given on the
    same grid (linear interpolation at the half step)."""
    x = np.empty((len(t), len(x0)))
    x[0] = x0
    for n in range(len(t) - 1):
        dt = t[n + 1] - t[n]
        a1, a2 = u1[n], u2[n]
        m1, m2 = 0.5 * (u1[n] + u1[n + 1]), 0.5 * (u2[n] + u2[n + 1])
        b1, b2 = u1[n + 1], u2[n + 1]
        k1 = f(x[n], a1, a2, p)
        k2 = f(x[n] + dt / 2 * k1, m1, m2, p)
        k3 = f(x[n] + dt / 2 * k2, m1, m2, p)
        k4 = f(x[n] + dt * k3, b1, b2, p)
        x[n + 1] = x[n] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def running_cost(x, u1, u2, w):
    """Integrand of J at every grid point (x has shape (n, 14))."""
    U = x[:, iS2] + x[:, iE2] + x[:, iR2]
    return (SIGMA * (x[:, iE1] + x[:, iE2])
            + (w["c_d"] + 0.5 * w["eta1"] * u1) * u1 * U
            + (w["c_q"] + 0.5 * w["eta2"] * u2) * u2 * x[:, iI2])


def objective(x, t, u1, u2, w):
    """J by the trapezoidal rule."""
    return np.trapezoid(running_cost(x, u1, u2, w), t)


def adjoint_rhs(lam, x, u1, u2, p, w):
    """d lambda / dt = - dH/dx = - dL/dx - J_f(x)^T lambda."""
    dose_cost = (w["c_d"] + 0.5 * w["eta1"] * u1) * u1        # d L / d U
    iso_cost = (w["c_q"] + 0.5 * w["eta2"] * u2) * u2         # d L / d I2
    dL = np.zeros(14)
    dL[iE1] += SIGMA
    dL[iE2] += SIGMA + dose_cost              # E2 is new cases AND part of U
    dL[iS2] += dose_cost
    dL[iR2] += dose_cost
    dL[iI2] += iso_cost
    return -dL - jacobian(x, u1, u2, p).T @ lam


def rk4_backward(x, t, u1, u2, p, w):
    """Solve the adjoint equations backwards from lambda(T) = 0."""
    lam = np.zeros_like(x)
    for n in range(len(t) - 1, 0, -1):
        dt = t[n] - t[n - 1]
        xm = 0.5 * (x[n] + x[n - 1])
        m1, m2 = 0.5 * (u1[n] + u1[n - 1]), 0.5 * (u2[n] + u2[n - 1])
        k1 = adjoint_rhs(lam[n], x[n], u1[n], u2[n], p, w)
        k2 = adjoint_rhs(lam[n] - dt / 2 * k1, xm, m1, m2, p, w)
        k3 = adjoint_rhs(lam[n] - dt / 2 * k2, xm, m1, m2, p, w)
        k4 = adjoint_rhs(lam[n] - dt * k3, x[n - 1], u1[n - 1], u2[n - 1], p, w)
        lam[n - 1] = lam[n] - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return lam


# ---------------------------------------------------------------- optimality
def switching(x, lam, u1, u2, p, w):
    """The parts of dH/du that do not involve the eta terms:
         dH/du1 = a1 + eta1*u1*U,   dH/du2 = a2 + eta2*u2*I2.
    a1 = c_d*U + (lambda_V12 - lambda_S2)*S2   (from S2 -> V12)
    a2 = c_q*I2 + lambda . (f(u2=1) - f(u2=0)) (f is linear in u2)."""
    a1 = (w["c_d"] * (x[:, iS2] + x[:, iE2] + x[:, iR2])
          + (lam[:, iV12] - lam[:, iS2]) * x[:, iS2])
    a2 = np.empty(len(x))
    for n in range(len(x)):
        df_du2 = f(x[n], u1[n], 1.0, p) - f(x[n], u1[n], 0.0, p)
        a2[n] = w["c_q"] * x[n, iI2] + lam[n] @ df_du2
    return a1, a2


def optimal_control(x0, t, p, w, u1_max, u2_max, max_iter=400, tol=1e-5, relax=0.5,
                    u1_init=0.0, u2_init=0.0):
    """Forward-backward sweep (Lenhart & Workman 2007).
    Repeat: states forward -> adjoints backward -> controls from the
    optimality condition, until the controls stop changing.
    The sweep finds a LOCAL optimum near its starting controls (u1_init,
    u2_init); use best_optimal_control to try several starts."""
    u1 = np.full(len(t), float(u1_init))
    u2 = np.full(len(t), float(u2_init))
    for it in range(1, max_iter + 1):
        x = rk4_forward(x0, t, u1, u2, p)
        lam = rk4_backward(x, t, u1, u2, p, w)
        a1, a2 = switching(x, lam, u1, u2, p, w)
        # dH/du = 0  ->  u = -a / (eta * people reached), kept inside the range
        U = np.maximum(x[:, iS2] + x[:, iE2] + x[:, iR2], 1e-9)
        I2 = np.maximum(x[:, iI2], 1e-9)
        new1 = np.clip(-a1 / (w["eta1"] * U), 0, u1_max)
        new2 = np.clip(-a2 / (w["eta2"] * I2), 0, u2_max)
        change = (np.abs(new1 - u1).sum() / max(np.abs(new1).sum(), 1e-12)
                  + np.abs(new2 - u2).sum() / max(np.abs(new2).sum(), 1e-12))
        u1 = relax * u1 + (1 - relax) * new1
        u2 = relax * u2 + (1 - relax) * new2
        if change < tol:
            break
    x = rk4_forward(x0, t, u1, u2, p)
    lam = rk4_backward(x, t, u1, u2, p, w)
    return dict(u1=u1, u2=u2, x=x, lam=lam, J=objective(x, t, u1, u2, w),
                iterations=it, converged=change < tol, change=change)


def best_optimal_control(x0, t, p, w, u1_max, u2_max):
    """The problem is NOT convex: a weak response only delays infections,
    while a strong one prevents them, so 'do nothing' can be a local optimum
    even when 'act strongly' is far better.  We therefore start the sweep from
    no control, half the maximum, and the maximum, and keep the lowest J."""
    runs = [optimal_control(x0, t, p, w, u1_max, u2_max,
                            u1_init=frac * u1_max, u2_init=frac * u2_max)
            for frac in (0.0, 0.5, 1.0)]
    best = min(runs, key=lambda r: r["J"])
    best["J_all_starts"] = [r["J"] for r in runs]
    return best


# ---------------------------------------------------------------- summaries
def outcomes(x, t, u1, u2):
    """Cases, doses and isolation person-days over the grid."""
    U = x[:, iS2] + x[:, iE2] + x[:, iR2]
    return dict(
        cases=(x[-1, iC1] + x[-1, iC2]) - (x[0, iC1] + x[0, iC2]),
        doses=np.trapezoid(u1 * U, t),
        doses_effective=np.trapezoid(u1 * x[:, iS2], t),   # doses reaching susceptibles
        isolation_days=np.trapezoid(u2 * x[:, iI2], t),
    )
