"""
Step 8 -- Checks of the optimal-control code (Stage 5).

  1. The controlled model with no isolation equals the original model (model.rhs).
  2. The complex-step Jacobian agrees with finite differences.
  3. The adjoint equations derived BY HAND in the paper (Section 6, eq. adjoint)
     agree with the code's adjoint (-dL/dx - J^T psi) at random states.
  4. Gradient check: the adjoint's prediction of how J changes when a control
     is nudged agrees with simply re-running the model (finite differences).
  5. With no control, the RK4 integrator reproduces the Stage 3 South Carolina fit.

Run from the project folder:   python src/08_check_control.py   (about 1 minute)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import rhs
from fitting import run_model
from control import (f, jacobian, adjoint_rhs, rk4_forward, rk4_backward, switching,
                     objective, SIGMA, GAMMA, EPS1, EPS2)

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
W = dict(c_d=0.02, c_q=0.1, eta1=1.0, eta2=0.1)


def adjoint_by_hand(psi, x, u1, u2, p, w):
    """The adjoint equations exactly as written in the paper."""
    (pS1, pV11, pV21, pE1, pI1, pR1, pC1,
     pS2, pV12, pV22, pE2, pI2, pR2, pC2) = psi
    S1, V11, V21, E1, I1, R1, C1, S2, V12, V22, E2, I2, R2, C2 = x
    b, th, N1, N2 = p["beta"], p["theta"], p["N1"], p["N2"]
    N = N1 + N2
    I2e = (1 - u2) * I2
    L1 = b * (th * I1 / N1 + (1 - th) * (I1 + I2e) / N)        # forces of infection
    L2 = b * (th * I2e / N2 + (1 - th) * (I1 + I2e) / N)
    a11, a12 = b * (th / N1 + (1 - th) / N), b * (1 - th) * (1 - u2) / N
    a21, a22 = b * (1 - th) / N, b * (1 - u2) * (th / N2 + (1 - th) / N)
    G1 = (pE1 - pS1) * S1 + (1 - EPS1) * (pE1 - pV11) * V11 + (1 - EPS2) * (pE1 - pV21) * V21
    G2 = (pE2 - pS2) * S2 + (1 - EPS1) * (pE2 - pV12) * V12 + (1 - EPS2) * (pE2 - pV22) * V22
    k1 = (w["c_d"] + 0.5 * w["eta1"] * u1) * u1        # marginal dose cost per person in U
    k2 = (w["c_q"] + 0.5 * w["eta2"] * u2) * u2        # marginal isolation cost per I2
    return np.array([
        L1 * (pS1 - pE1),                                   # psi_S1'
        (1 - EPS1) * L1 * (pV11 - pE1),                     # psi_V11'
        (1 - EPS2) * L1 * (pV21 - pE1),                     # psi_V21'
        -SIGMA + SIGMA * (pE1 - pI1 - pC1),                 # psi_E1'
        GAMMA * (pI1 - pR1) - a11 * G1 - a21 * G2,          # psi_I1'
        0.0,                                                # psi_R1'
        0.0,                                                # psi_C1'
        -k1 + L2 * (pS2 - pE2) + u1 * (pS2 - pV12),         # psi_S2'
        (1 - EPS1) * L2 * (pV12 - pE2),                     # psi_V12'
        (1 - EPS2) * L2 * (pV22 - pE2),                     # psi_V22'
        -SIGMA - k1 + SIGMA * (pE2 - pI2 - pC2),            # psi_E2'
        -k2 + GAMMA * (pI2 - pR2) - a12 * G1 - a22 * G2,    # psi_I2'
        -k1,                                                # psi_R2'
        0.0,                                                # psi_C2'
    ])


def main():
    p = dict(beta=15 * GAMMA, theta=0.19, N1=379580.0, N2=1277.0)
    p_model = dict(p, sigma=SIGMA, gamma=GAMMA, eps1=EPS1, eps2=EPS2)

    # 1
    worst = 0.0
    for _ in range(200):
        x = rng.uniform(0, 1000, 14)
        u1 = rng.uniform(0, 0.02)
        worst = max(worst, np.abs(f(x, u1, 0.0, p) - np.array(rhs(0, x, p_model, (0.0, u1)))).max())
    print(f"1. controlled model vs original (no isolation): largest difference {worst:.1e}")

    # 2
    x = rng.uniform(1, 1000, 14)
    J = jacobian(x, 0.01, 0.3, p)
    Jfd = np.column_stack([(f(x + 1e-4 * e, 0.01, 0.3, p) - f(x - 1e-4 * e, 0.01, 0.3, p)) / 2e-4
                           for e in np.eye(14)])
    print(f"2. complex-step Jacobian vs finite differences: relative error "
          f"{np.abs(J - Jfd).max() / np.abs(J).max():.1e}")

    # 3
    worst = 0.0
    for _ in range(200):
        x = rng.uniform(0, 1000, 14)
        psi = rng.uniform(-2, 2, 14)
        u1, u2 = rng.uniform(0, 0.01), rng.uniform(0, 0.5)
        a = adjoint_rhs(psi, x, u1, u2, p, W)
        b = adjoint_by_hand(psi, x, u1, u2, p, W)
        worst = max(worst, np.abs(a - b).max() / max(np.abs(a).max(), 1e-12))
    print(f"3. hand-derived adjoint (paper) vs code: largest relative difference {worst:.1e}")

    # 4 and 5 use the fitted South Carolina model
    e = pd.read_csv(ROOT / "results" / "fit_estimates.csv").set_index("outbreak").loc["upstate_sc"]
    N2, c22, I0 = e.N2eff, 0.17, e.I0
    p = dict(beta=15 * GAMMA, theta=e.theta, N1=380857 - N2, N2=N2)
    x0 = np.array([0, 0, 0, 0, 0, p["N1"], 0, (1 - c22) * N2 - I0, 0, c22 * N2,
                   I0 / 2, I0 / 2, 0, 0.0])
    t = np.arange(28, 294 + 1e-9, 0.25)
    xd = rk4_forward(x0, np.arange(0, 28 + 1e-9, 0.25),
                     np.zeros(113), np.zeros(113), p)[-1]
    u1, u2 = 0.004 * np.ones(len(t)), 0.2 * np.ones(len(t))
    x = rk4_forward(xd, t, u1, u2, p)
    lam = rk4_backward(x, t, u1, u2, p, W)
    a1, a2 = switching(x, lam, u1, u2, p, W)
    U = x[:, 7] + x[:, 10] + x[:, 12]
    v = np.sin(np.pi * (t - t[0]) / (t[-1] - t[0]))          # a smooth nudge
    for name, d, g, nudge in (("u1", 1e-4, a1 + W["eta1"] * u1 * U, (v, 0 * v)),
                              ("u2", 1e-3, a2 + W["eta2"] * u2 * x[:, 11], (0 * v, v))):
        up = objective(rk4_forward(xd, t, u1 + d * nudge[0], u2 + d * nudge[1], p),
                       t, u1 + d * nudge[0], u2 + d * nudge[1], W)
        dn = objective(rk4_forward(xd, t, u1 - d * nudge[0], u2 - d * nudge[1], p),
                       t, u1 - d * nudge[0], u2 - d * nudge[1], W)
        fd = (up - dn) / (2 * d)
        adj = np.trapezoid(g * v, t)
        print(f"4. gradient check {name}: re-running the model {fd:.4f}, adjoint {adj:.4f}, "
              f"relative error {abs(fd - adj) / abs(fd):.1e}")

    tt = np.arange(0, 238 + 1e-9, 0.25)
    xr = rk4_forward(x0, tt, np.zeros(len(tt)), np.zeros(len(tt)), p)
    e1, e2 = run_model(dict(R0=15, theta=e.theta, s1=0.0, N2eff=N2, I0=I0, k=e.k),
                       dict(N_total=380857, c22=c22), 17)
    print(f"5. no control, South Carolina: RK4 {xr[-1, 6] + xr[-1, 13]:.2f} cases, "
          f"Stage 3 solver {(e1 + e2).sum():.2f} cases")

    # 6. Proposition 4: isolation needed to stop each outbreak, exact vs 1 - 1/R_v
    from model import next_generation_matrix
    fits = pd.read_csv(ROOT / "results" / "fit_estimates.csv").set_index("outbreak")
    totals = {"west_texas": 28_956, "upstate_sc": 380_857, "short_creek": 515_102}
    coverage = {"west_texas": 0.30, "upstate_sc": 0.17, "short_creek": 0.077}
    for name, r in fits.iterrows():
        p1 = 1 - r.N2eff / totals[name]
        K0 = next_generation_matrix(15.0, r.theta, p1, r.s1, 1 - EPS2 * coverage[name])
        def Rv(u2):                        # isolation scales column 2 by (1 - u2)
            K = K0.copy()
            K[:, 1] *= (1 - u2)
            return max(abs(np.linalg.eigvals(K)))
        grid = np.linspace(0, 1, 100001)
        exact = grid[np.argmax([Rv(u) < 1 for u in grid])]
        print(f"6. {name:12s} isolation needed: exact {exact:.3f}, "
              f"Proposition 4 gives 1 - 1/R_v = {1 - 1 / Rv(0):.3f}")

    # 7. The sweep is a local method, so compare its solution (strategy "both")
    #    with a brute-force search over 11 x 11 constant strategies. The optimal
    #    J should be lower than every constant one.  (Needs 07_control.py results.)
    import importlib.util
    from itertools import product
    spec = importlib.util.spec_from_file_location("c7", Path(__file__).parent / "07_control.py")
    c7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c7)
    strategies = pd.read_csv(ROOT / "results" / "control_strategies.csv")
    for name in ("west_texas", "upstate_sc", "short_creek"):
        s = c7.setup(name)
        t = np.arange(0, s["T"] + 1e-9, c7.DT)
        k = int(round(s["t_detect"] / c7.DT))
        x_det = rk4_forward(s["x0"], t[:k + 1], np.zeros(k + 1), np.zeros(k + 1), s["p"])[-1]
        tt = t[k:]
        best = np.inf
        for a, b in product(np.linspace(0, 1, 11), np.linspace(0, 1, 11)):
            u1 = a * c7.U1_MAX * np.ones(len(tt))
            u2 = b * c7.U2_MAX * np.ones(len(tt))
            best = min(best, objective(rk4_forward(x_det, tt, u1, u2, s["p"]), tt, u1, u2,
                                       c7.WEIGHTS))
        J_opt = strategies[(strategies.outbreak == name) & (strategies.strategy == "both")].J.iloc[0]
        print(f"7. {name:12s} optimal J {J_opt:.1f} vs best of 121 constant strategies {best:.1f}")

    # 8. Non-convexity table of the paper (South Carolina, constant isolation from
    #    detection, no vaccination).  Saved for 09_check_paper_numbers.py.
    s = c7.setup("upstate_sc")
    t = np.arange(0, s["T"] + 1e-9, c7.DT)
    k = int(round(s["t_detect"] / c7.DT))
    x_none = rk4_forward(s["x0"], t, np.zeros(len(t)), np.zeros(len(t)), s["p"])
    x_det, tt = x_none[k], t[k:]
    cases_before = x_det[6] + x_det[13]
    rows = []
    from control import outcomes
    for level in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        u1, u2 = np.zeros(len(tt)), level * np.ones(len(tt))
        x = rk4_forward(x_det, tt, u1, u2, s["p"])
        o = outcomes(x, tt, u1, u2)
        rows.append({"u2": level, "cases": cases_before + o["cases"],
                     "isolation_days": o["isolation_days"],
                     "J": objective(x, tt, u1, u2, c7.WEIGHTS)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "nonconvexity_sc.csv", index=False)
    print("8. South Carolina, constant isolation (saved results/nonconvexity_sc.csv):")
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
