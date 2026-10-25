"""
Step 6 -- Sensitivity analysis (Stage 4, part 2).

Three questions:

A. Do the FITTED results depend on the inputs we fixed in Stage 2?
   We refit the Stage 3 model (constant transmission, R0 = 15) with each
   fixed input moved to the low or high end of its plausible range, one at a
   time ("one-at-a-time" or OAT), and once with weekly instead of 2-week data.

B. Which parameters drive R_v and c2* most strongly near the fitted values?
   The "normalised sensitivity index" (elasticity) answers: if parameter x
   rises by 1%, by what % does the output change?
        index = (d output / d x) * (x / output)
   An index of +1 means proportional; 0 means no effect.  Computed by a small
   central difference (x up and down by 1%).

C. Over the whole plausible parameter space, which inputs matter most?
   Latin hypercube sampling (LHS) draws 2,000 parameter sets that cover every
   range evenly.  The partial rank correlation coefficient (PRCC) between an
   input and an output measures their monotone association after removing the
   effect of all other inputs.  |PRCC| near 1 = strong, near 0 = none.

Outputs
  results/sensitivity_oat.csv
  results/elasticities.csv
  results/prcc.csv
  figures/fig_prcc.png

Run from the project folder:   python src/06_sensitivity.py
(A takes 5-15 minutes, running the three outbreaks in parallel; B and C are fast.)
"""

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc, rankdata, t as t_dist

from parameters import OUTBREAKS
from model import reproduction_number, critical_s2
from fitting import biweekly, fit, summaries

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
R0_REF = 15.0

# A. one-at-a-time scenarios: (label, changes to the setting, use weekly data?)
SCENARIOS = [
    ("baseline",               {},                    False),
    ("latent period 8 days",   {"sigma": 1 / 8},      False),
    ("latent period 12 days",  {"sigma": 1 / 12},     False),
    ("infectious period 6 d",  {"gamma": 1 / 6},      False),
    ("infectious period 9 d",  {"gamma": 1 / 9},      False),
    ("two-dose efficacy 0.93", {"eps2": 0.93},        False),
    ("two-dose efficacy 0.99", {"eps2": 0.99},        False),
    ("weekly data",            {"period_days": 7},    True),
]


def base_setting(name):
    e = OUTBREAKS[name]
    return dict(N_total=e["N1"]["value"] + e["N2"]["value"],
                N2eff_bounds=(50, e["N2"]["high"]), c22=e["cov2"]["value"])


def load_weekly(name):
    data = pd.read_csv(ROOT / "data" / "processed" / "outbreaks_weekly.csv")
    return data[data.outbreak == name].cases.values.astype(float)


# ---------------------------------------------------------------- A. OAT refits
def oat(name):
    weekly = load_weekly(name)
    stage3 = pd.read_csv(RESULTS / "fit_estimates.csv").set_index("outbreak").loc[name]
    warm = {k: stage3[k] for k in ("theta", "s1", "N2eff", "I0", "k")}
    rows = []
    for label, changes, use_weekly in SCENARIOS:
        setting = {**base_setting(name), **changes}
        cases = weekly if use_weekly else biweekly(weekly)
        par, ll = fit(cases, setting, fixed={"R0": R0_REF}, n_starts=4, warm_start=warm)
        s = summaries(par, setting, cases)
        rows.append({"outbreak": name, "scenario": label, "loglik": ll,
                     "theta": par["theta"], "N2eff": par["N2eff"], "s1": par["s1"], **s})
        print(f"[{name}] {label}: R_v {s['Rv']:.2f}  h* {100 * s['share_susceptibles']:.1f}%",
              flush=True)
    return rows


# ---------------------------------------------------------------- B. elasticities
def outputs(R0, theta, s1, s2, p2, eps2):
    """R_v and c2* from the closed-form results of Section 2."""
    Rv = reproduction_number(R0, theta, 1 - p2, s1, s2)
    c2 = (1 - critical_s2(R0, theta, 1 - p2, s1)) / eps2
    return Rv, c2


def elasticities(name):
    st = pd.read_csv(RESULTS / "fit_estimates.csv").set_index("outbreak").loc[name]
    setting = base_setting(name)
    x0 = dict(R0=R0_REF, theta=st.theta, s1=st.s1,
              s2=1 - 0.97 * setting["c22"], p2=st.N2eff / setting["N_total"], eps2=0.97)
    base = outputs(**x0)
    rows = []
    for param in x0:
        if x0[param] == 0:                       # e.g. s1 = 0 in South Carolina
            rows.append({"outbreak": name, "parameter": param,
                         "index_Rv": 0.0, "index_c2_star": 0.0})
            continue
        up = outputs(**{**x0, param: x0[param] * 1.01})
        down = outputs(**{**x0, param: x0[param] * 0.99})
        # (change in output / output) divided by (change in x / x) = 2%
        idx = [((u - d) / b) / 0.02 for u, d, b in zip(up, down, base)]
        rows.append({"outbreak": name, "parameter": param,
                     "index_Rv": idx[0], "index_c2_star": idx[1]})
    return rows


# ---------------------------------------------------------------- C. LHS-PRCC
PRCC_RANGES = {            # plausible ranges, from Stages 2-3
    "R0":    (8.0, 25.0),
    "theta": (0.05, 0.5),
    "s1":    (0.0, 0.08),
    "c22":   (0.0, 0.5),
    "p2":    (0.0005, 0.03),
    "eps2":  (0.93, 0.99),
}


def prcc(X, y):
    """Partial rank correlation of each column of X with y."""
    R = np.column_stack([rankdata(col) for col in X.T])
    ry = rankdata(y)
    n, k = R.shape
    result = []
    for j in range(k):
        others = np.column_stack([np.ones(n), np.delete(R, j, axis=1)])
        # remove the linear effect of the other (ranked) inputs ...
        res_x = R[:, j] - others @ np.linalg.lstsq(others, R[:, j], rcond=None)[0]
        res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
        # ... and correlate what is left
        r = np.corrcoef(res_x, res_y)[0, 1]
        t_stat = r * np.sqrt((n - 2 - (k - 1)) / (1 - r ** 2))
        p = 2 * t_dist.sf(abs(t_stat), df=n - 2 - (k - 1))
        result.append((r, p))
    return result


def lhs_prcc(n_samples=2000, seed=2026):
    names = list(PRCC_RANGES)
    lo = np.array([PRCC_RANGES[k][0] for k in names])
    hi = np.array([PRCC_RANGES[k][1] for k in names])
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=seed).random(n_samples), lo, hi)
    Rv, c2 = [], []
    for row in X:
        x = dict(zip(names, row))
        r, c = outputs(x["R0"], x["theta"], x["s1"], 1 - x["eps2"] * x["c22"],
                       x["p2"], x["eps2"])
        Rv.append(r)
        c2.append(min(c, 1.0))
    rows = []
    for out_name, y in (("Rv", np.array(Rv)), ("c2_star", np.array(c2))):
        for name, (r, p) in zip(names, prcc(X, y)):
            rows.append({"output": out_name, "parameter": name, "PRCC": r, "p_value": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    names = list(OUTBREAKS)

    # A
    with Pool(processes=len(names)) as pool:
        oat_rows = [row for rows in pool.map(oat, names) for row in rows]
    oat_df = pd.DataFrame(oat_rows)
    oat_df.to_csv(RESULTS / "sensitivity_oat.csv", index=False)
    print("\nA. ONE-AT-A-TIME REFITS (R0 = 15)")
    for n in names:
        g = oat_df[oat_df.outbreak == n]
        print(f"  {OUTBREAKS[n]['label']}")
        for _, r in g.iterrows():
            print(f"    {r.scenario:24s} R_v {r.Rv:5.2f}   h* {100 * r.share_susceptibles:5.1f}%"
                  f"   c2* {100 * r.c2_star:5.1f}%   susceptibles in network "
                  f"{r.susceptibles_in_network:6.0f}")
        print(f"    range of h*: {100 * g.share_susceptibles.min():.1f}"
              f"-{100 * g.share_susceptibles.max():.1f}%")

    # B
    el = pd.DataFrame([row for n in names for row in elasticities(n)])
    el.to_csv(RESULTS / "elasticities.csv", index=False)
    print("\nB. NORMALISED SENSITIVITY INDICES at the fitted values")
    print(el.pivot(index="parameter", columns="outbreak",
                   values=["index_Rv", "index_c2_star"]).round(2).to_string())

    # C
    pr = lhs_prcc()
    pr.to_csv(RESULTS / "prcc.csv", index=False)
    print("\nC. LHS-PRCC (2,000 samples)")
    print(pr.pivot(index="parameter", columns="output", values="PRCC").round(3).to_string())

    labels = {"R0": r"$\mathcal{R}_0$", "theta": r"$\theta$", "s1": r"$s_1$",
              "c22": r"$c_{2,2}$", "p2": r"$p_2$", "eps2": r"$\varepsilon_2$"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for ax, (out_name, title) in zip(axes, (("Rv", r"$\mathcal{R}_v$"),
                                            ("c2_star", r"critical coverage $c_2^*$"))):
        g = pr[pr.output == out_name]
        colors = ["#08306b" if v > 0 else "#d94801" for v in g.PRCC]
        ax.barh([labels[p] for p in g.parameter], g.PRCC, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("PRCC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.savefig(FIGURES / "fig_prcc.png", dpi=200)
    print("\nSaved results/sensitivity_oat.csv, elasticities.csv, prcc.csv, figures/fig_prcc.png")


if __name__ == "__main__":      # required on Windows when using multiprocessing
    main()
