"""
Step 4 -- Fit the two-group model to each outbreak (Stage 3).

What this script does, for each of the three outbreaks
  1. MAIN FIT with R0 held at 15 (the middle of the literature range).
  2. R0 PROFILE: hold R0 at 8, 10, ..., 25 and refit everything else.
     If the best log-likelihood barely changes, the data cannot tell these
     R0 values apart (R0 is "not identifiable").  We then check whether the
     quantities we care about (R_v, the coverage the community needs) stay
     the same along the profile.  If they do, they ARE identifiable.
  3. THETA PROFILE at R0 = 15: step theta away from its best value and refit
     the rest, until the fit becomes clearly worse.  The values of R_v and
     c2* inside the 95% region (2 x drop in log-likelihood <= 3.84) give the
     95% confidence intervals.
  4. West Texas only: refit with other values of the (unknown) community
     coverage, to see which results depend on that assumption.

The three outbreaks run in parallel, one per CPU core.  Expect about
10-15 minutes in total.

Outputs
  results/fit_estimates.csv         main estimates with 95% intervals
  results/profile_R0.csv            R0 profile (identifiability evidence)
  results/profile_theta.csv         theta profile (confidence intervals)
  results/texas_coverage_check.csv  West Texas coverage sensitivity
  figures/fig_model_fits.png        data vs fitted model
  figures/fig_profiles.png          R0 profile and what stays stable

Run from the project folder:   python src/04_fit_outbreaks.py
"""

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parameters import OUTBREAKS
from fitting import biweekly, fit, run_model, summaries, DAYS_PER_PERIOD

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

R0_REF = 15.0                               # reference value for the main fit
R0_GRID = [8, 10, 12, 15, 18, 21, 25]       # values used in the R0 profile
CHI2_95 = 3.84                              # 95% cut-off for 2 x log-likelihood drop
TEXAS_COVERAGE_CHECK = [0.10, 0.30, 0.50]


# ---------------------------------------------------------------- settings
def setting_for(name):
    """Fixed, sourced numbers for one outbreak (from parameters.py)."""
    e = OUTBREAKS[name]
    return dict(
        N_total=e["N1"]["value"] + e["N2"]["value"],   # whole outbreak area
        N2eff_bounds=(50, e["N2"]["high"]),           # network: 50 people up to whole community
        c22=e["cov2"]["value"],                       # community two-dose coverage
    )


def load_cases(name):
    data = pd.read_csv(ROOT / "data" / "processed" / "outbreaks_weekly.csv",
                       parse_dates=["week_start"])
    g = data[data.outbreak == name]
    return biweekly(g.cases.values), g.week_start.iloc[0]


# ---------------------------------------------------------------- one outbreak
def analyse(name):
    """All fits for one outbreak. Runs in its own process."""
    cases, start_date = load_cases(name)
    setting = setting_for(name)
    print(f"[{name}] started ({len(cases)} two-week periods, {int(cases.sum())} cases)",
          flush=True)

    # 1. main fit -------------------------------------------------------
    best, best_ll = fit(cases, setting, fixed={"R0": R0_REF}, n_starts=8)
    print(f"[{name}] main fit done: log-lik = {best_ll:.2f}", flush=True)

    # 2. R0 profile -----------------------------------------------------
    r0_rows, warm = [], best
    for R0 in R0_GRID:
        par, ll = fit(cases, setting, fixed={"R0": R0}, n_starts=3, warm_start=warm)
        warm = par
        r0_rows.append({"outbreak": name, "R0": R0, "loglik": ll,
                        **{k: par[k] for k in ("theta", "s1", "N2eff")},
                        **summaries(par, setting, cases)})
    print(f"[{name}] R0 profile done", flush=True)

    # 3. theta profile at R0 = 15 ----------------------------------------
    # Walk outwards from the best theta in 6% steps on each side until the
    # fit is clearly worse (2 x drop > 3.84 + margin) or 20 steps are used.
    th_rows = [{"outbreak": name, "theta": best["theta"], "loglik": best_ll,
                **{k: best[k] for k in ("s1", "N2eff")},
                **summaries(best, setting, cases)}]
    for direction in (+1, -1):
        warm, theta = best, best["theta"]
        for _ in range(20):
            theta = theta * np.exp(direction * 0.06)
            if not 0.005 < theta < 0.995:
                break
            par, ll = fit(cases, setting, fixed={"R0": R0_REF, "theta": theta},
                          n_starts=2, warm_start=warm)
            warm = par
            th_rows.append({"outbreak": name, "theta": theta, "loglik": ll,
                            **{k: par[k] for k in ("s1", "N2eff")},
                            **summaries(par, setting, cases)})
            if 2 * (best_ll - ll) > CHI2_95 + 2:
                break
    print(f"[{name}] theta profile done", flush=True)

    # 4. West Texas coverage check ----------------------------------------
    cov_rows = []
    if name == "west_texas":
        for c22 in TEXAS_COVERAGE_CHECK:
            s = {**setting, "c22": c22}
            par, ll = fit(cases, s, fixed={"R0": R0_REF}, n_starts=4, warm_start=best)
            cov_rows.append({"c22_assumed": c22, "loglik": ll, "theta": par["theta"],
                             "N2eff": par["N2eff"], **summaries(par, s, cases)})
        print(f"[{name}] coverage check done", flush=True)

    e1, e2 = run_model(best, setting, len(cases))
    return dict(name=name, cases=cases, start_date=start_date, best=best,
                best_ll=best_ll, fitted=(e1, e2), r0_rows=r0_rows,
                th_rows=th_rows, cov_rows=cov_rows, setting=setting)


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    names = list(OUTBREAKS)

    with Pool(processes=len(names)) as pool:      # one outbreak per CPU core
        out = {r["name"]: r for r in pool.map(analyse, names)}

    # ---- tables
    r0_prof = pd.DataFrame([row for n in names for row in out[n]["r0_rows"]])
    th_prof = pd.DataFrame([row for n in names for row in out[n]["th_rows"]])
    r0_prof.to_csv(RESULTS / "profile_R0.csv", index=False)
    th_prof.to_csv(RESULTS / "profile_theta.csv", index=False)
    pd.DataFrame(out["west_texas"]["cov_rows"]).to_csv(
        RESULTS / "texas_coverage_check.csv", index=False)

    rows = []
    for n in names:
        o = out[n]
        b, s = o["best"], summaries(o["best"], o["setting"], o["cases"])
        prof = th_prof[th_prof.outbreak == n]
        inside = prof[2 * (o["best_ll"] - prof.loglik) <= CHI2_95]   # 95% region
        row = {"outbreak": n, "loglik": o["best_ll"], "R0_fixed": R0_REF,
               "theta": b["theta"], "s1": b["s1"], "N2eff": b["N2eff"],
               "I0": b["I0"], "k": b["k"], **s,
               "observed_total": o["cases"].sum()}
        for q in ("R0_theta", "Rv", "c2_star", "share_susceptibles", "N2eff"):
            row[f"{q}_lo"], row[f"{q}_hi"] = inside[q].min(), inside[q].max()
        rows.append(row)
    est = pd.DataFrame(rows)
    est.to_csv(RESULTS / "fit_estimates.csv", index=False)

    # ---- printed summary
    print("\nMAIN ESTIMATES (R0 held at 15; 95% profile intervals)")
    for _, r in est.iterrows():
        print(f"  {OUTBREAKS[r.outbreak]['label']}")
        print(f"     R0*theta = {r.R0_theta:5.2f}  [{r.R0_theta_lo:.2f}, {r.R0_theta_hi:.2f}]")
        print(f"     R_v      = {r.Rv:5.2f}  [{r.Rv_lo:.2f}, {r.Rv_hi:.2f}]")
        print(f"     network N2eff = {r.N2eff:6.0f}  [{r.N2eff_lo:.0f}, {r.N2eff_hi:.0f}]")
        print(f"     coverage the network needs c2* = {100 * r.c2_star:5.1f}%  "
              f"[{100 * r.c2_star_lo:.1f}, {100 * r.c2_star_hi:.1f}]")
        print(f"     share of today's susceptibles to immunise = "
              f"{100 * r.share_susceptibles:5.1f}%  "
              f"[{100 * r.share_susceptibles_lo:.1f}, {100 * r.share_susceptibles_hi:.1f}]")
        print(f"     s1 = {r.s1:.4f}   model share of cases in network = "
              f"{100 * r.community_share:.0f}%   predicted total = {r.predicted_total:.0f} "
              f"(observed {r.observed_total:.0f})")

    print("\nR0 PROFILE (identifiability): spread across R0 = 8 ... 25")
    for n in names:
        g = r0_prof[r0_prof.outbreak == n]
        print(f"  {n:12s} log-lik range {g.loglik.max() - g.loglik.min():.2f}   "
              f"R_v {g.Rv.min():.2f}-{g.Rv.max():.2f}   "
              f"c2* {100 * g.c2_star.min():.1f}-{100 * g.c2_star.max():.1f}%   "
              f"theta {g.theta.min():.3f}-{g.theta.max():.3f}")

    print("\nWEST TEXAS: dependence on the assumed community coverage")
    for r in out["west_texas"]["cov_rows"]:
        print(f"  coverage {r['c22_assumed']:.2f}: log-lik {r['loglik']:.2f}  "
              f"R_v {r['Rv']:.2f}  c2* {100 * r['c2_star']:.1f}%  "
              f"share of susceptibles {100 * r['share_susceptibles']:.1f}%")

    # ---- Figure: data vs fits
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for ax, n in zip(axes, names):
        o = out[n]
        dates = o["start_date"] + pd.to_timedelta(
            np.arange(len(o["cases"])) * DAYS_PER_PERIOD, unit="D")
        e1, e2 = o["fitted"]
        ax.bar(dates, o["cases"], width=12, color="#c6dbef", label="reported cases")
        ax.plot(dates, e1 + e2, color="#08306b", lw=2, label="model, total")
        ax.plot(dates, e2, color="#2171b5", lw=1.3, ls="--", label="model, community network")
        ax.set_title(OUTBREAKS[n]["label"], loc="left", fontsize=11)
        ax.set_ylabel("cases per 2 weeks")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(FIGURES / "fig_model_fits.png", dpi=200)

    # ---- Figure: R0 profile
    colors = {"west_texas": "#08306b", "upstate_sc": "#2171b5", "short_creek": "#6baed6"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for n in names:
        g = r0_prof[r0_prof.outbreak == n]
        lab = OUTBREAKS[n]["label"]
        axes[0].plot(g.R0, 2 * (g.loglik.max() - g.loglik), "o-", color=colors[n], label=lab)
        axes[1].plot(g.R0, g.theta, "o-", color=colors[n])
        axes[2].plot(g.R0, 100 * g.c2_star, "o-", color=colors[n])
    axes[0].axhline(CHI2_95, color="#b2182b", ls="--", lw=1)
    axes[0].text(8.2, CHI2_95 + 0.2, "95% cut-off", color="#b2182b", fontsize=8)
    axes[0].set_ylim(-0.2, 6)
    axes[0].set_ylabel(r"$2\times$ drop in log-likelihood")
    axes[1].set_ylabel(r"best-fitting $\theta$")
    axes[2].set_ylabel("coverage the network needs, $c_2^*$ (%)")
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel(r"$\mathcal{R}_0$ (held fixed)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIGURES / "fig_profiles.png", dpi=200)

    print(f"\nSaved results/fit_estimates.csv, profile_R0.csv, profile_theta.csv, "
          f"texas_coverage_check.csv")
    print("Saved figures/fig_model_fits.png, figures/fig_profiles.png")


if __name__ == "__main__":      # required on Windows when using multiprocessing
    main()
