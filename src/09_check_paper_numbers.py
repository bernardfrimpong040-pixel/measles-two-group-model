"""
Step 9 -- Check that the numbers printed in the paper match the results files.

Every table in paper/main.tex (and the key numbers quoted in the text) is
rebuilt here from the CSV files in results/, formatted exactly as in the
paper, and searched for in main.tex.  A line that is not found means the
paper and the code disagree, and must be fixed before submission.

Run from the project folder, after scripts 01-08:
    python src/09_check_paper_numbers.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
TEX = " ".join(TEX.split())          # line breaks in the source do not matter
ORDER = ["west_texas", "upstate_sc", "short_creek"]

results = []          # (what is checked, expected text, found?)


def check(label, text, condition=True):
    """Pass if the text appears in main.tex (and an extra condition holds)."""
    text = " ".join(text.split())
    results.append((label, text, bool(condition) and text in TEX))


# ---------------------------------------------------------------- formatting
def comma(x):
    """1277 -> '1{,}277' as written in LaTeX."""
    s = f"{int(round(x)):,}"
    return s.replace(",", "{,}")


def f(x, d):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ---------------------------------------------------------------- data (Table 1)
weekly = pd.read_csv(ROOT / "data" / "processed" / "outbreaks_weekly.csv")
for n in ORDER:
    g = weekly[weekly.outbreak == n]
    check(f"data table, {n}", f"& {len(g)} & {int(g.cases.sum())} &")

# ---------------------------------------------------------------- fit (Table 3)
e = pd.read_csv(R / "fit_estimates.csv").set_index("outbreak").loc[ORDER]


def row(name, cells):
    return name + " & " + " & ".join(cells) + r"\\"


def ci(q, d, scale=1.0, unit=""):
    return [f"{scale * r[q]:.{d}f}{unit} [{scale * r[q + '_lo']:.{d}f}, {scale * r[q + '_hi']:.{d}f}]"
            for _, r in e.iterrows()]


check("fit table, R_v", row(r"$\mathcal R_v$ (start of outbreak)", ci("Rv", 2)))
check("fit table, R0*theta", row(r"$\mathcal R_0\theta$", ci("R0_theta", 2)))
check("fit table, network size", row(r"Network size $N_2^{\mathrm{eff}}$",
      [f"{comma(r.N2eff)} [{comma(r.N2eff_lo)}, {comma(r.N2eff_hi)}]" for _, r in e.iterrows()]))
check("fit table, susceptibles in network",
      row(r"Susceptibles in network $N_2^{\mathrm{eff}} s_2$",
          [comma(r.susceptibles_in_network) for _, r in e.iterrows()]))
c2 = [f"{100 * r.c2_star:.1f}\\% [{100 * r.c2_star_lo:.1f}, {100 * r.c2_star_hi:.1f}]"
      for _, r in e.iterrows()]
c2[0] += r"$^\dagger$"
check("fit table, c2*", row(r"Coverage needed $c_2^*$", c2))
check("fit table, h*", row(r"Share to immunise $h^*$",
      [f"{100 * r.share_susceptibles:.1f}\\% [{100 * r.share_susceptibles_lo:.1f}, "
       f"{100 * r.share_susceptibles_hi:.1f}]" for _, r in e.iterrows()]))
check("fit table, s1", row(r"$s_1$", [f(r.s1, 3) for _, r in e.iterrows()]))
check("fit table, community share", row("Model share of cases in network",
      [f"{100 * r.community_share:.0f}\\%" for _, r in e.iterrows()]))
check("fit table, cases", row("Cases: model / reported",
      [f"{r.predicted_total:.0f} / {r.observed_total:.0f}" for _, r in e.iterrows()]))
check("fit table, dispersion", row("Dispersion $k$",
      [f(r.k, 1) if r.k >= 10 else f(r.k, 2) for _, r in e.iterrows()]))
check("fit table, log-likelihood", row(r"Log-likelihood $\hat\ell$",
      [f"${r.loglik:.2f}$" for _, r in e.iterrows()]))

# ---------------------------------------------------------------- R0 profile (text)
prof = pd.read_csv(R / "profile_R0.csv")
spread = max(g.loglik.max() - g.loglik.min() for _, g in prof.groupby("outbreak"))
check("R0 profile: largest log-likelihood change", f"changes by at most {spread:.2f}")

# ---------------------------------------------------------------- Texas coverage check
tx = pd.read_csv(R / "texas_coverage_check.csv")
check("Texas coverage check, c2* range",
      f"{100 * tx.c2_star.min():.1f}\\% or {100 * tx.c2_star.max():.1f}\\%, while $h^*$ is "
      f"{100 * tx.share_susceptibles.iloc[0]:.1f}\\% or {100 * tx.share_susceptibles.iloc[-1]:.1f}\\%")
check("Texas coverage check, network sizes",
      f"{comma(tx.N2eff.iloc[0])}, {comma(tx.N2eff.iloc[1])} and {comma(tx.N2eff.iloc[2])} people")

# ---------------------------------------------------------------- time-varying (Table 4)
tv = pd.read_csv(R / "time_varying_models.csv", parse_dates=["change_date"])
for _, r in tv.iterrows():
    text = f"& {r.model} & ${r.loglik:.2f}$ & {r.AIC:.1f} & {r.dAIC:.1f} &"
    check(f"model comparison, {r.outbreak} {r.model}", text)
    if r.model != "H1":
        d = r.change_date
        check(f"model comparison, {r.outbreak} {r.model} change",
              f"$\\times{r.m:.2f}" + ("^c" if r.m <= 0.05 + 1e-9 or r.m >= 5 - 1e-9 else "")
              + f"$ from {d.day} {d.strftime('%b')}")

# ---------------------------------------------------------------- sensitivity (Table 5)
oat = pd.read_csv(R / "sensitivity_oat.csv")
labels = {"baseline": "Baseline", "latent period 8 days": "Latent period 8 days",
          "latent period 12 days": "Latent period 12 days",
          "infectious period 6 d": "Infectious period 6 days",
          "infectious period 9 d": "Infectious period 9 days",
          "two-dose efficacy 0.93": "Two-dose efficacy 0.93",
          "two-dose efficacy 0.99": "Two-dose efficacy 0.99", "weekly data": "Weekly data"}
for scen, label in labels.items():
    cells = []
    for n in ORDER:
        r = oat[(oat.outbreak == n) & (oat.scenario == scen)].iloc[0]
        cells += [f(r.Rv, 2), f"{100 * r.share_susceptibles:.1f}\\%"]
    check(f"sensitivity table, {scen}", row(label, cells))

el = pd.read_csv(R / "elasticities.csv")
idx = {n: el[(el.outbreak == n) & (el.parameter == "R0")].index_c2_star.iloc[0] for n in ORDER}
check("elasticities of c2* (text)",
      f"the computed indices are {idx['short_creek']:.2f}, {idx['upstate_sc']:.2f} and "
      f"{idx['west_texas']:.2f}")

pr = pd.read_csv(R / "prcc.csv").set_index(["output", "parameter"]).PRCC
check("PRCC, R_v ~ theta", f"$\\theta$ (PRCC {pr['Rv', 'theta']:.2f})")
check("PRCC, R_v ~ R0", f"$\\mathcal R_0$ ({pr['Rv', 'R0']:.2f})")
check("PRCC, R_v ~ c22", f"$c_{{2,2}}$ (${pr['Rv', 'c22']:.2f}$)")
check("PRCC, c2* ~ R0", f"$\\mathcal R_0$ ({pr['c2_star', 'R0']:.2f})")
check("PRCC, c2* ~ s1", f"$s_1$ ({pr['c2_star', 's1']:.2f})")

# ---------------------------------------------------------------- control (Tables 6-7)
st = pd.read_csv(R / "control_strategies.csv")
for n in ORDER:
    g = st[st.outbreak == n]
    check(f"control table, {n} no response", f"& no response & {g.cases_no_response.iloc[0]:.0f} &")
    for _, r in g.iterrows():
        doses = "0" if r.doses == 0 else f"{r.doses:.0f} ({r.doses_wasted:.0f})"
        text = (f"& {r.strategy} & {r.cases:.0f} & {r.averted_pct:.1f}\\% & {doses} & "
                f"{comma(r.isolation_days)}\\\\")
        check(f"control table, {n} {r.strategy}", text)

tm = pd.read_csv(R / "control_timing.csv")
names = {"west_texas": "West Texas", "upstate_sc": "Upstate SC", "short_creek": "Short Creek"}
for n in ORDER:
    g = tm[(tm.outbreak == n) & (tm.strategy == "both")].set_index("delay_days").averted_pct
    check(f"timing table, {n}",
          row(names[n], [f"{g[d]:.1f}\\%" for d in (0, 7, 14, 28, 56)]))

sc = pd.read_csv(R / "control_scenarios.csv")
scen_labels = {
    "observed uptake (1.4e-4/day)": r"Observed uptake, $u_1 = 1.4\times10^{-4}$ per day",
    "observed uptake + isolation 0.5": r"Observed uptake + isolation at rash onset ($u_2 = 0.5$)",
    "moderate vaccination 0.003/day": r"Moderate vaccination, $u_1 = 0.003$ per day",
    "intensive vaccination 0.01/day": r"Intensive vaccination, $u_1 = 0.01$ per day",
    "isolation + quarantine 0.8": r"Isolation + quarantine of contacts ($u_2 = 0.8$)",
}
for key, label in scen_labels.items():
    cells = [f"{sc[(sc.outbreak == n) & (sc.scenario == key)].averted_pct.iloc[0]:.1f}\\%"
             for n in ORDER]
    check(f"scenario table, {key}", row(label, cells))
const = [sc[(sc.outbreak == n) & (sc.scenario == "intensive vaccination 0.01/day")].doses.iloc[0]
         for n in ORDER]
check("constant-campaign doses (text)", f"{comma(const[0])}, {comma(const[1])} and {comma(const[2])} doses")
opt = [st[(st.outbreak == n) & (st.strategy == "vaccination only")].doses.iloc[0] for n in ORDER]
saved = [100 * (1 - o / c) for o, c in zip(opt, const)]
check("doses saved by stopping (text)",
      f"saves {saved[0]:.0f}\\%, {saved[1]:.0f}\\% and {saved[2]:.0f}\\% of the doses")

prof = pd.read_csv(R / "control_profiles.csv", parse_dates=["date"])
for n in ORDER:
    g = prof[(prof.outbreak == n) & (prof.strategy == "vaccination only") & (prof.u1 > 1e-4)]
    d = g.date.iloc[-1]
    check(f"vaccination stop date, {n}", f"{d.day}~{d.strftime('%B')}~{d.year}")

fr = pd.read_csv(R / "control_frontier.csv")
g = fr[fr.outbreak == "upstate_sc"].sort_values("c_d")
check("dose frontier, South Carolina (text)",
      f"{g.averted_pct.iloc[0]:.1f}\\% to {g.averted_pct.iloc[-1]:.1f}\\%")

# ---------------------------------------------------------------- non-convexity table
nc = pd.read_csv(R / "nonconvexity_sc.csv")
check("non-convexity table, cases", row("Cases", [comma(v) for v in nc.cases]))
check("non-convexity table, isolation days",
      row("Isolation person-days", [comma(v) for v in nc.isolation_days]))
check("non-convexity table, J", row("$J$", [f"{v:.0f}" for v in nc.J]))

# ---------------------------------------------------------------- under-reporting
ur = pd.read_csv(R / "under_reporting.csv").set_index(["outbreak", "multiplier"])
check("under-reporting (text)",
      f"$\\mathcal R_v = {ur.loc[('upstate_sc', 2)].Rv:.2f}$ (against "
      f"{ur.loc[('upstate_sc', 1)].Rv:.2f}) and {ur.loc[('short_creek', 3)].Rv:.2f} (against "
      f"{ur.loc[('short_creek', 1)].Rv:.2f})")
check("under-reporting networks (text)",
      f"from {comma(ur.loc[('upstate_sc', 1)].N2eff)} to {comma(ur.loc[('upstate_sc', 2)].N2eff)} "
      f"and from {comma(ur.loc[('short_creek', 1)].N2eff)} to {comma(ur.loc[('short_creek', 3)].N2eff)} people")

# ---------------------------------------------------------------- abstract and discussion
Rv = [e.loc[n].Rv for n in ORDER]
h = [100 * e.loc[n].share_susceptibles for n in ORDER]
check("abstract: R_v", f"were {Rv[0]:.2f}, {Rv[1]:.2f} and {Rv[2]:.2f}")
check("abstract: h*", f"{h[0]:.0f}\\%, {h[1]:.0f}\\% and {h[2]:.0f}\\% of each community")
check("discussion: R_v", f"({Rv[0]:.1f}, {Rv[1]:.1f} and {Rv[2]:.1f})")
check("discussion: h*", f"({h[0]:.0f}\\%, {h[1]:.0f}\\% and {h[2]:.0f}\\%)")
check("abstract: network sizes (a few hundred to about 1,300)",
      "a few hundred to about 1{,}300 people",
      e.N2eff.min() > 300 and round(e.N2eff.max(), -2) == 1300)
obs = sc[sc.scenario == "observed uptake (1.4e-4/day)"].averted_pct
check("abstract: observed uptake averts under 4%", "under 4\\% of cases", obs.max() < 4)
check("discussion: observed uptake range", f"{obs.min():.1f}--{obs.max():.1f}\\%")
iq = sc[sc.scenario == "isolation + quarantine 0.8"].averted_pct
check("abstract: isolation + quarantine 90-99%", "90--99\\%",
      round(iq.min()) == 90 and round(iq.max()) == 99)
g = tm[(tm.outbreak == "west_texas") & (tm.strategy == "both")].set_index("delay_days").averted_pct
per_week = (g[0] - g[28]) / 4
check("abstract: about ten points per week of delay (West Texas)",
      "about ten percentage points", 9 <= per_week <= 11)

# ---------------------------------------------------------------- report
bad = [r for r in results if not r[2]]
print(f"{len(results) - len(bad)} of {len(results)} checks passed\n")
for label, text, ok in results:
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
if bad:
    print("\nNot found in main.tex (expected text):")
    for label, text, _ in bad:
        print(f"  {label}:\n      {text}")
