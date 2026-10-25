"""
Step 7 -- Control analysis (Stage 5): what response would have worked, and when?

Uses the Stage 3 fits (constant transmission, R0 = 15, network size estimated)
and adds two controls in the community network (see control.py):
  u1 = catch-up vaccination rate (per day), u2 = share of infectiousness removed
       by isolation.

Response limits
  u1 <= 0.01 per day : an intensive campaign (about 26% of the unvaccinated
                        per month).  ASSUMPTION, varied in part C.
  u2 <= 0.5          : isolating cases at rash onset.  People are infectious
                        from 4 days before to 4 days after the rash (CDC), so
                        isolation at the rash removes at most 4 of 8 days.
  u2  = 0.8          : isolation plus quarantine of exposed contacts
                        (ASSUMPTION, part C only).
Observed uptake (part C): at the South Carolina community's charter school, the
share up to date rose from 17% (2024-25) to 21% (2025-26), i.e. 4 of every 83
unvaccinated students in a year:  u1 = -ln(1 - 0.04/0.83) / 365 = 1.4e-4 per day.

The response starts at DETECTION: the end of the first week in which
cumulative cases reach 3 (CDC: an outbreak is 3 or more related cases).

Parts
  A. Optimal strategies from detection (vaccination only, isolation only, both)
  B. Timing: the same strong response started earlier or later than detection
  C. Realistic scenarios: observed uptake; isolation with quarantine
  D. Dose frontier: cases averted against doses, as doses get dearer

Outputs
  results/control_strategies.csv, control_timing.csv, control_scenarios.csv,
  control_frontier.csv
  figures/fig_optimal_controls.png, fig_control_timing.png, fig_dose_frontier.png

Run from the project folder:   python src/07_control.py      (15-30 minutes)
"""

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from parameters import OUTBREAKS
from control import (rk4_forward, best_optimal_control, outcomes, objective,
                     iC1, iC2, SIGMA, iE1, iE2)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

R0_REF, GAMMA = 15.0, 1 / 8
DT = 0.5                                    # time step (days); 0.25 changes totals by < 0.3 cases
HORIZON_DAYS = 1095                         # 3 years: long enough for every outbreak to end.
# A shorter horizon is misleading: a response that holds R_v just above 1 turns
# the outbreak into a slow burn whose later cases fall outside the window.
# (South Carolina, isolation 0.5: 190 cases counted by day 294, 350 in total.)
PLOT_DAYS = 900                             # how much of the horizon the figure shows
U1_MAX, U2_MAX = 0.01, 0.5
U1_OBSERVED = -np.log(1 - 0.04 / 0.83) / 365
U2_QUARANTINE = 0.8
WEIGHTS = dict(c_d=0.02, c_q=0.1, eta1=1.0, eta2=0.1)   # 50 doses or 10 isolation-days = 1 case
DELAYS = list(range(-14, 99, 7))            # response start relative to detection (days)
DOSE_WEIGHTS = [0.002, 0.01, 0.05, 0.1, 0.2, 0.5]


# ---------------------------------------------------------------- set-up
def setup(name):
    """Fitted parameters, initial state, detection day and horizon."""
    e = OUTBREAKS[name]
    fit = pd.read_csv(RESULTS / "fit_estimates.csv").set_index("outbreak").loc[name]
    N_total = e["N1"]["value"] + e["N2"]["value"]
    N2 = fit.N2eff
    N1 = N_total - N2
    c22, s1, I0 = e["cov2"]["value"], fit.s1, fit.I0
    p = dict(beta=R0_REF * GAMMA, theta=fit.theta, N1=N1, N2=N2)
    x0 = np.array([s1 * N1, 0, 0, 0, 0, (1 - s1) * N1, 0,
                   (1 - c22) * N2 - I0, 0, c22 * N2, I0 / 2, I0 / 2, 0, 0.0])

    data = pd.read_csv(ROOT / "data" / "processed" / "outbreaks_weekly.csv",
                       parse_dates=["week_start"])
    g = data[data.outbreak == name].reset_index(drop=True)
    first_week = g.index[g.cumulative >= 3][0]
    t_detect = 7.0 * (first_week + 1)               # end of that week
    T = float(HORIZON_DAYS)
    return dict(p=p, x0=x0, t_detect=t_detect, T=T, start_date=g.week_start[0])


def simulate_constant(s, start, u1, u2):
    """Response at constant strength (u1, u2) from day 'start' to the end."""
    t = np.arange(0, s["T"] + 1e-9, DT)
    on = (t >= start).astype(float)
    x = rk4_forward(s["x0"], t, u1 * on, u2 * on, s["p"])
    return t, x, u1 * on, u2 * on


def total_cases(x):
    return x[-1, iC1] + x[-1, iC2]


# ---------------------------------------------------------------- one outbreak
def analyse(name):
    s = setup(name)
    t_all = np.arange(0, s["T"] + 1e-9, DT)
    zero = np.zeros(len(t_all))
    x_none = rk4_forward(s["x0"], t_all, zero, zero, s["p"])
    cases_none = total_cases(x_none)

    # state at detection; the response period starts there
    pre = t_all[t_all <= s["t_detect"] + 1e-9]
    x_detect = x_none[len(pre) - 1]
    cases_pre = x_detect[iC1] + x_detect[iC2]
    t = np.arange(s["t_detect"], s["T"] + 1e-9, DT)

    # A. optimal strategies ------------------------------------------------
    strat_rows, profiles = [], {}
    for label, u1m, u2m in (("vaccination only", U1_MAX, 0.0),
                            ("isolation only", 0.0, U2_MAX),
                            ("both", U1_MAX, U2_MAX)):
        r = best_optimal_control(x_detect, t, s["p"], WEIGHTS, u1m, u2m)
        o = outcomes(r["x"], t, r["u1"], r["u2"])
        cases = cases_pre + o["cases"]
        averted = cases_none - cases
        strat_rows.append({"outbreak": name, "strategy": label, "cases": cases,
                           "cases_no_response": cases_none, "averted": averted,
                           "averted_pct": 100 * averted / cases_none,
                           "doses": o["doses"], "doses_wasted": o["doses"] - o["doses_effective"],
                           "isolation_days": o["isolation_days"],
                           "doses_per_case_averted": o["doses"] / averted if o["doses"] > 0 else 0,
                           "J": r["J"], "converged": r["converged"],
                           "J_by_start": "/".join(f"{j:.1f}" for j in r["J_all_starts"])})
        profiles[label] = (t, r["u1"], r["u2"], r["x"])
    print(f"[{name}] A. optimal strategies done", flush=True)

    # B. timing -------------------------------------------------------------
    timing_rows = []
    for delay in DELAYS:
        start = max(s["t_detect"] + delay, 1.0)
        for label, u1, u2 in (("vaccination only", U1_MAX, 0.0),
                              ("isolation only", 0.0, U2_MAX),
                              ("both", U1_MAX, U2_MAX)):
            _, x, _, _ = simulate_constant(s, start, u1, u2)
            c = total_cases(x)
            timing_rows.append({"outbreak": name, "delay_days": delay, "strategy": label,
                                "cases": c, "averted_pct": 100 * (cases_none - c) / cases_none})
    print(f"[{name}] B. timing done", flush=True)

    # C. realistic scenarios -----------------------------------------------------
    scen_rows = []
    for label, u1, u2 in (("observed uptake (1.4e-4/day)", U1_OBSERVED, 0.0),
                          ("observed uptake + isolation 0.5", U1_OBSERVED, U2_MAX),
                          ("isolation + quarantine 0.8", 0.0, U2_QUARANTINE),
                          ("intensive vaccination 0.01/day", U1_MAX, 0.0),
                          ("moderate vaccination 0.003/day", 0.003, 0.0)):
        tt, x, a1, a2 = simulate_constant(s, s["t_detect"], u1, u2)
        o = outcomes(x, tt, a1, a2)
        c = total_cases(x)
        scen_rows.append({"outbreak": name, "scenario": label, "cases": c,
                          "averted_pct": 100 * (cases_none - c) / cases_none,
                          "doses": o["doses"], "isolation_days": o["isolation_days"]})
    print(f"[{name}] C. scenarios done", flush=True)

    # D. dose frontier (vaccination only) ------------------------------------------
    front_rows = []
    for c_d in DOSE_WEIGHTS:
        r = best_optimal_control(x_detect, t, s["p"], {**WEIGHTS, "c_d": c_d}, U1_MAX, 0.0)
        o = outcomes(r["x"], t, r["u1"], r["u2"])
        c = cases_pre + o["cases"]
        front_rows.append({"outbreak": name, "c_d": c_d, "doses": o["doses"],
                           "cases": c, "averted": cases_none - c,
                           "averted_pct": 100 * (cases_none - c) / cases_none})
    print(f"[{name}] D. dose frontier done", flush=True)

    return dict(name=name, strat=strat_rows, timing=timing_rows, scen=scen_rows,
                front=front_rows, profiles=profiles, t_all=t_all, x_none=x_none,
                t_detect=s["t_detect"], start_date=s["start_date"])


# ---------------------------------------------------------------- main
def main():
    names = list(OUTBREAKS)
    with Pool(processes=len(names)) as pool:
        out = {r["name"]: r for r in pool.map(analyse, names)}

    strat = pd.DataFrame([r for n in names for r in out[n]["strat"]])
    timing = pd.DataFrame([r for n in names for r in out[n]["timing"]])
    scen = pd.DataFrame([r for n in names for r in out[n]["scen"]])
    front = pd.DataFrame([r for n in names for r in out[n]["front"]])
    strat.to_csv(RESULTS / "control_strategies.csv", index=False)

    # optimal control profiles, one row per day (for tables and checking)
    prof_rows = []
    for n in names:
        for label, (t, u1, u2, x) in out[n]["profiles"].items():
            step = int(round(1 / DT))
            for i in range(0, len(t), step):
                prof_rows.append({"outbreak": n, "strategy": label, "day": t[i],
                                  "date": (out[n]["start_date"] + pd.Timedelta(days=float(t[i]))).date(),
                                  "u1": u1[i], "u2": u2[i],
                                  "new_cases": SIGMA * (x[i, iE1] + x[i, iE2])})
    profiles = pd.DataFrame(prof_rows)
    profiles.to_csv(RESULTS / "control_profiles.csv", index=False)
    print("\nWHEN THE OPTIMAL CAMPAIGN STOPS (last day with u1 above 1% of its maximum)")
    for n in names:
        for label in ("vaccination only", "both"):
            g = profiles[(profiles.outbreak == n) & (profiles.strategy == label)]
            on = g[g.u1 > 0.01 * U1_MAX]
            if len(on):
                print(f"  {OUTBREAKS[n]['label']:32s} {label:17s} vaccinates until {on.date.iloc[-1]}"
                      f" (day {on.day.iloc[-1]:.0f})")
    timing.to_csv(RESULTS / "control_timing.csv", index=False)
    scen.to_csv(RESULTS / "control_scenarios.csv", index=False)
    front.to_csv(RESULTS / "control_frontier.csv", index=False)

    print("\nA. OPTIMAL STRATEGIES FROM DETECTION "
          f"(u1 <= {U1_MAX}/day, u2 <= {U2_MAX}; 50 doses or 10 isolation-days = 1 case)")
    for n in names:
        g = strat[strat.outbreak == n]
        print(f"  {OUTBREAKS[n]['label']}  (no response: {g.cases_no_response.iloc[0]:.0f} cases;"
              f" detection day {out[n]['t_detect']:.0f})")
        for _, r in g.iterrows():
            print(f"    {r.strategy:17s} cases {r.cases:6.1f}  averted {r.averted_pct:5.1f}%  "
                  f"doses {r.doses:6.0f} (wasted {r.doses_wasted:4.0f})  "
                  f"isolation-days {r.isolation_days:5.0f}  doses/case averted "
                  f"{r.doses_per_case_averted:5.2f}  J by start {r.J_by_start}")

    print("\nB. TIMING: % of cases averted by a response starting d days after detection")
    for n in names:
        g = timing[timing.outbreak == n].pivot(index="delay_days", columns="strategy",
                                               values="averted_pct")
        print(f"  {OUTBREAKS[n]['label']}")
        print(g.round(1).to_string().replace("\n", "\n    "))

    print("\nC. REALISTIC SCENARIOS (from detection)")
    for n in names:
        print(f"  {OUTBREAKS[n]['label']}")
        for _, r in scen[scen.outbreak == n].iterrows():
            print(f"    {r.scenario:34s} averted {r.averted_pct:5.1f}%  doses {r.doses:6.0f}")

    print("\nD. DOSE FRONTIER (vaccination only)")
    print(front.pivot(index="c_d", columns="outbreak",
                      values=["doses", "averted_pct"]).round(1).to_string())

    # ---- figure: optimal controls ("both") and resulting cases
    colors = {"vaccination only": "#2171b5", "isolation only": "#d94801", "both": "#41ab5d"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row, n in zip(axes, names):
        o = out[n]
        day0 = o["start_date"]
        dates_all = day0 + pd.to_timedelta(o["t_all"], unit="D")
        new_none = SIGMA * (o["x_none"][:, iE1] + o["x_none"][:, iE2])
        row[0].plot(dates_all, new_none, color="#636363", lw=2, label="no response")
        for label, (t, u1, u2, x) in o["profiles"].items():
            d = day0 + pd.to_timedelta(t, unit="D")
            row[0].plot(d, SIGMA * (x[:, iE1] + x[:, iE2]), color=colors[label], lw=1.6,
                        label=f"optimal: {label}")
        row[0].axvline(day0 + pd.Timedelta(days=o["t_detect"]), color="black", ls=":", lw=1)
        row[0].set_ylabel("new cases per day")
        row[0].set_title(OUTBREAKS[n]["label"], loc="left", fontsize=11)
        t, u1, u2, _ = o["profiles"]["both"]
        d = day0 + pd.to_timedelta(t, unit="D")
        row[1].plot(d, 100 * u1 / U1_MAX, color="#2171b5", lw=2,
                    label=r"vaccination $u_1$ (% of maximum)")
        row[1].plot(d, 100 * u2 / U2_MAX, color="#d94801", lw=2, ls="--",
                    label=r"isolation $u_2$ (% of maximum)")
        row[1].set_ylim(-5, 105)
        row[1].set_ylabel("control (% of maximum)")
        row[1].set_title("optimal controls, strategy 'both'", loc="left", fontsize=10)
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(day0, day0 + pd.Timedelta(days=PLOT_DAYS))
            locator = mdates.AutoDateLocator(maxticks=6)       # few, readable date labels
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "fig_optimal_controls.png", dpi=200)

    # ---- figure: timing
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for ax, n in zip(axes, names):
        g = timing[timing.outbreak == n]
        for label in ("both", "isolation only", "vaccination only"):
            h = g[g.strategy == label]
            ax.plot(h.delay_days, h.averted_pct, "o-", ms=3, color=colors[label], label=label)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.set_title(OUTBREAKS[n]["label"], loc="left", fontsize=11)
        ax.set_xlabel("response start (days after detection)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("cases averted (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "fig_control_timing.png", dpi=200)

    # ---- figure: dose frontier
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for n, col in zip(names, ("#08306b", "#2171b5", "#6baed6")):
        g = front[front.outbreak == n].sort_values("doses")
        ax.plot(g.doses, g.averted_pct, "o-", color=col, label=OUTBREAKS[n]["label"])
    ax.set_xlabel("doses given (optimal campaign)")
    ax.set_ylabel("cases averted (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "fig_dose_frontier.png", dpi=200)
    print("\nSaved results/control_*.csv and figures/fig_optimal_controls.png, "
          "fig_control_timing.png, fig_dose_frontier.png")


if __name__ == "__main__":      # required on Windows when using multiprocessing
    main()
