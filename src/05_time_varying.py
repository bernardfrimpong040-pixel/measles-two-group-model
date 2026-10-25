"""
Step 5 -- Why did each outbreak stop?  (Stage 4, part 1)

In Stage 3 every outbreak was fitted with CONSTANT transmission, so the only
way the model could end an outbreak early was a small community network.
A public-health response (quarantine, extra vaccination) could also end it.
Here we let transmission change once, by a factor m, around day tc:

    beta(t) = beta * [1 + (m - 1) * expit((t - tc) / 3 days)]

    m < 1 : transmission fell (e.g. response, school holidays)
    m > 1 : transmission rose (e.g. spread into schools or churches)

and compare three explanations for each outbreak:

  H1  network only   constant transmission, network size estimated (Stage 3)
  H2  response only  network = the WHOLE community, transmission may change
  H3  both           network size estimated AND transmission may change

Models are compared with AIC = 2 x (number of free parameters) - 2 x log-lik.
Lower is better; differences below about 2 mean the data cannot choose.
We also compare each model's share of cases in the community with what
officials reported (about 90% in South Carolina), because that is outside
information the case curve does not contain.

R0 is held at 15 throughout (Stage 3 showed it cannot be estimated).
The three outbreaks run in parallel; expect 5-15 minutes.

Outputs
  results/time_varying_models.csv
  figures/fig_time_varying_fits.png

Run from the project folder:   python src/05_time_varying.py
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
R0_REF = 15.0

MODELS = {   # name: (description, estimate network size?, time-varying?)
    "H1": ("network only", True, False),
    "H2": ("response only", False, True),
    "H3": ("network + change", True, True),
}


def setting_for(name, n_periods):
    """Fixed numbers for one outbreak (same as Stage 3, plus the window for tc)."""
    e = OUTBREAKS[name]
    T = n_periods * DAYS_PER_PERIOD
    return dict(N_total=e["N1"]["value"] + e["N2"]["value"],
                N2eff_bounds=(50, e["N2"]["high"]),
                c22=e["cov2"]["value"],
                tc_bounds=(14, T - 14))       # change at least 2 weeks from either end


def load_cases(name):
    data = pd.read_csv(ROOT / "data" / "processed" / "outbreaks_weekly.csv",
                       parse_dates=["week_start"])
    g = data[data.outbreak == name]
    return biweekly(g.cases.values), g.week_start.iloc[0]


def analyse(name):
    cases, start_date = load_cases(name)
    setting = setting_for(name, len(cases))
    community_size = OUTBREAKS[name]["N2"]["value"]
    rows, curves = [], {}
    for model, (desc, estimate_network, time_varying) in MODELS.items():
        fixed = {"R0": R0_REF}
        if not estimate_network:
            fixed["N2eff"] = community_size          # whole community can be reached
        par, ll = fit(cases, setting, fixed=fixed, time_varying=time_varying,
                      n_starts=12 if time_varying else 8)
        # free parameters: theta, s1, I0, k, (+ N2eff), (+ m and tc); R0 is fixed
        n_free = 4 + (1 if estimate_network else 0) + (2 if time_varying else 0)
        s = summaries(par, setting, cases)
        row = {"outbreak": name, "model": model, "description": desc,
               "loglik": ll, "n_free": n_free, "AIC": 2 * n_free - 2 * ll,
               "theta": par["theta"], "s1": par["s1"], "N2eff": par["N2eff"],
               "k": par["k"], "m": par["m"],
               "change_date": (start_date + pd.Timedelta(days=float(par["tc"]))).date()
                              if time_varying else None,
               **s}
        rows.append(row)
        curves[model] = run_model(par, setting, len(cases))
        print(f"[{name}] {model} ({desc}) done: log-lik {ll:.2f}", flush=True)
    return dict(name=name, rows=rows, curves=curves, cases=cases, start_date=start_date)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    names = list(OUTBREAKS)
    with Pool(processes=len(names)) as pool:
        out = {r["name"]: r for r in pool.map(analyse, names)}

    table = pd.DataFrame([row for n in names for row in out[n]["rows"]])
    table["dAIC"] = table.AIC - table.groupby("outbreak").AIC.transform("min")
    table.to_csv(RESULTS / "time_varying_models.csv", index=False)

    print("\nMODEL COMPARISON (R0 = 15).  dAIC = AIC minus the best AIC for that outbreak")
    for n in names:
        print(f"\n  {OUTBREAKS[n]['label']}")
        for _, r in table[table.outbreak == n].iterrows():
            change = (f"m = {r.m:.2f} from {r.change_date}" if r.model != "H1"
                      else "no change")
            after = f"  R_v after = {r.Rv_after:.2f}" if r.model != "H1" else ""
            print(f"    {r.model} {r.description:17s} log-lik {r.loglik:7.2f}  "
                  f"AIC {r.AIC:6.1f}  dAIC {r.dAIC:4.1f}  | network {r.N2eff:6.0f}  "
                  f"{change}  R_v start = {r.Rv:.2f}{after}  "
                  f"share in community {100 * r.community_share:3.0f}%")

    # ---- figure: the three models against the data
    colors = {"H1": "#08306b", "H2": "#d94801", "H3": "#41ab5d"}
    styles = {"H1": "-", "H2": "--", "H3": "-."}
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for ax, n in zip(axes, names):
        o = out[n]
        dates = o["start_date"] + pd.to_timedelta(
            np.arange(len(o["cases"])) * DAYS_PER_PERIOD, unit="D")
        ax.bar(dates, o["cases"], width=12, color="#d9d9d9", label="reported cases")
        for model, (e1, e2) in o["curves"].items():
            r = table[(table.outbreak == n) & (table.model == model)].iloc[0]
            ax.plot(dates, e1 + e2, color=colors[model], ls=styles[model], lw=2,
                    label=f"{model} {MODELS[model][0]} (dAIC {r.dAIC:.1f})")
        ax.set_title(OUTBREAKS[n]["label"], loc="left", fontsize=11)
        ax.set_ylabel("cases per 2 weeks")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIGURES / "fig_time_varying_fits.png", dpi=200)
    print("\nSaved results/time_varying_models.csv and figures/fig_time_varying_fits.png")


if __name__ == "__main__":      # required on Windows when using multiprocessing
    main()
