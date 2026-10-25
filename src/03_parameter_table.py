"""
Step 3 -- Parameter table and first calculations (Stage 2).

What this script does
  1. Prints every parameter with its value, range, status and source,
     and saves the table to results/parameter_table.csv.
  2. Check 1: what if kindergarten coverage described the WHOLE general
     population?  Computes R0 * s1 for each outbreak.  A value above 1 would
     mean the general population could sustain measles by itself.
  3. Figure 2: how much two-dose coverage the community itself needs
     (critical coverage c2*, Corollary 1 of the paper) as the mixing
     parameter theta goes from 0 (random mixing) to 1 (fully separate).
     These are illustrative: R0 and s1 are only fixed properly in Stage 3.

Run from the project folder:   python src/03_parameter_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parameters import DISEASE, VACCINE, POPULATION_IMMUNITY, OUTBREAKS
from model import critical_s2, effective_susceptibility

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 1. table
rows = []
for group_name, group in (("disease", DISEASE), ("vaccine", VACCINE),
                          ("immunity", POPULATION_IMMUNITY)):
    for name, entry in group.items():
        rows.append({"outbreak": "all", "parameter": name, **entry})
for outbreak, entries in OUTBREAKS.items():
    for name, entry in entries.items():
        if name != "label":
            rows.append({"outbreak": outbreak, "parameter": name, **entry})

table = pd.DataFrame(rows, columns=["outbreak", "parameter", "value", "low", "high",
                                    "status", "source", "note"])
table.to_csv(RESULTS / "parameter_table.csv", index=False)

print("PARAMETER TABLE  (status 'estimated' = fitted to case data in Stage 3)\n")
for _, r in table.iterrows():
    rng = "" if r.low is None or pd.isna(r.low) else f"[{r.low:g}, {r.high:g}]"
    print(f"  {r.outbreak:12s} {r.parameter:16s} {r.value:>10g} {rng:20s} "
          f"{r.status:10s} {r.source}")

# ---------------------------------------------------------------- 2. check 1
R0 = DISEASE["R0"]["value"]
eps2 = VACCINE["eps2"]["value"]
print(f"\nCHECK 1 -- kindergarten coverage applied to the whole general population"
      f" (R0 = {R0:g})")
for outbreak, e in OUTBREAKS.items():
    cov1 = e["cov1"]["value"]
    s1 = effective_susceptibility(0.0, cov1, VACCINE["eps1"]["value"], eps2)
    verdict = "could sustain spread ALONE" if R0 * s1 > 1 else "below threshold"
    print(f"  {e['label']:32s} cov1 = {cov1:.3f}  s1 = {s1:.3f}  "
          f"R0*s1 = {R0 * s1:.2f}  -> {verdict}")
s1_max = POPULATION_IMMUNITY["s1"]["high"]
print(f"  With national serology (s1 <= {s1_max}): R0*s1 <= {R0 * s1_max:.2f}")
print("  Conclusion: school coverage overstates adult susceptibility; s1 must be"
      " estimated from the data (Stage 3).")

# ---------------------------------------------------------------- 3. figure 2
thetas = np.linspace(0, 1, 201)
s1_values = [0.02, 0.04, 0.06]          # plausible general-population values
colors = ["#9ecae1", "#4292c6", "#08306b"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
print(f"\nCRITICAL COMMUNITY COVERAGE c2* (R0 = {R0:g}, eps2 = {eps2})")
for ax, (outbreak, e) in zip(axes, OUTBREAKS.items()):
    N1, N2 = e["N1"]["value"], e["N2"]["value"]
    p1 = N1 / (N1 + N2)
    print(f"  {e['label']}: community share p2 = {1 - p1:.4f}")
    for s1, col in zip(s1_values, colors):
        c2_star = np.array([(1 - critical_s2(R0, th, p1, s1)) / eps2 for th in thetas])
        c2_star = np.minimum(c2_star, 1.0)      # cannot vaccinate more than 100%
        ax.plot(thetas, 100 * c2_star, color=col, lw=2, label=f"$s_1$ = {s1}")
        for th in (0.5, 0.9):
            i = int(round(th * 200))
            print(f"      s1 = {s1}: theta = {th}: c2* = {100 * c2_star[i]:5.1f}%")
    cov2 = e["cov2"]["value"]
    if e["cov2"]["status"] == "fixed":
        ax.axhline(100 * cov2, color="#b2182b", ls="--", lw=1.5,
                   label=f"reported community coverage ({100 * cov2:.0f}%)")
    ax.set_title(e["label"], loc="left", fontsize=11)
    ax.set_xlabel(r"within-community mixing $\theta$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 102)
axes[0].set_ylabel("two-dose coverage the community needs (%)")
# one shared legend below the three panels (South Carolina panel has every line)
handles, labels = axes[1].get_legend_handles_labels()
labels[-1] = "reported community coverage (SC 17%, Short Creek 8%)"
fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
fig.savefig(FIGURES / "fig2_critical_coverage.png", dpi=200)

print(f"\nSaved {RESULTS / 'parameter_table.csv'}")
print(f"Saved {FIGURES / 'fig2_critical_coverage.png'}")
