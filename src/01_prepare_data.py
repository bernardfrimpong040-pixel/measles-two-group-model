"""
Step 1 -- Build the outbreak case series used in the paper.

Source: Johns Hopkins University US Measles Data Repository
        https://github.com/CSSEGISandData/measles_data
        (Ahmadi, Dong & Gardner, JAMA 2025, doi:10.1001/jama.2025.17812)

We pin one version of the repository (DATA_COMMIT) so every number in the
paper can be reproduced exactly.

Outputs
  data/processed/outbreaks_weekly.csv   one row per outbreak-week
  figures/fig1_outbreak_curves.png      Figure 1 of the paper

Run from the project folder:   python src/01_prepare_data.py
"""

from pathlib import Path
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
DATA_COMMIT = "c361b387631ce7e31f9d9b28eb36b7642b8b9892"   # repo version used (23 Oct 2026)
BASE_URL = f"https://raw.githubusercontent.com/CSSEGISandData/measles_data/{DATA_COMMIT}/"
FILES = ["Top_states_time_series.csv", "measles_county_all_updates.csv"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# Official outbreak totals, used only to check our series (see README for sources)
OFFICIAL_TOTALS = {"west_texas": 762, "upstate_sc": 997}


# ---------------------------------------------------------------- helpers
def download_raw_files():
    """Download the two CSV files once; reuse them afterwards."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, target)


def week_start(dates):
    """Sunday that starts the MMWR (epidemiological) week containing each date."""
    return dates - pd.to_timedelta((dates.dt.day_of_week + 1) % 7, unit="D")


def county_weekly(county_df, counties, start, end):
    """Weekly sum of newly reported cases for a set of counties, zero-filled."""
    sub = county_df[county_df["location_name"].isin(counties)].copy()
    sub["week_start"] = week_start(sub["date"])
    weekly = sub.groupby("week_start")["value"].sum()
    all_weeks = pd.date_range(start, end, freq="W-SUN")
    return weekly.reindex(all_weeks, fill_value=0)


# ---------------------------------------------------------------- main
def main():
    download_raw_files()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    state = pd.read_csv(RAW / "Top_states_time_series.csv", parse_dates=["week_start"])
    county = pd.read_csv(RAW / "measles_county_all_updates.csv", parse_dates=["date"])

    outbreaks = []

    # 1. West Texas, 2025 -- statewide series by RASH ONSET week.
    #    Includes ~40 Texas cases not linked to the outbreak (travel etc.).
    tx = state[(state.week_start >= "2025-01-12") & (state.week_start <= "2025-08-31")]
    outbreaks.append(pd.DataFrame({
        "outbreak": "west_texas", "week_start": tx.week_start.values,
        "cases": tx.TX_cases.values, "date_basis": "rash_onset"}))

    # 2. Upstate South Carolina, 2025-26 -- Spartanburg + neighbours by REPORT week.
    #    Batch reporting around the holidays (174 cases in one week) is a
    #    reporting artefact, not biology; see the README for how we handle it.
    sc = county_weekly(county,
                       ["Spartanburg, South Carolina", "Cherokee, South Carolina",
                        "Greenville, South Carolina", "Pickens, South Carolina"],
                       "2025-09-07", "2026-04-26")
    outbreaks.append(pd.DataFrame({
        "outbreak": "upstate_sc", "week_start": sc.index, "cases": sc.values,
        "date_basis": "report_date"}))

    # 3. Short Creek (Hildale, UT / Colorado City, AZ), 2025-26 --
    #    Southwest Utah Health District + Mohave County, AZ, by REPORT week.
    sk = county_weekly(county,
                       ["Southwest Health District, Utah", "Mohave, Arizona"],
                       "2025-06-22", "2026-05-31")
    outbreaks.append(pd.DataFrame({
        "outbreak": "short_creek", "week_start": sk.index, "cases": sk.values,
        "date_basis": "report_date"}))

    data = pd.concat(outbreaks, ignore_index=True)
    data["week"] = data.groupby("outbreak").cumcount()          # week 0, 1, 2, ...
    data["cumulative"] = data.groupby("outbreak")["cases"].cumsum()
    data.to_csv(PROCESSED / "outbreaks_weekly.csv", index=False)

    # ---- checks against official totals
    print("\nOutbreak totals in our series vs official counts")
    for name, g in data.groupby("outbreak", sort=False):
        official = OFFICIAL_TOTALS.get(name, "n/a")
        print(f"  {name:12s} weeks={len(g):3d}  total={g.cases.sum():5d}  "
              f"official={official}  peak={g.cases.max()} "
              f"({g.loc[g.cases.idxmax(), 'week_start'].date()})")

    # ---- Figure 1
    titles = {"west_texas": "West Texas, 2025 (rash-onset week)",
              "upstate_sc": "Upstate South Carolina, 2025–26 (report week)",
              "short_creek": "Short Creek, UT–AZ, 2025–26 (report week)"}
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), constrained_layout=True)
    for ax, (name, g) in zip(axes, data.groupby("outbreak", sort=False)):
        ax.bar(g.week_start, g.cases, width=6, color="#1f4e79")
        ax.set_title(titles[name], loc="left", fontsize=11)
        ax.set_ylabel("Confirmed cases")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Week")
    fig.savefig(FIGURES / "fig1_outbreak_curves.png", dpi=200)
    print(f"\nSaved {PROCESSED / 'outbreaks_weekly.csv'}")
    print(f"Saved {FIGURES / 'fig1_outbreak_curves.png'}")


if __name__ == "__main__":
    main()
