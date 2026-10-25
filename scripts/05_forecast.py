# 05_forecast.py
# =============================================================================
# Purpose: make every forecast in the study (Analysis Plan, section 4).
#
# For each Friday forecast date ("origin"), each target (flu, RSV, COVID-19,
# RSV 65+), each horizon h = 1..4 weeks, and each state, it produces 23
# quantiles of weekly admissions from:
#   model A (linear quantile regression)  - without and with wastewater
#   model B (LightGBM quantile models)    - without and with wastewater
#   flat baseline
# The two arms of a model are identical except for the wastewater inputs.
#
# The script runs in three stages, each saving its results, so it can be
# stopped and restarted:
#   python scripts/05_forecast.py features   (about 5 minutes)
#   python scripts/05_forecast.py tune       (about 30-40 minutes)
#   python scripts/05_forecast.py forecast   (about 2 hours; resumable)
#
# KEY RULES THAT PREVENT INFORMATION LEAKS
#   * Features for any week come from the data as published on the Friday
#     right after it (hospital versions; pseudo-real-time wastewater).
#   * Training targets at an origin use only weeks already published by then.
#   * Tuning uses only the 2023-24 season; the chosen settings are saved once
#     and then frozen (the tune stage refuses to run again).
# =============================================================================

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from epiweeks import Week
from joblib import Parallel, delayed
from scipy.optimize import linprog

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (NHSN_PAUSE_FIRST_WEEK_END, NHSN_PAUSE_LAST_WEEK_END,  # noqa: E402
                    QUANTILES, weighted_interval_score)

PROC = os.path.join("data", "processed")
MODEL_DIR = os.path.join("data", "models")
FC_DIR = os.path.join("data", "forecasts")

# target -> which wastewater pathogen it uses
TARGETS = {"flu": "flu", "rsv": "rsv", "covid": "covid", "rsv_65plus": "rsv"}
HORIZONS = [1, 2, 3, 4]

BASE_FEATURES = ["lvl", "d1", "d2", "woy_sin", "woy_cos"]
WW_FEATURES = ["ww_lvl", "ww_d1", "ww_d2"]

# Tuning grid (section 4). Window = number of most recent training weeks.
WINDOW_GRID = [26, 52, 78]
SMOOTH_GRID = [1, 2, 3]                      # 1 = no smoothing
LGBM_GRID = [dict(num_leaves=7, n_estimators=100), dict(num_leaves=7, n_estimators=300),
             dict(num_leaves=15, n_estimators=100), dict(num_leaves=15, n_estimators=300)]
LGBM_FIXED = dict(learning_rate=0.05, min_child_samples=20, subsample=1.0,
                  colsample_bytree=1.0, verbose=-1, n_jobs=1)
TUNING_ORIGINS = pd.date_range("2023-12-01", "2024-04-26", freq="W-FRI")
TUNING_TARGETS = ["flu", "covid"]    # RSV reporting was voluntary in 2023-24

WW_GAP_WEEKS = 2                     # carry the last wastewater value forward <= 2 weeks
MIN_TRAIN_ROWS = 200                 # fewer training rows -> no forecast (both arms)
BASELINE_HISTORY_WEEKS = 52
N_JOBS = -1                          # use all CPU cores for LightGBM


# =============================================================================
# Data helpers
# =============================================================================
def load_inputs():
    versions = pd.read_csv(os.path.join(PROC, "nhsn_versions.csv.gz"),
                           parse_dates=["week_end", "report_date"])
    truth = pd.read_csv(os.path.join(PROC, "nhsn_truth.csv.gz"), parse_dates=["week_end"])
    ww = pd.read_csv(os.path.join(PROC, "ww_state_asof.csv.gz"), parse_dates=["origin", "week_end"])
    cov = pd.read_csv(os.path.join("data", "summary", "coverage_by_state.csv"))
    window = pd.read_csv(os.path.join(PROC, "evaluation_window.csv"))
    states = {t: sorted(cov.loc[(cov["pathogen"] == p) & cov["passes_main_rule"], "state"])
              for t, p in TARGETS.items()}
    return versions, truth, ww, states, window


def in_pause(weeks):
    weeks = pd.to_datetime(weeks)
    return (weeks >= NHSN_PAUSE_FIRST_WEEK_END) & (weeks <= NHSN_PAUSE_LAST_WEEK_END)


def vintage(versions, friday, first_report):
    """Hospital admissions as published by `friday`, one value per
    (target, state, week). Before a target's first saved version, its first
    version is used, cut to weeks that had ended by then (pseudo-real-time)."""
    parts = []
    for target, first in first_report.items():
        v = versions[versions["target"] == target]
        cutoff = max(friday, first)
        v = v[(v["report_date"] <= cutoff) & (v["week_end"] <= friday - pd.Timedelta(days=6))]
        parts.append(v.drop_duplicates(["target", "state", "week_end"], keep="last"))
    out = pd.concat(parts, ignore_index=True)
    return out.set_index(["target", "state", "week_end"])["value"]


def to_log(x):
    return np.log(np.asarray(x, dtype=float) + 1)


# =============================================================================
# Stage 1: features for every Friday
# =============================================================================
def wastewater_features(ww_f, anchor, smooth):
    """ww_lvl, ww_d1, ww_d2 for each (pathogen, state) from one Friday's table.

    Weekly values are carried forward for at most WW_GAP_WEEKS weeks, then
    smoothed with a trailing mean over `smooth` weeks."""
    weeks = pd.date_range(anchor - pd.Timedelta(weeks=7), anchor, freq="7D")
    out = []
    for (pathogen, state), g in ww_f.groupby(["pathogen", "state"]):
        s = g.set_index("week_end")["ww_value"].reindex(weeks).ffill(limit=WW_GAP_WEEKS)
        sm = s.rolling(smooth, min_periods=smooth).mean()
        lvl, l1, l2 = sm.iloc[-1], sm.iloc[-2], sm.iloc[-3]
        out.append((pathogen, state, lvl, lvl - l1, lvl - l2))
    return pd.DataFrame(out, columns=["pathogen", "state", "ww_lvl", "ww_d1", "ww_d2"])


def build_features(versions, ww):
    """One row per (target, state, anchor week) with the features as known on
    the Friday after the anchor week. Wastewater columns for each smoothing."""
    first_report = versions.groupby("target")["report_date"].min().to_dict()
    first_friday = versions["week_end"].min() + pd.Timedelta(days=6 + 14)
    last_friday = versions["report_date"].max()
    fridays = pd.date_range(first_friday, last_friday, freq="W-FRI")
    rows = []
    for i, f in enumerate(fridays):
        anchor = f - pd.Timedelta(days=6)                       # last week ended by f
        v = vintage(versions, f, first_report)
        y = v.unstack("week_end")
        need = [anchor - pd.Timedelta(weeks=k) for k in (0, 1, 2)]
        if not all(w in y.columns for w in need):
            continue
        y = y[need].dropna()
        if len(y) == 0 or in_pause(need).any():
            continue
        feat = pd.DataFrame({"lvl": to_log(y[need[0]]),
                             "d1": to_log(y[need[0]]) - to_log(y[need[1]]),
                             "d2": to_log(y[need[1]]) - to_log(y[need[2]]),
                             "y_anchor": y[need[0]]}, index=y.index).reset_index()
        woy = Week.fromdate(anchor.date()).week
        feat["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
        feat["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
        feat["friday"], feat["anchor"] = f, anchor
        feat["pathogen"] = feat["target"].map(TARGETS)

        ww_f = ww[ww["origin"] == f]
        for L in SMOOTH_GRID:
            wf = (wastewater_features(ww_f, anchor, L) if len(ww_f) else
                  pd.DataFrame(columns=["pathogen", "state"] + WW_FEATURES))
            wf = wf.rename(columns={c: f"{c}_s{L}" for c in WW_FEATURES})
            feat = feat.merge(wf, on=["pathogen", "state"], how="left")
        rows.append(feat)
        if i % 25 == 0:
            print(f"  features: {f.date()} ({i + 1}/{len(fridays)})")
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Models
# =============================================================================
def qr_fit_predict(X, y, X_new, taus):
    """Linear quantile regression, solved exactly as a linear program
    (Koenker's dual: maximise y'a subject to X'a = (1-tau) X'1, 0 <= a <= 1;
    the coefficients are the equality-constraint multipliers)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    Xn1 = np.column_stack([np.ones(len(X_new)), X_new])
    preds = np.empty((len(X_new), len(taus)))
    for j, tau in enumerate(taus):
        res = linprog(-y, A_eq=X1.T, b_eq=(1 - tau) * X1.sum(axis=0),
                      bounds=(0, 1), method="highs-ds")
        if res.status != 0:
            raise RuntimeError(f"quantile regression failed: {res.message}")
        beta = -res.eqlin.marginals
        preds[:, j] = Xn1 @ beta
    return preds


def _lgbm_one(X, y, X_new, tau, params):
    m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **LGBM_FIXED, **params)
    m.fit(X, y)
    return m.predict(X_new)


def lgbm_fit_predict(X, y, X_new, taus, params):
    cols = Parallel(n_jobs=N_JOBS)(delayed(_lgbm_one)(X, y, X_new, t, params) for t in taus)
    return np.column_stack(cols)


def training_rows(features, vint_o, target, h, origin, window, smooth, with_ww):
    """Training pairs for one origin: features at anchor a (as known then) and
    target = log(admissions in week a+h, as published by the origin) - lvl(a)."""
    anchor_o = origin - pd.Timedelta(days=6)
    last_anchor = anchor_o - pd.Timedelta(weeks=h)
    first_anchor = last_anchor - pd.Timedelta(weeks=window - 1)
    f = features[(features["target"] == target)
                 & features["anchor"].between(first_anchor, last_anchor)].copy()
    f["target_week"] = f["anchor"] + pd.Timedelta(weeks=h)
    f = f[~in_pause(f["target_week"])]
    key = pd.MultiIndex.from_arrays([f["target"], f["state"], f["target_week"]])
    f["y_future"] = vint_o.reindex(key).to_numpy()
    f = f.dropna(subset=["y_future"])
    f["z"] = to_log(f["y_future"]) - f["lvl"]
    cols = BASE_FEATURES + ([f"{c}_s{smooth}" for c in WW_FEATURES] if with_ww else [])
    f = f.dropna(subset=cols)
    return f, cols


def forecast_one(features, vint_o, target, h, origin, states, cfg, model, with_ww):
    """Quantile forecasts (admissions) for every state, one target/horizon/arm."""
    train, cols = training_rows(features, vint_o, target, h, origin,
                                cfg["window"], cfg["smooth"], with_ww)
    new = features[(features["target"] == target) & (features["friday"] == origin)
                   & features["state"].isin(states)].dropna(subset=cols)
    if len(train) < MIN_TRAIN_ROWS or len(new) == 0:
        return None
    X, z, Xn = train[cols].to_numpy(), train["z"].to_numpy(), new[cols].to_numpy()
    if model == "A":
        qz = qr_fit_predict(X, z, Xn, QUANTILES)
    else:
        qz = lgbm_fit_predict(X, z, Xn, QUANTILES, cfg["lgbm"])
    qz = np.sort(qz, axis=1)                         # fix any quantile crossing
    q = np.maximum(np.exp(new["lvl"].to_numpy()[:, None] + qz) - 1, 0)
    out = pd.DataFrame(q, columns=[f"q{t:.3f}" for t in QUANTILES])
    out.insert(0, "state", new["state"].to_numpy())
    out["n_train"] = len(train)
    return out


def baseline_forecast(vint_o, target, h, origin, states):
    """Flat baseline: median = latest published week; spread = quantiles of
    past h-week changes in that state (last 52 weeks), made symmetric."""
    anchor_o = origin - pd.Timedelta(days=6)
    rows = []
    for s in states:
        try:
            series = vint_o.loc[(target, s)].sort_index()
        except KeyError:
            continue
        series = series[~in_pause(series.index)]
        if anchor_o not in series.index:
            continue
        hist = series[series.index > anchor_o - pd.Timedelta(weeks=BASELINE_HISTORY_WEEKS)]
        hist = hist.asfreq("7D")
        diffs = (hist - hist.shift(h)).dropna().to_numpy()
        if len(diffs) < 8:
            continue
        sym = np.concatenate([diffs, -diffs])
        q = np.maximum(series[anchor_o] + np.quantile(sym, QUANTILES), 0)
        rows.append([s] + list(q))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["state"] + [f"q{t:.3f}" for t in QUANTILES])


# =============================================================================
# Stage 2: tuning on the 2023-24 season
# =============================================================================
def tune(features, versions, truth, states):
    os.makedirs(MODEL_DIR, exist_ok=True)
    chosen_path = os.path.join(MODEL_DIR, "chosen_settings.json")
    if os.path.exists(chosen_path):
        print("Settings are already frozen in", chosen_path)
        print("Delete that file ONLY if the analysis plan is formally changed.")
        return
    first_report = versions.groupby("target")["report_date"].min().to_dict()
    truth_s = truth.set_index(["target", "state", "week_end"])["value"]
    vintages = {o: vintage(versions, o, first_report) for o in TUNING_ORIGINS}   # compute once
    log = []

    def score(cfg, model, with_ww):
        """Mean over tuning targets of (mean WIS / mean baseline WIS)."""
        ratios = {}
        for target in TUNING_TARGETS:
            w_model, w_base = [], []
            for o in TUNING_ORIGINS:
                vint_o = vintages[o]
                for h in HORIZONS:
                    fc = forecast_one(features, vint_o, target, h, o, states[target], cfg, model, with_ww)
                    bl = baseline_forecast(vint_o, target, h, o, states[target])
                    if fc is None or bl is None:
                        continue
                    tw = o - pd.Timedelta(days=6) + pd.Timedelta(weeks=h)
                    m = fc.merge(bl, on="state", suffixes=("", "_b"))
                    key = pd.MultiIndex.from_arrays([[target] * len(m), m["state"], [tw] * len(m)])
                    y = truth_s.reindex(key).to_numpy()
                    ok = ~np.isnan(y)
                    if in_pause([tw]).any() or not ok.any():
                        continue
                    qc = [f"q{t:.3f}" for t in QUANTILES]
                    w_model.append(weighted_interval_score(y[ok], m.loc[ok, qc].to_numpy())["wis"])
                    w_base.append(weighted_interval_score(
                        y[ok], m.loc[ok, [c + "_b" for c in qc]].to_numpy())["wis"])
            ratios[target] = np.concatenate(w_model).mean() / np.concatenate(w_base).mean()
        return float(np.mean(list(ratios.values()))), ratios

    # 1. window length: model A without wastewater
    best = None
    for W in WINDOW_GRID:
        t0 = time.time()
        r, parts = score({"window": W, "smooth": 1}, "A", False)
        log.append(dict(step="window", model="A", window=W, score=r, **parts))
        print(f"  window {W}: relative WIS {r:.4f}  ({time.time() - t0:.0f}s)")
        best = (r, W) if best is None or r < best[0] else best
    W = best[1]

    # 2. LightGBM settings: model B without wastewater, chosen window
    best = None
    for i, p in enumerate(LGBM_GRID):
        t0 = time.time()
        r, parts = score({"window": W, "smooth": 1, "lgbm": p}, "B", False)
        log.append(dict(step="lgbm", model="B", window=W, params=json.dumps(p), score=r, **parts))
        print(f"  lightgbm {p}: relative WIS {r:.4f}  ({time.time() - t0:.0f}s)")
        best = (r, i) if best is None or r < best[0] else best
    P = LGBM_GRID[best[1]]

    # 3. smoothing window: model A WITH wastewater, chosen window
    best = None
    for L in SMOOTH_GRID:
        t0 = time.time()
        r, parts = score({"window": W, "smooth": L}, "A", True)
        log.append(dict(step="smooth", model="A", window=W, smooth=L, score=r, **parts))
        print(f"  smoothing {L}: relative WIS {r:.4f}  ({time.time() - t0:.0f}s)")
        best = (r, L) if best is None or r < best[0] else best
    L = best[1]

    pd.DataFrame(log).to_csv(os.path.join(MODEL_DIR, "tuning_results.csv"), index=False)
    chosen = {"window": W, "smooth": L, "lgbm": P,
              "tuned_on": [str(TUNING_ORIGINS[0].date()), str(TUNING_ORIGINS[-1].date())],
              "tuning_targets": TUNING_TARGETS, "frozen_at": pd.Timestamp.now().isoformat()}
    with open(chosen_path, "w") as fh:
        json.dump(chosen, fh, indent=2)
    print("\nFrozen settings:", json.dumps(chosen))


# =============================================================================
# Stage 3: forecasts for every evaluation origin
# =============================================================================
def forecast_all(features, versions, states, window):
    with open(os.path.join(MODEL_DIR, "chosen_settings.json")) as fh:
        cfg = json.load(fh)
    os.makedirs(FC_DIR, exist_ok=True)
    first_report = versions.groupby("target")["report_date"].min().to_dict()
    first_forecast = {t: d + pd.Timedelta(days=(4 - d.day_of_week) % 7)   # next Friday
                      for t, d in first_report.items()}
    origins = pd.date_range(window["first_eval_origin"].item(),
                            window["last_eval_origin"].item(), freq="W-FRI")
    print(f"Settings: {cfg}\n{len(origins)} forecast dates")

    for i, o in enumerate(origins):
        path = os.path.join(FC_DIR, f"forecasts_{o.date()}.parquet")
        if os.path.exists(path):
            continue
        t0 = time.time()
        vint_o = vintage(versions, o, first_report)
        parts = []
        for target in TARGETS:
            if o < first_forecast[target]:
                continue
            for h in HORIZONS:
                tw = o - pd.Timedelta(days=6) + pd.Timedelta(weeks=h)
                here = []                               # forecasts for this target and horizon
                bl = baseline_forecast(vint_o, target, h, o, states[target])
                if bl is not None:
                    here.append(bl.assign(model="baseline", arm="none", fallback=0))
                for model in ["A", "B"]:
                    no_ww = forecast_one(features, vint_o, target, h, o, states[target], cfg, model, False)
                    if no_ww is None:
                        continue
                    here.append(no_ww.assign(model=model, arm="no_ww", fallback=0))
                    ww = forecast_one(features, vint_o, target, h, o, states[target], cfg, model, True)
                    # states without usable wastewater get the no-wastewater forecast
                    got = set() if ww is None else set(ww["state"])
                    miss = no_ww[~no_ww["state"].isin(got)].assign(fallback=1)
                    pieces = [miss] if ww is None else [ww.assign(fallback=0), miss]
                    here.append(pd.concat(pieces, ignore_index=True).assign(model=model, arm="ww"))
                for p in here:
                    parts.append(p.assign(target=target, horizon=h, target_week=tw))
        if parts:
            out = pd.concat(parts, ignore_index=True)
            out["origin"] = o
            out.to_parquet(path, index=False)
        print(f"  {o.date()} ({i + 1}/{len(origins)}): {time.time() - t0:.0f}s")
    print("\nAll forecasts saved in data/forecasts/.")


# =============================================================================
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("features", "tune", "forecast"):
        print("Usage: python scripts/05_forecast.py features|tune|forecast")
        return
    stage = sys.argv[1]
    feat_path = os.path.join(PROC, "features.parquet")
    versions, truth, ww, states, window = load_inputs()

    if stage == "features":
        print("Building features for every Friday ...")
        feats = build_features(versions, ww)
        feats.to_parquet(feat_path, index=False)
        print(f"\nSaved {len(feats):,} rows to {feat_path}")
        have = feats.dropna(subset=["ww_lvl_s1"])
        print("Rows with wastewater features, by target:")
        print(have.groupby("target")["anchor"].agg(["size", "min", "max"]))
        return

    feats = pd.read_parquet(feat_path)
    if stage == "tune":
        print("Tuning on the 2023-24 season (flu and COVID-19) ...")
        tune(feats, versions, truth, states)
    else:
        forecast_all(feats, versions, states, window)


if __name__ == "__main__":
    main()
