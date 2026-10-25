# common.py
# =============================================================================
# Shared settings and helper functions used by the modelling and scoring
# scripts (05, 06). Keeping them in one place guarantees every script uses the
# same definitions.
# =============================================================================

import numpy as np
import pandas as pd

# The 23 quantile levels used by the CDC forecast hubs
QUANTILES = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                      0.95, 0.975, 0.99])

# Weeks when NHSN reporting was not mandatory (paused 28 Apr - 1 Nov 2024).
# Counts in these weeks are incomplete, so they are never used for training.
NHSN_PAUSE_FIRST_WEEK_END = pd.Timestamp("2024-05-04")
NHSN_PAUSE_LAST_WEEK_END = pd.Timestamp("2024-11-02")


def weighted_interval_score(y, q, levels=QUANTILES):
    """Weighted interval score (Bracher et al. 2021, PLoS Comput Biol 17:e1008618).

    y      : observed values, shape (n,)
    q      : predicted quantiles, shape (n, len(levels)), levels sorted ascending
    levels : the quantile levels; must be the median plus symmetric pairs
             (alpha/2, 1 - alpha/2).

    WIS = 1/(K + 1/2) * ( 1/2 * |y - median| + sum_k alpha_k/2 * IS_alpha_k )
    where the interval score of the central (1 - alpha) interval [l, u] is
    IS_alpha = (u - l) + 2/alpha * (l - y) * 1(y < l) + 2/alpha * (y - u) * 1(y > u).

    Returns a dict of arrays: wis and its three parts (spread, over-prediction,
    under-prediction), which add up to wis."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    levels = np.asarray(levels, dtype=float)
    n_lev = len(levels)
    mid = n_lev // 2
    if not np.isclose(levels[mid], 0.5) or not np.allclose(levels + levels[::-1], 1):
        raise ValueError("levels must be symmetric around a median of 0.5")
    K = mid                                           # number of central intervals
    median = q[:, mid]

    spread = np.zeros_like(y)
    over = 0.5 * np.maximum(median - y, 0)            # median term split into
    under = 0.5 * np.maximum(y - median, 0)           # over/under prediction
    for k in range(K):
        alpha = 2 * levels[k]                         # e.g. level 0.025 -> 95% interval
        lower, upper = q[:, k], q[:, n_lev - 1 - k]
        w = alpha / 2
        spread += w * (upper - lower)
        over += w * (2 / alpha) * np.maximum(lower - y, 0)
        under += w * (2 / alpha) * np.maximum(y - upper, 0)
    scale = 1 / (K + 0.5)
    return {"wis": scale * (spread + over + under),
            "spread": scale * spread,
            "overprediction": scale * over,
            "underprediction": scale * under}


def interval_coverage(y, q, level, levels=QUANTILES):
    """1 if y lies inside the central `level` interval (e.g. 0.95), else 0."""
    lo = int(np.argmin(np.abs(levels - (1 - level) / 2)))
    hi = int(np.argmin(np.abs(levels - (1 + level) / 2)))
    return ((np.asarray(y) >= q[:, lo]) & (np.asarray(y) <= q[:, hi])).astype(int)
