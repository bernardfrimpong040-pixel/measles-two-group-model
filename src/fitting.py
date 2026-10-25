"""
Tools for fitting the two-group model to an outbreak's case counts.

The idea in plain words
-----------------------
1. Pick values for the unknown parameters.
2. Run the ODE model and read off how many people became infectious in
   each 2-week period (the model's "expected cases").
3. Score how well the expected cases match the reported cases using the
   negative-binomial likelihood (a count distribution that allows extra
   noise, e.g. reporting backlogs).
4. Let an optimiser change the parameters to get the best score, starting
   from several guesses so it does not get stuck in a poor solution.

Unknown parameters
------------------
  R0     basic reproduction number                      [8, 25]
  theta  share of contacts kept inside the community    [0, 1]
  s1     susceptible fraction of the general population [0, 0.08]
  N2eff  effective size of the community's transmission network,
         i.e. the part of the community the outbreak could reach
  I0     number of infected people at the start of the series
  k      negative-binomial dispersion (small k = noisy reporting)

Any of these can be held FIXED by passing it in the dictionary `fixed`,
e.g. fixed={"R0": 15}.  That is how the profile likelihoods are built.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import gammaln, expit, logit

from model import rhs, reproduction_number, critical_s2

DAYS_PER_PERIOD = 14          # we fit 2-week totals
SIGMA, GAMMA = 1 / 10, 1 / 8  # latent and infectious periods (Stage 2, CDC)
EPS1, EPS2 = 0.93, 0.97       # vaccine efficacy (Stage 2, CDC)

# Each parameter: (lower bound, upper bound, type of transformation)
#   "logistic": any real number x -> lower + (upper - lower) * expit(x)
#   "log"     : any real number x -> exp(x)   (only needs to be positive)
BOUNDS = {
    "R0":    (8.0, 25.0, "logistic"),
    "theta": (0.0, 1.0, "logistic"),
    "s1":    (0.0, 0.08, "logistic"),   # 0.08 = 1 - 0.92 (NHANES serology)
    "N2eff": None,                       # set per outbreak (see setting)
    "I0":    (None, None, "log"),
    "k":     (None, None, "log"),
}
ORDER = ["R0", "theta", "s1", "N2eff", "I0", "k"]


# ---------------------------------------------------------------- data
def biweekly(weekly_cases):
    """Add weeks in pairs: [w0+w1, w2+w3, ...]. A last odd week is dropped."""
    weekly_cases = np.asarray(weekly_cases, dtype=float)
    n = len(weekly_cases) // 2 * 2
    return weekly_cases[:n].reshape(-1, 2).sum(axis=1)


# ---------------------------------------------------------------- parameters
def _bounds(name, setting):
    if name == "N2eff":
        lo, hi = setting["N2eff_bounds"]
        return lo, hi, "logistic"
    return BOUNDS[name]


def to_real(name, value, setting):
    """parameter value -> unbounded number the optimiser works with"""
    lo, hi, kind = _bounds(name, setting)
    if kind == "log":
        return np.log(value)
    frac = np.clip((value - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return logit(frac)


def from_real(name, x, setting):
    """unbounded number -> parameter value inside its bounds"""
    lo, hi, kind = _bounds(name, setting)
    if kind == "log":
        return np.exp(x)
    return lo + (hi - lo) * expit(x)


def free_names(fixed):
    return [n for n in ORDER if n not in fixed]


def unpack(x, setting, fixed):
    par = dict(fixed)
    for name, xi in zip(free_names(fixed), x):
        par[name] = from_real(name, xi, setting)
    return par


def pack(par, setting, fixed):
    return np.array([to_real(n, par[n], setting) for n in free_names(fixed)])


# ---------------------------------------------------------------- model run
def run_model(par, setting, n_periods):
    """Run the ODE model; return expected cases per 2-week period, split
    into (general population, community network).

    setting holds the fixed, sourced numbers for one outbreak:
      N_total  everyone in the outbreak area (general population + community)
      c22      two-dose MMR coverage in the community (Stage 2)
    People outside the community network belong to the general population,
    so N1 = N_total - N2eff keeps the total population fixed.
    """
    N2 = par["N2eff"]
    N1 = setting["N_total"] - N2
    c22 = setting["c22"]
    p = dict(beta=par["R0"] * GAMMA, theta=par["theta"], sigma=SIGMA, gamma=GAMMA,
             eps1=EPS1, eps2=EPS2, N1=N1, N2=N2)

    # General population: s1*N1 susceptible, the rest immune (put in R).
    # Community network: c22 with two doses, the rest unvaccinated;
    # the seed I0 is split equally between latent (E) and infectious (I).
    I0 = par["I0"]
    y0 = [par["s1"] * N1, 0, 0, 0, 0, (1 - par["s1"]) * N1, 0,
          (1 - c22) * N2 - I0, 0, c22 * N2, I0 / 2, I0 / 2, 0, 0]

    t_edges = np.arange(n_periods + 1) * DAYS_PER_PERIOD
    sol = solve_ivp(rhs, (0, t_edges[-1]), y0, args=(p, (0.0, 0.0)),
                    method="LSODA", t_eval=t_edges, rtol=1e-6, atol=1e-6)
    C1, C2 = sol.y[6], sol.y[13]       # cumulative people who became infectious
    return np.diff(C1), np.diff(C2)


# ---------------------------------------------------------------- likelihood
def negbin_loglik(observed, expected, k):
    """Negative-binomial log-likelihood, mean 'expected', dispersion k.
    Variance = mean + mean^2 / k, so a small k allows very noisy counts."""
    mu = np.maximum(expected, 1e-8)
    y = observed
    return np.sum(gammaln(y + k) - gammaln(k) - gammaln(y + 1)
                  + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def neg_loglik(x, cases, setting, fixed):
    par = unpack(x, setting, fixed)
    if par["I0"] >= (1 - setting["c22"]) * par["N2eff"]:
        return 1e10                                   # seed larger than the network
    try:
        e1, e2 = run_model(par, setting, len(cases))
    except Exception:
        return 1e10
    if len(e1) != len(cases) or not np.all(np.isfinite(e1 + e2)):
        return 1e10
    return -negbin_loglik(cases, e1 + e2, par["k"])


# ---------------------------------------------------------------- fitting
def random_start(setting, rng):
    """A random but sensible first guess for every parameter."""
    lo, hi = setting["N2eff_bounds"]
    return dict(R0=rng.uniform(10, 20), theta=rng.uniform(0.1, 0.6),
                s1=rng.uniform(0.002, 0.05),
                N2eff=np.exp(rng.uniform(np.log(lo * 1.5), np.log(hi * 0.5))),
                I0=np.exp(rng.uniform(np.log(0.5), np.log(10))),
                k=rng.uniform(1, 10))


def fit(cases, setting, fixed=None, n_starts=6, warm_start=None, seed=1):
    """Maximum-likelihood fit. Returns (parameters, log-likelihood).

    fixed      : parameters held at given values, e.g. {"R0": 15}
    warm_start : a previous solution to try first (used in profiles)
    """
    fixed = dict(fixed or {})
    rng = np.random.default_rng(seed)
    starts = []
    if warm_start is not None:
        starts.append({**warm_start, **fixed})
    while len(starts) < n_starts:
        starts.append({**random_start(setting, rng), **fixed})

    best = None
    for start in starts:
        x0 = pack(start, setting, fixed)
        res = minimize(neg_loglik, x0, args=(cases, setting, fixed),
                       method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-4, fatol=1e-5))
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x, setting, fixed), -best.fun


# ---------------------------------------------------------------- summaries
def summaries(par, setting, cases):
    """Quantities we report for a fitted parameter set."""
    N = setting["N_total"]
    p1 = 1 - par["N2eff"] / N
    s2 = 1 - EPS2 * setting["c22"]
    Rv = reproduction_number(par["R0"], par["theta"], p1, par["s1"], s2)
    s2_star = critical_s2(par["R0"], par["theta"], p1, par["s1"])
    e1, e2 = run_model(par, setting, len(cases))
    return dict(
        R0_theta=par["R0"] * par["theta"],            # within-network contacts
        Rv=Rv,                                        # reproduction number at the start
        c2_star=min(1.0, (1 - s2_star) / EPS2),      # two-dose coverage the network needs
        share_susceptibles=max(0.0, 1 - s2_star / s2),  # of today's susceptibles, share to immunise
        community_share=e2.sum() / (e1 + e2).sum(),  # model: share of cases in the network
        predicted_total=(e1 + e2).sum(),
    )
