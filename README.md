# Measles in under-vaccinated communities: a two-group model of three US outbreaks

Working title: *Why some measles outbreaks explode: a two-group model of vaccine-exempt
communities, fitted to three US outbreaks, 2025–2026.*

Everything is in Python. LaTeX is used only for the paper itself.

## Progress

| Stage | What | Status |
|---|---|---|
| 1 | Outbreak case data; two-group SV1V2EIR model; reproduction number and critical-coverage proofs, checked numerically | done |
| 2 | Every parameter sourced with a citation; first critical-coverage curves | done |
| 3 | Negative-binomial fits to all three outbreaks; profile likelihoods; identifiability result (Proposition 3) | done |
| 4 | Time-varying transmission; network size vs. response; sensitivity analysis (LHS–PRCC) | next |
| 5 | Optimal control: timing of catch-up vaccination and isolation; counterfactuals | |
| 6 | Write-up and submission (target 31 Dec 2026) | |

## Folder layout

```
measles_paper/
  README.md
  requirements.txt
  .gitignore
  src/
    01_prepare_data.py              builds the outbreak series + case-curve figure
    model.py                        the ODE model, R_v and critical coverage
    02_check_reproduction_number.py numerical checks of every formula in Section 2
    parameters.py                   every parameter value with its source (Stage 2)
    03_parameter_table.py           parameter table, check 1, critical-coverage figure (Stage 2)
    fitting.py                      likelihood, optimiser, summaries (Stage 3)
    04_fit_outbreaks.py             fits, profiles, Texas coverage check, 2 figures (Stage 3)
  data/raw/                         JHU files (downloaded automatically; not on GitHub)
  data/processed/outbreaks_weekly.csv
  results/
    parameter_table.csv             Stage 2
    fit_estimates.csv               Stage 3: main estimates with 95% intervals
    profile_R0.csv                  Stage 3: R0 profile (identifiability evidence)
    profile_theta.csv               Stage 3: theta profile (confidence intervals)
    texas_coverage_check.csv        Stage 3: West Texas coverage sensitivity
  figures/
    fig_outbreak_curves.png         weekly case data
    fig_critical_coverage.png       coverage the community needs vs. mixing (illustrative)
    fig_profiles.png                R0 profile: fit, best theta, critical coverage
    fig_model_fits.png              data vs. fitted model
  paper/                            the LaTeX paper (kept local; not on GitHub)
    main.tex                        complete paper draft (Overleaf: pdfLaTeX)
    references.bib
    figures/                        copies of the four figures, for Overleaf
    main.pdf                        compiled draft, for reading
```

The `paper/` folder is listed in `.gitignore`, so the paper stays off GitHub
until it is submitted. The repository tracks the code, data, results and figures.

Figure files are named after their content, not a number, because the paper's
figure numbers change as sections are added.

## Running it (Windows, PowerShell, from the project folder)

```
.venv\Scripts\activate
python src\01_prepare_data.py
python src\02_check_reproduction_number.py
python src\03_parameter_table.py
python src\04_fit_outbreaks.py
```

`04_fit_outbreaks.py` runs the three outbreaks in parallel and takes about
5–15 minutes. It prints a line as each step finishes.

First-time setup only: `py -3.13 -m venv .venv`, then activate and
`pip install -r requirements.txt`.

## Data

All case counts come from the Johns Hopkins University US Measles Data Repository
(github.com/CSSEGISandData/measles_data; Ahmadi, Dong & Gardner, *JAMA* 2025,
doi:10.1001/jama.2025.17812), pinned to commit `c361b38` (23 Oct 2026).

| Outbreak | Series used | Weeks | Cases in series | Official total |
|---|---|---|---|---|
| West Texas, Jan–Aug 2025 | Texas statewide, **rash-onset** week | 34 | 802 | 762 (Texas DSHS) |
| Upstate SC, Oct 2025–Apr 2026 | Spartanburg + Cherokee + Greenville + Pickens, **report** week | 34 | 990 | 997 (SC DPH) |
| Short Creek UT–AZ, Jun 2025–May 2026 | Southwest Utah HD + Mohave Co. AZ, **report** week | 50 | 550 | none published as one outbreak |

### Known data issues and how we handle them

1. **Texas includes about 40 non-outbreak cases** (travel-related and unlinked
   cases elsewhere in the state). Small next to 762; stated as a limitation.
2. **South Carolina is by report date, not rash onset.** The JHU README says the
   state file uses rash onset, but the SC column is identical to the Spartanburg
   report-date series. It contains a holiday backlog (174 cases in the week of
   28 Dec 2025) and a 57-case jump in mid-March 2026 after weeks of zeros, which
   looks like a reclassification.
3. **Short Creek has the same holiday backlog** (79 cases in the week of 28 Dec).

We fit 2-week totals with a negative-binomial likelihood; its dispersion term k
absorbs reporting noise (k is small for South Carolina, 0.67, and large for
Texas, 53).

## Parameters (Stage 2)

Full table with ranges and notes: `src/parameters.py` and `results/parameter_table.csv`.

| Parameter | Value | Source |
|---|---|---|
| Latent period | 10 days (rash ~14 days after exposure, infectious from 4 days before rash) | CDC clinical overview |
| Infectious period | 8 days (4 before to 4 after rash) | CDC clinical overview |
| One-dose / two-dose efficacy | 0.93 / 0.97 | CDC vaccine recommendations |
| R0 | not identifiable from case data (Stage 3); held at 15 | Guerra et al. 2017, *Lancet Infect Dis* |
| General-population susceptible fraction s1 | estimated, at most 0.08 | Lebo et al. 2015: 92.0% seropositive |
| West Texas | Gaines County 23,956; community coverage unknown (30% assumed, 10–50% checked) | Census; CDC MMWR 75(23), 2026 |
| Upstate SC | community 15,000 at 17% coverage; county coverage 88.9% | SC DPH via Healthbeat; NEJM 2026 |
| Short Creek | community 3,605 (2020 census; 7,547 in 2010) at 7.7% coverage | Census; NBC News (AZ DHS data) |

Key finding of Stage 2: kindergarten coverage cannot be used as the whole general
population's immunity (it would give R0·s1 = 1.44, 2.07, 3.51, i.e. self-sustaining
spread in the general population, which did not happen), so s1 is estimated.

## Results (Stage 3)

Fitted with R0 held at 15; brackets are 95% profile-likelihood intervals.

| | West Texas | Upstate SC | Short Creek |
|---|---|---|---|
| R_v (start of outbreak) | 3.00 [2.47, 3.71] | 2.40 [1.89, 3.24] | 1.64 [1.46, 2.47] |
| R0·theta | 3.92 [3.09, 4.99] | 2.83 [2.23, 3.83] | 1.75 [1.55, 2.67] |
| Effective network size | 678 [633, 833] | 1,277 [1,060, 1,940] | 375 [136, 1,060] |
| Coverage the network needs, c2* | 80.0%* [74.8, 84.3] | 67.2% [57.6, 76.6] | 45.1% [37.9, 64.7] |
| Share of susceptibles to immunise, h* | 68.4% [61.3, 74.3] | 58.3% [47.1, 69.3] | 39.2% [31.7, 59.8] |
| Reported community coverage | unknown | 17% | 7.7% |
| Cases: model / reported | 806 / 802 | 946 / 990 | 571 / 550 |

\* assumes 30% community coverage; 74.0% at 10% and 86.0% at 50%. h* stays 68–69%.

### Key findings of Stage 3

1. **Why a network size is needed.** With the sourced values (15,000 people at 17%
   coverage), the model predicts about 13,000 South Carolina cases against 990
   observed. The outbreak reached only part of the community, which officials
   described as "a small subset". We estimate the size of that network.
2. **Identifiability.** Across R0 = 8–25 the log-likelihood changes by at most 0.31.
   Case counts cannot separate R0 from theta, but their product R0·theta is stable,
   and so are R_v, c2* and h*. Proposition 3 in the paper proves why: for a small
   community, R_v ≈ R0·theta·s2, c2* ≈ (1 − 1/(R0·theta))/ε2 and h* ≈ 1 − 1/R_v.
3. **Robust quantity.** h* depends only on R_v, which the data measure, so it does not
   depend on the community's (often unknown) coverage. That is confirmed by the West
   Texas check.

### Known weaknesses (to address in Stage 4)

- South Carolina: the model gets the size right but not the timing (slow Oct–Nov,
  then a surge in Dec–Jan). Needs time-varying transmission.
- Short Creek: several waves; the model fits one smooth curve.
- Network size vs. public-health response as the reason each outbreak stopped:
  a constant-rate model cannot tell them apart.
- Check that estimates hold when fitting cumulative cases instead of 2-week totals.

### Inputs to verify before submission

- Washington County, UT coverage (79%): taken from a news report with no year stated.
- Hildale 2020 census count (1,127, down from 2,726 in 2010): possible undercount.
- West Texas: there is no published size for the affected community.
- Bibliography entries marked `CHECK` in `paper/references.bib`.
