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
| 4 | Time-varying transmission; network size vs. response (AIC); sensitivity analysis (OAT refits, elasticities, LHS–PRCC) | done |
| 5 | Optimal control: timing of catch-up vaccination and isolation; counterfactuals | next |
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
    fitting.py                      likelihood, optimiser, summaries (Stages 3-4)
    04_fit_outbreaks.py             fits, profiles, Texas coverage check, 2 figures (Stage 3)
    05_time_varying.py              network vs. response: 3 explanations compared (Stage 4)
    06_sensitivity.py               OAT refits, elasticities, LHS-PRCC (Stage 4)
  data/raw/                         JHU files (downloaded automatically; not on GitHub)
  data/processed/outbreaks_weekly.csv
  results/
    parameter_table.csv             Stage 2
    fit_estimates.csv               Stage 3: main estimates with 95% intervals
    profile_R0.csv                  Stage 3: R0 profile (identifiability evidence)
    profile_theta.csv               Stage 3: theta profile (confidence intervals)
    texas_coverage_check.csv        Stage 3: West Texas coverage sensitivity
    time_varying_models.csv         Stage 4: H1/H2/H3 with AIC
    sensitivity_oat.csv             Stage 4: one-at-a-time refits
    elasticities.csv                Stage 4: normalised sensitivity indices
    prcc.csv                        Stage 4: LHS-PRCC
  figures/
    fig_outbreak_curves.png         weekly case data
    fig_critical_coverage.png       coverage the community needs vs. mixing (illustrative)
    fig_profiles.png                R0 profile: fit, best theta, critical coverage
    fig_model_fits.png              data vs. fitted model
    fig_time_varying_fits.png       the three explanations vs. data
    fig_prcc.png                    PRCC bars for R_v and c2*
  paper/                            the LaTeX paper (kept local; not on GitHub)
    main.tex                        complete paper draft (Overleaf: pdfLaTeX)
    references.bib
    figures/                        copies of the six figures, for Overleaf
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
python src\05_time_varying.py
python src\06_sensitivity.py
```

Scripts 04, 05 and 06 each run the three outbreaks in parallel and take about
5–15 minutes each. They print a line as each step finishes. Run them in this
order: 06 reads the Stage 3 estimates written by 04.

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

## Results (Stage 4)

### Why did the outbreaks stop? (`05_time_varying.py`)

Transmission may change once by a factor m on day tc. Three explanations, R0 = 15,
compared by AIC (lower is better; differences under ~2 mean "cannot tell"):

| | H1 network only | H2 response only | H3 network + change |
|---|---|---|---|
| West Texas | dAIC 3.2 | 11.2 | 0.0* |
| Upstate SC | 1.7 | 0.0** | 1.3 (×2.84 on 27 Dec) |
| Short Creek | 0.0 | 4.1 | 2.6 |

\* Not credible: s1 sits at its 0.08 cap, so the general population sustains spread by
itself and carries 72% of cases, contradicting the MMWR description.
\** Not credible: needs 51% of cases outside the community (officials reported ~90%
inside) and a 95% cut in transmission at the edge of the search range.

The South Carolina H3 change falls in the holiday reporting-backlog week, so it is
most likely reporting, not transmission. **Conclusion:** a response alone does not
explain the outbreaks, and H1 (Stage 3) stays the main model.

### Sensitivity (`06_sensitivity.py`)

- **Refits with fixed inputs changed** (latent 8/12 d, infectious 6/9 d, two-dose
  efficacy 0.93/0.99, weekly data): h* stays within about 4 points of baseline
  (West Texas 66–70%, SC 55–62%, Short Creek 36–42%). R_v moves most with the
  latent and infectious periods: the data fix the growth rate, and a longer
  generation interval means a higher R for the same growth (Wallinga & Lipsitch 2007).
- **Elasticities** match Proposition 3: R_v index for R0 is exactly 1; c2* index
  for R0 is 1/(R0·theta − 1) = 1.33, 0.55, 0.34 (computed 1.32, 0.53, 0.33); for
  efficacy, exactly −1.
- **LHS–PRCC** (2,000 samples): R_v is driven by theta (0.96), R0 (0.92) and
  community coverage (−0.80); c2* by R0 (0.79), theta (0.67) and s1 (0.64), and
  not at all by current coverage (−0.02, p = 0.44), as it should be.

### Bug fixed in Stage 4

The negative-binomial dispersion k could run off towards infinity, where the
`gammaln` terms lose precision and the log-likelihood can even turn positive
(impossible). `fitting.py` now rejects k above 10,000. Stage 3 results are
unchanged by the fix; this was checked by rerunning `04_fit_outbreaks.py`.

### Known weaknesses

- South Carolina's two-phase timing is not captured; most likely a reporting artefact.
  A rash-onset series for South Carolina would settle it.
- Short Creek's several waves: the model fits one smooth curve.
- The case curves cannot fully exclude some contribution from the public-health
  response to how each outbreak ended.

### Inputs to verify before submission

- Washington County, UT coverage (79%): taken from a news report with no year stated.
- Hildale 2020 census count (1,127, down from 2,726 in 2010): possible undercount.
- West Texas: there is no published size for the affected community.
- Bibliography entries marked `CHECK` in `paper/references.bib`.
