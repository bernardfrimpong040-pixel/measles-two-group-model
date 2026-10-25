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
| 5 | Optimal control of vaccination and isolation (Pontryagin, forward-backward sweep); timing counterfactuals; realistic scenarios | done |
| 6 | Introduction and literature review, discussion, abstract; number checker; journal version (Bulletin of Mathematical Biology); cover letter | done |

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
    control.py                      controlled model, doses, Pontryagin sweep (Stage 5)
    07_control.py                   optimal strategies, timing, scenarios, frontier (Stage 5)
    08_check_control.py             8 checks of the control code, Proposition 4 and optimality (Stages 5-6)
    09_check_paper_numbers.py       checks that every table and key number in main.tex matches results/ (Stage 6)
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
    control_strategies.csv          Stage 5: optimal strategies from detection
    control_profiles.csv            Stage 5: optimal controls day by day
    control_timing.csv              Stage 5: response start vs cases averted
    control_scenarios.csv           Stage 5: realistic scenarios
    control_frontier.csv            Stage 5: doses vs cases averted
    under_reporting.csv             Stage 6: refits with case counts x2 (SC) and x3 (Short Creek)
    nonconvexity_sc.csv             Stage 6: constant isolation in SC (non-convexity table)
  figures/
    fig_outbreak_curves.png         weekly case data
    fig_critical_coverage.png       coverage the community needs vs. mixing (illustrative)
    fig_profiles.png                R0 profile: fit, best theta, critical coverage
    fig_model_fits.png              data vs. fitted model
    fig_time_varying_fits.png       the three explanations vs. data
    fig_prcc.png                    PRCC bars for R_v and c2*
    fig_optimal_controls.png        epidemic curves and optimal controls
    fig_control_timing.png          cases averted vs response start
    fig_dose_frontier.png           doses vs cases averted
  paper/                            the LaTeX paper (kept local; not on GitHub)
    main.tex                        complete paper draft (Overleaf: pdfLaTeX)
    references.bib
    figures/                        copies of the nine figures, for Overleaf
    main.pdf                        compiled working version, for reading
    make_bmb.py                     builds the journal version from main.tex
    sn_template/                    Springer Nature class and reference style
    bmb/                            journal version: main_bmb.tex, .bib, class, style, figures
    submission/cover_letter.tex     cover letter (and its PDF)
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
python src\07_control.py
python src\08_check_control.py
python src\09_check_paper_numbers.py
```

Scripts 04–07 each run the three outbreaks in parallel. 04–06 take about
5–15 minutes each and 07 about 15–30 minutes. They print a line as each step
finishes. Run them in this order: 06, 07 and 08 read the Stage 3 estimates
written by 04.

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

## Results (Stage 5): what response would have worked?

Two controls in the community network: catch-up vaccination u1 (per day) and isolation
u2 (share of infectiousness removed). Doses are counted explicitly: a campaign reaches
every unvaccinated person not currently infectious (S2 + E2 + R2), so doses to people
already infected or recovered are wasted. Limits: u1 ≤ 0.01/day (intensive; assumption);
u2 ≤ 0.5 (isolation at rash onset removes at most 4 of 8 infectious days, CDC); 0.8 with
quarantine (assumption). Observed uptake at the SC community school, 17% → 21% in a year,
gives u1 = 1.4 × 10⁻⁴/day. The response starts at detection (3 cumulative cases).

**Optimal strategies from detection** (`results/control_strategies.csv`):

| | Vaccination only | Isolation only | Both |
|---|---|---|---|
| West Texas (807 cases) | 24.6% averted, 315 doses (175 wasted) | 42.1% | 73.6% |
| Upstate SC (948) | 77.0%, 882 doses (162 wasted) | 63.1% | 97.5%, 539 doses (9 wasted) |
| Short Creek (584) | 87.3%, 266 doses | 95.3% | 97.0% |

**Realistic scenarios** (constant from detection): observed uptake averts 0.4%, 1.3%,
3.1%. Isolation + quarantine (0.8) averts 89.9%, 99.0%, 98.3%.

**Timing** (combined response, % averted when started 0 / 2 / 4 / 8 weeks after
detection): West Texas 74 / 55 / 36 / 10; SC 98 / 95 / 89 / 62; Short Creek
97 / 95 / 93 / 84. Delay costs follow R_v.

### Key findings of Stage 5

1. **Proposition 4 (critical isolation):** in the small-community limit, isolation
   multiplies R_v by (1 − u2), so the isolation needed is 1 − 1/R_v, the same number as
   h*. Exact values: West Texas 0.684, SC 0.583, Short Creek 0.392 (check 6 in
   `08_check_control.py`). So isolation at rash onset (≤ 0.5) can stop Short Creek but
   not West Texas.
2. **A weak response can be worse than none.** Partial isolation mostly delays
   infections and even uses *more* isolation-days (SC: 1,868 at u2 = 0.4 vs 79 at 0.7).
   The optimisation problem is non-convex, so the sweep runs from 3 starting controls.
3. **The two controls are complements:** isolation slows the outbreak so vaccination
   reaches people before they are infected (SC: 9 wasted doses with both, 162 without).
4. **Optimal campaigns stop early**, saving 57–92% of doses compared with a constant
   campaign running for the whole horizon, for the same benefit.

### Pitfalls found and fixed in Stage 5

- **Local optimum:** a sweep started from no control stays at "do nothing". Fixed by
  starting from none, half and full control and keeping the best.
- **Short horizon:** a response holding R_v just above 1 makes a slow burn; stopping
  8 weeks after the data undercounted SC isolation-only cases (190 vs 350). Fixed with a
  3-year horizon (incidence below 0.001/day at the end; 6 years gives the same totals).
- **Regularisation driving the answer:** a plain eta·u²/2 term kept isolation at 10%
  because early on there are few cases. Fixed by scaling it by the people reached.

### Checks (`08_check_control.py`)

1. Controlled model with no isolation equals the original model exactly.
2. Complex-step Jacobian vs finite differences: 4 × 10⁻¹⁰.
3. Hand-derived adjoint equations (as printed in the paper) vs code: 10⁻¹⁵.
4. Adjoint gradient vs re-running the model: 5 × 10⁻⁵ (vaccination), 4 × 10⁻⁶ (isolation).
5. No control: RK4 reproduces the Stage 3 SC fit (945.85 cases).
6. Proposition 4 against the exact eigenvalue.
7. Each optimal combined strategy beats all 121 constant strategies (Stage 6).
8. Non-convexity table for South Carolina, saved to results/nonconvexity_sc.csv (Stage 6).

### Known weaknesses

- Isolation + quarantine at 0.8 assumes high compliance, which may not hold in
  communities that distrust public-health authorities.
- South Carolina's two-phase timing is not captured; most likely a reporting artefact.
  A rash-onset series for South Carolina would settle it.
- Short Creek's several waves: the model fits one smooth curve.
- The case curves cannot fully exclude some contribution from the public-health
  response to how each outbreak ended.

## Stage 6: final paper and submission

### What was added
- **Introduction** with a verified literature review (heterogeneity and preferential
  mixing: Glasser 2016, Feng 2015, Masters 2020, Liu 2015; 2025 outbreak models:
  Chukwu 2026, Lao 2025, González-Parra 2025; identifiability: Tuncer & Le 2018,
  Chowell 2017; measles optimal control: Jaharuddin & Bakhtiar 2020, Kuddus 2023,
  Aldila 2025; projections: Kiang 2025).
- **Discussion** (summary, relation to previous work, implications, limitations, future
  work), **Conclusion**, **abstract** (246 words) and **keywords**.
- **Declarations**: data and code availability, funding, competing interests, ethics,
  AI use (described in Methods, Section 3.5, as the journal requires), contributions.
- **Under-reporting test**: with case counts doubled (SC) or tripled (Short Creek),
  R_v and h* barely move (2.40 → 2.42, 1.64 → 1.63; 58.3% → 58.6%, 39.2% → 39.6%);
  only the fitted network grows.
- **Check 7**: each optimal combined strategy beats all 121 constant strategies
  (J = 275.0 vs 310.1, 44.1 vs 60.2, 20.0 vs 26.8).
- **`09_check_paper_numbers.py`**: rebuilds every table and 86 key numbers from
  `results/` and searches for them in `main.tex`. It caught three errors that were then
  fixed (a 70% that should be 71%, and two double-rounded table entries).

### Journal: Bulletin of Mathematical Biology (back-up: Mathematical Biosciences)
- Fit: theory + fitted outbreaks + optimal control. No fee on the subscription route
  (open access would cost US$3,190). LaTeX accepted. Median 8 days to first decision.
- Requirements met: abstract 150–250 words; 4–6 keywords; data availability statement;
  Statements and Declarations; AI use documented in the Methods.
- `paper/bmb/` holds the journal version, generated by `make_bmb.py`. Always edit
  `paper/main.tex` and regenerate; never edit `main_bmb.tex` by hand.

### Submission checklist
1. Confirm the two references I could not open (they compile, but are incomplete):
   - `nejm2026sc`: open https://www.nejm.org/doi/full/10.1056/NEJMc2605131 and add the
     authors, volume, issue and pages.
   - `nbc2025shortcreek`: confirm the headline, authors (probably Marina Kopf and Joe
     Murphy) and date (late September 2025).
2. Read the whole paper once, slowly, as its author: every sentence must be one you can
   defend. Check that the AI-use statement in Section 3.5 is accurate for how you worked.
3. Make the GitHub repository **public**, then archive it on Zenodo (zenodo.org → log in
   with GitHub → enable the repository → create a release) to get a DOI. Add the DOI
   to the Data availability paragraph.
4. Run the full pipeline once more from a clean folder, then
   `python src\09_check_paper_numbers.py` (must say 86 of 86).
5. In `paper/`: `python make_bmb.py`, then compile `bmb/main_bmb.tex` on Overleaf
   (pdfLaTeX) and read the PDF.
6. Fill in the date on `submission/cover_letter.tex`.
7. Submit at the journal's site (Springer Nature submission system): upload
   `main_bmb.tex`, `references.bib`, `sn-jnl.cls`, `sn-mathphys-num.bst`, the nine figure
   PNGs, and the compiled PDF; paste the cover letter.
8. Optional suggested reviewers (look up current affiliations and emails yourself, and
   avoid anyone you have worked with): authors of the papers we build on, for example
   Zhilan Feng or John Glasser (preferential mixing and measles), Gerardo Chowell
   (fitting outbreak models), Necibe Tuncer (identifiability), Marisa Eisenberg
   (measles clustering, identifiability).

### Inputs to state as limitations (already in the paper)
- Washington County, UT (79%) is NBC News's own figure (with Stanford University), with
  no school year given. The nearest official figure is Utah DHHS 2021/22: 84.4% of
  kindergartners adequately immunised (not MMR-specific). It only sets the general
  population's coverage, which the fit does not use (s1 is estimated).
- Hildale's 2020 census count is 1,127 (Wikipedia, from data.census.gov); another source
  gives 1,145. It only affects the total population and the network's upper bound.
- West Texas: no published size or coverage for the affected community.
