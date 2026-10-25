# Measles in under-vaccinated communities: a two-group model of three US outbreaks

Working title: *Why some measles outbreaks explode: a two-group model of vaccine-exempt
communities, fitted to three US outbreaks, 2025–2026.*

Everything is in Python. LaTeX is used only for the paper itself.

## Progress

| Stage | What | Status |
|---|---|---|
| 1 | Outbreak case data; two-group SV1V2EIR model; reproduction number and critical-coverage proofs, checked numerically | done |
| 2 | Every parameter sourced with a citation; first critical-coverage curves | done |
| 3 | Fit R0, theta, s1 (and Texas community size and coverage) to each outbreak | next |
| 4 | Sensitivity analysis (LHS–PRCC), final-size relation | |
| 5 | Optimal control: timing of catch-up vaccination and isolation; counterfactuals | |
| 6 | Write-up and submission (target 31 Dec 2026) | |

## Folder layout

```
measles_paper/
  README.md
  requirements.txt
  .gitignore
  src/
    01_prepare_data.py              builds the outbreak series + Figure 1
    model.py                        the ODE model, R_v and critical coverage
    02_check_reproduction_number.py numerical checks of every formula in Section 2
    parameters.py                   every parameter value with its source (Stage 2)
    03_parameter_table.py           parameter table, check 1, Figure 2 (Stage 2)
  data/raw/                         JHU files (downloaded automatically; not on GitHub)
  data/processed/outbreaks_weekly.csv
  results/parameter_table.csv
  figures/
    fig1_outbreak_curves.png
    fig2_critical_coverage.png
  paper/                            the LaTeX paper (kept local; not on GitHub)
    main.tex                        complete paper draft (Overleaf: pdfLaTeX)
    references.bib
    figures/                        copies of the two figures, for Overleaf
    main.pdf                        compiled draft, for reading
```

The `paper/` folder is listed in `.gitignore`, so the paper stays off GitHub
until it is submitted. The repository tracks the code, data, results and figures.

## Running it (Windows, PowerShell, from the project folder)

```
.venv\Scripts\activate
python src\01_prepare_data.py
python src\02_check_reproduction_number.py
python src\03_parameter_table.py
```

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

Plan: fit 2-week totals with a negative-binomial likelihood (the dispersion term
absorbs reporting noise), and check that estimates barely change when we fit
cumulative cases instead.

## Parameters (Stage 2)

Full table with ranges and notes: `src/parameters.py` and `results/parameter_table.csv`.

| Parameter | Value | Source |
|---|---|---|
| Latent period | 10 days (rash ~14 days after exposure, infectious from 4 days before rash) | CDC clinical overview |
| Infectious period | 8 days (4 before to 4 after rash) | CDC clinical overview |
| One-dose / two-dose efficacy | 0.93 / 0.97 | CDC vaccine recommendations |
| R0 | estimated, search range 8–25 | Guerra et al. 2017, *Lancet Infect Dis* |
| General-population susceptible fraction s1 | estimated, at most 0.08 | Lebo et al. 2015: 92.0% seropositive |
| West Texas | Gaines County 23,956; community size and coverage estimated | Census; CDC MMWR 75(23), 2026 |
| Upstate SC | community 15,000 at 17% coverage; county coverage 88.9% | SC DPH via Healthbeat; NEJM 2026 |
| Short Creek | community 3,605 (2020 census) at 7.7% coverage | Census; NBC News (AZ DHS data) |

### Key finding of Stage 2

Kindergarten coverage cannot be used as the whole general population's immunity.
If it were, the general population in all three areas would have R0·s1 > 1
(1.44, 2.07, 3.51) and could sustain measles on its own, which did not happen.
So s1 is estimated in Stage 3. What makes it identifiable is the split of cases
between the community and everyone else.

### Inputs to verify before submission

- Washington County, UT coverage (79%): taken from a news report with no year stated.
- Hildale 2020 census count (1,127, down from 2,726 in 2010): possible undercount.
- West Texas: there is no published size for the affected community.
- Bibliography entries marked `CHECK` in `paper/references.bib`.
