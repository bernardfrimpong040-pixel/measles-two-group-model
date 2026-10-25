# Measles in under-vaccinated communities: a two-group model of three US outbreaks

Working title: *Why some measles outbreaks explode: a two-group SV1V2EIR model with
vaccine-exempt communities, fitted to the West Texas, Upstate South Carolina and
Short Creek (Utah–Arizona) outbreaks, 2025–26.*

Everything is in Python. LaTeX is used only for the paper itself.

## Folder layout

```
measles_paper/
  README.md
  requirements.txt
  src/
    01_prepare_data.py              builds the outbreak series + Figure 1
    model.py                        the ODE model, R_v and critical coverage
    02_check_reproduction_number.py numerical checks of every formula in Section 2
  data/raw/                         JHU files (downloaded automatically; not on GitHub)
  data/processed/outbreaks_weekly.csv
  figures/fig1_outbreak_curves.png
  paper/                            the LaTeX paper (kept local; not on GitHub)
    sec_model.tex                   Section 2 (model + proofs), \input into the paper
    references.bib
    preview.tex                     compiles Section 2 on its own (Overleaf: pdfLaTeX)
```

The `paper/` folder is listed in `.gitignore`, so the paper stays off GitHub
until it is submitted. The repository tracks the code, data and figures.

## Running it (Windows, VS Code terminal, from the project folder)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\01_prepare_data.py
python src\02_check_reproduction_number.py
```

## Data

All case counts come from the Johns Hopkins University US Measles Data Repository
(github.com/CSSEGISandData/measles_data; Ahmadi, Dong & Gardner, *JAMA* 2025,
doi:10.1001/jama.2025.17812), pinned to commit `c361b38` (23 Oct 2026).

| Outbreak | Series used | Weeks | Cases in series | Official total |
|---|---|---|---|---|
| West Texas, Jan–Aug 2025 | Texas statewide, **rash-onset** week | 34 | 802 | 762 (Texas DSHS) |
| Upstate SC, Oct 2025–Apr 2026 | Spartanburg + Cherokee + Greenville + Pickens, **report** week | 34 | 990 | 997 (SC DPH) |
| Short Creek UT–AZ, Jun 2025–May 2026 | Southwest Utah HD + Mohave Co. AZ, **report** week | 50 | 550 | none published as one outbreak |

### Known data issues and how we will handle them

1. **Texas includes about 40 non-outbreak cases** (travel-related and unlinked
   cases elsewhere in the state). Small next to 762; we state it as a limitation.
2. **South Carolina is by report date, not rash onset.** The JHU README says the
   state file uses rash onset, but the SC column is identical to the Spartanburg
   report-date series. It contains a holiday backlog (174 cases in the week of
   28 Dec 2025, 63 the next week) and a 57-case jump in mid-March 2026 after
   weeks of zeros, which looks like a reclassification.
3. **Short Creek has the same holiday backlog** (79 cases in the week of 28 Dec).

Plan: fit 2-week totals with a negative-binomial likelihood (the dispersion term
absorbs reporting noise), and check that the estimates barely change when we
fit cumulative cases instead. If SC DPH publishes a rash-onset epidemic curve, we
switch to it.

## What still needs sourcing (next step)

Every parameter value will carry a citation. To be filled in:
- size of each under-vaccinated community (N2) and of the surrounding population (N1)
- MMR coverage in each group (kindergarten exemption data by county / school)
- one-dose and two-dose vaccine efficacy (eps1, eps2)
- latent and infectious periods (1/sigma, 1/gamma)

## Plan to submission (target: 31 Dec 2026)

| Dates | Work |
|---|---|
| 25 Oct – 1 Nov | Data pipeline ✓, model and R_v ✓; source parameters; literature review |
| 2 – 15 Nov | Fit each outbreak; profile likelihood for theta vs community coverage |
| 16 – 22 Nov | Sensitivity analysis (LHS–PRCC); final-size relation |
| 23 Nov – 6 Dec | Optimal control: timing of catch-up vaccination and isolation; counterfactuals |
| 7 – 20 Dec | Write the paper |
| 21 – 31 Dec | Polish, check every number against the code, submit |
