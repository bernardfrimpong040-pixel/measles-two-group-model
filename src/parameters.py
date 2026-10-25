"""
Every parameter value used in the paper, with its source.

Each entry is a dictionary:
    value  : the baseline value used in the model
    low    : lower end of the range we will test (None = not used yet)
    high   : upper end of the range we will test
    status : "fixed"      -> taken from a source and kept fixed
             "estimated"  -> will be estimated from the case data in Stage 3
                             (value/low/high are only the starting guess and bounds)
    source : short key that matches an entry in paper/references.bib
    note   : how the number was obtained / what to watch out for

Time unit: days.  Coverage: fraction (0.93 = 93%).
"""

# ---------------------------------------------------------------- disease
# CDC: rash appears about 14 days after exposure; a person is infectious
# from 4 days before to 4 days after rash onset.
#   latent period     = 14 - 4 = 10 days  (exposure -> start of infectiousness)
#   infectious period = 4 + 4  =  8 days
DISEASE = {
    "latent_days": dict(
        value=10.0, low=8.0, high=12.0, status="fixed", source="cdc_clinical",
        note="14-day exposure-to-rash minus 4 infectious days before rash; "
             "range 8-12 is our sensitivity range"),
    "infectious_days": dict(
        value=8.0, low=6.0, high=9.0, status="fixed", source="cdc_clinical",
        note="4 days before to 4 days after rash; range is our sensitivity range"),
    "R0": dict(
        value=15.0, low=8.0, high=25.0, status="estimated", source="guerra2017",
        note="often quoted as 12-18, but Guerra et al. found estimates vary more "
             "widely; wide bounds so the data decide"),
}

# ---------------------------------------------------------------- vaccine
VACCINE = {
    "eps1": dict(value=0.93, low=0.39, high=1.00, status="fixed", source="cdc_vaccine",
                 note="one MMR dose; CDC range 39-100%"),
    "eps2": dict(value=0.97, low=0.67, high=1.00, status="fixed", source="cdc_vaccine",
                 note="two MMR doses; CDC range 67-100%"),
}

# ---------------------------------------------------------------- general-population immunity
# Kindergarten coverage describes 5-year-olds, not the whole population.
# Applying it to everyone makes the general population look so susceptible
# that it could sustain measles on its own -- which did not happen in any of
# the three outbreaks (see src/03_parameter_table.py, check 1).
# So the susceptible fraction of group 1, s1, is ESTIMATED in Stage 3, with
# an upper bound from national serology: 92.0% of the US population had
# measles antibodies in 2009-2010 (NHANES), so at most ~8% are susceptible.
POPULATION_IMMUNITY = {
    "s1": dict(value=0.04, low=0.0, high=0.08, status="estimated", source="lebo2015",
               note="upper bound 1 - 0.920 = 0.08 from NHANES seroprevalence "
                    "(95% CI 90.9-93.0%); 0.04 is a starting guess only"),
}

# ---------------------------------------------------------------- outbreaks
# Group 1 = general population of the outbreak area
# Group 2 = the under-vaccinated community where most cases occurred
# cov1, cov2 = two-dose MMR coverage (kindergarten data; all four states
#              require 2 doses for school entry, so we treat it as two-dose)
OUTBREAKS = {
    "west_texas": {
        "label": "West Texas 2025",
        "N1": dict(value=23_956, low=21_598, high=23_956, status="fixed",
                   source="census_gaines",
                   note="Gaines County: 21,598 (2020 census), 23,956 (2025 est.). "
                        "Cases also occurred in 8 nearby counties (MMWR)"),
        "N2": dict(value=5_000, low=1_000, high=15_000, status="estimated",
                   source="crisp2026",
                   note="NO published size for the close-knit community (MMWR names "
                        "none). Estimate from data; 5,000 is a placeholder only"),
        "cov1": dict(value=0.932, low=0.773, high=0.946, status="fixed",
                     source="crisp2026",
                     note="statewide kindergarten MMR 93.2% (2024-25); affected "
                          "counties ranged 77.3-94.6%"),
        "cov2": dict(value=0.30, low=0.0, high=0.80, status="estimated",
                     source="fortune2025",
                     note="no direct measure: Gaines K-12 exemptions ~14% (2023-24) "
                          "but home-schooled children are NOT counted (DSHS)"),
    },
    "upstate_sc": {
        "label": "Upstate South Carolina 2025-26",
        "N1": dict(value=380_857 - 15_000, low=327_997 - 15_000, high=380_857 - 15_000,
                   status="fixed", source="census_spartanburg",
                   note="Spartanburg County 327,997 (2020), 380,857 (2025 est.) "
                        "minus the community"),
        "N2": dict(value=15_000, low=10_000, high=20_000, status="fixed",
                   source="healthbeat2026",
                   note="SC DPH estimate of the Ukrainian- and Russian-speaking "
                        "community in Spartanburg County (~90% of cases)"),
        "cov1": dict(value=0.889, low=0.889, high=0.90, status="fixed",
                     source="nejm2026sc",
                     note="Spartanburg County MMR coverage 88.9%; all-vaccine "
                          "compliance 90% in 2024-25 (SC Daily Gazette)"),
        "cov2": dict(value=0.17, low=0.17, high=0.21, status="fixed",
                     source="healthbeat2026",
                     note="Global Academy (community charter school): 17% fully "
                          "vaccinated 2024-25, 21% in 2025-26 -- one school as proxy"),
    },
    "short_creek": {
        "label": "Short Creek (UT-AZ) 2025-26",
        "N1": dict(value=228_102 + 287_000 - 3_605, low=None, high=None,
                   status="fixed", source="census_mohave",
                   note="Mohave County 228,102 (2025 est.) + Southwest Utah health "
                        "district >287,000 (NBC) minus the community"),
        "N2": dict(value=3_605, low=3_605, high=7_547, status="fixed",
                   source="census_shortcreek",
                   note="Hildale 1,127 + Colorado City 2,478 (2020 census). The 2010 "
                        "total was 7,547; 2020 may be an undercount"),
        "cov1": dict(value=0.79, low=0.79, high=0.93, status="fixed",
                     source="nbc2025shortcreek",
                     note="Washington County UT kindergarten MMR ~79% (NBC; year not "
                          "stated) -- weakest number in the table, check against UDHHS"),
        "cov2": dict(value=0.077, low=0.077, high=0.40, status="fixed",
                     source="nbc2025shortcreek",
                     note="Cottonwood Elementary (Colorado City) kindergarten MMR 7.7% "
                          "(2024-25, AZ DHS); nearby Masada Charter 40%"),
    },
}
