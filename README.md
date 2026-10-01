# When Per-90 Misleads: Exposure and Transfer-Fee Prediction in European Soccer

This repository contains the analysis code, methodology, aggregate results, figures, and reproducibility documentation for an SSAC27 Research Paper Competition submission in the Soccer track.

## Research Question

How does low playing-time exposure affect transfer-fee prediction reliability, and does performance add value beyond structural information after exposure screening?

## Overview

Transfer-fee models often use per-90 performance statistics to compare players with different amounts of playing time. However, when those rates are calculated from extremely limited exposure, small denominators can produce unstable values and large prediction errors.

This study evaluates whether performance information adds predictive value beyond structural transfer-market information after accounting for low-exposure instability.

The main contribution is an exposure-aware evaluation of transfer-fee prediction across random holdout, development cross-validation, player-grouped validation, and future-season temporal validation.

## Primary Analytical Sample

The primary analysis contains:

- 1,059 transfers
- 863 unique players
- Big Five destination transfers
- Transfer seasons from 2018-19 through 2023-24
- Prior-season performance exposure of at least three recorded 90-minute equivalents, nominally 270 minutes

The underlying performance records used by the final primary sample span the 2017-18 through 2022-23 seasons.

The 270-minute exposure rule was introduced after diagnosing instability in extremely low-minute per-90 observations. It was fixed before the corrected reruns, but it was motivated by prior out-of-sample error inspection and should not be interpreted as an independently validated or optimal threshold.

Sensitivity analyses at nominal 450-minute and 900-minute thresholds produced broadly similar conclusions.

## Models

The dependent variable is the natural log of the recorded positive transfer fee.

Four common-sample linear models are evaluated:

1. Structural controls only
2. Structural controls plus traditional performance measures
3. Structural controls plus advanced performance measures
4. Combined structural and selected performance measures

Primary structural controls include centered quadratic age, position, destination league, and transfer season.

Historically unsupported contract variables, uncertain origin-league predictors, market value, deterministic redundant predictors, and audit identifiers are excluded from the primary specification.

Preprocessing and feature selection are performed using training data only.

## Validation Design

The analysis separates several validation targets:

- Random 80/20 holdout
- Five-fold development cross-validation with fold-local preprocessing and selection
- Five-fold player-grouped validation using stable player IDs
- Expanding-window future-season temporal validation

Player-grouped validation prevents the same player from appearing in both training and validation folds.

## Main Results

For the exposure-screened MIN270 population, the combined Model 4 achieved:

| Validation | Model 4 R² |
|---|---:|
| Random holdout | 0.414 |
| Development cross-validation mean | 0.434 |
| Player-grouped cross-validation mean | 0.428 |
| Pooled future-season temporal validation | 0.435 |

On the same primary population, the structural-only Model 1 achieved a random holdout R² of 0.260 and a pooled temporal R² of 0.296.

The Model 4 random holdout metrics were:

- R²: 0.414
- RMSE: 0.880
- MAE: 0.679

## Exposure-Stability Finding

Before exposure screening, Model 4 showed severe temporal instability in 2022-23.

Two very low-exposure observations accounted for approximately 51.1% of that season's squared prediction error.

The 2022-23 temporal R² changed from approximately -0.315 before screening to 0.414 in the corrected MIN270 analysis.

However, restricting the original predictions to the 228 observations that remained eligible under MIN270, without refitting the model, already produced an R² of approximately 0.424.

Therefore, the apparent recovery should not be interpreted as a retraining effect alone. Much of the difference reflects removal of unstable low-exposure observations and a change in the evaluated population.

## Position Analysis

Position-specific analysis was treated as secondary and interpretive.

Out-of-sample permutation importance suggested different predictive signals across roles, including goals per 90 for forwards and playing-time or passing-related measures for several other positions.

However:

- position-specific xG improvements were weak or mixed
- all position-specific xG improvement intervals included zero
- no planned position interaction remained significant after false-discovery-rate adjustment
- separate position-specific models did not outperform the pooled global model

These findings do not establish causal position-specific valuation effects.

## Data Sources

### Player Performance Data

Player performance data were obtained through the `worldfootballR` package using:

`worldfootballR::load_fb_big5_advanced_season_stats()`

The original collection used season end years 2018 through 2024 and the following player-level modules:

- standard
- shooting
- passing
- possession
- misc
- defense
- keepers
- keepers_adv

worldfootballR function documentation:

https://jaseziv.github.io/worldfootballR/reference/load_fb_big5_advanced_season_stats.html

The function used cached Big Five season data distributed through the worldfootballR data repository. The cache follows the pattern:

`https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_{MODULE}.rds`

Example shooting cache:

https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_shooting.rds

Provider:

https://fbref.com/

Sports Reference Terms of Use:

https://www.sports-reference.com/termsofuse.html

Sports Reference Data Use Policy:

https://www.sports-reference.com/data_use.html

### Transfer Data

Transfer and player metadata were obtained from:

**Football Data from Transfermarkt**

Dataset owner: David Cariboo

Kaggle dataset:

https://www.kaggle.com/datasets/davidcariboo/player-scores

Files required by the complete published workflow:

- `transfers.csv`
- `players.csv`
- `clubs.csv`
- `games.csv`
- `competitions.csv`
- `appearances.csv`

The last three support timing, identity and historical league audits. See [source fingerprints](docs/PUBLIC_SOURCE_FILE_FINGERPRINTS.csv). Local timestamps are not verified Kaggle version dates. Additional private FBref/reference inputs are required, as listed in [data/README.md](data/README.md).

The Kaggle page currently lists the dataset under the CC0: Public Domain license.

The exact historical Kaggle dataset version used during the Spring 2026 data collection could not be independently recovered from the preserved project metadata. The source dataset and files used are documented here without presenting the current Kaggle version as the historical version used in the original analysis.

Original data provider:

https://www.transfermarkt.com/

Transfermarkt Terms of Use:

https://www.transfermarkt.com/intern/anb

## Data Availability and Redistribution

This study relies on third-party player-level performance and transfer data.

Player-level raw and derived records are not redistributed in this repository because the author does not hold independent redistribution rights for all underlying third-party records.

The author reports that the SSAC27 Research Paper Competition organizers confirmed by email that, where third-party terms restrict redistribution of row-level data, the open-source requirement may instead be satisfied by providing:

- full analysis and modeling code
- a data dictionary and variable definitions
- exact source links and provenance
- documentation of data acquisition and processing
- aggregate results and figures
- reproducibility documentation

The private correspondence was not independently reviewed during this code QA. The author should retain it and confirm its scope. This statement is not an independent certification of competition compliance.

Accordingly, it does not include raw FBref data, cached player-level source files, raw Transfermarkt-derived player records, merged player-level analytical datasets, player-level predictions, or player-level residual files.

## Reproduction Status

Path-only release adaptations are documented in [PATH_ADAPTATIONS.md](docs/PATH_ADAPTATIONS.md). Numerical logic and frozen results are unchanged. No model was rerun in this QA. A fresh checkout is not self-sufficient: authorized matching historical inputs are required, and the exact historical Kaggle version remains unrecovered. Read [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before executing code.

Use `python code/prepare_public_workspace.py --check` for a non-modeling input check. The ordered workflow and separate initialization command are documented there. Missing helper code and the original design-evidence script are now included. All generated player-level material stays in ignored data/work/, never public results/.

## Repository Structure

```text
code/       path-adapted scripts and I/O-only workspace initializer
docs/       provenance, fingerprints, rights, reproduction and QA
results/    unchanged aggregate reference results
figures/    unchanged final figures
data/       README only; inputs/ and work/ are ignored
```

requirements.txt includes all directly imported Python dependencies, including statsmodels, matplotlib and the historical audit-stage xgboost dependency. R_DEPENDENCIES.md documents recorded reproduction versions, not a recovered original authoring lockfile.
