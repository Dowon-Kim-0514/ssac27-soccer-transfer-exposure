# Reproducibility

This repository provides the analysis code, methodology, data dictionary, aggregate outputs, and documentation needed to reconstruct the research workflow.

The underlying player-level data are not redistributed because they originate from third-party sources for which the author does not hold independent redistribution rights.

Researchers with authorized access to the original sources can reconstruct the analysis using the documented data sources and the scripts in this repository.

## Data Sources Required

### Player performance data

Player performance data were obtained through:

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

Function documentation:

https://jaseziv.github.io/worldfootballR/reference/load_fb_big5_advanced_season_stats.html

Cached Big Five data are distributed through the worldfootballR data repository using the pattern:

`https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_{MODULE}.rds`

### Transfer data

Transfer and player metadata were obtained from:

**Football Data from Transfermarkt**

Dataset owner: David Cariboo

Kaggle dataset:

https://www.kaggle.com/datasets/davidcariboo/player-scores

Files used:

- `transfers.csv`
- `players.csv`
- `clubs.csv`

The exact historical Kaggle version used during Spring 2026 could not be independently recovered from the preserved project metadata.

## Reproduction Workflow

A researcher with access to the source data should:

1. Obtain the required FBref/worldfootballR performance data.
2. Obtain the Transfermarkt-derived Kaggle files listed above.
3. Run the data-processing scripts in `code/`.
4. Construct the analytical sample using the documented timing, identity, and eligibility rules.
5. Apply the MIN270 primary exposure rule.
6. Run the corrected modeling and validation scripts.
7. Compare the resulting aggregate outputs with the CSV files in `results/`.

## Primary Analysis Definition

The primary analytical sample contains 1,059 transfers involving 863 players.

Primary eligibility requires at least three recorded 90-minute equivalents of prior-season exposure.

The target is the natural log of the recorded positive transfer fee.

The primary structural specification includes centered quadratic age, position, destination league, and transfer season.

Contract variables, uncertain origin-league predictors, market value, redundant structural predictors, and audit identifiers are excluded from the primary model.

## Validation

The final analysis includes:

- random holdout validation
- five-fold development cross-validation with fold-local preprocessing and feature selection
- player-grouped cross-validation
- expanding-window temporal validation

## Expected Aggregate Results

For Model 4 under the MIN270 primary sample:

- Random holdout R²: approximately 0.414
- Development cross-validation mean R²: approximately 0.434
- Player-grouped cross-validation mean R²: approximately 0.428
- Pooled temporal R²: approximately 0.435

Detailed outputs are available in `results/`.

## Data Not Included

This repository intentionally does not include:

- raw FBref player-level data
- cached FBref player-level files
- raw Transfermarkt-derived player records
- merged player-level analytical datasets
- player-level predictions
- player-level residuals
- observation-level split membership files

These exclusions follow third-party redistribution restrictions and SSAC27 guidance for open-source submissions using restricted row-level data.
