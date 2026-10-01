# Data Access and Redistribution

This study uses third-party soccer performance and transfer data.

## Player Performance Data

Player performance data were obtained through the `worldfootballR` package using:

`worldfootballR::load_fb_big5_advanced_season_stats()`

The original data collection used season end years 2018 through 2024 and the following player-level modules:

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

The function accessed cached Big Five season data distributed through the worldfootballR data repository using files of the form:

`https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_{MODULE}.rds`

Example:

https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_shooting.rds

Underlying provider:

https://fbref.com/

Sports Reference Terms of Use:

https://www.sports-reference.com/termsofuse.html

Sports Reference Data Use Policy:

https://www.sports-reference.com/data_use.html

## Transfer Data

Transfer and player metadata were obtained from the Kaggle dataset:

**Football Data from Transfermarkt**

Dataset owner:

David Cariboo

Dataset URL:

https://www.kaggle.com/datasets/davidcariboo/player-scores

Files used in the project:

- `transfers.csv`
- `players.csv`
- `clubs.csv`

The Kaggle page currently lists the dataset under the CC0: Public Domain license.

The exact historical Kaggle version used during the Spring 2026 data collection could not be independently recovered from the preserved project metadata. The current dataset page is therefore cited as the identified source dataset without claiming that the current version is identical to the historical version used in the original analysis.

Original data provider:

https://www.transfermarkt.com/

Transfermarkt Terms of Use:

https://www.transfermarkt.com/intern/anb

## Redistribution

Player-level raw and derived records are not redistributed in this repository.

This includes:

- raw FBref player-level files
- cached FBref player-level files
- raw Transfermarkt-derived records
- merged player-level analytical datasets
- player-level predictions
- player-level residuals
- observation-level validation membership files

The author does not independently claim redistribution rights over all third-party records used in the study.

## SSAC27 Open-Source Requirement

Before submission, the author contacted the SSAC27 Research Paper Competition organizers to clarify the open-source requirement for research using third-party data.

The organizers confirmed that, where third-party terms restrict redistribution of player-level records, the open-source requirement may be satisfied by providing:

- full analysis and modeling code
- a data dictionary and variable definitions
- exact source links and provenance
- documentation of data acquisition and processing
- aggregate results and figures
- reproducibility documentation

This repository follows that structure and intentionally excludes third-party player-level raw and derived records.
