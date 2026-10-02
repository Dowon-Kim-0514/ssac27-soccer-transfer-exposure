# Data Access and Rights

## Transfer Source
Football Data from Transfermarkt. Owner: David Cariboo (`davidcariboo`).

Dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores
Source repository: https://github.com/dcaribou/transfermarkt-datasets
Provider terms: https://www.transfermarkt.com/intern/anb

This is the source identified by the project author. The exact historical Spring 2026 Kaggle version could not be independently recovered. Current dataset versions must not be described as the exact historical input. The source's CC0 label is not a blanket clearance for every upstream record or for FBref-derived data.

The complete published workflow requires **six** Kaggle CSVs:

| File | Code use |
|---|---|
| transfers.csv | Original transfer construction in original_cleaning_from_cache.R |
| players.csv | Original join and build_phase2.py identity/contract audit |
| clubs.csv | Original destination/origin labels and Phase 2 corroboration |
| games.csv | Phase 2 observed season boundaries and club-season membership |
| competitions.csv | Phase 2 competition context |
| appearances.csv | Phase 2 dated player/club appearance evidence |

No additional Kaggle tables are read by the published scripts. Countries, national teams, game events, game lineups, player valuations and club_games files from the larger bundle are not required by this published workflow. This does not mean the six files alone are sufficient: historical FBref and reference inputs are also required.

PUBLIC_SOURCE_FILE_FINGERPRINTS.csv provides hashes, row counts (excluding header), and column counts for the preserved six originals. Local archived modification times do not establish Kaggle version dates, update dates or download dates. Matching filenames/schemas without matching bytes does not establish the exact snapshot.

## Performance Source
The original function was `worldfootballR::load_fb_big5_advanced_season_stats`, player-level, season end years 2018 through 2024. Modules: standard, shooting, passing, possession, misc, defense, keepers, keepers_adv. The final primary performance seasons are 2017-18 through 2022-23; transfers are 2018-19 through 2023-24.

Function: https://jaseziv.github.io/worldfootballR/reference/load_fb_big5_advanced_season_stats.html
Cache pattern: `https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_{MODULE}.rds`
Provider: https://fbref.com/
Terms: https://www.sports-reference.com/termsofuse.html
Data policy: https://www.sports-reference.com/data_use.html

Cached source acquisition is not equivalent to a verified historical publication timestamp. The archived combined file, standard module and advanced goalkeeper module remain required private inputs. See data/README.md. No data package license is inferred from the worldfootballR software license.

## Non-Redistribution and Organizer Guidance
No third-party player-level source files, derived records, predictions, residuals, crosswalks or split memberships are included. The author does not independently claim complete upstream redistribution rights. Source links and aggregate fingerprints are not substitutes for legal permission.

The previous public documentation reports that organizers accepted a code/documentation/aggregate-output alternative by email. That private correspondence was not independently inspected in this QA. Retain it privately and confirm its scope; this audit does not certify conference compliance or disclose the correspondence. Public reproducibility remains conditional on authorized access to matching historical inputs.
