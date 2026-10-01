# R Dependencies

Required for the published R acquisition/cleaning/lineage scripts:

| Package | Recorded audit runtime version | Use |
|---|---|---|
| worldfootballR | 0.6.2 | Cached acquisition; also loaded by original cleaning |
| readr | 2.2.0 | CSV I/O |
| dplyr | 1.2.1 | Joins, filters, feature construction |
| tidyr | 1.3.2 | Loaded original workflow dependency |
| stringdist | 0.9.17 | Original Jaro-Winkler identity matching |
| ggplot2 | 4.0.3 | Loaded by preserved R scripts |
| ggrepel | 0.9.8 | Loaded by preserved R scripts |
| patchwork | 1.3.2 | Loaded by preserved R scripts |

R 4.5.2, Apple Silicon macOS, from the preserved Phase 2 sessionInfo record. These are verified reproduction-runtime versions, NOT a recovered Spring 2026 authoring lockfile. Transitive dependencies must also be installed by R's package manager. The current cache and package availability may differ; no automatic current-version substitution is guaranteed equivalent.

`export_lineage.R` requires the repository root argument and sources `code/original_cleaning_from_cache.R`. Do not run the sourced helper directly. `original_fbref_fetch.R` also takes the root argument and writes only to ignored `data/work/acquisition/`. It fetches the current cache, not a guaranteed historical snapshot. No R script was evaluated during release QA; parse-only checks are recorded separately.
