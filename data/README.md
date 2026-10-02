# Private Local Inputs and Outputs

Do not commit `inputs/` or `work/`; both are ignored. Only this README belongs in public data/.

Authorized researchers must privately supply:

```text
data/inputs/kaggle/
  transfers.csv
  players.csv
  clubs.csv
  games.csv
  competitions.csv
  appearances.csv
data/inputs/reference/
  clean_data.csv
  fbref_combined.csv
data/inputs/fbref_tables/
  standard_data.csv
  keepers_adv_data.csv
```

The reference clean file is the preserved 1,622-row original, including original contract fields, not the revised MIN270 sample. It is required for lineage equality and Phase 2 preservation checks. `fbref_combined.csv` is the original combined performance snapshot. The two named module exports are additionally required by exposure and goalkeeper checks. All are third-party-derived records and are excluded from Git.

The optional `original_fbref_fetch.R` imports eight modules (standard, shooting, passing, possession, misc, defense, keepers, keepers_adv) for end years 2018-2024 and produces a combined file in ignored work/acquisition/. Current downloads are not guaranteed to match the historical reference and must not silently replace it. Source reconstruction without the historical inputs cannot be certified from hashes alone.

Run `python code/prepare_public_workspace.py --check` to see missing input paths without fitting or creating outputs. `--initialize` checks completeness, creates ignored work directories, copies the authorized clean baseline for legacy design reconstruction, and initializes a fresh local hash manifest. It never fits a model or modifies public results/figures.

Work layout: audit/, datasets/intermediate/, datasets/final/, models/, validation/, positions/, residuals/, figures/, tables/, reports/, acquisition/. Generated manifests and changelogs remain in work/. See docs/REPRODUCIBILITY.md for the ordered commands. No analysis command has been run during this release task.
