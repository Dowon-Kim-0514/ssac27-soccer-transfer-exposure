# Path-Only Adaptations

All original statistical constants and formulas are retained. Changed source is tracked in CODE_PROVENANCE.csv; new public I/O configuration has no statistical source counterpart. No model was executed.

| Files | Path-only change |
|---|---|
| All seven original public Python scripts | Root from parents[2] to repository/data/work; numbered phase folders renamed to audit, datasets, models, validation, positions, residuals, figures, tables, reports underneath ignored work/ |
| build_phase2.py | Explicit data/inputs/kaggle instead of recursive discovery in private snapshot; reference clean input from data/inputs/reference |
| finalize_phase2.py | Baseline clean input from data/inputs/reference; generated manifests/reports remain under work/ |
| per90_stability.py, analyze_positions.py | corrected_modeling import resolved beside the calling script in code/; standard and advanced-keeper tables from data/inputs/fbref_tables |
| analyze_residuals.py | analyze_positions import resolved beside the calling script in code/ |
| export_lineage.R | Repository argument; code/ cleaning helper; work/datasets/intermediate outputs; reference clean comparison input |
| original_cleaning_from_cache.R (added preserved dependency) | Replaces four personal absolute input paths with repository-root paths; suppressed exports redirected to ignored acquisition folder |
| original_models.py (added preserved dependency) | All historical clean/model/design evidence paths redirected to ignored work/audit; fitting logic unchanged and not run |
| original_fbref_fetch.R (added preserved acquisition dependency) | Repository argument and acquisition output paths only; original seasons, modules, joins and deduplication unchanged |
| prepare_public_workspace.py (new) | Filesystem checks, directory creation, private baseline copy and fresh hash-manifest configuration only; no fitting or row transformations |

The public root METHODOLOGY_CHANGELOG.md is preserved; executable scripts operate on a new work-local changelog. Publication results and figures are not destinations for scripts. Historical report text remains historically scoped; running an old finalizer does not certify the frozen submission anew.
