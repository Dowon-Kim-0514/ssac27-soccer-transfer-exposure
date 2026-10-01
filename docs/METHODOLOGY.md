# Methodology

Target: natural-log transfer fee. Observation: retained positive-fee player-season transfer, with the highest-fee move retained where applicable. Strict prior-period matching does not certify historical publication timestamps of cached tables.

Primary structural terms: age minus 25, its square, sub_position, destination league, transfer season. No age-threshold dummies in the primary age design. Season controls replace COVID indicators. Contract, uncertain origin league, deterministic UEFA lookups, market value, fees, audit flags and identifiers are excluded as predictors.

Traditional family: Gls_per90, Ast_per90, Mins_Per_90_Playing.

Advanced family: xG_Per, Succ_Take_per90, PrgC_per90, Cmp_percent_Total, KP_per90, Won_percent_Aerial, Recov_per90.

Model 4 uses their union with training-only candidate VIF screening and backward removal requiring simultaneous AIC and BIC improvement. Structural terms are protected. Missing values, levels, standardization and alias handling are training-local. Unknown categories use training-frequency-weighted encoding, including unseen future seasons.

Random split: seed 42, 847 train and 212 test. Development cross-validation: five folds on the 847 training records, with preprocessing and selection refitted inside each fold, not a separate hyperparameter-search inner loop. GroupKFold: five folds on all 1,059 records, grouping stable player IDs. Temporal: expanding windows beginning with 2018-19/2019-20 training and testing 2020-21, followed by the next three seasons; pooled evaluation contains 716 observations. Players may recur across temporal seasons, unlike the grouped evaluation target.

MIN450 and MIN900 are fixed sensitivities, not independently optimized models. Counts and scores differ partly because eligibility changes. No clipping replaces the primary analysis. Errors are log units, not currency. No causal valuation, stable residual premium, or optimal cutoff is established.
