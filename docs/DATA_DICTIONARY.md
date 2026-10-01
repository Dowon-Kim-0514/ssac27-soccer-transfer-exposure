# Data Dictionary

This describes the minimum primary modeling schema, not a released dataset. Archival audit columns are intentionally omitted from a future modeling export.

| Field | Meaning / use |
|---|---|
| original_clean_row_id | Stable record key, validation alignment only |
| player_id | Stable player key, grouped splitting only |
| transfer_season | Transfer season label, categorical structural control |
| to_league | Historically corroborated destination; GB1, ES1, L1, IT1, FR1 |
| sub_position | Recorded position label; not independently validated tactical role history |
| age_at_transfer | Age in years, used to construct fixed-centered age |
| age_centered | age_at_transfer minus 25 |
| age_centered_squared | Squared centered age |
| log_transfer_fee | Natural-log recorded positive fee; target only |
| Mins_Per_90_Playing | Rounded selected-club prior-season 90-minute equivalents; exposure and traditional predictor |
| Gls_per90 | Goals divided by recorded 90s |
| Ast_per90 | Assists divided by recorded 90s |
| xG_Per | Provider-supplied expected goals per 90; not recomputed from rounded 90s |
| Succ_Take_per90 | Successful take-ons divided by recorded 90s |
| PrgC_per90 | Progressive carries divided by recorded 90s |
| Cmp_percent_Total | Pass completion percentage; not minute-normalized |
| KP_per90 | Key passes divided by recorded 90s |
| Won_percent_Aerial | Aerial-duel win percentage; not minute-normalized |
| Recov_per90 | Ball recoveries divided by recorded 90s |

Rates are not capped in the primary analysis. Training-only imputation handles missing inputs. Names, URLs, market value, raw fees, origin context and contract information must not be added to the primary predictor list merely because the source archive contains them.
