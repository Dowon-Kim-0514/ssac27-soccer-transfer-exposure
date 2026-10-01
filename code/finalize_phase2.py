"""Document verified Phase 2 outputs, check preservation, and extend the manifest."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]/"data"/"work"
I, F, A = ROOT/'datasets/intermediate', ROOT/'datasets/final', ROOT/'audit'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def table(df):
    cols = list(df.columns)
    rows = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in df.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |')
    return '\n'.join(rows)

def write(path, title, body):
    path.write_text(f'# {title}\n\nPhase 2: DATA_AND_DESIGN_ONLY. No final fee models fitted.\n\n{body.strip()}\n', encoding='utf-8')

def main():
    summary = json.loads((I/'PHASE2_SUMMARY.json').read_text())
    variants = pd.read_csv(F/'DATASET_VARIANT_SUMMARY.csv')
    timing = pd.read_csv(I/'proxy_season_audit.csv')
    contracts = pd.read_csv(I/'contract_variable_audit.csv')
    identities = pd.read_csv(I/'fuzzy_match_identity_audit.csv')
    leagues = pd.read_csv(I/'club_league_timing_audit.csv')
    ranks = pd.read_csv(I/'revised_design_rank_summary.csv')
    vifs = pd.read_csv(I/'revised_full_design_vif.csv')
    dist = pd.read_csv(I/'dataset_variant_distributions.csv')
    ledger = pd.read_csv(I/'dataset_row_inclusion_ledger.csv')
    prior = pd.read_csv(I/'PHASE1_RESULTS_MANIFEST.csv')
    primary = variants.loc[variants.primary_recommendation].iloc[0]
    primary_name = primary.dataset_name
    modeled_timing = timing[timing.original_eligible]
    excluded = modeled_timing[~modeled_timing.strict_timing_keep]
    conservative = excluded[excluded.boundary_basis.eq('CONSERVATIVE_NEXT_OBSERVED_SEASON_START') & excluded.proxy_flag.eq(0)]
    compact = variants[['dataset_name','n','unique_player_names','unique_player_ids','rows_excluded','league_timing_concerns']]
    distribution_tables = []
    for kind in ['season','position']:
        pivot = dist[dist.distribution.eq(kind)].pivot(index='category', columns='dataset', values='n').fillna(0).astype(int)
        pivot.columns = [f'V{list(variants.dataset_name).index(x)+1}' for x in pivot.columns]
        distribution_tables.append(f'### {kind.title()} distribution\n\n' + table(pivot.reset_index()))
    distribution_text = '\n\n'.join(distribution_tables)

    write(A/'LEAKAGE_AUDIT.md', 'Performance Timing and Proxy Audit', f'''
## Decision and counts

All 1,622 cleaned rows are traced to transfer dates, selected FBref season/squad and archived fixtures. The original eligible 1,407 rows form Dataset A. Dataset B retains 1,195 and excludes 212: 190 proxies and 22 non-proxies. This is a date-based rule, not automatic deletion by proxy flag. No eligible proxy meets the boundary.

{table(compact.iloc[:2])}

Names are strings, not identities. Dataset A has 1,100 distinct names but 1,101 Transfermarkt player IDs. Use IDs for future grouped validation. The audit covers every performance record that reached the cleaned file, not unmatched raw transfers.

## Timing evidence and its limits

The archived games.csv supplies league/season dates. A complete directed home-away fixture ledger supplies an end date; the boundary is the next calendar day. For an incomplete ledger, the next observed season start is used as a conservative upper bound for completion. No historical dates are reconstructed from memory.

The 2019-20 Ligue 1 ledger is incomplete. Its last recorded match is not accepted as the formal cancellation date. **{len(conservative)} non-proxy French-league observations are excluded conservatively**, not proven to contain post-transfer events. The other {22-len(conservative)} non-proxy exclusions occur before the complete league-season boundary. A full-season aggregate can be risky even when labeled previous season, particularly during delayed 2019-20 schedules.

Selected-squad appearances corroborate possible post-transfer play where exact club aliases resolve. They are not used to rebuild truncated player statistics. A league-wide end rule may exclude a player whose own last appearance was earlier. No rows are silently reassigned to another season.

**Historical FBref publication timestamps are unavailable.** Strict pre-transfer means defensibly completed underlying performance period, not proof that this exact cached data vintage was publicly downloadable then. Later statistical revisions remain possible. Transfer dates are recorded effective dates, not verified negotiation or announcement dates.

{distribution_text}

V1-V5 follow the order in DATASET_VARIANT_SUMMARY.csv. Each exclusion is in dataset_row_inclusion_ledger.csv. Evidence: proxy_season_audit.csv and performance_season_bounds.csv under datasets/intermediate.
''')

    cmodel = contracts[contracts.original_eligible]
    write(A/'CONTRACT_VARIABLE_AUDIT.md', 'Contract Variable Validity Audit', f'''
## Source and verified construction

Both contract fields trace to archived players.csv:contract_expiration_date joined by player_id. The original formula is (snapshot expiry minus July 1 of Season_End_Year) / 365.25, **not** expiry minus actual transfer date. Every nonmissing original value was reproduced to numerical tolerance.

Original eligible sample: minimum {summary['contract_original_min']:.12f}, median {summary['contract_original_median']:.12f}, maximum {summary['contract_original_max']:.12f} years. Available: {cmodel.calculated_years_remaining.notna().sum()}; missing: {cmodel.calculated_years_remaining.isna().sum()}.

{table(cmodel.plausibility_flag.value_counts().rename_axis('Classification').reset_index(name='n'))}

No record is classified HISTORICALLY_PLAUSIBLE because none has a dated historical selling-club contract in the project inputs. QUESTIONABLE flags include a different snapshot current club, expiry before transfer, or more than six years from transfer to expiry. Six years is an audit heuristic, not a legal limit. These flags do not prove that every individual expiry is wrong. Otherwise, including missing expiry, the class is CANNOT_VERIFY.

## Future specifications

WITHOUT_CONTRACT is primary. Remove contract_expiration_date and contract_years_remaining from the primary dataset and predictors. Removing columns costs zero rows. WITH_CONTRACT is a deliberately timing-compromised diagnostic sensitivity, not a valid deployment estimate. Its comparison must use identical rows; the specifications JSON enables this using original_clean_row_id to recover preserved contract fields. Do not interpret performance improvement as evidence of historical validity.

The transfer-date-based recalculation in the audit is diagnostic only. It does not turn a current snapshot into historical information. No individual contracts were searched for or invented. Row evidence: contract_variable_audit.csv.
''')

    wrong = identities[identities.identity_confidence.eq('LIKELY_INCORRECT')]
    write(A/'FUZZY_MATCH_AUDIT.md', 'Fuzzy Player Identity Audit', f'''
## All accepted replacements

{table(identities.identity_confidence.value_counts().rename_axis('Confidence').reset_index(name='n'))}

All 212 original accepted replacements are accounted for. They affect 80 original eligible model rows. Only **one** eligible row is removed by strict identity screening after timing: Antonio Candela matched to Antonio Candreva. Counts among the 212 are not counts of removed model rows. Most CANNOT_VERIFY replacements do not reach the eligible modeling sample.

{table(wrong[['fuzzy_match_id','transfer_player_name','matched_fbref_name','player_date_of_birth','fbref_birth_year','original_model_rows_affected']])}

## Evidence rules

The candidate must have a unique record in the required performance season. Birth-year discrepancy of at least two years is LIKELY_INCORRECT. One-year discrepancy or incompatible broad position is AMBIGUOUS. Matching birth year and position plus same-Transfermarkt-ID appearance at an exactly resolved archived FBref squad in that season yields HIGH_CONFIDENCE. Matching birth/position with missing club corroboration yields LIKELY. Contradictory resolved club evidence is AMBIGUOUS. Insufficient season or identity evidence yields CANNOT_VERIFY.

These are reproducible rule-based evidence categories, not calibrated probabilities or 212 independently hand-verified identities. FBref provides birth year, not exact birth date. Nationalities are recorded but not scored because the coding systems lack a validated project crosswalk. Club aliases use only exact normalized archived names. Unresolved club aliases are not guessed.

Strict filtering excludes LIKELY_INCORRECT and CANNOT_VERIFY, but never automatically excludes AMBIGUOUS. No ambiguous row survives into Variant 4, so an identical ambiguity sensitivity file is not generated. Exact-name matches are labeled NOT_FUZZY_NOT_INDEPENDENTLY_AUDITED; correctness of those matches is not established by this task. Original matched values remain preserved, with suspect rows filtered rather than repaired by guessing.

Evidence: fuzzy_match_identity_audit.csv, fuzzy_matches_requiring_review.csv and fbref_club_alias_evidence.csv.
''')

    lmodel = leagues[leagues.original_eligible]
    status = lmodel.groupby(['from_classification','to_classification']).size().reset_index(name='n')
    write(A/'CLUB_LEAGUE_TIMING_AUDIT.md', 'Historical Club-League Audit', f'''
## Source finding

Original from_league and to_league derive from club snapshot competition labels, not a longitudinal promotion/relegation table. Archived games.csv domestic-league fixtures provide an independent club-season membership check. The available competition coverage is first tier, not a complete second-tier history.

{table(status)}

There are 293 concerns in the original 1,407 eligible rows. Variant 4 retains 248 concerns, including 79 unconfirmed Big 5 destinations. Variant 5 restricts to both source labels corroborated by transfer-season fixtures and an observed Big 5 destination: **{int(primary.n)} rows**.

This is an evidence/coverage restriction, not proof that all 248 excluded rows were mislabeled. Missing fixtures do not establish relegation. Changes between previous/current/next-season observed memberships flag possible promotion, relegation or incomplete coverage only. A last pre-transfer fixture can be stale; its date is retained and it is not used as a substitute historical classification.

## Scope of recommendation

Variant 5 is the conservative primary for a **fixture-corroborated two-sided first-tier sample**, not an unbiased census of all Big 5 destination transfers. This restriction can underrepresent promoted/relegated clubs and transactions involving lower tiers. Variant 4 is the broader coverage sensitivity. Neither restriction uses a fee outcome or model score.

Season-level corroboration is not exact transfer-day status and can use fixtures after the transfer to verify the season category retrospectively. A future real-time deployment claim requires dated membership/announcement records. No observed labels have overwritten original labels. No missing historical tables were reconstructed from memory.

Evidence: club_league_timing_audit.csv; both sides include snapshot, target/adjacent-season fixture memberships and last pre-transfer observation.
''')

    pv = vifs[vifs.dataset.eq(primary_name)].nlargest(10, 'VIF')
    write(A/'RANK_AND_MULTICOLLINEARITY_AUDIT.md', 'Full-Design Rank and Multicollinearity Audit', f'''
## Original dependencies reconfirmed

The saved original training matrix has 44 columns including intercept, rank 41. Numerical identities were rechecked: COVID equals the 2019-20 plus 2020-21 indicators; proxy equals the omitted reference-season indicator; fixed UEFA coefficient equals an intercept plus origin-league dummy lookup. Residual tolerances are below 1e-10, saved in original_design_dependencies_reconfirmed.json.

## Revised representation

Keep reference-coded transfer-season and origin-league fixed effects. Exclude covid_dummy, season_proxy_flag and UEFA_coeff_from as predictors. These fields may remain as audit metadata. Keep age, age squared, U21, O30, league_level_diff, position and destination-league controls. WITH_CONTRACT adds the questionable contract term only for sensitivity. Carry the eight original retained performance terms for design auditing, not a new feature-selection result.

{table(ranks[['dataset','columns_including_intercept','rank','max_full_design_vif']])}

All revised designs are full rank. **Full rank does not mean low multicollinearity.** The raw age/age-squared basis still has large VIFs:

{table(pv[['feature','VIF']])}

VIF is computed for every non-intercept numeric/dummy column against the full remaining design, using the inverse standardized correlation matrix. It is not candidate-only VIF or factor-level generalized VIF. Intercept VIF is not reported. Columns removed as deterministic have exclusion reasons in revised_feature_dictionary.csv, not a misleading finite VIF.

These X-only diagnostics use full-sample median/mode imputation solely to evaluate the design. Exported audit matrices must not be used for validation. Future imputation, scaling, encoding, feature selection and age centering must be trained inside each training fold. Centering age can improve conditioning but does not add information or resolve all age-segment correlation. Future folds must recheck rank and absent factor levels. No variables were dropped to improve R-squared; no final models or outcomes were fitted here.
''')

    write(F/'PRIMARY_DATASET_RECOMMENDATION.md', 'Primary Dataset Recommendation', f'''
## Recommended primary

**VARIANT_5_LEAGUE_CONFIRMED_WITHOUT_CONTRACT**, file `{primary.filename}`, n={int(primary.n)}, distinct player IDs={int(primary.unique_player_ids)}, distinct player-name strings={int(primary.unique_player_names)}.

Selection is based on timing, identity evidence and historical league corroboration, not predictive results. It estimates relationships within a restricted, fixture-corroborated sample. It is not a claim that all excluded observations are false or that unrestricted market patterns will agree.

## Dataset variants and costs

{table(compact)}

Starting from 1,407 eligible rows: timing excludes 212 (190 proxies, 22 non-proxies), leaving 1,195; contract removal costs zero rows; identity excludes one, leaving 1,194; two-sided league corroboration excludes 248, leaving {int(primary.n)}. Total exclusion is {1407-int(primary.n)} ({(1407-int(primary.n))/1407:.1%}). The original clean 1,622 to eligible 1,407 restriction is inherited, not a new Phase 2 correction.

## Sensitivities and future use

Variant 4 is the principal broader-coverage sensitivity. Variant 3 measures the identity-screening difference; Variant 2 preserves contracts for historical comparison; Variant 1 preserves all original eligible values. These are not directly comparable scores until evaluated on a common, valid split. WITH_CONTRACT versus WITHOUT_CONTRACT must be compared on the same IDs, with the former explicitly labeled historically unverified. Contract values can be joined from Variant 1 by original_clean_row_id for that diagnostic, not reinstated in the primary CSV.

Exclude contract fields, fixed UEFA lookup, COVID dummy and proxy dummy from the primary predictors. Do not turn all remaining CSV fields into predictors: use future_modeling_specifications.json and revised_feature_dictionary.csv. Market value, fees, IDs, URLs and audit flags are not authorized explanatory features. The target is log_transfer_fee.

Open uncertainties: historical data-release timestamps; exact transfer versus announcement dates; exact-name identity matches; nationality/club crosswalk gaps; historical contract provenance; historical day-specific league status; representativeness after league restriction; possible raw age polynomial ill-conditioning. Repeated player IDs require grouped and temporal out-of-sample validation in the next authorized phase. Original R-squared/CV figures are not results for this revised dataset.

{distribution_text}

## Reproduction and stop

Run tools/export_lineage.R with the revision root argument, then tools/build_phase2.py and tools/finalize_phase2.py. Use the existing recorded R/Python environments. The lineage stage reads the preserved snapshot; later stages only write Phase 2 outputs and the authorized root documentation. See intermediate/phase2_R_sessionInfo.txt and PHASE2_QA.json. No final Models 1-4, position analysis, residual-premium tests, or Sloan abstract were produced.
''')

    changelog = ROOT/'METHODOLOGY_CHANGELOG.md'
    heading = '\n## Phase 2: DATA_AND_DESIGN_ONLY\n'
    original_text = changelog.read_text().split(heading)[0]
    changelog.write_text(original_text + heading + f'''

This section supersedes the Phase 1 scope-stop statement for data/design work only. All original sources, reproduced models and Phase 1 evidence remain unchanged.

- Completed row-level timing, snapshot-contract, all-212 fuzzy identity and historical club-season evidence audits.
- Preserved 1,407-row original-comparable values; created 1,195-row strict-timing, no-contract, 1,194-row identity-screened, and {int(primary.n)}-row league-corroborated variants.
- Recommended the last variant as a conservative restricted-sample primary, Variant 4 as broader-coverage sensitivity. League evidence restriction is not proof of misclassification or a complete historical reconstruction.
- Removed contract columns from primary for unverified historical timing, not fit improvement. No rows lost solely to column removal.
- Removed three deterministic predictors from future structural designs; verified full rank and reported full-design VIF, including high polynomial-age VIF.
- Preserved every exclusion in a row ledger. No original performance values, fees, identity values or league labels silently repaired.
- Future fold-specific preprocessing, nested selection and temporal/player-ID validation remain unperformed. No final models, residual tests, position models or abstract. Original performance metrics cannot be attributed to corrected variants.

See datasets/final/PRIMARY_DATASET_RECOMMENDATION.md and audit/*_AUDIT.md Phase 2 reports. RESULTS_MANIFEST.csv labels new artifacts separately.
''', encoding='utf-8')

    # Verify preservation before extending the manifest. Only the authorized root
    # changelog is allowed to differ from the immutable Phase 1 manifest.
    preserved = 0
    for row in prior.itertuples():
        if row.relative_path in ['METHODOLOGY_CHANGELOG.md','RESULTS_MANIFEST.csv']:
            continue
        path = ROOT/row.relative_path
        assert path.is_file() and digest(path) == row.sha256, row.relative_path
        preserved += 1
    snapshot = ROOT.parent/'inputs/reference/clean_data.csv'
    clean = pd.read_csv(snapshot)
    eligible = clean[clean.log_transfer_fee.notna() & clean.from_league.notna()]
    a = pd.read_csv(F/'original_comparable_dataset.csv')
    pd.testing.assert_frame_equal(a[list(clean.columns)].reset_index(drop=True), eligible.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
    assert len(timing)==len(contracts)==len(leagues)==1622
    assert timing.original_clean_row_id.nunique()==1622
    assert len(identities)==212 and identities.fuzzy_match_id.nunique()==212
    assert ranks.rank_equals_columns.all()
    for row in variants.itertuples():
        d = pd.read_csv(F/row.filename)
        ll = ledger[ledger.dataset.eq(row.dataset_name)]
        assert len(d)==row.n and d.original_clean_row_id.is_unique
        assert set(d.original_clean_row_id)==set(ll.loc[ll.included,'original_clean_row_id'])
        assert len(ll)==1407 and not ll.loc[~ll.included,'reason'].eq('INCLUDED').any()
        common = [x for x in clean.columns if x in d]
        expected = clean.iloc[d.original_clean_row_id.to_numpy()-1][common].reset_index(drop=True)
        pd.testing.assert_frame_equal(d[common].reset_index(drop=True), expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        if not row.contract_predictor_present:
            assert not {'contract_years_remaining','contract_expiration_date'} & set(d.columns)
    p = pd.read_csv(F/primary.filename)
    assert p.strict_timing_keep.all() and not p.league_timing_concern.any()
    assert p.historical_big5_destination_status.eq('OBSERVED_BIG5').all()
    assert not p.identity_confidence.isin(['LIKELY_INCORRECT','CANNOT_VERIFY']).any()
    assert p.log_transfer_fee.notna().all()
    qa = dict(status='PASS', phase='DATA_AND_DESIGN_ONLY', final_models_fitted=False,
              phase1_artifacts_hash_preserved=preserved, original_comparable_values_preserved=True,
              all_variant_original_values_preserved=True, row_ledgers_verified=True,
              full_rank_all_variants=True, fuzzy_replacements_audited=212,
              primary_dataset=primary.filename, primary_n=int(primary.n),
              limitations='PASS verifies implementation and preservation, not historical truth or model validity.')
    (I/'PHASE2_QA.json').write_text(json.dumps(qa, indent=2))

    new_docs = [A/name for name in ['LEAKAGE_AUDIT.md','CONTRACT_VARIABLE_AUDIT.md','FUZZY_MATCH_AUDIT.md','CLUB_LEAGUE_TIMING_AUDIT.md','RANK_AND_MULTICOLLINEARITY_AUDIT.md']]
    paths = sorted(set([p for p in (ROOT/'datasets').rglob('*') if p.is_file() and not p.name.startswith('.') and '__pycache__' not in p.parts] + new_docs + [changelog]))
    rows = prior.to_dict('records')
    existing = {r['relative_path']:i for i,r in enumerate(rows)}
    serial = max(int(r['result_id'][3:]) for r in rows)
    for path in paths:
        rel = str(path.relative_to(ROOT))
        if rel in existing:
            idx = existing[rel]; rid = rows[idx]['result_id']
        else:
            serial += 1; rid = f'ART{serial:04d}'; idx = len(rows)
        entry = dict(result_id=rid,path=str(path),relative_path=rel,artifact_type=path.suffix.lstrip('.'),
                     methodology_version='PHASE2_DATA_AND_DESIGN_ONLY',source_reference='Preserved original snapshot and Phase 1 audit; no new fee models',
                     verification_status='PHASE2_QA_PASS',bytes=path.stat().st_size,sha256=digest(path),
                     modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat())
        if idx==len(rows): rows.append(entry)
        else: rows[idx]=entry
    pd.DataFrame(rows,columns=prior.columns).to_csv(ROOT/'RESULTS_MANIFEST.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(qa,indent=2))
    print(f'Manifest: {len(rows)} artifacts. Phase 2 complete. STOP before modeling.')

if __name__=='__main__':
    main()
