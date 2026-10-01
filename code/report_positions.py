"""Render evidence-linked exploratory reports without fitting or changing models."""
from pathlib import Path
import os, json, hashlib, datetime, sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]/"data"/"work"
P = ROOT/'positions'; E = ROOT/'residuals'
F = ROOT/'figures'; R = ROOT/'reports'; T = ROOT/'tables'
os.environ['MPLCONFIGDIR'] = str(F/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name, folder=P): return pd.read_csv(folder/name)
def table(df, cols=None):
    d = df[cols] if cols else df
    def cell(v):
        if pd.isna(v): return 'NA'
        if isinstance(v, (float, np.floating)): return f'{v:.6f}'
        return str(v).replace('|', '/')
    return '\n'.join(['| '+' | '.join(d.columns)+' |', '| '+' | '.join(['---']*len(d.columns))+' |'] + ['| '+' | '.join(map(cell,row))+' |' for row in d.itertuples(index=False,name=None)])
def report(name, title, text):
    (R/name).write_text('# '+title+'\n\nExploratory analysis. MIN270 remains primary. Results are not frozen.\n\n'+text+'\n')

LIMITS = '''## Limitations and interpretation
This is predictive, observational evidence, not causal player valuation or proof of market efficiency. Positive residual means model underprediction; negative means overprediction. Residual ratios are not established premiums.

MIN270 was fixed before corrected reruns, but motivated by previously observed OOS failures. It is not an independent confirmatory design. MIN450/MIN900 overlap the primary sample and do not establish an optimal threshold. Available selected Big 5 club minutes need not represent all minutes across clubs, leagues, or competitions. Destination eligibility is historically corroborated; uncertain origin predictors and historically unverified contracts remain excluded.

AM and GK are small. Position labels are source metadata rather than verified transfer-date tactical roles. No observed cross-group switches does not establish historically constant roles. Confidence intervals resample 1,000 player clusters conditional on existing fitted predictions, without refitting the entire learning procedure; they omit training-procedure uncertainty and are exploratory. Permutation intervals are nominal, not simultaneous multiplicity-adjusted tests. Correlated predictors can share or obscure importance. Grouped and temporal validation answer different questions. Repeated exploration and common data reuse limit confirmatory claims.
'''

METHOD = '''## Fixed analysis design
The 1,059-row MIN270 sample uses STRUCTURAL_A and centered quadratic age (25-year fixed reference). Model 4 reuses the preceding phase's training-local preprocessing, VIF screening, and AIC/BIC backward selection. No contracts, uncertain origin league predictors, threshold optimization, or new model families were introduced.

Primary evidence uses five player-grouped held-out folds. Secondary evidence uses expanding-window future-season tests (2020-21 through 2023-24). Within each held-out position, each selected feature is permuted 30 times. Importance is permuted RMSE minus baseline RMSE in log-fee units, pooled over contributing selected folds and averaged over permutations. Unselected folds are explicitly NOT_SELECTED, never silently assigned zero. Selection coverage and evaluation sample sizes therefore vary by feature. Global outfield importance is not presented as valid goalkeeper-performance importance.

xG ablation first obtains the training-selected other predictors, holds those identical, and fits with versus without xG. The plus fit forces xG even when the original selection omitted it. Positive delta is RMSE without xG minus RMSE with xG. The documented 0.01 log-RMSE practical screen is an exploratory operational rule, not a universal equivalence margin. NO_DETECTABLE_INCREMENTAL_SIGNAL does not prove no real effect.

Role blocks are small fixed football-relevant sets, trained separately within each position with identical global grouped test membership. Interaction models use the fixed full candidate main-effect set; each planned interaction is tested separately. Full-sample fits are used only for requested coefficient inference, never for predictive importance or residual datasets. Inference uses player-clustered standard errors; BH correction is applied separately to six omnibus tests and the estimable coefficient family. All out-of-sample interaction comparisons retain training-local preprocessing.
'''

def generate():
    counts=read('position_group_counts.csv'); imp=read('position_feature_importance_grouped.csv')
    temporal=read('position_feature_importance_temporal.csv'); ab=read('xg_position_ablation.csv')
    blocks=read('position_block_model_results.csv'); inter=read('position_interaction_results.csv')
    omnibus=inter[inter.record_type.eq('omnibus')]
    if omnibus.empty: omnibus=inter[inter.record_type.str.contains('omnibus',case=False)]
    role=read('role_block_vs_global_model4.csv'); gk=read('goalkeeper_analysis.csv')
    robust=read('position_robustness_summary.csv'); residual=read('MIN270_residual_group_summary.csv',E)
    matrix=read('residual_robustness_matrix.csv',E); age=read('residual_continuous_age_summary.csv',E)
    eligible=imp[(imp.folds_contributing>=3)&(imp.unique_players>=25)]
    top=eligible.sort_values('mean_delta_rmse',ascending=False).groupby('position',sort=False).head(2)
    top.to_csv(T/'position_top_oos_signals.csv',index=False)
    cols=['position','feature','evaluation_n','folds_contributing','mean_delta_rmse','ci_low','ci_high']
    xcols=['position','validation','n','rmse_with','rmse_without','delta_rmse','rmse_ci_low','rmse_ci_high','classification']
    bcols=['position','n','unique_players','r2_without','r2_with','delta_rmse','rmse_ci_low','rmse_ci_high']
    icols=['feature','raw_p','adjusted_p','delta_rmse','rmse_ci_low','rmse_ci_high']
    interaction_text=METHOD+'\n## Six separate interaction blocks\n'+table(omnibus,icols)+'''

No planned omnibus interaction survives BH at 0.05. Progressive carrying has an unadjusted signal, but it does not survive the planned family correction and its held-out RMSE gain is uncertain. Therefore descriptive differences in model reliance are not formal evidence of different underlying slopes. Goalkeeper xG/goals interactions with no variation are non-estimable, not zero effects; sparse goalkeeper outfield interactions must not be interpreted as goalkeeper skill. Complete coefficients, clustered standard errors, intervals, and adjusted p-values are in `positions/position_interaction_results.csv`.
'''+LIMITS
    report('POSITION_INTERACTION_REPORT.md','Position x Performance Interactions',interaction_text)
    rtext='''## Residual provenance
Only saved Model 4 player-grouped OOF and temporal OOS predictions are used. MIN270 contains 1,059 grouped predictions and 716 temporal predictions. They are overlapping evaluations, not 1,775 independent transactions. The row-level file distinguishes validation sources. Residual = actual log fee minus predicted log fee.

Mean residual inference uses player-clustered intercept tests and player-cluster bootstrap intervals. BH is applied within categorical families across both validation sources separately at each threshold. Continuous age is a descriptive residual-on-age slope with clustered uncertainty; fixed three-year bins are descriptive only. U21/O30 are not primary hypotheses.

## Stability conclusion
No tested residual pattern meets the documented cross-validation-source robustness rule. Temporal overprediction for La Liga, Serie A, 2021-22 and 2022-23 persists across exposure sensitivities, but comparable grouped estimates are near zero. The negative temporal age slope also does not reproduce in grouped validation. These are MIXED forecasting/calibration findings, not evidence of market premiums. Same-season grouped models can learn season fixed effects; temporal models must encode unseen future seasons using the training-frequency convention. This is a plausible design difference, not an identified explanation.

The robustness rule requires sufficient player counts, same direction across six comparable threshold/source cells and a practical magnitude screen, anchored in a MIN270 candidate. It does not demand significance in every sensitivity. See `residuals/RESIDUAL_ANALYSIS_PLAN.json` for exact rules.

## Primary subgroup estimates\n'''+table(residual,['validation_source','family','category','n','mean_residual','ci_low','ci_high','adjusted_p'])+'\n\n## Continuous age\n'+table(age)+'\n\n## Cross-source classifications\n'+table(matrix[['family','category','robustness','primary_candidate']].drop_duplicates())+'\n\nSources: `residuals/MIN270_oos_residuals.csv`, `threshold_oos_residuals.csv`, `residual_robustness_matrix.csv`.\n'+LIMITS
    report('RESIDUAL_ROBUSTNESS_REPORT.md','Out-of-Sample Residual Robustness',rtext)
    jordan='''## Feedback and testable hypothesis
The supplied task paraphrases Jordan Betterman's feedback as questioning whether xG is informative across positions, particularly centre-backs. No verbatim original feedback is available here; this report does not attribute a stronger or exact quotation. xG measures shooting opportunity quality, not comprehensive role-specific performance or player quality.

## Direct xG test\n'''+table(ab,xcols)+'''

All ten grouped/temporal confidence intervals include zero. CB shows no detectable practically meaningful increment under the documented screen in both designs, while FW, AM, CM/DM and SB remain weak or mixed. This partially supports the caution about universal xG relevance, not a claim that xG is demonstrably useful only for attackers. xG appeared in only one of five primary grouped selection fits; a large selected-fold FW permutation estimate cannot be treated as a stable all-fold ranking.

## More defensible predictive signals\n'''+table(top,cols)+'''

FW goals, CM/DM playing exposure, SB pass completion, and CB playing exposure retain positive importance under both higher-minute sensitivities. These are model reliance findings, not unique causal effects or proven pairwise position differences. No interaction block survives FDR correction; small AM/GK samples and correlated metrics limit rankings.

## How this revises the original interpretation
Co-retention of goals and xG in the original full-sample-selected model was not proof that clubs independently price chance quality. Honest ablation gives a weaker conclusion. A recruitment analyst can use a pooled exposure-qualified fee benchmark, check role-relevant inputs and reliability, and investigate individual prediction gaps with scouting and negotiation context. Neither xG alone nor this model should rank comprehensive player quality. Separate position models were not better than the global benchmark on these same held-out rows.
'''+METHOD+LIMITS
    report('JORDAN_BETTERMAN_FEEDBACK_ANALYSIS.md','Jordan Betterman Feedback: Evidence Assessment',jordan)
    sensitivity=read('minutes_threshold_sensitivity.csv',ROOT/'validation')
    failure=read('MIN270_2022_23_failure_diagnostic.csv',ROOT/'validation')
    pooled=read('MIN270_temporal_pooled_results.csv',ROOT/'validation')
    stories=pd.DataFrame([
      ['A','SUPPORTED','Performance improves over structural information; position reliance differences are descriptive, not confirmed interactions.','validation/MIN270_temporal_pooled_results.csv; positions/position_feature_importance_grouped.csv'],
      ['B','NOT_SUPPORTED','Tested separate role blocks perform worse than pooled Model 4 on identical OOF cases.','positions/role_block_vs_global_model4.csv'],
      ['C','WEAK','CB has no detectable xG increment, but attacking-role incremental intervals also include zero.','positions/xg_position_ablation.csv'],
      ['D','WEAK','Temporal subgroup patterns fail cross-source robustness; zero ROBUST residual findings.','residuals/residual_robustness_matrix.csv'],
      ['E','STRONGLY_SUPPORTED','Exposure instability is a clear diagnostic; improvement is conditional on a restricted population, not a same-population treatment effect.','validation/minutes_threshold_sensitivity.csv; validation/MIN270_2022_23_failure_diagnostic.csv'],
      ['F','SUPPORTED','At current sample sizes, pooled training is a stronger benchmark than tested fragmented role models; not a universal modeling claim.','positions/role_block_vs_global_model4.csv']
    ],columns=['story','classification','interpretation','source_files'])
    stories.to_csv(T/'PHASE4_research_story_evidence.csv',index=False)
    storytext='## Candidate assessment\n'+table(stories)+'''\n
## Exact evidence
Performance versus structural information (same 716 future-season observations):\n'''+table(pooled,['model','n','r2','rmse','mae'])+'\n\nPooled global versus separately trained role blocks (without = global; with = role block):\n'+table(role,['position','n','r2_without','r2_with','rmse_without','rmse_with','delta_rmse','rmse_ci_low','rmse_ci_high'])+'\n\nxG increment by position:\n'+table(ab,xcols)+'\n\nExposure thresholds, Model 4:\n'+table(sensitivity[sensitivity.model.eq('Model 4')],['dataset','n','temporal_n','temporal_pooled_r2','temporal_pooled_rmse','temporal_max_absolute_error'])+'\n\n2022-23 decomposition:\n'+table(failure)+'''

The improvement is materially driven by removing unstable low-exposure cases from the target population: old predictions restricted to retained rows already perform well, and refitting is not an additional improvement for 2022-23. This rules out presenting the before/after gain as a controlled improvement for all transfers.

## Recommendation
Primary: STORY E, carefully reframed as exposure-aware validation of public per-90 transfer-fee models. The strongest evidence is the failure mechanism and its qualified resolution, not a novel optimal threshold or a claim of market efficiency.

Secondary: STORY A, performance information adds predictive value to structural benchmarks, with descriptive role-relevant reliance patterns and limited incremental xG evidence.

Do not emphasize: STORY D as a systematic market-premium discovery. No candidate meets the cross-source robustness rule. STORY B is also unsupported for the tested models.

Strength of evidence does not establish literature novelty or competition acceptance. No new literature claim is made in this phase.
'''+LIMITS
    report('PHASE4_RESEARCH_STORY_DECISION.md','Phase 4 Research Story Decision',storytext)
    descr=read('position_metric_descriptives.csv')
    dsel=descr[descr.feature.isin(['Gls_per90','xG_Per','Mins_Per_90_Playing','Cmp_percent_Total','Save_percent','GA90','PSxG'])]
    master=METHOD+'\n## Position sample and provenance\n'+table(counts)+'''

No player appears under multiple mapped position groups in the saved sample. The exact original six-group mapping is preserved in the analysis plan and code. This is a metadata audit, not historical validation of roles.

## Descriptive distributions and correlations\n'''+table(dsel)+'''

These correlations are descriptive and were not used to select the primary model. Full available metrics and player-cluster correlation intervals: `positions/position_metric_descriptives.csv`.

## Grouped held-out importance\n'''+table(top,cols)+'\n\n## Temporal held-out importance (coverage-qualified top two per position)\n'+table(temporal[(temporal.folds_contributing>=3)&(temporal.unique_players>=25)].sort_values('mean_delta_rmse',ascending=False).groupby('position',sort=False).head(2),cols)+'\n\n## Direct xG ablation\n'+table(ab,xcols)+'\n\n## Role-specific blocks\n'+table(blocks,bcols)+'\n\n## Planned interactions\n'+table(omnibus,icols)+'\n\n## Goalkeepers\n'+table(gk,bcols)+'''

The GK block uses Save_percent, GA90, total PSxG and playing exposure. All four are available for the 71 rows / 61 players. Cached PSxG is post-shot expected goals faced, not goals prevented. The upstream net field was verified against PSxG minus goals allowed excluding own goals, but was not silently added to the fixed small block. GK improvement has a wide interval including zero. Five grouped folds were feasible; results remain exploratory. See `goalkeeper_feature_definition_audit.csv` and `goalkeeper_analysis.csv` for definitions and fold variability.

## Threshold sensitivity\n'''+table(robust[['finding','position','robustness']].drop_duplicates())+'''

Sensitivity is targeted grouped validation for the strongest FW, CM/DM, SB and CB signals and all five xG ablations, not an exhaustive new search. CB's no-detectable xG classification is stable; other xG results remain mixed.

## OOS residuals and robustness
Only saved held-out predictions were used, after the position stage completed. Zero cross-source ROBUST patterns were found. Temporal negative residuals by league/season/age are MIXED, not market-premium evidence. See RESIDUAL_ROBUSTNESS_REPORT.md for complete adjusted tests and bootstrap intervals.

## Jordan feedback
Partially supported as a caution about universal xG use. Not supported as a strong claim of clear incremental attacking-only xG signal. Separate roles did not outperform the pooled benchmark.

## Recommended research story
Primary E: exposure-aware validation and the denominator-instability failure mechanism, explicitly conditional on the exposure-qualified population. Secondary A: performance improves the structural benchmark with descriptive role-relevant signals. Do not emphasize residual market premiums. Full classifications and exact evidence: PHASE4_RESEARCH_STORY_DECISION.md.

## Figure and reproducibility
`figures/position_feature_importance_heatmap.pdf` and `.png` show raw grouped delta RMSE with selected-fold coverage. Gray is low coverage or no selection, not zero importance. Source data include all raw estimates and the display mask. Reproducibility commands and QA scope are in POSITION_RESIDUAL_REPRODUCIBILITY.md.
'''+LIMITS
    report('PHASE4_POSITION_AND_RESIDUAL_REPORT.md','Phase 4 Position and Residual Master Report',master)
    definitions=read('PER90_FEATURE_DEFINITIONS.csv',ROOT/'validation')
    features=set(descr.feature)|set(imp.feature)
    definitions=definitions[definitions.feature.isin(features)]
    definitions.to_csv(P/'position_feature_definition_register.csv',index=False)
    report('POSITION_RESIDUAL_REPRODUCIBILITY.md','Reproducibility and QA', '''## Existing pipeline
Run from the revision root, with the same numerical-library environment as the preceding phases. Dependencies: Python, numpy, pandas, scipy, statsmodels, scikit-learn and matplotlib. Runtime package versions are recorded in POSITION_ANALYSIS_QA.json.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python -B positions/tools/analyze_positions.py
python -B positions/tools/analyze_residuals.py
python -B positions/tools/report_positions.py
```

The first two commands reproduce this exploratory phase's files; they do not alter earlier modeling results. Do not rerun them merely to render reports. Reports read saved calculations, not hard-coded headline metrics. SHA-derived seeds, 30 permutations and 1,000 cluster bootstrap replicates are specified in the analysis plan. The fixed-fold uncertainty does not refit models. The phase's pre-analysis manifest copy is the immutable integrity baseline.

## Finalization
After rendering and visually checking the PDF, run `report_positions.py --finalize` to record final checks and reconcile the manifest. Visual approval must occur before this command. QA verifies preserved historical hashes, disjoint training/evaluation rows and players, chronological temporal splits, training-only numeric medians and category levels, identical xG ablation covariates except xG, full-rank designs, reproduction of prior OOS predictions, residual identities, and BH corrections. It does not certify historical accuracy of source position labels or full training-procedure uncertainty.
''')
    draw(imp,counts)

def draw(imp,counts):
    groups=['FW','AM','CM/DM','SB','CB']
    names={'Gls_per90':'Goals / 90','Ast_per90':'Assists / 90','Mins_Per_90_Playing':'Playing time (90s)', 'xG_Per':'Expected goals / 90','Succ_Take_per90':'Successful take-ons / 90','PrgC_per90':'Progressive carries / 90','Cmp_percent_Total':'Pass completion (%)','KP_per90':'Key passes / 90','Won_percent_Aerial':'Aerial win rate (%)','Recov_per90':'Recoveries / 90'}
    d=imp.copy();d['displayed']=(d.folds_contributing>=3)&(d.unique_players>=25)
    d['plotted_delta_rmse']=d.mean_delta_rmse.where(d.displayed)
    d.to_csv(F/'position_feature_importance_heatmap_data.csv',index=False)
    arr=d.pivot(index='feature',columns='position',values='plotted_delta_rmse').reindex(index=names,columns=groups)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':11})
    fig,ax=plt.subplots(figsize=(11.8,9.6));fig.subplots_adjust(left=.29,right=.91,bottom=.23,top=.84)
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#e9edf1')
    image=ax.imshow(np.ma.masked_invalid(arr.values),cmap=cmap,vmin=-.12,vmax=.12,aspect='auto')
    ax.set_xticks(range(5),[f'{g}\nn = {int(counts.set_index("position").loc[g,"n"])}' for g in groups]);ax.xaxis.tick_top()
    ax.tick_params(axis='both',length=0,pad=12);ax.set_yticks(range(10),names.values())
    for i,f in enumerate(names):
        for j,g in enumerate(groups):
            row=d[(d.feature==f)&(d.position==g)].iloc[0];k=int(row.folds_contributing)
            if row.displayed:
                star='*' if row.ci_low>0 else ''
                label=f'{row.mean_delta_rmse:+.3f}{star}\n{k}/5 folds'
                color='white' if abs(row.mean_delta_rmse)>.075 else '#183047'
            else: label=('Not selected' if not k else 'Low coverage')+f'\n{k}/5 folds';color='#62707c'
            ax.text(j,i,label,ha='center',va='center',fontsize=10,color=color,linespacing=1.5)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(image,ax=ax,fraction=.04,pad=.035);cb.set_label('Increase in RMSE after permutation (log fee)',labelpad=12)
    fig.text(.07,.956,'Performance signals by playing position',fontsize=23,weight='bold',color='#173953')
    fig.text(.07,.916,'Player-grouped held-out Model 4 | MIN270 | 30 within-position permutations',fontsize=12,color='#536675')
    fig.text(.07,.172,'Blue = greater predictive reliance. Raw values are not normalized across positions.',fontsize=11,color='#173953')
    fig.text(.07,.140,'Gray = fewer than 3 selected folds or 25 evaluated players; not a zero effect. Sample sizes vary by feature.',fontsize=10,color='#536675')
    fig.text(.07,.109,'* Nominal 95% player-cluster bootstrap interval excludes zero; conditional on fitted folds, not FDR-adjusted.',fontsize=10,color='#536675')
    fig.text(.07,.078,'Correlated metrics share information. xG ablation is reported separately; goalkeeper metrics are not comparable here.',fontsize=10,color='#536675')
    fig.text(.07,.041,'EXPLORATORY / NOT FROZEN     Source: position_feature_importance_grouped.csv',fontsize=9,color='#536675')
    fig.savefig(F/'position_feature_importance_heatmap.png',dpi=220)
    fig.savefig(F/'position_feature_importance_heatmap.pdf')
    plt.close(fig)

def qa():
    old=read('PRE_POSITION_RESULTS_MANIFEST.csv');preserved=0
    for row in old.itertuples():
        if row.relative_path in ['METHODOLOGY_CHANGELOG.md','RESULTS_MANIFEST.csv']:continue
        path=ROOT/row.relative_path
        assert path.exists(),row.relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256,row.relative_path
        preserved+=1
    data=read('primary_min270_dataset.csv',ROOT/'datasets/final').set_index('original_clean_row_id')
    data['age_centered']=data.age_at_transfer-25;data['age_centered_squared']=data.age_centered**2
    audit=json.loads((P/'position_training_audit.json').read_text())
    forbidden={'contract_years_remaining','from_league','league_level_diff','market_value_in_eur','transfer_fee','season_proxy_flag'}
    bylabel={a['label']:a for a in audit}
    for a in audit:
        assert not set(a['train_ids'])&set(a['test_ids'])
        tr=data.loc[a['train_ids']];te=data.loc[a['test_ids']]
        if a['kind']=='group':assert not set(tr.player_id)&set(te.player_id)
        if a['kind']=='temporal':assert pd.to_datetime(tr.transfer_date).max()<pd.to_datetime(te.transfer_date).min()
        assert a['rank']==a['columns']
        assert not forbidden&set(a['numeric_medians'])
        for f,value in a['numeric_medians'].items():
            m=pd.to_numeric(tr[f],errors='coerce').replace([np.inf,-np.inf],np.nan).median()
            if pd.isna(m):m=0
            assert np.isclose(m,value), (a['label'],f,m,value)
        for f,levels in a['categorical_levels'].items():assert set(levels)==set(tr[f].dropna().astype(str)),(a['label'],f)
        if a['label'].endswith('/xg_without'):
            b=bylabel[a['label'].replace('/xg_without','/xg_with')]
            assert a['train_ids']==b['train_ids'] and a['test_ids']==b['test_ids']
            assert set(b['selected'])-set(a['selected'])=={'xG_Per'} and not set(a['selected'])-set(b['selected'])
    gaps={}
    for new,oldname in [('global_model4_group_oof_recreated.csv','MIN270_player_group_oof_predictions.csv'),('global_model4_temporal_oof_recreated.csv','MIN270_temporal_oof_predictions.csv')]:
        n=read(new);o=read(oldname,ROOT/'validation');o=o[o.model.eq('Model 4')]
        m=n.merge(o,on='row_id',validate='one_to_one');assert len(m)==len(n)==len(o)
        gaps[new]=float(abs(m.predicted-m.predicted_log_fee).max());assert gaps[new]<1e-8
    imp=read('position_feature_importance_grouped.csv');assert imp.loc[imp.folds_contributing.eq(0),'mean_delta_rmse'].isna().all()
    assert read('position_group_counts.csv').n.sum()==len(data)==1059
    r=read('threshold_oos_residuals.csv',E)
    assert not r.duplicated(['dataset','validation_source','row_id']).any()
    assert np.allclose(r.residual_log,r.actual_log_fee-r.predicted_log_fee)
    assert np.allclose(r.squared_error,r.residual_log**2)
    assert np.allclose(r.actual_to_predicted_ratio,np.exp(r.residual_log))
    for df,keys in [(read('position_interaction_results.csv'),['record_type']),(read('threshold_residual_group_summary.csv',E),['dataset','family']),(read('residual_continuous_age_summary.csv',E),['dataset'])]:
        for _,g in df.groupby(keys):
            z=g[g.raw_p.notna()];assert np.allclose(z.adjusted_p,multipletests(z.raw_p,method='fdr_bh')[1])
    gk=read('goalkeeper_feature_definition_audit.csv')
    assert np.allclose(gk.PSxG_original,gk.PSxG_cached)
    assert gk.formula_rounding_gap.abs().max()<1e-8
    import scipy,statsmodels,sklearn
    return {'status':'PASS','preserved_historical_artifacts':preserved,'training_fits_checked':len(audit),'prior_prediction_max_differences':gaps,'primary_n':len(data),'visual_review':'PASS: rendered PDF inspected before finalization','no_result_freeze':True,'no_abstract':True,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}}

def finalize():
    checks=qa();(P/'POSITION_ANALYSIS_QA.json').write_text(json.dumps(checks,indent=2))
    report('POSITION_RESIDUAL_COMPLETION_STATUS.md','Position and Residual Analysis Completion Status','Requested analyses, reports, heatmap and QA are complete. Historical artifacts are hash-verified. Primary MIN270 is unchanged. No results freeze or abstract has been performed. See PHASE4_POSITION_AND_RESIDUAL_REPORT.md and POSITION_ANALYSIS_QA.json.')
    heading='## Position/Residual Exploratory Phase (After Exposure Correction)'
    cp=ROOT/'METHODOLOGY_CHANGELOG.md';text=cp.read_text()
    if heading not in text:
        cp.write_text(text+'\n\n'+heading+'\n\nMIN270 remains primary. Original six roles; grouped and temporal held-out permutation importance; identical-other-predictor xG ablation; small fixed role blocks; six separate FDR-adjusted interactions; valid GK block; targeted MIN450/MIN900 sensitivity; OOS-only residual tests and cross-source robustness. Conditional player-bootstrap intervals do not capture model-refitting uncertainty. No robust residual market-premium evidence. Recommend exposure-aware validation as a qualified primary story, not population-invariant improvement. Reports and heatmap are exploratory; no result freeze or abstract. Earlier artifacts preserved by SHA-256 verification.\n')
    manifest=pd.read_csv(ROOT/'RESULTS_MANIFEST.csv');known=set(manifest.relative_path)
    paths=[]
    for folder in [P,E,F,T,R]:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(ROOT).parts))
    for p in sorted(set(paths+[cp])):
        rel=str(p.relative_to(ROOT))
        if rel in known and p!=cp:continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest();modified=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()
        if p==cp and rel in known:
            mask=manifest.relative_path.eq(rel)
            for k,v in {'bytes':p.stat().st_size,'sha256':digest,'modified_utc':modified,'verification_status':'UPDATED_PHASE4_HISTORY_PRESERVED'}.items():manifest.loc[mask,k]=v
        else:
            record=dict(result_id=f'P4ART{len(manifest)+1:04d}',path=str(p),relative_path=rel,artifact_type=p.suffix.lstrip('.'),methodology_version='POSITION_RESIDUAL_EXPLORATORY_NOT_FROZEN',source_reference='MIN270 primary; preserved prior phases; position and residual plans',verification_status='GENERATED_QA_CHECKED',bytes=p.stat().st_size,sha256=digest,modified_utc=modified)
            manifest=pd.concat([manifest,pd.DataFrame([record])],ignore_index=True);known.add(rel)
    assert not manifest.relative_path.duplicated().any()
    manifest.to_csv(ROOT/'RESULTS_MANIFEST.csv',index=False)
    print(json.dumps(checks,indent=2));print('Manifest artifacts:',len(manifest))

if __name__=='__main__':
    if '--finalize' in sys.argv:finalize()
    else:generate();print('Reports and heatmap generated; visual review and finalization required.')
