"""Phase 4: fixed exposure thresholds; reuse Phase 3 estimators unchanged."""
from pathlib import Path
import importlib.util, json, hashlib, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'02_data/final';M=ROOT/'03_models';V=ROOT/'04_validation';T=ROOT/'08_tables';R=ROOT/'09_reports'
spec=importlib.util.spec_from_file_location('phase3',M/'tools/corrected_modeling.py')
p3=importlib.util.module_from_spec(spec);spec.loader.exec_module(p3)
MANUAL={'Gls_per90':'Gls','Ast_per90':'Ast','GA_per90':'G+A','npGoals_per90':'npGoals',
        'PrgC_per90':'PrgC_Carries','PrgDist_per90':'PrgDist_Carries','Final_Third_per90':'Final_Third_Carries',
        'PrgP_per90':'PrgP','Succ_Take_per90':'Succ_Take','KP_per90':'KP','Tkl_per90':'Tkl_Tackles',
        'TklInt_per90':'Tkl+Int','Int_per90':'Int','TklW_per90':'TklW','Clr_per90':'Clr','Blocks_per90':'Blocks_Blocks','Recov_per90':'Recov'}
PROVIDER={'xG_Per':'xG_Expected','xAG_Per':'xAG_Expected','npxG+xAG_Per':'npxG+xAG_Expected','GA90':'goalkeeper goals conceded (not retained as a raw count)'}
ALIASES=['Gls_Per','Ast_Per','G+A_Per','G_minus_PK_Per','G+A_minus_PK_Per','xG+xAG_Per','npxG_Per']

def save(x,name,folder=V): pd.DataFrame(x).to_csv(folder/name,index=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def errors(pred,source):
    out=pred.merge(source[['original_clean_row_id','player_name','position','sub_position','Mins_Per_90_Playing']],left_on='row_id',right_on='original_clean_row_id',validate='many_to_one')
    out['residual']=out.actual_log_fee-out.predicted_log_fee
    out['absolute_error']=out.residual.abs();out['squared_error']=out.residual**2
    return out
def error_metrics(g):
    return dict(n=len(g),mean_absolute_error=float(g.absolute_error.mean()),median_absolute_error=float(g.absolute_error.median()),
                rmse=float(np.sqrt(g.squared_error.mean())),max_absolute_error=float(g.absolute_error.max()),p99_absolute_error=float(g.absolute_error.quantile(.99)))

def main():
    # Preserve every prior artifact before producing any Phase 4 analysis output.
    if not (M/'PHASE3_RESULTS_MANIFEST.csv').exists():shutil.copy2(ROOT/'RESULTS_MANIFEST.csv',M/'PHASE3_RESULTS_MANIFEST.csv')
    plan=dict(phase='PHASE4_PER90_STABILITY',primary_threshold_90s=3.,sensitivity_thresholds_90s=[5.,10.],
              fixed_before_revised_performance=True,source='User-prespecified thresholds, not optimized on validation scores',
              threshold_column='Mins_Per_90_Playing',threshold_scale='Recorded rounded 90-minute equivalents; 270/450/900 are nominal minutes',
              structural='A',age_spec=1,seed=42,contract=False,origin_predictors=False,
              candidates=p3.CAND,phase3_implementation_sha256=sha(M/'tools/corrected_modeling.py'),
              selection='Unchanged training-local candidate VIF<=5 then backward simultaneous AIC/BIC improvement',
              nested_cv='5 outer development folds; preprocessing and selection inside training. Age fixed by request; no age tuning or inner hyperparameter grid',
              unknown_future_category='Unchanged training-frequency-weighted encoding',
              primary_winsorization=False,no_player_specific_corrections=True)
    (M/'PER90_MINUTES_ANALYSIS_PLAN.json').write_text(json.dumps(plan,indent=2))
    source=pd.read_csv(M/'common_modeling_sample.csv')
    assert len(source)==1115 and source.original_clean_row_id.is_unique
    exposure=source.Mins_Per_90_Playing
    assert exposure.notna().all() and exposure.ge(0).all()
    temporal=pd.read_csv(V/'temporal_oos_predictions.csv')
    e=errors(temporal,source);save(e,'PER90_PHASE3_temporal_errors.csv')

    # Manual ratios can be reconstructed exactly; cached provider rates are not
    # recomputed from the rounded denominator or silently replaced.
    definitions=[]
    for feature,numerator in MANUAL.items():
        valid=exposure.gt(0)&source[feature].notna()&source[numerator].notna()
        delta=(source.loc[valid,feature]-source.loc[valid,numerator]/exposure[valid]).abs()
        assert delta.max()<1e-10,feature
        definitions.append(dict(feature=feature,in_primary_file=True,in_model_candidates=feature in p3.CAND,
            family='PROJECT_COMPUTED_PER90',numerator=numerator,denominator='Recorded Mins_Per_90_Playing',
            source='original_cleaning_from_cache.R:172-188',verified_rows=int(valid.sum()),max_reconstruction_error=float(delta.max()),
            missing_n=int(source[feature].isna().sum()),nonfinite_nonmissing_n=int((source[feature].notna()&~np.isfinite(source[feature])).sum()),
            note='GA_per90 is goals plus assists, not goalkeeper goals conceded' if feature=='GA_per90' else 'No rate recomputation in revised datasets'))
    for feature,numerator in PROVIDER.items():
        definitions.append(dict(feature=feature,in_primary_file=True,in_model_candidates=feature in p3.CAND,
            family='PROVIDER_PER90',numerator=numerator,denominator='Provider minutes / 90; exact upstream calculation not independently available',
            source='original_fbref_fetch.R:standard_data or keepers_data -> clean_data',missing_n=int(source[feature].isna().sum()),
            nonfinite_nonmissing_n=int((source[feature].notna()&~np.isfinite(source[feature])).sum()),
            note='Already per90; rounded published 90s need not reproduce provider rate exactly'))
    for feature in ALIASES:
        definitions.append(dict(feature=feature,in_primary_file=False,in_model_candidates=False,family='UPSTREAM_PROVIDER_PER90_ALIAS',
            source='original_clean_lineage.csv / cached standard_data.csv',denominator='Provider minutes / 90',note='Available upstream, not retained in primary CSV and not added to model'))
    for feature in ['Cmp_percent_Total','Won_percent_Aerial','Save_percent','CS_percent','PSxG','Mins_Per_90_Playing']:
        definitions.append(dict(feature=feature,in_primary_file=True,in_model_candidates=feature in p3.CAND,
            family='EXPOSURE' if feature=='Mins_Per_90_Playing' else 'NOT_A_MINUTE_NORMALIZED_RATE',
            denominator='Pass attempts / aerial duels / shots faced / starts as appropriate; PSxG is cumulative' if feature!='Mins_Per_90_Playing' else 'minutes / 90',
            source='Cached provider tables',note='Do not classify percentages or cumulative PSxG as per90; percentage reliability can still depend on opportunity count'))
    save(definitions,'PER90_FEATURE_DEFINITIONS.csv')
    audit=[]
    for name,q in [('minimum',0),('p01',.01),('p025',.025),('p05',.05),('p10',.1),('p25',.25),('median',.5)]:
        audit.append(dict(record_type='exposure_quantile',statistic=name,value=float(exposure.quantile(q)),units='90s',n=len(source)))
    for threshold in [1,2,3,5,10]:
        low=source[exposure.lt(threshold)]
        audit.append(dict(record_type='low_exposure_count',threshold_90s=threshold,n=len(low),percent=100*len(low)/len(source)))
        for group in ['transfer_season','position','sub_position']:
            for category,g in source.groupby(group):
                n=int(g.Mins_Per_90_Playing.lt(threshold).sum())
                audit.append(dict(record_type='low_exposure_group_count',threshold_90s=threshold,group=group,category=category,n=n,group_total=len(g),percent=100*n/len(g)))
    save(audit,'PER90_EXPOSURE_AUDIT.csv')

    lineage=pd.read_csv(ROOT/'02_data/intermediate/original_clean_lineage.csv',low_memory=False).set_index('original_clean_row_id')
    raw_path=ROOT/'01_audit/reproduction/fresh_cache_attempt/downloaded_tables/standard_data.csv'
    raw=pd.read_csv(raw_path,low_memory=False)
    keys=['Player','Squad','Season_End_Year']
    assert not raw.duplicated(keys).any()
    raw=raw.set_index(keys)
    minute_rows=[]
    for row in source.itertuples():
        lin=lineage.loc[row.original_clean_row_id];key=(row.player_name,lin.Squad,lin.Season_End_Year)
        if key in raw.index:
            r=raw.loc[key];agree=np.isclose(r.Mins_Per_90_Playing,row.Mins_Per_90_Playing)
            minute_rows.append(dict(original_clean_row_id=row.original_clean_row_id,player_name=row.player_name,transfer_season=row.transfer_season,
                selected_squad=lin.Squad,performance_end_year=lin.Season_End_Year,recorded_90s=row.Mins_Per_90_Playing,
                nominal_minutes=row.Mins_Per_90_Playing*90,cached_exact_minutes=r.Min_Playing,cached_denominator_agrees=bool(agree),
                source='Phase 1 cached standard_data.csv; selected Big5 squad only'))
        else:
            minute_rows.append(dict(original_clean_row_id=row.original_clean_row_id,player_name=row.player_name,transfer_season=row.transfer_season,recorded_90s=row.Mins_Per_90_Playing,nominal_minutes=row.Mins_Per_90_Playing*90,cached_denominator_agrees=False,source='NO_EXACT_KEY_MATCH'))
    minute_rows=pd.DataFrame(minute_rows);save(minute_rows,'PER90_EXACT_MINUTES_CROSSCHECK.csv')
    examples=e[e.model.eq('Model 4')&e.transfer_season.eq('2022-23')&e.player_name.isin(['Neco Williams','Pablo Sarabia'])].copy()
    assert len(examples)==2
    examples=examples.merge(source[['original_clean_row_id']+p3.CAND],on='original_clean_row_id',suffixes=('','_rate'),validate='one_to_one').merge(minute_rows[['original_clean_row_id','cached_exact_minutes','nominal_minutes','selected_squad']],on='original_clean_row_id',validate='one_to_one')
    save(examples,'PER90_VERIFIED_EXAMPLES.csv')

    e['exposure_bin']=pd.cut(e.Mins_Per_90_Playing,[-np.inf,.25,.5,1,2,3,5,10,np.inf],right=False,labels=['<0.25','0.25-0.5','0.5-1','1-2','2-3','3-5','5-10','10+'])
    concentration=[]
    for season in ['ALL']+sorted(e.transfer_season.unique()):
        scope=e if season=='ALL' else e[e.transfer_season.eq(season)]
        for model,g in scope.groupby('model'):
            for category in e.exposure_bin.cat.categories:
                sub=g[g.exposure_bin.eq(category)]
                concentration.append(dict(record_type='EXPOSURE_BIN',test_season=season,model=model,exposure_bin=category,**error_metrics(sub)))
            if model=='Model 4':
                total=float(g.squared_error.sum())
                for k in [1,2,5,10]:
                    top=g.nlargest(k,'squared_error')
                    concentration.append(dict(record_type='TOP_K_SQUARED_ERROR',test_season=season,model=model,k=k,n=len(g),top_k_n=len(top),total_squared_error=total,top_squared_error=float(top.squared_error.sum()),percentage_total_squared_error=100*top.squared_error.sum()/total,top_row_ids=p3.js(top.row_id.tolist())))
    season22=e[e.model.eq('Model 4')&e.transfer_season.eq('2022-23')]
    concentration.append(dict(record_type='VERIFIED_TWO_EXAMPLES',test_season='2022-23',model='Model 4',n=len(season22),k=2,total_squared_error=season22.squared_error.sum(),top_squared_error=examples.squared_error.sum(),percentage_total_squared_error=100*examples.squared_error.sum()/season22.squared_error.sum(),top_row_ids=p3.js(examples.row_id.tolist())))
    save(concentration,'per90_error_concentration.csv')

    variants={'NO_MINIMUM':source}
    filenames={3:'primary_min270_dataset.csv',5:'sensitivity_min450_dataset.csv',10:'sensitivity_min900_dataset.csv'}
    ledger=[];summary=[]
    for cutoff,filename in filenames.items():
        label=f'MIN{cutoff*90}';d=source[exposure.ge(cutoff)].copy()
        save(d,filename,D);variants[label]=d
        exact=minute_rows[minute_rows.original_clean_row_id.isin(d.original_clean_row_id)]
        summary.append(dict(dataset=label,threshold_90s=cutoff,nominal_minutes=cutoff*90,n=len(d),excluded_n=len(source)-len(d),unique_players=d.player_id.nunique(),
            season_distribution=p3.counts(d.transfer_season),position_distribution=p3.counts(d.position),sub_position_distribution=p3.counts(d.sub_position),
            destination_distribution=p3.counts(d.to_league),cached_exact_minutes_below_nominal_n=int((exact.cached_denominator_agrees & exact.cached_exact_minutes.lt(cutoff*90)).sum())))
        for r in source.itertuples():ledger.append(dict(dataset=label,row_id=r.original_clean_row_id,player_id=r.player_id,recorded_90s=r.Mins_Per_90_Playing,included=r.Mins_Per_90_Playing>=cutoff,reason='INCLUDED' if r.Mins_Per_90_Playing>=cutoff else 'BELOW_PRESPECIFIED_EXPOSURE_THRESHOLD'))
    save(ledger,'PER90_threshold_row_ledger.csv');save(summary,'PER90_threshold_sample_summary.csv',T)
    distributions=[]
    for label,d in variants.items():
        for feature in list(MANUAL)+list(PROVIDER)+['Cmp_percent_Total','Won_percent_Aerial','Mins_Per_90_Playing']:
            s=d[feature].replace([np.inf,-np.inf],np.nan)
            distributions.append(dict(dataset=label,n=len(d),feature=feature,finite_n=int(s.notna().sum()),missing_n=int(d[feature].isna().sum()),nonfinite_nonmissing_n=int((d[feature].notna()&~np.isfinite(d[feature])).sum()),median=s.median(),p95=s.quantile(.95),p99=s.quantile(.99),maximum=s.max()))
    save(distributions,'PER90_feature_distributions_by_threshold.csv',T)

    # Fit unchanged Phase 3 code. Only eligibility changes; age is fixed at 1.
    packages={}
    for label,d in variants.items():
        if label=='NO_MINIMUM':continue
        models=(1,2,3,4) if label=='MIN270' else (1,4)
        a,b=train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
        tr,te=d.iloc[a],d.iloc[b];hold=[];preds=[]
        save([dict(row_id=int(r),split='train' if r in set(tr.original_clean_row_id) else 'test') for r in d.original_clean_row_id],f'{label}_holdout_membership.csv')
        for model in models:
            row,pred,fit=p3.evaluate(tr,te,model,1,label+'/holdout','random')
            row['n']=len(d);hold.append(row);preds.append(pred)
            art=dict(model=model,age_spec=1,structural='A',training_rows=list(map(int,tr.original_clean_row_id)),design=fit['prep'].__dict__,selected_columns=fit['cols'],coefficients=fit['coef'].tolist())
            (M/f'{label}_model{model}_holdout_fitted.json').write_text(json.dumps(art,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
        hold.append(dict(model='Training mean',n=len(d),train_n=len(tr),test_n=len(te),effective_predictors=0,train_r2=0.,**p3.metrics(te.log_transfer_fee,np.repeat(tr.log_transfer_fee.mean(),len(te)))))
        hold=pd.DataFrame(hold);save(hold,f'{label}_common_sample_holdout_results.csv',M);save(pd.concat(preds),f'{label}_holdout_predictions.csv')
        pack={'holdout':hold,'n':len(d)}
        for kind,population,prefix in [('random',tr,'nested'),('group',d,'player_group'),('temporal',d,'temporal')]:
            rr,pp=p3.validate(population,kind,label+'_'+prefix,models=models,fixed_age=1)
            save(rr,f'{label}_{prefix}_validation_results.csv');save(p3.summarize(rr),f'{label}_{prefix}_validation_summary.csv');save(pp,f'{label}_{prefix}_oof_predictions.csv')
            pack[kind]=rr;pack[kind+'_pred']=pp
        packages[label]=pack
        print('COMPLETED',label,'n=',len(d),flush=True)
    packages['NO_MINIMUM']=dict(n=len(source),holdout=pd.read_csv(M/'common_sample_holdout_results.csv'),
        group=pd.read_csv(V/'player_group_validation_results.csv'),temporal=pd.read_csv(V/'temporal_validation_results.csv'),temporal_pred=temporal)
    threshold_results=[]
    for label,pack in packages.items():
        for model in ['Model 1','Model 4']:
            h=pack['holdout'][pack['holdout'].model.eq(model)].iloc[0];g=pack['group'][pack['group'].model.eq(model)]
            tp=errors(pack['temporal_pred'][pack['temporal_pred'].model.eq(model)],source)
            scores=p3.metrics(tp.actual_log_fee,tp.predicted_log_fee)
            row=dict(dataset=label,model=model,n=pack['n'],holdout_r2=h.r2,holdout_rmse=h.rmse,holdout_mae=h.mae,
                grouped_r2_mean=g.r2.mean(),grouped_r2_sd=g.r2.std(ddof=1),grouped_rmse_mean=g.rmse.mean(),grouped_rmse_sd=g.rmse.std(ddof=1),grouped_mae_mean=g.mae.mean(),grouped_mae_sd=g.mae.std(ddof=1),
                temporal_n=len(tp),temporal_pooled_r2=scores['r2'],temporal_pooled_rmse=scores['rmse'],temporal_pooled_mae=scores['mae'],temporal_max_absolute_error=tp.absolute_error.max(),temporal_p99_absolute_error=tp.absolute_error.quantile(.99))
            for r in pack['temporal'][pack['temporal'].model.eq(model)].itertuples():row['temporal_r2_'+r.test_season]=r.r2
            threshold_results.append(row)
    save(threshold_results,'minutes_threshold_sensitivity.csv')
    pooled=[]
    min_errors=errors(packages['MIN270']['temporal_pred'],source)
    save(min_errors,'MIN270_temporal_errors.csv')
    for model,g in min_errors.groupby('model'):
        pooled.append({'model':model,**p3.metrics(g.actual_log_fee,g.predicted_log_fee),**error_metrics(g)})
    save(pooled,'MIN270_temporal_pooled_results.csv')
    after=min_errors[min_errors.model.eq('Model 4')&min_errors.transfer_season.eq('2022-23')]
    retained_before=season22[season22.row_id.isin(after.row_id)]
    diagnostic=[];top=[]
    for label,g in [('PHASE3_ALL',season22),('PHASE3_PREDICTIONS_RETAINED_ROWS_NO_REFIT',retained_before),('MIN270_REFIT_RETAINED_ROWS',after)]:
        diagnostic.append({'comparison':label,**p3.metrics(g.actual_log_fee,g.predicted_log_fee),**error_metrics(g)})
        top.append(g.nlargest(5,'absolute_error').assign(comparison=label))
    save(diagnostic,'MIN270_2022_23_failure_diagnostic.csv');save(pd.concat(top),'MIN270_2022_23_top5_errors.csv')
    save(p3.COEFS,'PER90_MINUTES_coefficient_stability.csv')
    save(p3.SELECTION,'PER90_MINUTES_feature_selection_by_fit.csv')
    stability=pd.DataFrame(p3.SELECTION);stability['evaluation']=stability.label.str.rsplit('/',n=1).str[0]
    save(stability.groupby(['evaluation','feature']).selected.agg(['sum','count','mean']).reset_index(),'PER90_MINUTES_feature_selection_frequency.csv')
    (V/'PER90_MINUTES_training_audit.json').write_text(json.dumps(p3.RUN_LOG,indent=2))
    (V/'PER90_MINUTES_split_audit.json').write_text(json.dumps(p3.SPLITS,indent=2))
    complete=dict(status='COMPLETE',primary_threshold=3.,primary_n=len(variants['MIN270']),min450_n=len(variants['MIN450']),min900_n=len(variants['MIN900']),fit_n=len(p3.RUN_LOG),no_winsorization=True,no_position_analysis=True,no_market_premium_interpretation=True,no_abstract=True)
    (M/'PER90_MINUTES_RUN_COMPLETE.json').write_text(json.dumps(complete,indent=2))
    print(json.dumps(complete,indent=2),flush=True)

if __name__=='__main__':main()
