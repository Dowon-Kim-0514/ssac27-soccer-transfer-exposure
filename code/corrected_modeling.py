"""Phase 3: common-sample, training-local linear modeling and honest validation."""
from pathlib import Path
import hashlib, json, shutil, sys, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]/"data"/"work"
D=ROOT/'datasets/final'; M=ROOT/'models'; V=ROOT/'validation'; T=ROOT/'tables'; R=ROOT/'reports'
SEED=42
TRAD=['Gls_per90','Ast_per90','Mins_Per_90_Playing']
ADV=['xG_Per','Succ_Take_per90','PrgC_per90','Cmp_percent_Total','KP_per90','Won_percent_Aerial','Recov_per90']
CAND=TRAD+ADV
FORBIDDEN={'market_value_in_eur','transfer_fee','transfer_fee_raw','log_transfer_fee','player_id','original_clean_row_id','covid_dummy','season_proxy_flag','UEFA_coeff_from','league_level_diff'}
AGE_LOG=[]; RUN_LOG=[]; SPLITS=[]; COEFS=[]; SELECTION=[]

def save(x,name,folder=V):
    pd.DataFrame(x).to_csv(folder/name,index=False)

def js(x): return json.dumps(x,sort_keys=True,default=lambda a: a.item() if hasattr(a,'item') else str(a))
def counts(x): return js({str(k):int(v) for k,v in x.value_counts(dropna=False).sort_index().items()})
def rowhash(d): return hashlib.sha256(','.join(map(str,sorted(d.original_clean_row_id))).encode()).hexdigest()
def metrics(y,p): return dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)))
def fitted(X,y):
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    return b,X@b
def vif(X):
    z=X[:,1:]; sd=z.std(axis=0)
    if not len(sd): return np.array([])
    z=(z-z.mean(axis=0))/sd
    return np.diag(np.linalg.pinv(z.T@z/len(z)))

class Design:
    """Training-only medians/modes/levels/scaling; unknown categories use train proportions."""
    def __init__(self,model,age=1,structural='A',contract=False,raw_age=False):
        self.performance=[] if model==1 else TRAD if model==2 else ADV if model==3 else CAND
        self.numeric=(['age_at_transfer','age_squared'] if raw_age else ['age_centered','age_centered_squared'])+(['U21_dummy','O30_dummy'] if age==2 else [])+(['contract_years_remaining'] if contract else [])+self.performance
        self.categorical=['sub_position','to_league','transfer_season']+(['from_league'] if structural=='B' else [])
        assert not FORBIDDEN.intersection(self.numeric+self.categorical)
        self.model=model

    def fit(self,d):
        self.medians={}; self.modes={}; self.levels={}; self.frequencies={}
        for c in self.numeric:
            s=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
            self.medians[c]=float(s.median()) if s.notna().any() else 0.
        for c in self.categorical:
            s=d[c]; self.modes[c]=str(s.mode().iloc[0]) if s.notna().any() else 'MISSING'
            x=s.fillna(self.modes[c]).astype(str)
            self.levels[c]=sorted(x.unique()); self.frequencies[c]=x.value_counts(normalize=True).to_dict()
        self.names=['const']+self.numeric+[c+'='+lev for c in self.categorical for lev in self.levels[c][1:]]
        a=self.raw(d)
        self.means=a[:,1:].mean(axis=0); self.scales=a[:,1:].std(axis=0)
        self.scales[self.scales<1e-12]=1.
        z=np.column_stack([np.ones(len(d)),(a[:,1:]-self.means)/self.scales])
        # Deterministic training-only alias handling protects sparse inner folds.
        # Preserve structural controls before candidate performance terms.
        order=[0]+[i for i,n in enumerate(self.names) if i and n not in self.performance]+[i for i,n in enumerate(self.names) if n in self.performance]
        keep=[]
        for i in order:
            trial=z[:,keep+[i]]
            if np.linalg.matrix_rank(trial)>len(keep):keep.append(i)
        self.keep=keep; self.aliases=[n for i,n in enumerate(self.names) if i not in keep]
        self.kept_names=[self.names[i] for i in keep]
        self.train_hash=rowhash(d)
        return self

    def raw(self,d):
        cols=[np.ones(len(d))]
        for c in self.numeric:
            cols.append(pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(self.medians[c]).to_numpy(float))
        for c in self.categorical:
            x=d[c].fillna(self.modes[c]).astype(str)
            known=x.isin(self.levels[c]).to_numpy()
            for lev in self.levels[c][1:]:
                cols.append(np.where(known,x.eq(lev).to_numpy(float),self.frequencies[c][lev]))
        return np.column_stack(cols)

    def transform(self,d):
        a=self.raw(d)
        return np.column_stack([np.ones(len(d)),(a[:,1:]-self.means)/self.scales])[:,self.keep]

def fit_model(train,model,age=1,structural='A',contract=False,forced_candidates=None):
    prep=Design(model,age,structural,contract).fit(train)
    X=prep.transform(train); y=train.log_transfer_fee.to_numpy()
    names=prep.kept_names; structural_idx=[i for i,n in enumerate(names) if n not in prep.performance]
    cand=[i for i,n in enumerate(names) if n in prep.performance]
    history=[]
    if forced_candidates is not None:
        cand=[i for i in cand if names[i] in forced_candidates]
        history.append(dict(step='MATCH_WITHOUT_CONTRACT_TRAINING_SELECTION',retained=forced_candidates))
    if model==4 and forced_candidates is None:
        while len(cand)>1:
            vc=vif(np.column_stack([np.ones(len(train)),X[:,cand]]))
            if vc.max()<=5+1e-10: break
            j=int(np.argmax(vc)); history.append(dict(step='CANDIDATE_VIF',removed=names[cand[j]],value=float(vc[j])));cand.pop(j)
        def criteria(cols):
            _,p=fitted(X[:,cols],y); n=len(y); k=len(cols)
            dev=n*np.log(max(float(np.sum((y-p)**2))/n,1e-30))
            return dev+2*k,dev+np.log(n)*k
        current=criteria(structural_idx+cand)
        while cand:
            choices=[]
            for c in cand:
                a,b=criteria(structural_idx+[j for j in cand if j!=c])
                if a<current[0]-1e-9 and b<current[1]-1e-9: choices.append((b,a,c))
            if not choices:break
            b,a,c=min(choices);cand.remove(c);current=(a,b)
            history.append(dict(step='TRAIN_ONLY_AIC_AND_BIC',removed=names[c],aic=float(a),bic=float(b)))
    selected=structural_idx+cand
    coef,p=fitted(X[:,selected],y)
    original_coef={}
    for j,c in enumerate(selected):
        original_col=prep.keep[c]
        original_coef[names[c]]=float(coef[j]/prep.scales[original_col-1]) if original_col else float(coef[j])
    # Convert the standardized intercept back to the raw encoded basis.
    original_coef['const']=float(coef[0]-sum(original_coef[names[c]]*prep.means[prep.keep[c]-1] for c in selected if prep.keep[c]))
    return dict(prep=prep,cols=selected,coef=coef,train_pred=p,names=[names[c] for c in selected],
                selected=[names[c] for c in cand],history=history,raw_coef=original_coef,
                max_vif=float(vif(X[:,selected]).max()),rank=int(np.linalg.matrix_rank(X[:,selected])))

def predict(fit,test): return fit['prep'].transform(test)[:,fit['cols']]@fit['coef']

def splits(d,kind):
    if kind=='random':return list(KFold(5,shuffle=True,random_state=SEED).split(d))
    if kind=='group':return list(GroupKFold(5).split(d,groups=d.player_id))
    seasons=sorted(d.transfer_season.unique());out=[]
    for i in range(2,len(seasons)):
        tr=np.flatnonzero(d.transfer_season.isin(seasons[:i]));te=np.flatnonzero(d.transfer_season.eq(seasons[i]))
        if len(tr)>=100 and len(te)>=20:out.append((tr,te))
    return out

def record_split(train,test,label,kind):
    assert not set(train.original_clean_row_id)&set(test.original_clean_row_id)
    overlap=len(set(train.player_id)&set(test.player_id))
    if kind=='group':assert overlap==0
    if kind=='temporal':assert pd.to_datetime(train.transfer_date).max()<pd.to_datetime(test.transfer_date).min()
    SPLITS.append(dict(label=label,kind=kind,train_row_ids=list(map(int,train.original_clean_row_id)),test_row_ids=list(map(int,test.original_clean_row_id)),player_overlap=overlap))

def choose_age(train,kind,label):
    """Paired inner validation with prespecified parsimony and stability safeguards."""
    out=[]; ss=splits(train,kind)
    if len(ss)<3:
        AGE_LOG.append(dict(label=label,selected_age=1,reason='Fewer than three historical inner splits; prespecified quadratic-only fallback',inner_folds=len(ss)))
        return 1
    for f,(a,b) in enumerate(ss,1):
        tr,va=train.iloc[a],train.iloc[b]
        record_split(tr,va,label+f'/age_inner_{f}',kind)
        for age in [1,2]:
            fit=fit_model(tr,1,age)
            m=metrics(va.log_transfer_fee,predict(fit,va))
            out.append(dict(fold=f,age=age,**m,max_vif=fit['max_vif'],u21=fit['raw_coef'].get('U21_dummy',0),o30=fit['raw_coef'].get('O30_dummy',0)))
    q=pd.DataFrame(out); rm=q.pivot(index='fold',columns='age',values='rmse'); gains=rm[1]-rm[2]
    se=float(gains.std(ddof=1)/np.sqrt(len(gains)))
    threshold=q[q.age.eq(2)]
    signs=all(max((threshold[c]>0).sum(),(threshold[c]<0).sum())>=np.ceil(.8*len(threshold)) for c in ['u21','o30'])
    stable=float(threshold.max_vif.max())<=10 and signs
    selected=2 if gains.mean()>max(se,0) and (gains>0).sum()>=np.ceil(.8*len(gains)) and stable else 1
    AGE_LOG.append(dict(label=label,selected_age=selected,inner_folds=len(ss),rmse_gain=float(gains.mean()),paired_gain_se=se,
                        threshold_max_vif=float(threshold.max_vif.max()),threshold_signs_stable=bool(signs),
                        reason='Thresholds require paired RMSE gain > SE, >=80% fold wins, VIF<=10 and >=80% sign consistency; otherwise simpler quadratic',fold_details=out))
    return selected

def evaluate(train,test,model,age,label,kind,structural='A',contract=False):
    record_split(train,test,label+f'/M{model}/{structural}',kind)
    forced=fit_model(train,model,age,structural,False)['selected'] if contract else None
    fit=fit_model(train,model,age,structural,contract,forced_candidates=forced);p=predict(fit,test)
    m=metrics(test.log_transfer_fee,p); res=test.log_transfer_fee.to_numpy()-p
    row=dict(label=label,model=f'Model {model}',structural=structural,age_spec=age,train_n=len(train),test_n=len(test),
             effective_predictors=len(fit['cols'])-1,train_r2=float(r2_score(train.log_transfer_fee,fit['train_pred'])),**m,
             mean_residual=float(res.mean()),median_residual=float(np.median(res)),selected_features=js(fit['selected']),
             full_design_rank=fit['rank'],full_design_columns=len(fit['cols']),max_full_design_vif=fit['max_vif'],
             train_row_hash=rowhash(train),test_row_hash=rowhash(test),contract=contract)
    pred=pd.DataFrame(dict(row_id=test.original_clean_row_id.to_numpy(),player_id=test.player_id.to_numpy(),actual_log_fee=test.log_transfer_fee.to_numpy(),predicted_log_fee=p,model=f'Model {model}',fold=label,transfer_season=test.transfer_season.to_numpy(),structural=structural))
    RUN_LOG.append(dict(**row,imputation=fit['prep'].medians,category_levels=fit['prep'].levels,aliases_removed=fit['prep'].aliases,
                        selection_history=fit['history'],train_ids=list(map(int,train.original_clean_row_id))))
    for feature,value in fit['raw_coef'].items():COEFS.append(dict(label=label,model=f'Model {model}',structural=structural,feature=feature,coefficient=value))
    if model==4:
        for c in CAND:SELECTION.append(dict(label=label,feature=c,selected=c in fit['selected']))
    return row,pred,fit

def validate(d,kind,prefix,models=(1,2,3,4),structural='A',fixed_age=None,contract=False):
    rows=[];preds=[]
    for fold,(a,b) in enumerate(splits(d,kind),1):
        tr,te=d.iloc[a],d.iloc[b];label=f'{prefix}/{kind}/{fold}'
        age=choose_age(tr,kind,label) if fixed_age is None else fixed_age
        for model in models:
            row,pred,_=evaluate(tr,te,model,age,label,kind,structural,contract)
            row.update(fold=fold,training_seasons=js(sorted(tr.transfer_season.unique())),test_season='|'.join(sorted(te.transfer_season.unique())))
            rows.append(row);preds.append(pred)
        base=np.repeat(tr.log_transfer_fee.mean(),len(te))
        rows.append(dict(label=label,model='Training mean',structural=structural,age_spec=age,train_n=len(tr),test_n=len(te),**metrics(te.log_transfer_fee,base),fold=fold,training_seasons=js(sorted(tr.transfer_season.unique())),test_season='|'.join(sorted(te.transfer_season.unique()))))
        print('Finished',label,flush=True)
    return pd.DataFrame(rows),pd.concat(preds,ignore_index=True)

def summarize(results):
    return results.groupby(['model']).agg(folds=('r2','size'),r2_mean=('r2','mean'),r2_sd=('r2','std'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index()

def add_age(d):
    d=d.copy();d['age_centered']=d.age_at_transfer-25;d['age_centered_squared']=d.age_centered**2
    assert d.age_at_transfer.notna().all()
    return d

def main():
    if not (M/'PHASE2_RESULTS_MANIFEST.csv').exists():shutil.copy2(ROOT/'RESULTS_MANIFEST.csv',M/'PHASE2_RESULTS_MANIFEST.csv')
    v4=pd.read_csv(D/'strict_pretransfer_identity_without_contract_dataset.csv')
    v5=pd.read_csv(D/'strict_pretransfer_identity_league_confirmed_without_contract_dataset.csv')
    audit=pd.read_csv(ROOT/'datasets/intermediate/club_league_timing_audit.csv').set_index('original_clean_row_id')
    ids=audit.index[audit.big5_destination_observed_in_transfer_season]
    primary=v4[v4.original_clean_row_id.isin(ids)].copy()
    assert primary.historical_big5_destination_status.eq('OBSERVED_BIG5').all()
    assert len(primary)==len(v4)-sum(~v4.original_clean_row_id.isin(ids))
    assert not {'contract_years_remaining','contract_expiration_date'}&set(primary)
    save(primary,'strict_pretransfer_identity_destination_confirmed_without_contract_dataset.csv',D)
    sample_rows=[]
    for label,d in [('V4',v4),('DESTINATION_CONFIRMED',primary),('V5',v5)]:
        freq=d.player_id.value_counts();missing=d[CAND].replace([np.inf,-np.inf],np.nan).isna()
        sample_rows.append(dict(dataset=label,n=len(d),unique_players=d.player_id.nunique(),unique_player_names=d.player_name.nunique(),
            season_distribution=counts(d.transfer_season),position_distribution=counts(d.position),destination_league_distribution=counts(d.to_league),origin_league_distribution=counts(d.from_league),
            missing_performance_cells=int(missing.sum().sum()),rows_with_missing_performance=int(missing.any(axis=1).sum()),missing_by_feature=js(missing.sum().to_dict()),
            repeated_players=int((freq>1).sum()),observations_of_repeated_players=int(freq[freq>1].sum()),origin_unconfirmed=int(d.original_clean_row_id.map(audit.from_classification).ne('SNAPSHOT_AGREES_WITH_OBSERVED_SEASON').sum())))
    save(sample_rows,'PRIMARY_SAMPLE_COMPARISON.csv',D)
    primary=add_age(primary);v4=add_age(v4);v5=add_age(v5)
    save(primary,'common_modeling_sample.csv',M)
    assert primary.original_clean_row_id.is_unique and primary.log_transfer_fee.notna().all()
    # Record the analysis plan before any target-based fitting or evaluation.
    plan=dict(seed=SEED,primary_n=len(primary),primary_structural='A',structural_A='Centered quadratic age, position, destination league, season; no origin or contract',
              structural_B='Add from_league only on two-sided corroborated matched rows; no league_level_diff',
              league_level_diff_exclusion='Sign of fixed hand-ranked GB1>ES1>L1>IT1>FR1 hierarchy, not historical competitive-strength measurement',
              traditional=TRAD,advanced=ADV,candidates=CAND,age_reference=25,
              age_rule='Inner Model 1 paired CV; retain thresholds only if gain > paired SE, >=80% wins, VIF<=10 and both signs >=80% consistent; <3 inner time folds defaults AGE_SPEC_1',
              model4_selection='Training-only candidate VIF>5 iterative pruning then backward deletion if BOTH AIC and BIC improve, choose lowest BIC then AIC',
              unseen_categories='Training-frequency-weighted dummy vector, not omitted reference category. Future season uses training observation-weighted average season effect; no future level fitted',
              evaluation='Holdout seed42 80/20; development-only nested5CV; full-sample player GroupKFold5 and expanding-window seasons 2020-21 onward, separately labeled',
              sensitivity_age_spec=1,sensitivity_age_rationale='Prespecified quadratic-only control; never selected using rows later serving as grouped/temporal validation',
              no_post_validation_tuning=True)
    (M/'ANALYSIS_PLAN.json').write_text(json.dumps(plan,indent=2))
    (M/'runtime_versions.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,packages={x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy']}),indent=2))

    # Raw/centered age equivalence is a numerical diagnostic, not model selection.
    diag=[]
    for age in [1,2]:
        designs=[]
        for raw in [True,False]:
            pr=Design(4,age,raw_age=raw).fit(primary)
            x=pr.raw(primary)[:,pr.keep]
            _,p=fitted(x,primary.log_transfer_fee.to_numpy()); designs.append(p)
            scaled=x/np.linalg.norm(x,axis=0);vv=vif(x)
            row=dict(age_spec=age,basis='raw' if raw else 'centered_25',n=len(primary),rank=int(np.linalg.matrix_rank(scaled)),columns=x.shape[1],condition_number=float(np.linalg.cond(x)),column_scaled_condition_number=float(np.linalg.cond(scaled)),max_vif=float(vv.max()))
            for name,val in zip(pr.kept_names[1:],vv):row['vif_'+name]=float(val)
            diag.append(row)
        delta=float(np.max(np.abs(designs[0]-designs[1])));assert delta<1e-8
        diag[-1]['max_abs_fitted_value_difference']=delta;diag[-2]['max_abs_fitted_value_difference']=delta
    save(diag,'age_centering_diagnostic.csv')

    tr_idx,te_idx=train_test_split(np.arange(len(primary)),test_size=.2,random_state=SEED)
    train,test=primary.iloc[tr_idx],primary.iloc[te_idx]
    save([dict(row_id=int(x),split='train' if x in set(train.original_clean_row_id) else 'test') for x in primary.original_clean_row_id],'random_holdout_membership.csv')
    age=choose_age(train,'random','primary_holdout_development')
    hold=[]
    for model in range(1,5):
        row,pred,fit=evaluate(train,test,model,age,'primary_holdout','random')
        row['n']=len(primary);hold.append(row);save(pred,f'model{model}_holdout_predictions.csv',M)
        artifact=dict(model=model,age_spec=age,structural='A',training_rows=list(map(int,train.original_clean_row_id)),
                      design=fit['prep'].__dict__,selected_columns=fit['cols'],coefficients=fit['coef'].tolist(),raw_coefficients=fit['raw_coef'])
        (M/f'model{model}_holdout_fitted.json').write_text(json.dumps(artifact,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    hold.append(dict(model='Training mean',n=len(primary),train_n=len(train),test_n=len(test),effective_predictors=0,train_r2=0.,**metrics(test.log_transfer_fee,np.repeat(train.log_transfer_fee.mean(),len(test)))))
    save(hold,'common_sample_holdout_results.csv',M)
    print('Holdout complete; selected age',age,flush=True)
    cv,cvpred=validate(train,'random','primary_nested_development')
    save(cv,'nested_validation_results.csv');save(summarize(cv),'nested_validation_summary.csv');save(cvpred,'nested_oof_predictions.csv')
    group,gpred=validate(primary,'group','primary_group')
    save(group,'player_group_validation_results.csv');save(summarize(group),'player_group_validation_summary.csv');save(gpred,'player_group_oof_predictions.csv')
    temporal,tpred=validate(primary,'temporal','primary_temporal')
    save(temporal,'temporal_validation_results.csv');save(tpred,'temporal_oos_predictions.csv')
    pooled=[]
    for model,g in tpred.groupby('model'):
        pooled.append(dict(model=model,n=len(g),**metrics(g.actual_log_fee,g.predicted_log_fee),mean_residual=float((g.actual_log_fee-g.predicted_log_fee).mean()),median_residual=float((g.actual_log_fee-g.predicted_log_fee).median())))
    save(pooled,'temporal_pooled_results.csv')
    freq=primary.player_id.value_counts()
    overlap=set(train.player_id)&set(test.player_id)
    (V/'player_dependence_summary.json').write_text(json.dumps(dict(unique_player_ids=primary.player_id.nunique(),repeated_players=int((freq>1).sum()),observations_of_repeated_players=int(freq[freq>1].sum()),holdout_overlapping_player_ids=len(overlap),holdout_rows_with_seen_player=int(test.player_id.isin(overlap).sum())),indent=2))

    # Matched origin predictor comparison uses exactly the same restricted rows.
    origin=[]
    for structural in ['A','B']:
        rr,_=validate(v5,'group',f'origin_matched_{structural}',models=(1,4),structural=structural,fixed_age=1)
        rr['validation']='group';origin.append(rr)
        rr,_=validate(v5,'temporal',f'origin_matched_{structural}',models=(1,4),structural=structural,fixed_age=1)
        rr['validation']='temporal';origin.append(rr)
    save(pd.concat(origin),'origin_predictor_comparison.csv')

    # Freeze the prespecified quadratic-only control for all sensitivity runs.
    # A development-selected age could otherwise leak into full-data group folds.
    sensitivity_age=1
    sensitivity=[]
    for label,d in [('PRIMARY',primary),('V4',v4),('V5',v5)]:
        a,b=train_test_split(np.arange(len(d)),test_size=.2,random_state=SEED)
        rr,_,_=evaluate(d.iloc[a],d.iloc[b],4,sensitivity_age,f'sensitivity_{label}/holdout','random')
        rr.update(dataset=label,validation='holdout',n=len(d));sensitivity.append(rr)
        for kind in ['group','temporal']:
            rr,_=validate(d,kind,f'sensitivity_{label}',models=(4,),fixed_age=sensitivity_age)
            for row in rr[rr.model.eq('Model 4')].to_dict('records'):
                row.update(dataset=label,validation=kind,n=len(d));sensitivity.append(row)
    save(sensitivity,'dataset_sensitivity_results.csv')

    original=pd.read_csv(D/'original_comparable_dataset.csv').set_index('original_clean_row_id')
    contracted=primary.copy();contracted['contract_years_remaining']=contracted.original_clean_row_id.map(original.contract_years_remaining)
    contracted=contracted[contracted.contract_years_remaining.notna()].copy()
    ctr=[]
    # Preserve primary holdout assignments rather than inventing a favorable split.
    tr=contracted[contracted.original_clean_row_id.isin(train.original_clean_row_id)]
    te=contracted[contracted.original_clean_row_id.isin(test.original_clean_row_id)]
    for with_contract in [False,True]:
        label='HISTORICALLY_UNVERIFIED_SENSITIVITY' if with_contract else 'MATCHED_WITHOUT_CONTRACT'
        rr,_,_=evaluate(tr,te,4,sensitivity_age,'contract_'+label,'random',contract=with_contract)
        rr.update(sensitivity=label,validation='holdout',n=len(contracted));ctr.append(rr)
        for kind in ['group','temporal']:
            rr,_=validate(contracted,kind,'contract_'+label,models=(4,),fixed_age=sensitivity_age,contract=with_contract)
            for row in rr[rr.model.eq('Model 4')].to_dict('records'):
                row.update(sensitivity=label,validation=kind,n=len(contracted));ctr.append(row)
    save(ctr,'contract_sensitivity_results.csv')
    # Selection is fold-local; report frequencies by evaluation family.
    stability=pd.DataFrame(SELECTION)
    stability['evaluation']=stability.label.str.rsplit('/',n=1).str[0]
    save(stability,'feature_selection_by_fit.csv')
    save(stability.groupby(['evaluation','feature']).selected.agg(['sum','count','mean']).reset_index().rename(columns={'sum':'selected_folds','count':'folds','mean':'selection_frequency'}),'feature_selection_stability.csv')
    save(COEFS,'coefficient_stability.csv')
    (V/'age_specification_decisions.json').write_text(json.dumps(AGE_LOG,indent=2))
    (V/'training_only_fit_audit.json').write_text(json.dumps(RUN_LOG,indent=2))
    (V/'all_validation_splits.json').write_text(json.dumps(SPLITS,indent=2))
    (M/'MODEL_RUN_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',primary_n=len(primary),selected_holdout_age=age,structural='A',models_fitted=len(RUN_LOG),no_position_analysis=True,no_market_residual_interpretation=True,no_abstract=True),indent=2))
    print('ALL REQUESTED MODEL RUNS COMPLETE',flush=True)

if __name__=='__main__': main()
