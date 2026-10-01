"""Fixed-design position exploration. All predictive comparisons use held-out rows."""
from pathlib import Path
import json, hashlib, importlib.util, shutil
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'05_position_analysis';E=ROOT/'06_residual_analysis';F=ROOT/'07_figures';T=ROOT/'08_tables';R=ROOT/'09_reports'
sp=importlib.util.spec_from_file_location('p3',ROOT/'03_models/tools/corrected_modeling.py');p3=importlib.util.module_from_spec(sp);sp.loader.exec_module(p3)
GROUPS=['FW','AM','CM/DM','SB','CB','GK'];OUTFIELD=GROUPS[:-1]
MAPPING={'Centre-Forward':'FW','Right Winger':'FW','Left Winger':'FW','Attacking Midfield':'AM','Second Striker':'AM',
         'Central Midfield':'CM/DM','Defensive Midfield':'CM/DM','Right Midfield':'CM/DM','Left Midfield':'CM/DM',
         'Left-Back':'SB','Right-Back':'SB','Centre-Back':'CB','Goalkeeper':'GK'}
BLOCKS={'FW':['Gls_per90','xG_Per','Ast_per90','KP_per90','Mins_Per_90_Playing'],
        'AM':['Ast_per90','KP_per90','PrgC_per90','Cmp_percent_Total','xAG_Per','Mins_Per_90_Playing'],
        'CM/DM':['Cmp_percent_Total','PrgC_per90','KP_per90','Recov_per90','Won_percent_Aerial','Mins_Per_90_Playing'],
        'SB':['PrgC_per90','Ast_per90','KP_per90','Cmp_percent_Total','Recov_per90','Mins_Per_90_Playing'],
        'CB':['Won_percent_Aerial','Cmp_percent_Total','Recov_per90','Clr_per90','TklInt_per90','Mins_Per_90_Playing'],
        'GK':['Save_percent','GA90','PSxG','Mins_Per_90_Playing']}
INTERACTIONS=['xG_Per','Gls_per90','Cmp_percent_Total','PrgC_per90','Won_percent_Aerial','KP_per90']
BOOT=1000;REPEATS=30;MIN_EFFECT=.01
FIT_LOG=[];PERM_LOG=[];SPLIT_LOG=[]

def seed(label):return int(hashlib.sha256(label.encode()).hexdigest()[:8],16)
def save(x,name,folder=P):pd.DataFrame(x).to_csv(folder/name,index=False)
def dump(x,name,folder=P): (folder/name).write_text(json.dumps(x,indent=2,default=lambda z:z.item() if hasattr(z,'item') else str(z)))
def rng(label):return np.random.default_rng(seed(label))
def prepare(name):
    d=pd.read_csv(ROOT/'02_data/final'/name);d['position_group']=d.sub_position.map(MAPPING)
    assert d.position_group.notna().all() and d.Mins_Per_90_Playing.ge(3).all()
    return d

def bootstrap_sums(ids,values,label):
    """Resample entire player clusters; fixed fitted predictions are not refitted."""
    _,idx=np.unique(np.asarray(ids),return_inverse=True);nclusters=idx.max()+1
    vals=np.asarray(values,float)
    if vals.ndim==1:vals=vals[:,None]
    sums=np.zeros((nclusters,vals.shape[1]));np.add.at(sums,idx,vals)
    weights=rng(label).multinomial(nclusters,np.repeat(1/nclusters,nclusters),size=BOOT)
    return weights@sums,nclusters

def paired(y,p0,p1,ids,label):
    e0=np.asarray(y)-p0;e1=np.asarray(y)-p1
    v=np.column_stack([np.ones(len(y)),e0**2,e1**2,abs(e0),abs(e1)])
    b,nc=bootstrap_sums(ids,v,label);rm=np.sqrt(b[:,1]/b[:,0])-np.sqrt(b[:,2]/b[:,0]);ma=(b[:,3]-b[:,4])/b[:,0]
    return dict(n=len(y),unique_players=nc,rmse_without=float(np.sqrt(np.mean(e0**2))),rmse_with=float(np.sqrt(np.mean(e1**2))),
        delta_rmse=float(np.sqrt(np.mean(e0**2))-np.sqrt(np.mean(e1**2))),rmse_ci_low=float(np.quantile(rm,.025)),rmse_ci_high=float(np.quantile(rm,.975)),
        mae_without=float(np.mean(abs(e0))),mae_with=float(np.mean(abs(e1))),delta_mae=float(np.mean(abs(e0))-np.mean(abs(e1))),
        mae_ci_low=float(np.quantile(ma,.025)),mae_ci_high=float(np.quantile(ma,.975)),
        r2_without=float(p3.metrics(y,p0)['r2']),r2_with=float(p3.metrics(y,p1)['r2']),ci_scope='Player-cluster bootstrap conditional on fitted folds; no model refit')

def classify_xg(row):
    if row['unique_players']<25 or row['n']<40 or row['folds']<3:return 'INSUFFICIENT_EVIDENCE'
    if row['delta_rmse']>=MIN_EFFECT and row['rmse_ci_low']>0:return 'CLEAR_INCREMENTAL_SIGNAL'
    if row['rmse_ci_high']<MIN_EFFECT:return 'NO_DETECTABLE_INCREMENTAL_SIGNAL'
    return 'WEAK_OR_MIXED_SIGNAL'

def custom_fit(train,features,interaction=None):
    prep=p3.Design(1,age=1,structural='A');prep.performance=list(features);prep.numeric=['age_centered','age_centered_squared']+list(features)
    prep.fit(train)
    base=prep.raw(train)[:,prep.keep];names=prep.kept_names.copy();cols=[];omitted=[]
    center=0.
    if interaction:
        raw=pd.to_numeric(train[interaction],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(prep.medians[interaction]).to_numpy()
        center=float(raw.mean())
        for group in GROUPS[1:]:
            c=(raw-center)*train.position_group.eq(group).to_numpy();trial=np.column_stack([base]+cols+[c]);norm=np.linalg.norm(trial,axis=0)
            if np.any(norm==0) or np.linalg.matrix_rank(trial/np.where(norm==0,1,norm))<trial.shape[1]:omitted.append(group)
            else:cols.append(c);names.append(interaction+' x '+group+' vs FW')
    X=np.column_stack([base]+cols)
    # Full-sample inference uses this fixed design only, never its fitted residuals.
    result=sm.OLS(train.log_transfer_fee.to_numpy(),X).fit()
    return dict(prep=prep,result=result,names=names,interaction=interaction,center=center,
                interaction_groups=[g for g in GROUPS[1:] if g not in omitted] if interaction else [],omitted=omitted,X=X)

def custom_predict(fit,test):
    base=fit['prep'].raw(test)[:,fit['prep'].keep]
    if fit['interaction']:
        f=fit['interaction'];raw=pd.to_numeric(test[f],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(fit['prep'].medians[f]).to_numpy()
        extra=[(raw-fit['center'])*test.position_group.eq(g).to_numpy() for g in fit['interaction_groups']]
        base=np.column_stack([base]+extra)
    return base@fit['result'].params

def log_fit(fit,train,test,label,kind,custom=False):
    assert not set(train.original_clean_row_id)&set(test.original_clean_row_id)
    if kind=='group':assert not set(train.player_id)&set(test.player_id)
    if kind=='temporal':assert pd.to_datetime(train.transfer_date).max()<pd.to_datetime(test.transfer_date).min()
    pre=fit['prep'];X=fit['X'] if custom else pre.transform(train)[:,fit['cols']]
    norms=np.linalg.norm(X,axis=0);rank=int(np.linalg.matrix_rank(X/np.where(norms==0,1,norms)))
    assert rank==X.shape[1],label
    FIT_LOG.append(dict(label=label,kind=kind,train_ids=train.original_clean_row_id.tolist(),test_ids=test.original_clean_row_id.tolist(),
        numeric_medians=pre.medians,categorical_levels=pre.levels,rank=rank,columns=X.shape[1],
        selected=fit.get('selected',pre.performance),aliases=pre.aliases,
        interaction=fit.get('interaction'),omitted_interactions=fit.get('omitted',[])))

def permutation_summary(parts,label,group,feature,folds_total,n_total,unique_total):
    if not parts:return dict(position=group,feature=feature,status='NOT_SELECTED_IN_ANY_EVALUABLE_FOLD',evaluation_n=0,position_evaluation_n=n_total,
        unique_players=0,position_unique_players=unique_total,folds_contributing=0,folds_available=folds_total,feature_selection_frequency=0.)
    base=np.concatenate([x['base_error'] for x in parts]);perm=np.concatenate([x['perm_error'] for x in parts]);ids=np.concatenate([x['ids'] for x in parts])
    n=len(base);delta=np.sqrt(np.mean(perm**2,axis=0))-np.sqrt(np.mean(base**2));dma=np.mean(abs(perm),axis=0)-np.mean(abs(base))
    vals=np.column_stack([np.ones(n),base**2,abs(base),perm**2,abs(perm)])
    b,nc=bootstrap_sums(ids,vals,label+group+feature)
    bootdelta=(np.sqrt(b[:,3:3+REPEATS]/b[:,[0]])-np.sqrt(b[:,[1]]/b[:,[0]])).mean(axis=1)
    foldmeans=[float((np.sqrt(np.mean(x['perm_error']**2,axis=0))-np.sqrt(np.mean(x['base_error']**2))).mean()) for x in parts]
    return dict(position=group,feature=feature,status='SELECTED_EVALUATED',evaluation_n=n,position_evaluation_n=n_total,unique_players=nc,position_unique_players=unique_total,
        folds_contributing=len(parts),folds_available=folds_total,feature_selection_frequency=len(parts)/folds_total,
        mean_delta_rmse=float(delta.mean()),median_delta_rmse=float(np.median(delta)),sd_delta_rmse=float(delta.std(ddof=1)),between_fold_sd=float(np.std(foldmeans,ddof=1)) if len(parts)>1 else np.nan,
        mean_delta_mae=float(dma.mean()),ci_low=float(np.quantile(bootdelta,.025)),ci_high=float(np.quantile(bootdelta,.975)),
        baseline_rmse=float(np.sqrt(np.mean(base**2))),baseline_mae=float(np.mean(abs(base))),permutation_repeats=REPEATS,
        uncertainty='Cluster-player bootstrap, fixed fits and permutations; selected folds only. SD across permutation replicates; between-fold SD separate.')

def global_position_eval(data,label,kind,wanted=None):
    wanted=wanted or {g:p3.CAND for g in OUTFIELD}
    pieces={(g,f):[] for g in wanted for f in wanted[g]};ablation=[];predictions=[]
    foldlist=p3.splits(data,kind);eval_rows=[]
    for fold,(a,b) in enumerate(foldlist,1):
        train,test=data.iloc[a],data.iloc[b];eval_rows.append(test)
        tag=f'{label}/{kind}/{fold}'
        model=p3.fit_model(train,4,age=1);log_fit(model,train,test,tag+'/global4',kind)
        predictions.append(pd.DataFrame(dict(row_id=test.original_clean_row_id,player_id=test.player_id,position=test.position_group,
              transfer_season=test.transfer_season,actual=test.log_transfer_fee,predicted=p3.predict(model,test),fold=fold)))
        # Other selected predictors are identical in both ablation fits. xG is
        # forced into the plus fit even if the original selection dropped it.
        shared=[x for x in model['selected'] if x!='xG_Per']
        minus=p3.fit_model(train,4,age=1,forced_candidates=shared)
        plus=p3.fit_model(train,4,age=1,forced_candidates=shared+['xG_Per'])
        log_fit(minus,train,test,tag+'/xg_without',kind);log_fit(plus,train,test,tag+'/xg_with',kind)
        ablation.append(pd.DataFrame(dict(row_id=test.original_clean_row_id,player_id=test.player_id,position=test.position_group,actual=test.log_transfer_fee,
              without=p3.predict(minus,test),with_xg=p3.predict(plus,test),fold=fold)))
        for group,features in wanted.items():
            sub=test[test.position_group.eq(group)]
            if len(sub)<2:continue
            base=sub.log_transfer_fee.to_numpy()-p3.predict(model,sub)
            for feature in features:
                status='NOT_SELECTED' if feature not in model['selected'] else 'CONSTANT_HELDOUT' if sub[feature].nunique(dropna=False)<=1 else 'SELECTED'
                PERM_LOG.append(dict(dataset=label,validation=kind,fold=fold,position=group,feature=feature,status=status,n=len(sub),baseline_rmse=float(np.sqrt(np.mean(base**2))),baseline_mae=float(np.mean(abs(base)))))
                if status=='NOT_SELECTED':continue
                r=rng(tag+group+feature);err=[]
                for repeat in range(REPEATS):
                    perm=sub.copy();perm[feature]=r.permutation(sub[feature].to_numpy())
                    err.append(sub.log_transfer_fee.to_numpy()-p3.predict(model,perm))
                pieces[(group,feature)].append(dict(base_error=base,perm_error=np.column_stack(err),ids=sub.player_id.to_numpy()))
    eval_data=pd.concat(eval_rows);summ=[]
    for (g,f),parts in pieces.items():
        sub=eval_data[eval_data.position_group.eq(g)]
        available=sum(int((data.iloc[b].position_group.eq(g)).sum()>=2) for a,b in foldlist)
        summ.append(dict(dataset=label,validation=kind,**permutation_summary(parts,label+kind,g,f,available,len(sub),sub.player_id.nunique())))
    abl=pd.concat(ablation,ignore_index=True);arows=[]
    for g in OUTFIELD:
        z=abl[abl.position.eq(g)]
        row=dict(dataset=label,position=g,validation=kind,folds=z.fold.nunique(),**paired(z.actual.to_numpy(),z.without.to_numpy(),z.with_xg.to_numpy(),z.player_id,label+kind+'xg'+g))
        row['classification']=classify_xg(row);arows.append(row)
    return pd.DataFrame(summ),pd.DataFrame(arows),pd.concat(predictions,ignore_index=True),abl

def descriptives(data):
    metrics=p3.CAND+['xAG_Per','npxG+xAG_Per','Clr_per90','TklInt_per90','Blocks_per90','Save_percent','GA90','PSxG']
    out=[]
    for group in GROUPS:
        d=data[data.position_group.eq(group)]
        candidates=metrics if group!='GK' else BLOCKS['GK']
        if group!='GK':candidates=[f for f in candidates if f not in ['Save_percent','GA90','PSxG']]
        for f in candidates:
            valid=d[[f,'log_transfer_fee','player_id']].replace([np.inf,-np.inf],np.nan).dropna();x=valid[f].to_numpy();y=valid.log_transfer_fee.to_numpy()
            cor=np.corrcoef(x,y)[0,1] if len(x)>2 and np.std(x)>0 else np.nan
            ci=[np.nan,np.nan]
            if np.isfinite(cor):
                b,_=bootstrap_sums(valid.player_id,np.column_stack([np.ones(len(x)),x,y,x*x,y*y,x*y]),'desc'+group+f)
                n,sx,sy,sxx,syy,sxy=b.T;den=np.sqrt(np.maximum((sxx-sx*sx/n)*(syy-sy*sy/n),0));r=np.divide(sxy-sx*sy/n,den,out=np.full_like(n,np.nan),where=den>0)
                ci=np.nanquantile(r,[.025,.975])
            s=valid[f]
            out.append(dict(position=group,feature=f,n=len(d),n_nonmissing=len(valid),median=s.median(),iqr=s.quantile(.75)-s.quantile(.25),p05=s.quantile(.05),p95=s.quantile(.95),correlation=cor,correlation_ci_low=ci[0],correlation_ci_high=ci[1],scope='Descriptive Pearson; player-cluster bootstrap; not predictive importance'))
    return pd.DataFrame(out)

def role_models(data):
    preds=[];folds=[]
    for fold,(a,b) in enumerate(p3.splits(data,'group'),1):
        globaltrain,globaltest=data.iloc[a],data.iloc[b]
        for group in GROUPS:
            tr=globaltrain[globaltrain.position_group.eq(group)];te=globaltest[globaltest.position_group.eq(group)]
            if len(te)<5 or tr.player_id.nunique()<20:raise ValueError('Inadequate role fold: '+group)
            base=custom_fit(tr,[]);block=custom_fit(tr,BLOCKS[group])
            assert len(tr)>block['X'].shape[1]+10
            log_fit(base,tr,te,f'role/{group}/{fold}/base','group',True);log_fit(block,tr,te,f'role/{group}/{fold}/block','group',True)
            p0,p1=custom_predict(base,te),custom_predict(block,te)
            m0,m1=p3.metrics(te.log_transfer_fee,p0),p3.metrics(te.log_transfer_fee,p1)
            folds.append(dict(position=group,fold=fold,train_n=len(tr),test_n=len(te),baseline_r2=m0['r2'],block_r2=m1['r2'],baseline_rmse=m0['rmse'],block_rmse=m1['rmse'],delta_rmse=m0['rmse']-m1['rmse'],baseline_mae=m0['mae'],block_mae=m1['mae']))
            preds.append(pd.DataFrame(dict(row_id=te.original_clean_row_id,player_id=te.player_id,position=group,fold=fold,actual=te.log_transfer_fee,baseline=p0,block=p1)))
    pred=pd.concat(preds,ignore_index=True);fd=pd.DataFrame(folds);out=[]
    for group,g in pred.groupby('position'):
        row=paired(g.actual.to_numpy(),g.baseline.to_numpy(),g.block.to_numpy(),g.player_id,'role'+group)
        row.update(position=group,folds=g.fold.nunique(),delta_r2=row['r2_with']-row['r2_without'],
          rmse_fold_sd_baseline=fd.loc[fd.position.eq(group),'baseline_rmse'].std(),rmse_fold_sd_block=fd.loc[fd.position.eq(group),'block_rmse'].std(),
          r2_fold_sd_baseline=fd.loc[fd.position.eq(group),'baseline_r2'].std(),r2_fold_sd_block=fd.loc[fd.position.eq(group),'block_r2'].std(),
          features=p3.js(BLOCKS[group]),validation='Global player-grouped assignments, role-only training; five matched held-out folds',exploratory_small_group=group in ['AM','GK'])
        out.append(row)
    save(fd,'position_block_fold_results.csv');save(pred,'position_block_oof_predictions.csv')
    return pd.DataFrame(out),pred

def interactions(data):
    results=[];predicts=[]
    for feature in INTERACTIONS:
        pred=[]
        for fold,(a,b) in enumerate(p3.splits(data,'group'),1):
            tr,te=data.iloc[a],data.iloc[b];base=custom_fit(tr,p3.CAND);aug=custom_fit(tr,p3.CAND,feature)
            log_fit(base,tr,te,f'interaction/{feature}/{fold}/base','group',True);log_fit(aug,tr,te,f'interaction/{feature}/{fold}/aug','group',True)
            pred.append(pd.DataFrame(dict(feature=feature,row_id=te.original_clean_row_id,player_id=te.player_id,fold=fold,actual=te.log_transfer_fee,baseline=custom_predict(base,te),interaction=custom_predict(aug,te))))
        z=pd.concat(pred,ignore_index=True);predicts.append(z)
        score=paired(z.actual.to_numpy(),z.baseline.to_numpy(),z.interaction.to_numpy(),z.player_id,'interaction'+feature)
        full=custom_fit(data,p3.CAND,feature);fit=full['result'].get_robustcov_results(cov_type='cluster',groups=data.player_id,use_correction=True,df_correction=True,use_t=True)
        start=len(full['prep'].kept_names);inds=list(range(start,len(full['names'])))
        if inds:
            mat=np.eye(len(full['names']))[inds];wald=fit.wald_test(mat,use_f=True,scalar=True)
            results.append(dict(feature=feature,record_type='OMNIBUS',term='Joint estimable interactions',raw_p=float(wald.pvalue),statistic=float(wald.statistic),estimable_terms=len(inds),**score))
        for group in GROUPS[1:]:
            term=feature+' x '+group+' vs FW'
            if group in full['omitted']:
                results.append(dict(feature=feature,record_type='COEFFICIENT',term=term,position=group,status='NOT_ESTIMABLE',raw_p=np.nan,**score));continue
            j=full['names'].index(term);ci=fit.conf_int()[j]
            results.append(dict(feature=feature,record_type='COEFFICIENT',term=term,position=group,status='ESTIMABLE',coefficient=float(fit.params[j]),standard_error=float(fit.bse[j]),ci_low=float(ci[0]),ci_high=float(ci[1]),raw_p=float(fit.pvalues[j]),**score))
    out=pd.DataFrame(results);out['adjusted_p']=np.nan
    for typ in ['OMNIBUS','COEFFICIENT']:
        mask=out.record_type.eq(typ)&out.raw_p.notna();out.loc[mask,'adjusted_p']=multipletests(out.loc[mask,'raw_p'],method='fdr_bh')[1]
    out['inference_scope']='Fixed all-ten-candidate specification, not selected Model4; player-cluster robust t/F; BH across six omnibus tests and all estimable interaction coefficients separately'
    save(pd.concat(predicts),'position_interaction_oof_predictions.csv')
    return out

def robustness(primary_imp,primary_ab,imp_sensitivity,ab_sensitivity,top):
    out=[]
    allimp=pd.concat([primary_imp]+imp_sensitivity);allab=pd.concat([primary_ab]+ab_sensitivity)
    for group,feature in top.items():
        z=allimp[allimp.validation.eq('group')&allimp.position.eq(group)&allimp.feature.eq(feature)].copy()
        if len(z)<3 or (z.folds_contributing<3).any() or (z.unique_players<25).any():status='INSUFFICIENT_EVIDENCE'
        elif (z.mean_delta_rmse>0).all() and z.loc[z.dataset.eq('MIN270'),'ci_low'].iloc[0]>0:status='ROBUST'
        elif (z.mean_delta_rmse<-.01).any():status='NOT_ROBUST'
        else:status='MIXED'
        for row in z.to_dict('records'):out.append(dict(finding='TOP_GROUPED_PERMUTATION',robustness=status,**row))
    for group in OUTFIELD:
        z=allab[allab.position.eq(group)&allab.validation.eq('group')].copy()
        if len(z)<3 or (z.unique_players<25).any():status='INSUFFICIENT_EVIDENCE'
        elif (z.classification.eq('CLEAR_INCREMENTAL_SIGNAL')).all() or (z.classification.eq('NO_DETECTABLE_INCREMENTAL_SIGNAL')).all():status='ROBUST'
        elif (z.delta_rmse>.01).any() and (z.delta_rmse<-.01).any():status='NOT_ROBUST'
        else:status='MIXED'
        for row in z.to_dict('records'):out.append(dict(finding='XG_ABLATION',feature='xG_Per',robustness=status,**row))
    return pd.DataFrame(out)

def main():
    if not (P/'PRE_POSITION_RESULTS_MANIFEST.csv').exists():shutil.copy2(ROOT/'RESULTS_MANIFEST.csv',P/'PRE_POSITION_RESULTS_MANIFEST.csv')
    dump(dict(primary='MIN270 unchanged',phase='POSITION_AND_RESIDUAL_EXPLORATION',mapping=MAPPING,blocks=BLOCKS,interaction_blocks=INTERACTIONS,
        bootstrap_repetitions=BOOT,permutations=REPEATS,practical_delta_rmse_screen=MIN_EFFECT,
        uncertainty='Conditional player-cluster bootstrap of held-out errors; no training-model refit; nominal intervals not simultaneous',
        xg_ablation='Global Model4 selected on training; other selected predictors fixed; force xG in plus and remove only xG in minus',
        role_validation='Use same global GroupKFold assignments for local role training and matched global comparison',
        interaction_inference='Fixed full candidate main-effects design, one interaction family at a time; cluster-robust inference, no stepwise inference',
        fdr='BH across six omnibus interaction tests; separately all interaction coefficients. Residual BH within family across methods per threshold.',
        small_group_rule='Exploratory AM/GK; minimum 25 players,40 rows,3 folds for xG/robustness class',
        clear_xg='Delta RMSE>=0.01 and conditional CI lower>0; no-detectable if CI upper<0.01; else mixed; 0.01 is a transparent screen, not universal football importance',
        primary_residual_candidate='abs(mean)>=0.05 log units and BH q<0.05 in at least one MIN270 source; no economic premium claim',
        residual_robustness='Same direction across six cells, >=4 point magnitudes>=0.05, >=20 players each, primary candidate evidence; no significance required in every sensitivity',
        no_freeze=True,no_abstract=True),'POSITION_ANALYSIS_PLAN.json')
    d=prepare('primary_min270_dataset.csv');assert len(d)==1059
    counts=[]
    for group in GROUPS:
        g=d[d.position_group.eq(group)];counts.append(dict(position=group,n=len(g),unique_players=g.player_id.nunique(),season_counts=p3.counts(g.transfer_season),destination_league_counts=p3.counts(g.to_league),median_age=g.age_at_transfer.median(),median_transfer_fee=g.transfer_fee_raw.median(),median_log_transfer_fee=g.log_transfer_fee.median(),median_90s=g.Mins_Per_90_Playing.median(),iqr_90s=g.Mins_Per_90_Playing.quantile(.75)-g.Mins_Per_90_Playing.quantile(.25)))
    save(counts,'position_group_counts.csv')
    change=d.groupby('player_id').position_group.nunique();changing=d[d.player_id.isin(change[change>1].index)]
    save(changing[['original_clean_row_id','player_id','player_name','transfer_season','sub_position','position_group']],'players_with_position_group_changes.csv')
    save(descriptives(d),'position_metric_descriptives.csv')
    # Inspect upstream keeper excess-goals field, without injecting new metrics.
    raw=pd.read_csv(ROOT/'01_audit/reproduction/fresh_cache_attempt/downloaded_tables/keepers_adv_data.csv')
    lineage=pd.read_csv(ROOT/'02_data/intermediate/original_clean_lineage.csv',low_memory=False).set_index('original_clean_row_id')
    keeper=[]
    for row in d[d.position_group.eq('GK')].itertuples():
        l=lineage.loc[row.original_clean_row_id];match=raw[raw.Player.eq(row.player_name)&raw.Squad.eq(l.Squad)&raw.Season_End_Year.eq(l.Season_End_Year)]
        if len(match)==1:
            x=match.iloc[0];net=x['PSxG+_per__minus__Expected'];derived=x.PSxG_Expected-(x.GA_Goals-x.OG_Goals)
            keeper.append(dict(row_id=row.original_clean_row_id,PSxG_original=row.PSxG,PSxG_cached=x.PSxG_Expected,PSxG_minus_goals_excluding_own_goals=derived,provider_PSxG_net=net,formula_rounding_gap=net-derived,
                decision='PSxG is total post-shot expected goals faced, not net saves. Net field available upstream but not added to fixed four-variable primary GK block.'))
    save(keeper,'goalkeeper_feature_definition_audit.csv')
    primary_imp=[];primary_ab=[];global_predictions={}
    for kind in ['group','temporal']:
        imp,ab,pred,ap=global_position_eval(d,'MIN270',kind)
        primary_imp.append(imp);primary_ab.append(ab);global_predictions[kind]=pred
        save(imp,'position_feature_importance_'+('grouped' if kind=='group' else 'temporal')+'.csv')
        save(pred,f'global_model4_{kind}_oof_recreated.csv');save(ap,f'xg_ablation_{kind}_predictions.csv')
        # Verify exact reproduction of previously saved genuine OOS predictions.
        previous=pd.read_csv(ROOT/'04_validation'/f'MIN270_{"player_group" if kind=="group" else "temporal"}_oof_predictions.csv')
        merged=pred.merge(previous[previous.model.eq('Model 4')],on='row_id',validate='one_to_one')
        assert len(merged)==len(pred) and np.allclose(merged.predicted,merged.predicted_log_fee,atol=1e-8)
        print('Completed primary importance/ablation',kind,flush=True)
    imp=pd.concat(primary_imp);ab=pd.concat(primary_ab)
    save(imp,'position_feature_importance_summary.csv');save(ab,'xg_position_ablation.csv')
    blocks,rolepred=role_models(d);save(blocks,'position_block_model_results.csv');save(blocks[blocks.position.eq('GK')],'goalkeeper_analysis.csv')
    matched=rolepred.merge(global_predictions['group'][['row_id','predicted']],on='row_id',validate='one_to_one')
    matchedrows=[]
    for group,z in [('ALL',matched)]+list(matched.groupby('position')):
        matchedrows.append(dict(position=group,**paired(z.actual.to_numpy(),z.predicted.to_numpy(),z.block.to_numpy(),z.player_id,'local_vs_global'+group)))
    save(matchedrows,'role_block_vs_global_model4.csv')
    save(interactions(d),'position_interaction_results.csv')
    print('Completed role models and interactions',flush=True)
    top={}
    for group in ['FW','CM/DM','SB','CB']:
        candidates=imp[imp.validation.eq('group')&imp.position.eq(group)&imp.folds_contributing.ge(3)&imp.unique_players.ge(25)].sort_values(['ci_low','mean_delta_rmse'],ascending=False)
        if len(candidates):top[group]=candidates.iloc[0].feature
    dump(top,'selected_robustness_targets.json')
    si=[];sa=[]
    for label,file in [('MIN450','sensitivity_min450_dataset.csv'),('MIN900','sensitivity_min900_dataset.csv')]:
        dd=prepare(file);wanted={g:[f] for g,f in top.items()}
        ii,aa,_,_=global_position_eval(dd,label,'group',wanted)
        si.append(ii);sa.append(aa)
    save(pd.concat(si),'threshold_top_signal_importance.csv');save(pd.concat(sa),'threshold_xg_ablation.csv')
    save(robustness(imp,ab,si,sa,top),'position_robustness_summary.csv')
    save(PERM_LOG,'position_permutation_fold_status.csv')
    dump(FIT_LOG,'position_training_audit.json')
    dump(dict(status='COMPLETE',no_results_frozen=True,no_abstract=True,primary_n=len(d),models=len(FIT_LOG)),'POSITION_ANALYSIS_COMPLETE.json')
    print('POSITION ANALYSES COMPLETE; residual stage may now start.',flush=True)

if __name__=='__main__':main()
