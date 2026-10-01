"""Conditional OOS calibration diagnostics, only after position work completes."""
from pathlib import Path
import json, importlib.util
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[2];P=ROOT/'05_position_analysis';E=ROOT/'06_residual_analysis'
sp=importlib.util.spec_from_file_location('pos',P/'tools/analyze_positions.py');pos=importlib.util.module_from_spec(sp);sp.loader.exec_module(pos)

def mean_summary(g,label):
    y=g.residual_log.to_numpy();mu=float(y.mean());n=len(y)
    b,k=pos.bootstrap_sums(g.player_id,np.column_stack([np.ones(n),y]),label)
    boot=b[:,1]/b[:,0];scores=g.assign(centered=y-mu).groupby('player_id').centered.sum().to_numpy()
    se=float(np.sqrt(k/(k-1)*np.sum(scores**2)/n**2)) if k>1 else np.nan
    p=float(2*stats.t.sf(abs(mu/se),k-1)) if se>0 else np.nan
    return dict(n=n,unique_players=k,mean_residual=mu,median_residual=float(np.median(y)),mae=float(np.mean(abs(y))),rmse=float(np.sqrt(np.mean(y*y))),
        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),cluster_standard_error=se,raw_p=p,
        test='Intercept-only mean=0, player-cluster robust t; conditional on trained OOF models')

def main():
    assert json.loads((P/'POSITION_ANALYSIS_COMPLETE.json').read_text())['status']=='COMPLETE'
    pos.dump(dict(scope='Only saved Model4 OOF/OOS errors, never fitted training residuals',categorical_families=['destination_league','position_group','transfer_season'],
        age='Linear residual-age slope and prespecified 3-year bins, descriptive; no U21/O30 primary tests',
        tests='Cluster-robust mean=0 t tests; BH per family across validation sources within threshold',
        uncertainty='1000 player-cluster bootstrap draws; models not refitted; not full-procedure uncertainty',
        candidate='abs(mean)>=0.05 and q<0.05 in a MIN270 source, >=20 players',
        robust='All six cells present, >=20 players each, same sign, >=4 magnitudes>=0.05 and primary candidate; does not require all p-values significant',
        no_freeze=True),'RESIDUAL_ANALYSIS_PLAN.json',E)
    allrows=[];summaries=[];slopes=[];bins=[]
    for dataset,filename in [('MIN270','primary_min270_dataset.csv'),('MIN450','sensitivity_min450_dataset.csv'),('MIN900','sensitivity_min900_dataset.csv')]:
        data=pos.prepare(filename)
        for method,prefix in [('grouped','player_group'),('temporal','temporal')]:
            p=pd.read_csv(ROOT/'04_validation'/f'{dataset}_{prefix}_oof_predictions.csv');p=p[p.model.eq('Model 4')].copy()
            d=p.merge(data[['original_clean_row_id','player_name','to_league','position_group','age_at_transfer','Mins_Per_90_Playing']],left_on='row_id',right_on='original_clean_row_id',validate='one_to_one')
            assert len(d)==len(p)
            d['dataset']=dataset;d['validation_source']=method;d['residual_log']=d.actual_log_fee-d.predicted_log_fee
            d['absolute_error']=abs(d.residual_log);d['squared_error']=d.residual_log**2;d['actual_to_predicted_ratio']=np.exp(d.residual_log)
            allrows.append(d)
            for family,column in [('destination_league','to_league'),('position_group','position_group'),('transfer_season','transfer_season')]:
                for category,g in d.groupby(column):
                    summaries.append(dict(dataset=dataset,validation_source=method,family=family,category=category,**mean_summary(g,dataset+method+family+str(category))))
            x=d.age_at_transfer.to_numpy(float)-25;y=d.residual_log.to_numpy()
            fit=sm.OLS(y,np.column_stack([np.ones(len(x)),x])).fit().get_robustcov_results(cov_type='cluster',groups=d.player_id,use_correction=True,df_correction=True,use_t=True)
            b,nc=pos.bootstrap_sums(d.player_id,np.column_stack([np.ones(len(x)),x,y,x*x,x*y]),dataset+method+'age')
            n,sx,sy,sxx,sxy=b.T;sl=(sxy-sx*sy/n)/(sxx-sx*sx/n)
            slopes.append(dict(dataset=dataset,validation_source=method,n=len(d),unique_players=nc,slope_log_residual_per_year=float(fit.params[1]),standard_error=float(fit.bse[1]),raw_p=float(fit.pvalues[1]),ci_low=float(np.quantile(sl,.025)),ci_high=float(np.quantile(sl,.975)),scope='Descriptive linear age trend, not nonlinear search or an age premium'))
            d['age_bin_start']=(np.floor(d.age_at_transfer/3)*3).astype(int)
            for start,g in d.groupby('age_bin_start'):
                bins.append(dict(dataset=dataset,validation_source=method,age_bin=f'{start}-{start+2}',age_mid=start+1,n=len(g),mean_residual=g.residual_log.mean(),median_residual=g.residual_log.median(),scope='Descriptive 3-year bins; no bin-wise hypothesis tests'))
    allrows=pd.concat(allrows,ignore_index=True);summary=pd.DataFrame(summaries);slope=pd.DataFrame(slopes)
    summary['adjusted_p']=np.nan;slope['adjusted_p']=np.nan
    for (_,family),idx in summary.groupby(['dataset','family']).groups.items():
        good=summary.loc[idx].raw_p.dropna().index;summary.loc[good,'adjusted_p']=multipletests(summary.loc[good,'raw_p'],method='fdr_bh')[1]
    for dataset,idx in slope.groupby('dataset').groups.items():slope.loc[idx,'adjusted_p']=multipletests(slope.loc[idx,'raw_p'],method='fdr_bh')[1]
    pos.save(allrows[allrows.dataset.eq('MIN270')],'MIN270_oos_residuals.csv',E)
    pos.save(allrows,'threshold_oos_residuals.csv',E)
    pos.save(summary[summary.dataset.eq('MIN270')],'MIN270_residual_group_summary.csv',E)
    pos.save(summary,'threshold_residual_group_summary.csv',E)
    pos.save(slope,'residual_continuous_age_summary.csv',E);pos.save(bins,'residual_age_bins.csv',E)
    matrix=[]
    for (family,category),g in summary.groupby(['family','category']):
        primary=g[g.dataset.eq('MIN270')]
        candidate=bool(((primary.mean_residual.abs()>=.05)&primary.adjusted_p.lt(.05)&primary.unique_players.ge(20)).any())
        sign_consistent=bool((g.mean_residual>0).all() or (g.mean_residual<0).all())
        magnitude=int(g.mean_residual.abs().ge(.05).sum())
        if len(g)<6 or (g.unique_players<20).any():status='INSUFFICIENT_EVIDENCE'
        elif candidate and sign_consistent and magnitude>=4:status='ROBUST'
        elif candidate and not sign_consistent and (g.mean_residual>.05).any() and (g.mean_residual<-.05).any():status='NOT_ROBUST'
        elif candidate:status='MIXED'
        else:status='INSUFFICIENT_EVIDENCE'
        for row in g.to_dict('records'):matrix.append(dict(robustness=status,primary_candidate=candidate,same_direction_all_cells=sign_consistent,meaningful_magnitude_cells=magnitude,estimate=row['mean_residual'],estimate_units='log fee mean residual',**row))
    # Age remains continuous. A five-year slope contrast puts its magnitude on
    # the same 0.05-log-unit screen without introducing youth/veteran hypotheses.
    primary=slope[slope.dataset.eq('MIN270')]
    candidate=bool(((primary.slope_log_residual_per_year.abs()*5>=.05)&primary.adjusted_p.lt(.05)).any())
    same=bool((slope.slope_log_residual_per_year>0).all() or (slope.slope_log_residual_per_year<0).all())
    meaningful=int((slope.slope_log_residual_per_year.abs()*5>=.05).sum())
    status='ROBUST' if candidate and same and meaningful>=4 else 'MIXED' if candidate else 'INSUFFICIENT_EVIDENCE'
    for row in slope.to_dict('records'):
        matrix.append(dict(family='continuous_age',category='linear_age_slope',robustness=status,primary_candidate=candidate,same_direction_all_cells=same,meaningful_magnitude_cells=meaningful,estimate=row['slope_log_residual_per_year'],estimate_units='log residual per year',**row))
    pos.save(matrix,'residual_robustness_matrix.csv',E)
    pos.dump(dict(status='COMPLETE',source_rows_by_method=allrows.groupby(['dataset','validation_source']).size().reset_index(name='n').to_dict('records'),
                  training_residuals_used=False,results_frozen=False,abstract_written=False),'RESIDUAL_ANALYSIS_COMPLETE.json',E)
    print('OOS residual diagnostics complete; no results frozen.',flush=True)

if __name__=='__main__':main()
