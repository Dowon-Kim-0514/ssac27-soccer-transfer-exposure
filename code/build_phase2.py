"""Timing/entity/provenance audits and design diagnostics. No fee models are fitted."""
from pathlib import Path
import csv, datetime, hashlib, json, re, unicodedata, shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
I=ROOT/'02_data/intermediate'
F=ROOT/'02_data/final'
A=ROOT/'01_audit'
S=ROOT/'00_original_snapshot/Soccer_Transfer_Research_2026-09-22/original_sources/coding'
RAW=next(S.rglob('transfers.csv')).parent
BIG={'GB1','ES1','L1','IT1','FR1'}
COMP={'Premier League':'GB1','La Liga':'ES1','Bundesliga':'L1','Serie A':'IT1','Ligue 1':'FR1'}

def save(df,name,folder=I):
    df.to_csv(folder/name,index=False)

def norm(x):
    if pd.isna(x): return ''
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower())

def dt(x):
    return pd.to_datetime(x,errors='coerce',utc=True,format='mixed').tz_convert(None) if not isinstance(x,pd.Series) else pd.to_datetime(x,errors='coerce',utc=True,format='mixed').dt.tz_convert(None)

def js(x): return json.dumps(x,ensure_ascii=False,sort_keys=True)
def counts(s): return {str(k):int(v) for k,v in s.value_counts(dropna=False).sort_index().items()}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    I.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True)
    prior_manifest=I/'PHASE1_RESULTS_MANIFEST.csv'
    if not prior_manifest.exists():shutil.copy2(ROOT/'RESULTS_MANIFEST.csv',prior_manifest)
    clean=pd.read_csv(S/'data/clean_data.csv')
    lin=pd.read_csv(I/'original_clean_lineage.csv',low_memory=False)
    tr=pd.read_csv(I/'original_transfer_dedup_lineage.csv')
    fb=pd.read_csv(I/'original_fbref_dedup_lineage.csv',low_memory=False)
    fuzzy=pd.read_csv(I/'original_fuzzy_replacements.csv')
    players=pd.read_csv(RAW/'players.csv',low_memory=False)
    clubs=pd.read_csv(RAW/'clubs.csv')
    games=pd.read_csv(RAW/'games.csv',low_memory=False)
    competitions=pd.read_csv(RAW/'competitions.csv')
    assert len(clean)==len(lin)==1622 and len(fuzzy)==212
    assert clean.player_name.equals(lin.player_name)
    assert np.array_equal(clean.transfer_fee_raw,lin.transfer_fee)
    assert np.array_equal(clean.Season_End_Year,lin.Season_End_Year)
    eligible=clean.log_transfer_fee.notna()&clean.from_league.notna()
    assert eligible.sum()==1407
    key=['player_id','transfer_date','from_club_id','to_club_id','transfer_fee']
    original_names=tr[key+['player_name']].rename(columns={'player_name':'transfer_player_name'})
    lin=lin.merge(original_names,on=key,how='left',validate='many_to_one',sort=False)
    assert lin.transfer_player_name.notna().all()
    assert (lin.original_clean_row_id.to_numpy()==np.arange(1,1623)).all()
    lin['original_eligible']=eligible.to_numpy()
    lin['transfer_dt']=dt(lin.transfer_date)
    lin['transfer_year']=lin.transfer_season.str[:2].astype(int)+2000
    lin['performance_year']=lin.Season_End_Year.astype(int)-1
    lin['performance_league']=lin.Comp.map(COMP)
    games['date_dt']=dt(games.date)
    league_games=games[games.competition_type.eq('domestic_league')].copy()
    byseason={(c,int(y)):g.copy() for (c,y),g in league_games.groupby(['competition_id','season'])}
    bounds=[]
    for (c,y),g in byseason.items():
        if c not in BIG or not 2017<=y<=2023:continue
        teams=set(g.home_club_id)|set(g.away_club_id)
        pairs=set(zip(g.home_club_id,g.away_club_id))
        complete=len(teams)>=2 and len(pairs)==len(teams)*(len(teams)-1) and g[['home_club_goals','away_club_goals']].notna().all().all()
        nxt=byseason.get((c,y+1))
        next_start=nxt.date_dt.min() if nxt is not None else pd.NaT
        # A non-complete fixture ledger cannot establish a cancellation/end date.
        # Its next observed season start supplies a conservative availability bound.
        cutoff=g.date_dt.max()+pd.Timedelta(days=1) if complete else next_start
        bounds.append(dict(competition_id=c,performance_start_year=y,fixture_n=len(g),club_n=len(teams),
            unique_directed_pairs=len(pairs),full_home_away_schedule_observed=bool(complete),
            first_fixture=g.date_dt.min(),last_fixture=g.date_dt.max(),next_season_first_fixture=next_start,
            strict_available_from=cutoff,bound_basis='COMPLETE_FIXTURE_LEDGER_PLUS_ONE_DAY' if complete else 'CONSERVATIVE_NEXT_OBSERVED_SEASON_START'))
    bounds=pd.DataFrame(bounds);save(bounds,'performance_season_bounds.csv')
    bmap={(r.competition_id,int(r.performance_start_year)):r for r in bounds.itertuples()}

    # Exact normalized club aliases only; no guessed translations or fuzzy club joins.
    aliases={}
    def add_alias(cid,name):
        if pd.notna(cid) and norm(name):aliases.setdefault(norm(name),set()).add(int(cid))
    for r in clubs.itertuples():add_alias(r.club_id,r.name)
    for r in tr.itertuples():add_alias(r.from_club_id,r.from_club_name);add_alias(r.to_club_id,r.to_club_name)
    for r in games.itertuples():add_alias(r.home_club_id,r.home_club_name);add_alias(r.away_club_id,r.away_club_name)
    club_map={}
    alias_rows=[]
    for (squad,comp),g in fb.groupby(['Squad','Comp']):
        ids=aliases.get(norm(squad),set())
        cid=next(iter(ids)) if len(ids)==1 else None
        club_map[(squad,comp)]=cid
        alias_rows.append(dict(fbref_squad=squad,fbref_comp=comp,matched_club_id=cid,candidate_ids=js(sorted(ids)),basis='UNIQUE_EXACT_NORMALIZED_ARCHIVED_ALIAS' if cid else 'UNRESOLVED_OR_NONUNIQUE'))
    save(pd.DataFrame(alias_rows),'fbref_club_alias_evidence.csv')

    # Appearance evidence is used for identity and possible post-transfer events,
    # never to recompute or truncate the original season performance statistics.
    app=pd.read_csv(RAW/'appearances.csv',usecols=['game_id','player_id','player_club_id','date'])
    app=app[app.player_id.isin(set(tr.player_id))].merge(games[['game_id','season','competition_id']],on='game_id',how='inner',validate='many_to_one')
    app['date_dt']=dt(app.date)
    appgrp=app.groupby(['player_id','season','player_club_id','competition_id']).agg(n=('game_id','size'),first=('date_dt','min'),last=('date_dt','max'))
    app_by_player_season=app.groupby(['player_id','season']).player_club_id.apply(lambda x:sorted(set(int(v) for v in x)))

    timing=[]
    for r in lin.itertuples():
        b=bmap.get((r.performance_league,r.performance_year))
        cutoff=b.strict_available_from if b is not None else pd.NaT
        defend=pd.notna(r.transfer_dt) and pd.notna(cutoff) and r.transfer_dt>=cutoff
        prior=r.performance_year<r.transfer_year;same=r.performance_year==r.transfer_year
        cid=club_map.get((r.Squad,r.Comp))
        ak=(r.player_id,r.performance_year,cid,r.performance_league)
        ae=appgrp.loc[ak] if cid is not None and ak in appgrp.index else None
        post=pd.notna(r.transfer_dt) and ae is not None and ae['last']>=r.transfer_dt
        if defend:
            status='DEFENSIBLE_COMPLETED_PERFORMANCE_PERIOD'
            reason=f'Transfer on/after {pd.Timestamp(cutoff).date()}; {b.bound_basis}. Original season aggregate retained.'
        elif b is None or pd.isna(cutoff) or pd.isna(r.transfer_dt):
            status='CANNOT_ESTABLISH_PRETRANSFER_TIMING';reason='Missing date, mapped league, or defensible season boundary.'
        else:
            status='PERFORMANCE_PERIOD_NOT_DEFENSIBLY_COMPLETE'
            reason=f'Transfer before conservative availability boundary {pd.Timestamp(cutoff).date()}; same/prior-season label alone is insufficient.'
        timing.append(dict(original_clean_row_id=r.original_clean_row_id,original_eligible=bool(r.original_eligible),player_name=r.player_name,
            transfer_player_name=r.transfer_player_name,player_id=int(r.player_id),transfer_date=r.transfer_date,
            transfer_season=clean.transfer_season.iloc[r.original_clean_row_id-1],performance_season=f'{r.performance_year}-{str(r.performance_year+1)[-2:]}',
            performance_league=r.performance_league,fbref_squad=r.Squad,prior_season=bool(prior),same_season=bool(same),
            proxy_flag=int(clean.season_proxy_flag.iloc[r.original_clean_row_id-1]),
            observed_season_last_fixture=b.last_fixture if b is not None else pd.NaT,strict_available_from=cutoff,
            boundary_basis=b.bound_basis if b is not None else 'UNAVAILABLE',possible_post_transfer_information=not bool(defend),
            corroborating_post_transfer_appearance=bool(post),selected_squad_last_archived_appearance=ae['last'] if ae is not None else pd.NaT,
            timing_classification=status,strict_timing_keep=bool(defend),reason=reason,
            publication_timestamp_verified=False))
    timing=pd.DataFrame(timing);save(timing,'proxy_season_audit.csv')

    # Trace contract fields to the archived players snapshot, not guessed contracts.
    pmap=players.set_index('player_id')
    contracts=[]
    for r in lin.itertuples():
        p=pmap.loc[r.player_id];expiry=dt(p.contract_expiration_date)
        base=pd.Timestamp(int(r.Season_End_Year),7,1)
        original_years=(expiry-base).days/365.25 if pd.notna(expiry) else np.nan
        transfer_years=(expiry-r.transfer_dt).total_seconds()/86400/365.25 if pd.notna(expiry) and pd.notna(r.transfer_dt) else np.nan
        actual=clean.contract_years_remaining.iloc[r.original_clean_row_id-1]
        assert (pd.isna(actual) and pd.isna(original_years)) or np.isclose(actual,original_years,rtol=0,atol=1e-10)
        reasons=[]
        if pd.isna(expiry):category='CANNOT_VERIFY';reasons.append('Snapshot expiry missing.')
        else:
            if pd.notna(p.current_club_id) and int(p.current_club_id)!=int(r.from_club_id):reasons.append('Player snapshot current club differs from historical selling club.')
            if transfer_years<0:reasons.append('Snapshot expiry precedes transfer date.')
            if transfer_years>6:reasons.append('More than six years from transfer to snapshot expiry; audit screening flag, not a legal limit.')
            category='QUESTIONABLE' if reasons else 'CANNOT_VERIFY'
        reasons.append('No dated historical selling-club contract record or field-validity timestamp exists in these project inputs.')
        contracts.append(dict(original_clean_row_id=r.original_clean_row_id,original_eligible=bool(r.original_eligible),player=r.player_name,
            player_id=int(r.player_id),transfer_date=r.transfer_date,transfer_season=clean.transfer_season.iloc[r.original_clean_row_id-1],
            contract_expiration_value=p.contract_expiration_date,source_field='players.csv:contract_expiration_date',
            original_calculation_reference_date=base.date(),calculated_years_remaining=original_years,
            snapshot_expiry_years_from_actual_transfer=transfer_years,snapshot_current_club_id=p.current_club_id,
            historical_selling_club_id=r.from_club_id,player_snapshot_last_season=p.last_season,
            plausibility_flag=category,historical_timing_established=False,reason=' '.join(reasons)))
    contracts=pd.DataFrame(contracts);save(contracts,'contract_variable_audit.csv')

    # Every accepted replacement receives a traceable rule-based confidence label.
    required=lambda s:'17/18' if s=='17/18' else f'{int(s[:2])-1:02d}/{int(s[-2:])-1:02d}'
    pos_expected={'Attack':'FW','Midfield':'MF','Defender':'DF','Goalkeeper':'GK'}
    fmap={};identity=[]
    for fi,r in enumerate(fuzzy.itertuples(),1):
        trans=tr[tr.player_name.eq(r.player_name)&tr.transfer_season.eq(r.transfer_season)]
        assert len(trans)==1
        t=trans.iloc[0];season=required(r.transfer_season)
        cand=fb[fb.Player.eq(r.best_match)&fb.transfer_season.eq(season)]
        allcand=fb[fb.Player.eq(r.best_match)]
        c=cand.iloc[0] if len(cand)==1 else None
        birth_year=pd.Timestamp(t.date_of_birth).year if pd.notna(t.date_of_birth) else None
        fb_birth=float(c.Born) if c is not None and pd.notna(c.Born) else None
        bymatch=birth_year==fb_birth if birth_year is not None and fb_birth is not None else None
        expected=pos_expected.get(t.position)
        posmatch=expected in str(c.Pos).split(',') if c is not None and expected else None
        cid=club_map.get((c.Squad,c.Comp)) if c is not None else None
        seasonyear=int(season[:2])+2000
        ak=(t.player_id,seasonyear,cid,COMP.get(c.Comp) if c is not None else None)
        clubapp=cid is not None and ak in appgrp.index
        altclubs=app_by_player_season.get((t.player_id,seasonyear),[])
        reasons=[]
        if c is None:
            confidence='CANNOT_VERIFY';reasons.append('No unique FBref row in the required performance season; candidate seasons are recorded, not substituted.')
        elif fb_birth is not None and birth_year is not None and abs(fb_birth-birth_year)>=2:
            confidence='LIKELY_INCORRECT';reasons.append('Birth year differs by at least two years.')
        elif bymatch is False or posmatch is False:
            confidence='AMBIGUOUS';reasons.append('Birth year differs by one year or broad position is incompatible; role/metadata changes possible.')
        elif bymatch and posmatch and clubapp:
            confidence='HIGH_CONFIDENCE';reasons.append('Birth year, broad position, exact archived squad alias and same-player-ID appearances in required season agree.')
        elif bymatch and posmatch and (cid is None or not altclubs):
            confidence='LIKELY';reasons.append('Birth year and broad position agree; club crosswalk or appearance coverage insufficient for high confidence.')
        elif bymatch and posmatch:
            confidence='AMBIGUOUS';reasons.append('Birth year and position agree, but archived player appearances do not corroborate the resolved FBref squad.')
        else:
            confidence='CANNOT_VERIFY';reasons.append('Insufficient independent birth-year/position evidence.')
        impact=lin[lin.transfer_player_name.eq(r.player_name)&lin.transfer_season.eq(r.transfer_season)]
        row=dict(fuzzy_match_id=f'F{fi:03d}',transfer_player_name=r.player_name,matched_fbref_name=r.best_match,
            jaro_winkler_family_distance=r.match_score,transfer_season=r.transfer_season,required_performance_season=season,
            candidate_fbref_season=c.transfer_season if c is not None else '',all_candidate_seasons='|'.join(sorted(allcand.transfer_season.unique())),
            player_id=int(t.player_id),player_date_of_birth=t.date_of_birth,tm_nationality=t.country_of_citizenship,
            fbref_nationality=c.Nation if c is not None else '',nationality_verification='NOT_SCORED: different coding systems; no validated crosswalk',
            fbref_birth_year=fb_birth,birth_year_match=bymatch,fbref_exact_birth_date_available=False,
            origin_club=t.from_club_name,destination_club=t.to_club_name,fbref_squad=c.Squad if c is not None else '',
            fbref_club_id=cid,tm_position=t.position,fbref_position=c.Pos if c is not None else '',position_compatible=posmatch,
            same_id_season_squad_appearance=bool(clubapp),archived_appearance_club_ids=js(altclubs),
            fbref_player_url=c.Url if c is not None else '',manual_exception_status='ACCEPTED_NOT_ONE_OF_TWO_EXPLICITLY_REJECTED_PAIRS',
            identity_confidence=confidence,audit_reason=' '.join(reasons),clean_rows_affected=len(impact),
            original_model_rows_affected=int(impact.original_eligible.sum()),
            clean_row_ids=js(impact.original_clean_row_id.astype(int).tolist()))
        identity.append(row);fmap[(r.player_name,r.transfer_season)]=row
    identity=pd.DataFrame(identity);save(identity,'fuzzy_match_identity_audit.csv')
    save(identity[identity.identity_confidence.isin(['AMBIGUOUS','LIKELY_INCORRECT','CANNOT_VERIFY'])],'fuzzy_matches_requiring_review.csv')

    # Historical league membership is evidenced by archived fixtures, not memory.
    membership={};clubmatches={}
    for r in league_games.itertuples():
        for cid in [r.home_club_id,r.away_club_id]:
            membership.setdefault((int(cid),int(r.season)),set()).add(r.competition_id)
            clubmatches.setdefault(int(cid),[]).append((r.date_dt,r.competition_id,int(r.season)))
    clubmatches={k:sorted(v) for k,v in clubmatches.items()}
    cm=clubs.set_index('club_id');leagueaudit=[]
    for r in lin.itertuples():
        yy=r.transfer_year;entry=dict(original_clean_row_id=r.original_clean_row_id,original_eligible=bool(r.original_eligible),
            player_name=r.player_name,player_id=int(r.player_id),transfer_date=r.transfer_date,transfer_season=clean.transfer_season.iloc[r.original_clean_row_id-1])
        concerns=[];transition=False
        for side in ['from','to']:
            cid=int(getattr(r,side+'_club_id'));label=getattr(r,side+'_league')
            this=membership.get((cid,yy),set());prev=membership.get((cid,yy-1),set());nxt=membership.get((cid,yy+1),set())
            hist=[x for x in clubmatches.get(cid,[]) if pd.notna(r.transfer_dt) and x[0]<r.transfer_dt]
            last=hist[-1] if hist else (pd.NaT,'',None)
            if not this:status='NO_TRANSFER_SEASON_FIRST_TIER_FIXTURE_EVIDENCE'
            elif len(this)>1:status='MULTIPLE_COMPETITIONS_REQUIRE_REVIEW'
            elif pd.isna(label) or label not in this:status='SNAPSHOT_DIFFERS_FROM_OBSERVED_SEASON'
            else:status='SNAPSHOT_AGREES_WITH_OBSERVED_SEASON'
            if status!='SNAPSHOT_AGREES_WITH_OBSERVED_SEASON':concerns.append(side+':'+status)
            changed=prev!=this or this!=nxt
            transition=transition or changed
            entry.update({side+'_club_id':cid,side+'_league_snapshot':label,side+'_league_observed_transfer_season':'|'.join(sorted(this)),
                side+'_league_observed_previous_season':'|'.join(sorted(prev)),side+'_league_observed_next_season':'|'.join(sorted(nxt)),
                side+'_last_observed_pretransfer_league':last[1],side+'_last_observed_pretransfer_game':last[0],
                side+'_club_snapshot_last_season':cm.loc[cid,'last_season'] if cid in cm.index else np.nan,
                side+'_classification':status,side+'_possible_promotion_relegation_or_coverage_change':changed})
        toset=membership.get((int(r.to_club_id),yy),set())
        entry.update(big5_destination_observed_in_transfer_season=bool(toset&BIG),
            historical_big5_destination_status='OBSERVED_BIG5' if toset&BIG else ('OBSERVED_OTHER_FIRST_TIER' if toset else 'UNCONFIRMED_FROM_AVAILABLE_FIXTURES'),
            league_timing_concern=bool(concerns),possible_promotion_relegation_timing=bool(transition),
            audit_reason='; '.join(concerns) if concerns else 'Snapshot agrees with observed season-level fixtures; exact transfer-day league status/publication not independently established.',
            source='clubs.csv snapshot vs games.csv domestic_league club-season appearances',historical_labels_replaced=False)
        leagueaudit.append(entry)
    leagueaudit=pd.DataFrame(leagueaudit);save(leagueaudit,'club_league_timing_audit.csv')

    # Original columns remain unchanged; provenance/decision columns are added.
    base=clean.copy();base.insert(0,'original_clean_row_id',lin.original_clean_row_id)
    base['player_id']=lin.player_id.astype(int);base['transfer_date']=lin.transfer_date
    base['transfer_player_name']=lin.transfer_player_name
    base['fbref_player_url']=lin.Url;base['performance_season']=timing.performance_season
    base['strict_timing_keep']=timing.strict_timing_keep;base['timing_classification']=timing.timing_classification
    base['contract_audit_class']=contracts.plausibility_flag
    base['identity_confidence']=[fmap.get((r.transfer_player_name,r.transfer_season),{}).get('identity_confidence','NOT_FUZZY_NOT_INDEPENDENTLY_AUDITED') for r in lin.itertuples()]
    base['is_original_fuzzy_match']=base.identity_confidence.ne('NOT_FUZZY_NOT_INDEPENDENTLY_AUDITED')
    base['league_timing_concern']=leagueaudit.league_timing_concern
    base['possible_promotion_relegation_timing']=leagueaudit.possible_promotion_relegation_timing
    base['historical_big5_destination_status']=leagueaudit.historical_big5_destination_status
    original=base.loc[eligible].copy()
    strict=original[original.strict_timing_keep].copy()
    without=strict.drop(columns=['contract_expiration_date','contract_years_remaining']).copy()
    idkeep=~without.identity_confidence.isin(['LIKELY_INCORRECT','CANNOT_VERIFY'])
    identity_strict=without[idkeep].copy()
    primary=identity_strict[~identity_strict.league_timing_concern & identity_strict.historical_big5_destination_status.eq('OBSERVED_BIG5')].copy()
    variants={
        'DATASET_A_ORIGINAL_COMPARABLE':('original_comparable_dataset.csv',original),
        'DATASET_B_STRICT_PRETRANSFER':('strict_pretransfer_dataset.csv',strict),
        'VARIANT_3_STRICT_PRETRANSFER_WITHOUT_CONTRACT':('strict_pretransfer_without_contract_dataset.csv',without),
        'VARIANT_4_STRICT_PRETRANSFER_IDENTITY_WITHOUT_CONTRACT':('strict_pretransfer_identity_without_contract_dataset.csv',identity_strict),
        'VARIANT_5_LEAGUE_CONFIRMED_WITHOUT_CONTRACT':('strict_pretransfer_identity_league_confirmed_without_contract_dataset.csv',primary)}
    # Retain ambiguous cases in primary; removing them is a named sensitivity only.
    if primary.identity_confidence.eq('AMBIGUOUS').any():
        variants['SENSITIVITY_EXCLUDE_AMBIGUOUS']=('strict_pretransfer_identity_unambiguous_without_contract_dataset.csv',primary[~primary.identity_confidence.eq('AMBIGUOUS')].copy())
    ledger=[];summaries=[];distributions=[]
    for name,(filename,d) in variants.items():
        save(d,filename,F)
        included=set(d.original_clean_row_id)
        for row in original.itertuples():
            keep=row.original_clean_row_id in included
            reason='INCLUDED'
            if not keep:
                if not row.strict_timing_keep:reason='TIMING_NOT_DEFENSIBLE'
                elif row.identity_confidence in ['LIKELY_INCORRECT','CANNOT_VERIFY']:reason='STRICT_IDENTITY:'+row.identity_confidence
                elif row.identity_confidence=='AMBIGUOUS':reason='OPTIONAL_AMBIGUITY_SENSITIVITY'
                elif row.league_timing_concern or row.historical_big5_destination_status!='OBSERVED_BIG5':reason='LEAGUE_SEASON_MEMBERSHIP_NOT_CORROBORATED'
                else:raise AssertionError('Unexplained exclusion')
            ledger.append(dict(dataset=name,original_clean_row_id=row.original_clean_row_id,player_id=row.player_id,included=keep,reason=reason))
        summaries.append(dict(dataset_name=name,filename=filename,n=len(d),unique_player_names=d.player_name.nunique(),unique_player_ids=d.player_id.nunique(),
            rows_before=len(original),rows_excluded=len(original)-len(d),unique_player_ids_before=original.player_id.nunique(),
            seasons=js(counts(d.transfer_season)),position_distribution=js(counts(d.position)),proxy_observations=int(d.season_proxy_flag.sum()),
            fuzzy_matches=int(d.is_original_fuzzy_match.sum()),questionable_fuzzy_matches=int(d.identity_confidence.isin(['AMBIGUOUS','LIKELY_INCORRECT','CANNOT_VERIFY']).sum()),
            identity_categories=js(counts(d.identity_confidence)),contract_predictor_present='contract_years_remaining' in d,
            contract_available_n=int(d.contract_years_remaining.notna().sum()) if 'contract_years_remaining' in d else 0,
            contract_audit_categories=js(counts(d.contract_audit_class)),league_timing_concerns=int(d.league_timing_concern.sum()),
            possible_promotion_relegation_or_coverage_changes=int(d.possible_promotion_relegation_timing.sum()),
            target_missingness=int(d.log_transfer_fee.isna().sum()),primary_recommendation=name=='VARIANT_5_LEAGUE_CONFIRMED_WITHOUT_CONTRACT'))
        for typ,col in [('season','transfer_season'),('position','position')]:
            for value,n in counts(d[col]).items():distributions.append(dict(dataset=name,distribution=typ,category=value,n=n))
    save(pd.DataFrame(summaries),'DATASET_VARIANT_SUMMARY.csv',F)
    save(pd.DataFrame(distributions),'dataset_variant_distributions.csv')
    save(pd.DataFrame(ledger),'dataset_row_inclusion_ledger.csv')

    # Rank/VIF diagnostics concern X only. No y, scores, CV or final model fitting.
    structural=['age_at_transfer','age_squared','U21_dummy','O30_dummy','league_level_diff']
    categorical=['sub_position','from_league','to_league','transfer_season']
    performance=['Gls_per90','Mins_Per_90_Playing','xG_Per','PrgC_per90','Cmp_percent_Total','KP_per90','Won_percent_Aerial','Recov_per90']
    specs={label:dict(numeric_structural=structural+(['contract_years_remaining'] if label=='WITH_CONTRACT' else []),categorical_fixed_effects=categorical,
        original_model4_performance_for_rank_audit=performance,additional_original_candidates_not_reselected=['Ast_per90','Succ_Take_per90'],
        excluded_redundant_predictors=['covid_dummy','season_proxy_flag','UEFA_coeff_from'],
        future_preprocessing='Fit numeric medians, categorical modes and one-hot levels inside each future training fold; no full-sample audit matrix may be reused for validation.') for label in ['WITH_CONTRACT','WITHOUT_CONTRACT']}
    (F/'future_modeling_specifications.json').write_text(json.dumps(specs,indent=2))
    ranks=[];vifrows=[];designmap=[]
    for name,(filename,d) in variants.items():
        spec='WITH_CONTRACT' if 'contract_years_remaining' in d else 'WITHOUT_CONTRACT'
        nums=specs[spec]['numeric_structural']+performance
        num=d[nums].replace([np.inf,-np.inf],np.nan).astype(float)
        medians=num.median();num=num.fillna(medians)
        cats=d[categorical].copy()
        for c in categorical:cats[c]=cats[c].fillna(cats[c].mode().iloc[0]).astype(str)
        enc=pd.get_dummies(cats,drop_first=True,dtype=float)
        X=pd.concat([pd.Series(1.,index=d.index,name='const'),num,enc],axis=1)
        a=X.to_numpy(float);scaled=a/np.linalg.norm(a,axis=0)
        rank=int(np.linalg.matrix_rank(scaled));assert rank==a.shape[1],(name,rank,a.shape)
        z=a[:,1:];z=(z-z.mean(axis=0))/z.std(axis=0)
        corr=z.T@z/len(z);vifs=np.diag(np.linalg.inv(corr))
        singular=np.linalg.svd(scaled,compute_uv=False)
        designfile=f'rank_design_{name}.csv';save(X,designfile)
        for c,v in zip(X.columns[1:],vifs):vifrows.append(dict(dataset=name,specification=spec,feature=c,VIF=float(v),
            reason_for_inclusion='Original retained performance term' if c in performance else 'Nonredundant structural control or reference-coded fixed effect',
            scope='Full-sample X-only audit; NOT a training/validation artifact'))
        ranks.append(dict(dataset=name,specification=spec,n=len(d),columns_including_intercept=a.shape[1],rank=rank,
            rank_equals_columns=rank==a.shape[1],condition_number_column_scaled=float(singular[0]/singular[-1]),max_full_design_vif=float(vifs.max()),
            preprocessing_scope='Full-sample medians/modes for diagnostic only; future modeling must refit within training folds'))
        designmap.append(dict(dataset=name,reference_categories={c:sorted(cats[c].unique())[0] for c in categorical},numeric_imputation_medians=medians.to_dict()))
    save(pd.DataFrame(ranks),'revised_design_rank_summary.csv')
    save(pd.DataFrame(vifrows),'revised_full_design_vif.csv')
    (I/'design_audit_preprocessing.json').write_text(json.dumps(designmap,indent=2))
    origX=pd.read_csv(A/'reproduction/audit_evidence/model4_training_design.csv')
    ox=origX.to_numpy(float);oldrank=int(np.linalg.matrix_rank(ox/np.linalg.norm(ox,axis=0)))
    assert oldrank==41 and origX.shape[1]==44
    deps={
        'covid_minus_2019_20_and_2020_21':float(abs(origX.covid_dummy-origX['transfer_season_2019-20']-origX['transfer_season_2020-21']).max()),
        'proxy_minus_reference_season':float(abs(origX.season_proxy_flag-(1-origX.filter(regex='^transfer_season_').sum(axis=1))).max()),
        'UEFA_minus_origin_lookup':float(abs(origX.UEFA_coeff_from-(82.55+7.7*origX.from_league_GB1-22.9*origX.from_league_FR1-11.5*origX.from_league_IT1-9.7*origX.from_league_L1)).max())}
    assert max(deps.values())<1e-10
    (I/'original_design_dependencies_reconfirmed.json').write_text(json.dumps(deps,indent=2))
    dictionary=[]
    for c in base.columns:
        if c in structural:role='NUMERIC_STRUCTURAL';reason='Retained structural concept; age polynomial terms kept despite basis-related VIF.'
        elif c in categorical:role='CATEGORICAL_FIXED_EFFECT';reason='Reference-code exactly once; learn levels in future training fold.'
        elif c in performance:role='ORIGINAL_MODEL4_PERFORMANCE';reason='Carried forward for rank audit only; no new selection or fit.'
        elif c in ['Ast_per90','Succ_Take_per90']:role='AVAILABLE_CANDIDATE_NOT_IN_RANK_DESIGN';reason='Original candidate; no future selection decision made in this phase.'
        elif c=='contract_years_remaining':role='SENSITIVITY_ONLY';reason='Snapshot expiry not historically established; exclude primary.'
        elif c in ['covid_dummy','season_proxy_flag','UEFA_coeff_from']:role='AUDIT_METADATA_NOT_PREDICTOR';reason='Deterministic in retained season/origin fixed effects; exclude from both designs.'
        elif c=='log_transfer_fee':role='TARGET';reason='Natural-log observed fee; never an X predictor.'
        else:role='PROVENANCE_OR_UNUSED_FIELD';reason='Not authorized as a primary predictor; retained for audit/traceability where present.'
        dictionary.append(dict(feature=c,role=role,primary_inclusion=role in ['NUMERIC_STRUCTURAL','CATEGORICAL_FIXED_EFFECT','ORIGINAL_MODEL4_PERFORMANCE'],
            with_contract_inclusion=role in ['NUMERIC_STRUCTURAL','CATEGORICAL_FIXED_EFFECT','ORIGINAL_MODEL4_PERFORMANCE','SENSITIVITY_ONLY'],
            reason_for_inclusion=reason if role in ['NUMERIC_STRUCTURAL','CATEGORICAL_FIXED_EFFECT','ORIGINAL_MODEL4_PERFORMANCE'] else '',
            reason_for_exclusion=reason if role not in ['NUMERIC_STRUCTURAL','CATEGORICAL_FIXED_EFFECT','ORIGINAL_MODEL4_PERFORMANCE'] else '',
            VIF_location='02_data/intermediate/revised_full_design_vif.csv; dummy-level and dataset-specific',
            original_value_modified=False))
    save(pd.DataFrame(dictionary),'revised_feature_dictionary.csv',F)
    summary=dict(phase='PHASE2_DATA_AND_DESIGN_ONLY',original_clean_n=1622,original_model_n=len(original),
        timing_classifications_clean=counts(timing.timing_classification),timing_classifications_model=counts(timing.loc[eligible,'timing_classification']),
        original_proxy_model=int(original.season_proxy_flag.sum()),proxy_retained_strict=int(strict.season_proxy_flag.sum()),
        nonproxy_excluded_for_timing=int((~original.strict_timing_keep & original.season_proxy_flag.eq(0)).sum()),
        contract_model_categories=counts(contracts.loc[eligible,'plausibility_flag']),
        contract_original_min=float(original.contract_years_remaining.min()),contract_original_median=float(original.contract_years_remaining.median()),contract_original_max=float(original.contract_years_remaining.max()),
        fuzzy_total=len(identity),fuzzy_categories=counts(identity.identity_confidence),fuzzy_affecting_model=int(identity.original_model_rows_affected.sum()),
        strict_identity_removal_from_original=int(original.identity_confidence.isin(['LIKELY_INCORRECT','CANNOT_VERIFY']).sum()),
        strict_identity_removal_after_timing=len(without)-len(identity_strict),league_confirmation_removal=len(identity_strict)-len(primary),ambiguous_in_primary=int(primary.identity_confidence.eq('AMBIGUOUS').sum()),
        league_concern_original=int(original.league_timing_concern.sum()),league_concern_primary=int(primary.league_timing_concern.sum()),
        destination_unconfirmed_primary=int(primary.historical_big5_destination_status.ne('OBSERVED_BIG5').sum()),
        original_rank=oldrank,original_columns=44,variants=summaries,rank_summaries=ranks,final_models_fitted=False)
    (I/'PHASE2_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    assert original[clean.columns].reset_index(drop=True).equals(clean.loc[eligible].reset_index(drop=True))
    assert primary.strict_timing_keep.all() and not primary.identity_confidence.isin(['LIKELY_INCORRECT','CANNOT_VERIFY']).any()
    assert 'contract_years_remaining' not in primary
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
