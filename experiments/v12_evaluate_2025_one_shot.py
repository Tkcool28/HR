"""Frozen one-shot evaluator for the authorized 2025 v1.2 holdout."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score,brier_score_loss,log_loss,roc_auc_score)
from sklearn.linear_model import LogisticRegression

SEED=20260904
OBVIOUS=[
    'batter_barrel_rate_season','batter_barrel_rate_career',
    'batter_xwoba_on_contact_season','batter_xwoba_on_contact_career',
    'batter_avg_ev_season',
]


def matrix(df,cols,means=None):
    X=df[cols].to_numpy(dtype=np.float32,copy=True)
    if means is None: means=np.nanmean(X,axis=0).astype(np.float32)
    if np.isnan(means).any(): raise RuntimeError('all-NaN feature mean')
    rr,cc=np.where(np.isnan(X));
    if len(rr): X[rr,cc]=means[cc]
    if not np.isfinite(X).all(): raise RuntimeError('nonfinite model matrix')
    return X,means


def ordered(g,score):
    return g.sort_values([score,'game_pk','batter_id'],ascending=[False,True,True])


def add_ranks(d,score,name):
    out=pd.Series(index=d.index,dtype='int32')
    for _,g in d.groupby('game_date',sort=True):
        z=ordered(g,score); out.loc[z.index]=np.arange(1,len(z)+1,dtype=np.int32)
    return out.astype('int32')


def pct_mask(d,rank_col,p):
    sizes=d.groupby('game_date').size(); n=d.game_date.map(lambda x:max(1,math.ceil(p*int(sizes.loc[x]))))
    return d[rank_col].le(n)


def rate(d,m):
    x=d.loc[m]
    return {'n':int(len(x)),'hr':int(x.hr_in_game.sum()),'hr_rate':float(x.hr_in_game.mean()) if len(x) else None,
            'n_dates':int(x.game_date.nunique()) if len(x) else 0}


def bootstrap_pair(d,ma,mb,reps,seed):
    dates=np.array(sorted(d.game_date.drop_duplicates().to_numpy()))
    if len(dates)<100: raise RuntimeError('too few 2025 slate dates')
    pos={x:i for i,x in enumerate(dates)}
    def agg(mask):
        a=d.loc[mask,['game_date','hr_in_game']].groupby('game_date').hr_in_game.agg(['sum','count'])
        s=np.zeros(len(dates)); n=np.zeros(len(dates))
        for day,row in a.iterrows():
            k=np.datetime64(pd.Timestamp(day).to_datetime64()); i=pos[k]; s[i]=row['sum']; n[i]=row['count']
        return s,n
    sa,na=agg(ma); sb,nb=agg(mb)
    rng=np.random.default_rng(seed); delta=np.empty(reps); ra=np.empty(reps); rb=np.empty(reps)
    at=0
    while at<reps:
        k=min(1000,reps-at); ix=rng.integers(0,len(dates),size=(k,len(dates)))
        ra[at:at+k]=sa[ix].sum(1)/na[ix].sum(1); rb[at:at+k]=sb[ix].sum(1)/nb[ix].sum(1)
        delta[at:at+k]=rb[at:at+k]-ra[at:at+k]; at+=k
    return {
        'reference_obvious_mean':float(ra.mean()),'full73_mean':float(rb.mean()),
        'delta_mean':float(delta.mean()),'delta_median':float(np.median(delta)),
        'delta_ci95_low':float(np.quantile(delta,.025)),'delta_ci95_high':float(np.quantile(delta,.975)),
        'prob_delta_gt_0':float((delta>0).mean()),'replicates':int(reps),'cluster':'slate_date',
    }


def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); ids=np.clip(np.digitize(p,edges[1:-1]),0,bins-1)
    total=len(y); val=0.0; rows=[]
    for i in range(bins):
        m=ids==i
        if not m.any(): continue
        obs=float(y[m].mean()); pred=float(p[m].mean()); w=float(m.sum()/total)
        val+=w*abs(obs-pred); rows.append({'bin':i,'n':int(m.sum()),'mean_pred':pred,'observed':obs})
    return float(val),rows


def calibration_fit(y,p):
    q=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6); logit=np.log(q/(1-q)).reshape(-1,1)
    lr=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000).fit(logit,y)
    return {'intercept':float(lr.intercept_[0]),'slope':float(lr.coef_[0,0]),
            'mean_pred':float(q.mean()),'observed_rate':float(np.mean(y))}


def composite_obvious(d,med):
    parts=[]
    for c in OBVIOUS:
        v=pd.to_numeric(d[c],errors='coerce').fillna(float(med[c]))
        parts.append(v.groupby(d.game_date).rank(method='average',pct=True))
    return pd.concat(parts,axis=1).mean(axis=1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',required=True); ap.add_argument('--feature-list',required=True)
    ap.add_argument('--contract',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--reps',type=int,default=10000)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if args.reps!=10000: raise RuntimeError('frozen primary bootstrap requires exactly 10000 reps')
    cols=json.loads(Path(args.feature_list).read_text()); con=json.loads(Path(args.contract).read_text())
    if len(cols)!=73 or int(con['best_round'])!=194: raise RuntimeError('frozen architecture contract mismatch')
    d=pd.read_parquet(args.matrix); d['game_date']=pd.to_datetime(d.game_date).dt.normalize()
    if set(d.year.unique())-set(range(2015,2026)): raise RuntimeError('matrix year escaped')
    if d.duplicated(['game_pk','batter_id']).any(): raise RuntimeError('duplicate batter-game')
    fit=d[d.year<=2023].copy(); cal=d[d.year==2024].copy(); hold=d[d.year==2025].copy()
    if min(len(fit),len(cal),len(hold))<=0: raise RuntimeError('empty frozen partition')
    if not hold.groupby('game_pk').size().eq(18).all(): raise RuntimeError('2025 target universe not 18/game')
    Xfit,means=matrix(fit,cols); Xcal,_=matrix(cal,cols,means); Xh,_=matrix(hold,cols,means)
    yf=fit.hr_in_game.to_numpy(dtype=np.int8); yc=cal.hr_in_game.to_numpy(dtype=np.int8); yh=hold.hr_in_game.to_numpy(dtype=np.int8)
    params={'objective':'binary:logistic','eval_metric':'rmse','tree_method':'hist','seed':42,'nthread':-1,**con['best_params']}
    bst=xgb.train(params,xgb.DMatrix(Xfit,label=yf,feature_names=cols),num_boost_round=194,verbose_eval=False)
    p24=bst.predict(xgb.DMatrix(Xcal,feature_names=cols)); iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p24,yc)
    raw=bst.predict(xgb.DMatrix(Xh,feature_names=cols)); pc=np.asarray(iso.predict(raw),dtype=float)
    hold['p_raw']=raw; hold['p_cal']=pc

    # Frozen obvious-power proxy. Missing-value medians may use historical <=2024 feature data only.
    if any(c not in d.columns for c in OBVIOUS): raise RuntimeError('obvious proxy feature missing')
    med=d[d.year<=2024][OBVIOUS].median(numeric_only=True)
    if med.isna().any(): raise RuntimeError('obvious proxy all-NaN median')
    hold['obvious_power_score']=composite_obvious(hold,med)
    hold['model_rank']=add_ranks(hold,'p_raw','model_rank'); hold['obvious_rank']=add_ranks(hold,'obvious_power_score','obvious_rank')
    m5=pct_mask(hold,'model_rank',.05); o5=pct_mask(hold,'obvious_rank',.05)

    base=float(yh.mean()); primary={
        'year':2025,'n_target_rows':int(len(hold)),'n_games':int(hold.game_pk.nunique()),'n_slate_dates':int(hold.game_date.nunique()),
        'base_hr_rate':base,'full73_top5':rate(hold,m5),'obvious_top5':rate(hold,o5),
        'observed_full73_minus_obvious_pp':float((hold.loc[m5,'hr_in_game'].mean()-hold.loc[o5,'hr_in_game'].mean())*100),
        'full73_top5_minus_base_pp':float((hold.loc[m5,'hr_in_game'].mean()-base)*100),
        'paired_bootstrap':bootstrap_pair(hold,o5,m5,args.reps,SEED),
    }
    # Save the primary result before secondary analysis.
    (out/'01_primary_2025_top5.json').write_text(json.dumps(primary,indent=2))

    raw_ece,raw_rel=ece(yh,raw); cal_ece,cal_rel=ece(yh,pc)
    metrics={
        'raw_brier':float(brier_score_loss(yh,raw)),'cal_brier':float(brier_score_loss(yh,pc)),
        'raw_auc':float(roc_auc_score(yh,raw)),'cal_auc':float(roc_auc_score(yh,pc)),
        'raw_ap':float(average_precision_score(yh,raw)),'cal_ap':float(average_precision_score(yh,pc)),
        'raw_logloss':float(log_loss(yh,np.clip(raw,1e-7,1-1e-7))),
        'cal_logloss':float(log_loss(yh,np.clip(pc,1e-7,1-1e-7))),
        'raw_ece_10bin':raw_ece,'cal_ece_10bin':cal_ece,
        'raw_calibration_fit':calibration_fit(yh,raw),'cal_calibration_fit':calibration_fit(yh,pc),
    }
    tails={}
    for label,p in [('top10pct',.10),('top5pct',.05),('top2pct',.02),('top1pct',.01)]:
        tails[label]=rate(hold,pct_mask(hold,'model_rank',p))
    for n in [1,2,4,8]: tails[f'top{n}_per_day']=rate(hold,hold.model_rank.le(n))

    mo=m5 & ~o5; shared=m5&o5
    structural={
        'top5_overlap_n':int(shared.sum()),'top5_overlap_fraction_of_full73':float(shared.sum()/max(1,m5.sum())),
        'full73_only_n':int(mo.sum()),
        'full73_only_hr_rate':float(hold.loc[mo,'hr_in_game'].mean()) if mo.any() else None,
        'full73_only_obvious_depth_mean':None,'full73_only_obvious_depth_median':None,
        'full73_only_obvious_rank_mean':float(hold.loc[mo,'obvious_rank'].mean()) if mo.any() else None,
        'full73_only_obvious_rank_median':float(hold.loc[mo,'obvious_rank'].median()) if mo.any() else None,
    }
    slate=hold.groupby('game_date').size()
    if mo.any():
        den=hold.loc[mo,'game_date'].map(lambda x:max(1,int(slate.loc[x])-1)).to_numpy(float)
        depth=(hold.loc[mo,'obvious_rank'].to_numpy(float)-1)/den
        structural['full73_only_obvious_depth_mean']=float(depth.mean()); structural['full73_only_obvious_depth_median']=float(np.median(depth))

    predcols=['game_pk','batter_id','game_date','year','hr_in_game','p_raw','p_cal','obvious_power_score','model_rank','obvious_rank']
    hold[predcols].to_parquet(out/'2025_predictions_and_outcomes.parquet',index=False)
    pd.DataFrame(raw_rel).to_csv(out/'2025_raw_reliability.csv',index=False); pd.DataFrame(cal_rel).to_csv(out/'2025_cal_reliability.csv',index=False)
    result={
        'freeze':{'fit_years':'2015-2023','calibration_year':2024,'holdout_year':2025,'n_features':73,'rounds':194,
                  'feature_list_sha256':hashlib.sha256(Path(args.feature_list).read_bytes()).hexdigest(),
                  'posthoc_tuning_or_subgroup_selection':False},
        'primary':primary,'secondary_model_metrics':metrics,'secondary_tail_metrics':tails,'secondary_structural':structural,
    }
    (out/'02_complete_2025_result.json').write_text(json.dumps(result,indent=2))
    print('=== FROZEN 2025 PRIMARY RESULT ==='); print(json.dumps(primary,indent=2),flush=True)
    print('=== FROZEN SECONDARY METRICS ==='); print(json.dumps({'model':metrics,'tails':tails,'structural':structural},indent=2),flush=True)

if __name__=='__main__': main()
