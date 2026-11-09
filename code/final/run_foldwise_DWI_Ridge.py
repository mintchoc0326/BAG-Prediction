#!/usr/bin/env python3
"""Nested known-site DWI Ridge: raw input, two ComBat modes, matched no-ComBat arm.

All learned preprocessing and alpha selection are fitted inside training folds.
Held-out age is absent from transform APIs. Training age can inform additive ComBat.
This validates implementation boundaries, not clinical labels or scanner protocols.
"""
import argparse
import hashlib
import html
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from foldwise_dwi import ROOT, DwiPreprocessor, load_eligible


def supported_splits(meta,n_splits,seed):
    """First feasible batch-stratified grouped split, without inspecting ages/errors."""
    batches=meta.batch.to_numpy();groups=meta.participant_group.to_numpy()
    cn=meta.is_cn.to_numpy()
    for attempt in range(100):
        cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed+attempt)
        splits=list(cv.split(np.zeros(len(meta)),batches,groups))
        feasible=True
        for train,test in splits:
            assert set(groups[train]).isdisjoint(set(groups[test]))
            support=pd.Series(batches[train][cn[train]]).value_counts()
            if any(support.get(b,0)<2 for b in set(batches[test])|set(batches[train])):
                feasible=False;break
        if feasible:
            return splits,seed+attempt
    raise ValueError('No feasible known-site split; inspect CN participant support, do not use test data to fit')


def metrics(age,prediction):
    return dict(n_scans=len(age),MAE=float(mean_absolute_error(age,prediction)),
                RMSE=float(np.sqrt(mean_squared_error(age,prediction))),
                R2=float(r2_score(age,prediction)) if len(age)>1 else None,
                Pearson_r=float(pearsonr(age,prediction)[0]) if len(age)>2 and np.std(age)>0 and np.std(prediction)>0 else None)


def tune(X,meta,indices,splits,alphas,harmonize,age_trained_mean_only=False,preprocessor_class=DwiPreprocessor):
    """Inner OOF CN predictions; preprocessing fitted once per inner split, not per alpha."""
    inner_meta=meta.iloc[indices].reset_index(drop=True)
    predictions=np.full((len(indices),len(alphas)),np.nan)
    audit=[]
    for inner_fold,(train_local,val_local) in enumerate(splits,1):
        train,val=indices[train_local],indices[val_local]
        prep=preprocessor_class(harmonize,age_trained_mean_only).fit(
            X[train],meta.batch.iloc[train],
            training_ages=meta.age.iloc[train] if age_trained_mean_only else None)
        fit=prep.transform(X[train],meta.batch.iloc[train])
        validation=prep.transform(X[val],meta.batch.iloc[val])
        for column,alpha in enumerate(alphas):
            model=Ridge(alpha=alpha,solver='lsqr',tol=1e-6).fit(fit,meta.age.iloc[train])
            predictions[val_local,column]=model.predict(validation)
        audit.append(dict(inner_fold=inner_fold,n_train=len(train),n_validation=len(val),
                          train_groups=meta.participant_group.iloc[train].nunique(),
                          validation_groups=meta.participant_group.iloc[val].nunique(),
                          retained_edges=int(prep.mask_.sum()),
                          training_transform_max_error=prep.training_transform_max_error_))
    if not np.isfinite(predictions).all():
        raise AssertionError('Missing inner OOF predictions')
    scores=np.mean(np.abs(predictions-inner_meta.age.to_numpy()[:,None]),axis=0)
    selected=int(np.argmin(scores))
    # Only outer-training CN ages/predictions enter calibration.
    slope,intercept=np.polyfit(inner_meta.age,predictions[:,selected]-inner_meta.age,1)
    return float(alphas[selected]),float(slope),float(intercept),scores,audit,predictions[:,selected]


def probe(Xtrain,Xtest,train_meta,test_meta):
    scanner=LogisticRegression(C=.1,max_iter=1000,solver='lbfgs').fit(Xtrain,train_meta.batch)
    score=float(balanced_accuracy_score(test_meta.batch,scanner.predict(Xtest)))
    train_known=train_meta.sex.isin(['M','F']).to_numpy()
    test_known=test_meta.sex.isin(['M','F']).to_numpy()
    sex_score=None
    if train_meta.loc[train_known,'sex'].nunique()==2 and test_known.sum()>1:
        model=LogisticRegression(C=.1,max_iter=1000).fit(Xtrain[train_known],train_meta.loc[train_known,'sex'])
        sex_score=float(balanced_accuracy_score(test_meta.loc[test_known,'sex'],model.predict(Xtest[test_known])))
    return dict(scanner_balanced_accuracy=score,sex_balanced_accuracy=sex_score,
                scanner_classes_in_test=int(test_meta.batch.nunique()),sex_test_scans=int(test_known.sum()))


def run(X,meta,args,output):
    output.mkdir(parents=True,exist_ok=True)
    preprocessor_class=getattr(args,'preprocessor_class',DwiPreprocessor)
    modality=getattr(args,'modality','DWI')
    outer,split_seed=supported_splits(meta,args.outer_folds,args.seed)
    fold_rows=[];prediction_rows=[];audit_rows=[];split_rows=[];search_rows=[];probe_rows=[]
    for fold,(outer_train,test) in enumerate(outer,1):
        train=outer_train[meta.is_cn.iloc[outer_train].to_numpy()]
        cn_meta=meta.iloc[train].reset_index(drop=True)
        inner,inner_seed=supported_splits(cn_meta,args.inner_folds,args.seed+fold*100)
        for role,idx in [('outer_train',outer_train),('outer_test',test)]:
            for i in idx:
                split_rows.append(dict(fold=fold,role=role,row_index=int(i),participant_group=meta.participant_group.iloc[i]))
        for inner_fold,(it,iv) in enumerate(inner,1):
            for role,idx in [('inner_train',train[it]),('inner_validation',train[iv])]:
                for i in idx:
                    split_rows.append(dict(fold=fold,inner_fold=inner_fold,role=role,row_index=int(i),participant_group=meta.participant_group.iloc[i]))
        for harmonize,age_trained_mean_only,arm in [
                (False,False,'no_combat'),(True,False,'combat_batch_only'),
                (True,True,'combat_train_age_mean_only')]:
            start=time.perf_counter()
            alpha,slope,intercept,scores,audits,calibration=tune(X,meta,train,inner,args.alphas,harmonize,age_trained_mean_only,preprocessor_class)
            for a in audits:
                audit_rows.append(dict(arm=arm,outer_fold=fold,**a))
            for candidate,score in zip(args.alphas,scores):
                search_rows.append(dict(arm=arm,fold=fold,alpha=candidate,inner_oof_MAE=score,selected=candidate==alpha))
            prep=preprocessor_class(harmonize,age_trained_mean_only).fit(
                X[train],meta.batch.iloc[train],
                training_ages=meta.age.iloc[train] if age_trained_mean_only else None)
            trained=prep.transform(X[train],meta.batch.iloc[train])
            heldout=prep.transform(X[test],meta.batch.iloc[test])
            ridge=Ridge(alpha=alpha,solver='lsqr',tol=1e-6).fit(trained,meta.age.iloc[train])
            prediction=ridge.predict(heldout)
            bag=prediction-meta.age.iloc[test].to_numpy()
            cbag=bag-(slope*meta.age.iloc[test].to_numpy()+intercept)
            for local,i in enumerate(test):
                prediction_rows.append(dict(arm=arm,fold=fold,row_index=int(i),scan_id=meta.scan_id.iloc[i],
                    participant_group=meta.participant_group.iloc[i],cohort=meta.cohort.iloc[i],
                    batch=meta.batch.iloc[i],diagnosis=meta.clinical_diagnosis.iloc[i],is_cn=bool(meta.is_cn.iloc[i]),
                    age=float(meta.age.iloc[i]),predicted_age=float(prediction[local]),
                    BAG=float(bag[local]),cBAG=float(cbag[local]),bias_slope=slope,bias_intercept=intercept))
            cn_test=meta.is_cn.iloc[test].to_numpy()
            fold_rows.append(dict(arm=arm,fold=fold,alpha=alpha,n_training_CN=len(train),
                retained_edges=int(prep.mask_.sum()),bias_slope=slope,bias_intercept=intercept,
                fit_seconds=time.perf_counter()-start,**metrics(meta.age.iloc[test].to_numpy()[cn_test],prediction[cn_test])))
            probe_rows.append(dict(arm=arm,fold=fold,**probe(trained,heldout,meta.iloc[train].reset_index(drop=True),meta.iloc[test].reset_index(drop=True))))
            joblib.dump(dict(preprocessor=prep,ridge=ridge,bias_slope=slope,bias_intercept=intercept,
                             outer_train_indices=train,outer_test_indices=test,inner_seed=inner_seed),
                        output/f'{arm}_fold{fold}.joblib')
            # Private calibration predictions permit independent provenance checks.
            pd.DataFrame({'row_index':train,'inner_oof_predicted_age':calibration}).to_csv(output/f'{arm}_fold{fold}_calibration_private.csv',index=False)
            print(f'{arm} fold={fold} alpha={alpha:g} CN test MAE={fold_rows[-1]["MAE"]:.3f}',flush=True)
        pd.DataFrame(prediction_rows).to_csv(output/'oof_predictions_private.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(output/'fold_metrics.csv',index=False)
    pred=pd.DataFrame(prediction_rows)
    for arm,rows in pred.groupby('arm'):
        if rows.row_index.nunique()!=len(meta) or len(rows)!=len(meta):
            raise AssertionError('Each eligible scan must have exactly one outer OOF prediction per arm')
    pd.DataFrame(split_rows).to_csv(output/'splits_private.csv',index=False)
    pd.DataFrame(audit_rows).to_csv(output/'preprocessing_audit.csv',index=False)
    pd.DataFrame(search_rows).to_csv(output/'alpha_search.csv',index=False)
    pd.DataFrame(probe_rows).to_csv(output/'retention_probes.csv',index=False)
    result=[]
    for (arm,cohort),rows in pred.groupby(['arm','cohort']):
        cn=rows[rows.is_cn]
        if len(cn)>1:
            result.append(dict(arm=arm,cohort=cohort,population='CN',n_participants=cn.participant_group.nunique(),**metrics(cn.age.to_numpy(),cn.predicted_age.to_numpy())))
    for arm,rows in pred.groupby('arm'):
        cn=rows[rows.is_cn]
        result.append(dict(arm=arm,cohort='Pooled',population='CN',n_participants=cn.participant_group.nunique(),**metrics(cn.age.to_numpy(),cn.predicted_age.to_numpy())))
    summary=pd.DataFrame(result);summary.to_csv(output/'oof_metrics.csv',index=False)
    dependence=[]
    for arm,rows in pred[pred.is_cn].groupby('arm'):
        for label,part in [('Pooled',rows)]+list(rows.groupby('cohort')):
            if len(part)>2:
                dependence.append(dict(arm=arm,cohort=label,n_scans=len(part),
                                       raw_BAG_age_r=float(pearsonr(part.age,part.BAG)[0]),
                                       corrected_BAG_age_r=float(pearsonr(part.age,part.cBAG)[0])))
    pd.DataFrame(dependence).to_csv(output/'residual_age_dependence.csv',index=False)
    render(pred,summary,pd.DataFrame(probe_rows),output,modality)
    return dict(outer_split_seed=split_seed,eligible_scans=len(meta),eligible_participants=meta.participant_group.nunique(),
                CN_scans=int(meta.is_cn.sum()),status='complete')


def render(pred,summary,probes,output,modality='DWI'):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm,rows in pred[pred.is_cn].groupby('arm'):
        color={'no_combat':'#4878A8','combat_batch_only':'#BC5B58','combat_train_age_mean_only':'#59926C'}[arm]
        axes[0].scatter(rows.age,rows.predicted_age,s=8,alpha=.3,color=color,label=arm)
        axes[1].scatter(rows.age,rows.cBAG,s=8,alpha=.3,color=color,label=arm)
    axes[0].set(xlabel='Chronological age',ylabel='OOF predicted age')
    axes[1].set(xlabel='Chronological age',ylabel='OOF corrected BAG')
    axes[1].axhline(0,lw=.7,color='#999999')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout()
    for ext in ['png','pdf','svg']:
        fig.savefig(output/f'oof_predictions.{ext}',dpi=180)
    plt.close(fig)
    text=('<!doctype html><meta charset="utf-8"><style>body{font:14px Arial;margin:32px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}img{max-width:100%}</style>'
          '<h1>Fold-contained DWI Ridge: known-site validation</h1>'
          '<p>Raw connectomes; canonical participant grouping; no HABS all-CN override; batch-only ComBat. '
          'The batch-only arm fits without covariates; the mean-only arm uses training ages to learn additive site offsets. No held-out ages enter either transform; sex/diagnosis never enter feature transforms. '
          'Age is used for supervised prediction and downstream BAG correction only. '
          'All alpha selection and calibration use outer-training CN only; all arms use identical folds and feature masks.</p>'
          '<h2>CN OOF age prediction</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.3f}')+
          '<h2>Held-out retention probes</h2><p>Fixed-regularization probes, trained on the same CN training features. '
          'Scanner predictability is descriptive and can reflect cohort/age confounding; lower scanner accuracy does not by itself prove better harmonization. '
          'Sex predictability is one limited biological-retention probe, not proof all biological effects survive.</p>'+probes.to_html(index=False,float_format=lambda x:f'{x:.3f}')+
          '<h2>Residual age dependence (CN)</h2>'+pd.read_csv(output/'residual_age_dependence.csv').to_html(index=False,float_format=lambda x:f'{x:.3f}')+'<img src="oof_predictions.svg"><h2>Limits</h2><p>Known supported sites only; sparse batches and ambiguous clinical joins excluded with recorded reasons. '
          'Clinical source labels, age/visit matching, independent identity checks and ADRC protocol consistency remain unverified. '
          'The archived eligibility manifest already excludes some original samples. No claim of unseen-site generalization, '
          'causal scanner removal, validated clinical biomarkers, or fully preserved biology is made. '
          'Held-out ages are used in performance scoring and cBAG arithmetic, never harmonization. '
          'Raw prediction metrics and corrected BAG are reported separately.</p>')
    (output/'validation_report.html').write_text(text.replace('DWI Ridge',modality+' Ridge'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/final/DWI/foldwise_Ridge_validated_20261106')
    parser.add_argument('--outer-folds',type=int,default=5)
    parser.add_argument('--inner-folds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--min-cn-participants',type=int,default=5)
    parser.add_argument('--alphas',type=float,nargs='+',default=[.1,1,10,100,1000,10000,100000])
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing results are not overwritten')
    if min(args.outer_folds,args.inner_folds)<2 or min(args.alphas)<=0:
        parser.error('Need >=2 folds and positive alphas')
    X,meta,exclusions,sources=load_eligible(min_cn_participants=args.min_cn_participants)
    args.output.mkdir(parents=True)
    meta.to_csv(args.output/'eligible_metadata_private.csv',index=False)
    exclusions.to_csv(args.output/'eligibility_audit_private.csv',index=False)
    exclusions.groupby(['cohort','exclusion_reason'],dropna=False).size().rename('scans').reset_index().to_csv(args.output/'eligibility_counts.csv',index=False)
    manifest=dict(status='running',seed=args.seed,outer_folds=args.outer_folds,inner_folds=args.inner_folds,
                  min_cn_participants=args.min_cn_participants,alphas=args.alphas,sources=sources,
                  input_representation='log1p upper-triangle of symmetrized raw 84x84 DWI',
                  harmonization='three arms: none; batch-only location/scale ComBat; training-age mean-only ComBat. No held-out age or sex/diagnosis covariates in transforms',
                  scope='known-site restricted population; clinical/protocol validation pending')
    for name in ['foldwise_dwi.py','run_foldwise_DWI_Ridge.py']:
        source=Path(__file__).with_name(name)
        manifest.setdefault('code_sha256',{})[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        (args.output/name).write_bytes(source.read_bytes())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest.update(run(X,meta,args,args.output))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
