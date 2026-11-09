#!/usr/bin/env python3
"""Aggregate interpretation diagnostics for completed DWI/fMRI foldwise runs.

Cohort/batch mean-age baselines are diagnostic non-imaging comparators, fitted
only on outer-training CN. Bootstrap intervals condition on fixed OOF models;
they do not capture refitting, model-selection or split uncertainty. No private
prediction records or participant IDs are exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def analyze(output,replicates=3000,seed=20261106):
    output.mkdir(parents=True,exist_ok=True)
    age_tables=[];baseline_tables=[];comparisons=[];dependence=[];probe_tables=[];performance=[];sources={}
    for modality in ['DWI','fMRI']:
        run=ROOT/'results/final'/modality/'foldwise_Ridge_validated_20261106'
        pred=pd.read_csv(run/'oof_predictions_private.csv')
        meta=pd.read_csv(run/'eligible_metadata_private.csv')
        splits=pd.read_csv(run/'splits_private.csv')
        cn=pred[pred.arm.eq('no_combat') & pred.is_cn].copy()
        for name in ['oof_predictions_private.csv','eligible_metadata_private.csv','splits_private.csv']:
            sources[str((run/name).relative_to(ROOT))]=hashlib.sha256((run/name).read_bytes()).hexdigest()
        ages=cn.groupby('cohort').agg(scans=('age','size'),participants=('participant_group','nunique'),
                   age_mean=('age','mean'),age_SD=('age','std'),age_min=('age','min'),age_max=('age','max')).reset_index()
        ages.insert(0,'modality',modality);age_tables.append(ages)
        records=[]
        for fold,test in cn.groupby('fold'):
            idx=splits[splits.fold.eq(fold) & splits.role.eq('outer_train')].row_index.to_numpy(dtype=int)
            train=meta.iloc[idx];train=train[train.is_cn]
            assert set(train.participant_group).isdisjoint(test.participant_group)
            for level in ['pooled','cohort','batch']:
                estimate=np.full(len(test),train.age.mean()) if level=='pooled' else test[level].map(train.groupby(level).age.mean()).to_numpy()
                assert np.isfinite(estimate).all()
                records.extend(dict(baseline=level,absolute_error=float(error)) for error in abs(estimate-test.age.to_numpy()))
        baseline=pd.DataFrame(records).groupby('baseline').absolute_error.mean().rename('MAE').reset_index()
        baseline.insert(0,'modality',modality);baseline_tables.append(baseline)
        wide=pred[pred.is_cn].pivot(index='row_index',columns='arm',values='predicted_age').join(cn.set_index('row_index')[['age','cohort','participant_group']])
        rng=np.random.default_rng(seed)
        for arm in ['combat_batch_only','combat_train_age_mean_only']:
            d=wide[['cohort','participant_group']].copy()
            d['difference']=abs(wide[arm]-wide.age)-abs(wide.no_combat-wide.age)
            g=d.groupby(['cohort','participant_group']).agg(difference=('difference','sum'),n=('difference','size')).reset_index()
            numerator=np.zeros(replicates);denominator=np.zeros(replicates)
            for _,part in g.groupby('cohort'):
                counts=rng.multinomial(len(part),np.full(len(part),1/len(part)),size=replicates)
                numerator+=counts@part.difference.to_numpy();denominator+=counts@part.n.to_numpy()
            low,high=np.percentile(numerator/denominator,[2.5,97.5])
            comparisons.append(dict(modality=modality,arm=arm,reference='no_combat',
                 paired_MAE_difference_years=float(d.difference.mean()),fixed_OOF_CI_low=low,fixed_OOF_CI_high=high))
        for arm,part in pred[pred.is_cn].groupby('arm'):
            age=part.age-part.groupby('cohort').age.transform('mean')
            bag=part.cBAG-part.groupby('cohort').cBAG.transform('mean')
            dependence.append(dict(modality=modality,arm=arm,pooled_cBAG_age_r=part.age.corr(part.cBAG),
                  within_cohort_centered_cBAG_age_r=age.corr(bag),predicted_age_SD=part.predicted_age.std(),age_SD=part.age.std()))
        probes=pd.read_csv(run/'retention_probes.csv').groupby('arm')[['scanner_balanced_accuracy','sex_balanced_accuracy']].mean().reset_index()
        probes.insert(0,'modality',modality);probe_tables.append(probes)
        perf=pd.read_csv(run/'oof_metrics.csv');perf.insert(0,'modality',modality);performance.append(perf)
    tables={'CN_age_distributions':pd.concat(age_tables),'training_only_age_baselines':pd.concat(baseline_tables),
            'paired_MAE_differences':pd.DataFrame(comparisons),'residual_age_diagnostics':pd.DataFrame(dependence),
            'mean_retention_probes':pd.concat(probe_tables),'prediction_metrics':pd.concat(performance)}
    sections=['<h1>Interpretation of fold-contained Ridge results</h1>',
              '<p>Modality populations and folds differ: DWI/fMRI values are not a matched modality comparison. '
              'Cohort/batch mean-age predictors are diagnostic non-imaging comparators, not proposed imaging-only models. '
              'Bootstrap intervals use paired participant-cluster resampling stratified by cohort and condition on fixed OOF models. '
              'They exclude refitting, selection and split uncertainty. Probe scores evaluate all eligible held-out subjects; prediction MAE here evaluates CN.</p>']
    for name,table in tables.items():
        table.to_csv(output/(name+'.csv'),index=False)
        sections+=['<h2>'+name.replace('_',' ')+'</h2>',table.to_html(index=False,float_format=lambda x:f'{x:.4f}')]
    limitations=['Low pooled cBAG-age correlation does not establish cohort-specific calibration.',
                 'Scanner/batch predictability mixes acquisition, cohort and biological differences.',
                 'Sex predictability alone cannot establish biological preservation.',
                 'Source diagnoses, clinical visit matching, protocol consistency and eligibility restrictions remain unresolved.',
                 'No clinical association, causal conclusion or model-refitting uncertainty is estimated.']
    provenance=dict(seed=seed,bootstrap_replicates=replicates,sources=sources,limitations=limitations)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    css='<style>body{font:14px Arial;margin:32px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}h2{margin-top:32px}</style>'
    (output/'analysis_report.html').write_text('<!doctype html><meta charset="utf-8">'+css+''.join(sections))
    print(output/'analysis_report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/final/foldwise_analysis_20261106')
    p.add_argument('--bootstrap',type=int,default=3000)
    args=p.parse_args()
    if args.bootstrap<100:p.error('Use at least 100 bootstrap replicates')
    analyze(args.output,args.bootstrap)
