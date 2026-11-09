#!/usr/bin/env python3
"""Independently audit a completed run's folds, OOF coverage and calibration.

No participant records are printed; writes aggregate validation_checks.json.
Synthetic held-out-information invariance tests are in test_foldwise_dwi.py.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from foldwise_dwi import ROOT


def audit(output):
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='complete','Training run is incomplete'
    meta=pd.read_csv(output/'eligible_metadata_private.csv')
    pred=pd.read_csv(output/'oof_predictions_private.csv')
    for arm,rows in pred.groupby('arm'):
        assert len(rows)==len(meta) and rows.row_index.nunique()==len(meta)
        for fold in range(1,manifest['outer_folds']+1):
            model=joblib.load(output/f'{arm}_fold{fold}.joblib')
            train,test=model['outer_train_indices'],model['outer_test_indices']
            assert set(meta.participant_group.iloc[train]).isdisjoint(meta.participant_group.iloc[test])
            assert meta.is_cn.iloc[train].all()
            calibration=pd.read_csv(output/f'{arm}_fold{fold}_calibration_private.csv')
            assert set(calibration.row_index)==set(train)
            ages=meta.age.iloc[calibration.row_index].to_numpy()
            a,b=np.polyfit(ages,calibration.inner_oof_predicted_age.to_numpy()-ages,1)
            np.testing.assert_allclose([a,b],[model['bias_slope'],model['bias_intercept']],atol=1e-10)
            held=rows[rows.fold==fold]
            assert set(held.row_index)==set(test)
            np.testing.assert_allclose(held.cBAG,held.predicted_age-held.age-(a*held.age+b),atol=1e-9)
    for fold in range(1,manifest['outer_folds']+1):
        masks=[joblib.load(output/f'{arm}_fold{fold}.joblib')['preprocessor'].mask_ for arm in pred.arm.unique()]
        assert all(np.array_equal(masks[0],mask) for mask in masks[1:])
    for source,digest in manifest['code_sha256'].items():
        assert hashlib.sha256((output/source).read_bytes()).hexdigest()==digest
    splits=pd.read_csv(output/'splits_private.csv')
    for fold,rows in splits.groupby('fold'):
        heldout=set(rows.loc[rows.role.eq('outer_test'),'participant_group'])
        assert heldout.isdisjoint(rows.loc[rows.role.ne('outer_test'),'participant_group'])
        for inner,part in rows[rows.inner_fold.notna()].groupby('inner_fold'):
            assert set(part.loc[part.role.eq('inner_train'),'participant_group']).isdisjoint(part.loc[part.role.eq('inner_validation'),'participant_group'])
    preprocessing=pd.read_csv(output/'preprocessing_audit.csv')
    error=float(preprocessing.training_transform_max_error.max())
    assert error<1e-8
    result=dict(status='passed',outer_folds=manifest['outer_folds'],arms=pred.arm.nunique(),
                eligible_scans=len(meta),eligible_participants=meta.participant_group.nunique(),
                CN_scans=int(meta.is_cn.sum()),CN_participants=meta.loc[meta.is_cn,'participant_group'].nunique(),
                oof_rows=len(pred),inner_preprocessing_fits=len(preprocessing),max_train_apply_error=error,
                checks=['complete OOF coverage','participant-disjoint outer and inner splits','CN-only fitting',
                        'identical feature masks across arms','training-only calibration provenance',
                        'saved cBAG arithmetic','training source snapshot hashes'],
                versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','neuroCombat','joblib']})
    (output/'validation_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'results/final/DWI/foldwise_Ridge_validated_20261106')
    args=parser.parse_args();audit(args.run)
