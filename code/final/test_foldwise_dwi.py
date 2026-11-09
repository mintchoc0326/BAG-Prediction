"""Meaningful leakage-boundary checks; synthetic data only."""
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from foldwise_dwi import DwiPreprocessor
from run_foldwise_DWI_Ridge import supported_splits,tune,run


class FoldBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(81)
        n=96
        cls.meta=pd.DataFrame({'batch':np.repeat(['A','B'],n//2),
             'participant_group':np.repeat([f'p{i}' for i in range(n//2)],2),
             'age':rng.uniform(50,85,n),'sex':np.tile(['M','F'],n//2),
             'is_cn':True,'scan_id':[f's{i}' for i in range(n)],'cohort':'SYNTHETIC',
             'clinical_diagnosis':'NORMAL'})
        cls.X=np.exp(rng.normal(1,.3,(n,24))+cls.meta.age.to_numpy()[:,None]*.01+
                     (cls.meta.batch.to_numpy()=='B')[:,None]*.4)

    def test_frozen_transform_and_known_batch(self):
        X=self.X;batch=self.meta.batch.to_numpy()
        train=np.r_[0:32,48:80];test=np.r_[32:48,80:96]
        prep=DwiPreprocessor(True).fit(X[train],batch[train])
        self.assertLess(prep.training_transform_max_error_,1e-8)
        digest=joblib.hash(prep)
        whole=prep.transform(X[test],batch[test])
        alone=prep.transform(X[test[:1]],batch[test[:1]])
        np.testing.assert_allclose(whole[:1],alone,atol=1e-12)
        changed=X[test].copy();changed[1:]*=100
        np.testing.assert_allclose(whole[:1],prep.transform(changed,batch[test])[:1],atol=1e-12)
        self.assertEqual(digest,joblib.hash(prep))
        with self.assertRaisesRegex(ValueError,'Unseen batches'):
            prep.transform(X[:1],['NEW'])
        self.assertNotIn('age',inspect.signature(prep.transform).parameters)

    def test_age_trained_additive_transform_matches_combat(self):
        train=np.r_[0:32,48:80];test=np.r_[32:48,80:96]
        prep=DwiPreprocessor(True,True).fit(self.X[train],self.meta.batch.iloc[train],self.meta.age.iloc[train])
        self.assertLess(prep.training_transform_max_error_,1e-8)
        digest=joblib.hash(prep)
        predicted=prep.transform(self.X[test],self.meta.batch.iloc[test])
        np.testing.assert_allclose(predicted[:1],prep.transform(self.X[test[:1]],self.meta.batch.iloc[test[:1]]),atol=1e-12)
        self.assertEqual(digest,joblib.hash(prep))
        outer,_=supported_splits(self.meta,3,42);train,test=outer[0]
        inner,_=supported_splits(self.meta.iloc[train].reset_index(drop=True),2,52)
        original=tune(self.X,self.meta,train,inner,[1.,10.],True,True)
        changed=self.meta.copy();changed.loc[test,'age']+=1000
        altered=tune(self.X,changed,train,inner,[1.,10.],True,True)
        np.testing.assert_allclose(original[:3],altered[:3],atol=0,rtol=0)
        np.testing.assert_array_equal(original[-1],altered[-1])

    def test_inner_choices_ignore_outer_test_ages(self):
        outer,_=supported_splits(self.meta,3,42)
        train,test=outer[0]
        inner,_=supported_splits(self.meta.iloc[train].reset_index(drop=True),2,52)
        a=tune(self.X,self.meta,train,inner,[1.,10.],True)
        altered=self.meta.copy();altered.loc[test,'age']+=1000
        b=tune(self.X,altered,train,inner,[1.,10.],True)
        np.testing.assert_allclose(a[:3],b[:3],atol=0,rtol=0)
        np.testing.assert_array_equal(a[-1],b[-1])

    def test_end_to_end_grouped_oof_and_calibration_provenance(self):
        args=SimpleNamespace(outer_folds=3,inner_folds=2,seed=42,alphas=[1.,10.])
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            status=run(self.X,self.meta,args,output)
            self.assertEqual(status['status'],'complete')
            pred=pd.read_csv(output/'oof_predictions_private.csv')
            for arm,part in pred.groupby('arm'):
                self.assertEqual(len(part),len(self.X))
                self.assertEqual(part.row_index.nunique(),len(self.X))
                for fold in part.fold.unique():
                    saved=joblib.load(output/f'{arm}_fold{fold}.joblib')
                    train=saved['outer_train_indices'];test=saved['outer_test_indices']
                    self.assertTrue(set(self.meta.participant_group.iloc[train]).isdisjoint(self.meta.participant_group.iloc[test]))
                    calibration=pd.read_csv(output/f'{arm}_fold{fold}_calibration_private.csv')
                    self.assertTrue(set(calibration.row_index).issubset(set(train)))
                    a,b=np.polyfit(self.meta.age.iloc[calibration.row_index],
                                  calibration.inner_oof_predicted_age-self.meta.age.iloc[calibration.row_index].to_numpy(),1)
                    np.testing.assert_allclose([a,b],[saved['bias_slope'],saved['bias_intercept']],rtol=1e-10,atol=1e-10)
                    test_rows=part[part.fold==fold]
                    np.testing.assert_allclose(test_rows.cBAG,test_rows.predicted_age-test_rows.age-(a*test_rows.age+b),atol=1e-9)


if __name__=='__main__':
    unittest.main()
