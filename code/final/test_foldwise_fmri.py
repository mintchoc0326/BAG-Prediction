"""fMRI signed-correlation and leakage-boundary tests using synthetic data."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from foldwise_fmri import FmriPreprocessor
from run_foldwise_DWI_Ridge import run,supported_splits,tune
import test_foldwise_dwi as dwi_tests
from audit_foldwise_run import audit


class FmriBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dwi_tests.FoldBoundaries.setUpClass()
        cls.X=np.tanh(np.log(dwi_tests.FoldBoundaries.X)-1.6)
        cls.meta=dwi_tests.FoldBoundaries.meta.copy()

    def test_signed_fisher_domain(self):
        self.assertTrue((self.X<0).any())
        prep=FmriPreprocessor(False).fit(self.X,self.meta.batch)
        actual=prep.transform(self.X,self.meta.batch,standardize=False)
        np.testing.assert_allclose(actual,np.arctanh(self.X[:,prep.mask_]),atol=1e-12)
        self.assertTrue(np.isfinite(prep._features(np.array([[-1.,1.]]))).all())
        with self.assertRaisesRegex(ValueError,'Invalid fMRI'):
            prep._features(np.array([[1.1]]))

    def test_age_isolation_and_train_apply_equivalence(self):
        outer,_=supported_splits(self.meta,3,42);train,test=outer[0]
        inner,_=supported_splits(self.meta.iloc[train].reset_index(drop=True),2,52)
        for additive in [False,True]:
            prep=FmriPreprocessor(True,additive).fit(self.X[train],self.meta.batch.iloc[train],
                       self.meta.age.iloc[train] if additive else None)
            self.assertLess(prep.training_transform_max_error_,1e-8)
            state=joblib.hash(prep)
            whole=prep.transform(self.X[test],self.meta.batch.iloc[test])
            np.testing.assert_allclose(whole[:1],prep.transform(self.X[test[:1]],self.meta.batch.iloc[test[:1]]),atol=1e-12)
            self.assertEqual(state,joblib.hash(prep))
            original=tune(self.X,self.meta,train,inner,[1.,10.],True,additive,FmriPreprocessor)
            changed=self.meta.copy();changed.loc[test,'age']+=1000
            altered=tune(self.X,changed,train,inner,[1.,10.],True,additive,FmriPreprocessor)
            np.testing.assert_allclose(original[:3],altered[:3],rtol=0,atol=0)
            np.testing.assert_array_equal(original[-1],altered[-1])
        self.assertNotIn('age',inspect.signature(FmriPreprocessor.transform).parameters)

    def test_nested_fmri_run_artifact_audit(self):
        args=SimpleNamespace(outer_folds=3,inner_folds=2,seed=42,alphas=[1.,10.],preprocessor_class=FmriPreprocessor,modality='fMRI')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            self.meta.to_csv(output/'eligible_metadata_private.csv',index=False)
            result=run(self.X,self.meta,args,output)
            (output/'manifest.json').write_text(json.dumps(dict(result,outer_folds=3,code_sha256={})))
            audit(output)
            self.assertIn('fMRI Ridge',(output/'validation_report.html').read_text())


if __name__=='__main__':
    unittest.main()
