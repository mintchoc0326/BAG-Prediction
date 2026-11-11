"""Synthetic end-to-end checks of the unified pipeline's scientific boundaries."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .calibration import BiasCorrection
from .preprocessing import MultiPreprocessor
from .splits import make_plan, audit_plan
from .engine import run
from .audit import audit
from .reporting import report


class PipelineTests(unittest.TestCase):
    def test_raw_manifest_paths_resolve_against_data_root(self):
        from foldwise_dwi import load_eligible

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'data/final/ADNI'
            metadata.mkdir(parents=True)
            matrices = root / 'raw'
            matrices.mkdir()
            clinical = pd.DataFrame({
                'CONNECTOME_KEY_USED_FOR_INTERSECTION': ['S0', 'S1'],
                'PTID': ['P0', 'P1'], 'DX_Label_harmonized': ['CN', 'CN']})
            clinical.to_csv(metadata / 'ADNI_harmonized_metadata(in).csv', index=False)
            for i in range(2):
                np.savetxt(matrices / f'scan{i}.csv', np.full((84, 84), .2), delimiter=',')
            for modality in ['DWI', 'fMRI']:
                rows = pd.DataFrame({
                    'DATASET': ['ADNI', 'ADNI'], '__id__': ['S0', 'S1'],
                    'AGE': [60., 70.], 'SEX': ['F', 'M'], 'DATASET_SITE': ['A', 'A'],
                    'filepath': ['raw/scan0.csv', str(matrices / 'scan1.csv')]})
                if modality == 'DWI':
                    manifest = root / 'results/final/DWI/metadata/metadata_used_for_joint_combat.csv'
                else:
                    rows = rows.rename(columns={'__id__': 'connectome_key', 'AGE': 'Age',
                                                'SEX': 'Sex', 'filepath': 'file_path'})
                    manifest = root / 'results/final/fMRI/evaluation_inputs/fMRI_metadata_aligned.csv'
                manifest.parent.mkdir(parents=True, exist_ok=True)
                rows.to_csv(manifest, index=False)
                values, meta, _, _ = load_eligible(root, 1, modality)
                self.assertEqual(values.shape, (2, 3486))
                np.testing.assert_allclose(values, .2)
                self.assertEqual(meta.participant_group.nunique(), 2)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(31)
        n = 60
        cls.meta = pd.DataFrame(
            {
                "cohort": np.repeat(["C1", "C2"], 30),
                "batch": np.repeat(["B1", "B2"], 30),
                "participant_group": np.repeat([f"P{i}" for i in range(30)], 2),
                "scan_id": [f"S{i}" for i in range(n)],
                "is_cn": True,
                "clinical_diagnosis": "NORMAL",
                "sex": np.tile(["F", "M"], 30),
                "age": rng.uniform(50, 80, n),
            }
        )
        age = cls.meta.age.to_numpy()[:, None]
        cls.raw = {
            "DWI": np.exp(rng.normal(0, 0.2, (n, 3486)) + 0.01 * age),
            "fMRI": np.tanh(rng.normal(0, 0.2, (n, 3486)) + 0.003 * (age - 65)),
        }
        cls.settings = dict(
            outer_folds=3,
            inner_folds=2,
            seed=42,
            learners=["ridge"],
            harmonization_modes=["none", "age_mean"],
            calibration="cohort",
            ridge_alphas=[10.0, 100.0],
            gine_grid=[dict(layers=1, width=16, residual=False, threshold=0.25)],
            training=dict(max_epochs=1, patience=1, batch_size=16, threads=2),
            validation_only=True,
        )

    def test_multimodal_age_free_frozen_transform(self):
        train = np.r_[0:20, 30:50]
        test = np.r_[20:30, 50:60]
        prep = MultiPreprocessor("age_mean").fit(
            {m: x[train] for m, x in self.raw.items()},
            self.meta.batch.iloc[train],
            self.meta.age.iloc[train],
        )
        digest = joblib.hash(prep)
        full = prep.matrix(
            {m: x[test] for m, x in self.raw.items()}, self.meta.batch.iloc[test]
        )
        solo = prep.matrix(
            {m: x[test[:1]] for m, x in self.raw.items()},
            self.meta.batch.iloc[test[:1]],
        )
        np.testing.assert_allclose(full[:1], solo, atol=1e-10)
        self.assertEqual(digest, joblib.hash(prep))
        self.assertNotIn("age", inspect.signature(prep.transform).parameters)
        for p in prep.transforms.values():
            self.assertLess(p.training_transform_max_error_, 1e-8)

    def test_group_plan_and_label_independence_from_age(self):
        a = make_plan(self.meta, 3, 2, 42)
        audit_plan(a, self.meta)
        changed = self.meta.copy()
        changed.age += 1000
        b = make_plan(changed, 3, 2, 42)
        self.assertEqual(a, b)
        broken = json.loads(json.dumps(a))
        broken["folds"][0]["inner"][0]["early_stop"].append(
            broken["folds"][0]["inner"][0]["early_train"][0]
        )
        with self.assertRaises(AssertionError):
            audit_plan(broken, self.meta)

    def test_cohort_bias_fixed_application(self):
        age = self.meta.age.to_numpy()
        cohort = self.meta.cohort.to_numpy()
        offset = np.where(cohort == "C1", 2.0, -3.0)
        prediction = age + 0.2 * age + offset
        cal = BiasCorrection("cohort").fit(age, prediction, cohort)
        np.testing.assert_allclose(cal.apply(prediction, age, cohort), 0.0, atol=1e-10)
        digest = joblib.hash(cal)
        cal.apply(prediction[:5], age[:5] + 100, cohort[:5])
        self.assertEqual(digest, joblib.hash(cal))

    def test_ridge_pipeline_fusion_report_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            np.savez_compressed(prepared / "paired_raw_private.npz", **self.raw)
            self.meta.to_csv(prepared / "paired_metadata_private.csv", index=False)
            (prepared / "manifest.json").write_text('{"status":"complete"}')
            target = root / "run"
            run(prepared, target, self.settings)
            result = audit(target)
            self.assertEqual(result["completed_tasks"], 18)
            report(target, root / "report")
            self.assertTrue((root / "report/index.html").exists())
            # Resume uses exactly the same data/settings/code and does not refit completed tasks.
            before = (
                (target / "fold1/none/DWI/ridge/model_private.joblib")
                .stat()
                .st_mtime_ns
            )
            run(prepared, target, self.settings, resume=True)
            self.assertEqual(
                before,
                (target / "fold1/none/DWI/ridge/model_private.joblib")
                .stat()
                .st_mtime_ns,
            )
            changed = dict(self.settings, seed=99)
            with self.assertRaises(ValueError):
                run(prepared, target, changed, resume=True)

    def test_signed_graph_residual_and_gine_task(self):
        import torch
        from .gnn import graphs, GINE
        from torch_geometric.data import Batch
        from .engine import evaluate_task

        indices = np.r_[0:20, 30:50]
        prep = MultiPreprocessor("none").fit(
            {m: x[indices] for m, x in self.raw.items()},
            self.meta.batch.iloc[indices],
            self.meta.age.iloc[indices],
        )
        raw = {m: x[:2] for m, x in self.raw.items()}
        values = prep.graph_values(raw, self.meta.batch.iloc[:2])
        items = graphs(raw, values, 0.5, prep.transforms)
        self.assertEqual(items[0].x.shape, (84, 168))
        self.assertEqual(items[0].edge_attr.shape[1], 2)
        self.assertTrue((items[0].edge_attr[:, 1] < 0).any())
        model = GINE(168, 2, 2, 32, True)
        batch = Batch.from_data_list(items)
        loss = model(batch).square().mean()
        loss.backward()
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        plan = make_plan(self.meta, 3, 2, 42)
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_task(
                self.raw,
                self.meta,
                plan["folds"][0],
                "gine",
                "age_mean",
                self.settings,
                Path(tmp),
                42,
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue((Path(tmp) / "checkpoint_private.pt").exists())
            history = pd.read_csv(Path(tmp) / "learning_history.csv")
            self.assertIn("validation_mse_z", history)

    def test_verified_clinical_schema_and_clustered_models(self):
        from .clinical import clinical

        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            task = run / "fold1/age_mean/DWI/ridge"
            task.mkdir(parents=True)
            (run / "manifest.json").write_text('{"status":"complete","settings":{}}')
            prediction = self.meta.copy()
            prediction["predicted_age"] = prediction.age + rng.normal(
                0, 3, len(prediction)
            )
            prediction["cBAG"] = prediction.predicted_age - prediction.age
            prediction.to_csv(task / "predictions_private.csv", index=False)
            source = self.meta[["cohort", "scan_id", "sex"]].copy()
            source["diagnosis"] = np.tile(["CN", "MCI", "AD"], 20)
            source["apoe"] = np.tile([0, 0, 1, 1], 15)
            source["diabetes"] = np.tile([0, 1, 0, 0, 1], 12)
            source["hba"] = rng.uniform(5, 8, len(source))
            source.to_csv(root / "clinical.csv", index=False)
            variables = {
                "cognitive_status": dict(
                    column="diagnosis",
                    kind="categorical",
                    levels=["CN", "MCI", "AD"],
                    mapping={x: x for x in ["CN", "MCI", "AD"]},
                    verified=True,
                ),
                "APOE4": dict(
                    column="apoe",
                    kind="categorical",
                    levels=["no", "yes"],
                    mapping={"0": "no", "1": "yes"},
                    verified=True,
                ),
                "sex": dict(
                    column="sex",
                    kind="categorical",
                    levels=["F", "M"],
                    mapping={"F": "F", "M": "M"},
                    verified=True,
                ),
                "T2D": dict(
                    column="diabetes",
                    kind="categorical",
                    levels=["no", "yes"],
                    mapping={"0": "no", "1": "yes"},
                    verified=True,
                    type2_definition_verified=True,
                ),
                "HbA1c": dict(
                    column="hba", kind="continuous", unit="percent", verified=True
                ),
            }
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    dict(
                        source_csv=str(root / "clinical.csv"),
                        source_verified=True,
                        visit_alignment_verified=True,
                        variables=variables,
                    )
                )
            )
            result = clinical(run, spec, root / "out")
            self.assertEqual(result["status"], "complete_exploratory")
            results = pd.read_csv(root / "out/associations.csv")
            self.assertIn("q_BH_all_exploratory_tests", results)
            self.assertIn("chronological_age", results.target.to_list())
            self.assertIn("HbA1c", results.target.to_list())

    def test_clinical_gate(self):
        from .clinical import clinical

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "spec.json"
            spec.write_text('{"source_verified":false}')
            result = clinical(root, spec, root / "output")
            self.assertEqual(result["status"], "blocked_by_unverified_source")


if __name__ == "__main__":
    unittest.main()
