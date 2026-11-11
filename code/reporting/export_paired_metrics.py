#!/usr/bin/env python3
"""Export only cohort aggregates from completed, audited paired Ridge/GINE runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from brainage_pipeline.reporting import bootstrap_metrics
from run_foldwise_DWI_Ridge import metrics


def export(ridge_run, gine_run, output, replicates=3000):
    output = Path(output)
    source = []
    references = {}
    rows = []
    preparation = None
    plan_reference = None
    for learner, run in [('ridge', Path(ridge_run)), ('gine', Path(gine_run))]:
        manifest = json.loads((run / 'manifest.json').read_text())
        audit = json.loads((run / 'audit.json').read_text())
        if (manifest['status'] != 'complete' or manifest['population'] != 'paired'
                or manifest['settings'].get('validation_only', False)
                or audit['status'] != 'passed'):
            raise ValueError('Require complete, audited, nonpilot paired runs')
        if manifest.get('training_fraction', 1) != 1:
            raise ValueError('Require full supported training populations')
        if learner == 'gine':
            grid = manifest['settings']['gine_grid']
            if not grid or any(c['layers'] != 2 or c['width'] != 32
                               or c.get('residual', False) for c in grid):
                raise ValueError('This figure requires the fixed GINE 2x32 reference arm')
        fingerprint = manifest['prepared_manifest_sha256']
        plan = json.loads((run / 'fold_plan_private.json').read_text())
        if preparation is not None and (fingerprint != preparation or plan != plan_reference):
            raise ValueError('Different prepared data or participant fold plans')
        preparation, plan_reference = fingerprint, plan
        blocks = []
        paths = sorted(run.glob(f'fold*/*/*/{learner}/predictions_private.csv'))
        for path in paths:
            d = pd.read_csv(path)
            d = d.loc[d.is_cn].copy()
            d['fold'] = int(path.parts[-5].removeprefix('fold'))
            d['mode'], d['modality'] = path.parts[-4:-2]
            blocks.append(d)
        if len(paths) != 30:
            raise ValueError('Expected five folds x two modes x three modalities')
        predictions = pd.concat(blocks, ignore_index=True)
        for (mode, modality), part in predictions.groupby(['mode', 'modality']):
            if mode not in {'none', 'age_mean'} or modality not in {'DWI', 'fMRI', 'fusion'}:
                raise ValueError('Unexpected comparison condition')
            if part.row_index.duplicated().any():
                raise ValueError('Repeated evaluated observations')
            columns = ['fold', 'row_index', 'cohort', 'participant_group', 'age']
            alignment = part.sort_values(['fold', 'row_index'])[columns].reset_index(drop=True)
            if references:
                pd.testing.assert_frame_equal(next(iter(references.values())), alignment)
            references[(learner, mode, modality)] = alignment
            model = {'DWI': 'DWI', 'fMRI': 'fMRI', 'fusion': 'EARLY_FUSION'}[modality]
            if learner == 'gine':
                model += '_GINE_2X32'
            for cohort, d in [('Pooled', part)] + list(part.groupby('cohort')):
                rows.append(dict(mode=mode, modality=modality, learner=learner,
                                 cohort=cohort, model=model,
                                 gine_arm='gine_2x32' if learner == 'gine' else '',
                                 n_participants=d.participant_group.nunique(),
                                 **metrics(d.age.to_numpy(), d.predicted_age.to_numpy()),
                                 **bootstrap_metrics(d, replicates)))
        source.append(dict(learner=learner, run=run.name, audit_status=audit['status'],
                           manifest_sha256=hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest(),
                           prediction_files={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in paths}))
    reference = next(iter(references.values()))
    if not np.isfinite(reference.age).all():
        raise ValueError('Invalid reference ages')
    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'paired_metrics_six_models.csv', index=False)
    provenance = dict(
        population='paired CN outer OOF', n_scans=len(reference),
        n_participants=reference.participant_group.nunique(),
        gine_arm='fixed 2x32 reference, not selected from outer scores',
        default_mode='age_mean', alternate_mode='none', sources=source,
        confidence_intervals=f'{replicates} cohort-stratified participant bootstrap draws; fixed OOF models, no refitting',
        checks=['complete audited nonpilot full-training runs',
                'identical prepared fingerprint and participant fold plan',
                'identical CN observations across all models, modalities and modes',
                'metrics recomputed from original held-out predictions'],
        limitations=['Development reanalysis, not external validation',
                     'HABS has only six CN scans in the current paired data',
                     'Clinical labels, dates, protocols and ROI order remain unverified',
                     'Restricted paired population differs from historical 387-scan benchmark'])
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Exported {len(result)} aggregate rows; no participant records exported')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ridge-run', required=True, type=Path)
    parser.add_argument('--gine-run', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/current')
    parser.add_argument('--bootstrap', type=int, default=3000)
    args = parser.parse_args()
    export(args.ridge_run, args.gine_run, args.output, args.bootstrap)
