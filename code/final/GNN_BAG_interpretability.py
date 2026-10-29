#!/usr/bin/env python3
"""OOF input attribution for the existing matched-sample fusion GNN.

Run: python code/GNN_BAG_interpretability.py
Requires the environment used by GNN_DWI_fMRI_fusion.py. Loads exact saved
outer-fold checkpoints, including training-only normalizers. No model is
retrained. Run the updated training script first to create outer_fold_models/.

Integrated gradients (IG) jointly attributes standardized node features and
normalized edge attributes. Baseline: training-mean node features (z=0), with
observed diagonal held fixed, and zero edge attributes. Topology is fixed:
this is representation attribution, NOT attribution through top-k selection,
log1p or per-graph max normalization. Signed contributions are in age years;
for fixed chronological age they also explain the BAG prediction difference
from the reference. They are not causal effects or clinical associations.

Outputs: per-scan compressed attribution arrays; connection/ROI/modality
summaries with equal participant weighting; fold metrics, prediction checks,
IG completeness residuals, and modality reference-occlusion results. ROI
numbers are zero-based matrix indices. Names use the existing atlas audit
table when available; its per-scan verification limitation is retained.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

import GNN_DWI_fMRI_fusion as gnn

ROOT = Path(__file__).resolve().parents[2]
UPPER = np.triu_indices(84, 1)
MODALITIES = ('DWI', 'fMRI')


def reference(graph):
    base = graph.clone()
    base.x = torch.zeros_like(graph.x)
    diagonal = torch.arange(84, device=graph.x.device)
    base.x[diagonal, diagonal] = graph.x[diagonal, diagonal]
    base.edge_attr = torch.zeros_like(graph.edge_attr)
    return base


def age_output(model, pair, age_mean, age_std):
    return model(*pair).reshape(()) * age_std + age_mean


def integrated_gradients(model, pair, mean, std, steps):
    """Joint straight-line IG, trapezoidal quadrature; no parameter gradients."""
    model.eval()
    bases = [reference(g) for g in pair]
    observed = [v for g in pair for v in (g.x, g.edge_attr)]
    baseline = [v for g in bases for v in (g.x, g.edge_attr)]
    differences = [x - b for x, b in zip(observed, baseline)]
    totals = [torch.zeros_like(x) for x in observed]
    for k in range(steps + 1):
        alpha = k / steps
        values = [(b + alpha * d).detach().requires_grad_(True)
                  for b, d in zip(baseline, differences)]
        interpolated = [g.clone() for g in pair]
        for m, g in enumerate(interpolated):
            g.x, g.edge_attr = values[2*m:2*m+2]
        output = age_output(model, interpolated, mean, std)
        grads = torch.autograd.grad(output, values, allow_unused=True)
        weight = 0.5 if k in (0, steps) else 1.0
        for total, grad in zip(totals, grads):
            if grad is not None:
                total.add_(grad.detach(), alpha=weight / steps)
    attrs = [(d * t).detach().cpu().numpy() for d, t in zip(differences, totals)]
    with torch.no_grad():
        prediction = float(age_output(model, pair, mean, std))
        base_prediction = float(age_output(model, bases, mean, std))
    residual = prediction - base_prediction - sum(float(a.sum()) for a in attrs)
    return attrs, bases, prediction, base_prediction, residual


def connection_channels(graph, node_attr, edge_attr):
    # Combine both node-feature appearances and both directed edge appearances.
    node = node_attr[UPPER] + node_attr.T[UPPER]
    edge = np.zeros((84, 84), dtype=float)
    src, dst = graph.edge_index.detach().cpu().numpy()
    np.add.at(edge, (src, dst), edge_attr.ravel())
    return node, edge[UPPER] + edge.T[UPPER]


def summarize(arrays, metadata, out):
    """First average visits within participant, then summarize participants."""
    groups = metadata.participant_group.to_numpy()
    people = np.unique(groups)
    person = np.stack([arrays[groups == p].mean(axis=0) for p in people])
    # arrays: scans x modalities x 3 channels(node, edge, combined) x connections
    rows = []
    roi_values = np.zeros((len(person), 2, 84))
    for m, modality in enumerate(MODALITIES):
        for c, channel in enumerate(('node_feature', 'edge_attribute', 'combined')):
            values = person[:, m, c]
            for j, (a, b) in enumerate(zip(*UPPER)):
                rows.append(dict(modality=modality, channel=channel, roi_i=int(a), roi_j=int(b),
                                 mean_signed_years=float(values[:, j].mean()),
                                 mean_absolute_years=float(abs(values[:, j]).mean()),
                                 n_participants=len(people)))
        # Allocate half of each connection's absolute attribution to each endpoint.
        for j, (a, b) in enumerate(zip(*UPPER)):
            strength = abs(person[:, m, 2, j]) / 2
            roi_values[:, m, a] += strength
            roi_values[:, m, b] += strength
    pd.DataFrame(rows).sort_values('mean_absolute_years', ascending=False).to_csv(out / 'connection_importance.csv', index=False)
    roi_rows = [dict(modality=modality, roi_index=i,
                     mean_incident_absolute_years=float(roi_values[:, m, i].mean()),
                     n_participants=len(people))
                for m, modality in enumerate(MODALITIES) for i in range(84)]
    pd.DataFrame(roi_rows).sort_values('mean_incident_absolute_years', ascending=False).to_csv(out / 'roi_importance.csv', index=False)
    return person


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'results/final/Fusion/Weighted_Ridge')
    parser.add_argument('--gnn-dir', type=Path, default=ROOT / 'results/final/Fusion/GNN_fusion')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/final/Fusion/Attribution')
    parser.add_argument('--mode', choices=gnn.MODES, default='GNN_FUSION')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--ig-steps', type=int, default=64)
    parser.add_argument('--ig-max-steps', type=int, default=512)
    parser.add_argument('--ig-tolerance-years', type=float, default=0.05)
    parser.add_argument('--limit-per-fold', type=int, default=None, help='Smoke test only; metrics then cover a subset.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    # Interactive kernels supply their own -f connection-file argument.
    # Explicit main([...]) arguments and ordinary CLI validation stay strict.
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)
    if args.ig_steps < 2 or not set(args.folds) <= set(range(1, 6)) or len(set(args.folds)) != len(args.folds):
        parser.error('Need >=2 IG steps and distinct fold IDs between 1 and 5')
    if args.ig_max_steps < args.ig_steps or args.ig_tolerance_years <= 0:
        parser.error('IG max steps must be >= initial steps and tolerance must be positive')
    if args.limit_per_fold is not None and args.limit_per_fold < 1:
        parser.error('--limit-per-fold must be positive')
    if args.device != 'auto':
        gnn.DEVICE = torch.device(args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cn = pd.read_csv(args.baseline_dir / 'CN_OOF_predictions.csv')
    for fold in args.folds:
        path = args.gnn_dir / 'outer_fold_models' / f'{args.mode}_fold_{fold}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'{path} missing. Run updated GNN_DWI_fMRI_fusion.py first; old final/reconstructed models cannot substitute for these checkpoints.')
    expected = pd.read_csv(args.gnn_dir / 'CN_OOF_predictions.csv')
    keys = ['dataset', 'scan_id']
    if cn.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError('Duplicate scan keys')
    if cn.groupby('participant_group').FOLD.nunique().max() != 1 or set(cn.FOLD) != set(range(1, 6)):
        raise ValueError('Invalid participant-grouped folds')
    assignments = pd.read_csv(args.gnn_dir / 'outer_fold_assignments.csv')
    check = cn[keys + ['FOLD', 'unified_age', 'participant_group']].merge(
        assignments, on=keys, suffixes=('_input', '_saved'), validate='one_to_one')
    if len(check) != len(cn) or len(assignments) != len(cn):
        raise ValueError('Saved GNN samples differ from input')
    for field in ['FOLD', 'unified_age', 'participant_group']:
        if not (check[field + '_input'] == check[field + '_saved']).all():
            raise ValueError('Saved GNN assignment mismatch: ' + field)
    y = cn.unified_age.to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite ages')
    graphs = [(gnn.graph_from_edges(gnn.features(r.connectome_dwi, 'DWI'), True),
               gnn.graph_from_edges(gnn.features(r.connectome_fmri, 'fMRI'), False)) for r in cn.itertuples()]
    fingerprint = gnn.sample_graph_fingerprint(cn, graphs)
    sample_rows, modality_rows, fold_rows, all_channels = [], [], [], []
    for fold in args.folds:
        train = np.flatnonzero(cn.FOLD.to_numpy() != fold)
        test = np.flatnonzero(cn.FOLD.to_numpy() == fold)
        if set(cn.iloc[train].participant_group) & set(cn.iloc[test].participant_group):
            raise ValueError('Participant leakage')
        path = args.gnn_dir / 'outer_fold_models' / f'{args.mode}_fold_{fold}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint.get('schema_version') != 1 or checkpoint['mode'] != args.mode
                or checkpoint['fold'] != fold or checkpoint['input_fingerprint'] != fingerprint
                or checkpoint['train_indices'] != train.tolist() or checkpoint['test_indices'] != test.tolist()):
            raise ValueError(f'Checkpoint/sample/fold mismatch: {path}')
        if (checkpoint['hidden'] != gnn.HIDDEN or checkpoint['dropout'] != gnn.DROPOUT
                or checkpoint['retain_positive_percent'] != gnn.RETAIN_PERCENT):
            raise ValueError('Checkpoint architecture/preprocessing differs from current implementation')
        n_epochs = checkpoint['epochs']
        stats = checkpoint['node_normalizers']
        mean, std = checkpoint['age_mean'], checkpoint['age_std']
        model = gnn.FusionGNN(args.mode).to(gnn.DEVICE)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        print(f'Loaded fold {fold}, mode={args.mode}, epochs={n_epochs}, device={gnn.DEVICE}', flush=True)
        # Verify exact checkpoint predictions against both model inference and saved OOF CSV.
        checkpoint_predictions = checkpoint['test_predictions'].numpy()
        reproduced = gnn.predict(model, gnn.loader(graphs, y, test, stats), mean, std)
        saved = cn.iloc[test][keys].merge(expected[keys + [f'PREDICTED_AGE_{args.mode}_OOF']],
                                        on=keys, validate='one_to_one')
        csv_predictions = saved[f'PREDICTED_AGE_{args.mode}_OOF'].to_numpy()
        if (len(saved) != len(test) or not np.allclose(checkpoint_predictions, csv_predictions, atol=1e-4, rtol=0)
                or not np.allclose(reproduced, checkpoint_predictions, atol=1e-3, rtol=0)):
            raise ValueError('Saved model predictions do not match checkpoint/OOF CSV. Use files from the same training run and check environment.')
        if args.limit_per_fold:
            test = test[:args.limit_per_fold]
        predictions = []
        for index in test:
            row = cn.iloc[index]
            pair = []
            for m in range(2):
                graph = graphs[index][m].clone()
                graph.x = (graph.x - stats[m][0]) / stats[m][1]
                pair.append(Batch.from_data_list([graph]).to(gnn.DEVICE))
            steps_used = args.ig_steps
            while True:
                attrs, bases, prediction, base_prediction, residual = integrated_gradients(model, pair, mean, std, steps_used)
                if abs(residual) <= args.ig_tolerance_years or steps_used >= args.ig_max_steps:
                    break
                steps_used = min(steps_used * 2, args.ig_max_steps)
            if not np.isfinite(prediction) or not all(np.isfinite(a).all() for a in attrs):
                raise ValueError('Nonfinite prediction or attribution')
            channels = []
            key = dict(dataset=row.dataset, scan_id=row.scan_id, participant_group=row.participant_group, fold=fold)
            for m, modality in enumerate(MODALITIES):
                node, edge = connection_channels(pair[m], attrs[2*m], attrs[2*m+1])
                combined = node + edge
                channels.append(np.stack([node, edge, combined]))
                occluded = list(pair)
                occluded[m] = bases[m]
                with torch.no_grad():
                    occluded_prediction = float(age_output(model, occluded, mean, std))
                modality_rows.append(dict(**key, modality=modality,
                    signed_IG_years=float(combined.sum()), absolute_connection_IG_years=float(abs(combined).sum()),
                    prediction_minus_reference_occlusion_years=prediction-occluded_prediction,
                    absolute_error_increase_years=abs(occluded_prediction-y[index])-abs(prediction-y[index])))
            all_channels.append(np.stack(channels))
            predictions.append(prediction)
            sample_rows.append(dict(**key, chronological_age=y[index], predicted_age=prediction,
                                   BAG=prediction-y[index], reference_predicted_age=base_prediction,
                                   IG_steps=steps_used, IG_quality_pass=abs(residual) <= args.ig_tolerance_years,
                                   IG_completeness_residual_years=residual))
            safe = hashlib.sha256(f'{row.dataset}/{row.scan_id}'.encode()).hexdigest()[:16]
            np.savez_compressed(out / f'fold{fold}_{safe}_attribution.npz',
                node_DWI=attrs[0], edge_DWI=attrs[1], node_fMRI=attrs[2], edge_fMRI=attrs[3],
                edge_index_DWI=pair[0].edge_index.cpu().numpy(), edge_index_fMRI=pair[1].edge_index.cpu().numpy())
            print(f'  {row.dataset}/{row.scan_id}: IG residual={residual:.4g} years', flush=True)
        fold_rows.append(dict(fold=fold, model=args.mode, selected_epochs=n_epochs, n_test=len(test), **gnn.metrics(y[test], predictions)))
        del model
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(out / 'scan_attribution_quality.csv', index=False)
    # Individual-graph inference should agree with saved batched OOF inference.
    column = f'PREDICTED_AGE_{args.mode}_OOF'
    comparison = samples.merge(expected[keys + [column]], on=keys, validate='one_to_one')
    comparison['prediction_difference_years'] = comparison.predicted_age - comparison[column]
    comparison.to_csv(out / 'saved_prediction_check.csv', index=False)
    max_difference = float(comparison.prediction_difference_years.abs().max())
    if max_difference > 0.001:
        raise ValueError(f'Individual-graph predictions differ from saved OOF by {max_difference:.6f} years')
    pd.DataFrame(fold_rows).to_csv(out / 'fold_metrics.csv', index=False)
    arrays = np.stack(all_channels)
    summarize(arrays, samples, out)
    for fold in args.folds:
        fold_out = out / f'fold_{fold}'
        fold_out.mkdir(exist_ok=True)
        mask = samples.fold.eq(fold).to_numpy()
        summarize(arrays[mask], samples.loc[mask], fold_out)
    modality = pd.DataFrame(modality_rows)
    modality.to_csv(out / 'modality_per_scan.csv', index=False)
    values = ['signed_IG_years', 'absolute_connection_IG_years', 'prediction_minus_reference_occlusion_years', 'absolute_error_increase_years']
    participant = modality.groupby(['participant_group', 'modality'])[values].mean().reset_index()
    participant.to_csv(out / 'modality_per_participant.csv', index=False)
    modality_summary = participant.groupby('modality')[values].agg(['mean', 'std'])
    modality_summary.columns = ['_'.join(c) for c in modality_summary.columns]
    modality_summary.to_csv(out / 'modality_summary.csv')
    folds = []
    for fold in args.folds:
        table = pd.read_csv(out / f'fold_{fold}/roi_importance.csv')
        table['fold'] = fold
        table['rank'] = table.groupby('modality').mean_incident_absolute_years.rank(ascending=False, method='min')
        folds.append(table)
    # Label only using the existing audited table; retain its verification scope.
    mapping_path = ROOT / 'results/final/Fusion/roi_order_audit/roi_order_mapping.csv'
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        if len(mapping) != 84 or set(mapping.matrix_position_1based) != set(range(1, 85)):
            raise ValueError('ROI mapping must contain exactly positions 1..84')
        mapping['roi_index'] = mapping.matrix_position_1based - 1
        mapping.to_csv(out / 'roi_labels_with_verification_scope.csv', index=False)
        names = mapping.set_index('roi_index').lookup_name.to_dict()
        for directory in [out] + [out / f'fold_{f}' for f in args.folds]:
            table = pd.read_csv(directory / 'connection_importance.csv')
            table['roi_i_name'] = table.roi_i.map(names)
            table['roi_j_name'] = table.roi_j.map(names)
            table.to_csv(directory / 'connection_importance.csv', index=False)
            table = pd.read_csv(directory / 'roi_importance.csv')
            table['roi_name'] = table.roi_index.map(names)
            table.to_csv(directory / 'roi_importance.csv', index=False)
    connection_folds = []
    for fold in args.folds:
        table = pd.read_csv(out / f'fold_{fold}/connection_importance.csv')
        table = table[table.channel.eq('combined')].copy()
        table['rank'] = table.groupby('modality').mean_absolute_years.rank(ascending=False, method='min')
        connection_folds.append(table)
    pd.concat(connection_folds).groupby(['modality', 'roi_i', 'roi_j']).agg(
        mean_rank=('rank', 'mean'), sd_rank=('rank', 'std'),
        folds_in_top100=('rank', lambda x: int((x <= 100).sum()))).to_csv(out / 'connection_fold_stability.csv')
    stability = pd.concat(folds)
    stability.groupby(['modality', 'roi_index']).agg(mean_rank=('rank', 'mean'), sd_rank=('rank', 'std'),
        folds_in_top10=('rank', lambda x: int((x <= 10).sum()))).to_csv(out / 'roi_fold_stability.csv')
    summary = dict(mode=args.mode, folds=args.folds, scans=len(samples), participants=samples.participant_group.nunique(),
        ig_steps=args.ig_steps, ig_max_steps=args.ig_max_steps,
        ig_tolerance_years=args.ig_tolerance_years, n_IG_quality_failures=int((~samples.IG_quality_pass).sum()), device=str(gnn.DEVICE), subset_run=args.limit_per_fold is not None,
        max_abs_IG_residual_years=float(samples.IG_completeness_residual_years.abs().max()),
        interpretation_scope='Exact saved outer-fold checkpoints used for OOF evaluation; no retraining',
        max_abs_prediction_difference_years=float(comparison.prediction_difference_years.abs().max()),
        input_sha256=hashlib.sha256((args.baseline_dir / 'CN_OOF_predictions.csv').read_bytes()).hexdigest(),
        limitations=['Fixed-topology transformed-input IG; does not explain graph threshold decisions.',
            'Reference node features are training means with observed diagonal fixed; edge attributes are zero.',
            'Reference occlusion retains topology and may be out of distribution; no causal interpretation.',
            'Mean absolute attribution is computed after within-participant signed averaging across visits.',
            'ROI scores allocate half of each absolute combined connection attribution to each endpoint.',
            'Saved predictions are checked against checkpoint inference and the OOF CSV; tolerance 0.001 years.',
            'Same upstream ComBat and age-filtering limitations as original analysis; CN attribution only.',
            'Feature columns are connectivity to ROI j, not volume/thickness or named biological factors.'])
    (out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'METHODS.txt').write_text(__doc__ + '\n\n' + json.dumps(summary, indent=2))
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
