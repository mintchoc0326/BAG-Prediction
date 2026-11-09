#!/usr/bin/env python3
"""Controlled DWI GINE capacity diagnostics on existing harmonized data.

This study inherits global-ComBat leakage and the previously selected 50% topology.
It diagnoses optimization/capacity; it is NOT an unbiased model-selection benchmark.
Outputs are isolated from the original GNN_BAG outputs. Participant-level files
stay under ignored results/. Default: five grouped folds, three architectures.
"""
import argparse
import copy
import hashlib
import html
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
import GNN_DWI_BAG as base

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = [('GINE_2x32', 2, 32), ('GINE_3x32', 3, 32), ('GINE_3x64', 3, 64)]
LIMITATION = ('Exploratory capacity diagnostic: upstream global ComBat and prior '
              'OOF-selected 50% topology are reused; not leakage-free generalization. '
              'CN labels retain the existing HABS override; participant grouping '
              'uses the existing visit-ID parser pending independent validation.')


class CapacityGINE(nn.Module):
    def __init__(self, layers, width, node_dim=84):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            input_dim = node_dim if layer == 0 else width
            mlp = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, width))
            self.convs.append(GINEConv(mlp, edge_dim=1))
            self.norms.append(nn.BatchNorm1d(width))
        self.dropout = nn.Dropout(base.DROPOUT)
        self.regressor = nn.Sequential(nn.Linear(width, width//2), nn.ReLU(),
                                       nn.Dropout(base.DROPOUT), nn.Linear(width//2, 1))

    def forward(self, data):
        x = data.x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = F.relu(norm(conv(x, data.edge_index, data.edge_attr)))
            if i < len(self.convs)-1:
                x = self.dropout(x)
        return self.regressor(global_mean_pool(x, data.batch)).squeeze(-1)


def evaluate(model, graphs, mean, std):
    prediction = base.predict_standardized(model, graphs)*std + mean
    age = np.array([float(g.y.item())*std+mean for g in graphs])
    return prediction, float(mean_absolute_error(age, prediction)), float(mean_squared_error(age, prediction))


def fit(graphs, train, validation, name, layers, width, fold, args):
    base.set_seed(args.seed+fold)
    age = np.array([float(graphs[i].y.item()) for i in train])
    mean, std = float(age.mean()), max(float(age.std()), 1e-8)
    train_graphs = base.scale_age_targets(train, graphs, mean, std)
    val_graphs = base.scale_age_targets(validation, graphs, mean, std)
    loader = DataLoader(train_graphs, batch_size=base.BATCH_SIZE, shuffle=True)
    model = CapacityGINE(layers, width).to(base.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    best, best_epoch, state = np.inf, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.max_epochs+1):
        loss = base.train_one_epoch(model, loader, optimizer)
        _, train_mae, train_mse = evaluate(model, train_graphs, mean, std)
        _, val_mae, val_mse = evaluate(model, val_graphs, mean, std)
        history.append(dict(architecture=name, fold=fold, epoch=epoch,
                            optimization_mse_z=loss, train_mse_z=train_mse/std**2,
                            validation_mse_z=val_mse/std**2, train_mae_years=train_mae,
                            validation_mae_years=val_mae))
        pd.DataFrame(history).to_csv(args.output/f'{name}_fold{fold}_history.csv', index=False)
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f'{name} fold={fold} epoch={epoch} train={train_mae:.3f} val={val_mae:.3f}', flush=True)
        if epoch-best_epoch >= args.patience:
            break
    elapsed = time.perf_counter()-start
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, layers=layers, width=width, age_mean=mean,
                    age_std=std, best_epoch=best_epoch, limitation=LIMITATION),
               args.output/f'{name}_fold{fold}.pt')
    record = dict(architecture=name, fold=fold, layers=layers, width=width,
                  parameters=sum(p.numel() for p in model.parameters()),
                  n_train=len(train), n_validation=len(validation), best_epoch=best_epoch,
                  epochs_run=len(history), best_validation_mae=best,
                  train_mae_at_best=history[best_epoch-1]['train_mae_years'],
                  generalization_gap_at_best=best-history[best_epoch-1]['train_mae_years'],
                  fit_seconds=elapsed)
    return model, mean, std, record


def plot_histories(output, histories, loss=False):
    folds = sorted(histories.fold.unique())
    train_column = 'train_mse_z' if loss else 'train_mae_years'
    val_column = 'validation_mse_z' if loss else 'validation_mae_years'
    fig, axes = plt.subplots(len(folds), 3, figsize=(13, 3.1*len(folds)), squeeze=False)
    for column, (name, _, _) in enumerate(ARCHITECTURES):
        for row, fold in enumerate(folds):
            ax = axes[row, column]
            h = histories[(histories.architecture == name) & (histories.fold == fold)]
            ax.plot(h.epoch, h[train_column], color='#4878A8', label='Training (eval mode)')
            ax.plot(h.epoch, h[val_column], color='#BC5B58', label='Validation')
            best_epoch = int(h.loc[h.validation_mae_years.idxmin(), 'epoch'])
            ax.axvline(best_epoch, color='#999999', lw=.7)
            ax.set_title(f'{name} | fold {fold}', fontsize=10)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (standardized age)' if loss else 'MAE (years)')
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        stem = 'learning_curves_loss' if loss else 'learning_curves'
        fig.savefig(output/f'{stem}.{ext}', dpi=180)
    plt.close(fig)


def render_report(output):
    histories = pd.concat([pd.read_csv(p) for p in output.glob('*_history.csv')], ignore_index=True)
    plot_histories(output, histories)
    plot_histories(output, histories, loss=True)
    metrics = pd.read_csv(output/'fold_metrics.csv')
    summary = metrics.groupby('architecture', sort=False).agg(
        parameters=('parameters','first'), folds_completed=('fold','count'),
        mean_validation_mae=('best_validation_mae','mean'), mean_test_mae=('test_mae_years','mean'),
        sd_test_mae=('test_mae_years','std'), mean_gap=('generalization_gap_at_best','mean'),
        mean_fit_seconds=('fit_seconds','mean'), median_best_epoch=('best_epoch','median'))
    summary.to_csv(output/'capacity_summary.csv')
    css = '<style>body{font:15px Arial,sans-serif;max-width:1300px;margin:30px auto;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style>'
    text = ('<!doctype html><meta charset="utf-8">'+css+'<h1>DWI GINE capacity diagnostics</h1>'
            +'<p>'+html.escape(LIMITATION)+'</p>'
            +'<p>All models use identical grouped train/validation/test splits and fixed 50% positive-edge topology. '
            'Early stopping uses validation MAE; outer test scores are descriptive and do not select a winning architecture. '
            'Mean test MAE is the unweighted mean across folds; fold SD is not a confidence interval. '
            'One seed is used; reruns with independent seeds are needed to assess initialization variability.</p>'
            +'<h2>Capacity and fold summaries</h2>'+summary.to_html(float_format=lambda x:f'{x:.3f}')
            +'<h2>Training and validation MAE</h2><p>Both curves use evaluation mode. The vertical line marks the epoch selected by validation MAE. Raw curves are not smoothed.</p><img src="learning_curves.svg">'
            +'<h2>Training and validation loss</h2><p>Evaluation-mode MSE uses training-only standardized age targets; early stopping remains based on MAE.</p><img src="learning_curves_loss.svg">'
            +'<h2>Per-fold records</h2>'+metrics.to_html(index=False,float_format=lambda x:f'{x:.3f}'))
    (output/'capacity_report.html').write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/final/DWI/GINE_capacity_study')
    parser.add_argument('--max-epochs', type=int, default=250)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--render-only', action='store_true', help='Rebuild plots/report from completed histories')
    parser.add_argument('--first-fold-only', action='store_true', help='Pilot only; not full CV')
    args = parser.parse_args()
    if args.render_only:
        manifest = json.loads((args.output/'run_manifest.json').read_text())
        if manifest.get('status') != 'complete':
            parser.error('Run is incomplete; wait for training to finish before rendering.')
        print(render_report(args.output).to_string())
        return
    if args.max_epochs < 1 or args.patience < 1 or args.folds < 2:
        parser.error('Epochs/patience must be positive and folds >= 2')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'run_manifest.json').exists():
        parser.error('Output already contains a run; choose a new --output directory')
    torch.set_num_threads(4)
    print(LIMITATION, flush=True)
    aligned, dataset_col, id_col, path_col, metadata_path = base.load_inputs()
    aligned = aligned[aligned['_is_cn']].reset_index(drop=True)
    graphs = [base.matrix_to_graph(base.load_matrix(Path(path)), float(age), 50)[0]
              for path, age in zip(aligned[path_col], aligned['_age'])]
    groups = aligned['_participant_group'].to_numpy()
    if len(np.unique(groups)) < args.folds:
        raise ValueError('Not enough participant groups for requested folds')
    splits, split_rows = [], []
    for fold, (outer_train, test) in enumerate(GroupKFold(args.folds).split(np.arange(len(graphs)), groups=groups), 1):
        inner_train, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed+fold).split(outer_train, groups=groups[outer_train]))
        train, validation = outer_train[inner_train], outer_train[inner_val]
        for a, b in [(train, validation), (train, test), (validation, test)]:
            assert set(groups[a]).isdisjoint(set(groups[b]))
        splits.append((fold, train, validation, test))
        for role, indices in [('train', train), ('validation', validation), ('test', test)]:
            for index in indices:
                split_rows.append(dict(fold=fold, role=role, row_index=int(index), participant_group=groups[index],
                                       dataset=aligned.iloc[index][dataset_col], scan_id=aligned.iloc[index][id_col]))
        if args.first_fold_only:
            break
    pd.DataFrame(split_rows).to_csv(args.output/'split_manifest_private.csv', index=False)
    manifest = dict(limitation=LIMITATION, seed=args.seed, folds=args.folds, first_fold_only=args.first_fold_only,
                    max_epochs=args.max_epochs, patience=args.patience, threshold_percent=50,
                    architectures=ARCHITECTURES, device=str(base.DEVICE), torch_version=torch.__version__,
                    n_scans=len(graphs), n_participants=len(np.unique(groups)),
                    learning_rate=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
                    dropout=base.DROPOUT, batch_size=base.BATCH_SIZE,
                    metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status='running')
    (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    records = []
    for fold, train, validation, test in splits:
        for name, layers, width in ARCHITECTURES:
            model, mean, std, record = fit(graphs, train, validation, name, layers, width, fold, args)
            test_graphs = base.scale_age_targets(test, graphs, mean, std)
            prediction, mae, mse = evaluate(model, test_graphs, mean, std)
            record.update(test_mae_years=mae, test_rmse_years=float(np.sqrt(mse)), n_test=len(test))
            records.append(record)
            pd.DataFrame(records).to_csv(args.output/'fold_metrics.csv', index=False)
            pd.DataFrame(dict(row_index=test, predicted_age=prediction)).to_csv(args.output/f'{name}_fold{fold}_predictions_private.csv', index=False)
            histories = pd.concat([pd.read_csv(p) for p in args.output.glob('*_history.csv')], ignore_index=True)
            # Plot only when all architectures of the current fold are available.
            if name == ARCHITECTURES[-1][0]:
                plot_histories(args.output, histories)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = pd.DataFrame(records).groupby('architecture', sort=False).agg(
        parameters=('parameters', 'first'), folds_completed=('fold', 'count'),
        mean_validation_mae=('best_validation_mae', 'mean'), mean_test_mae=('test_mae_years', 'mean'),
        sd_test_mae=('test_mae_years', 'std'), mean_gap=('generalization_gap_at_best', 'mean'),
        mean_fit_seconds=('fit_seconds', 'mean'), median_best_epoch=('best_epoch', 'median'))
    summary.to_csv(args.output/'capacity_summary.csv')
    manifest['status'] = 'complete'
    (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    render_report(args.output)
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
