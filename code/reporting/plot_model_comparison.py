#!/usr/bin/env python3
"""Plot current participant-nested six-model cohort metrics (no training).

Run from the repository root:
    python code/reporting/plot_model_comparison.py
    python code/reporting/plot_model_comparison.py --no-ci
    python code/reporting/plot_model_comparison.py --before-after

Dependencies: numpy, pandas, matplotlib.
Edit MODEL_STYLES for colors/labels; FIGSIZE and FONT_SIZE for layout.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Model ID, legend label, color: blue, yellow, red, green, purple, orange.
MODEL_STYLES = [
    ('DWI', 'Ridge–DWI', '#4878A8'),
    ('fMRI', 'Ridge–fMRI', '#C6A83E'),
    ('EARLY_FUSION', 'Ridge–fusion (early)', '#BC5B58'),
    ('DWI_GINE_2X32', 'GINE–DWI (2×32)', '#59926C'),
    ('fMRI_GINE_2X32', 'GINE–fMRI (2×32)', '#8D74AA'),
    ('EARLY_FUSION_GINE_2X32', 'GINE–fusion (2×32)', '#D08B4B'),
]
METRICS = [('MAE', 'A. MAE (years)'), ('RMSE', 'B. RMSE (years)'),
           ('R2', 'C. R²'), ('Pearson_r', 'D. Pearson r')]
COHORT_ORDER = ['ADNI', 'ADRC', 'HABS']
FIGSIZE = (12, 8)
FONT_SIZE = 10
BAR_WIDTH = 0.125


def plot_comparison(data, output, show_ci=True, mode='age_mean'):
    data = data.loc[data['mode'].eq(mode)].copy()
    if data.empty:
        raise ValueError(f'No aggregate results for mode {mode}')
    data = data.rename(columns={'cohort': 'dataset', **{
        metric + '_CI_' + end: metric + '_ci_' + end
        for metric, _ in METRICS for end in ['low', 'high']}})
    cohorts = [c for c in COHORT_ORDER if c in data['dataset'].unique()]
    if not cohorts:
        raise ValueError('No recognized cohorts in input CSV.')
    indexed = data.set_index(['dataset', 'model'])
    if indexed.index.has_duplicates:
        raise ValueError('Expected one row per dataset/model.')
    for cohort in cohorts:
        rows = indexed.loc[[(cohort, model) for model, _, _ in MODEL_STYLES]]
        if rows.n_scans.nunique() != 1 or rows.n_participants.nunique() != 1:
            raise ValueError('Models must use identical matched cohort counts.')
    with plt.rc_context({'font.size': FONT_SIZE, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                         'axes.linewidth': .6, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
        x = np.arange(len(cohorts))
        for ax, (metric, title) in zip(axes.flat, METRICS):
            for j, (model, label, color) in enumerate(MODEL_STYLES):
                rows = indexed.loc[[(c, model) for c in cohorts]]
                positions = x + (j - (len(MODEL_STYLES)-1)/2) * BAR_WIDTH
                # Solid fills, without hatching or dashed outlines.
                ax.bar(positions, rows[metric], width=BAR_WIDTH*.94,
                       color=color, edgecolor='white', linewidth=.4)
                if show_ci:
                    low, high = rows[metric+'_ci_low'], rows[metric+'_ci_high']
                    ax.vlines(positions, low, high, color='#333333', lw=.8)
                    ax.hlines(low, positions-.018, positions+.018, color='#333333', lw=.8)
                    ax.hlines(high, positions-.018, positions+.018, color='#333333', lw=.8)
            ax.set_title(title, loc='left', fontsize=11, weight='normal')
            counts = [int(indexed.loc[(c, MODEL_STYLES[0][0]), 'n_scans']) for c in cohorts]
            ax.set_xticks(x, [f'{c}\nn={n} scans' for c, n in zip(cohorts, counts)])
            ax.grid(axis='y', color='#ECECEC', linewidth=.5, linestyle='-')
            ax.set_axisbelow(True)
            ax.axhline(0, color='#777777', lw=.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#777777')
            ax.tick_params(width=.6, length=3, color='#777777')
            ax.margins(x=.07)
            if metric in ('MAE', 'RMSE'):
                ax.set_ylim(bottom=0)
        fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in MODEL_STYLES],
                   loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False,
                   handlelength=1.3, handleheight=.8, columnspacing=2, fontsize=10)
        label = 'Training-age mean-only ComBat' if mode == 'age_mean' else 'No ComBat'
        reference = indexed.loc[[(c, MODEL_STYLES[0][0]) for c in cohorts]]
        scans, participants = int(reference.n_scans.sum()), int(reference.n_participants.sum())
        fig.text(.5, .012, label + f' · Participant-nested OOF · {scans} CN scans / {participants} participants',
                 ha='center', fontsize=9, color='#555555')
        fig.tight_layout(rect=[.015, .045, .995, .91], h_pad=2.2, w_pad=2.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ('png', 'pdf', 'svg'):
            target = output.with_suffix('.'+extension)
            fig.savefig(target, dpi=220)
            if extension == 'svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
            print(target)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT/'reports/current/paired_metrics_six_models.csv')
    parser.add_argument('--output', type=Path,
                        help='Output stem; defaults to the matching figure name in reports/current')
    parser.add_argument('--before-after', action='store_true',
                        help='Eight panels: no-ComBat versus training-age mean-only ComBat')
    parser.add_argument('--mode', choices=['age_mean', 'none'], default='age_mean')
    parser.add_argument('--no-ci', action='store_true', help='Hide confidence interval error bars')
    args = parser.parse_args()
    output = args.output or ROOT / 'reports/current' / (
        'paired_model_performance_before_after' if args.before_after else 'paired_model_performance')
    if args.before_after:
        plot_before_after(pd.read_csv(args.input), output, show_ci=not args.no_ci)
    else:
        plot_comparison(pd.read_csv(args.input), output, show_ci=not args.no_ci, mode=args.mode)


def plot_before_after(data, output, show_ci=True):
    """Compare separately trained declared arms on shared axes and observations."""
    keys = [(cohort, model) for cohort in COHORT_ORDER for model, _, _ in MODEL_STYLES]
    indexed = {}
    for mode in ['none', 'age_mean']:
        frame = data.loc[data['mode'].eq(mode)].set_index(['cohort', 'model'])
        if frame.index.has_duplicates:
            raise ValueError('Expected one row per cohort/model/mode')
        frame = frame.loc[keys]
        for cohort in COHORT_ORDER:
            part = frame.loc[cohort]
            if part.n_scans.nunique() != 1 or part.n_participants.nunique() != 1:
                raise ValueError('All models must use identical matched cohort counts')
        indexed[mode] = frame
    pd.testing.assert_frame_equal(
        indexed['none'][['n_scans', 'n_participants']],
        indexed['age_mean'][['n_scans', 'n_participants']])
    with plt.rc_context({'font.size': 10, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                         'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey='col')
        fig.subplots_adjust(left=.055, right=.99, bottom=.14, top=.83,
                            hspace=.50, wspace=.27)
        x = np.arange(len(COHORT_ORDER))
        for row, mode in enumerate(['none', 'age_mean']):
            frame = indexed[mode]
            for col, (metric, title) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(f'{chr(65 + row * 4 + col)}. {title.split(". ", 1)[1]}',
                             loc='left', fontsize=12, weight='normal', pad=10)
                for j, (model, _, color) in enumerate(MODEL_STYLES):
                    values = frame.loc[[(c, model) for c in COHORT_ORDER]]
                    positions = x + (j - 2.5) * BAR_WIDTH
                    ax.bar(positions, values[metric], width=BAR_WIDTH * .94,
                           color=color, edgecolor='white', linewidth=.4)
                    if show_ci:
                        low, high = values[metric + '_CI_low'], values[metric + '_CI_high']
                        ax.vlines(positions, low, high, color='#333333', lw=.75)
                        ax.hlines(low, positions - .018, positions + .018, color='#333333', lw=.75)
                        ax.hlines(high, positions - .018, positions + .018, color='#333333', lw=.75)
                counts = [int(frame.loc[(c, MODEL_STYLES[0][0]), 'n_scans']) for c in COHORT_ORDER]
                ax.set_xticks(x, [f'{c}\nn={n}' for c, n in zip(COHORT_ORDER, counts)])
                ax.grid(axis='y', color='#ECECEC', linewidth=.5)
                ax.set_axisbelow(True)
                ax.axhline(0, color='#777777', lw=.6)
                ax.spines[['top', 'right']].set_visible(False)
                ax.spines[['left', 'bottom']].set_color('#777777')
                ax.tick_params(width=.6, length=3, color='#777777', labelleft=True)
                ax.margins(x=.07)
        # Determine limits only after both rows have been drawn. This prevents
        # setting the first row's lower bound from freezing a too-small range.
        for col, (metric, _) in enumerate(METRICS):
            low_key = metric + '_CI_low' if show_ci else metric
            high_key = metric + '_CI_high' if show_ci else metric
            low = min(float(frame[low_key].min()) for frame in indexed.values())
            high = max(float(frame[high_key].max()) for frame in indexed.values())
            if metric in ('MAE', 'RMSE'):
                axes[0, col].set_ylim(0, high * 1.07)
            elif metric == 'Pearson_r':
                axes[0, col].set_ylim(-1.08, 1.08)
            else:
                low, high = min(low, 0), max(high, 0)
                pad = max((high - low) * .07, .05)
                axes[0, col].set_ylim(low - pad, high + pad)
        fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in MODEL_STYLES],
                   loc='upper center', bbox_to_anchor=(.5, .995), ncol=3,
                   frameon=False, handlelength=1.3, columnspacing=3, fontsize=11)
        fig.text(.055, .90, 'Before harmonization · No ComBat', fontsize=12)
        fig.text(.055, .48, 'After harmonization · Training-age mean-only ComBat', fontsize=12)
        reference = indexed['none'].loc[[(c, MODEL_STYLES[0][0]) for c in COHORT_ORDER]]
        scans, participants = int(reference.n_scans.sum()), int(reference.n_participants.sum())
        footer = f'Same participant folds and CN observations · {scans} scans / {participants} participants · Matched cohorts; n = scans'
        if show_ci:
            footer += ' · 95% participant-bootstrap intervals'
        fig.text(.5, .065, footer, ha='center', fontsize=9, color='#555555')
        fig.text(.5, .03, 'Both arms use training-only QC/scaling and separate nested model fitting; harmonization is fitted on training folds only.',
                 ha='center', fontsize=9, color='#555555')
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf', 'svg']:
            path = output.with_suffix('.' + ext)
            fig.savefig(path, dpi=220)
            if ext == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            print(path)
        plt.close(fig)


if __name__ == '__main__':
    main()
