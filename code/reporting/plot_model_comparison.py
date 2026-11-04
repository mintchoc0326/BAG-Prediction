#!/usr/bin/env python3
"""Plot cohort/model comparisons from aggregate CSV metrics (no training).

Run from the repository root:
    python code/reporting/plot_model_comparison.py
    python code/reporting/plot_model_comparison.py --no-ci

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
    ('WEIGHTED_EARLY_FUSION', 'Ridge–fusion (weighted)', '#BC5B58'),
    ('DWI_GNN', 'GNN–DWI', '#59926C'),
    ('fMRI_GNN', 'GNN–fMRI', '#8D74AA'),
    ('GNN_FUSION', 'GNN–fusion', '#D08B4B'),
]
METRICS = [('MAE', 'A. MAE (years)'), ('RMSE', 'B. RMSE (years)'),
           ('R2', 'C. R²'), ('Pearson_r', 'D. Pearson r')]
COHORT_ORDER = ['ADNI', 'ADRC', 'HABS', 'ADDECODE', 'ADDecode']
FIGSIZE = (12, 8)
FONT_SIZE = 10
BAR_WIDTH = 0.125


def plot_comparison(data, output, show_ci=True):
    cohorts = [c for c in COHORT_ORDER if c in data['dataset'].unique()]
    if not cohorts:
        raise ValueError('No recognized cohorts in input CSV.')
    indexed = data.set_index(['dataset', 'model'])
    if indexed.index.has_duplicates:
        raise ValueError('Expected one row per dataset/model.')
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
        fig.tight_layout(rect=[.015, .015, .995, .91], h_pad=2.2, w_pad=2.5)
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
                        default=ROOT/'reports/brain_age/paired_metrics_six_models.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'reports/brain_age/paired_model_performance')
    parser.add_argument('--no-ci', action='store_true', help='Hide confidence interval error bars')
    args = parser.parse_args()
    plot_comparison(pd.read_csv(args.input), args.output, show_ci=not args.no_ci)


if __name__ == '__main__':
    main()
