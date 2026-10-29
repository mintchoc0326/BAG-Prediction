#!/usr/bin/env python3
"""Build English OOF benchmark figures and reproduce every native slide table.

No training or participant-level export. Requires numpy, pandas, matplotlib.
Embedded slide tables are a dated presentation snapshot, not recomputed analyses.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap
import warnings
import zipfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['MAE', 'RMSE', 'R2', 'Pearson_r']
MODELS = {
    'DWI': ('Ridge', 'DWI-only'),
    'fMRI': ('Ridge', 'fMRI-only'),
    'WEIGHTED_EARLY_FUSION': ('Ridge', 'Fusion (weighted early)'),
    'DWI_GNN': ('GNN', 'DWI-only'),
    'fMRI_GNN': ('GNN', 'fMRI-only'),
    'GNN_FUSION': ('GNN', 'Fusion'),
    'EARLY_FUSION': ('Ridge', 'Fusion (early)'),
    'LATE_MEAN': ('Ridge', 'Fusion (late mean)'),
}
MAIN = list(MODELS)[:6]
COLORS = ['#2878B5', '#E89B35', '#36965B', '#77B5DE', '#F2C57C', '#88C999']
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
# Filled with all native tables from both reviewed decks; no individual data.
SLIDE_SNAPSHOT = [{'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 5, 'table': 1, 'context': 'METHOD DETAILS | Data preparation and quality-control: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['i; n', 'Scan index; number of eligible scans', 'A participant may contribute multiple visits; i is not a participant index.'], ['N; Aᵢ', 'N = 84 ROIs; Aᵢ is 84 × 84', 'A matrix entry is connectivity between two ROIs, not a regional volume.'], ['vecᵤ; xᵢ', 'Upper triangle; xᵢ ∈ ℝ³⁴⁸⁶', 'Take each unordered ROI pair once; omit the diagonal.'], ['d; X', 'd = N(N−1)/2 = 3,486; X is n × d', 'Rows are scans; columns refer to the same ROI pair across scans.'], ['q(i); visit(i)', 'Participant identifier; visit identifier', 'q(i) defines grouping in cross-validation; visits remain together.'], ['b(i); cᵢ', 'Technical batch; biological design vector', 'b(i) combines dataset and site; cᵢ encodes covariates to preserve.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 6, 'table': 1, 'context': 'STUDY DESIGN | Cohort sizes and the paired evaluation set | report_tables/Table1_samples.csv; DWI metadata; modality CV summaries', 'rows': [['Analysis', 'Scans', 'Participants / scope'], ['DWI harmonization', '1,234', 'ADNI 314; HABS 487; ADRC 347; ADDECODE 86'], ['fMRI harmonization', '847', 'ADNI, HABS, ADRC'], ['DWI CN OOF models', '916', '667 participants; HABS all-CN override'], ['fMRI CN OOF models', '642', '525 participants'], ['Paired DWI + fMRI', '847', '679 participants'], ['Shared fusion CN OOF', '387', '330 participants']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 8, 'table': 1, 'context': 'METHOD DETAILS | DWI harmonization: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['xᵢⱼ; uᵢⱼ', 'Raw weight; log1p weight, dimensionless', 'i indexes scans; j indexes unique connections; u = log(1+x).'], ['αⱼ; βⱼ; cᵢ', 'Edge intercept; covariate coefficients; design vector', 'cᵢ contains age and sex in the saved DWI run; diagnosis is absent.'], ['γᵦⱼ; δᵦⱼ', 'Batch location shift; positive batch scale', 'b = b(i) is dataset/site, not the Ridge intercept.'], ['σⱼ; εᵢⱼ', 'Edge residual SD; standardized residual', 'Feature standardization precedes empirical Bayes batch estimation.'], ['γ*ᵦⱼ; δ*ᵦⱼ', 'Empirical Bayes batch parameter estimates', 'Shrink noisy batch estimates using information across connections.'], ['H; expm1', 'Harmonized quantity; inverse log1p', 'Reconstruct symmetric weights and apply the export nonnegative convention.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 10, 'table': 1, 'context': 'METHOD DETAILS | fMRI harmonization: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['rᵢⱼ; zᵢⱼ', 'Correlation; Fisher-z feature', 'r is clipped inside (−1,1); z = arctanh(r).'], ['cᵢ; βⱼ', 'Covariate design vector; edge coefficients', 'Age is continuous; sex and diagnosis are categorical design columns.'], ['αⱼ; b(i)', 'Feature intercept; dataset/site batch index', 'Each eligible scan belongs to one batch; singleton batches are removed.'], ['γᵦⱼ; δᵦⱼ', 'Batch shift; batch scale', 'Adjustment removes modeled technical location and scale effects.'], ['σⱼ; γ*; δ*', 'Residual SD; empirical Bayes batch estimates', 'The detailed ComBat diagram shows standardization and reconstruction.'], ['zᴴ; rᴴ', 'Adjusted z feature; adjusted correlation', 'rᴴ = tanh(zᴴ); reconstruct an 84 × 84 symmetric matrix with diagonal 1.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 14, 'table': 1, 'context': 'METHOD DETAILS | Harmonization evaluation: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['X; μ; V₂', 'Feature matrix; column means; two PCA loadings', 'T = (X−μ)V₂ is n × 2; separate before/after fits need not share axes.'], ['K; recallₖ', 'Number of classes; TPₖ/(TPₖ+FNₖ)', 'Balanced accuracy averages sensitivity equally across classes.'], ['yᵢ; ŷᵢ; ȳ', 'Actual age; estimated age; evaluation mean', 'Age and age errors are measured in years.'], ['MAE; RMSE', 'Mean absolute error; root mean squared error', 'MAE = mean |ŷ−y|; RMSE = sqrt(mean (ŷ−y)²).'], ['R²', '1 − residual sum of squares / total sum of squares', 'R² < 0 means worse squared error than predicting the evaluation mean.'], ['Target-specific n', 'Eligible scans for each diagnostic target', 'Sample sizes and class counts differ; do not interpret all rows as one set.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 18, 'table': 1, 'context': 'HARMONIZATION | Biological signal preservation is mixed | report_tables/Table3_harmonization.csv', 'rows': [['Diagnostic', 'DWI before → after', 'fMRI before → after'], ['Sex balanced accuracy', '0.782 → 0.766', '0.631 → 0.640'], ['Diagnosis balanced accuracy', '0.294 → 0.151', '0.300 → 0.227'], ['Age MAE (years)', '6.22 → 6.71', '7.64 → 8.00'], ['Age R²', '0.588 → 0.516', '0.379 → 0.307']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 21, 'table': 1, 'context': 'METHOD DETAILS | Single-modality Ridge: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['xᵢ; Zᵢ', 'Raw edge vector; standardized edge vector', 'Both have d = 3,486 entries; DWI and fMRI are fitted separately.'], ['μⱼ; σⱼ', 'Training-column mean and standard deviation', 'Re-estimate inside every inner split; outer-test observations never fit them.'], ['β ∈ ℝᵈ; b ∈ ℝ', 'Edge coefficients; unpenalized age intercept', 'ŷ = b + Zβ; β converts standardized feature values into age years.'], ['λ; ‖β‖₂²', 'Nonnegative penalty; sum of squared coefficients', 'λ is sklearn Ridge alpha; larger values shrink coefficients more strongly.'], ['Tₒ; Vₒ', 'Outer-training set; outer-test set for fold o', 'All visits from a participant belong to one outer fold.'], ['λ*; OOF', 'Inner-CV selected penalty; out-of-fold prediction', 'Refit on Tₒ using λ*; predict Vₒ once. Full-data ComBat remains upstream.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 24, 'table': 1, 'context': 'METHOD DETAILS | Single-modality GNN: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['v,u; N(v); Eᵣ', 'ROI nodes; neighbors of v; selected edge set', 'Use r for retained fraction to distinguish it from a participant index.'], ['hᵥ⁽⁰⁾; hᵥ⁽ˡ⁾', '84-value connectivity row; layer-l embedding', 'h⁽⁰⁾ ∈ ℝ⁸⁴; hidden vectors ∈ ℝ⁶⁴; three GINE layers.'], ['eᵤᵥ; Wₗ', 'Scalar edge weight; learned edge projection', 'Project scalar edges to the current node-feature dimension before addition.'], ['mᵥ⁽ˡ⁾; εₗ', 'Summed neighbor message; self-feature coefficient', 'ε is fixed at PyG default zero; the MLP transforms self + neighbor information.'], ['g; Head(g)', '64-value mean-pooled graph embedding; scalar output', 'Regressor: 64 → 32 → 1. Its output is standardized age.'], ['μᵧ; σᵧ; ŷ', 'Training age mean / SD; predicted age in years', 'MSE trains standardized age; grouped inner validation selects epochs.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 26, 'table': 1, 'context': 'SINGLE MODALITY | Ridge leads the full-cohort unimodal analyses | DWI_Ridge_CV_metrics.json; fMRI_Ridge_CV_metrics.json; selected GNN threshold summaries', 'rows': [['Modality / model', 'CN scans', 'MAE (yr)', 'RMSE (yr)', 'R²', 'Pearson r'], ['DWI Ridge', '916', '5.98', '7.89', '0.519', '0.724'], ['DWI GNN (top 50%)', '916', '7.56', '9.61', '0.287', '0.537'], ['fMRI Ridge', '642', '7.33', '9.49', '0.406', '0.637'], ['fMRI GNN (top 10%)', '642', '9.31', '11.97', '0.056', '0.305']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 29, 'table': 1, 'context': 'METHOD DETAILS | BAG and age-bias correction: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['f₋fold(i)', 'Predictor trained without scan i’s outer fold', 'Participant-level grouping also excludes other visits from that participant.'], ['ŷᵢ; yᵢ', 'Predicted age; chronological age, both in years', 'ŷ is the raw prediction before subtracting the BAG age trend.'], ['BAGᵢ', 'Raw brain age gap = ŷᵢ − yᵢ', 'Positive values indicate older predicted age relative to chronological age.'], ['a; b', 'Age-bias intercept (years); slope (years/year)', 'These are calibration coefficients, distinct from Ridge β and intercept.'], ['â; b̂; cBAGᵢ', 'Fitted coefficients; BAGᵢ − (â+b̂yᵢ)', 'The CN OOF population defines the reference age trend.'], ['Reference vs inference', 'Descriptive calibration vs independent validation', 'Fitting and assessing correction on one reference does not independently validate it.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 32, 'table': 1, 'context': 'METHOD DETAILS | Cross-modality pairing and training selection: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['D; F', 'Superscripts for DWI and fMRI', 'Superscripts identify modalities; they are not mathematical powers.'], ['keyᵢ', 'Dataset / participant / normalized visit key', 'Key agreement does not establish acquisition-date proximity.'], ['dᵢ; τ', 'Absolute age disagreement; tolerance τ = 0.01 yr', 'Exclude unresolved ages from training; do not silently replace them.'], ['CNᴰ; CNᶠ', 'Modality-specific cognitively normal indicators', 'DWI HABS all-CN override is retained; fMRI requires explicit CN.'], ['∧; sex_ok', 'Logical AND; no explicit sex disagreement', 'Every required condition must hold; missing sex is allowed in this model.'], ['n; P; q(i)', '387 scans; 330 participants; scan-to-participant map', 'Pairing yields 847 scans; policy and consistency filters define the evaluated subset.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 33, 'table': 1, 'context': 'MULTIMODAL FUSION | Paired-sample selection changes the population | report_tables/Table1_samples.csv; metadata/fusion_metadata_audit_summary.json', 'rows': [['Cohort', 'Paired scans', 'Evaluated CN scans', 'Evaluated participants'], ['ADNI', '245', '131', '74'], ['ADRC', '338', '243', '243'], ['HABS', '264', '13', '13'], ['Total', '847', '387', '330']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 34, 'table': 1, 'context': 'MULTIMODAL FUSION | Fusion combines modalities in three ways | Ridge_baseline/fusion_summary.json; Weighted_Ridge/fusion_summary.json; GNN_fusion/GNN_fusion_summary.json', 'rows': [['Approach', 'Implementation'], ['Ridge early fusion', 'Concatenate standardized DWI and fMRI edges: 6,972 features'], ['Weighted Ridge early fusion', '[Z_DWI, w × Z_fMRI]; tune w and alpha jointly within training folds'], ['Ridge late mean', 'Average the two unimodal age predictions'], ['GNN fusion', 'Separate two-layer GINE encoders; concatenate pooled embeddings; nonlinear age head']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 36, 'table': 1, 'context': 'METHOD DETAILS | Ridge fusions: early, weighted and late: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Zᴰ; Zᶠ', 'DWI and fMRI standardized edge vectors', 'Each has 3,486 columns; fit modality columns using training data only.'], ['[ ; ]; Zᴱ', 'Feature concatenation; early-fusion vector', 'Zᴱ = [Zᴰ;Zᶠ] contains 6,972 features.'], ['w; Zᵂ', 'fMRI block scale; weighted concatenation', 'Zᵂ=[Zᴰ;wZᶠ]; DWI scale is one; w is not a contribution percentage.'], ['λ*; w*', 'Penalty and weight jointly selected by inner CV', 'Search w ∈ {0,0.25,0.5,1,2,4}; λ is Ridge alpha.'], ['β; b', '6,972 edge coefficients; scalar intercept', 'Only β is penalized; never standardize again after block weighting.'], ['ŷᴰ; ŷᶠ; ŷlate', 'Two unimodal OOF ages; arithmetic mean', 'Late fusion averages predictions in years, not coefficients or embeddings.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 38, 'table': 1, 'context': 'METHOD DETAILS | GNN fusion: two modality-specific encoders: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Hₘ⁽⁰⁾; m', 'Node input matrix; modality m ∈ {D,F}', '84 × 84 input per modality; DWI log1p weights, fMRI signed correlations.'], ['Eₘ; eᵤᵥ,ₘ', 'Positive top-50% topology; normalized scalar weight', 'e = selected positive weight / graph positive maximum; bidirectional edges.'], ['Hₘ⁽ˡ⁾; gₘ', 'ROI embeddings; pooled modality vector', 'Two GINE layers, hidden 32; mean+max pooling makes gₘ ∈ ℝ⁶⁴.'], ['g=[gᴰ;gᶠ]', 'Concatenated fusion representation ∈ ℝ¹²⁸', 'A separate encoder is learned for each modality; weights are not shared.'], ['Head; μᵧ; σᵧ', '128→32→1 MLP; training age mean and SD', 'Head predicts standardized age; multiply by σᵧ and add μᵧ for age years.'], ['μₘ,j; σₘ,j', 'Training node-feature column statistics', 'Pool training graph nodes per modality; fresh outer refit follows inner epoch selection.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 40, 'table': 1, 'context': 'MULTIMODAL FUSION | The fusion gain is modest and method dependent | report_tables/Table4_models_pooled.csv; Weighted_Ridge/fusion_summary.json', 'rows': [['Model', 'MAE (yr)', 'RMSE (yr)', 'R²', 'r'], ['DWI Ridge', '6.66', '8.49', '0.620', '0.788'], ['Ridge early fusion', '6.69', '8.62', '0.609', '0.780'], ['Weighted Ridge fusion', '6.47', '8.38', '0.630', '0.794'], ['Ridge late mean', '6.57', '8.48', '0.621', '0.799'], ['GNN fusion', '9.07', '11.44', '0.311', '0.558']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 41, 'table': 1, 'context': 'MULTIMODAL FUSION | Pooled performance masks cohort differences | report_tables/Table5_models_by_cohort.csv', 'rows': [['Cohort', 'CN scans', 'DWI Ridge MAE', 'Weighted Ridge MAE', 'GNN fusion MAE'], ['ADNI', '131', '7.15', '6.85', '8.91'], ['ADRC', '243', '6.52', '6.39', '9.04'], ['HABS', '13', '4.12', '4.16', '11.24']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 43, 'table': 1, 'context': 'METHOD DETAILS | Interpretability: from OOF models to attribution: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['f; x; x′', 'Saved outer-fold age predictor; observed input; reference', 'f outputs age years; x jointly contains both modalities’ node and edge attributes.'], ['x′ node / edge', 'Training-mean node values; zero edge attributes', 'Node mean corresponds to standardized zero; keep observed diagonal fixed.'], ['k; t; x(t)', 'Input coordinate; path fraction; interpolated input', 'k is not a fold; t ∈ [0,1]; x(t)=x′+t(x−x′); topology is unchanged.'], ['∂f/∂xₖ; IGₖ', 'Local model sensitivity; integrated contribution', 'IG has age-year units; it explains f(x)−f(x′), not a causal effect.'], ['S; rIG', 'Integration intervals; completeness residual', 'Trapezoidal quadrature; S = 64 initially, adapt up to 512; |rIG| ≤ 0.05 yr.'], ['Δpred; rIG', 'Prediction consistency vs integration error', 'Require inference/checkpoint difference ≤ 0.001 yr; IG completeness is assessed separately.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 48, 'table': 1, 'context': 'METHOD DETAILS | Interpretability summaries and robustness checks: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Aₛ,ᵢⱼ; p; Vₚ', 'Scan connection IG; participant; participant visits', 'i,j now index ROI endpoints; s indexes scans to avoid an index collision.'], ['A node; A edge', 'Node-profile attribution; edge-attribute attribution', 'Sum both symmetric node entries and both directed edge entries for each pair.'], ['Āₚ,ᵢⱼ; Iᵢⱼ', 'Signed visit average; mean participant absolute score', 'Average signed scan values first, then take absolute values across participants.'], ['ROIᵢ', '½ Σⱼ≠ᵢ Iᵢⱼ, in attribution years', 'Each connection contributes half its absolute importance to each endpoint.'], ['rank; top-k frequency', 'Per-fold ranking; count of folds in top k', 'k = 100 for connections; k = 10 for ROIs; frequency ranges from 0 to 5.'], ['x₋ₘ,x′ₘ; ΔMAEₘ', 'Keep other input; replace modality m; error change', 'Average within participant, then across participants; positive = worse prediction.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 49, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | Attribution quality limits the strength of interpretation | GNN_interpretability/analysis_summary.json; reconstruction_check.csv', 'rows': [['Quality check', 'Observed result'], ['IG integration steps', '64 initially; up to 512'], ['Completeness tolerance', '0.05 years'], ['Scans failing completeness', '6 / 387 (1.6%)'], ['Maximum absolute IG residual', '0.119 years'], ['Maximum difference from original OOF prediction', '13.18 years']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 50, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | fMRI carries larger attribution in this fusion model | GNN_interpretability/modality_summary.csv', 'rows': [['Modality', 'Mean signed IG (yr)', 'Mean absolute connection IG', 'Occlusion error increase (yr)'], ['DWI', '-6.96', '45.17', '2.55'], ['fMRI', '10.33', '128.54', '3.36']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 52, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | Leading connections and stability offer complementary views | GNN_interpretability/connection_importance.csv; connection_fold_stability.csv', 'rows': [['Connection (fMRI)', 'Mean |IG| (yr)', 'Folds in top 100'], ['lh_precuneus ↔ rh_precuneus', '0.415', '5/5'], ['lh_entorhinal ↔ rh_entorhinal', '0.233', '5/5'], ['lh_isthmuscingulate ↔ lh_precuneus', '0.195', '5/5'], ['lh_inferiorparietal ↔ lh_precuneus', '0.183', '3/5'], ['lh_insula ↔ rh_insula', '0.170', '5/5']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable.pptx', 'slide': 54, 'table': 1, 'context': 'APPENDIX | Result provenance and run selection | Paths are relative to the project results directory.', 'rows': [['Section', 'Primary saved evidence'], ['Harmonization', 'final/Fusion/report_tables/Table3_harmonization.csv'], ['DWI single modality', 'final/DWI/{Ridge_BAG,GNN_BAG}/*summary*.json / *metrics.json'], ['fMRI single modality', 'final/fMRI/{Ridge_BAG,GNN_BAG}'], ['Shared fusion benchmark', 'final/Fusion/report_tables/Table1,4,5,6*.csv'], ['Interpretability', 'explore/GNN_attribution/*.csv; analysis_summary.json']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 5, 'table': 1, 'context': 'METHOD DETAILS | Data preparation and quality-control: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['i; n', 'Scan index; number of eligible scans', 'A participant may contribute multiple visits; i is not a participant index.'], ['N; Aᵢ', 'N = 84 ROIs; Aᵢ is 84 × 84', 'A matrix entry is connectivity between two ROIs, not a regional volume.'], ['vecᵤ; xᵢ', 'Upper triangle; xᵢ ∈ ℝ³⁴⁸⁶', 'Take each unordered ROI pair once; omit the diagonal.'], ['d; X', 'd = N(N−1)/2 = 3,486; X is n × d', 'Rows are scans; columns refer to the same ROI pair across scans.'], ['q(i); visit(i)', 'Participant identifier; visit identifier', 'q(i) defines grouping in cross-validation; visits remain together.'], ['b(i); cᵢ', 'Technical batch; biological design vector', 'b(i) combines dataset and site; cᵢ encodes covariates to preserve.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 6, 'table': 1, 'context': 'STUDY DESIGN | Cohort sizes and the paired evaluation set | report_tables/Table1_samples.csv; DWI metadata; modality CV summaries', 'rows': [['Analysis', 'Scans', 'Participants / scope'], ['DWI harmonization', '1,234', 'ADNI 314; HABS 487; ADRC 347; ADDECODE 86'], ['fMRI harmonization', '847', 'ADNI, HABS, ADRC'], ['DWI CN OOF models', '916', '667 participants; HABS all-CN override'], ['fMRI CN OOF models', '642', '525 participants'], ['Paired DWI + fMRI', '847', '679 participants'], ['Shared fusion CN OOF', '387', '330 participants']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 8, 'table': 1, 'context': 'METHOD DETAILS | DWI harmonization: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['xᵢⱼ; uᵢⱼ', 'Raw weight; log1p weight, dimensionless', 'i indexes scans; j indexes unique connections; u = log(1+x).'], ['αⱼ; βⱼ; cᵢ', 'Edge intercept; covariate coefficients; design vector', 'cᵢ contains age and sex in the saved DWI run; diagnosis is absent.'], ['γᵦⱼ; δᵦⱼ', 'Batch location shift; positive batch scale', 'b = b(i) is dataset/site, not the Ridge intercept.'], ['σⱼ; εᵢⱼ', 'Edge residual SD; standardized residual', 'Feature standardization precedes empirical Bayes batch estimation.'], ['γ*ᵦⱼ; δ*ᵦⱼ', 'Empirical Bayes batch parameter estimates', 'Shrink noisy batch estimates using information across connections.'], ['H; expm1', 'Harmonized quantity; inverse log1p', 'Reconstruct symmetric weights and apply the export nonnegative convention.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 10, 'table': 1, 'context': 'METHOD DETAILS | fMRI harmonization: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['rᵢⱼ; zᵢⱼ', 'Correlation; Fisher-z feature', 'r is clipped inside (−1,1); z = arctanh(r).'], ['cᵢ; βⱼ', 'Covariate design vector; edge coefficients', 'Age is continuous; sex and diagnosis are categorical design columns.'], ['αⱼ; b(i)', 'Feature intercept; dataset/site batch index', 'Each eligible scan belongs to one batch; singleton batches are removed.'], ['γᵦⱼ; δᵦⱼ', 'Batch shift; batch scale', 'Adjustment removes modeled technical location and scale effects.'], ['σⱼ; γ*; δ*', 'Residual SD; empirical Bayes batch estimates', 'The detailed ComBat diagram shows standardization and reconstruction.'], ['zᴴ; rᴴ', 'Adjusted z feature; adjusted correlation', 'rᴴ = tanh(zᴴ); reconstruct an 84 × 84 symmetric matrix with diagonal 1.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 14, 'table': 1, 'context': 'METHOD DETAILS | Harmonization evaluation: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['X; μ; V₂', 'Feature matrix; column means; two PCA loadings', 'T = (X−μ)V₂ is n × 2; separate before/after fits need not share axes.'], ['K; recallₖ', 'Number of classes; TPₖ/(TPₖ+FNₖ)', 'Balanced accuracy averages sensitivity equally across classes.'], ['yᵢ; ŷᵢ; ȳ', 'Actual age; estimated age; evaluation mean', 'Age and age errors are measured in years.'], ['MAE; RMSE', 'Mean absolute error; root mean squared error', 'MAE = mean |ŷ−y|; RMSE = sqrt(mean (ŷ−y)²).'], ['R²', '1 − residual sum of squares / total sum of squares', 'R² < 0 means worse squared error than predicting the evaluation mean.'], ['Target-specific n', 'Eligible scans for each diagnostic target', 'Sample sizes and class counts differ; do not interpret all rows as one set.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 18, 'table': 1, 'context': 'HARMONIZATION | Biological signal preservation is mixed | report_tables/Table3_harmonization.csv', 'rows': [['Diagnostic', 'DWI before → after', 'fMRI before → after'], ['Sex balanced accuracy', '0.782 → 0.766', '0.631 → 0.640'], ['Diagnosis balanced accuracy', '0.294 → 0.151', '0.300 → 0.227'], ['Age MAE (years)', '6.22 → 6.71', '7.64 → 8.00'], ['Age R²', '0.588 → 0.516', '0.379 → 0.307']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 21, 'table': 1, 'context': 'METHOD DETAILS | Single-modality Ridge: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['xᵢ; Zᵢ', 'Raw edge vector; standardized edge vector', 'Both have d = 3,486 entries; DWI and fMRI are fitted separately.'], ['μⱼ; σⱼ', 'Training-column mean and standard deviation', 'Re-estimate inside every inner split; outer-test observations never fit them.'], ['β ∈ ℝᵈ; b ∈ ℝ', 'Edge coefficients; unpenalized age intercept', 'ŷ = b + Zβ; β converts standardized feature values into age years.'], ['λ; ‖β‖₂²', 'Nonnegative penalty; sum of squared coefficients', 'λ is sklearn Ridge alpha; larger values shrink coefficients more strongly.'], ['Tₒ; Vₒ', 'Outer-training set; outer-test set for fold o', 'All visits from a participant belong to one outer fold.'], ['λ*; OOF', 'Inner-CV selected penalty; out-of-fold prediction', 'Refit on Tₒ using λ*; predict Vₒ once. Full-data ComBat remains upstream.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 24, 'table': 1, 'context': 'METHOD DETAILS | Single-modality GNN: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['v,u; N(v); Eᵣ', 'ROI nodes; neighbors of v; selected edge set', 'Use r for retained fraction to distinguish it from a participant index.'], ['hᵥ⁽⁰⁾; hᵥ⁽ˡ⁾', '84-value connectivity row; layer-l embedding', 'h⁽⁰⁾ ∈ ℝ⁸⁴; hidden vectors ∈ ℝ⁶⁴; three GINE layers.'], ['eᵤᵥ; Wₗ', 'Scalar edge weight; learned edge projection', 'Project scalar edges to the current node-feature dimension before addition.'], ['mᵥ⁽ˡ⁾; εₗ', 'Summed neighbor message; self-feature coefficient', 'ε is fixed at PyG default zero; the MLP transforms self + neighbor information.'], ['g; Head(g)', '64-value mean-pooled graph embedding; scalar output', 'Regressor: 64 → 32 → 1. Its output is standardized age.'], ['μᵧ; σᵧ; ŷ', 'Training age mean / SD; predicted age in years', 'MSE trains standardized age; grouped inner validation selects epochs.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 26, 'table': 1, 'context': 'SINGLE MODALITY | Ridge leads the full-cohort unimodal analyses | DWI_Ridge_CV_metrics.json; fMRI_Ridge_CV_metrics.json; selected GNN threshold summaries', 'rows': [['Modality / model', 'CN scans', 'MAE (yr)', 'RMSE (yr)', 'R²', 'Pearson r'], ['DWI Ridge', '916', '5.98', '7.89', '0.519', '0.724'], ['DWI GNN (top 50%)', '916', '7.56', '9.61', '0.287', '0.537'], ['fMRI Ridge', '642', '7.33', '9.49', '0.406', '0.637'], ['fMRI GNN (top 10%)', '642', '9.31', '11.97', '0.056', '0.305']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 29, 'table': 1, 'context': 'METHOD DETAILS | BAG and age-bias correction: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['f₋fold(i)', 'Predictor trained without scan i’s outer fold', 'Participant-level grouping also excludes other visits from that participant.'], ['ŷᵢ; yᵢ', 'Predicted age; chronological age, both in years', 'ŷ is the raw prediction before subtracting the BAG age trend.'], ['BAGᵢ', 'Raw brain age gap = ŷᵢ − yᵢ', 'Positive values indicate older predicted age relative to chronological age.'], ['a; b', 'Age-bias intercept (years); slope (years/year)', 'These are calibration coefficients, distinct from Ridge β and intercept.'], ['â; b̂; cBAGᵢ', 'Fitted coefficients; BAGᵢ − (â+b̂yᵢ)', 'The CN OOF population defines the reference age trend.'], ['Reference vs inference', 'Descriptive calibration vs independent validation', 'Fitting and assessing correction on one reference does not independently validate it.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 32, 'table': 1, 'context': 'METHOD DETAILS | Cross-modality pairing and training selection: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['D; F', 'Superscripts for DWI and fMRI', 'Superscripts identify modalities; they are not mathematical powers.'], ['keyᵢ', 'Dataset / participant / normalized visit key', 'Key agreement does not establish acquisition-date proximity.'], ['dᵢ; τ', 'Absolute age disagreement; tolerance τ = 0.01 yr', 'Exclude unresolved ages from training; do not silently replace them.'], ['CNᴰ; CNᶠ', 'Modality-specific cognitively normal indicators', 'DWI HABS all-CN override is retained; fMRI requires explicit CN.'], ['∧; sex_ok', 'Logical AND; no explicit sex disagreement', 'Every required condition must hold; missing sex is allowed in this model.'], ['n; P; q(i)', '387 scans; 330 participants; scan-to-participant map', 'Pairing yields 847 scans; policy and consistency filters define the evaluated subset.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 33, 'table': 1, 'context': 'MULTIMODAL FUSION | Paired-sample selection changes the population | report_tables/Table1_samples.csv; metadata/fusion_metadata_audit_summary.json', 'rows': [['Cohort', 'Paired scans', 'Evaluated CN scans', 'Evaluated participants'], ['ADNI', '245', '131', '74'], ['ADRC', '338', '243', '243'], ['HABS', '264', '13', '13'], ['Total', '847', '387', '330']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 34, 'table': 1, 'context': 'MULTIMODAL FUSION | Fusion combines modalities in three ways | Ridge_baseline/fusion_summary.json; Weighted_Ridge/fusion_summary.json; GNN_fusion/GNN_fusion_summary.json', 'rows': [['Approach', 'Implementation'], ['Ridge early fusion', 'Concatenate standardized DWI and fMRI edges: 6,972 features'], ['Weighted Ridge early fusion', '[Z_DWI, w × Z_fMRI]; tune w and alpha jointly within training folds'], ['Ridge late mean', 'Average the two unimodal age predictions'], ['GNN fusion', 'Separate two-layer GINE encoders; concatenate pooled embeddings; nonlinear age head']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 36, 'table': 1, 'context': 'METHOD DETAILS | Ridge fusions: early, weighted and late: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Zᴰ; Zᶠ', 'DWI and fMRI standardized edge vectors', 'Each has 3,486 columns; fit modality columns using training data only.'], ['[ ; ]; Zᴱ', 'Feature concatenation; early-fusion vector', 'Zᴱ = [Zᴰ;Zᶠ] contains 6,972 features.'], ['w; Zᵂ', 'fMRI block scale; weighted concatenation', 'Zᵂ=[Zᴰ;wZᶠ]; DWI scale is one; w is not a contribution percentage.'], ['λ*; w*', 'Penalty and weight jointly selected by inner CV', 'Search w ∈ {0,0.25,0.5,1,2,4}; λ is Ridge alpha.'], ['β; b', '6,972 edge coefficients; scalar intercept', 'Only β is penalized; never standardize again after block weighting.'], ['ŷᴰ; ŷᶠ; ŷlate', 'Two unimodal OOF ages; arithmetic mean', 'Late fusion averages predictions in years, not coefficients or embeddings.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 38, 'table': 1, 'context': 'METHOD DETAILS | GNN fusion: two modality-specific encoders: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Hₘ⁽⁰⁾; m', 'Node input matrix; modality m ∈ {D,F}', '84 × 84 input per modality; DWI log1p weights, fMRI signed correlations.'], ['Eₘ; eᵤᵥ,ₘ', 'Positive top-50% topology; normalized scalar weight', 'e = selected positive weight / graph positive maximum; bidirectional edges.'], ['Hₘ⁽ˡ⁾; gₘ', 'ROI embeddings; pooled modality vector', 'Two GINE layers, hidden 32; mean+max pooling makes gₘ ∈ ℝ⁶⁴.'], ['g=[gᴰ;gᶠ]', 'Concatenated fusion representation ∈ ℝ¹²⁸', 'A separate encoder is learned for each modality; weights are not shared.'], ['Head; μᵧ; σᵧ', '128→32→1 MLP; training age mean and SD', 'Head predicts standardized age; multiply by σᵧ and add μᵧ for age years.'], ['μₘ,j; σₘ,j', 'Training node-feature column statistics', 'Pool training graph nodes per modality; fresh outer refit follows inner epoch selection.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 40, 'table': 1, 'context': 'MULTIMODAL FUSION | The fusion gain is modest and method dependent | report_tables/Table4_models_pooled.csv; Weighted_Ridge/fusion_summary.json', 'rows': [['Model', 'MAE (yr)', 'RMSE (yr)', 'R²', 'r'], ['DWI Ridge', '6.66', '8.49', '0.620', '0.788'], ['Ridge early fusion', '6.69', '8.62', '0.609', '0.780'], ['Weighted Ridge fusion', '6.47', '8.38', '0.630', '0.794'], ['Ridge late mean', '6.57', '8.48', '0.621', '0.799'], ['GNN fusion', '9.07', '11.44', '0.311', '0.558']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 41, 'table': 1, 'context': 'MULTIMODAL FUSION | Pooled performance masks cohort differences | report_tables/Table5_models_by_cohort.csv', 'rows': [['Cohort', 'CN scans', 'DWI Ridge MAE', 'Weighted Ridge MAE', 'GNN fusion MAE'], ['ADNI', '131', '7.15', '6.85', '8.91'], ['ADRC', '243', '6.52', '6.39', '9.04'], ['HABS', '13', '4.12', '4.16', '11.24']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 43, 'table': 1, 'context': 'METHOD DETAILS | Interpretability: from OOF models to attribution: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['f; x; x′', 'Saved outer-fold age predictor; observed input; reference', 'f outputs age years; x jointly contains both modalities’ node and edge attributes.'], ['x′ node / edge', 'Training-mean node values; zero edge attributes', 'Node mean corresponds to standardized zero; keep observed diagonal fixed.'], ['k; t; x(t)', 'Input coordinate; path fraction; interpolated input', 'k is not a fold; t ∈ [0,1]; x(t)=x′+t(x−x′); topology is unchanged.'], ['∂f/∂xₖ; IGₖ', 'Local model sensitivity; integrated contribution', 'IG has age-year units; it explains f(x)−f(x′), not a causal effect.'], ['S; rIG', 'Integration intervals; completeness residual', 'Trapezoidal quadrature; S = 64 initially, adapt up to 512; |rIG| ≤ 0.05 yr.'], ['Δpred; rIG', 'Prediction consistency vs integration error', 'Require inference/checkpoint difference ≤ 0.001 yr; IG completeness is assessed separately.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 48, 'table': 1, 'context': 'METHOD DETAILS | Interpretability summaries and robustness checks: notation | Implementation-verified notation | See preceding diagram and speaker notes.', 'rows': [['Symbol / object', 'Definition and dimension', 'How to interpret it'], ['Aₛ,ᵢⱼ; p; Vₚ', 'Scan connection IG; participant; participant visits', 'i,j now index ROI endpoints; s indexes scans to avoid an index collision.'], ['A node; A edge', 'Node-profile attribution; edge-attribute attribution', 'Sum both symmetric node entries and both directed edge entries for each pair.'], ['Āₚ,ᵢⱼ; Iᵢⱼ', 'Signed visit average; mean participant absolute score', 'Average signed scan values first, then take absolute values across participants.'], ['ROIᵢ', '½ Σⱼ≠ᵢ Iᵢⱼ, in attribution years', 'Each connection contributes half its absolute importance to each endpoint.'], ['rank; top-k frequency', 'Per-fold ranking; count of folds in top k', 'k = 100 for connections; k = 10 for ROIs; frequency ranges from 0 to 5.'], ['x₋ₘ,x′ₘ; ΔMAEₘ', 'Keep other input; replace modality m; error change', 'Average within participant, then across participants; positive = worse prediction.']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 49, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | Attribution quality limits the strength of interpretation | GNN_interpretability/analysis_summary.json; reconstruction_check.csv', 'rows': [['Quality check', 'Observed result'], ['IG integration steps', '64 initially; up to 512'], ['Completeness tolerance', '0.05 years'], ['Scans failing completeness', '6 / 387 (1.6%)'], ['Maximum absolute IG residual', '0.119 years'], ['Maximum difference from original OOF prediction', '13.18 years']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 50, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | fMRI carries larger attribution in this fusion model | GNN_interpretability/modality_summary.csv', 'rows': [['Modality', 'Mean signed IG (yr)', 'Mean absolute connection IG', 'Occlusion error increase (yr)'], ['DWI', '-6.96', '45.17', '2.55'], ['fMRI', '10.33', '128.54', '3.36']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 52, 'table': 1, 'context': 'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS | Leading connections and stability offer complementary views | GNN_interpretability/connection_importance.csv; connection_fold_stability.csv', 'rows': [['Connection (fMRI)', 'Mean |IG| (yr)', 'Folds in top 100'], ['lh_precuneus ↔ rh_precuneus', '0.415', '5/5'], ['lh_entorhinal ↔ rh_entorhinal', '0.233', '5/5'], ['lh_isthmuscingulate ↔ lh_precuneus', '0.195', '5/5'], ['lh_inferiorparietal ↔ lh_precuneus', '0.183', '3/5'], ['lh_insula ↔ rh_insula', '0.170', '5/5']]}, {'deck': 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx', 'slide': 54, 'table': 1, 'context': 'APPENDIX | Result provenance and run selection | Paths are relative to the project results directory.', 'rows': [['Section', 'Primary saved evidence'], ['Harmonization', 'final/Fusion/report_tables/Table3_harmonization.csv'], ['DWI single modality', 'final/DWI/{Ridge_BAG,GNN_BAG}/*summary*.json / *metrics.json'], ['fMRI single modality', 'final/fMRI/{Ridge_BAG,GNN_BAG}'], ['Shared fusion benchmark', 'final/Fusion/report_tables/Table1,4,5,6*.csv'], ['Interpretability', 'explore/GNN_attribution/*.csv; analysis_summary.json']]}]


def extract_slide_tables(paths):
    """Extract exact displayed cell text, retaining slide/deck provenance."""
    result = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            slides = sorted((n for n in archive.namelist()
                             if re.fullmatch(r'ppt/slides/slide\d+.xml', n)),
                            key=lambda n: int(re.search(r'slide(\d+)', n).group(1)))
            for name in slides:
                root = ET.fromstring(archive.read(name))
                texts = [n.text or '' for n in root.findall('.//a:t', NS)]
                for i, table in enumerate(root.findall('.//a:tbl', NS), 1):
                    cells = [[' '.join(n.text or '' for n in c.findall('.//a:t', NS))
                              for c in row.findall('a:tc', NS)]
                             for row in table.findall('a:tr', NS)]
                    result.append(dict(deck=path.name,
                        slide=int(re.search(r'slide(\d+)', name).group(1)),
                        table=i, context=' | '.join(texts[:3]), rows=cells))
    return result


def metrics(y, predictions, weights=None):
    """Scan-weighted MAE/RMSE/R²/Pearson r; constant targets return NaN R²/r."""
    y = np.asarray(y, float)
    p = np.asarray(predictions, float)
    if p.ndim == 1:
        p = p[:, None]
    w = np.ones((1, len(y))) if weights is None else np.atleast_2d(weights)
    count = w.sum(axis=1, keepdims=True)
    ymean = (w @ y[:, None]) / count
    pmean = (w @ p) / count
    sse = w @ ((p - y[:, None]) ** 2)
    total = (w @ (y[:, None] ** 2)) - count * ymean**2
    pv = (w @ (p**2)) - count * pmean**2
    cov = (w @ (y[:, None] * p)) - count * ymean * pmean
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(total > 1e-10, 1 - sse / total, np.nan)
        corr = np.where((total > 1e-10) & (pv > 1e-10), cov / np.sqrt(total * pv), np.nan)
    return np.stack([(w @ abs(p-y[:, None])) / count,
                     np.sqrt(sse/count), r2, np.clip(corr, -1, 1)], axis=-1)


def summarize(frame, columns, bootstraps, seed, benchmark):
    rows = []
    for cohort in [*sorted(frame.dataset.unique()), 'Pooled']:
        sub = frame if cohort == 'Pooled' else frame[frame.dataset == cohort]
        if sub.participant_group.isna().any():
            raise ValueError('Missing participant group')
        groups, inv = np.unique(sub.participant_group.astype(str), return_inverse=True)
        y = sub.age.to_numpy(float)
        p = sub[list(columns.values())].to_numpy(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite ages/predictions; refusing model-specific row dropping')
        point = metrics(y, p)[0]
        # Resample whole participants, preserving all their visits. For pooled
        # estimates resample within cohort, keeping cohort participant counts fixed.
        rng = np.random.default_rng(seed)
        counts = np.zeros((bootstraps, len(groups)), dtype=int)
        group_cohort = sub.assign(group=sub.participant_group.astype(str)).groupby('group').dataset
        if (group_cohort.nunique() != 1).any():
            raise ValueError('Participant group crosses datasets')
        labels = group_cohort.first().reindex(groups).to_numpy()
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)), bootstraps)
        samples = metrics(y, p, counts[:, inv])
        for j, model in enumerate(columns):
            family, modality = MODELS[model]
            row = dict(benchmark=benchmark, dataset=cohort, model=model,
                       estimator=family, modality=modality, n_scans=len(sub),
                       n_participants=len(groups))
            for k, metric in enumerate(METRICS):
                valid = samples[:, j, k][np.isfinite(samples[:, j, k])]
                lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                row.update({metric:point[j,k], metric+'_ci_low':lo,
                            metric+'_ci_high':hi, metric+'_valid_bootstraps':len(valid)})
            rows.append(row)
    return pd.DataFrame(rows)


def load_paired(root):
    directory = root/'results/final/Fusion'
    data = pd.read_csv(directory/'GNN_fusion/CN_OOF_predictions.csv')
    keys = ['dataset', 'scan_id']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate paired scan keys')
    if data.groupby('participant_group').FOLD.nunique().max() != 1:
        raise ValueError('Participant visits cross folds')
    for folder in ['Ridge_baseline', 'Weighted_Ridge']:
        other = pd.read_csv(directory/folder/'CN_OOF_predictions.csv')
        joined = data.merge(other, on=keys, suffixes=('', '_other'), validate='one_to_one')
        if len(joined) != len(data) or len(other) != len(data):
            raise ValueError('Paired source scan sets differ')
        for col in other:
            if col.startswith('PREDICTED_AGE_') or col in ['unified_age', 'FOLD']:
                if not np.allclose(joined[col], joined[col+'_other'], rtol=0, atol=1e-8):
                    raise ValueError(f'Inconsistent {folder} {col}')
    return data.rename(columns={'unified_age':'age'})


def full_unimodal(root, bootstraps, seed):
    frames = []
    for modality in ['DWI', 'fMRI']:
        base = root/f'results/final/{modality}'
        ridge = pd.read_csv(base/f'Ridge_BAG/{modality}_Ridge_CN_OOF_predictions.csv')
        ridge = ridge.rename(columns={'DATASET':'dataset', 'PARTICIPANT_GROUP':'participant_group',
            'CHRONOLOGICAL_AGE':'age', 'PREDICTED_AGE_OOF':'prediction'})
        gnn = pd.read_csv(base/f'GNN_BAG/{modality}_GNN_BAG_CN_OOF_predictions.csv')
        gnn = gnn.rename(columns={'_manifest_dataset':'dataset', '_participant_group':'participant_group',
            '_age':'age', f'BRAIN_AGE_GNN_{modality}_OOF':'prediction'})
        # Verify models cover the same modality-specific scans and targets.
        a = ridge.rename(columns={'CONNECTOME_ID':'key'})
        b = gnn.rename(columns={'_manifest_id':'key'})
        paired = a.merge(b, on=['dataset','key'], suffixes=('_r','_g'), validate='one_to_one')
        if len(paired) != len(a) or len(a) != len(b) or not np.allclose(paired.age_r, paired.age_g):
            raise ValueError(f'{modality} full-cohort model samples differ')
        for frame, model in [(ridge, modality), (gnn, modality+'_GNN')]:
            frames.append(summarize(frame, {model:'prediction'}, bootstraps, seed,
                                    'Full unimodal CN OOF (different modality populations)'))
    return pd.concat(frames, ignore_index=True)


def plot_benchmark(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cohorts = [x for x in ['ADNI','ADRC','HABS','ADDECODE','ADDecode']
               if x in summary.dataset.unique()]
    titles = ['A. MAE (years)', 'B. RMSE (years)', 'C. R²', 'D. Pearson r']
    x = np.arange(len(cohorts)); width = .125
    for ax, metric, title in zip(axes.flat, METRICS, titles):
        for j, model in enumerate(MAIN):
            s = summary.set_index(['dataset','model']).loc[[(c,model) for c in cohorts]]
            positions = x + (j - 2.5)*width
            ax.bar(positions, s[metric], width=width*.94, color=COLORS[j],
                   edgecolor='white', linewidth=.4, hatch='//' if j >= 3 else None)
            # Draw percentile intervals directly; they need not contain the estimate.
            ax.vlines(positions, s[metric+'_ci_low'], s[metric+'_ci_high'], color='#333333', lw=.8)
            ax.hlines(s[metric+'_ci_low'], positions-.018, positions+.018, color='#333333', lw=.8)
            ax.hlines(s[metric+'_ci_high'], positions-.018, positions+.018, color='#333333', lw=.8)
        ax.set_title(title, loc='left', weight='bold', fontsize=13)
        ax.set_xticks(x, [f'{c}\nn={int(summary[summary.dataset==c].iloc[0].n_scans)} scans' for c in cohorts])
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.axhline(0, color='#777777', lw=.6)
        if metric in ['MAE','RMSE']:ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    labels = ['Ridge: DWI-only','Ridge: fMRI-only','Ridge: weighted early fusion',
              'GNN: DWI-only','GNN: fMRI-only','GNN: fusion']
    fig.legend(handles=[Patch(facecolor=c, label=l, hatch='//' if j>=3 else None)
                        for j,(c,l) in enumerate(zip(COLORS,labels))],
               loc='upper center', bbox_to_anchor=(.5,.955), ncol=3, frameon=False)
    fig.suptitle('Brain-age prediction by cohort and model', fontsize=19, weight='bold', y=.995)
    fig.text(.5,.025,'Raw CN out-of-fold predictions on matched scans; participant-cluster bootstrap 95% CIs.\n'
             'Lower MAE/RMSE and higher R²/r are better. HABS: 13 participants; ADDecode: no paired fusion data.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.88])
    for ext in ['png','pdf','svg']:
        fig.savefig(output/f'paired_model_performance.{ext}', dpi=220)
    plt.close(fig)


def html_table(rows):
    return '<table>'+''.join('<tr>'+''.join(
        f'<{"th" if i==0 else "td"}>{html.escape(str(cell))}</{"th" if i==0 else "td"}>'
        for cell in row)+'</tr>' for i,row in enumerate(rows))+'</table>'


def draw_table(rows, title, subtitle=''):
    widths = [.24,.32,.44] if len(rows[0])==3 else [1/len(rows[0])]*len(rows[0])
    wrapped = [[textwrap.fill(str(cell), max(12,int(150*widths[j])), break_long_words=True)
                for j,cell in enumerate(row)] for row in rows]
    heights = [max(len(c.splitlines()) for c in row)*.21+.22 for row in wrapped]
    fig, ax = plt.subplots(figsize=(16, max(3,sum(heights)+1.6)))
    ax.axis('off');ax.set_title(title+'\n'+subtitle, fontsize=13, loc='left', pad=18)
    table = ax.table(cellText=wrapped[1:], colLabels=wrapped[0], colWidths=widths,
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False);table.set_fontsize(9)
    available=sum(heights)
    for (i,j),cell in table.get_celld().items():
        cell.set_height(heights[i]/available*.94)
        cell.set_edgecolor('#D9E2EC')
        if i==0:cell.set_facecolor('#183B56');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#EDF4F8' if i%2 else 'white')
    fig.tight_layout()
    return fig


def formatted_rows(frame):
    rows=[['Model','Scans / participants','MAE [95% CI]','RMSE [95% CI]','R² [95% CI]','Pearson r [95% CI]']]
    for _,r in frame.iterrows():
        label=f'{r.estimator}: {r.modality}'
        values=[]
        for m in METRICS:
            precision=2 if m in ['MAE','RMSE'] else 3
            values.append(f'{r[m]:.{precision}f} [{r[m+"_ci_low"]:.{precision}f}, {r[m+"_ci_high"]:.{precision}f}]')
        rows.append([label,f'{r.n_scans} / {r.n_participants}',*values])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/brain_age')
    parser.add_argument('--bootstrap',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--slides',type=Path,nargs='*',help='Optional PPTX files; otherwise reproduce embedded reviewed tables')
    parser.add_argument('--tables-only',action='store_true',help='Reproduce slide tables without private prediction data')
    args=parser.parse_args()
    if args.bootstrap<100:parser.error('Use at least 100 bootstrap replicates')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    slide_tables=extract_slide_tables(args.slides) if args.slides else SLIDE_SNAPSHOT
    if not slide_tables:raise ValueError('No slide tables found')
    sections=[]
    table_dir=output/'slide_tables';table_dir.mkdir(exist_ok=True)
    with PdfPages(output/'all_slide_tables.pdf') as pdf:
        for i,t in enumerate(slide_tables,1):
            filename=f'table_{i:02d}_slide_{t["slide"]:02d}'
            with (table_dir/(filename+'.csv')).open('w',newline='') as f:csv.writer(f).writerows(t['rows'])
            caption=f'{t["deck"]} | Slide {t["slide"]}, table {t["table"]}'
            sections.append('<h2>'+html.escape(caption)+'</h2><p>'+html.escape(t['context'])+'</p>'+html_table(t['rows']))
            # Suppress missing-glyph warnings for mathematical notation; exact Unicode
            # text is preserved in CSV and HTML even when a local PDF font lacks it.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Glyph .* missing from .*font')
                fig=draw_table(t['rows'],f'Slide {t["slide"]}: table {t["table"]}',t['deck'])
                pdf.savefig(fig);plt.close(fig)
    (output/'slide_table_manifest.json').write_text(json.dumps(slide_tables,indent=2,ensure_ascii=False)+'\n')
    css='<style>body{font:15px Arial;margin:36px;color:#183B56}table{border-collapse:collapse;width:100%;margin-bottom:30px}td,th{border:1px solid #ccd7e0;padding:10px;text-align:left}th{background:#183B56;color:white}tr:nth-child(even){background:#edf4f8}img{max-width:100%}h2{margin-top:36px}</style>'
    (output/'all_slide_tables.html').write_text('<!doctype html><meta charset="utf-8">'+css+'<h1>All presentation tables</h1><p>Exact displayed values from both reviewed decks; snapshot prepared 2026-10-29. These are reproduced tables, not newly recomputed analyses. Archived IG tables describe reconstructed models.</p>'+''.join(sections))
    if args.tables_only:
        print(f'Reproduced {len(slide_tables)} slide tables in {output}');return
    data=load_paired(args.root)
    columns={m:f'PREDICTED_AGE_{m}_OOF' for m in MODELS}
    summary=summarize(data,columns,args.bootstrap,args.seed,'Matched CN OOF')
    summary.to_csv(output/'paired_metrics_all_models.csv',index=False)
    primary=summary[summary.model.isin(MAIN)]
    primary.to_csv(output/'paired_metrics_six_models.csv',index=False)
    full=full_unimodal(args.root,args.bootstrap,args.seed)
    full.to_csv(output/'full_unimodal_metrics.csv',index=False)
    availability=[]
    for cohort in ['ADNI','ADRC','HABS','ADDECODE']:
        for model in MAIN:
            for label, frame in [('Matched CN OOF',primary),('Full unimodal CN OOF',full)]:
                found=frame[(frame.dataset.str.upper()==cohort.upper()) & (frame.model==model)]
                availability.append(dict(dataset=cohort,model=model,benchmark=label,
                    n_scans=int(found.iloc[0].n_scans) if len(found) else 0,
                    status='Available' if len(found) else 'Not available in saved CN OOF predictions'))
    pd.DataFrame(availability).to_csv(output/'model_availability.csv',index=False)
    plot_benchmark(primary,output)
    body=['<h1>Brain-age prediction performance</h1><p>Raw CN OOF predictions; common paired benchmark. Ridge fusion means weighted early fusion. Additional fusion variants appear in the CSV.</p><img src="paired_model_performance.svg">']
    markdown=['# Brain-age prediction performance','', 'Raw CN OOF predictions; participant-cluster bootstrap 95% confidence intervals. Ridge fusion = weighted early fusion.','']
    with PdfPages(output/'performance_tables.pdf') as pdf:
        for cohort in [*sorted(set(primary.dataset.unique())-{'Pooled'}),'Pooled']:
            sub=primary[primary.dataset==cohort].set_index('model').loc[MAIN].reset_index()
            rows=formatted_rows(sub)
            body.extend(['<h2>'+cohort+'</h2>',html_table(rows)])
            markdown.extend(['## '+cohort,'','| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |',*['| '+' | '.join(r)+' |' for r in rows[1:]],''])
            fig=draw_table(rows,cohort+' — matched CN OOF performance','MAE/RMSE in years; participant-cluster bootstrap 95% CIs')
            pdf.savefig(fig);fig.savefig(output/f'performance_table_{cohort}.png',dpi=180);plt.close(fig)
    body.append('<h2>Full unimodal populations (supplement)</h2><p>DWI and fMRI use different populations; do not compare these estimates as a matched fusion benchmark. ADDecode has no saved CN OOF predictions, including DWI; its metrics are not available.</p>')
    for cohort in sorted(full.dataset.unique()):body.extend(['<h3>'+cohort+'</h3>',html_table(formatted_rows(full[full.dataset==cohort]))])
    (output/'performance_report.html').write_text('<!doctype html><meta charset="utf-8">'+css+''.join(body)+'<p><a href="all_slide_tables.html">All original slide tables</a></p>')
    (output/'performance_tables.md').write_text('\n'.join(markdown))
    sources=list((args.root/'results/final/Fusion').glob('*/CN_OOF_predictions.csv'))
    sources+=list((args.root/'results/final').glob('*/Ridge_BAG/*CN_OOF_predictions.csv'))
    sources+=list((args.root/'results/final').glob('*/GNN_BAG/*CN_OOF_predictions.csv'))
    provenance=dict(bootstrap_replicates=args.bootstrap,seed=args.seed,ci='Participant-cluster percentile 95% CI; pooled resampling stratified by cohort; fixed OOF predictions, no refitting',point_estimates='Scan-weighted; all participant visits kept together in bootstrap',slide_tables=len(slide_tables),sources={str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},limitations=['Upstream ComBat used all scans and age covariates; this is not fold-contained harmonization.','Intervals condition on fitted models and existing model/threshold selection; no model-retraining uncertainty or significance claim.','HABS paired benchmark has only 13 participants; ADDecode has no saved CN OOF predictions for any model.','Slide snapshots preserve displayed historical values; IG tables describe reconstructed models, not verified original checkpoints.','CSV and HTML retain exact slide-table Unicode; PDF mathematical glyph support depends on installed fonts.'])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Created {len(summary)} paired rows, {len(full)} full-unimodal rows, and {len(slide_tables)} slide tables in {output}')


if __name__=='__main__':main()
