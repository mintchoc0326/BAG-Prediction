"""Update the comprehensive deck from current aggregate evidence; retain historical figures.

Run with python-pptx available. No participant-level inputs are read.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

try:
    import pptx
except ImportError:
    sys.path.insert(0, '/tmp/avery_slides_deps')
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.opc.packuri import PackURI
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/presentation'
NAME = 'Connectome_Brain_Age_Results_English_Editable_with_Pipelines'
TARGET = OUT / (NAME + '.pptx')
PUBLISHED = ROOT / 'reports/brain_age/slides'
BACKUP = OUT / 'archive_before_pipeline_update_20261109'
N = ROOT / 'reports/new_pipeline'
G = N / 'gine_study'
BACKUP.mkdir(exist_ok=True)
for ext in ('pptx', 'pdf'):
    src = OUT / (NAME + '.' + ext)
    dst = BACKUP / src.name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

prs = Presentation(BACKUP / (NAME + '.pptx'))
W, H = prs.slide_width / Inches(1), prs.slide_height / Inches(1)
INK, MUTED, ACCENT, GRID = '222222', '526477', '007F86', 'DCE3E8'
sources = set()
checks = []


def records(path):
    sources.add(path.relative_to(ROOT).as_posix())
    with path.open() as f:
        return list(csv.DictReader(f))


def txt(s, x, y, w, h, value, size=19, bold=False, color=INK):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.03)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for i, line in enumerate(str(value).split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = 'Arial'
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(6)
    return sh


def new(title, source, note='', section='CURRENT PIPELINE | 9 NOV 2026'):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string('FFFFFF')
    txt(s, .48, .18, 12.2, .3, section, 10, True, ACCENT)
    txt(s, .48, .65, 12.3, .8, title, 27, True)
    txt(s, .5, 7.06, 11.5, .27, source, 8, color=MUTED)
    s.notes_slide.notes_text_frame.text = note + '\nEvidence: ' + source
    return s


def bullets(title, lines, source, takeaway='', note=''):
    s = new(title, source, note)
    y = 1.65
    step = min(.94, 4.8 / max(len(lines), 1))
    for line in lines:
        txt(s, .68, y, 11.9, step - .08, '• ' + line, 20 if len(lines) <= 5 else 18)
        y += step
    if takeaway:
        txt(s, .68, 6.47, 11.9, .48, takeaway, 16, True, ACCENT)
    return s


def table(title, headers, rows, source, takeaway='', widths=None, note=''):
    s = new(title, source, note)
    height = min(4.65, .6 * (len(rows) + 1))
    tb = s.shapes.add_table(len(rows) + 1, len(headers), Inches(.65), Inches(1.65),
                            Inches(12.0), Inches(height)).table
    if widths:
        for c, w in zip(tb.columns, widths):
            c.width = Inches(w)
    for i, row in enumerate([headers] + rows):
        for j, v in enumerate(row):
            c = tb.cell(i, j)
            c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string('EDF0F3' if i == 0 else 'FFFFFF')
            c.margin_left = c.margin_right = Inches(.10)
            c.margin_top = c.margin_bottom = Inches(.06)
            for p in c.text_frame.paragraphs:
                p.font.name = 'Arial'
                p.font.size = Pt(16 if len(rows) <= 7 else 14)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor.from_string(INK)
    if takeaway:
        txt(s, .68, 6.4, 11.9, .55, takeaway, 16, True, ACCENT)
    return s


def flow(title, steps, source, takeaway, note=''):
    s = new(title, source, note)
    n = len(steps)
    boxw = (12.0 - (n - 1) * .30) / n
    for i, (heading, body) in enumerate(steps):
        x = .65 + i * (boxw + .30)
        sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2),
                                Inches(boxw), Inches(3.3))
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string('F7F9FA')
        sh.line.color.rgb = RGBColor.from_string(GRID)
        txt(s, x + .12, 2.16, boxw - .24, .75, heading, 19, True, ACCENT)
        txt(s, x + .12, 3.02, boxw - .24, 2.0, body, 17)
        if i < n - 1:
            txt(s, x + boxw, 3.35, .30, .5, '→', 20, True)
    txt(s, .68, 6.1, 11.9, .75, takeaway, 18, True)
    return s


def figure(title, path, source, takeaway):
    sources.add(path.relative_to(ROOT).as_posix())
    s = new(title, source)
    iw, ih = Image.open(path).size
    scale = min(11.8 / iw, 4.65 / ih)
    ww, hh = iw * scale, ih * scale
    s.shapes.add_picture(str(path), Inches((W - ww) / 2), Inches(1.55),
                         width=Inches(ww), height=Inches(hh))
    txt(s, .68, 6.43, 11.9, .50, takeaway, 16, True, ACCENT)
    return s


def f(row, field='MAE', digits=2):
    return f'{float(row[field]):.{digits}f}'


def select(rows, **keys):
    found = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in keys.items())]
    assert len(found) == 1, (keys, len(found))
    return found[0]


ridge = records(N / 'matched_ridge/performance.csv')
compare = records(G / 'ridge_gine_comparison.csv')
delta = records(G / 'paired_MAE_differences.csv')
harm = records(G / 'harmonization_paired_MAE_differences.csv')
stop = records(G / 'early_stopping_summary.csv')
study = records(G / 'study_status.csv')
counts = records(N / 'sample_counts.csv')
cal = records(N / 'matched_ridge/age_calibration.csv')

# Keep original figures and numerical results only as a clearly separated appendix.
historical_numbers = {15, 16, 17, 18, 26, 27, 33, 39, 40, 41, 46, 49, 50, 51, 52}
historical = []
for i, slide in enumerate(list(prs.slides), 1):
    sid = prs.slides._sldIdLst[i - 1]
    if i in historical_numbers:
        historical.append(sid)
    else:
        checks.append({'removed_old_slide': i})
for i in range(len(prs.slides) - 1, -1, -1):
    sid = prs.slides._sldIdLst[i]
    if sid not in historical:
        prs.part.drop_rel(sid.rId)
        prs.slides._sldIdLst.remove(sid)

# python-pptx allocates new slide names from the slide count. Rename retained
# parts so deleted original slide numbers cannot collide with new slide parts.
for i, s in enumerate(prs.slides, 1):
    s.part._partname = PackURI(f'/ppt/slides/slideHistorical{i}.xml')

s = new('Connectome Brain Age: Audit, Revised Pipeline and Results',
        'Current source + audited aggregate runs | Updated 9 November 2026')
txt(s, .72, 1.85, 11.8, 1.0, 'Participant-nested evaluation of DWI, signed fMRI and early fusion', 30, True)
txt(s, .72, 3.25, 11.8, 1.7, 'Audit findings → training-only transformations → nested prediction\nMatched Ridge and full-budget GINE → calibration and remaining validation', 23)
txt(s, .72, 5.7, 11.8, .8, 'Current development results first; historical benchmarks and reconstructed attribution in the appendix.', 19, color=MUTED)

bullets('Three points to take away', [
    'The audit identified full-sample preprocessing, outer-score threshold selection and mixed calibration provenance.',
    'The revised evaluation nests preprocessing, selection and stopping within participant-level training partitions.',
    'Matched Ridge and completed GINE arms are available; clinical source validation and anatomical interpretation remain unresolved.'
], 'audit_findings.tsv; audit_to_implementation.csv; completed-run audit.json',
    'Implementation validation supports the evaluation design; it does not verify the underlying clinical data.')

table('Audit findings and the implemented response', ['Finding', 'Revised behavior'], [
    ['Global ComBat + full-sample feature mask', 'Training-only QC, ComBat and scaling at every fit boundary'],
    ['Threshold chosen using reported OOF scores', 'Choose sparsity within inner validation'],
    ['Stopping and selection need separation', 'Separate inner stopping split; fixed-epoch outer refit'],
    ['CN final predictions include in-sample fits', 'Evaluate held-out predictions; calibrate from training inner OOF'],
    ['Unverified pairing and reconstructed attribution', 'Explicit pairing exclusions; original checkpoint attribution']
], 'reports/pipeline_audit/audit_findings.tsv; reports/new_pipeline/audit_to_implementation.csv', widths=[4.3, 7.7])

flow('The revised pipeline: fitting stays within training', [
    ('1  Register', 'Raw connectomes\nMetadata evidence\nCanonical participant groups'),
    ('2  Split', 'Frozen outer folds\nCN training reference\nSeparate inner / stopping folds'),
    ('3  Fit', 'Training-only QC\nAdditive ComBat\nScaling + selected model'),
    ('4  Evaluate', 'Outer held-out age\nTraining-OOF calibration\nSaved states and audits')
], 'code/brainage_pipeline/{data,splits,preprocessing,engine,calibration}.py',
    'No outer-test age enters feature transformation, configuration choice or early stopping.')

table('Current eligible populations and matched comparison', ['Population', 'Scans', 'Participants', 'CN scans'],
      [[r['population'], r['scans'], r['participants'], r['CN_scans']] for r in counts],
      'reports/new_pipeline/sample_counts.csv',
      'Matched CN performance uses 237 scans / 189 participants; the full paired registry has 318 scans / 252 participants.')

flow('Cross-modality pairing: conservative exclusions', [
    ('575 keys', 'Common scan keys\nUnique source matching\nMetadata agreement required'),
    ('250 excluded', 'Age disagreement\nTolerance: 0.01 years\nNo age averaging'),
    ('7 excluded', 'Insufficient paired\nCN batch support\nNested fits must be feasible'),
    ('318 retained', '252 participants\n237 CN scans\nDates still require verification')
], 'reports/new_pipeline/pairing_counts.csv; code/brainage_pipeline/data.py',
    'Agreeing keys and metadata do not establish contemporaneous DWI and fMRI acquisition.')

table('Inputs, transforms and notation', ['Symbol / object', 'Definition'], [
    ['Aᴰ, Aᶠ', 'Symmetric 84 × 84 DWI and fMRI connectivity matrices'],
    ['xᴰ ∈ R³⁴⁸⁶', 'Upper-triangle DWI edges; fixed log1p transformation'],
    ['xᶠ ∈ R³⁴⁸⁶', 'Upper-triangle signed fMRI edges; clipped Fisher-z transformation'],
    ['b, y, g', 'Known acquisition batch; age target; canonical participant group'],
    ['CN reference', 'Diagnosis defines reference eligibility; it is not an imaging predictor']
], 'code/brainage_pipeline/data.py; code/final/{harmonization,foldwise_fmri}.py',
    'Age is the supervised target; no sex, APOE or metabolic variables enter the age-prediction features.')

bullets('Feature QC and edge reconstruction are training-only', [
    'Apply fixed modality transforms before learning feature statistics.',
    'Fit feature masks on the corresponding training rows, including nonconstant and residual-variance checks.',
    'Fit harmonization and StandardScaler only after training-only edge selection.',
    'Restore graph values to their original ROI-pair positions; excluded edges map to scaled zero and cannot enter topology.',
    'Frozen transforms reject unsupported batches and invalid input instead of silently adapting to test data.'
], 'code/final/harmonization.py; code/brainage_pipeline/preprocessing.py')

bullets('Harmonization: compare declared modes', [
    'No-ComBat control retains the same training-only QC and scaling boundaries.',
    'Training-age mean-only ComBat estimates additive batch offsets using training ages.',
    'Held-out transformation requires imaging and known batch metadata, without chronological age.',
    'Batch-only ComBat is an available sensitivity mode; the completed matched comparisons report none versus age_mean.',
    'Known-batch eligibility restricts the population; this does not evaluate generalization to an unseen acquisition batch.'
], 'code/brainage_pipeline/preprocessing.py; code/final/harmonization.py',
    'Reduced technical predictability is not proof that all biological variation is preserved.')

table('Additive harmonization: fit once, apply frozen offsets', ['Step', 'Operation'], [
    ['Fit on training rows', 'Estimate training-age-adjusted γ̂[b,j] and pooled scale σ̂[j]'],
    ['Mean-only adjustment', 'xᴴ[i,j] = x[i,j] − σ̂[j] γ̂[bᵢ,j]'],
    ['Training scaling', 'z[i,j] = (xᴴ[i,j] − μ̂train[j]) / ŝtrain[j]'],
    ['Apply to held-out rows', 'Reuse mask, offsets and scaler; do not re-estimate parameters'],
    ['Age boundary', 'Training age informs offsets; test age is unnecessary for this transform']
], 'code/final/harmonization.py::_adjust; code/brainage_pipeline/preprocessing.py',
    'The mean-only age terms cancel during application; this formula is not the general location-scale ComBat rule.')

bullets('Harmonization diagnostics require held-out evidence', [
    'Compare age-prediction performance under no-ComBat and additive harmonization.',
    'Assess technical and sex predictability on held-out representations with training-fitted probes.',
    'Fit PCA to training representations and project held-out scans in a shared coordinate system.',
    'Inspect cohort-specific age distributions and within-cohort cBAG residual trends.',
    'Technical batch, cohort and biology overlap; visual mixing and probe scores alone cannot establish preservation.'
], 'code/brainage_pipeline/diagnostics.py; code/brainage_pipeline/reporting.py')

flow('Nested evaluation: outer test is reserved for scoring', [
    ('Outer folds', '5 participant folds\nAll visits grouped\nTrain on CN reference'),
    ('Inner folds', '3 participant folds\nFit preprocessing anew\nSelect α or graph sparsity'),
    ('Outer refit', 'Refit training transforms\nFit selected configuration\nUse fixed GINE epochs'),
    ('Outer test', 'One held-out prediction\nFrozen calibration\nNo selection or stopping')
], 'code/brainage_pipeline/{splits,engine}.py; configs/ridge.json',
    'DWI, fMRI and fusion share paired observations and frozen participant folds.')

flow('GINE stopping is separate from inner validation', [
    ('Early training', 'Subset of inner training\nFit own preprocessing\nOptimize model weights'),
    ('Stopping split', 'Participant-disjoint\nSelect epoch count\nNo inner validation access'),
    ('Inner refit', 'Refit all inner training\nUse selected epochs\nThen score inner validation'),
    ('Outer refit', 'All outer CN training\nMedian selected epochs\nNo outer early stopping')
], 'code/brainage_pipeline/engine.py; code/brainage_pipeline/gnn.py',
    'Maximum 250 epochs, patience 25; stored histories distinguish optimization, stopping and refit stages.')

bullets('Ridge: a fully nested imaging baseline', [
    'Fit a linear age predictor using transformed, training-standardized imaging edges.',
    'Choose α from 0.1, 1, 10, 100, 1,000, 10,000 and 100,000 using inner OOF MAE.',
    'Re-estimate preprocessing and Ridge on all outer CN training rows after selecting α.',
    'Early fusion concatenates independently standardized DWI and fMRI features.',
    'Evaluate raw held-out age predictions; BAG calibration is a separate downstream operation.'
], 'code/brainage_pipeline/engine.py; configs/ridge.json',
    'β̂ = argminβ ||y − Xβ − c||² + α||β||²; α is never selected from outer-test scores.')

table('Graph construction: signed features and per-scan topology', ['Component', 'Current definition'], [
    ['Nodes', '84 ROIs; each node contains its complete standardized connectivity profile'],
    ['DWI topology', 'Positive raw connectivity; retain per-scan top 25% or 50% of eligible edges'],
    ['fMRI topology', 'Rank raw magnitude; retain positive and negative connections'],
    ['Edge attributes', 'Signed transformed / standardized edge values'],
    ['QC and sparsity', 'Training mask removes invalid edges; sparsity fraction selected by inner validation'],
    ['Directionality', 'Each selected undirected pair becomes two directed edges for message passing']
], 'code/brainage_pipeline/gnn.py::graphs',
    'Unimodal node matrix: 84 × 84; edge attributes: one channel.')

flow('Current GINE architecture', [
    ('Graph input', 'Node profiles\nSigned edge attributes\nSelected topology'),
    ('GINE layers', 'MLP message updates\nBatch normalization\nReLU + dropout'),
    ('Readout', 'Global mean pooling\nOptional residual skips\nNo max-pooling branch'),
    ('Age head', 'Width → width/2 → 1\nTraining target scaling\nReturn age in years')
], 'code/brainage_pipeline/gnn.py::GINE',
    'AdamW: learning rate 5×10⁻⁴; weight decay 5×10⁻⁴; dropout 0.3; batch size 32.')

table('GINE notation and capacity arms', ['Quantity', 'Definition'], [
    ['Message update', 'hᵥ′ = MLP((1 + ε)hᵥ + Σᵤ ReLU(hᵤ + E(eᵤᵥ)))'],
    ['Layer output', 'ReLU(BN(hᵥ′) + S(hᵥ)) for residual arms; omit S otherwise'],
    ['Readout / age', 'g = meanᵥ hᵥ; age = μtrain + σtrain × head(g)'],
    ['Prespecified capacities', '1 layer × 16; 2 × 32; 3 × 64; 2 × 32 with residual skips'],
    ['Within-arm selection', 'Capacity fixed for each ablation; select sparsity inside inner validation']
], 'code/brainage_pipeline/gnn.py; configs/gine_*.json',
    'Outer-score capacity comparisons are development analyses; choosing a winner needs further validation.')

flow('Fusion now combines modalities at the input level', [
    ('DWI branch', 'Training-fit transform\nStandardized profile\nModality-specific mask'),
    ('fMRI branch', 'Training-fit transform\nSigned profile\nModality-specific mask'),
    ('Early fusion', 'Ridge: concatenate edges\nGINE: concatenate profiles\nUnion graph topology'),
    ('One predictor', 'One Ridge model or\none GINE encoder\nMatched participant folds')
], 'code/brainage_pipeline/{preprocessing,gnn,engine}.py',
    'Historical weighted / late Ridge and dual-encoder GNN results are retained only in the appendix.')

table('Fusion dimensions and modality alignment', ['Object', 'Current representation'], [
    ['Ridge fusion vector', 'Concatenated retained DWI and fMRI features; width depends on training masks'],
    ['GINE node input', '[DWI profile | fMRI profile] ∈ R¹⁶⁸ for each of 84 nodes'],
    ['GINE edge attributes', '[standardized DWI value, standardized fMRI value] ∈ R²'],
    ['Fusion topology', 'Union of per-modality selected ROI pairs; bidirectional message passing'],
    ['Comparison boundary', 'Same observations and participant folds as the unimodal matched models']
], 'code/brainage_pipeline/gnn.py::graphs; preprocessing.py::matrix')

flow('BAG calibration uses the outer training reference', [
    ('Inner OOF', 'CN outer-training rows\nSelected model candidate\nPredict without own inner fit'),
    ('Fit age trend', 'BAG = prediction − age\nGlobal and cohort trends\nRecord sparse-cohort fallback'),
    ('Outer prediction', 'Model fitted on outer train\nPredict held-out scans\nKeep age-model output raw'),
    ('Apply correction', 'Subtract frozen age trend\nSave global and cohort cBAG\nNo test-coefficient fitting')
], 'code/brainage_pipeline/calibration.py; engine.py',
    'Calibration uses selected-candidate inner OOF predictions; external confirmation is still needed.')

table('BAG and cBAG: equations and reference limits', ['Quantity', 'Definition'], [
    ['Raw BAG', 'BAGᵢ = ŷᵢ − yᵢ'],
    ['Training-reference trend', 'Fit BAGinner-OOF = a + b × age on CN outer-training rows'],
    ['Global cBAG', 'cBAGglobal,ᵢ = ŷᵢ − yᵢ − (aglobal + bglobal × yᵢ)'],
    ['Cohort-aware cBAG', 'Use cohort-specific a,b when ≥10 training reference scans and age varies'],
    ['Fallback / age usage', 'Sparse cohorts use global trend; test age is used for BAG arithmetic, not imaging transformation']
], 'code/brainage_pipeline/calibration.py::BiasCorrection',
    'Neither low pooled age correlation nor corrected BAG alone establishes a clinical biomarker.')

bullets('Prediction and calibration provenance is auditable', [
    'Save original outer model weights, preprocessing state and selected configuration.',
    'Persist training indices, participant folds, inner OOF calibration predictions and coefficients locally.',
    'Audit split boundaries, training-only fit indices, complete outer predictions and reconstructed cBAG arithmetic.',
    'Record data/source fingerprints; incompatible resume requests are rejected.',
    'Synthetic boundary tests validate the software; source diagnoses, dates, protocols and ROI ordering need independent evidence.'
], 'code/brainage_pipeline/audit.py; code/test_pipeline.py; completed-run audit.json')

rows = []
for modality in ['DWI', 'fMRI', 'fusion']:
    a = select(ridge, mode='age_mean', modality=modality, cohort='Pooled')
    b = select(ridge, mode='none', modality=modality, cohort='Pooled')
    rows.append([modality, f(a), f"{f(a, 'MAE_CI_low')}–{f(a, 'MAE_CI_high')}", f(a, 'R2', 3), f(b)])
table('Matched Ridge: current held-out age performance',
      ['Modality', 'Additive MAE', '95% CI', 'R²', 'No-ComBat MAE'], rows,
      'reports/new_pipeline/matched_ridge/performance.csv',
      '237 CN scans / 189 participants; fusion does not improve on DWI Ridge in this matched population.',
      note='MAE in years. Intervals: 3,000 cohort-stratified participant bootstrap draws of fixed OOF predictions; no refitting uncertainty.')

arms = ['ridge', 'gine_1x16', 'gine_2x32', 'gine_3x64', 'gine_2x32_skip']
for mode, label in [('age_mean', 'Additive harmonization'), ('none', 'No ComBat')]:
    rows = []
    for arm in arms:
        vals = [f(select(compare, arm=arm, mode=mode, modality=m, cohort='Pooled')) for m in ['DWI', 'fMRI', 'fusion']]
        rows.append([arm.replace('gine_', 'GINE ').replace('ridge', 'Ridge'), *vals])
    table('Full-budget comparison: ' + label, ['Model arm', 'DWI MAE', 'fMRI MAE', 'Fusion MAE'], rows,
          'reports/new_pipeline/gine_study/ridge_gine_comparison.csv',
          'All arms use the same 237 CN scans / 189 participants and participant folds; MAE in years.')

rows = []
for m in ['DWI', 'fMRI', 'fusion']:
    r = select(delta, arm='gine_2x32', reference='ridge', mode='age_mean', modality=m)
    rows.append([m, f(r, 'delta_MAE'), f"{f(r, 'CI_low')}–{f(r, 'CI_high')}"])
table('GINE 2×32 versus Ridge: paired differences', ['Modality', 'ΔMAE (GINE − Ridge)', '95% fixed-OOF CI'], rows,
      'reports/new_pipeline/gine_study/paired_MAE_differences.csv',
      'Positive values favor Ridge. These intervals condition on trained OOF models; they do not include model refitting.')

figure('Capacity and residual comparisons', G / 'architecture_comparison.png',
       'reports/new_pipeline/gine_study/architecture_comparison.png; performance.csv',
       'Report every prespecified arm; capacity selection based on these outer scores requires further validation.')

rows = []
for arm in ['ridge', 'gine_2x32']:
    for m in ['DWI', 'fMRI', 'fusion']:
        r = select(harm, arm=arm, modality=m)
        rows.append([arm, m, f(r, 'delta_MAE', 3), f"{f(r, 'CI_low', 3)}–{f(r, 'CI_high', 3)}"])
table('Additive ComBat versus no ComBat: paired sensitivity', ['Arm', 'Modality', 'ΔMAE (additive − none)', '95% CI'], rows,
      'reports/new_pipeline/gine_study/harmonization_paired_MAE_differences.csv',
      'No consistent accuracy gain across these contrasts; the preprocessing mode is declared rather than selected on outer scores.')

rows = []
for cohort in ['ADNI', 'ADRC', 'HABS']:
    r = select(ridge, mode='age_mean', modality='DWI', cohort=cohort)
    rows.append([cohort, r['n_scans'], r['n_participants'], f(r), f(r, 'R2', 3)])
table('Cohort-specific performance: DWI Ridge', ['Cohort', 'CN scans', 'Participants', 'MAE (yr)', 'R²'], rows,
      'reports/new_pipeline/matched_ridge/performance.csv',
      'Pooled performance mixes cohort age distributions; HABS has only six retained CN scans.')

rows = []
for cohort in ['Pooled', 'ADNI', 'ADRC', 'HABS']:
    vals = [select(cal, mode='age_mean', modality='DWI', cohort=cohort, measure=v) for v in ['BAG', 'cBAG_global', 'cBAG']]
    rows.append([cohort, *[f(r, 'age_r', 3) for r in vals]])
table('Calibration diagnostics: inspect each cohort', ['DWI Ridge', 'r(age, BAG)', 'r(age, global cBAG)', 'r(age, cohort cBAG)'], rows,
      'reports/new_pipeline/matched_ridge/age_calibration.csv',
      'These are held-out residual diagnostics, not evidence of clinical validity or complete age independence.')

for m in ['DWI', 'fMRI', 'fusion']:
    figure('Training size: ' + m + ' GINE 2×32', G / f'training_size_gine_2x32/training_size_age_mean_{m}.png',
           'gine_study/training_size_gine_2x32/training_size_metrics.csv | additive mode',
           'Mean outer-fold MAE, not pooled MAE; fixed outer tests. 25%, 50% and 60% failed batch-support preflight.')

size_gine = records(G / 'training_size_gine_2x32/training_size_metrics.csv')
size_ridge = records(G / 'training_size_ridge/training_size_metrics.csv')
rows = []
for m in ['DWI', 'fMRI', 'fusion']:
    for fraction in [.75, .9, 1.0]:
        values = []
        counts_here = []
        for data in [size_ridge, size_gine]:
            part = [r for r in data if r['mode'] == 'age_mean' and r['modality'] == m
                    and float(r['training_fraction']) == fraction]
            assert len(part) == 5
            values.append(f"{sum(float(r['MAE']) for r in part) / len(part):.2f}")
            counts_here.append(sum(float(r['n_training_CN']) for r in part) / len(part))
        assert counts_here[0] == counts_here[1]
        rows.append([m, f'{fraction:.0%}', f'{counts_here[0]:.1f}', *values])
table('Training-size comparison: Ridge and GINE 2×32',
      ['Modality', 'Fraction', 'Mean CN train scans', 'Ridge MAE', 'GINE MAE'], rows,
      'gine_study/training_size_{ridge,gine_2x32}/training_size_metrics.csv',
      'Additive mode; arithmetic mean of five outer-fold MAEs. Fractions refer to participant groups, not scan counts.')

rows = []
for arm in ['gine_1x16', 'gine_2x32', 'gine_3x64', 'gine_2x32_skip']:
    r = select(stop, arm=arm, fraction='1.0', mode='age_mean', modality='DWI')
    rows.append([arm.replace('gine_', ''), r['parameters'], f(r, 'selected_epoch_median', 0),
                 f(r, 'mean_training_MAE'), f(r, 'mean_stopping_MAE')])
table('Learning diagnostics: DWI additive-mode GINE', ['Arm', 'Parameters', 'Median epoch', 'Train MAE', 'Stopping MAE'], rows,
      'reports/new_pipeline/gine_study/early_stopping_summary.csv',
      'Training and stopping MAEs are inner early-fit diagnostics, not outer-test performance.')

completed = [r for r in study if r['status'] == 'complete']
assert len(completed) == 8, 'Reassess slide status if the experiment inventory changes'
bullets('What has completed, and what remains unverified', [
    'Matched Ridge: 30 outer tasks; complete predictions for six modality × preprocessing combinations.',
    'Four full-training GINE capacity / residual arms: each completed 30 outer tasks and passed its artifact audit.',
    '75% and 90% training-size runs completed for Ridge and GINE 2×32; 25%, 50% and 60% failed batch-support preflight.',
    'Short-budget GINE validation is software evidence and is excluded from trained-model comparisons.',
    'Clinical labels, acquisition dates/protocols and ROI ordering remain unverified.'
], 'gine_study/study_status.csv; matched_ridge and GINE run audit.json',
    'Status is taken from per-run completion records, rather than older summary text.')

flow('Interpretability uses original saved outer checkpoints', [
    ('Load', 'Original outer weights\nExact preprocessing\nMatching selected config'),
    ('Verify', 'Held-out sample identity\nPrediction consistency\nSource / state provenance'),
    ('Compute', 'Gradient × input\nNode / edge sensitivity\nHeld-out embeddings'),
    ('Interpret', 'Numeric ROI indices first\nLabels require verified order\nNo causal claims')
], 'code/brainage_pipeline/downstream.py',
    'The current method is local gradient-times-input sensitivity; archived integrated-gradient figures used reconstructed models.')

bullets('Embeddings and anatomical labels need separate checks', [
    'Extract embeddings from the original outer checkpoint for training and held-out observations.',
    'Fit the PCA basis to training embeddings separately for each fold, then project that fold’s held-out observations.',
    'Do not pool raw embedding coordinates across independently fitted models.',
    'Age, cohort, sex and verified clinical variables are overlays rather than age-prediction inputs.',
    'Keep ROI labels numeric until the 84-region order is verified; network assignments are not guessed.'
], 'code/brainage_pipeline/downstream.py; diagnostics.py')

bullets('Clinical analysis: implemented interface, gated inference', [
    'Verify source diagnosis codes and the CN reference before interpreting clinical group effects.',
    'Confirm imaging-to-clinical visit alignment, APOE coding, diabetes definition and HbA1c units.',
    'The interface supports complete-case analyses with cohort, age and sex covariates, participant-cluster uncertainty and BH adjustment.',
    'Clinical inference requires a reviewed specification and nonpilot held-out predictions.',
    'Generic diabetes fields do not establish type 2 diabetes; missing definitions are not filled in by assumption.'
], 'code/brainage_pipeline/clinical.py; configs/clinical_spec.json',
    'No verified clinical association or anatomical finding is claimed from these implementation checks.')

bullets('Limits that still determine interpretation', [
    'This is a development reanalysis following inspection of earlier results, rather than untouched external validation.',
    'Known-batch eligibility and pairing exclusions produce a restricted matched population.',
    'Canonical IDs, diagnosis provenance, visit dates and ADRC acquisition protocols need source confirmation.',
    'Bootstrap intervals describe uncertainty in fixed OOF predictions and omit refitting uncertainty.',
    'Residual cohort effects and unverified ROI order constrain biomarker and anatomical conclusions.'
], 'code/brainage_pipeline/README.md; gine_study/README.md')

bullets('Next decisions for the project', [
    'Prioritize source verification of reference diagnosis, participant identity and imaging / clinical visit dates.',
    'Resolve metadata conflicts with source evidence before expanding the paired population.',
    'Use a prespecified validation design before choosing a final capacity or preprocessing mode from development results.',
    'Verify ROI ordering before attaching anatomical names to checkpoint-based sensitivity.',
    'Run clinical association analyses only after definitions, units and alignment have been reviewed.'
], 'Audit follow-up | source-data verification and validation priorities')

table('Reproduction and result provenance', ['Artifact', 'Authoritative location'], [
    ['Current entry point', 'code/run_pipeline.py'],
    ['Methods / configuration', 'code/brainage_pipeline/ and configs/'],
    ['Matched Ridge aggregates', 'reports/new_pipeline/matched_ridge/'],
    ['Full-budget study aggregates', 'reports/new_pipeline/gine_study/'],
    ['Original audit evidence', 'reports/pipeline_audit/audit_findings.tsv'],
    ['This editable deck builder', 'code/final/slides/update_comprehensive_pipeline.py']
], 'Local runtime predictions/checkpoints stay under ignored results/final/new_pipeline/',
    'Slide tables are loaded from aggregate CSVs; historical figures below retain their original numbers.')

bullets('Appendix: historical benchmarks and reconstructed attribution', [
    'The following figures preserve the earlier full-sample harmonization and 387-CN-scan benchmark.',
    'They use different populations and evaluation procedures from the current 237-CN-scan matched results.',
    'Historical graph thresholds were compared using their reported OOF scores.',
    'Historical attribution used reconstructed models; it is not current original-checkpoint sensitivity.',
    'These archived numbers are retained for context and must not be read as updated validation results.'
], 'Original comprehensive deck snapshot retained in results/presentation/archive_before_pipeline_update_20261109/')

# Move the preserved original slides after all current content.
for sid in historical:
    prs.slides._sldIdLst.remove(sid)
    prs.slides._sldIdLst.append(sid)
for sid in historical:
    s = next(sl for sl in prs.slides if sl.slide_id == int(sid.id))
    for sh in list(s.shapes):
        if sh.has_text_frame and sh.top < Inches(.5):
            sh._element.getparent().remove(sh._element)
        elif sh.has_text_frame and abs(sh.top - Inches(.65)) < Inches(.02):
            sh.text = 'Historical: ' + sh.text.removeprefix('Historical: ')
            for p in sh.text_frame.paragraphs:
                p.font.name = 'Arial'
                p.font.size = Pt(24)
                p.font.bold = True
                p.font.color.rgb = RGBColor.from_string(INK)
    txt(s, .48, .18, 12.2, .3, 'HISTORICAL APPENDIX | DIFFERENT POPULATION / EVALUATION | NOT UPDATED RESULTS', 10, True, ACCENT)
    s.notes_slide.notes_text_frame.text += '\nHISTORICAL APPENDIX: preserved original numerical results. Not current nested-pipeline evaluation. Reconstructed attribution is not current-checkpoint output.'

violations = []
for i, s in enumerate(prs.slides, 1):
    # Remove inherited page numbers, then renumber the complete deck.
    for sh in list(s.shapes):
        if sh.has_text_frame and sh.left > Inches(12) and sh.top > Inches(7):
            sh._element.getparent().remove(sh._element)
    txt(s, 12.25, 7.03, .6, .3, f'{i:02}', 10, color=MUTED)
    for sh in s.shapes:
        if sh.left < 0 or sh.top < 0 or sh.left + sh.width > prs.slide_width + Inches(.03) or sh.top + sh.height > prs.slide_height + Inches(.03):
            violations.append({'slide': i, 'shape': sh.name})
assert not violations, violations
prs.save(TARGET)
PUBLISHED.mkdir(parents=True, exist_ok=True)
shutil.copy2(TARGET, PUBLISHED / TARGET.name)
manifest = {
    'updated': '2026-11-09', 'slides': len(prs.slides),
    'current_slides': len(prs.slides) - len(historical),
    'historical_appendix_slides': len(historical),
    'source_files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources)},
    'shape_bounds': 'passed', 'source_data': 'aggregate only',
    'backup': str(BACKUP.relative_to(ROOT)),
}
(OUT / 'comprehensive_pipeline_update_validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({k: v for k, v in manifest.items() if k != 'source_files'}, indent=2))
