#!/usr/bin/env python3
"""Inventory local cohort clinical tables without exporting participant records.

Counts refer to source rows, not unique imaging participants. Names suggest
candidate measures; units, sentinel codes and visit matching remain unverified.
Default outputs stay under ignored results/. No clinical association is fitted.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'ADNI': 'data/final/ADNI/ADNI_harmonized_metadata(in).csv',
    'HABS': 'data/final/HABS/HABS_harmonized_metadata(in).csv',
    'ADRC': 'data/final/ADRC/ADRC_harmonized_metadata_merged.csv',
    'ADDecode': 'data/final/ADDecode/AD_DECODE_harmonized_metadata.csv',
}
DOMAINS = [
    ('existing_model_output', r'bag|predicted.?age|^pc\d+$|pca'),
    ('demographics', r'age|sex|gender|educ|race|ethnic'),
    ('linkage_or_provenance', r'(^|_)(id|ptid|rid|subject|visit|viscode|visdate|date|key|source|file|exam|match|runno|dataset|site|scanner)($|_)|connectome|^idrna$'),
    ('cognition', r'moca|mmse|adas|cdr|cognition|cognitive|memory|executive|processing.speed|language|visuospatial|global.cog|craft|benson|trail|ravlt|avlt|digit|dspan|fluency|mint|neurobat|^animals$|^veg$|jolo|udsbent|story|delayed|ufov'),
    ('genetics', r'apoe|genotype|^risk$|risk_for_ad|family|fam_num'),
    ('diagnosis', r'diagnos|dx_|dxsum|research.group|normcog|normcong|dement|mci_|ad_01|cog_status'),
    ('biomarkers', r'abeta|^ab40$|^ab42$|tau|nfl|gfap|amy[l]?|atn|fdgad|hippatr|csf'),
    ('cardiometabolic', r'bmi|height|weight|systolic|diastolic|bpsys|bpdia|vsbp|pulse|hypertens|diabet|cardio|^map$'),
    ('psychiatric_or_functional', r'npi|gds|faq|depress|^dep|anxiet|bipold|schizo'),
]
CANDIDATES = {
    'Chronological age': ['AGE','VISIT_AGE','age'],
    'Sex': ['PTGENDER','SUBJECT_SEX','Sex','sex'],
    'Diabetes candidate (type unverified)': ['DIABETES','vrf_bmi_PHC_Diabetes','clinical__CDX_Diabetes','CDX_Diabetes_x','CDX_Diabetes_y','IMH_Diabetes'],
    'Diabetes type candidate': ['DIABTYPE'],
    'HbA1c (units unverified)': ['clinical__BW_HBA1c','BW_HBA1c_x','BW_HBA1c_y','HbA1c','HBA1C'],
    'MoCA': ['MOCA_total_corrected','MOCA_TOTAL','MOCATOTS','MOCA','moca_MOCA'],
    'MMSE': ['MMSE_total','MMSE','mmse_MMSCORE','clinical__MMSE_Total','MMSE_Total_x','MMSE_Total_y'],
    'CDR global': ['CDGLOBAL','cdr_CDGLOBAL','CDR','CDR_GLOBAL','clinical__CDR_Global','CDR_Global_x','CDR_Global_y'],
    'CDR sum of boxes': ['CDRSB','cdr_CDRSB','CDR_Sum'],
    'Diagnosis': ['DX_Label_harmonized','Research Group','DIAGNOSIS','NORMCOG'],
    'APOE e4 dosage': ['APOE4_dosage','APOE4_count','APOE_e4_count'],
    'APOE e4 carrier': ['APOE4_carriage','APOE4_carrier'],
    'Global cognition composite': ['Global_Cognition_Composite','cognition_composite'],
    'Education': ['PTEDUCAT','EDUC','EDUCATION','education','Years_of_Education','ID_Education'],
    'BMI': ['BMI_corrected','BMI','BMI_calculated'],
}


ANALYSIS_PLAN = """Preliminary cBAG analysis plan — 4 November 2026

1. Resolve reference labels and linkage before inference.
The HABS all-CN override conflicts with source diagnosis labels. Review diagnosis source and visit alignment; unknown is not CN and APOE is not diagnosis. ADRC has repeated connectome keys across clinical rows: do not select the first row. Use acquisition and assessment dates to define a visit-matching rule, then audit ambiguous matches. Exact-key matching alone does not establish contemporaneous clinical assessment.

2. Obtain held-out predictions and independent bias calibration.
Partition all evaluated participants before any learned preprocessing. In each outer fold, learn harmonization, scaling and model choices using outer training only. Train the normative brain-age model using confirmed CN reference participants. Use inner OOF CN predictions from outer training to estimate BAG = a*age+b. Apply the frozen trend to outer-held-out BAG. Disease participants evaluated in that fold must not influence preprocessing/model fitting or calibration. Keep raw BAG and cBAG. Check calibration by age and cohort without refitting to the test set. Existing saved cBAG and global-ComBat performance are exploratory, not replacements for this design.

3. Primary clinical question: cognition within each cohort.
Candidate primary endpoint: MoCA for ADNI, ADRC and ADDecode, subject to scale/version review. For HABS, MMSE is an available candidate; it is not interchangeable with MoCA. Fit a prespecified association model within each cohort, for example score ~ cBAG + age + sex + acquisition site where estimable. Add education in a defined sensitivity subset when available; report changes in sample composition. Report coefficient per year of cBAG, standardized effect, confidence interval, sample size and missingness. This is an association, not causal evidence or a diagnostic-performance claim.

4. Secondary questions.
Evaluate diagnosis-group differences and APOE e4 dosage/carriage only after verifying label coding and visit provenance. Keep diagnosis out of the primary harmonization design for diagnosis-association analyses. Treat biomarkers and vascular measures as exploratory until assay units, specimen type, coding, and overlap are established. Existing cognition composites are cohort-specific derived measures; shared column names do not establish common scales, and derivation/residualization must be reviewed before use.

5. Repeated measures, missingness and multiplicity.
Start with one prespecified eligible imaging/clinical visit per participant; if longitudinal analyses are justified later, use a participant-level model. Define the visit window after inspecting date availability and before association results. Report complete-case denominators and exclusions separately for each endpoint; do not impute clinical outcomes or replace missing scores by zero. Define primary and secondary testing families before fitting; apply an explicit multiplicity rule to secondary tests. No p-values are generated by this inventory.

6. Cross-cohort synthesis and deliverables.
Analyze cohorts separately first. Combine only measures with verified definitions and compatible populations, using a prespecified synthesis and heterogeneity assessment. Do not pool MoCA with MMSE or assume that all cohort composites are comparable. Deliver an imaging-linked variable matrix, a label/visit-resolution log, a frozen analysis specification, and association tables using held-out cBAG. Current inventory counts describe source rows or explicitly labeled exact-key matches, not final analysis-ready participants.
"""


def domain(column):
    for label, pattern in DOMAINS:
        if re.search(pattern, column, re.I):
            return label
    return 'other_requires_review'


def inventory(root, output):
    output.mkdir(parents=True, exist_ok=True)
    variables, tables, candidates, provenance = [], [], [], []
    imaging_variables, coverage, diagnosis_counts, linked_diagnoses = [], [], [], []
    imaging = pd.read_csv(root/'results/final/DWI/harmonized_connectomes/harmonized_DWI_manifest.csv')
    for cohort, relative in SOURCES.items():
        path = root/relative
        frame = pd.read_csv(path, low_memory=False)
        provenance.append(dict(cohort=cohort, source=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        id_candidates = ['PTID','Subject','subject_id','ID','IDRNA']
        id_column = next((c for c in id_candidates if c in frame), None)
        tables.append(dict(cohort=cohort, source=relative, source_rows=len(frame), columns=len(frame.columns),
                           provisional_id_column=id_column,
                           distinct_nonmissing_source_ids=frame[id_column].dropna().astype(str).str.strip().nunique() if id_column else None,
                           rows_with_missing_source_id=int(frame[id_column].isna().sum()) if id_column else None,
                           population='Source-table inventory; not an imaging-linked analysis cohort'))
        for column in frame:
            text = frame[column].astype('string').str.strip()
            missing = frame[column].isna() | text.str.lower().isin(['','na','n/a','nan','none','null','<na>'])
            numeric = pd.to_numeric(frame[column].where(~missing), errors='coerce')
            # Report possible codes, but never remove them without a data dictionary.
            sentinel = numeric.isin([-9,-99,-999,999,9999])
            variables.append(dict(cohort=cohort, source=relative, variable=column, domain=domain(column),
                                  source_rows=len(frame), nonmissing_rows=int((~missing).sum()),
                                  missing_percent=round(float(missing.mean()*100),2),
                                  numeric_rows=int(numeric.notna().sum()), potential_sentinel_rows=int(sentinel.sum()),
                                  units='unverified', scale_definition='unverified',
                                  linkage_status='not joined to imaging; visit timing unverified'))
        key_column = 'CONNECTOME_KEY_USED_FOR_INTERSECTION'
        if key_column in frame:
            keys = frame[key_column].astype('string').str.strip().str.upper()
            scan_keys = imaging.loc[imaging.DATASET.str.upper().eq(cohort.upper()), 'connectome_key'].astype('string').str.strip().str.upper()
            unique = keys.notna() & keys.ne('') & ~keys.duplicated(keep=False)
            matched = unique & keys.isin(scan_keys)
            subset = frame.loc[matched]
            coverage.append(dict(cohort=cohort, source_rows=len(frame), imaging_scans=len(scan_keys),
                                 uniquely_matched_source_rows=len(subset),
                                 imaging_keys_with_unique_source_match=int(scan_keys.isin(keys[matched]).sum()),
                                 source_rows_with_duplicated_key=int((keys.notna() & keys.ne('') & keys.duplicated(keep=False)).sum()),
                                 status='Exact normalized key only; clinical visit timing and diagnosis provenance unverified'))
            if 'DX_Label_harmonized' in subset:
                for label, count in subset.DX_Label_harmonized.fillna('MISSING').value_counts().items():
                    linked_diagnoses.append(dict(cohort=cohort, source_diagnosis_label=label, uniquely_matched_rows=int(count),
                                                 status='Exact-key subset only; dates and diagnosis provenance unverified'))
            for column in frame:
                text = subset[column].astype('string').str.strip().str.lower()
                missing = subset[column].isna() | text.isin(['','na','n/a','nan','none','null','<na>'])
                imaging_variables.append(dict(cohort=cohort, variable=column, matched_source_rows=len(subset),
                                              nonmissing_rows=int((~missing).sum()),
                                              missing_percent=round(float(missing.mean()*100),2) if len(subset) else None,
                                              status='Unique exact-key match; timing unverified'))
        if 'DX_Label_harmonized' in frame:
            for value, count in frame.DX_Label_harmonized.fillna('MISSING').value_counts().items():
                diagnosis_counts.append(dict(cohort=cohort, source_diagnosis_label=value, source_rows=int(count),
                                             status='Source labels; clinical provenance and visit alignment unverified'))
        lookup = {c.lower():c for c in frame.columns}
        for construct, names in CANDIDATES.items():
            found = [lookup[n.lower()] for n in names if n.lower() in lookup]
            if not found:
                candidates.append(dict(cohort=cohort, construct=construct, candidate_variable='', nonmissing_rows=0,
                                       status='No predefined candidate column found; absence not established'))
            for column in dict.fromkeys(found):
                record = next(v for v in variables if v['cohort']==cohort and v['variable']==column)
                candidates.append(dict(cohort=cohort, construct=construct, candidate_variable=column,
                                       nonmissing_rows=record['nonmissing_rows'],
                                       status='Candidate only; definition, codes and visit linkage need review'))
    v, t, c = pd.DataFrame(variables), pd.DataFrame(tables), pd.DataFrame(candidates)
    pd.DataFrame(imaging_variables).to_csv(output/'imaging_linked_variable_inventory.csv', index=False)
    pd.DataFrame(coverage).to_csv(output/'imaging_linkage_coverage.csv', index=False)
    pd.DataFrame(linked_diagnoses).to_csv(output/'imaging_linked_diagnosis_counts.csv', index=False)
    pd.DataFrame(diagnosis_counts).to_csv(output/'source_diagnosis_counts.csv', index=False)
    v.to_csv(output/'variable_inventory.csv', index=False)
    t.to_csv(output/'source_table_summary.csv', index=False)
    c.to_csv(output/'candidate_cross_cohort_measures.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(sources=provenance,
        limitations=['Source rows are not imaging-linked participants.', 'Domain labels use column-name heuristics.',
                     'Missingness does not interpret undocumented sentinel codes.',
                     'No scale equivalence, units, diagnosis validity or composite provenance has been established.',
                     'Alternative raw sources and Excel workbooks are not yet inventoried.']), indent=2))
    sections = ['<h1>Clinical variable inventory</h1>',
                '<p>Preliminary schema and missingness inventory of four existing harmonized metadata tables. '
                'Counts refer to source rows, not imaging-linked participants. No participant values are exported.</p>',
                '<h2>Source tables</h2>', t.to_html(index=False, escape=True),
                '<h2>Candidate imaging linkage</h2>', pd.DataFrame(coverage).to_html(index=False, escape=True),
                '<p>Duplicate source keys are excluded, not arbitrarily resolved. Exact ID matching does not verify visit timing.</p>',
                '<h2>Source diagnosis labels</h2>', pd.DataFrame(diagnosis_counts).to_html(index=False, escape=True),
                '<p>HABS source labels include MCI and AD, conflicting with the existing all-CN training override. Resolve provenance before normative training.</p>',
                '<h2>Candidate cross-cohort measures</h2>', c.to_html(index=False, escape=True)]
    for cohort in SOURCES:
        sections += ['<h2>'+html.escape(cohort)+'</h2>', v[v.cohort==cohort].drop(columns=['source','cohort']).to_html(index=False, escape=True)]
    (output/'preliminary_cBAG_plan.txt').write_text(ANALYSIS_PLAN)
    sections += ['<h2>Preliminary cBAG analysis plan</h2>', '<pre style="white-space:pre-wrap;font:inherit">'+html.escape(ANALYSIS_PLAN)+'</pre>']
    css='<style>body{font:14px Arial,sans-serif;margin:32px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f3f3f3}h2{margin-top:32px}</style>'
    (output/'clinical_inventory.html').write_text('<!doctype html><meta charset="utf-8">'+css+''.join(sections))
    print(t[['cohort','source_rows','columns','distinct_nonmissing_source_ids']].to_string(index=False))
    print('Variable inventory:', len(v), 'columns; output:', output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'results/final/clinical_inventory_20261104')
    args=parser.parse_args()
    inventory(args.root,args.output)
