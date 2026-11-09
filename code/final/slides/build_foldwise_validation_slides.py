#!/usr/bin/env python3
"""Native editable slides for a completed fold-contained DWI Ridge validation."""
import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[3]
NAVY='20364D';BLUE='EAF0F6';GREEN='E7F0E9';PALE='F7EDDF'


def build(run,output):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest.get('status')!='complete':
        raise ValueError('Only completed runs can be presented as validation results')
    modality=manifest.get('modality','DWI')
    metrics=pd.read_csv(run/'oof_metrics.csv')
    exclusions=pd.read_csv(run/'eligibility_counts.csv').fillna('')
    eligible=pd.read_csv(run/'eligible_metadata_private.csv')
    probes=pd.read_csv(run/'retention_probes.csv')
    r=Presentation();r.slide_width=Inches(13.333);r.slide_height=Inches(7.5)
    def box(s,x,y,w,h,text,fill=BLUE,size=20):
        shape=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(fill)
        shape.line.color.rgb=RGBColor.from_string('CCD5DE')
        shape.text_frame.word_wrap=True
        for i,line in enumerate(text.split('\n')):
            p=shape.text_frame.paragraphs[0] if i==0 else shape.text_frame.add_paragraph()
            p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string(NAVY)
            p.line_spacing=1.0;p.space_after=Pt(0)
        return shape
    def slide(title):
        s=r.slides.add_slide(r.slide_layouts[6])
        box(s,.4,.25,12.5,.7,title.replace('DWI',modality),'FFFFFF',27)
        box(s,.4,7.05,12.5,.25,f'6 November 2026 | {modality} Ridge implementation validation | Known supported sites only','FFFFFF',10)
        return s
    def table(s,headers,rows,x=.5,y=1.5,w=12.3,h=3.7):
        shape=s.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(w),Inches(h))
        t=shape.table
        for i,row in enumerate([headers]+rows):
            for j,value in enumerate(row):
                c=t.cell(i,j);c.text=str(value);c.fill.solid();c.fill.fore_color.rgb=RGBColor.from_string(BLUE if i==0 else 'FFFFFF')
                for p in c.text_frame.paragraphs:
                    p.font.name='Arial';p.font.size=Pt(16 if len(headers)<6 else 14)
                    p.font.color.rgb=RGBColor.from_string(NAVY)
        return shape

    s=slide('What is now implemented and tested')
    box(s,.6,1.35,12.1,1.2,'Five-fold outer participant CV + three-fold inner CV\nThree matched preprocessing arms; Ridge trained on CN source labels only',GREEN,19)
    box(s,.6,2.9,5.85,3.5,'Implemented\n• Raw connectome input\n• Fold-specific edge QC and scaling\n• Training-only ComBat and alpha selection\n• Inner-OOF age-bias calibration\n• Saved folds, models and provenance',BLUE,22)
    box(s,6.75,2.9,5.95,3.5,f'Scope\n• {modality} Ridge baseline\n• Known supported acquisition batches\n• Canonical clinical IDs and labels\n\nClinical source validity and acquisition protocols still require review.',PALE,22)

    s=slide('Implemented pipeline: training boundaries are explicit')
    box(s,.6,1.2,12.1,.65,('Raw fMRI correlations → fixed Fisher-z edges → participant outer split' if modality=='fMRI' else 'Raw 84 × 84 DWI → fixed per-scan symmetrization / upper triangle → participant outer split'),BLUE,19)
    box(s,.6,2.2,5.85,3.25,'OUTER TRAINING\nSource-labeled CN rows only\n↓ Inner grouped CV\nFit edge mask → ComBat → scaler → Ridge\nChoose alpha from inner OOF MAE\nFit BAG ~ age using inner OOF CN BAG\n↓ Refit selected pipeline on outer-training CN',GREEN,18)
    box(s,6.75,2.2,5.95,3.25,'OUTER HELD-OUT PARTICIPANTS\nUse frozen edge mask, offsets and scaler\n↓ Ridge prediction\nScore raw predicted age against actual age\n↓ BAG = prediction − age\n↓ cBAG = BAG − training-estimated age trend\nNo held-out age enters feature transformation.',BLUE,18)
    box(s,.6,5.8,12.1,.8,'All three arms use identical folds and training-derived edge masks. Non-CN rows are evaluated only when their participants are held out.',PALE,19)

    s=slide('How can age-informed ComBat avoid held-out ages?')
    box(s,.6,1.3,12.1,1.1,'Training model: x = intercept + β × training age + batch offset + residual\nMean-only ComBat adjusts batch location; its batch scale is fixed at 1.',BLUE,20)
    box(s,.6,2.8,12.1,1.2,'Additive held-out transformation: x_harmonized = x − σ_training × γ_batch\nThe subject-specific age terms cancel; no held-out age is needed.',GREEN,19)
    box(s,.6,4.4,12.1,1.6,'Three prespecified comparisons\n1. No ComBat   2. Batch-only location/scale ComBat   3. Training-age mean-only ComBat\nSex, diagnosis, APOE and metabolic variables are never brain-age predictor features.',BLUE,19)
    box(s,.6,6.2,12.1,.45,'This is an additive alternative, not full age-conditioned location/scale correction. Biology preservation must still be assessed.','FFFFFF',15)

    s=slide('Validation checks: what was actually tested')
    box(s,.6,1.3,12.1,4.95,'1. Training/apply equivalence\nFrozen transforms reproduce the ComBat training output, including age-trained mean-only mode.\n\n2. Held-out information isolation\nChanging outer-test ages leaves inner alpha choice and calibration unchanged.\n\n3. Independent-sample transformation\nA sample transforms identically alone or with altered companion test samples; fitted state stays unchanged. Unknown batches raise an error.\n\n4. End-to-end provenance\nParticipant groups remain disjoint; each eligible scan gets one OOF prediction per arm; calibration uses only outer-training rows.',GREEN,19)

    s=slide('Analysis population and exclusions')
    rows=[]
    for cohort,d in eligible.groupby('cohort'):
        rows.append([cohort,len(d),d.participant_group.nunique(),int(d.is_cn.sum())])
    table(s,['Cohort','Eligible scans','Participants','CN scans'],rows,h=3)
    excluded=exclusions[exclusions.exclusion_reason.ne('')]
    reasons='; '.join(f'{int(row.scans)} {row.cohort}: '+str(row.exclusion_reason).replace('_',' ') for row in excluded.itertuples())
    box(s,.6,4.9,12.1,1.5,f'Archived input universe: {int(exclusions.scans.sum())} scans. Retained: {len(eligible)} scans, including {int(eligible.is_cn.sum())} CN scans.\nExcluded: {reasons}.\nThese restrictions define the current estimand; they are not arbitrary first-row resolutions.',PALE,18)

    s=slide('Matched outer-OOF prediction results: CN reference samples')
    pooled=metrics[metrics.cohort.eq('Pooled')]
    names={'no_combat':'No ComBat','combat_batch_only':'Batch-only ComBat','combat_train_age_mean_only':'Training-age mean-only'}
    rows=[[names[row.arm],f'{row.MAE:.3f}',f'{row.RMSE:.3f}',f'{row.R2:.3f}',f'{row.Pearson_r:.3f}'] for row in pooled.itertuples()]
    table(s,['Preprocessing','MAE (years)','RMSE (years)','R²','Pearson r'],rows,h=2.8)
    box(s,.6,4.8,12.1,1.5,f'Same {int(eligible.is_cn.sum())} CN scans / {eligible.loc[eligible.is_cn,"participant_group"].nunique()} CN participants; identical outer and inner folds.\nRaw age predictions are scored before BAG correction. These are descriptive comparisons, not a significance test or a basis for post-hoc outer-test model selection.\nChanged eligibility and labels prevent direct comparison with the old benchmark.',PALE,19)

    s=slide('Scanner reduction and biological retention: limited probes')
    grouped=probes.groupby('arm')[['scanner_balanced_accuracy','sex_balanced_accuracy']].mean()
    rows=[[names[arm],f'{row.scanner_balanced_accuracy:.3f}',f'{row.sex_balanced_accuracy:.3f}'] for arm,row in grouped.iterrows()]
    table(s,['Preprocessing','Mean scanner balanced accuracy','Mean sex balanced accuracy'],rows,h=2.8)
    box(s,.6,4.8,12.1,1.5,'Fixed-regularization probes are fitted on outer-training CN and evaluated on held-out participants.\nSite predictability is confounded by cohort/age differences; lower accuracy does not prove that only technical effects were removed.\nAge prediction and residual cBAG–age dependence are reported separately.',PALE,19)

    s=slide('Remaining work before a project-wide scientific claim')
    box(s,.6,1.3,12.1,4.95,'Resolve clinical and acquisition provenance\nVerify HABS source diagnoses, ADRC visit matching, canonical identities, acquisition dates and ADRC protocol consistency.\n\nExtend the validated boundaries\nApply the same folds and training-only preprocessing to GINE, fMRI and fusion; nest graph sparsity, architecture choice and early stopping.\n\nClinical validation\nComplete T2D/HbA1c availability checks; generate modality-specific held-out cBAG; prespecify clinical tests.\n\nInterpret results cautiously\nKnown-site restricted population, one split seed, provisional source labels. No claim of unseen-site generalization or full biological preservation.',BLUE,19)
    output.mkdir(parents=True,exist_ok=True)
    target=output/f'{modality}_foldwise_validation_editable.pptx';r.save(target)
    check=Presentation(target)
    for slide_ in check.slides:
        for shape in slide_.shapes:
            assert shape.left>=0 and shape.top>=0 and shape.left+shape.width<=check.slide_width and shape.top+shape.height<=check.slide_height
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'results/final/DWI/foldwise_Ridge_validated_20261106')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/pipeline_audit/foldwise_validation_20261106')
    args=parser.parse_args();build(args.run,args.output)
