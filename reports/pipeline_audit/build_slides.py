"""Build native, editable PowerPoint audit slides; python-pptx required."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/pipeline_audit'
r = Presentation(); r.slide_width=Inches(13.333); r.slide_height=Inches(7.5)
NAVY='16324F'; BLUE='E7EFF7'; RED='F9E3E1'; GREEN='E1F0E8'
def box(s,x,y,w,h,text,fill=BLUE,size=21):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(fill); sh.line.color.rgb=RGBColor.from_string('BCCBDA')
    tf=sh.text_frame; tf.word_wrap=True
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.color.rgb=RGBColor.from_string(NAVY)
    return sh

def slide(title,subtitle='Source audit | DWI | 03 November 2026 | No training rerun'):
    s=r.slides.add_slide(r.slide_layouts[6])
    box(s,.35,.2,12.6,.7,title,NAVY,28)
    for p in s.shapes[-1].text_frame.paragraphs: p.font.color.rgb=RGBColor(255,255,255)
    box(s,.35,7.02,12.6,.27,subtitle,'FFFFFF',10)
    return s

def arrow(s,x,y): box(s,x,y,.45,.5,'→','FFFFFF',24)

s=slide('DWI harmonization & cross-validation: audit before new experiments')
box(s,.6,1.3,12.1,1.25,'Finding: the current end-to-end evaluation is not leakage-free.\nGlobal ComBat precedes subject-level cross-validation.',RED,25)
box(s,.6,2.9,5.8,3.5,'Existing safeguards\n• Participant-grouped outer CV\n• Participant-grouped early-stopping split\n• Training-only target age scaling\n\nThese protect model fitting, but not upstream ComBat.',GREEN)
box(s,6.7,2.9,6,3.5,'Required changes\n• Fit learned preprocessing inside each split\n• Select topology / architecture inside outer training\n• Define allowed harmonization covariates\n• Cross-fit clinical reference predictions',RED)

s=slide('Current implementation: global preprocessing comes first')
items=[('Raw DWI\n4 cohorts',BLUE),('Global edge QC\n+ log1p',RED),('Joint ComBat\nAGE / SEX / optional DX',RED),('Saved harmonized\n84 × 84 matrices',BLUE)]
for i,(t,c) in enumerate(items):
    box(s,.5+i*3.22,1.35,2.7,1.3,t,c,19)
    if i<3: arrow(s,3.22+i*3.22,1.72)
box(s,.6,3,12.1,1.1,'Then: CN GroupKFold → grouped 80/20 early stopping → outer OOF predictions',BLUE,23)
box(s,.6,4.4,12.1,1.25,'Repeat for five graph thresholds → select by pooled OOF MAE → report selected OOF score',RED,23)
box(s,.6,5.95,12.1,.65,'Evidence: run_DWI_harmonization.py:2226–2253, 2542–2675; GNN_DWI_BAG.py:710–753, 940–1016','FFFFFF',14)

s=slide('Audit findings: confirmed code behavior and its implications')
rows=[('Global ComBat','All retained samples enter neuroCombat before CV. Held-out distributions influence the representation.','2638; GNN:46'),('Global feature mask','Finite / nonconstant edge mask uses all rows. Learn data-dependent masks on training rows.','2226–2253'),('Threshold selection','Five candidates ranked on pooled outer OOF MAE. Selected score is a model-selection estimate.','GNN:940–1016'),('Covariates','True AGE always preserved; GROUP conditional on completeness/rank. Define age-conditioned use and diagnosis circularity.','2416–2675'),('Participant grouping','Only terminal _Y<number> removed. Grouping exists; actual visit IDs still require a data audit.','GNN:145–155')]
for i,(a,b,c) in enumerate(rows):
    y=1.2+i*1.05; box(s,.5,y,2.4,.9,a,BLUE,18); box(s,3,y,9.8,.9,b+' [source '+c+']',RED if i<4 else BLUE,17)

s=slide('Proposed pipeline: split first; fit within the split')
box(s,.6,1.2,12.1,.7,'Freeze eligible sample IDs + canonical participant groups + outer folds',BLUE,23)
box(s,.6,2.15,5.65,3.35,'OUTER TRAINING ONLY\n1. Inner grouped train / validation splits\n2. Fit edge QC + harmonization on inner train\n3. Transform inner validation with frozen state\n4. Select topology, capacity, epoch budget\n5. Refit on all outer training with frozen choices',GREEN,21)
box(s,6.8,2.15,5.9,3.35,'OUTER TEST: LOCKED\n• Never used for preprocessing fit or selection\n• Apply outer-training transform\n• Predict once per chosen model\n• Save raw age predictions and fold IDs\n• Aggregate OOF metrics after all folds',BLUE,21)
box(s,.6,5.8,12.1,.9,'Gate: verify a train/apply harmonizer that preserves required covariate effects; never concatenate test rows to refit.',RED,20)

s=slide('Define the scientific target before choosing harmonization')
box(s,.6,1.3,5.85,4.9,'Known-site, new-participant prediction\n• Group by participant in every split\n• Ensure training support for each test batch\n• Learn correction only from training rows\n• Predefine unsupported-batch handling\n\nAge-conditioned analysis\nTrue AGE may be available, but results must explicitly describe that conditioning.',BLUE,22)
box(s,6.75,1.3,5.95,4.9,'Unseen-site generalization\n• Hold out a site / cohort explicitly\n• Standard known-batch parameters do not resolve an unseen batch\n• Any target-site adaptation needs its own declared evaluation design\n\nClinical diagnosis analyses\nAvoid preserving diagnosis in the primary transform when later testing diagnosis associations.',RED,22)

s=slide('cBAG: separate reference calibration from evaluated subjects')
box(s,.6,1.3,12.1,1.3,'Current: fit BAG ~ AGE on CN OOF predictions, then subtract that trend from final-model BAG.\nCN final-model predictions are in-sample; existing cBAG is not fully cross-fitted.',RED,23)
box(s,.6,3,12.1,2.2,'Proposed: within each outer training set, obtain inner OOF CN predictions and estimate the age-bias trend.\nFreeze the trend and apply it to the outer-held-out BAG. Keep all visits of a subject together.\nFor clinical comparisons, use held-out predictions for all evaluated subjects or a prespecified independent clinical set.',GREEN,22)
box(s,.6,5.6,12.1,.8,'cBAG = predicted age − chronological age − (training-reference slope × age + intercept)','FFFFFF',22)
box(s,.6,6.4,12.1,.35,'Evidence: GNN_DWI_BAG.py:1140–1188. Clinical design remains provisional pending variable inventory.','FFFFFF',13)

s=slide('Next steps: controlled capacity study and clinical inventory')
box(s,.6,1.3,5.85,4.95,'Learning curves / capacity\nCurrent: 3 GINE layers, width 64, dropout 0.30\nProposed: 2 × 32; 3 × 32; current 3 × 64\nLog epoch-level train + validation MAE and loss, best epoch, parameters and runtime.\nUse eval-mode training MAE for comparable curves.\nSelect configurations inside outer training.\nCurrent fit_one_fold does not persist full curves.',BLUE,21)
box(s,6.75,1.3,5.95,4.95,'Clinical inventory / cBAG\nInventory ADNI, HABS, ADRC, ADDecode separately: definitions, units, missingness, visit linkage and overlap.\nVerify HABS CN override from diagnosis evidence.\nPredefine primary associations and covariates after inventory.\n\nThis deck is a source audit and proposal. No corrected performance claims or new experiments yet.',BLUE,21)

OUT.mkdir(parents=True,exist_ok=True)
path=OUT/'DWI_pipeline_audit_editable.pptx'; r.save(path)
# Verify native editable content and slide bounds.
check=Presentation(path)
for s in check.slides:
    for sh in s.shapes:
        assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=check.slide_width and sh.top+sh.height<=check.slide_height
assert len(check.slides)==7
print(path)
print('Validated 7 slides; all content is native editable text/shapes; all shapes within slide bounds.')
