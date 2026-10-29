import sys
sys.path.insert(0,'/tmp/avery_slides_deps')
from pathlib import Path
import json, csv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'results/presentation'
D=ROOT/'results/final/DWI'; F=ROOT/'results/final/fMRI'; U=ROOT/'results/final/Fusion'; I=ROOT/'results/explore/GNN_attribution'
def rows(p):
 with open(p) as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text())
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
navy='142B45'; teal='007F86'; blue='477AB7'; orange='D9883D'; gray='526477'; light='F3F6F9'
def text(s,x,y,w,h,t,size=20,color=navy,bold=False):
 box=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h));tf=box.text_frame;tf.word_wrap=True
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(12)
 return box
def slide(title,kicker='RESULTS',source='',note=''):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('FFFFFF')
 text(s,.48,.18,12,.3,kicker,11,teal,True);text(s,.48,.65,12.3,.8,title,29,navy,True)
 text(s,.5,7.06,11.8,.29,source,9,gray);text(s,12.25,7.03,.6,.3,f'{len(prs.slides):02}',10,gray)
 s.notes_slide.notes_text_frame.text=note+'\nSource files: '+source
 return s
def bullets(s,items,x=.6,y=1.7,w=12,size=22):text(s,x,y,w,4.9,'\n'.join('• '+v for v in items),size)
def call(s,t,y=6.35):text(s,.6,y,12.1,.55,t,17,teal,True)
def table(s,headers,data,x=.6,y=1.75,w=12.1,h=3.7,size=16):
 sh=s.shapes.add_table(len(data)+1,len(headers), Inches(x), Inches(y), Inches(w), Inches(h));tb=sh.table
 for i,row in enumerate([headers]+data):
  for j,v in enumerate(row):
   c=tb.cell(i,j);c.text=str(v);c.fill.solid();c.fill.fore_color.rgb=RGBColor.from_string(navy if i==0 else ('F0F5F8' if i%2 else 'FFFFFF'))
   for p in c.text_frame.paragraphs:p.font.name='Aptos';p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else navy)
 return tb
def chart(s,cats,series,x=.7,y=1.75,w=11.8,h=4.3):
 data=CategoryChartData();data.categories=cats
 for name,vals in series:data.add_series(name,vals)
 c=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h),data).chart
 c.has_legend=len(series)>1
 if c.has_legend:c.legend.position=XL_LEGEND_POSITION.BOTTOM
 c.category_axis.tick_labels.font.size=Pt(14);c.value_axis.tick_labels.font.size=Pt(12)
 for idx,se in enumerate(c.series):se.format.fill.solid();se.format.fill.fore_color.rgb=RGBColor.from_string([teal,blue,orange][idx%3])
 c.plots[0].has_data_labels=True;c.plots[0].data_labels.font.size=Pt(12);c.plots[0].data_labels.number_format='0.00'
 return c
def image(s,p,x,y,w,h):
 im=Image.open(p);iw,ih=im.size;scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
 s.shapes.add_picture(str(p), Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
source='Saved project outputs | Prepared 28 October 2026'
s=slide('Connectome Brain Age: From Harmonization to Interpretation','RESEARCH RESULTS',source)
text(s,.65,2,11.8,1.6,'DWI & fMRI\nRidge, GNN and multimodal fusion',36,navy,True)
text(s,.65,4.3,11.8,1.3,'ADNI · HABS · ADRC · ADDECODE (DWI only)\nAn evidence-based summary of existing analyses',23,gray)
call(s,'Editable text, tables and native PowerPoint charts; source details in speaker notes.')
s=slide('Three findings frame the current results',source='report_tables/Table4_models_pooled.csv; GNN_interpretability/analysis_summary.json')
bullets(s,['Harmonization substantially reduces dataset and site predictability; biological signal preservation is mixed.','Ridge outperforms GNN in the reported unimodal analyses and the shared paired-sample comparison.','Weighted Ridge fusion has the lowest paired-sample MAE: 6.47 years; its gain over DWI Ridge is modest.','GNN attribution is exploratory: reconstructed predictions differ from the original OOF predictions.'])
s=slide('One pipeline, two evaluation populations','STUDY DESIGN',source='README.md; report_tables/Table1_samples.csv; Table2_features.csv')
for x,title,body in [(.6,'01  Harmonize','84 × 84 connectomes\n3,486 unique edges\nJoint dataset/site ComBat'),(4.8,'02  Predict','CN normative age models\nRidge and GNN\nSingle modality + fusion'),(9,'03  Interpret','BAG and age bias\nOOF fusion GNN IG\nROI / edge / modality')]:
 text(s,x,2,3.8,.7,title,25,teal,True);text(s,x,2.95,3.7,2.1,body,21)
call(s,'Full modality cohorts answer a different question from the shared 387-scan fusion benchmark.')
s=slide('Cohort sizes and the paired evaluation set','STUDY DESIGN',source='report_tables/Table1_samples.csv; DWI metadata; modality CV summaries')
table(s,['Analysis','Scans','Participants / scope'],[['DWI harmonization','1,234','ADNI 314; HABS 487; ADRC 347; ADDECODE 86'],['fMRI harmonization','847','ADNI, HABS, ADRC'],['DWI CN OOF models','916','667 participants; HABS all-CN override'],['fMRI CN OOF models','642','525 participants'],['Paired DWI + fMRI','847','679 participants'],['Shared fusion CN OOF','387','330 participants']],h=4.3,size=16)
call(s,'CN = cognitively normal; OOF = out-of-fold. Cohort definitions are policy dependent.')
s=slide('Harmonization targets technical variability','HARMONIZATION',source='README.md; run_DWI_harmonization.py; run_fMRI_harmonization.py',note='DWI diagnosis preservation is conditional in the code. Inspect saved combat_covariates.csv for the exact run-specific covariate set.')
bullets(s,['QC and metadata matching precede feature extraction; each matrix yields 3,486 upper-triangle edges.','DWI: log1p transformation before ComBat; fMRI: Fisher-z transformation before ComBat.','Technical batch: DATASET_SITE, combining cohort and acquisition site.','Biological covariates: age and sex; fMRI also preserves diagnosis. DWI diagnosis inclusion is conditional.','Adjusted edges are reconstructed into symmetric 84 × 84 connectomes.'],size=21)
s=slide('Dataset and site predictability decrease sharply','HARMONIZATION',source='report_tables/Table3_harmonization.csv',note='Balanced accuracy from saved harmonization diagnostics. Different targets use different eligible subsets. Chance references: DWI dataset 1/4; fMRI dataset 1/3; site 1/29 and 1/24. These are descriptive diagnostics after full-data harmonization.')
chart(s,['DWI: dataset (4 classes)','DWI: site batch (29)','fMRI: dataset (3)','fMRI: site batch (24)'],[('Before',[.991557,.720693,.913175,.254869]),('After',[.334015,.013624,.428722,.011181])])
call(s,'Balanced accuracy; lower technical predictability supports reduced batch information.')
s=slide('PCA provides a visual check of cohort mixing','HARMONIZATION',source='DWI and fMRI harmonization_evaluation/*PCA*by_dataset.png')
for p,x,y in [(D/'harmonization_evaluation/DWI_PCA_before_ComBat_by_dataset.png',.5,1.65),(D/'harmonization_evaluation/DWI_PCA_after_ComBat_by_dataset.png',6.7,1.65),(F/'harmonization_evaluation/fMRI_PCA_before_ComBat_by_dataset.png',.5,4.05),(F/'harmonization_evaluation/fMRI_PCA_after_ComBat_by_dataset.png',6.7,4.05)]:image(s,p,x,y,6.1,2.35)
text(s,.7,6.58,11.9,.3,'Top: DWI     Bottom: fMRI     Left: before     Right: after',15,gray)
s=slide('Biological signal preservation is mixed','HARMONIZATION',source='report_tables/Table3_harmonization.csv')
table(s,['Diagnostic','DWI before → after','fMRI before → after'],[['Sex balanced accuracy','0.782 → 0.766','0.631 → 0.640'],['Diagnosis balanced accuracy','0.294 → 0.151','0.300 → 0.227'],['Age MAE (years)','6.22 → 6.71','7.64 → 8.00'],['Age R²','0.588 → 0.516','0.379 → 0.307']],h=3.4,size=20)
text(s,.7,5.4,11.8,.85,'Sex prediction is broadly retained. Age prediction worsens and diagnosis prediction decreases; these diagnostics do not establish complete biological preservation.',21)
s=slide('Evaluation uses participant grouping','MODELING',source='Ridge CV summaries; GNN threshold summaries; fusion_summary.json')
bullets(s,['CN scans train normative age models; all visits from each participant stay within one outer fold.','Ridge: five outer folds, four grouped inner folds for alpha selection; standardization is fitted within training splits.','Unimodal GNN: graph sparsity is compared using pooled CN OOF performance.','Paired benchmark: Ridge and GNN share 387 scans, 330 participants and the same outer folds.','Upstream ComBat was fitted on all data while preserving age; current CV estimates retain leakage risk.'],size=21)
s=slide('Ridge leads the full-cohort unimodal analyses','SINGLE MODALITY',source='DWI_Ridge_CV_metrics.json; fMRI_Ridge_CV_metrics.json; selected GNN threshold summaries')
table(s,['Modality / model','CN scans','MAE (yr)','RMSE (yr)','R²','Pearson r'],[['DWI Ridge','916','5.98','7.89','0.519','0.724'],['DWI GNN (top 50%)','916','7.56','9.61','0.287','0.537'],['fMRI Ridge','642','7.33','9.49','0.406','0.637'],['fMRI GNN (top 10%)','642','9.31','11.97','0.056','0.305']],h=3.5,size=18)
call(s,'DWI and fMRI use different populations; compare models within each modality.')
s=slide('Graph sparsity has modality-specific effects','SINGLE MODALITY',source='DWI_GNN_threshold_comparison.csv; fMRI_GNN_threshold_comparison.csv')
d=rows(D/'GNN_BAG/DWI_GNN_threshold_comparison.csv');f=rows(F/'GNN_BAG/fMRI_GNN_threshold_comparison.csv')
order=['10','25','50','75','all']
chart(s,['Top 10%','Top 25%','Top 50%','Top 75%','All positive'],[('DWI MAE',[float(next(r for r in d if r['retain_positive_percent']==v)['MAE_years']) for v in order]),('fMRI MAE',[float(next(r for r in f if r['retain_positive_percent']==v)['MAE_years']) for v in order])])
call(s,'Thresholds were selected on the reported OOF comparison; an independent estimate is still needed.')
s=slide('BAG captures prediction error; age bias needs care','BRAIN AGE GAP',source='Ridge CV summaries; report_tables/Table4_models_pooled.csv')
bullets(s,['BAG = predicted brain age − chronological age. Positive BAG means an older predicted age.','cBAG subtracts the linear age-related BAG trend estimated from CN OOF predictions.','Paired raw BAG–age correlations remain strongly negative: DWI Ridge −0.599; weighted Ridge −0.616; GNN fusion −0.825.','A correction fitted and assessed on the same OOF reference is descriptive; downstream inference needs appropriate independent or cross-fitted calibration.','Full-cohort CN predictions are in-sample after final refitting; use OOF outputs for performance comparisons.'],size=21)
s=slide('Paired-sample selection changes the population','MULTIMODAL FUSION',source='report_tables/Table1_samples.csv; metadata/fusion_metadata_audit_summary.json')
table(s,['Cohort','Paired scans','Evaluated CN scans','Evaluated participants'],[['ADNI','245','131','74'],['ADRC','338','243','243'],['HABS','264','13','13'],['Total','847','387','330']],h=3.1,size=20)
text(s,.7,5.15,12,1.05,'250 HABS scans have age disagreements between modalities. Training requires ages to agree within 0.01 year; only 13 HABS scans enter this benchmark.',21)
call(s,'Visit-ID pairing is available; acquisition-date proximity has not been verified.')
s=slide('Fusion combines modalities in three ways','MULTIMODAL FUSION',source='Ridge_baseline/fusion_summary.json; Weighted_Ridge/fusion_summary.json; GNN_fusion/GNN_fusion_summary.json')
table(s,['Approach','Implementation'],[['Ridge early fusion','Concatenate standardized DWI and fMRI edges: 6,972 features'],['Weighted Ridge early fusion','[Z_DWI, w × Z_fMRI]; tune w and alpha jointly within training folds'],['Ridge late mean','Average the two unimodal age predictions'],['GNN fusion','Separate two-layer GINE encoders; concatenate pooled embeddings; nonlinear age head']],h=3.7,size=19)
call(s,'Fusion GNN uses fixed top 50% positive edges for both modalities; hidden size = 32.')
pooled=rows(U/'report_tables/Table4_models_pooled.csv');names={'DWI':'DWI Ridge','fMRI':'fMRI Ridge','EARLY_FUSION':'Ridge early fusion','WEIGHTED_EARLY_FUSION':'Weighted Ridge fusion','LATE_MEAN':'Ridge late mean','DWI_GNN':'DWI GNN','fMRI_GNN':'fMRI GNN','GNN_FUSION':'GNN fusion'}
s=slide('Weighted Ridge has the lowest paired-sample MAE','MULTIMODAL FUSION',source='report_tables/Table4_models_pooled.csv',note='All values are raw CN OOF metrics from the same 387 scans and 330 participants. No significance claim is supported by these point estimates.')
chart(s,[names[r['model']] for r in pooled],[('MAE (years)',[float(r['MAE_years']) for r in pooled])],h=4.65)
call(s,'Shared benchmark: 387 CN scans / 330 participants / identical outer folds.',y=6.55)
s=slide('The fusion gain is modest and method dependent','MULTIMODAL FUSION',source='report_tables/Table4_models_pooled.csv; Weighted_Ridge/fusion_summary.json')
table(s,['Model','MAE (yr)','RMSE (yr)','R²','r'],[[names[r['model']],f"{float(r['MAE_years']):.2f}",f"{float(r['RMSE_years']):.2f}",f"{float(r['R2']):.3f}",f"{float(r['Pearson_r']):.3f}"] for r in pooled if r['model'] in ['DWI','EARLY_FUSION','WEIGHTED_EARLY_FUSION','LATE_MEAN','GNN_FUSION']],h=3.5,size=19)
text(s,.7,5.55,11.8,.8,'Weighted Ridge improves MAE by 0.184 years (2.8%) vs DWI Ridge. Unweighted concatenation does not improve on DWI alone.',21)
s=slide('Pooled performance masks cohort differences','MULTIMODAL FUSION',source='report_tables/Table5_models_by_cohort.csv')
co=rows(U/'report_tables/Table5_models_by_cohort.csv');mods=['DWI','WEIGHTED_EARLY_FUSION','GNN_FUSION']
table(s,['Cohort','CN scans','DWI Ridge MAE','Weighted Ridge MAE','GNN fusion MAE'],[[c,n]+[f"{float(next(r for r in co if r['dataset']==c and r['model']==m)['MAE_years']):.2f}" for m in mods] for c,n in [('ADNI','131'),('ADRC','243'),('HABS','13')]],h=2.8,size=19)
text(s,.7,5,11.8,1.1,'ADNI R² is negative for all paired models despite positive pooled R². HABS estimates rely on only 13 scans. These are pooled-training OOF results, not leave-one-cohort-out tests.',21)
s=slide('Interpretability explains the reconstructed fusion GNN','INTERPRETABILITY',source='GNN_interpretability/METHODS.txt; analysis_summary.json')
bullets(s,['Integrated gradients (IG) attributes standardized node features and normalized edge attributes on held-out CN scans.','Five reconstructed outer-fold models explain 387 scans from 330 participants.','Reference: training-mean node features with observed diagonal retained, and zero edge attributes; graph topology stays fixed.','Participant summaries average signed attributions across visits before taking absolute values.','Attributions describe model behavior relative to this reference; they are not causal effects or clinical associations.'],size=21)
s=slide('Attribution quality limits the strength of interpretation','INTERPRETABILITY',source='GNN_interpretability/analysis_summary.json; reconstruction_check.csv')
table(s,['Quality check','Observed result'],[['IG integration steps','64 initially; up to 512'],['Completeness tolerance','0.05 years'],['Scans failing completeness','6 / 387 (1.6%)'],['Maximum absolute IG residual','0.119 years'],['Maximum difference from original OOF prediction','13.18 years']],h=3.5,size=20)
call(s,'These attributions explain reconstructed models; exact original-model attribution is not established.')
s=slide('fMRI carries larger attribution in this fusion model','INTERPRETABILITY',source='GNN_interpretability/modality_summary.csv')
m=rows(I/'modality_summary.csv')
table(s,['Modality','Mean signed IG (yr)','Mean absolute connection IG','Occlusion error increase (yr)'],[[r['modality'],f"{float(r['signed_IG_years_mean']):.2f}",f"{float(r['absolute_connection_IG_years_mean']):.2f}",f"{float(r['absolute_error_increase_years_mean']):.2f}"] for r in m],h=2.3,size=19)
text(s,.7,4.55,11.7,1.5,'Larger fMRI attribution reflects this model and its reference. It does not imply superior standalone prediction. Absolute connection contributions can accumulate and cancel; they are not predicted ages or modality percentages.',21)
call(s,'Reference occlusion retains topology and may produce inputs outside the training distribution.')
s=slide('Top ROI attributions emphasize fMRI regions','INTERPRETABILITY',source='GNN_interpretability/roi_importance.csv; roi_labels_with_verification_scope.csv',note='ROI index is zero-based. Atlas naming is supported by the existing label/source audit, without per-scan regeneration. ROI importance assigns half of absolute combined connection attribution to each endpoint.')
r=rows(I/'roi_importance.csv')[:7]
chart(s,[a['roi_name'].replace('ctx_','').replace('_',' ') for a in r],[('Mean incident absolute IG (years)',[float(a['mean_incident_absolute_years']) for a in r])],h=4.45)
call(s,'Top overall rows are fMRI. Region labels follow the existing atlas audit; per-scan order is unverified.')
s=slide('Leading connections and stability offer complementary views','INTERPRETABILITY',source='GNN_interpretability/connection_importance.csv; connection_fold_stability.csv')
connections=[r for r in rows(I/'connection_importance.csv') if r['channel']=='combined'][:5]
stab={(r['modality'],r['roi_i'],r['roi_j']):r for r in rows(I/'connection_fold_stability.csv')}
table(s,['Connection (fMRI)','Mean |IG| (yr)','Folds in top 100'],[[r['roi_i_name'].replace('ctx_','')+' ↔ '+r['roi_j_name'].replace('ctx_',''),f"{float(r['mean_absolute_years']):.3f}",stab[(r['modality'],r['roi_i'],r['roi_j'])]['folds_in_top100']+'/5'] for r in connections],h=3.5,size=17)
call(s,'Ranks describe reconstructed CN models and fixed topology; avoid biomarker claims.')
s=slide('What the current evidence supports','CONCLUSIONS',source='Saved harmonization, model-comparison and attribution outputs')
bullets(s,['Technical dataset/site information is substantially reduced after harmonization, with mixed preservation diagnostics.','Ridge is the stronger baseline in these experiments; weighted fusion yields a small paired-sample improvement.','GNN fusion improves on the paired unimodal GNNs, but remains behind Ridge.','Attribution highlights candidate connections for investigation; reconstruction discrepancies constrain conclusions.','Next validation priorities: fold-contained harmonization, independent cohort testing, repeated seeds, and exact saved-checkpoint attribution.'],size=21)
s=slide('Result provenance and run selection','APPENDIX',source='Paths are relative to the project results directory.',note='Older DWI summaries report 429 CN scans and inconsistent legacy metrics. They were not mixed with the selected 916-scan threshold summary. Legacy fMRI_brain_age_Ridge outputs were not used in preference to current Ridge_BAG summaries.')
table(s,['Section','Primary saved evidence'],[['Harmonization','DWI_fMRI_fusion/report_tables/Table3_harmonization.csv'],['DWI single modality','DWI_4cohort_joint_ComBat/{Ridge_BAG,GNN_BAG}/*summary*.json / *metrics.json'],['fMRI single modality','ADNI_HABS_ADRC_fMRI_joint_site_harmonization/{Ridge_BAG,GNN_BAG}'],['Shared fusion benchmark','DWI_fMRI_fusion/report_tables/Table1,4,5,6*.csv'],['Interpretability','DWI_fMRI_fusion/GNN_interpretability/*.csv; analysis_summary.json']],h=3.7,size=16)
text(s,.7,5.7,11.8,.7,'Current selected-threshold summaries are used for unimodal GNNs. Older 429-scan DWI summaries and legacy fMRI Ridge outputs are kept separate.',18)
path=OUT/'Connectome_Brain_Age_Results_English_Editable.pptx';prs.save(path)
# Structural validation: text, charts, notes and geometry.
check=Presentation(path);problems=[]
for n,s in enumerate(check.slides,1):
 for sh in s.shapes:
  if sh.left<0 or sh.top<0 or sh.left+sh.width>check.slide_width+10000 or sh.top+sh.height>check.slide_height+10000:problems.append((n,sh.name))
assert not problems,problems
print('Created',path,'slides',len(check.slides),'charts',sum(sh.has_chart for s in check.slides for sh in s.shapes))
