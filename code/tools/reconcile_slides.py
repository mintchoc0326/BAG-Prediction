"""Reconcile the reviewed deck with current code, archived attribution and PCA outputs."""
from pathlib import Path
import sys,math,json,shutil
try:import pptx
except ImportError:sys.path.insert(0,'/tmp/avery_slides_deps')
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from PIL import Image
R=Path(__file__).resolve().parents[2];P=R/'results/presentation';path=P/'Connectome_Brain_Age_Results_English_Editable.pptx'
r=Presentation(path)
navy='142B45';teal='007F86';gray='526477';blue='477AB7'
def txt(s,x,y,w,h,value,size=18,color=navy,bold=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 for i,l in enumerate(value.split('\n')):
  pp=tf.paragraphs[0] if i==0 else tf.add_paragraph();pp.text=l;pp.font.name='Aptos';pp.font.size=Pt(size);pp.font.color.rgb=RGBColor.from_string(color);pp.font.bold=bold
 return sh
def title(s):return next((sh.text for sh in s.shapes if sh.has_text_frame and abs(sh.top-Inches(.65))<100),'')
def settext(sh,val):
 ps=sh.text_frame.paragraphs;old=ps[0].runs[0] if ps[0].runs else None
 size=(old.font.size if old else None) or ps[0].font.size or Pt(18);bold=(old.font.bold if old and old.font.bold is not None else ps[0].font.bold)
 color=navy
 if old:
  try:color=str(old.font.color.rgb)
  except Exception:pass
 sh.text=val
 for pp in sh.text_frame.paragraphs:pp.font.name='Aptos';pp.font.size=size;pp.font.bold=bold;pp.font.color.rgb=RGBColor.from_string(color)
def page(name,kicker='METHOD STATUS'):
 s=r.slides.add_slide(r.slide_layouts[6]);txt(s,.48,.18,12,.3,kicker,11,teal,True);txt(s,.48,.65,12.3,.8,name,29,navy,True)
 txt(s,.5,7.06,11.8,.29,'Current code and saved-result audit | results/checks/organization/Pipeline_Audit.md',9,gray)
 txt(s,12.25,7.03,.6,.3,'00',10,gray);return s
# Preserve a copy of the pre-audit deliverables.
backup=R/'results/checks/organization/before/presentation';backup.mkdir(parents=True,exist_ok=True)
for ext in ['pptx','pdf']:
 src=P/f'Connectome_Brain_Age_Results_English_Editable.{ext}'
 if not (backup/src.name).exists():shutil.copy2(src,backup/src.name)
# Data preserving native vector replacement for charts that export incorrectly through LibreOffice.
chart_data=[]
for sn,s in enumerate(list(r.slides),1):
 for sh in list(s.shapes):
  if not sh.has_chart:continue
  c=sh.chart;cats=[v.label for v in c.plots[0].categories];series=[(se.name,list(se.values)) for se in c.series]
  chart_data.append({'slide_title':title(s),'categories':cats,'series':series})
  x,y,w,h=[v/Inches(1) for v in [sh.left,sh.top,sh.width,sh.height]]
  sh._element.getparent().remove(sh._element)
  left=x+3.25;plotw=w-4.08;top=y+.18;usable=h-.9;rh=usable/len(cats)
  maximum=max(max(vals) for _,vals in series)
  limit=1 if maximum<=1 else (math.ceil(maximum/2)*2)
  def line(xx,yy,ww,hh,col):
   rr=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(xx),Inches(yy),Inches(max(ww,.008)),Inches(hh));rr.fill.solid();rr.fill.fore_color.rgb=RGBColor.from_string(col);rr.line.fill.background()
  for k in range(6):
   xx=left+plotw*k/5;line(xx,top-.03,.009,usable,'DFE5EB');txt(s,xx-.2,top+usable+.06,.55,.25,f'{limit*k/5:g}',10,gray)
  for i,cat in enumerate(cats):
   yy=top+i*rh;txt(s,x,yy+.05,3.15,rh,str(cat),13 if len(cats)>6 else 15)
   bh=min(.22,rh*.68/len(series));group=bh*len(series)
   for j,(name,vals) in enumerate(series):
    by=yy+(rh-group)/2+j*bh;val=vals[i];length=plotw*val/limit
    line(left,by,length,bh*.82,[teal,blue,'D9883D'][j%3]);txt(s,left+length+.06,by-.04,.7,.28,f'{val:.3f}' if limit==1 else f'{val:.2f}',11,navy,True)
  for j,(name,_) in enumerate(series):
   lx=left+j*2.6;line(lx,top+usable+.55,.19,.13,[teal,blue,'D9883D'][j%3]);txt(s,lx+.26,top+usable+.49,2.35,.3,name,11,gray)
if chart_data:(R/'results/checks/organization/chart_data.json').write_text(json.dumps(chart_data,indent=2))
# Update only current-workflow diagrams and notation; historical numerical results stay historical.
current_titles=['Interpretability pipeline: from OOF models to attribution','Interpretability: from OOF models to attribution: notation','Integrated gradients: reference and numerical checks']
for s in r.slides:
 t=title(s)
 if t in current_titles:
  for sh in s.shapes:
   if sh.has_text_frame:
    v=sh.text
    v=v.replace('1  Reconstruct fold models','1  Load original fold model').replace('f = f_reconstructed,fold','f = f_saved,fold').replace('Recorded epochs; explain held-out CN only','Original weights + normalizers; no retraining')
    v=v.replace('Reconstruction difference reaches 13.18 years; report these as reconstructed-model explanations.','Prediction gate: checkpoint ↔ OOF CSV ≤ 0.0001 yr; inference ↔ checkpoint ≤ 0.001 yr.')
    v=v.replace('Reconstructed age predictor','Saved outer-fold age predictor')
    v=v.replace('reconstructed outer-fold predictor','saved outer-fold predictor').replace('reconstructed age predictor','saved outer-fold age predictor').replace('reconstructed-model','saved-model')
    v=v.replace('Δrecon = f(x)−ŷoriginal','Δpred = f(x)−ŷoriginal').replace('Check integration and reconstruction','Check integration and prediction').replace('inspect Δrecon separately','verify prediction consistency separately')
    v=v.replace('These explain reconstructed CN predictions relative to a reference, not top-k decisions or causal biology.','The current code explains original saved OOF models; corresponding verified outputs are not yet available here.')
    if v!=sh.text:settext(sh,v)
   if sh.has_table:
    for row in sh.table.rows:
     for cell in row.cells:
      for pp in cell.text_frame.paragraphs:
       for rr in pp.runs:rr.text=rr.text.replace('Reconstructed age predictor','Saved outer-fold age predictor').replace('Δrecon; rIG','Δpred; rIG').replace('Reconstruction error vs integration error','Prediction consistency vs integration error').replace('Separate checks: max |Δrecon| = 13.18 yr; six scans fail IG completeness.','Require inference/checkpoint difference ≤ 0.001 yr; IG completeness is assessed separately.')
 archived=t in ['Interpretability explains the reconstructed fusion GNN','Attribution quality limits the strength of interpretation','fMRI carries larger attribution in this fusion model','Top ROI attributions emphasize fMRI regions','Leading connections and stability offer complementary views'] or t.startswith('Archived attribution:')
 if archived:
  for sh in s.shapes:
   if sh.has_text_frame and sh.top<Inches(.5):settext(sh,'ARCHIVED ATTRIBUTION — RECONSTRUCTED MODELS')
  if t=='Interpretability explains the reconstructed fusion GNN':
   for sh in s.shapes:
    if sh.has_text_frame and sh.text==t:settext(sh,'Archived attribution: reconstructed fusion GNN')
  txt(s,.65,6.78,12,.24,'Historical results: results/explore/GNN_attribution. Not the output of the current saved-checkpoint workflow.',10,'A35B22')
  s.notes_slide.notes_text_frame.text+='\nAUDIT: Numerical results are archived reconstructed-model outputs. Current checkpoint-based attribution has not produced a verified replacement in this workspace.'
 for sh in s.shapes:
  if sh.has_text_frame and 'GNN attribution is exploratory: reconstructed predictions differ' in sh.text:
   settext(sh,sh.text.replace('GNN attribution is exploratory: reconstructed predictions differ from the original OOF predictions.','Available attribution is archived; the current code uses original saved OOF checkpoints, whose verified outputs are pending.'))
# Replace the old PCA collage with two readable modality-specific paired plots.
if not any(title(s)=='PCA by dataset: DWI' for s in r.slides):
 for s in list(r.slides):
  if title(s)!='PCA provides a visual check of cohort mixing':continue
  for sh in list(s.shapes):
   if sh.top>=Inches(1.4) and sh.top<Inches(6.99):sh._element.getparent().remove(sh._element)
  for sh in s.shapes:
   if sh.has_text_frame and sh.text=='PCA provides a visual check of cohort mixing':settext(sh,'PCA by dataset: DWI')
  for mod,target in [('DWI',s),('fMRI',page('PCA by dataset: fMRI','HARMONIZATION'))]:
   im=R/f'results/final/{mod}/harmonization_evaluation/{mod}_PCA_by_dataset_before_after.png';iw,ih=Image.open(im).size
   ww=12.1;hh=ww*ih/iw;target.shapes.add_picture(str(im),Inches(.6),Inches(1.55),width=Inches(ww),height=Inches(hh))
   target.notes_slide.notes_text_frame.text='Fresh rerun of the current harmonization source. Dataset colors and legend match. Before/after PCA fit independently after StandardScaler; fMRI also median-imputes. All eligible scans retained. Source: '+str(im)
   if mod=='fMRI':
    el=r.slides._sldIdLst[-1];r.slides._sldIdLst.remove(el);idx=list(r.slides._sldIdLst).index(s.slide_id and next(z for z in r.slides._sldIdLst if int(z.id)==s.slide_id));r.slides._sldIdLst.insert(idx+1,el)
  break
if not any(title(s)=='Current attribution code and available outputs' for s in r.slides):
 s=page('Current attribution code and available outputs')
 txt(s,.7,1.75,11.8,.8,'Current code: load original outer-fold checkpoints → verify predictions → compute IG.',24,teal,True)
 txt(s,.7,2.9,11.8,2.5,'Each checkpoint contains the learned weights, node normalizers, age statistics, fold/sample identity and held-out predictions.\nThe current workspace does not yet contain the required outer_fold_models directory.\nThe following attribution figures are retained as archived reconstructed-model results; their 13.18-year mismatch is not a property of the updated checkpoint workflow.',21)
 txt(s,.7,6,11.8,.65,'No new attribution results or improved consistency were claimed without running and verifying the current workflow.',17,teal,True)
 el=r.slides._sldIdLst[-1];r.slides._sldIdLst.remove(el)
 idx=next(i for i,sl in enumerate(r.slides) if title(sl).startswith('Archived attribution:'))
 r.slides._sldIdLst.insert(idx,el)
for i,s in enumerate(r.slides,1):
 for sh in s.shapes:
  if sh.has_text_frame and sh.left>Inches(12) and sh.top>Inches(7):settext(sh,f'{i:02}')
  if sh.has_text_frame:
   for pp in sh.text_frame.paragraphs:
    for rr in pp.runs:
     rr.text=rr.text.replace('DWI_4cohort_joint_ComBat/','final/DWI/').replace('ADNI_HABS_ADRC_fMRI_joint_site_harmonization/','final/fMRI/').replace('DWI_fMRI_fusion/','final/Fusion/').replace('GNN_interpretability/*.csv','results/explore/GNN_attribution/*.csv')
for s in r.slides:
 t=title(s)
 for sh in s.shapes:
  if sh.has_table:
   for row in sh.table.rows:
    for cell in row.cells:
     for pp in cell.text_frame.paragraphs:
      for rr in pp.runs:
       rr.text=rr.text.replace('DWI_fMRI_fusion/GNN_interpretability/','explore/GNN_attribution/').replace('DWI_fMRI_fusion/','final/Fusion/').replace('DWI_4cohort_joint_ComBat/','final/DWI/').replace('ADNI_HABS_ADRC_fMRI_joint_site_harmonization/','final/fMRI/')
  if sh.has_text_frame:
   if t=='What the current evidence supports' and 'Attribution highlights candidate' in sh.text:
    settext(sh,sh.text.replace('Attribution highlights candidate connections for investigation; reconstruction discrepancies constrain conclusions.','Archived attribution suggests exploratory candidates; verified original-checkpoint attribution results remain pending.'))
   if t=='PCA by dataset: DWI' and sh.top>Inches(7) and sh.left<Inches(12):settext(sh,'results/checks/PCA/figures/DWI_PCA_by_dataset_before_after.png')
   if t.startswith('Interpretability pipeline:') and 'METHODS.txt' in sh.text and sh.top>Inches(7):settext(sh,'code/final/GNN_BAG_interpretability.py | Current checkpoint workflow; result generation pending')
# Restore explicit slide roles after text replacement; inherited theme defaults are too large for captions.
for s in r.slides:
 seen=set()
 for sh in list(s.shapes):
  if not sh.has_text_frame:continue
  sig=(sh.text,sh.left,sh.top,sh.width,sh.height)
  if sig in seen and sh.text.startswith('Historical results:'):
   sh._element.getparent().remove(sh._element);continue
  seen.add(sig)
  size=col=bold=None
  if abs(sh.top-Inches(.65))<100:size,col,bold=29,navy,True
  elif sh.top<Inches(.5):size,col,bold=11,teal,True
  elif sh.top>Inches(7):size,col,bold=(10 if sh.left>Inches(12) else 9),gray,False
  elif sh.text.startswith('Prediction gate:'):size,col,bold=14,teal,True
  elif sh.text.startswith('f; x; x′  '):size,col,bold=13,gray,False
  if size:
   for pp in sh.text_frame.paragraphs:
    pp.font.size=Pt(size);pp.font.bold=bold;pp.font.color.rgb=RGBColor.from_string(col)
    for rr in pp.runs:rr.font.size=Pt(size);rr.font.bold=bold;rr.font.color.rgb=RGBColor.from_string(col)
  if sh.shape_type==1 and sh.text and len(sh.text_frame.paragraphs)==3:
   for j,pp in enumerate(sh.text_frame.paragraphs):
    small=sh.height<Inches(1.3);sz=([15,13,11] if small else [18,17,13])[j];cc=[teal,navy,gray][j]
    pp.font.size=Pt(sz);pp.font.bold=j==0;pp.font.color.rgb=RGBColor.from_string(cc);pp.space_after=Pt(4 if small else 8)
    for rr in pp.runs:rr.font.size=Pt(sz);rr.font.bold=j==0;rr.font.color.rgb=RGBColor.from_string(cc)
r.save(path);shutil.copyfile(path,P/'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx')
print('Reconciled deck:',len(r.slides),'slides; charts are editable vectors; attribution status corrected.')
