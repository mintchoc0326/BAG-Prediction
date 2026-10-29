from pathlib import Path
p=Path(__file__).resolve().with_name('add_pipelines.py')
exec(compile(p.read_text().rsplit('path=OUT/',1)[0],str(p),'exec'))
# Each step is a single native shape containing title, equation and explanation.
def node(s,x,y,w,h,title,formula,explain,small=False):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string('EAF3F5');sh.line.color.rgb=RGBColor.from_string(teal)
 tf=sh.text_frame;tf.clear();tf.word_wrap=True;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 tf.margin_left=tf.margin_right=Inches(.1)
 for i,(val,sz,col) in enumerate([(title,15 if small else 18,teal),(formula,13 if small else 17,navy),(explain,11 if small else 13,gray)]):
  pp=tf.paragraphs[0] if i==0 else tf.add_paragraph();pp.text=val;pp.alignment=PP_ALIGN.CENTER;pp.font.name='DejaVu Sans' if i==1 else 'Aptos';pp.font.size=Pt(sz);pp.font.bold=i==0;pp.font.color.rgb=RGBColor.from_string(col);pp.space_after=Pt(4 if small else 8)
 return sh
P={
'Data preparation and quality-control pipeline':[
('1  Discover inputs','Aᵢ ∈ ℝ⁸⁴ˣ⁸⁴','Connectome matrices + cohort metadata'),('2  Standardize IDs','key = dataset + participant + visit','Harmonize cohort-specific identifiers'),('3  Check matrices','shape(Aᵢ) = (84, 84)','Require numeric, finite entries'),('4  Extract unique edges','xᵢ = vecᵤ(Aᵢ)\np = 84 × 83 / 2 = 3,486','Upper triangle; exclude the diagonal'),('5  Align covariates','X ∈ ℝⁿˣ³⁴⁸⁶','Match age / sex / diagnosis; retain audit')],
'DWI harmonization pipeline':[
('1  Structural edges','xᵢ = vecᵤ(Aᵢ)','Four cohorts; 3,486 unique connections'),('2  Transform weights','uᵢⱼ = log(1 + xᵢⱼ)','Reduce skew; apply edge QC'),('3  Model batch effects','uᵢⱼ = αⱼ + cᵢᵀβⱼ\n+ γᵦⱼ + δᵦⱼ εᵢⱼ','b: dataset/site; c: preserved covariates'),('4  Remove batch effects','uᴴᵢⱼ = (uᵢⱼ − αⱼ − cᵢᵀβⱼ\n− γ̂ᵦⱼ)/δ̂ᵦⱼ + αⱼ + cᵢᵀβⱼ','Schematic ComBat; empirical Bayes estimates'),('5  Reconstruct matrices','xᴴᵢⱼ = exp(uᴴᵢⱼ) − 1','Restore symmetric matrices; nonnegative export')],
'fMRI harmonization pipeline':[
('1  Functional edges','rᵢ = vecᵤ(Aᵢ)','ADNI / HABS / ADRC; 3,486 correlations'),('2  Fisher-z transform','zᵢⱼ = arctanh(rᵢⱼ)','Clip correlations before transformation'),('3  Model batch effects','zᵢⱼ = αⱼ + cᵢᵀβⱼ\n+ γᵦⱼ + δᵦⱼ εᵢⱼ','c: age, sex, diagnosis; b: dataset/site'),('4  Adjust in z-space','zᴴᵢⱼ = (zᵢⱼ − αⱼ − cᵢᵀβⱼ\n− γ̂ᵦⱼ)/δ̂ᵦⱼ + αⱼ + cᵢᵀβⱼ','Schematic ComBat; preserve modeled biology'),('5  Restore correlations','rᴴᵢⱼ = tanh(zᴴᵢⱼ)','Rebuild symmetric matrices; diagonal = 1')],
'Harmonization evaluation pipeline':[
('1  Align diagnostic inputs','X_before ↔ X_after','Use matching eligible scans per target'),('2  Inspect PCA','T = (X − μ) V₂','Separate PCA views; color dataset / site'),('3  Measure technical signal','BA = (1/K) Σₖ recallₖ','Predict dataset and dataset/site batch'),('4  Assess biological signal','MAE = (1/n) Σᵢ |ŷᵢ − yᵢ|','Age MAE; sex / diagnosis balanced accuracy'),('5  Compare before / after','R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²','Lower technical signal; assess biological loss')],
'Single-modality Ridge pipeline':[
('1  Extract CN features','xᵢ ∈ ℝ³⁴⁸⁶','DWI or fMRI; train on CN scans'),('2  Split by participant','train groups ∩ test groups = ∅','Five outer folds; visits stay together'),('3  Scale training features','Zᵢⱼ = (xᵢⱼ − μⱼ)/σⱼ','μ / σ from each training split only'),('4  Tune Ridge penalty','min over β,b:\nΣtrain(yᵢ − b − Zᵢβ)² + λ‖β‖₂²','Four grouped inner folds; b unpenalized'),('5  Predict held-out age','ŷᵢ = b̂ + Zᵢβ̂','Refit selected λ on outer-training data')],
'Single-modality GNN pipeline':[
('1  Construct node features','hᵥ⁽⁰⁾ = A[v, :] ∈ ℝ⁸⁴','DWI nonnegative; fMRI signed profiles'),('2  Select positive topology','Eₚ = {(u,v): Aᵤᵥ > 0,\nAᵤᵥ ≥ Q₁₋ₚ(A⁺)}','Compare top 10 / 25 / 50 / 75 / 100%'),('3  Aggregate messages','mᵥ⁽ˡ⁾ = Σᵤ∈N(v)\nReLU(hᵤ⁽ˡ⁾ + Wₗeᵤᵥ)','Project scalar edge weights into feature space'),('4  Update ROI embeddings','hᵥ⁽ˡ⁺¹⁾ = MLPₗ(\n(1 + εₗ)hᵥ⁽ˡ⁾ + mᵥ⁽ˡ⁾)','Three GINE layers + normalization / activation'),('5  Pool and predict','g = meanᵥ hᵥ⁽³⁾\nŷ = μᵧ + σᵧ · Head(g)','CN grouped OOF; select sparsity by MAE')],
'Cross-modality pairing and training selection':[
('1  Pair modalities','keyᴰᵢ = keyᶠᵢ','Dataset + participant + normalized visit'),('2  Audit ages','dᵢ = |ageᴰᵢ − ageᶠᵢ|','Record conflicts; do not adjudicate silently'),('3  Apply CN policies','CNᵢ = CNᴰᵢ ∧ CNᶠᵢ','Retain the existing DWI HABS override'),('4  Retain eligible scans','eligibleᵢ = CNᵢ ∧ (dᵢ ≤ 0.01)\n∧ no explicit sex conflict','Age tolerance in years; missing sex allowed'),('5  Freeze shared folds','n = 387 scans; P = 330','Same participant-grouped outer folds')],
'BAG and age-bias correction pipeline':[
('1  Collect OOF predictions','ŷᵢ = f₋fold(i)(xᵢ)','CN participant excluded from training'),('2  Compute raw BAG','BAGᵢ = ŷᵢ − yᵢ','Positive: predicted older than actual age'),('3  Estimate age bias','(â,b̂) = argminₐ,ᵦ\nΣCN-OOF[BAGᵢ − (a + byᵢ)]²','Fit age-related trend on the CN reference'),('4  Subtract age trend','cBAGᵢ = BAGᵢ − (â + b̂yᵢ)','Age-adjusted descriptive brain age gap'),('5  Summarize by cohort','mean cBAG = (1/n) Σᵢ cBAGᵢ','Separate OOF from final in-sample predictions')],
'Interpretability pipeline: from OOF models to attribution':[
('1  Reconstruct fold models','f = f_reconstructed,fold','Recorded epochs; explain held-out CN only'),('2  Verify predictions','Δᵢ = f(xᵢ) − ŷoriginal,ᵢ','Inspect differences before interpretation'),('3  Define reference path','x(t) = x′ + t(x − x′)','Training-mean node / zero-edge reference'),('4  Integrate gradients','IGₖ = (xₖ − x′ₖ)\n× ∫₀¹ ∂f(x(t))/∂xₖ dt','Node + edge attribution; fixed graph topology'),('5  Check completeness','r = f(x) − f(x′) − Σₖ IGₖ','Tolerance: |r| ≤ 0.05 yr; up to 512 steps')],
'Interpretability summaries and robustness checks':[
('1  Average visits','Āₚ,ᵢⱼ = meanᵥ∈visits(p) Aᵥ,ᵢⱼ','Signed connection IG; equal participant weight'),('2  Rank connections','Iᵢⱼ = meanₚ |Āₚ,ᵢⱼ|','Take absolute value after visit averaging'),('3  Score regions','ROIᵢ = ½ Σⱼ≠ᵢ Iᵢⱼ','Allocate half to each connection endpoint'),('4  Check rank stability','Sᵢⱼ = countfolds(rankᵢⱼ ≤ 100)','ROI: top 10; connections: top 100'),('5  Reference occlusion','ΔMAEₘ = meanₚ meanᵥ\n[|f(x₋ₘ,x′ₘ) − y| − |f(x) − y|]','Replace one modality; positive = higher error')]
}
for s in prs.slides:
 if not any(sh.has_text_frame and sh.text=='EDITABLE PIPELINE' for sh in s.shapes):continue
 title=next(sh.text for sh in s.shapes if sh.has_text_frame and abs(sh.top-Inches(.65))<100)
 caption=next((sh.text for sh in s.shapes if sh.has_text_frame and Inches(6.3)<=sh.top<Inches(6.8)),'')
 for sh in list(s.shapes):
  if Inches(1.5)<=sh.top<Inches(6.95):sh._element.getparent().remove(sh._element)
 if title in P:
  coords=[(.6,1.72),(4.78,1.72),(8.96,1.72),(8.96,3.88),(4.78,3.88)]
  for (x,y),(t,f,e) in zip(coords,P[title]):node(s,x,y,3.77,1.85,t,f,e)
  arrow(s,4.37,2.645,4.78,2.645);arrow(s,8.55,2.645,8.96,2.645);arrow(s,10.845,3.57,10.845,3.88);arrow(s,8.96,4.805,8.55,4.805)
  text(s,.7,4.05,3.4,1.45,'READING THE FLOW\n1 → 2 → 3\n             ↓\n         5 ← 4',16,gray)
  # Preserve methodological detail in speaker notes; keep slide readable.
  text(s,.65,6.13,12,.65,caption,15,teal,True)
 elif title=='Ridge fusion pipelines: early, weighted and late':
  lanes=[('EARLY', [('Features','xᴰ, xᶠ','3,486 edges each'),('Scale','Z = (X − μ)/σ','Training splits only'),('Join','Zᴱ = [Zᴰ ; Zᶠ]','6,972 features'),('Tune penalty','min Σ(y − b − Zᴱβ)²\n+ λ‖β‖²','Grouped inner CV'),('Predict','ŷ = b̂ + Zᴱβ̂','Held-out outer fold')]),('WEIGHTED',[('Features','xᴰ, xᶠ','Same paired CN scans'),('Scale','Z = (X − μ)/σ','Before block weighting'),('Weight & join','Zᵂ = [Zᴰ ; wZᶠ]','w is a feature scale'),('Tune jointly','(λ*, w*) = argmin\nMAE_inner(λ, w)','Grouped inner CV'),('Predict','ŷ = b̂ + Zᵂβ̂','Refit selected λ and w')]),('LATE MEAN',[('Features','xᴰ, xᶠ','Same outer folds'),('Separate tuning','fᴰ = Ridgeᴰ\nfᶠ = Ridgeᶠ','Scaler / λ per modality'),('Separate ages','ŷᴰ = fᴰ(xᴰ)\nŷᶠ = fᶠ(xᶠ)','Held-out predictions'),('Average','ŷ = (ŷᴰ + ŷᶠ)/2','Equal prediction weights'),('BAG','BAG = ŷ − y','Report pooled OOF metrics')])]
  for k,(label,steps) in enumerate(lanes):
   y=1.82+k*1.43;text(s,.62,y-.28,12,.28,label,11,teal,True)
   for j,(t,f,e) in enumerate(steps):
    x=.6+j*2.47;node(s,x,y,2.2,1.13,t,f,e,True)
    if j<4:arrow(s,x+2.2,y+.565,x+2.47,y+.565)
  text(s,.65,6.4,12,.5,'D / F: DWI / fMRI; μ / σ: training statistics; λ: Ridge penalty; w: fMRI block weight.',14,gray)
 elif title=='GNN fusion pipeline: two modality-specific encoders':
  for y,mod,f in [(1.8,'DWI','Aᴰ = log(1 + clip(A,0,∞))'),(3.95,'fMRI','Aᶠ = signed correlations')]:
   node(s,.6,y,3.15,1.65,mod+' graph',f+'\nE = top 50% positive edges','Training-only node statistics')
   node(s,4.15,y,3.15,1.65,'GINE encoder × 2','h⁽ˡ⁺¹⁾ = GINEₗ(h⁽ˡ⁾, e)\ngₘ = [meanᵥ hᵥ ; maxᵥ hᵥ]','Separate modality-specific weights')
   arrow(s,3.75,y+.825,4.15,y+.825)
  node(s,7.85,2.85,2.05,1.7,'Concatenate','g = [gᴰ ; gᶠ]','D / F: modality embeddings')
  node(s,10.35,2.85,2.38,1.7,'Age prediction','ŷ = μᵧ + σᵧ\n× MLP(g)','Restore standardized age to years')
  arrow(s,7.3,2.625,7.85,3.4);arrow(s,7.3,4.775,7.85,4);arrow(s,9.9,3.7,10.35,3.7)
  text(s,.65,6.2,12,.6,caption,15,teal,True)
 s.notes_slide.notes_text_frame.text+='\nEquations are embedded in their corresponding editable pipeline shapes. Symbol key: i=scan; j=edge; v/u=ROI nodes; p=participant or retained fraction according to context; y=age; hats=estimated parameters. ComBat equations are schematic, suppressing feature residual standardization and empirical Bayes details.'
# Save both established filenames.
import shutil
path=OUT/'Connectome_Brain_Age_Results_English_Editable.pptx';prs.save(path)
shutil.copyfile(path,OUT/'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx')
check=Presentation(path);assert len(check.slides)==36
print('Saved 36 slides; formulas are inside native editable pipeline nodes.')
