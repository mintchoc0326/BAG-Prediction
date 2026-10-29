from pathlib import Path
p=Path(__file__).resolve().with_name('equations_in_boxes.py')
exec(compile(p.read_text().split('# Save both established filenames.')[0],str(p),'exec'))
# Detailed notation accompanies each diagram, rather than compressing definitions into footnotes.
G={
'Data preparation and quality-control pipeline':[
('i; n','Scan index; number of eligible scans','A participant may contribute multiple visits; i is not a participant index.'),('N; Aᵢ','N = 84 ROIs; Aᵢ is 84 × 84','A matrix entry is connectivity between two ROIs, not a regional volume.'),('vecᵤ; xᵢ','Upper triangle; xᵢ ∈ ℝ³⁴⁸⁶','Take each unordered ROI pair once; omit the diagonal.'),('d; X','d = N(N−1)/2 = 3,486; X is n × d','Rows are scans; columns refer to the same ROI pair across scans.'),('q(i); visit(i)','Participant identifier; visit identifier','q(i) defines grouping in cross-validation; visits remain together.'),('b(i); cᵢ','Technical batch; biological design vector','b(i) combines dataset and site; cᵢ encodes covariates to preserve.')],
'DWI harmonization pipeline':[
('xᵢⱼ; uᵢⱼ','Raw weight; log1p weight, dimensionless','i indexes scans; j indexes unique connections; u = log(1+x).'),('αⱼ; βⱼ; cᵢ','Edge intercept; covariate coefficients; design vector','cᵢ contains age and sex in the saved DWI run; diagnosis is absent.'),('γᵦⱼ; δᵦⱼ','Batch location shift; positive batch scale','b = b(i) is dataset/site, not the Ridge intercept.'),('σⱼ; εᵢⱼ','Edge residual SD; standardized residual','Feature standardization precedes empirical Bayes batch estimation.'),('γ*ᵦⱼ; δ*ᵦⱼ','Empirical Bayes batch parameter estimates','Shrink noisy batch estimates using information across connections.'),('H; expm1','Harmonized quantity; inverse log1p','Reconstruct symmetric weights and apply the export nonnegative convention.')],
'fMRI harmonization pipeline':[
('rᵢⱼ; zᵢⱼ','Correlation; Fisher-z feature','r is clipped inside (−1,1); z = arctanh(r).'),('cᵢ; βⱼ','Covariate design vector; edge coefficients','Age is continuous; sex and diagnosis are categorical design columns.'),('αⱼ; b(i)','Feature intercept; dataset/site batch index','Each eligible scan belongs to one batch; singleton batches are removed.'),('γᵦⱼ; δᵦⱼ','Batch shift; batch scale','Adjustment removes modeled technical location and scale effects.'),('σⱼ; γ*; δ*','Residual SD; empirical Bayes batch estimates','The detailed ComBat diagram shows standardization and reconstruction.'),('zᴴ; rᴴ','Adjusted z feature; adjusted correlation','rᴴ = tanh(zᴴ); reconstruct an 84 × 84 symmetric matrix with diagonal 1.')],
'Harmonization evaluation pipeline':[
('X; μ; V₂','Feature matrix; column means; two PCA loadings','T = (X−μ)V₂ is n × 2; separate before/after fits need not share axes.'),('K; recallₖ','Number of classes; TPₖ/(TPₖ+FNₖ)','Balanced accuracy averages sensitivity equally across classes.'),('yᵢ; ŷᵢ; ȳ','Actual age; estimated age; evaluation mean','Age and age errors are measured in years.'),('MAE; RMSE','Mean absolute error; root mean squared error','MAE = mean |ŷ−y|; RMSE = sqrt(mean (ŷ−y)²).'),('R²','1 − residual sum of squares / total sum of squares','R² < 0 means worse squared error than predicting the evaluation mean.'),('Target-specific n','Eligible scans for each diagnostic target','Sample sizes and class counts differ; do not interpret all rows as one set.')],
'Single-modality Ridge pipeline':[
('xᵢ; Zᵢ','Raw edge vector; standardized edge vector','Both have d = 3,486 entries; DWI and fMRI are fitted separately.'),('μⱼ; σⱼ','Training-column mean and standard deviation','Re-estimate inside every inner split; outer-test observations never fit them.'),('β ∈ ℝᵈ; b ∈ ℝ','Edge coefficients; unpenalized age intercept','ŷ = b + Zβ; β converts standardized feature values into age years.'),('λ; ‖β‖₂²','Nonnegative penalty; sum of squared coefficients','λ is sklearn Ridge alpha; larger values shrink coefficients more strongly.'),('Tₒ; Vₒ','Outer-training set; outer-test set for fold o','All visits from a participant belong to one outer fold.'),('λ*; OOF','Inner-CV selected penalty; out-of-fold prediction','Refit on Tₒ using λ*; predict Vₒ once. Full-data ComBat remains upstream.')],
'Single-modality GNN pipeline':[
('v,u; N(v); Eᵣ','ROI nodes; neighbors of v; selected edge set','Use r for retained fraction to distinguish it from a participant index.'),('hᵥ⁽⁰⁾; hᵥ⁽ˡ⁾','84-value connectivity row; layer-l embedding','h⁽⁰⁾ ∈ ℝ⁸⁴; hidden vectors ∈ ℝ⁶⁴; three GINE layers.'),('eᵤᵥ; Wₗ','Scalar edge weight; learned edge projection','Project scalar edges to the current node-feature dimension before addition.'),('mᵥ⁽ˡ⁾; εₗ','Summed neighbor message; self-feature coefficient','ε is fixed at PyG default zero; the MLP transforms self + neighbor information.'),('g; Head(g)','64-value mean-pooled graph embedding; scalar output','Regressor: 64 → 32 → 1. Its output is standardized age.'),('μᵧ; σᵧ; ŷ','Training age mean / SD; predicted age in years','MSE trains standardized age; grouped inner validation selects epochs.')],
'Cross-modality pairing and training selection':[
('D; F','Superscripts for DWI and fMRI','Superscripts identify modalities; they are not mathematical powers.'),('keyᵢ','Dataset / participant / normalized visit key','Key agreement does not establish acquisition-date proximity.'),('dᵢ; τ','Absolute age disagreement; tolerance τ = 0.01 yr','Exclude unresolved ages from training; do not silently replace them.'),('CNᴰ; CNᶠ','Modality-specific cognitively normal indicators','DWI HABS all-CN override is retained; fMRI requires explicit CN.'),('∧; sex_ok','Logical AND; no explicit sex disagreement','Every required condition must hold; missing sex is allowed in this model.'),('n; P; q(i)','387 scans; 330 participants; scan-to-participant map','Pairing yields 847 scans; policy and consistency filters define the evaluated subset.')],
'BAG and age-bias correction pipeline':[
('f₋fold(i)','Predictor trained without scan i’s outer fold','Participant-level grouping also excludes other visits from that participant.'),('ŷᵢ; yᵢ','Predicted age; chronological age, both in years','ŷ is the raw prediction before subtracting the BAG age trend.'),('BAGᵢ','Raw brain age gap = ŷᵢ − yᵢ','Positive values indicate older predicted age relative to chronological age.'),('a; b','Age-bias intercept (years); slope (years/year)','These are calibration coefficients, distinct from Ridge β and intercept.'),('â; b̂; cBAGᵢ','Fitted coefficients; BAGᵢ − (â+b̂yᵢ)','The CN OOF population defines the reference age trend.'),('Reference vs inference','Descriptive calibration vs independent validation','Fitting and assessing correction on one reference does not independently validate it.')],
'Interpretability pipeline: from OOF models to attribution':[
('f; x; x′','Reconstructed age predictor; observed input; reference','f outputs age years; x jointly contains both modalities’ node and edge attributes.'),('x′ node / edge','Training-mean node values; zero edge attributes','Node mean corresponds to standardized zero; keep observed diagonal fixed.'),('k; t; x(t)','Input coordinate; path fraction; interpolated input','k is not a fold; t ∈ [0,1]; x(t)=x′+t(x−x′); topology is unchanged.'),('∂f/∂xₖ; IGₖ','Local model sensitivity; integrated contribution','IG has age-year units; it explains f(x)−f(x′), not a causal effect.'),('S; rIG','Integration intervals; completeness residual','Trapezoidal quadrature; S = 64 initially, adapt up to 512; |rIG| ≤ 0.05 yr.'),('Δrecon; rIG','Reconstruction error vs integration error','Separate checks: max |Δrecon| = 13.18 yr; six scans fail IG completeness.')],
'Interpretability summaries and robustness checks':[
('Aₛ,ᵢⱼ; p; Vₚ','Scan connection IG; participant; participant visits','i,j now index ROI endpoints; s indexes scans to avoid an index collision.'),('A node; A edge','Node-profile attribution; edge-attribute attribution','Sum both symmetric node entries and both directed edge entries for each pair.'),('Āₚ,ᵢⱼ; Iᵢⱼ','Signed visit average; mean participant absolute score','Average signed scan values first, then take absolute values across participants.'),('ROIᵢ','½ Σⱼ≠ᵢ Iᵢⱼ, in attribution years','Each connection contributes half its absolute importance to each endpoint.'),('rank; top-k frequency','Per-fold ranking; count of folds in top k','k = 100 for connections; k = 10 for ROIs; frequency ranges from 0 to 5.'),('x₋ₘ,x′ₘ; ΔMAEₘ','Keep other input; replace modality m; error change','Average within participant, then across participants; positive = worse prediction.')],
'Ridge fusion pipelines: early, weighted and late':[
('Zᴰ; Zᶠ','DWI and fMRI standardized edge vectors','Each has 3,486 columns; fit modality columns using training data only.'),('[ ; ]; Zᴱ','Feature concatenation; early-fusion vector','Zᴱ = [Zᴰ;Zᶠ] contains 6,972 features.'),('w; Zᵂ','fMRI block scale; weighted concatenation','Zᵂ=[Zᴰ;wZᶠ]; DWI scale is one; w is not a contribution percentage.'),('λ*; w*','Penalty and weight jointly selected by inner CV','Search w ∈ {0,0.25,0.5,1,2,4}; λ is Ridge alpha.'),('β; b','6,972 edge coefficients; scalar intercept','Only β is penalized; never standardize again after block weighting.'),('ŷᴰ; ŷᶠ; ŷlate','Two unimodal OOF ages; arithmetic mean','Late fusion averages predictions in years, not coefficients or embeddings.')],
'GNN fusion pipeline: two modality-specific encoders':[
('Hₘ⁽⁰⁾; m','Node input matrix; modality m ∈ {D,F}','84 × 84 input per modality; DWI log1p weights, fMRI signed correlations.'),('Eₘ; eᵤᵥ,ₘ','Positive top-50% topology; normalized scalar weight','e = selected positive weight / graph positive maximum; bidirectional edges.'),('Hₘ⁽ˡ⁾; gₘ','ROI embeddings; pooled modality vector','Two GINE layers, hidden 32; mean+max pooling makes gₘ ∈ ℝ⁶⁴.'),('g=[gᴰ;gᶠ]','Concatenated fusion representation ∈ ℝ¹²⁸','A separate encoder is learned for each modality; weights are not shared.'),('Head; μᵧ; σᵧ','128→32→1 MLP; training age mean and SD','Head predicts standardized age; multiply by σᵧ and add μᵧ for age years.'),('μₘ,j; σₘ,j','Training node-feature column statistics','Pool training graph nodes per modality; fresh outer refit follows inner epoch selection.')]
}
original=list(prs.slides._sldIdLst);extra={}
# Replace the uninformative reading aid with useful local notation on the diagram itself.
for idx,s in enumerate(list(prs.slides)):
 title=next((sh.text for sh in s.shapes if sh.has_text_frame and abs(sh.top-Inches(.65))<100),'')
 if title not in G:continue
 for sh in list(s.shapes):
  if sh.has_text_frame and sh.text.startswith('READING THE FLOW'):sh._element.getparent().remove(sh._element)
 if title in P:
  local=G[title][:3]
  text(s,.65,3.95,3.65,.3,'NOTATION AT A GLANCE',12,teal,True)
  text(s,.65,4.35,3.65,1.35,'\n'.join(a+'  '+b for a,b,c in local),13,gray)
 # The next slide is a full-size notation key, with no abbreviated symbol footnotes.
 ns=slide(title.replace(' pipeline','')+': notation','METHOD DETAILS',source='Implementation-verified notation | See preceding diagram and speaker notes.')
 tb=table(ns,['Symbol / object','Definition and dimension','How to interpret it'],G[title],y=1.7,h=4.8,size=16)
 tb.columns[0].width=Inches(1.9);tb.columns[1].width=Inches(4.25);tb.columns[2].width=Inches(5.95)
 for row in tb.rows:
  for cell in row.cells:
   for pp in cell.text_frame.paragraphs:pp.space_after=Pt(2)
 text(ns,.65,6.64,12,.3,'Indices and units are local to this diagram; modality superscripts are labels, and hats indicate fitted estimates.',12,gray)
 ns.notes_slide.notes_text_frame.text='Detailed symbol definitions for '+title+'\n'+'\n'.join(' | '.join(r) for r in G[title])
 extra[idx]=[prs.slides._sldIdLst[-1]]
# Correct the DWI covariate statement to match the saved run.
for s in prs.slides:
 for sh in s.shapes:
  if sh.has_text_frame:
   for pp in sh.text_frame.paragraphs:
    for run in pp.runs:
     run.text=run.text.replace('DWI diagnosis inclusion is conditional.','Saved DWI run preserves age and sex only.').replace('c: preserved covariates','c: age + sex in this saved run')
# Additional expanded diagrams are placed immediately after their notation page.
def detailed(anchor,title,source,blocks,legend,bottom):
 ns=slide(title,'MODEL DETAIL',source=source)
 positions=[(.6,1.7),(6.95,1.7),(.6,4.05),(6.95,4.05)]
 for (x,y),(t,f,e) in zip(positions,blocks):node(ns,x,y,5.75,1.8,t,f,e)
 arrow(ns,6.35,2.6,6.95,2.6);arrow(ns,9.825,3.5,9.825,4.05);arrow(ns,6.95,4.95,6.35,4.95)
 text(ns,.65,6.05,12,.42,legend,13,gray);text(ns,.65,6.52,12,.42,bottom,14,teal,True)
 ns.notes_slide.notes_text_frame.text+='\n'+legend+'\n'+bottom
 at=next(i for i,s in enumerate(list(prs.slides)[:len(original)]) if any(sh.has_text_frame and sh.text==anchor for sh in s.shapes))
 extra[at].append(prs.slides._sldIdLst[-1])
detailed('fMRI harmonization pipeline','ComBat: standardize, adjust, restore','run_fMRI_harmonization.py; saved DWI combat_covariates.csv',[
('1  Remove modeled biological mean','qᵢⱼ = α̂ⱼ + cᵢᵀβ̂ⱼ\nsᵢⱼ = (vᵢⱼ − qᵢⱼ)/σ̂ⱼ','v: log1p DWI or Fisher-z fMRI; σ̂: pooled residual SD'),('2  Estimate technical batch effects','sᵢⱼ = γᵦⱼ + δᵦⱼ εᵢⱼ\n(γ*ᵦⱼ, δ*ᵦⱼ) = EB estimates','Empirical Bayes shrinks location / scale estimates across edges'),('4  Restore feature units','vᴴᵢⱼ = σ̂ⱼ sᴴᵢⱼ + qᵢⱼ','Then expm1 for DWI or tanh for fMRI; reconstruct matrices'),('3  Adjust standardized values','sᴴᵢⱼ = (sᵢⱼ − γ*ᵦⱼ)/δ*ᵦⱼ','δ* denotes the estimated SD scale; divide by sqrt(variance)')],'i: scan; j: edge; b: dataset/site batch; c: preserved design; H: harmonized; *: empirical Bayes estimate.','Current ComBat uses all eligible scans before CV; the displayed prediction estimates retain upstream leakage risk.')
detailed('Single-modality Ridge pipeline','Nested validation: what is fitted where','Ridge_GNN_BAG.py; Ridge_fMRI_BAG.py; fusion scripts',[
('1  Hold out outer participants','Tₒ ∩ Vₒ = ∅ at participant level','Five outer folds; Vₒ is never used for scaling or tuning'),('2  Tune inside Tₒ only','λ* = argminλ meaninner MAE(λ)\n(λ*,w*) for weighted fusion','Refit scaler inside each of four grouped inner splits'),('4  Pool held-out predictions','ŷᵢ,OOF = fₒ(xᵢ), i ∈ Vₒ\nMAE = meanᵢ |ŷᵢ,OOF − yᵢ|','Each scan contributes one outer-test prediction; visits share a fold'),('3  Refit on all of Tₒ','μₒ,σₒ ← X[Tₒ]\nfₒ ← Ridge(λ*; Tₒ)','Apply frozen training statistics and coefficients to Vₒ')],'o: outer fold; Tₒ: outer-training scans; Vₒ: outer-test scans; inner: participant-grouped tuning partitions.','This nesting applies to prediction models; upstream full-data harmonization is not nested.')
detailed('Single-modality GNN pipeline','GINE: from connectivity to predicted age','GNN_DWI_BAG.py; GNN_fMRI_BAG.py',[
('1  Form edge-aware messages','mᵥ⁽ˡ⁾ = Σᵤ∈N(v) ReLU(hᵤ⁽ˡ⁾ + Wₗeᵤᵥ)','Layer 0: h ∈ ℝ⁸⁴; later layers: h ∈ ℝ⁶⁴; edge e is scalar'),('2  Mix self and neighbor information','aᵥ⁽ˡ⁺¹⁾ = MLPₗ((1+εₗ)hᵥ⁽ˡ⁾ + mᵥ⁽ˡ⁾)','εₗ=0 in this configuration; learned Wₗ matches feature dimension'),('4  Predict standardized age','g = (1/84)Σᵥ hᵥ⁽³⁾\nŷ = μᵧ + σᵧ Head(g)','Head: 64→32→1; train MSE against (y−μᵧ)/σᵧ'),('3  Normalize and activate','hᵥ⁽ˡ⁺¹⁾ = ReLU(BNₗ(aᵥ⁽ˡ⁺¹⁾))','Three layers; dropout after the first two; BN = batch normalization')],'v/u: ROI nodes; l: message-passing layer; N(v): selected neighbors; μᵧ/σᵧ: inner-training age statistics.','Inner 20% grouped validation selects checkpoints; sparsity selection reuses the reported pooled OOF comparison.')
detailed('Interpretability pipeline: from OOF models to attribution','Integrated gradients: reference and numerical checks','GNN_BAG_interpretability.py; analysis_summary.json',[
('1  Define an explicit reference','x′node = training mean; x′edge = 0\nx(t) = x′ + t(x−x′)','Keep observed node diagonal and edge topology; interpolate both modalities'),('2  Integrate each input coordinate','IGₖ = (xₖ−x′ₖ) ∫₀¹ ∂f(x(t))/∂xₖ dt','Trapezoidal quadrature; f outputs chronological-age prediction in years'),('4  Map inputs to connections','Aᵢⱼ = IGnode,ᵢⱼ + IGnode,ⱼᵢ\n+ IGedge,i→j + IGedge,j→i','Then average signed visits per participant before absolute importance'),('3  Check integration and reconstruction','rIG = f(x)−f(x′)−Σₖ IGₖ\nΔrecon = f(x)−ŷoriginal','Adapt 64→512 intervals; flag |rIG|>0.05 years; inspect Δrecon separately')],'k: transformed-input coordinate; t: path position; i/j: ROI endpoints; f: reconstructed outer-fold predictor.','These explain reconstructed CN predictions relative to a reference, not top-k decisions or causal biology.')
# Reassemble with notation and details directly following their associated pipeline.
for el in list(prs.slides._sldIdLst):prs.slides._sldIdLst.remove(el)
for i,el in enumerate(original):
 prs.slides._sldIdLst.append(el)
 for more in extra.get(i,[]):prs.slides._sldIdLst.append(more)
for n,s in enumerate(prs.slides,1):
 for sh in s.shapes:
  if sh.has_text_frame and sh.left>Inches(12) and sh.top>Inches(7):
   sh.text=f'{n:02}';sh.text_frame.paragraphs[0].font.size=Pt(10)
# Keep notation consistent between diagrams and their detailed keys.
for s in prs.slides:
 title=next((sh.text for sh in s.shapes if sh.has_text_frame and abs(sh.top-Inches(.65))<100),'')
 if title=='Single-modality GNN pipeline':
  for sh in s.shapes:
   if sh.has_text_frame:
    for pp in sh.text_frame.paragraphs:
     for rr in pp.runs:rr.text=rr.text.replace('Eₚ','Eᵣ').replace('Q₁₋ₚ','Q₁₋ᵣ')
 if title=='Interpretability summaries and robustness checks':
  for sh in s.shapes:
   if sh.has_text_frame:
    for pp in sh.text_frame.paragraphs:
     for rr in pp.runs:rr.text=rr.text.replace('Aᵥ,ᵢⱼ','Aₛ,ᵢⱼ').replace('meanᵥ∈visits(p)','meanₛ∈visits(p)').replace('meanᵥ','meanₛ')
# Validate editable shapes and page bounds before saving.
for k,s in enumerate(prs.slides,1):
 for sh in s.shapes:
  assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=prs.slide_width+10000 and sh.top+sh.height<=prs.slide_height+10000,(k,sh.name)
path=OUT/'Connectome_Brain_Age_Results_English_Editable.pptx';prs.save(path)
import shutil
shutil.copyfile(path,OUT/'Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx')
print('Saved',len(prs.slides),'slides; 12 notation pages + 4 expanded method diagrams added.')
