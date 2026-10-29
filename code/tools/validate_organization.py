"""Read-only validation of organized paths, Python-source provenance, and model imports."""
from pathlib import Path
import sys,ast,json,csv,hashlib,tokenize,io
R=Path(__file__).resolve().parents[2];A=R/'results/checks/organization';B=A/'before/code'
report={'syntax':[],'manifests':[],'harmonization_calls':{},'latest_sources':{}}
for p in (R/'code/final').rglob('*.py'):ast.parse(p.read_text());report['syntax'].append(str(p.relative_to(R)))
for p in (R/'code/tools').glob('*.py'):ast.parse(p.read_text());report['syntax'].append(str(p.relative_to(R)))
def tokens(s):
 return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type not in [tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT,tokenize.ENDMARKER]]
report['latest_sources']['DWI']='Authoritative source: code/final/run_DWI_harmonization.py; notebooks excluded by user instruction.'
report['latest_sources']['fMRI']='Authoritative source: code/final/run_fMRI_harmonization.py; notebooks excluded by user instruction.'
for m in ['DWI','fMRI']:
 old=ast.parse((B/f'run_{m}_harmonization.py').read_text());new=ast.parse((R/f'code/final/run_{m}_harmonization.py').read_text())
 def calls(t):return [ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='neuroCombat']
 assert calls(old)==calls(new);report['harmonization_calls'][m]='Identical NeuroCombat calls before and after organization.'
for p in (R/'results/final').rglob('*manifest*.csv'):
 rows=list(csv.DictReader(p.open()));tested=0;missing=[]
 for row in rows:
  for k,v in row.items():
   if v and v.startswith('/mnt/') and ('connectome' in k.lower() or 'path' in k.lower()):
    tested+=1
    if not Path(v).exists():missing.append(v)
 assert not missing,(str(p),missing[:3]);report['manifests'].append({'file':str(p.relative_to(R)),'rows':len(rows),'tested_paths':tested,'missing':0})
sys.path.insert(0,str(R/'code/final'))
from prepare_DWI_fMRI_fusion_metadata import ROOT,DWI,FMRI,load_modality,audit_pairs
assert ROOT==R and DWI==R/'results/final/DWI' and FMRI==R/'results/final/fMRI'
d=load_modality(DWI/'harmonized_connectomes/harmonized_DWI_manifest.csv',DWI/'metadata/metadata_used_for_joint_combat.csv','dwi')
f=load_modality(FMRI/'harmonized_connectomes/harmonized_fMRI_manifest.csv',FMRI/'02_canonical_metadata.csv','fmri')
pair=audit_pairs(d.merge(f,on=['dataset','scan_id'],validate='one_to_one'),.01)
assert len(d)==1234 and len(f)==847 and len(pair)==847
assert pair.path_exists_dwi.all() and pair.path_exists_fmri.all()
eligible=pair.candidate_cn_existing_policies & pair.age_status.eq('agrees_within_tolerance') & ~pair.sex_conflict & pair.metadata_found_dwi & pair.metadata_found_fmri
assert eligible.sum()==387
report['pairing']={'dwi':len(d),'fmri':len(f),'pairs':len(pair),'eligible_before_matrix_qc':int(eligible.sum()),'all_paired_paths_exist':True}
import fmri_bag_cv,fusion_transformers,GNN_DWI_fMRI_fusion,GNN_BAG_interpretability
report['imports']='Current grouped-CV, weighting, GNN fusion and attribution modules import successfully.'
report['original_fold_checkpoints_present']=(R/'results/final/Fusion/GNN_fusion/outer_fold_models').exists()
report['archived_attribution']={'path':'results/explore/GNN_attribution','available':(R/'results/explore/GNN_attribution/analysis_summary.json').exists()}
(A/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
