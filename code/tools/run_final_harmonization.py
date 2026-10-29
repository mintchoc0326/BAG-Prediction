"""Run the authoritative Python harmonization sources with snapshots and validation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,os,json,hashlib,shutil,subprocess,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
LOG=R/'results/checks/Harmonization'/stamp;BACK=R/'results/explore/Harmonization_backup'/stamp
LOG.mkdir(parents=True);BACK.mkdir(parents=True)
report={'run_id':stamp,'python':sys.executable,'sources':{},'backup':str(BACK),'outputs':{},'runs':{}}
for mod in ['DWI','fMRI']:
 source=R/f'code/final/run_{mod}_harmonization.py'
 report['sources'][mod]={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 shutil.copy2(source,LOG/source.name)
 shutil.copytree(R/f'results/final/{mod}',BACK/mod)
prep=R/'data/final/ADRC/connectomes/DWI/harmonize_prep_joint'
if prep.exists():shutil.copytree(prep,BACK/'ADRC_prep')
(LOG/'run_manifest.json').write_text(json.dumps(report,indent=2))
print('Snapshots saved:',BACK,flush=True)
def run(mod):
 log=LOG/f'{mod}.log';env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 with log.open('w') as f:
  result=subprocess.run([sys.executable,'-u',str(R/f'code/final/run_{mod}_harmonization.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(mod,'exit code',result.returncode,'log',log,flush=True)
 return mod,{'returncode':result.returncode,'log':str(log)}
with ThreadPoolExecutor(max_workers=2) as pool:
 for mod,result in pool.map(run,['DWI','fMRI']):report['runs'][mod]=result
(LOG/'run_manifest.json').write_text(json.dumps(report,indent=2))
if any(x['returncode'] for x in report['runs'].values()):raise SystemExit('A harmonization run failed; inspect recorded logs.')
for mod in ['DWI','fMRI']:
 folder=R/f'results/final/{mod}';previous=BACK/mod
 if mod=='DWI':
  pre=folder/'arrays/DWI_log1p_before_combat.npy';post=folder/'arrays/DWI_joint_combat.npy'
  x=np.load(pre);y=np.load(post);old=np.load(previous/'arrays/DWI_joint_combat.npy')
  md=pd.read_csv(folder/'metadata/metadata_used_for_joint_combat.csv');keys=['DATASET','__id__']
  oldmd=pd.read_csv(previous/'metadata/metadata_used_for_joint_combat.csv')
  covariates=pd.read_csv(folder/'metadata/combat_covariates.csv').columns.tolist()
  mf=folder/'harmonized_connectomes/harmonized_DWI_manifest.csv'
 else:
  pre=folder/'evaluation_inputs/fMRI_X_raw_edges.csv';post=folder/'evaluation_inputs/fMRI_X_combat_edges.csv'
  predf=pd.read_csv(pre,index_col=0);postdf=pd.read_csv(post,index_col=0)
  assert predf.index.equals(postdf.index) and predf.columns.equals(postdf.columns)
  x=predf.to_numpy();y=postdf.to_numpy();old=pd.read_csv(previous/'evaluation_inputs/fMRI_X_combat_edges.csv',index_col=0).to_numpy()
  md=pd.read_csv(folder/'evaluation_inputs/fMRI_metadata_aligned.csv',index_col=0);keys=['DATASET','connectome_key']
  oldmd=pd.read_csv(previous/'evaluation_inputs/fMRI_metadata_aligned.csv',index_col=0)
  covariates=['Age','Sex','Diagnosis'];mf=folder/'harmonized_connectomes/harmonized_fMRI_manifest.csv'
 assert x.shape==y.shape and len(md)==len(x)
 assert np.isfinite(x).all() and np.isfinite(y).all()
 same_keys=md[keys].reset_index(drop=True).equals(oldmd[keys].reset_index(drop=True))
 delta=float(np.max(np.abs(y-old))) if old.shape==y.shape else None
 same_features=delta is not None and np.allclose(y,old,rtol=0,atol=1e-10)
 manifest=pd.read_csv(mf);assert len(manifest)==len(md)
 missing=[];max_asym=0.;max_diag=0.
 for name in manifest.harmonized_connectome:
  p=Path(name)
  if not p.exists():missing.append(name);continue
  a=pd.read_csv(p,index_col=0).to_numpy()
  if a.shape!=(84,84):
   # Plain matrix CSVs may have no header or index.
   a=pd.read_csv(p,header=None).to_numpy()
  assert a.shape==(84,84) and np.isfinite(a).all(),(p,a.shape)
  max_asym=max(max_asym,float(np.max(np.abs(a-a.T))))
  max_diag=max(max_diag,float(np.max(np.abs(np.diag(a)-(1 if mod=='fMRI' else 0)))))
 assert not missing and max_asym<1e-10 and max_diag<1e-10
 status={'scans':len(md),'features':y.shape[1],'cohort_counts':md.DATASET.value_counts().to_dict(),'covariates':covariates,'manifest':str(mf),'matrices_checked':len(manifest),'missing_matrices':0,'maximum_asymmetry':max_asym,'maximum_diagonal_error':max_diag,'same_sample_keys_as_previous':same_keys,'maximum_harmonized_feature_change':delta,'same_features_within_1e-10':bool(same_features),'downstream_results_status':'Inputs unchanged within tolerance; prior model outputs retained.' if same_keys and same_features else 'Inputs changed; prior model outputs must be considered stale pending rerun.'}
 report['outputs'][mod]=status
 (folder/'harmonization_run.json').write_text(json.dumps({'run_id':stamp,'source':report['sources'][mod],'log':report['runs'][mod]['log'],**status},indent=2))
 print('VERIFIED',mod,json.dumps(status),flush=True)
(LOG/'run_manifest.json').write_text(json.dumps(report,indent=2))
(R/'results/checks/Harmonization/latest.json').write_text(json.dumps({'run_id':stamp,'manifest':str(LOG/'run_manifest.json')},indent=2))
print('COMPLETE:',LOG/'run_manifest.json',flush=True)
