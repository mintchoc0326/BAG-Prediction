"""Plot fresh ComBat inputs/outputs following evaluation PCA preprocessing.

DWI: log1p edges -> StandardScaler -> PCA(2, seed=42).
fMRI: Fisher-z edges -> median imputation -> StandardScaler -> PCA(2, seed=42).
Before and after are fitted independently, as in the existing evaluation code.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/final'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','pdf.fonttype':42})
colors={'ADNI':'#4477AA','HABS':'#EE7733','ADRC':'#228833','ADDECODE':'#AA3377'}
records={}
for mod in ['DWI','fMRI']:
 folder=RUN/mod
 OUT=folder/'harmonization_evaluation';OUT.mkdir(parents=True,exist_ok=True)
 if mod=='DWI':
  paths=[folder/'arrays/DWI_log1p_before_combat.npy',folder/'arrays/DWI_joint_combat.npy',folder/'metadata/metadata_used_for_joint_combat.csv']
  arrays=[np.load(p).astype(np.float32) for p in paths[:2]];meta=pd.read_csv(paths[2]);key='DATASET'
  keep=np.isfinite(arrays[0]).all(axis=0)&np.isfinite(arrays[1]).all(axis=0)
 else:
  paths=[folder/'evaluation_inputs'/n for n in ['fMRI_X_raw_edges.csv','fMRI_X_combat_edges.csv','fMRI_metadata_aligned.csv']]
  dfs=[pd.read_csv(p,index_col=0) for p in paths[:2]];md=pd.read_csv(paths[2],index_col=0)
  assert dfs[0].index.equals(dfs[1].index) and dfs[0].index.equals(md.index),'fMRI row keys differ'
  assert dfs[0].columns.equals(dfs[1].columns),'fMRI edge order differs'
  arrays=[d.to_numpy(dtype=np.float32) for d in dfs];meta=md.reset_index(drop=True);key='DATASET'
  for a in arrays:a[~np.isfinite(a)]=np.nan
  keep=~np.all(np.isnan(arrays[0]),axis=0)&~np.all(np.isnan(arrays[1]),axis=0)
 assert arrays[0].shape==arrays[1].shape and len(meta)==len(arrays[0])
 labels=meta[key].astype('string').str.strip().str.upper();valid=labels.notna()&~labels.isin(['','NAN','NONE','UNKNOWN','NA','N/A'])
 meta=meta.loc[valid].reset_index(drop=True);labels=labels.loc[valid].reset_index(drop=True)
 cohorts=[c for c in colors if c in set(labels)]
 fig,axs=plt.subplots(1,2,figsize=(14.4,6.1));stages={};score_tables=[]
 for ax,a,stage in zip(axs,arrays,['Before ComBat','After ComBat']):
  x=a[valid.to_numpy()][:,keep]
  if mod=='fMRI':x=SimpleImputer(strategy='median').fit_transform(x)
  scaled=StandardScaler().fit_transform(x)
  pca=PCA(n_components=2,random_state=42);scores=pca.fit_transform(scaled)
  for c in cohorts:
   mask=labels.eq(c).to_numpy(dtype=bool)
   ax.scatter(scores[mask,0],scores[mask,1],s=20,alpha=.7,color=colors[c],label=f'{c} (n = {mask.sum():,})',edgecolors='none',rasterized=False)
  ev=pca.explained_variance_ratio_*100
  ax.set_xlabel(f'PC1 ({ev[0]:.1f}% variance)',fontsize=14);ax.set_ylabel(f'PC2 ({ev[1]:.1f}% variance)',fontsize=14)
  ax.set_title(stage,fontsize=19,fontweight='bold',color='#142B45',pad=14)
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18,zorder=0);ax.set_axisbelow(True);ax.margins(.07)
  ax.text(.02,.98,f'{mod} · {len(labels):,} scans',transform=ax.transAxes,va='top',fontsize=12,color='#526477')
  stage_key=stage.replace(' ','_').lower();stages[stage_key]={'n':len(labels),'features':int(keep.sum()),'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'independent_fit':True}
  sc=meta.copy();sc['stage']=stage;sc['PC1']=scores[:,0];sc['PC2']=scores[:,1];score_tables.append(sc)
 handles,legend_labels=axs[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=len(cohorts),frameon=False,markerscale=1.6,fontsize=13)
 fig.text(.5,.025,'Separate standardized PCA fits; axes are specific to each panel. All eligible scans are shown.',ha='center',fontsize=11,color='#526477')
 fig.subplots_adjust(left=.08,right=.985,bottom=.24,top=.88,wspace=.28)
 stem=OUT/f'{mod}_PCA_by_dataset_before_after'
 for ext in ['png','svg','pdf']:fig.savefig(str(stem)+'.'+ext,dpi=400,facecolor='white',bbox_inches='tight')
 plt.close(fig)
 pd.concat(score_tables,ignore_index=True).to_csv(str(stem)+'_scores.csv',index=False)
 records[mod]={'inputs':[{ 'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'preprocessing':__doc__,'cohort_counts':labels.value_counts().to_dict(),'stages':stages}
 (OUT/'PCA_by_dataset_provenance.json').write_text(json.dumps(records[mod],indent=2))
 print(mod,json.dumps(records[mod]['cohort_counts']),json.dumps(stages),flush=True)
(ROOT/'results/checks/Harmonization/PCA_provenance.json').write_text(json.dumps(records,indent=2))
print('Saved by-dataset PCA plots in results/final/{DWI,fMRI}/harmonization_evaluation/')
