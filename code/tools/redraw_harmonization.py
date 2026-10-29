from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ROOT=Path(__file__).resolve().parents[2]
df=pd.read_csv(ROOT/'results/final/Fusion/report_tables/Table3_harmonization.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(15,6),sharex=True)
colors=['#007F86','#477AB7']
for ax,mod in zip(axes,['DWI','fMRI']):
 sub=df[(df.modality==mod)&df.target.isin(['dataset','dataset_site_batch'])].set_index('target')
 y=np.array([1,0]);height=.25
 for offset,key,label,color in [(height/2,'before','Before ComBat',colors[0]),(-height/2,'after','After ComBat',colors[1])]:
  vals=[sub.loc[t,key] for t in ['dataset','dataset_site_batch']]
  ax.barh(y+offset,vals,height=height,color=color,label=label,zorder=3)
  for yy,v in zip(y+offset,vals):ax.text(v+.014,yy,f'{v:.3f}',va='center',fontsize=13,fontweight='bold',color='#142B45')
 counts=[int(sub.loc[t,'n_classes']) for t in ['dataset','dataset_site_batch']]
 ax.set_yticks(y,[f'Dataset\n({counts[0]} classes)',f'Dataset / site\n({counts[1]} classes)'])
 for yy,k in zip(y,counts):
  ax.vlines(1/k,yy-.36,yy+.36,colors='#8A95A1',linestyles='dashed',linewidth=1.5,zorder=4)
 ax.set_xlim(0,1.12);ax.set_ylim(-.55,1.55);ax.set_title(mod,loc='left',fontsize=20,fontweight='bold',color='#142B45',pad=18)
 ax.set_xlabel('Balanced accuracy',fontsize=14,labelpad=12)
 ax.xaxis.set_major_locator(MultipleLocator(.2));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 ax.grid(axis='x',color='#DFE5EB',linewidth=.8,zorder=0);ax.set_axisbelow(True)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
 ax.spines['bottom'].set_color('#A7B1BC');ax.tick_params(axis='y',length=0,pad=10)
fig.suptitle('Dataset and site predictability decrease after harmonization',x=.04,ha='left',fontsize=23,fontweight='bold',color='#142B45',y=.98)
handles,labels=axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
handles.append(Line2D([0],[0],color='#8A95A1',linestyle='--',linewidth=1.5));labels.append('Nominal chance (1 / number of classes)')
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.10),ncol=3,frameon=False,fontsize=12)
fig.text(.04,.045,'Lower predictability indicates reduced technical batch information; these are full-data harmonization diagnostics.',fontsize=12,color='#526477')
fig.subplots_adjust(left=.13,right=.96,top=.78,bottom=.28,wspace=.38)
out=ROOT/'results/presentation/figures/Harmonization_Dataset_Site_Predictability'
for ext in ['png','svg','pdf']:fig.savefig(str(out)+'.'+ext,dpi=320,facecolor='white',bbox_inches='tight')
print(out)
