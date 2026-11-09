#!/usr/bin/env python3
"""fMRI nested Ridge: signed Fisher-z edges, training-only ComBat/calibration.

Reuses the DWI nested-CV control flow, without overwriting historical fMRI results.
Full modality populations differ; these are not matched DWI/fMRI comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
from foldwise_dwi import ROOT,load_eligible
from foldwise_fmri import FmriPreprocessor
from run_foldwise_DWI_Ridge import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/final/fMRI/foldwise_Ridge_validated_20261106')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-cn-participants',type=int,default=5)
    p.add_argument('--alphas',type=float,nargs='+',default=[.1,1,10,100,1000,10000,100000])
    args=p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory; existing results are not overwritten')
    if min(args.outer_folds,args.inner_folds)<2 or min(args.alphas)<=0:
        p.error('Need >=2 folds and positive alphas')
    X,meta,exclusions,sources=load_eligible(min_cn_participants=args.min_cn_participants,modality='fMRI')
    args.output.mkdir(parents=True)
    meta.to_csv(args.output/'eligible_metadata_private.csv',index=False)
    exclusions.to_csv(args.output/'eligibility_audit_private.csv',index=False)
    exclusions.groupby(['cohort','exclusion_reason'],dropna=False).size().rename('scans').reset_index().to_csv(args.output/'eligibility_counts.csv',index=False)
    manifest=dict(status='running',modality='fMRI',seed=args.seed,outer_folds=args.outer_folds,inner_folds=args.inner_folds,
                  min_cn_participants=args.min_cn_participants,alphas=args.alphas,sources=sources,
                  input_representation='fixed Fisher-z of signed upper triangle of symmetrized raw 84x84 Pearson correlations',
                  harmonization='none; batch-only location/scale; training-age mean-only. No held-out ages in transforms',
                  scope='known-site restricted population; clinical/protocol validation pending; modality-specific folds')
    for name in ['foldwise_dwi.py','foldwise_fmri.py','run_foldwise_DWI_Ridge.py','run_foldwise_fMRI_Ridge.py']:
        source=Path(__file__).with_name(name)
        manifest.setdefault('code_sha256',{})[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        (args.output/name).write_bytes(source.read_bytes())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    args.preprocessor_class=FmriPreprocessor;args.modality='fMRI'
    manifest.update(run(X,meta,args,args.output))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
