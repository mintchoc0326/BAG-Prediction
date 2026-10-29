"""Run the existing harmonization scripts in isolated output directories."""
from pathlib import Path
import ast,sys,json,hashlib
from IPython.display import display
ROOT=Path(__file__).resolve().parents[2]
mod=sys.argv[1]
source=ROOT/'code/final'/f'run_{mod}_harmonization.py'
out=ROOT/'results/checks/PCA'/mod
out.mkdir(parents=True,exist_ok=True)
tree=ast.parse(source.read_text())
class Redirect(ast.NodeTransformer):
 def visit_Assign(self,node):
  self.generic_visit(node)
  names=[t.id for t in node.targets if isinstance(t,ast.Name)]
  replacement=None
  if ('RESULTS_DIR' if mod=='DWI' else 'OUTPUT_DIR') in names:replacement=out
  if mod=='DWI' and 'ADRC_PREP_DIR' in names:replacement=out/'ADRC_harmonize_prep'
  if 'OUTPUT_CONNECTOME_DIR' in names:replacement=out/'harmonized_connectomes'
  if replacement is not None:node.value=ast.parse('Path('+repr(str(replacement))+')',mode='eval').body
  return node
ns={'__name__':'__main__','__file__':str(source),'display':display}
exec(compile(ast.fix_missing_locations(Redirect().visit(tree)),str(source),'exec'),ns)
report={'modality':mod,'source_script':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output':str(out),'before_shape':list(ns['X_before_combat' if mod=='DWI' else 'X_raw'].shape),'after_shape':list(ns['X_combat'].shape)}
(out/'rerun_provenance.json').write_text(json.dumps(report,indent=2));print('RERUN COMPLETE',report,flush=True)
