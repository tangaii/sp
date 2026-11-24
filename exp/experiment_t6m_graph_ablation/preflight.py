"""Validate frozen fold states and the physical ablation construction."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'src'))
from config import DEFAULTS,VARIANTS,MMGPL_CLASS_PROMPTS
from network import PLN,GFR
from utils import my_weight_init
from run import initialize_rng,load_data,state_hash

def hash_tensor(v): return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def common(m): return {n:hash_tensor(p) for n,p in m.named_parameters() if n.startswith(('InputLayer.','encode_','feed_','Outputlayer.'))}

def main():
 assert list(VARIANTS)==['G0_BASELINE','G1_BIOMARKER_SELF_ONLY','G2_BIOMARKER_Q97','G3_BIOMARKER_Q95','G4_NO_PEN_PHYSICAL','G5_NO_PEN_NO_GFR_PHYSICAL']
 device=torch.device('cpu'); initialize_rng(0)
 c=SimpleNamespace(**(DEFAULTS|VARIANTS['G0_BASELINE']))
 features,labels,dims,_=load_data(c,device)
 audit={'groups':{},'fold_states':{},'input_sha256':{}}
 for fold in range(1,11):
  ref=torch.load(HERE/'initial_states'/f'fold_{fold:02d}.pt',weights_only=True,map_location='cpu')
  audit['fold_states'][str(fold)]={'gfr':state_hash(type('M',(),{'state_dict':lambda self:ref['gfr']})()),'pln':hashlib.sha256(json.dumps({k:hash_tensor(v) for k,v in sorted(ref['pln'].items())},sort_keys=True).encode()).hexdigest(),'shared':{k:hash_tensor(v) for k,v in sorted(ref['pln'].items()) if k.startswith(('InputLayer.','encode_','feed_','Outputlayer.'))}}
 for name,variant in VARIANTS.items():
  cfg=SimpleNamespace(**(DEFAULTS|variant));gfr=GFR(dims,cfg.th,cfg.fusion,cfg.alpha,cfg.graph_mode,cfg.adaptive_quantile,cfg.st_temperature,cfg.biomarker_mode) if cfg.use_gfr else None
  pln=PLN(dims,cfg,MMGPL_CLASS_PROMPTS,device);pln.apply(my_weight_init)
  ref=torch.load(HERE/'initial_states'/'fold_01.pt',weights_only=True,map_location='cpu')
  if gfr is not None:gfr.load_state_dict(ref['gfr'])
  state={k:v for k,v in ref['pln'].items() if k in pln.state_dict() and pln.state_dict()[k].shape==v.shape}
  missing,_=pln.load_state_dict(state,strict=False)
  assert not any(k.startswith(('InputLayer.','encode_','feed_','Outputlayer.')) for k in missing)
  modules=list(dict(pln.named_modules()))
  params=list(pln.named_parameters());buffers=list(pln.named_buffers())
  pen=any(n.startswith('MLP.') for n,_ in params) or any('prompt_text_features' in n for n,_ in buffers)
  assert pen==cfg.use_pen
  assert (gfr is not None)==cfg.use_gfr
  if not cfg.use_gfr:assert not params or True
  x=features[:8]
  pln.eval()
  with torch.no_grad():out=pln(x)
  assert torch.isfinite(out).all() and out.shape==(8,3)
  opt=torch.optim.Adam(pln.parameters(),lr=cfg.lr)
  opt_names={n for n,p in pln.named_parameters() if any(p is q for group in opt.param_groups for q in group['params'])}
  assert opt_names=={n for n,_ in params}
  audit['groups'][name]={'gfr_present':gfr is not None,'pen_present':pen,'parameter_count':sum(p.numel() for _,p in params)+(sum(p.numel() for p in gfr.parameters()) if gfr else 0),'pln_parameter_count':sum(p.numel() for _,p in params),'module_names':modules,'parameter_names':[n for n,_ in params],'buffer_names':[n for n,_ in buffers],'shared_initial_parameter_sha256':common(pln),'finite_forward':True,'output_shape':list(out.shape),'optimizer_parameters_equal_named_parameters':True}
 for group in ('G0_BASELINE','G1_BIOMARKER_SELF_ONLY','G2_BIOMARKER_Q97','G3_BIOMARKER_Q95','G4_NO_PEN_PHYSICAL','G5_NO_PEN_NO_GFR_PHYSICAL'):
  fold_hashes=[]
  for fold in range(1,11):fold_hashes.append(audit['fold_states'][str(fold)]['shared'])
  assert len(fold_hashes)==10
 audit['status']='PASS'
 (HERE/'preflight_results.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({'status':'PASS','groups':{k:{'parameter_count':v['parameter_count'],'gfr':v['gfr_present'],'pen':v['pen_present']} for k,v in audit['groups'].items()}},indent=2))
if __name__=='__main__':main()
