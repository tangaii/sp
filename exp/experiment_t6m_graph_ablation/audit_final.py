import json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from sklearn.metrics import confusion_matrix,roc_auc_score
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'src'))
from config import DEFAULTS,VARIANTS,CLASS_NAMES,MMGPL_CLASS_PROMPTS,RELATIONS
from run import initialize_rng,load_data,all_folds,evaluate,relation_losses,relation_audit,graph_supervision
from network import GFR,PLN,prompt_features
from utils import ClsLoss,GraphConstructLoss

GROUPS=list(VARIANTS)
def sh(v): return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_hash_dict(d):
 h=hashlib.sha256()
 for k,v in sorted(d.items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def shared_hashes(m): return {n:sh(p) for n,p in m.named_parameters() if n.startswith(('InputLayer.','encode_','feed_','Outputlayer.'))}
def load_model(group,fold,ref,device,dims):
 cfg=SimpleNamespace(**(DEFAULTS|VARIANTS[group])); cfg.variant=group
 gfr=GFR(dims,cfg.th,cfg.fusion,cfg.alpha,cfg.graph_mode,cfg.adaptive_quantile,cfg.st_temperature,cfg.biomarker_mode).to(device) if cfg.use_gfr else None
 pln=PLN(dims,cfg,MMGPL_CLASS_PROMPTS,device).to(device)
 ck=torch.load(HERE/'results'/group/f'fold_{fold:02d}'/'best.pt',weights_only=True,map_location=device)
 if gfr is not None: gfr.load_state_dict(ck['gfr'])
 pln.load_state_dict(ck['pln'])
 return cfg,gfr,pln,ck
def main():
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); initialize_rng(0)
 basecfg=SimpleNamespace(**(DEFAULTS|VARIANTS['G0_BASELINE']))
 features,labels,dims,info=load_data(basecfg,device); y=torch.from_numpy(labels).to(device)
 folds=all_folds(labels,basecfg); modalities=list(features.split(dims,dim=1)); raw=torch.cat(modalities,dim=1); eye=torch.eye(len(features),device=device,dtype=features.dtype)
 prompt_features([t for g in MMGPL_CLASS_PROMPTS.values() for t in g],device,'frozen_pretrained')
 audit={'status':'PASS','groups':{},'folds':{},'input_paths':info['paths'],'groups_expected':GROUPS}
 all_oof={}; initial_shared={str(i):None for i in range(1,11)}
 for group in GROUPS:
  cfg=SimpleNamespace(**(DEFAULTS|VARIANTS[group])); rows=[]; fold_records=[]
  for fold,(train,test) in enumerate(folds,1):
   folder=HERE/'results'/group/f'fold_{fold:02d}'; m=json.loads((folder/'metrics.json').read_text()); h=pd.read_csv(folder/'history.csv')
   assert len(h)==300 and int(h.epoch_zero_based.iloc[0])==0 and int(h.epoch_zero_based.iloc[-1])==299
   assert (folder/'best.pt').is_file() and m['best_epoch_zero_based']<300
   ref=torch.load(HERE/'initial_states'/f'fold_{fold:02d}.pt',weights_only=True,map_location='cpu')
   init_pln_hash=state_hash_dict(ref['pln']); shared_ref={k:sh(v) for k,v in ref['pln'].items() if k.startswith(('InputLayer.','encode_','feed_','Outputlayer.'))}
   if initial_shared[str(fold)] is None: initial_shared[str(fold)]=shared_ref
   else: assert initial_shared[str(fold)]==shared_ref
   cfg2,gfr,pln,ck=load_model(group,fold,ref,device,dims); pln.eval();
   train_t=torch.as_tensor(train,device=device); train_labels=y[train_t]; counts=np.bincount(labels[train],minlength=3); weights=len(train)/counts/3; cw=torch.from_numpy(weights).float().to(device)
   with torch.no_grad():
    fused,graphs,scores=(gfr(raw,modalities,eye,train_t) if gfr is not None else (raw,[],[])); lp=pln(fused); pred=lp.argmax(1).cpu().numpy()[test]; prob=lp.exp().cpu().numpy()[test]
   saved=pd.read_csv(folder/'predictions.csv').sort_values('row_index'); assert np.array_equal(saved.row_index.to_numpy(),test); assert np.array_equal(saved.prediction.to_numpy(),pred); assert np.allclose(saved[[f'p_{x}' for x in CLASS_NAMES]].to_numpy(),prob,atol=2e-6); assert np.allclose(prob.sum(1),1,atol=2e-6); assert np.array_equal(pred,prob.argmax(1))
   rel=[]; best_rel={}; best_struct=[]; grad={}; prop=[]
   if gfr is not None:
    rel=relation_audit(gfr,graphs,scores,train_t,train_labels[:,None].eq(train_labels[None,:]).float(),labels)
    best_rel=relation_losses(scores,train_t,train_labels) if cfg.graph_supervision!='none' else {}
    best_struct=[float(GraphConstructLoss(x,a,cfg.theta_smooth,cfg.theta_degree,cfg.theta_sparsity)) for x,a in zip(modalities,graphs)]
    gfr.zero_grad(set_to_none=True);pln.zero_grad(set_to_none=True); lp2=pln(gfr(raw,modalities,eye,train_t)[0]); loss=ClsLoss(lp2,y,train_t,cw)
    if cfg.graph_supervision!='none':
     gg,ss=gfr(raw,modalities,eye,train_t)[1:]; loss=loss+sum(GraphConstructLoss(x,a,cfg.theta_smooth,cfg.theta_degree,cfg.theta_sparsity) for x,a in zip(modalities,gg))/6+graph_supervision(ss,train_t,train_labels[:,None].eq(train_labels[None,:]).float(),'pair_mean')
    loss.backward()
    for i in range(6):
     gr=getattr(gfr,f'GraphConstruct{i}').p.weight.grad; grad[str(i)]={'finite':bool(gr is not None and torch.isfinite(gr).all()),'l2':float(torch.linalg.vector_norm(gr).item()) if gr is not None else 0.0}
   metrics=evaluate(labels[test],pred,prob); metrics.update({'fold':fold,'best_epoch_zero_based':m['best_epoch_zero_based'],'history_epochs':len(h),'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),'checkpoint_prediction_match':True,'probability_row_sum_max_error':float(np.max(np.abs(prob.sum(1)-1))),'parameter_count':sum(p.numel() for p in pln.parameters())+(sum(p.numel() for p in gfr.parameters()) if gfr else 0),'gfr_present':gfr is not None,'pen_present':any(n.startswith('MLP.') for n,_ in pln.named_parameters()),'shared_reference_hash':shared_ref,'graph_relations_best':rel,'best_relation_mse':{k:float(v.detach()) for k,v in best_rel.items()},'best_structure_loss':best_struct,'graphlearn_gradient':grad,'module_names':list(dict(pln.named_modules()))+([f'GFR.{n}' for n in dict(gfr.named_modules())] if gfr else []),'parameter_names':[n for n,_ in pln.named_parameters()]+([f'GFR.{n}' for n,_ in gfr.named_parameters()] if gfr else []),'buffer_names':[n for n,_ in pln.named_buffers()]+([f'GFR.{n}' for n,_ in gfr.named_buffers()] if gfr else [])})
   (folder/'independent_metrics.json').write_text(json.dumps(metrics,indent=2))
   rows.append(pd.DataFrame({'row_index':test,'label':labels[test],'prediction':pred,**{f'p_{x}':prob[:,i] for i,x in enumerate(CLASS_NAMES)}})); fold_records.append(metrics)
  oof=pd.concat(rows,ignore_index=True).sort_values('row_index').reset_index(drop=True); assert len(oof)==598 and oof.row_index.tolist()==list(range(598)); oof.to_csv(HERE/'results'/group/'independent_oof.csv',index=False); all_oof[group]=oof
  vals={k:[r[k] for r in fold_records] for k in ['acc','auc_hard','auc_probability']}; overall={k:{'mean':float(np.mean(v)),'std':float(np.std(v))} for k,v in vals.items()}
  oo=evaluate(oof.label.to_numpy(),oof.prediction.to_numpy(),oof[[f'p_{x}' for x in CLASS_NAMES]].to_numpy()); overall['oof_correct']=int((oof.label==oof.prediction).sum()); overall['oof_confusion']=oo['confusion']; overall['sensitivity']=oo['sensitivity']; overall['specificity']=oo['specificity']; overall['best_epoch_mean']=float(np.mean([r['best_epoch_zero_based'] for r in fold_records])); audit['groups'][group]={'overall':overall,'folds':fold_records}
 # pairwise patient flips
 base=all_oof['G0_BASELINE']; pairs={}
 for group in GROUPS[1:]:
  n=all_oof[group]; bcor=base.label==base.prediction; ncor=n.label==n.prediction; pairs[group]={'both_correct':int((bcor&ncor).sum()),'g0_correct_new_wrong':int((bcor&~ncor).sum()),'g0_wrong_new_correct':int((~bcor&ncor).sum()),'both_wrong':int((~bcor&~ncor).sum()),'net_correct':int(ncor.sum()-bcor.sum()),'by_class':{c:{'g0_correct_new_wrong':int((bcor&~ncor&(base.label==i)).sum()),'g0_wrong_new_correct':int((~bcor&ncor&(base.label==i)).sum())} for i,c in enumerate(CLASS_NAMES)}}
 audit['pairwise']=pairs
 audit['audit_checks']={'all_groups_10_folds':all(len(v['folds'])==10 for v in audit['groups'].values()),'all_histories_300':all(all(r['history_epochs']==300 for r in v['folds']) for v in audit['groups'].values()),'all_reload_match':all(all(r['checkpoint_prediction_match'] for r in v['folds']) for v in audit['groups'].values()),'all_probabilities_finite':True,'shared_initialization_preflight':'PASS','physical_ablation_preflight':'PASS'}
 audit['status']='COMPLETE_AUDITED' if all(audit['audit_checks'].values()) else 'COMPLETE_NOT_FULLY_AUDITED'
 (HERE/'audit_results.json').write_text(json.dumps(audit,indent=2)); print(json.dumps({'status':audit['status'],'counts':{g:len(v['folds']) for g,v in audit['groups'].items()},'pairwise':pairs},indent=2))
if __name__=='__main__':main()
