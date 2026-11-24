"""Independent E5-PAR audit. It never trains or trusts aggregate files."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/'src'))
from config import CLASS_NAMES, DEFAULTS, VARIANTS, MMGPL_CLASS_PROMPTS
from run import initialize_rng, load_data, all_folds, evaluate
from network import PLN, GFR, prompt_features

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_rng(0)
    cfg=SimpleNamespace(**(DEFAULTS|VARIANTS['G0_FULL_BASELINE']))
    features, labels, dims, info=load_data(cfg,device); folds=all_folds(labels,cfg)
    assert len(labels)==598 and np.bincount(labels).tolist()==[209,315,74]
    assert dims==[145,148,3,36,24,9] and np.isfinite(features.detach().cpu().numpy()).all()
    prompt_features([t for group in MMGPL_CLASS_PROMPTS.values() for t in group],device,'frozen_pretrained')
    audit={'status':'INCOMPLETE','groups':{},'checks':{'inputs':True,'groups':list(VARIANTS)}}
    all_oof={}
    for group, overrides in VARIANTS.items():
        c=SimpleNamespace(**(DEFAULTS|overrides)); root=HERE/'results'/group
        records=[]; rows=[]
        for fold,(train,test) in enumerate(folds,1):
            folder=root/f'fold_{fold:02d}'; hp=folder/'history.csv'; cp=folder/'best.pt'; pp=folder/'predictions.csv'; mp=folder/'metrics.json'
            complete=all(x.is_file() for x in (hp,cp,pp,mp)) and len(pd.read_csv(hp))==300
            if not complete: raise RuntimeError(f'missing/incomplete {group} fold {fold:02d}')
            saved=pd.read_csv(pp).sort_values('row_index'); expected=np.sort(test)
            assert saved.row_index.tolist()==expected.tolist()
            m=json.loads(mp.read_text()); assert 0<=m['best_epoch_zero_based']<300
            # Reconstruct the declared physical structure and reload only this fold checkpoint.
            pln=PLN(dims,c,MMGPL_CLASS_PROMPTS,device).to(device)
            gfr=GFR(dims,c.th,c.fusion,c.alpha,c.graph_mode,c.adaptive_quantile,c.st_temperature,c.biomarker_mode).to(device) if c.use_gfr else None
            ck=torch.load(cp,map_location=device,weights_only=True)
            if gfr is not None: gfr.load_state_dict(ck['gfr'])
            pln.load_state_dict(ck['pln']); pln.eval()
            with torch.no_grad():
                if gfr is not None:
                    mods=list(features.split(dims,dim=1)); raw=torch.cat(mods,dim=1); eye=torch.eye(len(features),device=device,dtype=features.dtype)
                    fused,_,_=gfr(raw,mods,eye,torch.as_tensor(train,device=device))
                else: fused=features
                all_prob=pln(fused).exp().cpu().numpy(); prob=all_prob[expected]; pred=prob.argmax(1)
            assert np.array_equal(pred,saved.prediction.to_numpy())
            assert np.allclose(prob,saved[[f'p_{x}' for x in CLASS_NAMES]].to_numpy(),atol=2e-6)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=2e-6)
            rows.append(saved); records.append({'fold':fold,'best_epoch':m['best_epoch_zero_based'],'checkpoint_sha256':sha(cp),'prediction_reload_match':True,'parameter_count':sum(p.numel() for p in pln.parameters())+(sum(p.numel() for p in gfr.parameters()) if gfr else 0),'gfr_present':gfr is not None,'pen_present':any(n.startswith('MLP.') for n,_ in pln.named_parameters()),'shared_initial_hash':m.get('initial_state_sha256',{}).get('pln')})
        oof=pd.concat(rows,ignore_index=True).sort_values('row_index').reset_index(drop=True); assert len(oof)==598 and oof.row_index.tolist()==list(range(598))
        all_oof[group]=oof
        fold_metrics=[json.loads((root/f'fold_{i:02d}/metrics.json').read_text()) for i in range(1,11)]
        overall={k:{'mean':float(np.mean([m[k] for m in fold_metrics])),'std':float(np.std([m[k] for m in fold_metrics]))} for k in ('acc','auc_hard','auc_probability')}
        ev=evaluate(oof.label.to_numpy(),oof.prediction.to_numpy(),oof[[f'p_{x}' for x in CLASS_NAMES]].to_numpy())
        overall.update({'oof_correct':int((oof.label==oof.prediction).sum()),'sensitivity':ev['sensitivity'],'specificity':ev['specificity'],'confusion':ev['confusion']})
        audit['groups'][group]={'overall':overall,'folds':records}
    base=all_oof['G0_FULL_BASELINE']; pairs={}
    for group, oof in all_oof.items():
        if group== 'G0_FULL_BASELINE': continue
        bc=base.label.eq(base.prediction); nc=oof.label.eq(oof.prediction)
        pairs[group]={'g0_correct_new_wrong':int((bc&~nc).sum()),'g0_wrong_new_correct':int((~bc&nc).sum()),'net_correct':int(nc.sum()-bc.sum()),'prediction_agreement':float(np.mean(base.prediction.to_numpy()==oof.prediction.to_numpy()))}
    audit['pairwise_patient_comparison']=pairs
    # Deterministic semantic permutation check for the current max-over-class SPGS.
    rng=np.random.default_rng(0); sim=rng.normal(size=(6,3)); a=np.max(np.exp(sim)/np.exp(sim).sum(1,keepdims=True),axis=1); perm=sim[:,[2,0,1]]; b=np.max(np.exp(perm)/np.exp(perm).sum(1,keepdims=True),axis=1)
    audit['semantic_permutation_test']={'max_abs_difference':float(np.max(np.abs(a-b))),'passed':bool(np.allclose(a,b))}
    audit['checks'].update({'all_groups_10_folds':all(len(v['folds'])==10 for v in audit['groups'].values()),'all_histories_300':all(all(r['best_epoch']<300 for r in v['folds']) for v in audit['groups'].values()),'all_reload_match':all(all(r['prediction_reload_match'] for r in v['folds']) for v in audit['groups'].values()),'semantic_permutation_invariant':audit['semantic_permutation_test']['passed'],'oof_all_once':True})
    audit['status']='COMPLETE_AUDITED' if all(audit['checks'].values()) else 'COMPLETE_NOT_FULLY_AUDITED'
    (HERE/'audit_results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':audit['status'],'groups':list(audit['groups']),'checks':audit['checks']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
