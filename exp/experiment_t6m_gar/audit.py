import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
ROOT=Path(__file__).resolve().parent
GROUPS=['G0_BASELINE','G1_MODAL_ADAPTIVE_GRAPH','G2_GRAPH_GRADIENT_ALIGNMENT','G3_CLASS_BALANCED_GRAPH','G4_COMBINED_GAR']
CLASSES=['NC','MCI','AD']
def metrics(df):
 y=df.label.to_numpy(); p=df.prediction.to_numpy(); prob=df[[f'p_{x}' for x in CLASSES]].to_numpy(); one=np.eye(3)[y]; hard=np.eye(3)[p]; cm=confusion_matrix(y,p,labels=[0,1,2]); tp=np.diag(cm); sup=cm.sum(1); fp=cm.sum(0)-tp; neg=cm.sum()-sup
 return {'acc':float((y==p).mean()),'auc_hard':float(roc_auc_score(one,hard)), 'auc_probability':float(roc_auc_score(one,prob)), 'sensitivity':dict(zip(CLASSES,(tp/sup).tolist())), 'specificity':dict(zip(CLASSES,(1-fp/neg).tolist())), 'confusion':cm.tolist(),'correct':int((y==p).sum()),'mci_errors':int(((y==1)&(p!=1)).sum())}
out={}
for g in GROUPS:
 d=ROOT/'results'/g/'out_of_fold_predictions.csv'; df=pd.read_csv(d).sort_values('row_index')
 if len(df)!=598 or df.row_index.nunique()!=598: raise RuntimeError(g+' invalid OOF')
 fold=[]
 for f in range(1,11):
  x=pd.read_csv(ROOT/'results'/g/f'fold_{f:02d}'/'predictions.csv'); fold.append(metrics(x)|{'fold':f})
 overall=json.loads((ROOT/'results'/g/'summary.json').read_text())['overall']
 recomputed=metrics(df)
 graph=[]
 for f in range(1,11):
  m=json.loads((ROOT/'results'/g/f'fold_{f:02d}'/'metrics.json').read_text()); graph += [{'fold':f,**r} for r in m['graph_relations']]
 out[g]={'overall':overall,'oof':recomputed,'folds':fold,'graph_relations':graph,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results'/g).glob('fold_*/metrics.json')}}
(ROOT/'audit').mkdir(exist_ok=True)
(ROOT/'audit'/'audit.json').write_text(json.dumps(out,indent=2))
rows=[]
for g in GROUPS:
 o=out[g]['overall']; rows.append({'method':g,'ACC':f"{100*o['acc']['mean']:.2f} ± {100*o['acc']['std']:.2f}",'Hard-AUC':f"{100*o['auc_hard']['mean']:.2f} ± {100*o['auc_hard']['std']:.2f}",'Prob-AUC':f"{100*o['auc_probability']['mean']:.2f} ± {100*o['auc_probability']['std']:.2f}",'NC SEN':f"{100*o['sensitivity']['NC']['mean']:.2f} ± {100*o['sensitivity']['NC']['std']:.2f}",'MCI SEN':f"{100*o['sensitivity']['MCI']['mean']:.2f} ± {100*o['sensitivity']['MCI']['std']:.2f}",'AD SEN':f"{100*o['sensitivity']['AD']['mean']:.2f} ± {100*o['sensitivity']['AD']['std']:.2f}",'correct_oof':out[g]['oof']['correct'],'mci_errors_oof':out[g]['oof']['mci_errors']})
pd.DataFrame(rows).to_csv(ROOT/'summary.csv',index=False)
