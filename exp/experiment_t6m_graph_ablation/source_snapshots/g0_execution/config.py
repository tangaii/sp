"""Single immutable configuration for TADPOLE experiment 4."""
from pathlib import Path
CLASS_NAMES=('NC','MCI','AD')
DEFAULTS=dict(n_hidden=10,n_head=4,nlayer=2,dropout=0.0,attention_dropout=0.0,alpha=1.0,nmodal=6,nclass=3,lr=0.012,lambda_graph=1.0,eta_classification=1.0,epochs=300,folds=10,seed=0,split_seed=10,reg=0.0036,th=0.9,theta_smooth=1.0,theta_degree=0.5,theta_sparsity=0.0,graph_layers=2,optimizer='Adam',clip_init='frozen_pretrained',prompts='mmgpl_class_prompts',prompt_format='figure_6_concepts_with_class',pen='public',fusion='legacy',graph_supervision='pair_mean',graph_mode='fixed',adaptive_quantile=0.97,st_temperature=0.1,beta=0.0,biomarker_mode='normal',use_gfr=True,use_pen=True)
VARIANTS={
'G0_BASELINE':dict(),
'G1_BIOMARKER_SELF_ONLY':dict(biomarker_mode='self_only'),
'G2_BIOMARKER_Q97':dict(biomarker_mode='quantile',adaptive_quantile=0.97),
'G3_BIOMARKER_Q95':dict(biomarker_mode='quantile',adaptive_quantile=0.95),
'G4_NO_PEN_PHYSICAL':dict(use_pen=False,pen='none'),
'G5_NO_PEN_NO_GFR_PHYSICAL':dict(use_pen=False,use_gfr=False,pen='none',graph_supervision='none'),
}
assert len(VARIANTS)==6 and len(set(VARIANTS))==6
RELATIONS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
MMGPL_CONCEPTS={'NC':('Normal brain volume','Normal ventricle morphology','Symmetrical brain structure','Uniform cortical thickness'),'MCI':('Mild hippocampal atrophy','Mild brain volume reduction','Mild ventricular enlargement','Mild local cortical thinning'),'AD':('Pronounced hippocampal atrophy','Marked local cortical thinning','Significant brain volume reduction','Significant ventricular enlargement')}
MMGPL_CLASS_PROMPTS={k:tuple(f'{k}: {x}' for x in v) for k,v in MMGPL_CONCEPTS.items()}
MMGPL_CONCEPTS=MMGPL_CONCEPTS
PUBLIC_PROMPTS=MMGPL_CLASS_PROMPTS
TADPOLE_PROMPTS=MMGPL_CLASS_PROMPTS
MMGPL_PAPER='inputs/2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf'
PROVENANCE={'historical_exact_status':'under investigation','evaluation':'val=test outer-fold first maximal accuracy','rfe':'completed before outer CV','clip_sha256':'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'}
