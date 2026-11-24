"""Freeze foldwise G0 reference states before any formal training."""
import sys, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from config import DEFAULTS, VARIANTS, MMGPL_CLASS_PROMPTS
from network import GFR, PLN
from run import load_data, initialize_rng

cfg=SimpleNamespace(**(DEFAULTS|VARIANTS['G0_BASELINE']))
cfg.variant='G0_BASELINE'
device=torch.device('cuda')
initialize_rng(0)
features, labels, dims, info=load_data(cfg,device)
for fold in range(1,11):
    gfr=GFR(dims,cfg.th,cfg.fusion,cfg.alpha,cfg.graph_mode,cfg.adaptive_quantile,cfg.st_temperature,cfg.biomarker_mode).to(device)
    pln=PLN(dims,cfg,MMGPL_CLASS_PROMPTS,device).to(device)
    from utils import my_weight_init
    pln.apply(my_weight_init)
    state={'gfr':gfr.state_dict(),'pln':pln.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'rng_pre_model_cpu':None}
    torch.save(state,HERE/'initial_states'/f'fold_{fold:02d}.pt')
    del gfr,pln
    print(f'initialized fold {fold}/10',flush=True)
