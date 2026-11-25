"""Frozen E6-SGCI configuration for six-modality TADPOLE."""
from pathlib import Path

CLASS_NAMES = ('NC', 'MCI', 'AD')
DEFAULTS = dict(
    n_hidden=10, n_head=4, nlayer=2, dropout=0.0, attention_dropout=0.0,
    alpha=1.0, nmodal=6, nclass=3, lr=0.012, weight_decay=0.0036,
    reg=0.0036, lambda_graph=1.0, eta_classification=1.0, epochs=300,
    folds=10, seed=0, split_seed=10, th=0.9, theta_smooth=1.0,
    theta_degree=0.5, theta_sparsity=0.0, graph_layers=2, optimizer='Adam',
    clip_init='frozen_pretrained', prompts='mmgpl_class_prompts',
    prompt_format='figure_6_concepts_with_class', pen='public', fusion='legacy',
    graph_supervision='pair_mean', graph_mode='fixed', adaptive_quantile=0.97,
    st_temperature=0.1, beta=0.0, biomarker_mode='normal', use_gfr=True,
    use_pen=True, semantic_null=False, unweighted_classification=False,
    calibration_ratio=None, calibration_eps=1e-12,
)

VARIANTS = {
    'G0_FULL_LEGACY': dict(pen='public'),
    'G1_RES_ONLY_LEGACY': dict(pen='res_only'),
    'G2_RES_ONLY_R04': dict(pen='res_only', calibration_ratio=0.04),
    'G3_REAL_SPGS_R04': dict(pen='public', calibration_ratio=0.04),
    'G4_NULL_SPGS_R04': dict(pen='public', semantic_null=True, calibration_ratio=0.04),
    'G5_RES_ONLY_R20': dict(pen='res_only', calibration_ratio=0.20),
    'G6_REAL_SPGS_R20': dict(pen='public', calibration_ratio=0.20),
    'G7_NULL_SPGS_R20': dict(pen='public', semantic_null=True, calibration_ratio=0.20),
}

RELATIONS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
MMGPL_CONCEPTS = {
    'NC': ('Normal brain volume', 'Normal ventricle morphology', 'Symmetrical brain structure', 'Uniform cortical thickness'),
    'MCI': ('Mild hippocampal atrophy', 'Mild brain volume reduction', 'Mild ventricular enlargement', 'Mild local cortical thinning'),
    'AD': ('Pronounced hippocampal atrophy', 'Marked local cortical thinning', 'Significant brain volume reduction', 'Significant ventricular enlargement'),
}
MMGPL_CLASS_PROMPTS = {key: tuple(f'{key}: {value}' for value in values) for key, values in MMGPL_CONCEPTS.items()}
PUBLIC_PROMPTS = MMGPL_CLASS_PROMPTS
TADPOLE_PROMPTS = MMGPL_CLASS_PROMPTS
MMGPL_PAPER = Path('inputs/2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf')
PROVENANCE = {
    'experiment': 'E6-SGCI Semantic Guidance Causal Isolation',
    'historical_exact_status': 'requires G0 verification against E5',
    'evaluation': 'outer-fold test accuracy, first epoch attaining maximum',
    'rfe': 'completed before outer CV and frozen',
    'legacy_gfr': 'preserved, including relation 0 overwrite behavior',
    'res_only': 'released-code-compatible legacy_recalibrate with explicit unit part strength',
    'calibration': 'unlabelled full-240-vector delta norm ratio, detached scale',
}
