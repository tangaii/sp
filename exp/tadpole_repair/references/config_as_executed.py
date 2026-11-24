"""Presets fixed before evaluation of the TADPOLE repairs."""

PAPER = 'https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf'
RELEASE = 'https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a'
AUTHOR_LOG = RELEASE.replace('/tree/', '/blob/') + '/result_Tadpole.txt'
CLASS_NAMES = ('NC', 'MCI', 'AD')

PUBLIC_PROMPTS = {
    'group_0': (
        'Nromal brain volume',
        'Uniform cortical thickness',
        'Symmetrical brain structure',
        'Normal ventricular morphology',
    ),
    'group_1': (
        'Altered brain volume',
        'Abnormal functional connectivity',
        'White matter integrity issues',
        'Different activation patterns',
    ),
}

TADPOLE_PROMPTS = {
    'NC': (
        'Preserved hippocampal volume in a cognitively normal older adult',
        'Normal memory and cognition for age',
        'Preserved medial temporal brain structure',
        'Preserved daily function with normal cognitive testing',
    ),
    'MCI': (
        'Mild cognitive impairment with subtle hippocampal volume loss',
        'Mild episodic memory impairment',
        'Subtle medial temporal atrophy',
        'Mild cognitive decline with mostly preserved daily function',
    ),
    'AD': (
        'Significant hippocampal atrophy: AD subjects often show significant atrophy of the hippocampal region',
        'Alzheimer disease with impaired episodic memory',
        'Medial temporal cortical atrophy in Alzheimer disease',
        'Distinct cognitive decline with impaired daily function in Alzheimer disease',
    ),
}

DEFAULTS = {
    'n_hidden': 10, 'n_head': 4, 'nlayer': 2, 'dropout': 0.0, 'attention_dropout': 0.0,
    'alpha': 1.0, 'nmodal': 6, 'nclass': 3, 'lr': 0.012,
    'lambda_graph': 1.0, 'eta_classification': 1.0,
    'epochs': 300, 'folds': 10, 'seed': 0, 'split_seed': 10,
    'reg': 0.0036, 'th': 0.9, 'theta_smooth': 1.0, 'theta_degree': 0.5,
    'theta_sparsity': 0.0, 'graph_layers': 2, 'optimizer': 'Adam',
    'data': 'deterministic', 'clip_init': 'frozen_pretrained', 'prompts': 'public',
    'pen': 'public', 'fusion': 'legacy', 'graph_supervision': 'none',
}

VARIANTS = {
    'previous_control': {'data': 'previous', 'attention_dropout': 0.1},
    'paper_config': {},
    'clip_xavier': {'clip_init': 'xavier_linear'},
    'tadpole_prompts': {'prompts': 'tadpole'},
    'pen_equations': {'prompts': 'tadpole', 'pen': 'equations_5_6'},
    'gfr_sum': {'fusion': 'sigmoid_sum'},
    'gfr_softmax': {'fusion': 'softmax_sum'},
    'graph_supervision': {'graph_supervision': 'pair_mean'},
    'graph_frobenius': {'graph_supervision': 'frobenius_squared'},
    'combined': {'prompts': 'tadpole', 'pen': 'equations_5_6',
                 'fusion': 'sigmoid_sum', 'graph_supervision': 'pair_mean'},
}

PROVENANCE = {
    'paper': PAPER,
    'release': RELEASE,
    'author_log': AUTHOR_LOG,
    'paper_section_3_1': {'dropout': 0.0, 'lr': 0.012, 'hidden_value': 10,
                         'encoding_layers': 2, 'heads': 4, 'alpha': 1.0, 'folds': 10},
    'paper_equation_7': {'lambda_graph': 1.0, 'eta_classification': 1.0},
    'interpretation_of_hidden_value': 'per-head width following released argument semantics',
    'released_main_defaults': [
        'reg', 'th', 'theta_smooth', 'theta_degree', 'theta_sparsity', 'seed', 'split_seed',
    ],
    'epoch_budget_source': '300 epochs in each of ten folds in result_Tadpole.txt',
    'unpublished': ['author TADPOLE command line', 'exact prompt list',
                    'full per-subject split manifest'],
    'class_weight_rule': 'inverse training frequency in original Counter insertion order (legacy)',
    'evaluation': 'full-cohort transductive graph and first maximal outer-test-fold ACC',
    'graph_supervision_target': 'binary label equality on train-train pairs only',
    'graph_supervision_scores': 'pre-threshold cosine matrix; threshold retained for propagation and LS',
}
