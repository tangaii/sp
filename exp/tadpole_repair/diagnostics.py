"""Check prompt initialization, message fusion and graph supervision semantics."""

import json

import torch

from config import DEFAULTS, PUBLIC_PROMPTS, TADPOLE_PROMPTS
from prepare import HERE, sha256, write_json
from run import graph_supervision, initialize_rng
from network import CLIP_WEIGHTS, GFR, encode_prompts, prompt_features


def flatten(groups):
    return [prompt for group in groups.values() for prompt in group]


def tensor_stats(a, b):
    return {
        'shape': list(a.shape),
        'exactly_equal': torch.equal(a, b),
        'max_absolute_difference': (a - b).abs().max().item(),
        'mean_matching_prompt_cosine': torch.nn.functional.cosine_similarity(a, b).mean().item(),
    }


def check_clip(device):
    public_prompts = flatten(PUBLIC_PROMPTS)
    cached = prompt_features(public_prompts, device, 'frozen_pretrained')
    reused = prompt_features(public_prompts, device, 'frozen_pretrained')
    live = encode_prompts(public_prompts, device)
    xavier = prompt_features(public_prompts, device, 'xavier_linear')
    tadpole = prompt_features(flatten(TADPOLE_PROMPTS), device, 'frozen_pretrained')
    assert cached is reused
    assert torch.equal(cached, live) and not torch.equal(cached, xavier)
    assert torch.isfinite(xavier).all() and xavier.shape == (8, 1024)
    assert torch.isfinite(tadpole).all() and tadpole.shape == (12, 1024)
    assert not any(x.requires_grad for x in (cached, xavier, tadpole))
    return {
        'weights_sha256': sha256(CLIP_WEIGHTS),
        'frozen_cached_vs_frozen_live': tensor_stats(cached, live),
        'frozen_pretrained_vs_xavier': tensor_stats(cached, xavier),
        'tadpole_prompt_groups': list(TADPOLE_PROMPTS),
        'tadpole_feature_shape': list(tadpole.shape),
    }


def check_fusion(device):
    input_dims = [4, 4, 3, 2, 3, 2]
    raw = torch.randn(16, sum(input_dims), device=device)
    modalities = list(raw.split(input_dims, dim=1))
    eye = torch.eye(len(raw), device=device)
    gfr = GFR(input_dims, DEFAULTS['th'], 'legacy', DEFAULTS['alpha']).to(device)
    with torch.no_grad():
        first_layer = gfr.MessagePassing0.gcn2.W
        original = first_layer.clone()
        changes = {}
        for mode in ('legacy', 'sigmoid_sum', 'softmax_sum'):
            gfr.fusion = mode
            first_layer.copy_(original)
            original_output = gfr(raw, modalities, eye)[0]
            first_layer.zero_()
            altered_output = gfr(raw, modalities, eye)[0]
            changes[mode] = (original_output - altered_output).abs().max().item()
        assert changes['legacy'] == 0 and all(
            changes[mode] > 0 for mode in ('sigmoid_sum', 'softmax_sum')
        )
    return {'max_fused_change_after_zeroing_first_relation_message': changes}


def check_graph_supervision(device):
    train = torch.tensor([0, 1, 3], device=device)
    labels = torch.tensor([0, 0, 2, 1, 2], device=device)
    target = labels[train, None].eq(labels[train][None, :]).float()
    score = torch.eye(len(labels), device=device, requires_grad=True)
    pair_mean = graph_supervision([score] * 6, train, target, 'pair_mean')
    frobenius = graph_supervision([score] * 6, train, target, 'frobenius_squared')
    torch.testing.assert_close(pair_mean * target.numel(), frobenius)
    grad, = torch.autograd.grad(pair_mean, score)
    assert grad[0, 1] < 0 and grad[1, 0] < 0
    in_train = torch.zeros(len(labels), device=device, dtype=torch.bool)
    in_train[train] = True
    cross_or_test_pairs = ~(in_train[:, None] & in_train[None, :])
    assert torch.count_nonzero(grad[cross_or_test_pairs]) == 0
    return {
        'no_direct_supervision_gradient_for_pairs_with_outer_test_node': True,
        'below_threshold_same_class_pair_gradient': grad[0, 1].item(),
        'mean_to_frobenius_factor': target.numel(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_rng(DEFAULTS['seed'])
    summary = {
        'clip': check_clip(device),
        'fusion': check_fusion(device),
        'graph_supervision': check_graph_supervision(device),
    }
    write_json(HERE / 'diagnostics.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
