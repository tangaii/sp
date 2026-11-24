"""Fail-fast semantic audit for the CBSR supervision objective."""

import torch

from run import RELATIONS, graph_supervision, relation_masks


def main():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    masks = relation_masks(labels)
    assert tuple(masks) == RELATIONS
    assert all(not mask.diagonal().any() for mask in masks.values())
    assert sum(int(mask.sum()) for mask in masks.values()) == 30
    assert all(not (masks[a] & masks[b]).any() for i, a in enumerate(RELATIONS)
               for b in RELATIONS[i + 1:])
    for relation, mask in masks.items():
        assert bool(mask.any())
        expected = relation[0] == relation[1]
        assert expected == (relation in RELATIONS[:3])

    scores = torch.full((1, 6, 6), 0.2, requires_grad=True)
    with torch.no_grad():
        scores[0].fill_diagonal_(0.7)
    targets = labels[:, None].eq(labels[None, :]).float()
    pair_index = torch.arange(6)
    cb = graph_supervision([scores[0]], pair_index, targets, 'class_balanced', labels)
    manual = []
    for relation in RELATIONS:
        target = 1.0 if relation[0] == relation[1] else 0.0
        manual.append((scores[0][masks[relation]] - target).square().mean())
    expected_loss = 0.5 * (torch.stack(manual[:3]).mean() + torch.stack(manual[3:]).mean())
    assert torch.allclose(cb, expected_loss, atol=1e-7, rtol=0)

    pair_scores = torch.tensor(0.2, requires_grad=True)
    relation_results = {}
    for relation in RELATIONS:
        pair_scores.grad = None
        target = 1.0 if relation[0] == relation[1] else 0.0
        loss = (pair_scores - target).square()
        loss.backward()
        direction = -pair_scores.grad.item()
        relation_results[str(relation)] = {'target': target, 'gradient_descent_direction': direction}
        assert direction > 0 if target == 1 else direction < 0

    print({'status': 'PASS', 'relation_pair_counts': {str(k): int(v.sum()) for k, v in masks.items()},
           'cb_loss': cb.item(), 'gradient_directions': relation_results})


if __name__ == '__main__':
    main()
