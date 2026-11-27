"""Read-only gradient-path audit for the sealed DARG Best checkpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner
from source_runtime import TADPOLE_MODALITIES

HISTORICAL = {
    "H0": (Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"), "R0"),
    "H1": (Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"), "S02"),
}
GROUPS = ("H0", "H1", "T1", "T2", "T3", "T4")


def norm_and_finite(values):
    present = [value.detach() for value in values if value is not None]
    if not present:
        return 0.0, True
    finite = all(bool(torch.isfinite(value).all()) for value in present)
    norm = math.sqrt(sum(float(value.float().square().sum()) for value in present))
    return norm, finite


def source_checkpoint(group: str, fold: int) -> Path:
    if group in HISTORICAL:
        root, name = HISTORICAL[group]
        return root / "groups" / name / "best_checkpoints" / f"fold{fold:02d}.pt"
    return ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_np, y_np, _, splits_raw = runner.load_data()
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_np, dtype=torch.long, device=device)
    splits = {int(row["fold"]): row for row in splits_raw}
    rows = []

    for group in GROUPS:
        config = runner.group_config(group)
        for fold in range(1, 11):
            payload = torch.load(source_checkpoint(group, fold), map_location="cpu", weights_only=False)
            model = runner.build_model(config, fold, device, spectral_stats=payload["spectral_stats"])
            model.load_state_dict(payload["model"], strict=True)
            model.darg.enter_positive_dcr()
            model.eval()
            train = torch.as_tensor(splits[fold]["train_idx"], dtype=torch.long, device=device)
            pair = payload["relation_pairs"].to(device)
            same = payload["relation_targets"].to(device)
            with torch.enable_grad():
                _, aux = model(x, labels, train, pair, same, dpsl_enabled=False)
                core = model.darg.core
                configured_anchor = model.darg.cadr.alignment_anchor_index
                anchor_index = -1 if configured_anchor is None else int(configured_anchor)
                encoder_params = [list(encoder.parameters()) for encoder in core.encoders]
                scorer_params = [list(scorer.parameters()) for scorer in core.scorers]
                discriminator_params = list(core.discriminator.parameters())
                pair_parameters = [parameter for module in encoder_params + scorer_params for parameter in module]
                pair_gradients = torch.autograd.grad(aux["pair_loss"], pair_parameters,
                                                    retain_graph=True, allow_unused=True)
                pair_encoder_norms, pair_scorer_norms = [], []
                offset = 0
                for modules, output in ((encoder_params, pair_encoder_norms),
                                        (scorer_params, pair_scorer_norms)):
                    for params in modules:
                        count = len(params)
                        norm, finite = norm_and_finite(pair_gradients[offset:offset + count])
                        output.append({"norm": norm, "finite": finite})
                        offset += count

                pair_reprs = aux["pair_reprs"]
                condition = F.one_hot(same.long(), num_classes=2).to(pair_reprs[0].dtype)
                adv_logits, adv_targets = [], []
                direct_logits, direct_targets = [], []
                direct_repr_gradients = []
                for modality, relation in enumerate(pair_reprs):
                    target = torch.full((len(relation),), modality, dtype=torch.long, device=device)
                    adv_relation = relation.detach() if modality == anchor_index else model.darg.cadr.grad_reverse(
                        relation, core.grl_scale)
                    adv_logits.append(core.discriminator(adv_relation, condition))
                    direct_logits.append(core.discriminator(relation, condition))
                    adv_targets.append(target)
                    direct_targets.append(target)
                adv_loss = F.cross_entropy(torch.cat(adv_logits), torch.cat(adv_targets))
                direct_loss = F.cross_entropy(torch.cat(direct_logits), torch.cat(direct_targets))
                adv_repr_grads = torch.autograd.grad(adv_loss, pair_reprs, retain_graph=True, allow_unused=True)
                direct_repr_grads = torch.autograd.grad(direct_loss, pair_reprs, retain_graph=True, allow_unused=True)
                grad_scale_ratios = []
                for modality, (adv_grad, direct_grad) in enumerate(zip(adv_repr_grads, direct_repr_grads)):
                    adv_norm, adv_finite = norm_and_finite([adv_grad])
                    direct_norm, direct_finite = norm_and_finite([direct_grad])
                    ratio = (adv_norm / direct_norm) if direct_norm else None
                    expected = None if modality == anchor_index else 0.03
                    passed = (adv_finite and direct_finite and
                              (abs(ratio - expected) < 1e-6 if expected is not None
                               else adv_grad is None or adv_norm == 0.0))
                    grad_scale_ratios.append({"modality_index": modality,
                                              "modality": TADPOLE_MODALITIES[modality],
                                              "adversarial_repr_gradient_norm": adv_norm,
                                              "direct_repr_gradient_norm": direct_norm,
                                              "ratio": ratio, "expected_ratio": expected,
                                              "finite": adv_finite and direct_finite,
                                              "pass": bool(passed)})

                adversarial_parameters = [parameter for modules in encoder_params for parameter in modules]
                adversarial_parameters.extend(discriminator_params)
                adversarial_gradients = torch.autograd.grad(aux["adv_loss"], adversarial_parameters,
                                                            allow_unused=True)
                encoder_adv_norms, discriminator_gradients = [], []
                offset = 0
                for params in encoder_params:
                    count = len(params)
                    norm, finite = norm_and_finite(adversarial_gradients[offset:offset + count])
                    encoder_adv_norms.append({"norm": norm, "finite": finite})
                    offset += count
                discriminator_gradients = adversarial_gradients[offset:]
                discriminator_adv_norm, discriminator_adv_finite = norm_and_finite(discriminator_gradients)

            pair_ok = all(row["finite"] and row["norm"] > 0.0
                          for row in pair_encoder_norms + pair_scorer_norms)
            detached_anchor_ok = (anchor_index < 0 or
                                  (encoder_adv_norms[anchor_index]["finite"]
                                   and encoder_adv_norms[anchor_index]["norm"] == 0.0))
            grl_ok = (all(row["finite"] and row["norm"] > 0.0
                          for index, row in enumerate(encoder_adv_norms)
                          if index != anchor_index)
                      and detached_anchor_ok
                      and discriminator_adv_finite and discriminator_adv_norm > 0.0
                      and all(row["pass"] for row in grad_scale_ratios))
            rows.append({"group": group, "fold": fold, "pair_loss": float(aux["pair_loss"].detach()),
                         "adv_loss": float(aux["adv_loss"].detach()),
                         "relation_encoder_pair_bce_gradients": pair_encoder_norms,
                         "relation_scorer_pair_bce_gradients": pair_scorer_norms,
                         "alignment_anchor_index": (None if anchor_index < 0 else anchor_index),
                         "encoder_grl_gradients": [
                             {"modality": TADPOLE_MODALITIES[index], **row}
                             for index, row in enumerate(encoder_adv_norms)],
                         "discriminator_adversarial_gradient_norm": discriminator_adv_norm,
                         "discriminator_adversarial_gradient_finite": discriminator_adv_finite,
                         "grl_repr_gradient_scale": grad_scale_ratios,
                         "pair_bce_path_pass": pair_ok, "grl_path_pass": grl_ok,
                         "status": "PASS" if pair_ok and grl_ok else "FAIL"})
            del model, payload, aux
            if device.type == "cuda":
                torch.cuda.empty_cache()

    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS" if passed == len(rows) == 60 else "FAIL",
        "device": str(device), "folds": len(rows), "folds_passed": passed,
        "darg_modalities": list(TADPOLE_MODALITIES),
        "grl_scale": 0.03,
        "checks": {
            "pair_bce_nonzero_finite_gradients_to_all_relation_encoders_and_scorers": True,
            "grl_nonzero_finite_gradients_to_non_anchor_encoders": True,
            "configured_alignment_anchor_detached_when_present": True,
            "grl_repr_gradient_ratio_is_0.03": True,
            "adversarial_discriminator_gradient_nonzero_finite": True,
        },
        "fold_results": rows,
    }
    target = ROOT / "audits/gradient_replay.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "folds": len(rows), "folds_passed": passed,
                      "path": str(target)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
