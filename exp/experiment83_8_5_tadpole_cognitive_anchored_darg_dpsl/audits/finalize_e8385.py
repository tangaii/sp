"""Assemble the formal report and append one verified handoff entry."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from aggregate_e8385 import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CONTRASTS, METRICS, ROOT, document, require, run

GROUPS = run.FORMAL_GROUPS
HEADING = "## Experiment 83.8.5 · TADPOLE COGNITIVE_TEST-Anchored DARG + DPSL"
FORMAL_HEADING = re.compile(rb"(?m)^## Experiment 83\.8\.5(?=\s|$)")
AUDIT_FILES = (
    "method_fidelity.json", "warmup_pairing_preflight.json", "r0_identity.json",
    "r0_regression.json", "resource_probe.json", "training_seal.json", "checkpoint_replay.json",
    "mechanism_summary.json", "log_format_audit.json",
)
RESULT_FILES = ("group_summary.csv", "fold_metrics.csv", "oof_predictions.csv", "paired_contrasts.csv")
METHODS = (
    ("R0", "无", "无", "0.01", "Y3/Z0 历史完整 G4，逐折重放"),
    ("R1", "4", "无", "0.01", "仅指定 COGNITIVE_TEST 为对抗对齐参考"),
    ("R2", "无", "无", "0.10", "仅提高 COGNITIVE_TEST 的 gate scale"),
    ("R3", "4", "无", "0.10", "联合 R1 和 R2"),
    ("R4", "4", "4", "0.10", "在 R3 基础上对 COGNITIVE_TEST 启用历史 CORR 边规则"),
)


def markdown_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |",
         "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def passed_attempts():
    tasks = {(group, f"{fold:02d}") for group in run.TRAIN_GROUPS for fold in range(1, 11)}
    verified, errors, resumed, evidence = set(), [], [], {}
    for path in sorted((ROOT / "logs").glob("formal_training*.log")):
        log = path.read_text(encoding="utf-8")
        verified.update(re.findall(r"(?m)^(?:TRAINED|VERIFIED_PREVIOUS) (R[1-4]) fold (\d{2})$", log))
        errors.extend(line for line in log.splitlines() if line.startswith("ERROR "))
        resumed.extend(line for line in log.splitlines() if line.startswith("RESUME group="))
        evidence[str(path.relative_to(ROOT))] = run.sha_file(path)
    require(verified == tasks, "Formal process records do not account for all 40 new folds")
    return {"logs": evidence, "errors": errors, "resume_events": resumed,
            "additional_attempt_logs": max(0, len(evidence) - 1)}


def collect_evidence():
    run.require_training_gate(run.TRAIN_GROUPS)
    run.load_data()
    audit_docs = {Path(name).stem: document(f"audits/{name}") for name in AUDIT_FILES}
    tables = {Path(name).stem: pd.read_csv(ROOT / "results" / name, float_precision="round_trip")
              for name in RESULT_FILES}
    method, warmup = audit_docs["method_fidelity"], audit_docs["warmup_pairing_preflight"]
    r0, regression = audit_docs["r0_identity"], audit_docs["r0_regression"]
    replay, mechanism, logs = (audit_docs[key] for key in
                               ("checkpoint_replay", "mechanism_summary", "log_format_audit"))
    seal = audit_docs["training_seal"]
    require(method["status"] == method["data_identity"] == warmup["status"] == "PASS"
            and warmup["reference_folds"] == 10 and warmup["treatment_folds"] == 40
            and regression["status"] == "PASS_EXACT" and regression["compared_folds"] == 10,
            "Method gate or historical regression is incomplete")
    require(r0["status"] == "PASS" and r0["R0"] == "REUSED_VERIFIED_FORMAL"
            and r0["best_replayed"] == r0["latest_restored"] == r0["deterministic_next_steps"] == 10,
            "Historical checkpoint identity was not verified")
    require(replay["status"] == mechanism["status"] == logs["status"] == "PASS"
            and replay["best_replayed"] == replay["latest_restored"]
            == replay["deterministic_next_steps"] == mechanism["folds"] == 50,
            "One or more final independent audits are incomplete")
    require(all(mechanism[key] == "PASS" for key in (
        "GRAPH_ANCHOR_AUDIT", "DARG_MECHANISM_AUDIT", "DPSL_MECHANISM_AUDIT")),
        "Graph or mechanism audit is incomplete")
    groups = tables["group_summary"].set_index("group")
    folds, oof, contrasts = (tables[key] for key in ("fold_metrics", "oof_predictions", "paired_contrasts"))
    protocol, sources = document("protocol.json"), document("source_manifest.json")
    require(tuple(groups.index) == GROUPS and len(folds) == 50 and len(oof) == 2990
            and len(contrasts) == 14 and (groups["folds"] == 10).all() and (groups["patients"] == 598).all(),
            "The formal tables have incomplete group, fold, or patient coverage")
    require(all(contrasts.loc[contrasts["mode"] == mode, "contrast"].tolist()
                == [f"{after}-{before}" for after, before in CONTRASTS] for mode in ("full", "native")),
            "One or more of the preregistered comparisons are missing")
    require(run.sha_file(ROOT / "results/oof_predictions.csv") == mechanism["patient_audit_sha256"]
            and logs["template_sha256"] == protocol["log_template_sha256"]
            and logs["workspace_template_sha256"] in (logs["template_sha256"], None),
            "Reviewed log template or OOF CSV changed after independent audit")
    require(all(folds[field].eq(value).all() for field, value in (
        ("data_sha256", run.DATA_SHA), ("split_sha256", run.SPLIT_SHA), ("seed", 0))),
        "Fold rows do not use the registered data and seed")
    historical_source = r0["groups"]["R0"]["1"]["identity"]["implementation_sha256"]
    completion, best_rng = {}, {}
    for group in GROUPS:
        rows = folds.loc[folds["group"] == group].sort_values("fold")
        patients = oof.loc[oof["group"] == group].sort_values("sample_index")
        current_source = seal["implementation_sha256"] if group != "R0" else historical_source
        status = "REUSED_VERIFIED_FORMAL" if group == "R0" else "TRAINED_AUDITED_FORMAL"
        require(len(rows) == 10 and rows["fold"].tolist() == list(range(1, 11))
                and rows["status"].eq(status).all() and rows["model_source_sha256"].eq(current_source).all()
                and rows["warmup_exact"].all() and patients["sample_index"].tolist() == list(range(598)),
                f"{group} failed formal identity, warmup, or OOF coverage")
        current_log = ROOT / f"groups/{group}/logs/result_Tadpole.txt"
        info = logs["groups"][group]
        require(info["status"] == "PASS" and info["folds"] == 10
                and info["epoch_counts"] == rows["last_epoch"].tolist()
                and info["log_sha256"] == run.sha_file(current_log),
                f"{group} log audit no longer matches completed folds")
        require(mechanism["groups"][group]["folds"] == 10 and len(mechanism["fold_evidence"][group]) == 10,
                f"{group} mechanism audit has incomplete fold coverage")
        best_rng[group] = 0
        for row in rows.itertuples():
            observed = replay["groups"][group][str(row.fold)]
            mechanism_fold = mechanism["fold_evidence"][group][row.fold - 1]
            require(observed["best_sha256"] == row.best_sha256 == mechanism_fold["best_sha256"]
                    and observed["latest_sha256"] == row.latest_sha256
                    and observed["best"]["status"] == observed["latest"]["restore_status"]
                    == observed["first_strict_best_and_stop"] == mechanism_fold["status"] == "PASS"
                    and observed["latest"]["next_step_deterministic"] and observed["warmup_exact"],
                    f"{group} fold {row.fold} is not tied to the independently audited checkpoints")
            best_rng[group] += observed["best_rng_present"]
        require(best_rng[group] == (0 if group == "R0" else 10), f"{group} unexpected Best RNG schema")
        best, last = rows["best_epoch"], rows["last_epoch"]
        completion[group] = {
            "folds": len(rows), "status": status, "warmup_exact_folds": int(rows["warmup_exact"].sum()),
            "best_epoch_range": [int(best.min()), int(best.max())],
            "last_epoch_range": [int(last.min()), int(last.max())],
            "early_stops": int((last < run.MAX_EPOCHS).sum()), "epoch_cap_reached": int((last == run.MAX_EPOCHS).sum()),
            "epoch_records": int(last.sum()), "sum_fold_training_seconds": float(rows["train_seconds"].sum()),
        }
    go_groups = [group for group in run.TRAIN_GROUPS if groups.at[group, "dual_primary_improvement"]]
    paper_groups = [group for group in GROUPS if groups.at[group, "PAPER_REFERENCE_MEAN_REACHED"]]
    return {
        "audits": audit_docs, "groups": groups, "contrasts": contrasts, "protocol": protocol, "sources": sources,
        "completion": completion, "best_rng": best_rng, "historical_source": historical_source,
        "go": go_groups, "paper": paper_groups, "formal_attempts": passed_attempts(),
    }


def score_table(evidence, mode="full"):
    prefix = "Native_" if mode == "native" else ""
    rows = []
    for group, row in evidence["groups"].iterrows():
        metrics = [f"{100*row[f'{prefix}{name}_mean']:.4f} ± {100*row[f'{prefix}{name}_std']:.4f}"
                   for name in METRICS]
        rows.append([group, "10/10", *metrics, f"{row[f'{prefix}correct']}/598",
                     f"{row[f'{prefix}NC_correct']}/209", f"{row[f'{prefix}MCI_correct']}/315",
                     f"{row[f'{prefix}AD_correct']}/74"])
    return markdown_table(("组", "折", "ACC %", "AUC_original %", "AUC_probability %", "正确",
                           "NC 正确", "MCI 正确", "AD 正确"), rows)


def patient_pair_table(evidence, mode):
    rows = []
    for row in evidence["contrasts"].loc[lambda frame: frame["mode"] == mode].to_dict("records"):
        deltas = [f"{row[f'delta_{key}_pp']:+.4f}" for key in METRICS]
        wins = ["/".join(str(row[f"{key}_fold_{tag}"]) for tag in ("W", "T", "L")) for key in METRICS]
        interval = f"[{row['patient_ACC_CI95_low_pp']:+.3f}, {row['patient_ACC_CI95_high_pp']:+.3f}]"
        rows.append([row["contrast"], *deltas, *wins, f"{row['corrected']}/{row['damaged']}",
                     f"{row['net_correct']:+d}", interval, f"{row['McNemar_exact_p']:.4f}",
                     f"{row['McNemar_Holm_p']:.4f}"])
    return markdown_table(("对比", "ΔACC", "Δ硬 AUC", "Δ概率 AUC", "ACC 胜/平/负", "硬 AUC 胜/平/负",
                           "概率 AUC 胜/平/负", "纠正/误伤", "净正确", "患者 ΔACC 95% 区间",
                           "McNemar p", "Holm p"), rows)


def paper_table(evidence):
    rows = []
    for group, item in evidence["groups"].iterrows():
        rows.append([group, f"{item['paper_ACC_delta_pp']:+.4f}", f"{item['paper_AUC_original_delta_pp']:+.4f}",
                     "是" if group in evidence["go"] else "否", "是" if group in evidence["paper"] else "否"])
    return markdown_table(("组", "相对论文 ΔACC", "相对论文 Δ硬 AUC", "超过 Z0 双主指标",
                           "PAPER_REFERENCE_MEAN_REACHED"), rows)


def graph_table(evidence):
    rows = []
    for group, item in evidence["audits"]["mechanism_summary"]["groups"].items():
        cognitive = item["graph_means"][4]
        counts = [f"{cognitive[key]:.1f}" for key in (
            "original_effective_edges", "candidate_edges_before_support_intersection", "scored_edges",
            "modified_edges", "positive_score_edges", "negative_score_edges")]
        rows.append([group, *counts, f"{cognitive['q_mean']:.4f} / {cognitive['q_std']:.4f}",
                     f"{cognitive['gate_effective']:.7f}", f"{cognitive['adjacency_change_relative']:.3e}",
                     f"{cognitive['gcn_relative_change_vs_disease_neutral']:.3e}",
                     f"{cognitive['fusion_weighted_gcn_message_raw_l2_ratio']:.3e}",
                     f"{cognitive['fusion_weighted_dcr_message_raw_l2_ratio']:.3e}"])
    return markdown_table(("组", "有效边", "候选边", "评分边", "修改边", "正分边", "负分边", "q 均值/SD",
                           "有效 gate", "边权相对 Δ", "GCN 相对 Δ", "GCN 融合/原始范数",
                           "Direct 融合/原始范数"), rows)


def intervention_table(evidence, keys):
    rows = []
    for group, item in evidence["audits"]["mechanism_summary"]["groups"].items():
        for key, display in keys:
            contrast = item["contrasts"][key]
            per_class = " / ".join(f"{counts['corrected']}↗ {counts['damaged']}↘"
                                   for counts in contrast["classes"].values())
            rows.append([group, display, contrast["flips"], f"{contrast['net_correct']:+d}", per_class,
                         f"{contrast['cam_relative_delta_mean']:.3e}",
                         f"{contrast['enhanced_cam_relative_delta_mean']:.3e}",
                         f"{contrast['max_abs_logit_delta']:.3e}", f"{contrast['max_abs_prob_delta']:.3e}",
                         f"{contrast['mean_abs_true_margin_delta_mean']:.3e}"])
    return markdown_table(("组", "加入 Full 的对照起点", "argmax 变化", "净正确", "NC / MCI / AD 纠正与误伤",
                           "CAM 相对 Δ", "增强 CAM 相对 Δ", "最大 |Δlogit|", "最大 |Δ概率|",
                           "平均 |Δ真类 margin|"), rows)


def relation_table(evidence):
    rows = []
    comparisons = evidence["audits"]["mechanism_summary"]["heldout_relation_geometry_contrasts"]
    for contrast, modalities in comparisons.items():
        for name, item in modalities.items():
            d = item["mean_conditioned_cosine_distance_delta"]
            rows.append([contrast, name, item["encoder_changed_folds"], item["scorer_changed_folds"],
                         f"{d['same']:+.5f}", f"{d['different']:+.5f}", f"{100*item['mean_same_auc_delta']:+.4f}"])
    return markdown_table(("对比", "模态", "编码器变化折", "评分器变化折", "同病质心距离 Δ",
                           "异病质心距离 Δ", "V–V 同病/异病 ΔAUC pp"), rows)


def training_nll_table(evidence):
    rows = []
    for group, item in evidence["audits"]["mechanism_summary"]["groups"].items():
        values = item["train_class_nll_eval_mode"]
        rows.append([group, *[
            f"{values[key]['full_minus_control_mean']:+.3e} ({values[key]['full_lower_nll_folds']}/10)"
            for key in ("score_permuted", "disease_neutral")]])
    return markdown_table(("组", "Full − Score-Permuted NLL (Full 更低折数)",
                           "Full − Disease-Neutral NLL (Full 更低折数)"), rows)


def execution_table(evidence):
    rows = [[group, item["status"], f"{item['folds']}/10", f"{item['best_epoch_range'][0]}–{item['best_epoch_range'][1]}",
             f"{item['last_epoch_range'][0]}–{item['last_epoch_range'][1]}", item["early_stops"],
             item["epoch_cap_reached"], item["epoch_records"]]
            for group, item in evidence["completion"].items()]
    return markdown_table(("组", "状态", "完成折", "Best epoch", "Latest epoch", "提前停止折",
                           "达到上限折", "真实逐轮记录数"), rows)


def trial_table(evidence):
    probe = evidence["audits"]["resource_probe"]
    rows = [[item["workers"], item["status"], f"{item['steps_per_second']:.4f}",
             f"{item['gpu_utilization_percent_median']:g}", item["gpu_device_peak_mib"]]
            for item in probe["trials"]]
    return markdown_table(("并发", "探针", "更新/秒", "GPU 利用率中位数 %", "峰值 MiB"), rows)


def log_table(evidence):
    return markdown_table(("组", "状态", "记录行数", "训练日志 SHA-256"),
                          [[group, item["status"], item["line_count"], f"`{item['log_sha256']}`"]
                           for group, item in evidence["audits"]["log_format_audit"]["groups"].items()])


def scientific_summary(evidence):
    groups = evidence["groups"]
    contrasts = evidence["contrasts"].loc[lambda frame: frame["mode"] == "full"].set_index("contrast")
    r3 = contrasts.loc["R3-R0"]
    r4 = contrasts.loc["R4-R3"]
    additive_departure = {
        name: 100 * (groups.at["R3", f"{name}_mean"] - groups.at["R1", f"{name}_mean"]
                     - groups.at["R2", f"{name}_mean"] + groups.at["R0", f"{name}_mean"])
        for name in METRICS}
    go, paper = ", ".join(evidence["go"]) or "无", ", ".join(evidence["paper"]) or "无"
    uncertain = [name for name, row in contrasts.iterrows()
                 if row["ACC_fold_CI95_low_pp"] <= 0 <= row["ACC_fold_CI95_high_pp"]
                 or row["AUC_original_fold_CI95_low_pp"] <= 0 <= row["AUC_original_fold_CI95_high_pp"]]
    patient_uncertain = [name for name, row in contrasts.iterrows()
                         if row["patient_ACC_CI95_low_pp"] <= 0 <= row["patient_ACC_CI95_high_pp"]]
    holm = [name for name, row in contrasts.iterrows() if row["McNemar_Holm_p"] < 0.05]
    lower_than_both = all(contrasts.loc[name, column] < 0 for name in ("R3-R1", "R3-R2")
                          for column in ("delta_ACC_pp", "delta_AUC_original_pp"))
    probability_only = [group for group in run.TRAIN_GROUPS if group not in evidence["go"]
                        and groups.at[group, "Z0_AUC_probability_delta_pp"] > 0]
    relation = evidence["audits"]["mechanism_summary"]["heldout_relation_geometry_contrasts"]
    if evidence["go"]:
        verdict = f"相对 Z0 的预注册性能判定为 GO，符合组为 {go}。"
    else:
        verdict = "相对 Z0 的预注册性能判定为 NO-GO，继续保留 Y3/Z0 作为完整 G4 基准。"
    lines = [
        f"{verdict}达到论文两项参考均值的组为 {paper}。",
        f"R3 相对 R0 的 ΔACC 为 {r3['delta_ACC_pp']:+.4f} pp，ΔAUC_original 为 "
        f"{r3['delta_AUC_original_pp']:+.4f} pp，净正确 {r3['net_correct']:+d} 人。R3 的描述性交互量 "
        f"R3−R1−R2+R0 为 ACC {additive_departure['ACC']:+.4f} pp、"
        f"AUC_original {additive_departure['AUC_original']:+.4f} pp。",
        f"R4 相对 R3 的 ΔACC 为 {r4['delta_ACC_pp']:+.4f} pp，ΔAUC_original 为 "
        f"{r4['delta_AUC_original_pp']:+.4f} pp，净正确 {r4['net_correct']:+d} 人。",
        f"Full 对比中至少一项主指标的折配对区间跨 0 的对比为 {', '.join(uncertain) or '无'}。"
        f"患者 ΔACC 区间跨 0 的对比为 {', '.join(patient_uncertain) or '无'}。"
        f"患者正确性 McNemar 的七项 Holm 校正后 p < 0.05 的对比为 {', '.join(holm) or '无'}。",
    ]
    if lower_than_both:
        lines.insert(2, "R3 的两项主指标均低于 R1 和 R2，当前数据未显示锚定与门控带来分类收益互补。")
    if probability_only:
        lines.insert(1, f"相对 Z0 的概率 AUC 辅助收益见于 {', '.join(probability_only)}。")
    for contrast, items in relation.items():
        others = [value for name, value in items.items() if name != "COGNITIVE_TEST"]
        counts = [sum(item["mean_conditioned_cosine_distance_delta"][label] < 0 for item in others)
                  for label in ("same", "different")]
        auc_mean = 100 * float(np.mean([item["mean_same_auc_delta"] for item in others]))
        lines.append(f"{contrast} 的其他五模态中，同病和异病质心距离缩小的模态分别为 "
                     f"{counts[0]}/5 和 {counts[1]}/5，平均同病/异病 AUC 变化 {auc_mean:+.4f} pp。")
    return "\n\n".join(lines)


def mechanism_conclusions(evidence):
    groups = evidence["audits"]["mechanism_summary"]["groups"]
    counts = {name: sum(group["contrasts"][f"{name}_to_full"]["flips"] for group in groups.values())
              for name in ("direct_off", "disease_neutral", "anchor_neutral", "cognitive_graph_off", "score_permuted")}
    max_logit = max(item["contrasts"][f"{name}_to_full"]["max_abs_logit_delta"]
                    for item in groups.values() for name in counts)
    nll_spread = max(abs(result["full_minus_control_mean"]) for item in groups.values()
                     for result in item["train_class_nll_eval_mode"].values())
    r4 = groups["R4"]["graph_means"][4]
    r4_class = groups["R4"]["contrasts"]["native_to_full"]["classes"]
    r4_effect = ", ".join(f"{name} {value['corrected']} 人" for name, value in r4_class.items() if value["corrected"])
    return (
        "由各组固定 checkpoint 的汇总可见，Direct-Off、Disease-Neutral、Anchor-Neutral、"
        f"Cognitive-Graph-Off 和 Score-Permuted 回到 Full 分别改变 "
        f"{'/'.join(str(count) for count in counts.values())} 个 argmax，最大 |Δlogit| 为 {max_logit:.3e}。"
        f"R4 认知门控图边扩展后的 GCN 消息相对原图变化为 {r4['gcn_relative_change_vs_disease_neutral']:.3e}。"
        f"训练患者分类 NLL 的组均差最大绝对值为 {nll_spread:.3e}，未出现一致下降。"
        f"R4 的 DPSL 固定 checkpoint 纠正 {r4_effect or '0 人'}，误伤 "
        f"{groups['R4']['contrasts']['native_to_full']['damaged']} 人。"
    )


def restoration_commands(evidence):
    workers = evidence["audits"]["resource_probe"]["worker_count"]
    return "\n".join([
        "```bash", f"cd {ROOT}",
        "export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0",
        'e8385_recovery_log="logs/formal_training_recovery_$(date -u +%Y%m%dT%H%M%SZ).log"',
        f'python3 -u code/run_e8385.py --groups R1 R2 R3 R4 --workers {workers} > "$e8385_recovery_log" 2>&1',
        "python3 -u audits/replay_e8385.py", "python3 -u audits/write_author_logs_e8385.py",
        "python3 -u audits/audit_log_format_e8385.py", "python3 -u audits/mechanism_e8385.py",
        "python3 -u audits/aggregate_e8385.py", "python3 -u audits/finalize_e8385.py", "```",
    ])


def execution_notes(evidence):
    attempts = evidence["formal_attempts"]
    return (f"正式训练记录到 {len(attempts['errors'])} 个折错误及 "
            f"{attempts['additional_attempt_logs']} 次额外启动，"
            f"{len(attempts['resume_events'])} 次从 Latest 恢复。"
            "预检审计器曾两次中止。第一次改为从同一初始化图重建关系对，第二次改用历史 RNG 恢复 API。"
            "两次失败均保留日志，修复后完整预检通过并封印训练源码。")


def make_report(evidence):
    seal = evidence["audits"]["training_seal"]
    sources = evidence["sources"]
    repositories = sources["repositories"]
    graph = evidence["audits"]["mechanism_summary"]["groups"]
    max_ratio = max(item["dpsl_residual_ratio_max"] for item in graph.values())
    max_budget = max(item["dpsl_shared_ratio_max"] for item in graph.values())
    return f"""# E83.8.5 · TADPOLE COGNITIVE_TEST-Anchored DARG + DPSL

R0 历史核验 10/10。R1–R4 真实训练 40/40。Best 独立重放、Latest 恢复和确定性下一步均为 50/50。

## 来源与协议

论文 MICCAI 2025 Table 2 报告 ACC 94.81 ± 2.95% 和 AUC 95.59 ± 2.57%。作者仓库 `{repositories["author"]["commit"]}` 有 GFR、GraphLearn、GCN、CAM、PEN、SPGS、RES 和训练日志来源。作者示例训练实现采用 ABIDE 四模态，正式实验沿用 SABER 六模态适配。SABER 核查提交 `{repositories["SABER"]["local_checkout_commit"]}`，TADPOLE N3 核查提交 `{repositories["TADPOLE"]["verified_N3_commit"]}`。历史 Y3 短引用 `cbaaf1c` 未解析出完整提交。Y3/Z0 的本地封印源码、配置、原始 checkpoint 和独立 GPU 重放建立了传承身份。 [SPromptGL 论文]({sources["paper"]["url"]}) · [源码核查](../audits/source_review.md) · [来源清单](../source_manifest.json)

NC 209、MCI 315、AD 74，标签 0/1/2。模态依序为 UCSFFSX 145、UCBERKELEYAV45 148、UPENNBIOMK9 3、RISK_FACTOR 36、COGNITIVE_TEST 24、ROI_AVERAGE 9，共 365 维。数据 SHA-256 `{run.DATA_SHA}`，分折 SHA-256 `{run.SPLIT_SHA}`。十折使用 `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)`，训练 seed=0，每折设置一次。沿用全队列均值填补、标准化、RFE，转导特征图、GraphLearn 0.9 阈值和历史第二模态覆盖行为。val=test 用于首次严格 ACC 提升的 Best 选择。外层测试标签不进入关系对和训练损失，历史预处理及重复使用验证标签限制对独立新患者的泛化解释。 [数据清单](../data_manifest.json) · [协议](../protocol.json)

完整 G4 为六模态 GFR、CAM、RES、DARG 和 DPSL SCMR-DA-SRPE。沿用 MMGPL-12、冻结 RN50、DPSL 历史置信门控和优化器，以及 Y3 平方 Frobenius ALoss 系数 0.01 从第 11 轮启用。每折用 4,096 个确定性训练对，同病与异病各 2,048。BCE、条件对抗、关系保留系数依次为 0.05、0.01、0.01。十轮 warmup、450 轮上限和 `wait > 200` 延续历史实现。对齐、图边锚定和增强门控从第 11 轮才产生组差异。

P0.1 的 COGNITIVE_TEST 在四项关系指标上均为 10/10 折第一，V–V 与 V–T 的六关系 AUC 为 0.8325 和 0.8501，同病/异病 AUC 为 0.7153 和 0.7238。正式训练只依据该结果选取锚点。 [P0.1 归档](../sources/reference/P0.1/results/modality_summary.csv)

## 实验组与方法保真

{markdown_table(("组", "对齐参考索引", "图边锚定索引", "认知模态 gate scale", "定义"), METHODS)}

其他五模态尺度统一 0.01，各组六模态初始门控概率 0.1。COGNITIVE_TEST 名称、索引 4 和特征顺序保持原样。对抗参考只在条件对抗路径 detach，其他模态沿用 GRL，参考编码器仍接受 BCE 等训练梯度。普通候选为原图有效边与 mutual-topK(12) 交集，评分阈值为 0.45/0.55。R4 仅在 COGNITIVE_TEST 的全部原始有效边采用连续 `2q−1`，正负重加权均依赖原边。各组沿用 Positive-DCR 及 Direct-DCR 二次门控。

R0 与 Z0 的原图、关系分数、GCN 路径、Native/Full、分类及训练梯度按原始实现逐折精确回归。R1–R4 预检逐组核对参数梯度、参考 detach、GRL、门控优化、图边范围、数值有限性和对称性。扰动外层测试标签后关系对、目标、训练损失和特征路径一致。全部 40 个新增折使用历史 N3/Y3 十轮 warmup，共享初始化、模型、优化器、Python/NumPy/PyTorch/CUDA RNG 及第 11 轮前 DPSL 谱统计均精确配对。warmup 训练损失真实执行，日志所用参考指标已独立逐轮复算并证明与历史相等。 [方法预检](../audits/method_fidelity.json) · [Warmup](../audits/warmup_pairing_preflight.json)

## 资源、训练和重放

PPU-ZW810E，GPU0，98,304 MiB。Python {sources["dependencies"]["python"]}，PyTorch {sources["dependencies"]["torch"]}，CUDA ABI {sources["dependencies"]["torch_cuda_abi"]}。每 worker 一条 CPU 计算线程。探针覆盖完整训练步和评估，用实测最高吞吐的 4 个 worker 运行正式训练。

{trial_table(evidence)}

{execution_table(evidence)}

训练封印 `{seal["implementation_sha256"]}`，Y3 历史模型源码身份 `{evidence["historical_source"]}`。50/50 逐折核对数据和分折、训练患者与测试患者索引、训练配置和源码、checkpoint、硬预测及 float32 概率。Latest 的完整 RNG 均恢复并做两次独立下一步，模型、优化器、损失和 RNG 一致。R0 的 10 份原始 Best 只用于推理核验，完整 RNG 存于原始 Latest。40 份新 Best 同时保存完整 RNG。每组 598 名患者 OOF 恰好一次。{execution_notes(evidence)} [重放记录](../audits/checkpoint_replay.json) · [训练封印](../audits/training_seal.json)

## Full 十折性能

ACC 和 AUC_original 是两项主指标。AUC_original 用硬预测 one-hot 的 macro OVR。AUC_probability 用概率 macro OVR。表中为十折均值 ± 总体标准差。

{score_table(evidence)}

### Full 同折同患者差值

指标差值和区间单位为百分点。以折和真实类别分层的配对患者 bootstrap 区间，{BOOTSTRAP_DRAWS:,} 次抽样，seed={BOOTSTRAP_SEED}。CSV 还给出以折为抽样单位的三个指标区间，各类纠正和误伤，McNemar 精确检验及按模式七项 Holm 校正。折间训练集重叠使这些配对推断适用于本队列的描述性比较。

{patient_pair_table(evidence, "full")}

### Native 同折同患者差值

Native 已经过相应组的联合训练，推理去掉 DPSL 残差。下表反映联合训练后 Native 路径的差异。

{score_table(evidence, "native")}

{patient_pair_table(evidence, "native")}

### 与论文参考均值之差

{paper_table(evidence)}

{scientific_summary(evidence)}

## 图边与关系打分

下表是十折 Best 的 COGNITIVE_TEST 均值。边数按无序、非自环、有效图边。P0.1 的平均 13,123.9 个有效边和 2,491.6 个候选的 0.0140 覆盖率用 178,503 个全部潜在患者对作分母。其全部候选均被评分。

{graph_table(evidence)}

R4 的评分集合逐折等于原始认知模态有效边，原图支持、对称性和对角线通过检查，边权为有限值。同一组同一 checkpoint 的认知图干预保持其他五模态的邻接、GCN 消息和 Direct-DCR 残差逐元素相等。六模态的完整边计数、门控、分数分布和传播范数在审计 JSON。对抗参考身份仅用于训练，推理时单独移除该身份的 logit 逐元素相等。 [机制证据](../audits/mechanism_summary.json)

## DARG 同状态机制

以下是从同 checkpoint 的各推理对照回到 Full 的患者差异。Direct-Off 保留重加权图和 GCN。Disease-Neutral 回到原图并关闭直接残差。Anchor-Neutral 恢复普通候选与阈值。Cognitive-Graph-Off 仅关闭认知关系图调整。Score-Permuted 在评分支持内置换关系分数，检验分数与边的匹配在推理中的作用。真类 margin 是真类 logit 与最大其他类 logit 之差。

{intervention_table(evidence, (
       ("direct_off_to_full", "Direct-Off"), ("disease_neutral_to_full", "Disease-Neutral"),
       ("anchor_neutral_to_full", "Anchor-Neutral"), ("cognitive_graph_off_to_full", "Cognitive-Graph-Off"),
       ("score_permuted_to_full", "Score-Permuted")))}

以相同 Best 在 eval 模式下计算训练患者分类 NLL。负差值表示原始关系分数或完整疾病图对应较低训练患者 NLL。该结果衡量固定参数下的即时作用。

{training_nll_table(evidence)}

以下为测试患者 V–V 关系诊断。先生成关系表示和分数，再以真实类别划分同病与异病用于度量。距离是同一关系类别下该模态与认知模态的质心余弦距离。联合训练参数差异、对照几何及推理受控比较衡量不同作用。

{relation_table(evidence)}

## DPSL 同状态机制

Native 和 Residual-Off 在 50 折逐元素相等，且上游 CAM 保持一致。推理残差最大比例 {max_ratio:.8f}（上限 0.08）。最新 checkpoint 中历史共享梯度更新的最大实测比例 {max_budget:.8f}（预算 0.001），四族参数均逐轮检查。

{intervention_table(evidence, (("native_to_full", "Native / Residual-Off"),))}

{mechanism_conclusions(evidence)}

## 训练日志与恢复

作者模板 SHA-256 `{evidence["audits"]["log_format_audit"]["template_sha256"]}`，工作区原模板与作者提交中模板逐字节相同。新组真实逐轮记录使用 `/450`，只记录实跑轮次。两项同名分类损失按作者代码显示同一个实测分类损失。第三项显示作者未平方的图损失观测值，训练反向传播仍用历史 Y3 平方版本。R0 文件从真实历史记录校验重建，和继承日志逐字节相等。

{log_table(evidence)}

从封存 Latest 恢复时保留旧训练日志并使用新日志路径。复跑会校验已完成折，按原 RNG、优化器与历史轨迹恢复未完成折。后续重放和汇总可独立执行。

{restoration_commands(evidence)}
"""


def make_handoff(evidence):
    sources = evidence["sources"]["repositories"]
    mechanism = evidence["audits"]["mechanism_summary"]["groups"]
    lines = []
    for group, item in mechanism.items():
        cognitive = item["graph_means"][4]
        native = item["contrasts"]["native_to_full"]
        direct, disease = (item["contrasts"][key] for key in ("direct_off_to_full", "disease_neutral_to_full"))
        lines.append([group, f"{cognitive['gate_effective']:.7f}",
                      f"{cognitive['original_effective_edges']:.1f}/{cognitive['scored_edges']:.1f}",
                      f"{direct['corrected']}/{direct['damaged']}", f"{disease['corrected']}/{disease['damaged']}",
                      f"{native['corrected']}/{native['damaged']}"])
    return f"""{HEADING}

- 状态 **COMPLETE_AUDITED**。正式目录 `{ROOT}`。R0 继承核验 10 折，R1–R4 新训练 40 折。
- 论文为 MICCAI 2025 SPromptGL。作者源码 `{sources["author"]["commit"]}`，SABER `{sources["SABER"]["local_checkout_commit"]}`，TADPOLE N3 `{sources["TADPOLE"]["verified_N3_commit"]}`。Y3 短 Git 引用 `cbaaf1c` 未解析，本地封印与原始 checkpoint 身份独立核验通过。 [论文]({evidence["sources"]["paper"]["url"]}) · [来源清单]({ROOT}/source_manifest.json)
- 数据 `{run.DATA_SHA}`。分折 `{run.SPLIT_SHA}`。历史模型源码 `{evidence["historical_source"]}`。正式封印 `{evidence["audits"]["training_seal"]["implementation_sha256"]}`。
- PPU-ZW810E GPU0 98,304 MiB，探针比较 1/2/4/8 并发后选 4 worker。seed=0，历史 10 轮 warmup、450 上限和 wait > 200，val=test，CV 前预处理和 RFE，转导特征图。保留 Y3 ALoss、DARG 双门控、DPSL 0.08/0.001、冻结 RN50 和 MMGPL-12。
- R0 无锚定及弱 gate。R1 仅参考索引 4。R2 仅认知 gate scale 提到 0.1。R3 合并 R1+R2。R4 再对认知索引 4 使用全部有效原边和历史连续关系分数。全部初始 gate 概率为 0.1，其他模态 scale 0.01。P0.1 只确定锚点。方法保真、测试标签扰动、图边支持和 warmup 配对通过，40/40 新折初始、warmup 与 RNG 精确匹配。
- Best/Latest 和确定性下一步均 50/50。R0 原 Best 核验推理，其训练恢复使用带完整 RNG 的 Latest。五组各 598 名患者 OOF 恰好一次。原模板 SHA-256 `{evidence["audits"]["log_format_audit"]["template_sha256"]}`，五组日志格式审计均通过，R0 继承日志逐字节重建核对。{execution_notes(evidence)}

{score_table(evidence)}

{patient_pair_table(evidence, "full")}

{markdown_table(("组", "认知 gate", "有效边/评分边", "Direct-Off→Full 纠正/误伤",
                  "Disease-Neutral→Full 纠正/误伤", "DPSL Native→Full 纠正/误伤"), lines)}

{mechanism_conclusions(evidence)}

{scientific_summary(evidence)}

完整结果含 Native/Full、DARG 同状态干预、各类变化和论文差值。 [正式报告]({ROOT}/results/final_report.md) · [最终审计]({ROOT}/audits/final_audit.json) · [日志格式核验]({ROOT}/audits/log_format_audit.json)

{restoration_commands(evidence)}
"""


def append_handoff(section):
    path = ROOT.parent / "handoff.md"
    require(path.is_file(), "Historical handoff file is missing")
    addition = b"\n\n" + section.encode("utf-8")
    header = HEADING.encode("utf-8")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        previous = stream.read()
        if not FORMAL_HEADING.search(previous):
            prefix = previous
            stream.write(addition)
            stream.flush()
            os.fsync(stream.fileno())
        else:
            require(previous.endswith(addition), "An existing formal handoff differs from the verified evidence")
            prefix = previous[:-len(addition)]
        stream.seek(0)
        observed = stream.read()
        require(observed == prefix + addition and len(FORMAL_HEADING.findall(observed)) == 1
                and observed.splitlines().count(header) == 1, "Formal handoff readback or uniqueness failed")
    return {"status": "PASS", "path": str(path), "formal_sections": 1, "prefix_bytes": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "append_sha256": hashlib.sha256(addition).hexdigest(),
            "readback_sha256": hashlib.sha256(observed).hexdigest()}


def main():
    with run.lock(ROOT / "audits/training.lock"):
        evidence = collect_evidence()
        report = ROOT / "results/final_report.md"
        run.atomic_write(report, lambda stream: stream.write(make_report(evidence).encode("utf-8")))
        handoff = append_handoff(make_handoff(evidence))
        files = [f"audits/{name}" for name in AUDIT_FILES]
        files += [f"results/{name}" for name in (*RESULT_FILES, report.name)]
        files += [f"groups/{group}/logs/result_Tadpole.txt" for group in GROUPS]
        files += ["source_manifest.json", "protocol.json", "data_manifest.json"]
        flags = ("EXECUTION_AUDIT", "METHOD_FIDELITY", "DATA_IDENTITY", "WARMUP_PAIRING",
                 "CHECKPOINT_REPLAY", "RNG_REPLAY", "GRAPH_ANCHOR_AUDIT", "DARG_MECHANISM_AUDIT",
                 "DPSL_MECHANISM_AUDIT", "LOG_FORMAT_AUDIT", "TRAINING_SEAL")
        audit = {
            "experiment": run.EXPERIMENT, "status": "COMPLETE_AUDITED",
            "verified_utc": datetime.now(timezone.utc).isoformat(), "experiment_directory": str(ROOT),
            **dict.fromkeys(flags, "PASS"),
            "PERFORMANCE_GO_NO_GO": "GO" if evidence["go"] else "NO-GO",
            "dual_primary_improvement_groups": evidence["go"],
            "PAPER_REFERENCE_MEAN_REACHED": evidence["paper"],
            "official_folds": 50, "new_trained_folds": 40, "reused_verified_folds": 10,
            "best_replayed": 50, "latest_restored": 50, "deterministic_next_steps": 50,
            "warmup_exact_new_folds": 40, "best_full_rng_by_group": evidence["best_rng"],
            "oof_patients_each_once_per_group": 598, "groups": evidence["completion"],
            "data_sha256": run.DATA_SHA, "split_sha256": run.SPLIT_SHA,
            "training_implementation_sha256": evidence["audits"]["training_seal"]["implementation_sha256"],
            "historical_implementation_sha256": evidence["historical_source"],
            "historical_y3_git_short_ref_resolved": False,
            "formal_attempts": evidence["formal_attempts"],
            "evidence_sha256": {name: run.sha_file(ROOT / name) for name in files},
            "handoff_readback": handoff, "finalizer_sha256": run.sha_file(Path(__file__)),
        }
        run.write_json(ROOT / "audits/final_audit.json", audit)
    print(audit["status"], audit["PERFORMANCE_GO_NO_GO"], flush=True)


if __name__ == "__main__":
    main()
