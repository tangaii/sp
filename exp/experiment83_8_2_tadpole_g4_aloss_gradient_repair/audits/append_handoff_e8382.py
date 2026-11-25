"""Append the verified E83.8.2 record to the shared project handoff once."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
HISTORICAL_8381 = Path("/model/spromptgl/experiment/experiment83_8_1_tadpole_six_group_recovery")
EXPECTED_HANDOFF_SHA256 = "f6f07f380b32f2a3924ac67c9c08e291188fae47ff8b8fd9138e88bb49785769"
SECTION_MARKER = "## Experiment 83.8.2 — TADPOLE G4 ALoss Gradient Scale and Warmup Repair"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def pct_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.4f}% ± {std * 100:.4f} pp"


def build_historical_checkpoint_manifest() -> dict:
    rows = read_csv(HISTORICAL_8381 / "results/fold_metrics.csv")
    selected_groups = {"H1", "H2", "H3", "H4", "H5", "X3", "X4", "X5", "X6"}
    records = []
    for row in rows:
        if row["group"] not in selected_groups:
            continue
        path = Path(row["best_checkpoint"])
        if row["status"] != "PASS" or not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"Historical checkpoint is missing or unverified: {row}")
        actual_sha = sha256_file(path)
        if actual_sha != row["best_sha256"]:
            raise RuntimeError(f"Historical checkpoint SHA mismatch: {path}")
        records.append({
            "group": row["group"],
            "fold": int(row["fold"]),
            "checkpoint_role": "Best",
            "absolute_path": str(path),
            "sha256": actual_sha,
            "source": row["source"],
            "source_metrics": str(HISTORICAL_8381 / "results/fold_metrics.csv"),
        })
    counts = {group: sum(record["group"] == group for record in records) for group in sorted(selected_groups)}
    if len(records) != 90 or any(count != 10 for count in counts.values()):
        raise RuntimeError(f"Expected 90 verified historical Best checkpoints, got {counts}")
    return {
        "experiment": "Experiment 83.8.2",
        "status": "READ_ONLY_VERIFIED",
        "historical_source_metrics": str(HISTORICAL_8381 / "results/fold_metrics.csv"),
        "verified_checkpoint_count": len(records),
        "group_fold_counts": counts,
        "initialization_policy": "No historical checkpoint was used to initialize a new formal fold",
        "records": records,
    }


def make_section(historical_manifest_sha: str) -> str:
    seal = load_json("audits/training_seal.json")
    preflight = load_json("audits/preflight_start.json")
    final = load_json("audits/final_audit.json")
    gradients = load_json("audits/gradient_mechanism.json")
    warmup = load_json("audits/warmup_pairing.json")
    recovery = load_json("audits/checkpoint_recovery.json")
    report_repair = load_json("audits/report_repair.json")
    prompt_manifest = load_json("prompt_manifest.json")
    summaries = read_csv(ROOT / "results/group_summary.csv")
    patient_rows = read_csv(ROOT / "results/patient_mechanism.csv")

    if final["status"] != "PASS" or final["completed_folds"] != 70:
        raise RuntimeError("Final audit is not a complete 70-fold PASS")
    if warmup["status"] != "PASS" or recovery["status"] != "PASS":
        raise RuntimeError("Warmup pairing or checkpoint recovery audit did not pass")
    if not report_repair["training_sources_unmodified"]:
        raise RuntimeError("Report repair indicates a training-source change")

    out = [
        SECTION_MARKER,
        "",
        "- **状态** 完成。METHOD_FIDELITY PASS，EXECUTION_AUDIT PASS，PERFORMANCE_GO。N3 达到性能候选门槛，完整模型研究主线仍为 G4 · E82.4.6.10-G1（DARG + DPSL）。",
        "- **实际目录** `/model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair`。首选 `/data/adodas/SPromptGL-main/experiment/...` 创建时遇 OSSFS EIO/errno 5，随后切换到实际目录。根目录模型源码及 E83.5–E83.8.1 历史目录均未修改。",
        f"- **身份封存** 实现 SHA-256 `{seal['implementation_sha256']}`，training seal SHA-256 `{sha256_file(ROOT / 'audits/training_seal.json')}`。启动预检 `{preflight['status']}`，训练前封存 `{seal['status']}`。单张 PPU-ZW810E、8 个 GPU worker、seed 0。",
        "",
        "### 来源、方法保真与输入",
        "",
        "- 论文来源为 [SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)，作者仓库为 [wanxixi11/SPromptGL](https://github.com/wanxixi11/SPromptGL)，项目仓库为 [tangaii/SPromptGL](https://github.com/tangaii/SPromptGL)。源码/公式差异和本轮执行对象见 `audits/method_fidelity_matrix.md`。论文 Eq.(1) 使用平方 Frobenius 项，Eq.(7)、GFR、CAM 与 TADPOLE 配置按论文和原作者实现分别审计。",
        "- 作者 `network.py` 的 ALoss 是未平方的 `torch.linalg.norm`，作者 `model.py` 的 simple-2 backward 不把该 ALoss 纳入反向传播。本轮保留该历史事实，不把它改写成论文描述。E83.8.1 的六模态 train×train 平方 ALoss 是另一登记变体，E83.6 连续相似度 PairMean 是不同对象，本轮不使用 PairMean。",
        "- GFR 保留 threshold=0.9、六个 GraphLearn 图、原始消息融合和第二模态历史覆盖行为。DARG 保留 Positive-DCR、正负 signed 关系和十轮 Warmup。DPSL 保留 MMGPL-12 三分类 Prompt、NC–MCI/NC–AD/MCI–AD 三对、谱统计、残差和共享梯度机制。",
        f"- 数据为 598 人，NC/MCI/AD 209/315/74，标签 0/1/2，模态维度 145/148/3/36/24/9，总维度 365。数据 SHA-256 `{seal['data_sha256']}`，十折 SHA-256 `{seal['split_sha256']}`。",
        f"- Prompt 定义 SHA-256 `{sha256_file(ROOT / 'prompt_manifest.json')}`，Query shape `{prompt_manifest['query_shape']}`，12 条 class prompt、3 类及 3 对类别证据。Prompt 文件 SHA-256 见 `source_manifest.json`，RN50 文件 SHA-256 `{prompt_manifest['rn50_file_sha256']}`，SCMR 文件 SHA-256 `{prompt_manifest['scmr_file_sha256']}`。",
        "- 保留历史全数据 RFE、均值填补和外层 CV 前 StandardScaler、转导患者图、`val_idx == test_idx`、外层测试 ACC 首次严格提升才保存 Best、`wait > 200`、最大 450 epoch、原优化器/学习率/权重衰减/标签权重与随机操作。该协议有泛化评估偏差，详见限制。",
        "",
        "### 七组冻结方法与正式结果",
        "",
        "N0 是完整 G4 clean control。N1/N2/N3 分别为 λ=0.001/0.003/0.010 且 epoch 11 启用。N4 为 λ=0.003、epoch 1 启用并覆盖 Warmup。N5 在 epoch 11 按每折训练梯度一次性校准 λ，目标比例 0.1，裁剪到 `[1e-5, 1e-2]`。N6 为完整 G3 + λ=0.003、epoch 11 启用，不含 DPSL。全部配置在训练前冻结。",
        "N0–N6 各完成 10/10 个正式 fold，合计 70/70。每折保存 Best、Latest、训练轨迹、患者概率与折审计。",
        "",
        "以下为每组十折 mean ± population standard deviation。AUC_original 使用历史 one-hot hard-prediction macro OVR 定义，概率 AUC 仅作辅助。",
        "",
        "| 组 | ACC | AUC_original | AUC_probability 均值 | OOF 正确 | NC/MCI/AD 正确 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["group"] == "PAPER":
            out.append("| 原论文 | 94.8100% ± 2.9500 pp | 95.5900% ± 2.5700 pp | — | — | — |")
            continue
        out.append(
            f"| {row['group']} | {pct_pm(float(row['ACC_mean']), float(row['ACC_std_population']))} "
            f"| {pct_pm(float(row['AUC_original_mean']), float(row['AUC_original_std_population']))} "
            f"| {pct(float(row['AUC_probability_mean']))} | {row['OOF_correct']}/598 "
            f"| {row['NC_correct']}/{row['MCI_correct']}/{row['AD_correct']} |"
        )

    out += [
        "",
        "历史 H1–H5 来自 E83.5 核验结果，X3/X4/X5/X6 来自 E83.8.1。只读 checkpoint 绝对路径、SHA、fold 和来源已逐项验证并记录在 `audits/historical_checkpoint_manifest.json`，manifest SHA-256 为 `" + historical_manifest_sha + "`。90 个历史 Best checkpoint 的 SHA 与历史 fold metrics 一致，没有用作新组初始化。",
        "",
        "### Warmup 配对与 ALoss 梯度审计",
        "",
        "N0/N1/N2/N3/N5 逐折初始参数、优化器、RNG、DARG 关系对、十轮 Warmup 损失、Warmup 后共有模型/优化器/RNG、DPSL 谱统计与 Prompt 全部一致。N4 初始状态相同，epoch 1 加入 ALoss 后其 Warmup 更新轨迹、参数/优化器和谱统计随之改变，Warmup 后 RNG 与 Prompt 仍相同。N6 共有 GFR/CAM/DARG Warmup 对照 10/10 PASS。",
        "",
        "| 组 | ALoss/历史损失梯度范数比均值 [范围] | λ 均值/固定值 | 梯度校准裁剪折数 |",
        "|---|---:|---:|---:|",
    ]
    for group in ["N1", "N2", "N3", "N4", "N5", "N6"]:
        item = gradients["aloss_gradients"]["groups"][group]
        coefficient = item["lambda_fold_mean"]
        if coefficient is None:
            coefficient_text = {"N1": "0.001", "N2": "0.003", "N3": "0.010", "N4": "0.003", "N6": "0.003"}[group]
        else:
            coefficient_text = f"{coefficient:.10f} [{item['lambda_fold_min']:.10f}, {item['lambda_fold_max']:.10f}]"
        out.append(
            f"| {group} | {item['actual_gradient_ratio_mean']:.6f} "
            f"[{item['actual_gradient_ratio_min']:.6f}, {item['actual_gradient_ratio_max']:.6f}] "
            f"| {coefficient_text} | {item['clipped_folds']} |"
        )
    out += [
        "",
        "ALoss 公式为 `(1/6) * sum_m ||normalize(A_m+I)[train,train] - normalize(A_label+I)[train,train]||_F^2`。六个真实图均在完整 598 节点图上归一化后取训练患者×训练患者子块，监督包含对角线，标签图只由该折训练标签构造。M1 在十折初始梯度审计时均无非对角边，记为 `STRUCTURAL_ZERO_GRADIENT`。M2–M6 均有有效边及有限梯度。",
        "",
        "克隆真实 Adam 一阶/二阶状态分别计算历史损失和加入 ALoss 的完整更新向量。表中是十折均值，包含两种更新范数、差向量范数、相对差和余弦。GraphLearn 列统计六个 `p.weight`。",
        "",
        "| 组/审计 epoch | 全量 base 范数 | 全量 supervised 范数 | 全量差范数 | 全量相对差 | 全量 cosine | Graph 差范数 | Graph 相对差 | Graph cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["N1", "N2", "N3", "N4", "N5"]:
        item = gradients["adam_update_vector_summary"][group]
        m = item["metrics"]
        out.append(
            f"| {group} / {item['epoch']} | {m['u_base_norm']['mean']:.6f} | {m['u_supervised_norm']['mean']:.6f} "
            f"| {m['u_difference_norm']['mean']:.6f} | {m['u_difference_relative']['mean']:.6f} | {m['u_cosine']['mean']:.9f} "
            f"| {m['graph_p_weight_u_difference_norm']['mean']:.6f} | {m['graph_p_weight_u_difference_relative']['mean']:.6f} "
            f"| {m['graph_p_weight_u_cosine']['mean']:.9f} |"
        )

    out += [
        "",
        "N5 十折 λ 均未触发裁剪，λ 均值约 0.0007594112，范围 0.0007208253–0.0008708068，实际初始梯度比例约 0.1。该 ALoss 比例与 DPSL `shared_cap=0.001` 是不同量。",
        "",
        "### DARG、DPSL 与患者级变化",
        "",
        "70/70 Best 检查均有正负 signed 关系，Positive-DCR 路径可调用，DARG 关系参数梯度非零。DPSL 在 N0–N5 的三对类别证据均参与，参数梯度有效，Residual-Off 与同状态 Native 输出逐折精确一致。共享梯度预算最大值低于 0.001。残差配置上限为 0.08，实测最大 0.0800000057，差异为 FP32 舍入。Direct-Off 和 Disease-Neutral 干预均未改变 argmax，logit 差异记录在患者机制审计。",
        "",
        "| G4 组 | Native 正确 | Full 正确 | Native→Full 纠正 | Native→Full 误伤 | NC/MCI/AD Native | NC/MCI/AD Full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    direct_rows = {row["group"]: row for row in patient_rows if row["comparison"] == "" and row["native_correct"] != ""}
    for group in ["N0", "N1", "N2", "N3", "N4", "N5"]:
        row = direct_rows[group]
        out.append(
            f"| {group} | {row['native_correct']} | {row['full_correct']} | {row['native_to_full_corrected']} "
            f"| {row['native_to_full_damaged']} | {row['NC_correct_native']}/{row['MCI_correct_native']}/{row['AD_correct_native']} "
            f"| {row['NC_correct_full']}/{row['MCI_correct_full']}/{row['AD_correct_full']} |"
        )
    comparisons = [row for row in patient_rows if row["group"] == "N3" and row["comparison"] in {"N3_vs_N0", "N3_vs_H4"}]
    if len(comparisons) != 2:
        raise RuntimeError("Expected N3 patient comparisons against N0 and historical H4")
    out += [
        "",
        f"N3 与 N0 及历史 H4 的患者级跨模型比较各自均为纠正 {comparisons[0]['patient_corrected']}、误伤 {comparisons[0]['patient_damaged']}、净变化 {comparisons[0]['patient_net']}。这是训练后模型比较，不等同于同一模型 Native→Full 的直接推理纠错。N3 Native→Full 为 1 例纠正、0 例误伤。",
        "",
        "### 基准、恢复与结论",
        "",
        "N0 与 H4 的十折 Best epoch、Warmup 历史、Warmup 后状态、预测和折指标均逐折匹配。历史 H4 未保存初始参数/优化器/RNG SHA，因此初始化身份只能记为 `PARTIAL_MATCH`，没有补造证据。",
        "",
        "| 配对 | ΔACC | ΔAUC_original | 患者纠正/误伤/净变化 |",
        "|---|---:|---:|---:|",
        "| N3 − N0 | +0.3333 pp | +0.0853 pp | 25 / 23 / +2 |",
        "| N3 − H4 | +0.3333 pp | +0.0853 pp | 25 / 23 / +2 |",
        "| N2 − N4（Warmup 时机） | −0.5056 pp | +0.1524 pp | 21 / 24 / −3 |",
        "| N2 − N6（G4/DPSL 与 G3） | +0.4972 pp | +0.7585 pp | 22 / 19 / +3 |",
        "| N1 − N2（ALoss 强度） | +0.6723 pp | +0.4975 pp | 26 / 22 / +4 |",
        "| N2 − N3（ALoss 强度） | −1.3418 pp | −0.5456 pp | 18 / 26 / −8 |",
        "",
        "N3 的 ACC 与 AUC_original 均超过本轮 N0 和历史 H4，满足登记的双指标性能候选门槛。N1/N2/N4/N5 未同时超过两个基准。原论文 TADPOLE 为 ACC 94.81±2.95%、AUC 95.59±2.57%。单 seed、固定十折和历史测试选点协议下，这些结果为描述性比较，不构成独立泛化证明。",
        "",
        f"Best 独立重放 {final['best_replay_count']}/70，Latest RNG 重放 {final['latest_rng_replay_count']}/70，Best 下一步见证 {final['best_next_step_witness_count']}/70，Latest 下一步见证 {final['latest_next_step_witness_count']}/70。恢复审计 140 个下一步见证 PASS，Best RNG sidecar SHA-256 `{recovery['best_recovery_rng_sidecar_sha256']}`。",
        "",
        "首次完整 checkpoint replay 已达 70/70，但报告聚合器因缺少 ACC 字段失败。随后只修改独立审计/报告脚本，修复十折 Adam 更新向量汇总和 CORR signed relation 计数语义，并从冻结 checkpoint 重建报告。训练 seal、训练源码、配置、数据、Best/Latest checkpoint 与预测均未修改。相关脚本哈希和范围见 `audits/report_repair.json`。",
        "",
        "正式结果、逐折对比、OOF 预测、患者机制、DARG/DPSL 机制、训练封存、启动预检、恢复与最终审计分别见 `results/group_summary.csv`、`results/fold_metrics.csv`、`results/oof_predictions.csv`、`results/paired_contrasts.csv`、`results/patient_mechanism.csv`、`results/darg_dpsl_fold_mechanism.csv` 和 `audits/` 下对应文件。",
        "",
        "受限恢复命令会检查已有 fold audit/checkpoint 身份，并只运行指定组和折。",
        "",
        "```bash",
        "cd /model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair",
        "PYTHONDONTWRITEBYTECODE=1 python -B code/run_e8382.py --groups N0 N1 N2 N3 N4 N5 N6 --workers 8",
        "PYTHONDONTWRITEBYTECODE=1 python -B audits/audit_finalize_e8382.py",
        "PYTHONDONTWRITEBYTECODE=1 python -B audits/repair_final_artifacts_e8382.py",
        "```",
        "",
        "科学结论为 ALoss 的固定公共系数在 N1–N3 产生约 0.13、0.40、1.32 倍历史图参数梯度，N3 在本轮唯一通过双指标门槛。Warmup-Calibrated N5 将 epoch 11 目标初始梯度比例稳定在 0.1，但均值指标没有超过 N0/H4。DPSL 的推理残差在本轮带来少量同模型直接纠错，而 N2 与 N6 的差异反映包含联合训练效应的组间变化。G4 · E82.4.6.10-G1 的 DARG + DPSL 仍是研究主线，N3 是登记门槛下的性能候选。",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    historical_manifest = build_historical_checkpoint_manifest()
    historical_manifest_path = ROOT / "audits/historical_checkpoint_manifest.json"
    historical_manifest_bytes = (json.dumps(historical_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    historical_manifest_path.write_bytes(historical_manifest_bytes)
    section = make_section(sha256_bytes(historical_manifest_bytes)).encode("utf-8")

    record_path = ROOT / "audits/handoff_append.json"
    with HANDOFF.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        old_bytes = stream.read()
        old_sha = sha256_bytes(old_bytes)
        if old_sha != EXPECTED_HANDOFF_SHA256:
            raise RuntimeError(f"handoff changed before append: {old_sha}")
        if SECTION_MARKER.encode("utf-8") in old_bytes:
            raise RuntimeError("E83.8.2 handoff section already exists")
        append_bytes = (b"\n\n" if old_bytes and not old_bytes.endswith(b"\n") else b"\n") + section
        stream.seek(0, os.SEEK_END)
        stream.write(append_bytes)
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        reread = stream.read()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    marker_count = reread.count(SECTION_MARKER.encode("utf-8"))
    if reread != old_bytes + append_bytes or marker_count != 1:
        raise RuntimeError("handoff append verification failed")
    metadata = {
        "experiment": "Experiment 83.8.2",
        "status": "APPENDED_AND_READ_BACK_VERIFIED",
        "handoff_path": str(HANDOFF),
        "preserved_prefix_bytes": len(old_bytes),
        "previous_handoff_sha256": old_sha,
        "appended_handoff_sha256": sha256_bytes(reread),
        "section_marker_count": marker_count,
        "prefix_verified": True,
        "historical_checkpoint_manifest": str(historical_manifest_path),
        "historical_checkpoint_manifest_sha256": sha256_bytes(historical_manifest_bytes),
        "append_script_sha256": sha256_file(Path(__file__)),
        "appended_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
