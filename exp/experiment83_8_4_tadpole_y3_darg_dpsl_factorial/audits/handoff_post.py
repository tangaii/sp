"""Append the audited Experiment 83.8.4 result to the shared handoff."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
MARKER = "## Experiment 83.8.4 — Y3 DARG/DPSL Factorial Optimization"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def percent(value):
    return f"{float(value) * 100:.4f}%"


def build_section(summary, analysis, final_audit, mechanism, seal, source_manifest, log_audit, replay):
    metrics = summary.set_index("group")
    metric_rows = []
    gate_rows = []
    for group in ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5"):
        row = metrics.loc[group]
        metric_rows.append(
            f"| {group} | {percent(row.ACC_mean)} ± {percent(row.ACC_population_std)} "
            f"| {percent(row.AUC_original_mean)} ± {percent(row.AUC_original_population_std)} "
            f"| {percent(row.AUC_probability_mean)} ± {percent(row.AUC_probability_population_std)} "
            f"| {int(row.OOF_correct)}/598 ({int(row.NC_correct)}/{int(row.MCI_correct)}/{int(row.AD_correct)}) |"
        )
        item = mechanism["groups"][group]
        modalities = item["darg_by_modality"].values()
        gate_mean = float(np.mean([value["gate_effective_mean"] for value in modalities]))
        message_mean = float(np.mean([value["gcn_message_norm_ratio_mean"] for value in modalities]))
        direct_mean = float(np.mean([value["weighted_direct_message_norm_ratio_mean"] for value in modalities]))
        gate_rows.append(
            f"| {group} | {gate_mean:.6f} | {message_mean:.6f} | {direct_mean:.6f} "
            f"| {item['direct_off_changed']} | {item['disease_neutral_changed']} |"
        )

    repos = source_manifest["repositories"]
    marker_bytes = MARKER.encode("utf-8")
    result_status = analysis["status"]
    next_group = analysis["next_group"]
    both = analysis.get("groups_both_primary_above_n3", [])
    log_groups = log_audit["groups"]
    lines = [
        "",
        "",
        MARKER,
        "",
        f"- **状态** `{final_audit['status'] == 'PASS' and 'COMPLETE_AUDITED' or final_audit['status']}`，科研判断 `{result_status}`。保留候选组 `{next_group}`。",
        f"- **唯一正式目录** `{ROOT}`。首选 `/data/adodas/SPromptGL-main/experiment/experiment83_8_4_tadpole_y3_darg_dpsl_factorial` 创建时返回 OSSFS EIO/errno 5，随后只使用 `{ROOT}`。",
        f"- **仓库来源** 原作者 `wanxixi11/SPromptGL` commit `{repos['author']['commit']}`；SABER 主项目保留请求锚点 `{repos['saber_main'].get('requested_e8383_anchor', 'cbaaf1c')}`，本地 manifest commit `{repos['saber_main'].get('local_manifest_commit', repos['saber_main'].get('commit'))}`；E83.8.2 来自独立仓库 `tangaii/sp` 固定 commit `{repos['experiment_repo']['commit']}`。论文 SHA-256 `{repos['paper'].get('sha256', source_manifest.get('paper_sha256', 'recorded in source_manifest'))}`。逐文件来源 SHA 在 `source_manifest.json`。",
        f"- **训练 seal** 实现 SHA `{seal['implementation_sha256']}`，seal 文件 SHA `{final_audit['training_seal_sha256']}`。数据 SHA `{final_audit['data_sha256']}`；split SHA `{final_audit['split_sha256']}`；Prompt manifest SHA `{source_manifest['training_inputs']['prompt_manifest_sha256']}`；Prompt 向量 SHA `{json.dumps(source_manifest['training_inputs']['prompt_files_sha256'], sort_keys=True)}`。",
        "- **锁定协议与资源** TADPOLE 598人，NC/MCI/AD `209/315/74`，标签 `0/1/2`，六模态维度 `145/148/3/36/24/9`，CAM 240维。保留全数据 RFE、均值填补、外层 CV 前 StandardScaler、转导患者图、val=test、StratifiedKFold 10折 random_state=10、seed 0、450 epoch、`wait > 200`、首次严格测试折 ACC Best、DARG 10 epoch warmup、MMGPL-12/RN50 冻结文本特征、DPSL 三类别对、残差上限 0.08、共享梯度上限 0.001。实际资源为单张 PPU-ZW810E（98 GB），总 worker 上限 8。",
        f"- **正式训练与回归** Z0 为历史 Y3 验证复用，Z1–Z5 各完成 10/10 个新训 fold，共 `{final_audit['formal_folds']}/60`。Z0 与 E83.8.3-Y3 的十折 Best/Latest、关系对、预测、概率和三项 AUC/ACC 精确相同。Best 重放 `{replay['best_replayed']}/60`，Latest 恢复 `{replay['latest_restored']}/60`，下一步确定性 `{replay['deterministic_next_steps']}/60`。",
        "",
        "| 组 | ACC mean ± population SD | AUC_original mean ± population SD | AUC_probability mean ± population SD | OOF正确 (NC/MCI/AD正确) |",
        "|---|---:|---:|---:|---:|",
        *metric_rows,
        "",
        f"- **固定参考** E83.8.2 G4/N0 为 ACC `92.9746%`、AUC_original `94.0967%`、AUC_probability `96.2710%`；E83.8.2-N3 为 `93.3079%`、`94.1819%`、`95.8360%`。论文参考 ACC `94.81±2.95%`、AUC `95.59±2.57%`。最高 ACC 组 `{analysis['highest_ACC_group']}`，最高 AUC_original 组 `{analysis['highest_AUC_original_group']}`，最高 probability AUC 组 `{analysis['highest_AUC_probability_group']}`。同时超过 N3 两项主指标的组 `{', '.join(both) if both else '无'}`。达到论文两项参考均值 `{analysis['paper_primary_means_reached']}`。",
        "- **DARG 门控机制** Z0/Z3 保留历史双重 gate，Z1/Z2/Z4/Z5 仅移除直接残差的第二次 gate，direct_scale 按预注册值为 0.02 或 0.10。六个 TADPOLE 模态均按实例配置隔离，RISK_FACTOR 未进入 CORR 特殊分支。signed 边、候选边、原始/统一图差、逐模态 gate、消息范数、关系梯度和 Direct-Off/Disease-Neutral 分类级结果见 `audits/mechanism_summary.json` 与 `results/patient_mechanism.csv`。",
        "",
        "| 组 | 六模态平均有效gate | GCN消息/原始范数 | Positive-DCR直接消息/原始范数 | Direct-Off改变人数 | Disease-Neutral改变人数 |",
        "|---|---:|---:|---:|---:|---:|",
        *gate_rows,
        "",
        "- **DPSL 置信保护** Z0/Z1/Z2 使用历史 `g=4*p1*p2`，Z3/Z4/Z5 使用 `g=4*p1*p2/((p1+p2)^2+1e-12)` 并限制在 [0,1]。概率来自 detached 原生 logits，不新增参数或标签输入。逐折 Native/Full ACC、AUC、纠正/误伤、NC/MCI/AD类别分解、margin-g 关系、高低置信度残差变化、三个类别对参与、最大残差和共享梯度预算已记录。组间 Native 变化作为联合训练结果，Native→Full 变化作为直接推理修正。",
        "- **ALoss** 六组均保留 N3 平方 Frobenius 公式，λ=0.010，epoch 11启用，使用训练折标签。作者日志第三项保留原论文模型的未平方六图平均观测，不替代本次训练 ALoss。",
        f"- **原论文格式日志** 作者模板 SHA `{log_audit['template_sha256']}`。Z0–Z5 每组一份日志，完整保留三项 `trn-loss-MF`、原字段顺序与精度，分母为真实 450 epoch。格式审计 `{len(log_groups)}/6 PASS`。逐组日志 SHA 为 `{json.dumps({g: log_groups[g]['log_sha256'] for g in sorted(log_groups)}, sort_keys=True)}`，逐 epoch 值、折边界、早停轮数、Best 测试指标、checkpoint SHA 与记录 SHA 见 `audits/log_manifest.json` 和 `audits/log_format_audit.json`。",
        "- **六组精确定义** Z0=双重 gate/0.02/old，Z1=单次 gate/0.02/old，Z2=单次 gate/0.10/old，Z3=双重 gate/0.02/multiclass，Z4=单次 gate/0.02/multiclass，Z5=单次 gate/0.10/multiclass。所有组保留 Y3 的六模态 GFR、CAM、RES、分类头、DARG signed edges/Positive-DCR/关系编码器/辅助损失、DPSL SCMR-DA-SRPE/医学 Prompt/三类对/谱库/优化器、MMGPL-12 冻结 RN50、平方 ALoss。",
        "- **成对统计** `paired_contrasts.csv` 包含 Z1–Z5 对 Z0、Z0–Z5 对 N3/G4、不同 DARG 门控、DPSL 保护和组合增量的逐折差值，以及患者 OOF 纠正/误伤和精确 McNemar 检验。折间训练集重叠且该数据已经历多轮选优，不把细小均值差异称为稳定泛化提升。",
        f"- **恢复命令** `python {ROOT}/code/run_e8384.py --groups Z1 Z2 Z3 Z4 Z5 --workers 8`。Z0 是已验证复用的历史产物。中断恢复使用每折 Latest 中的模型、optimizer、RNG、关系对、DPSL谱统计/EMA和逐 epoch 日志历史。",
        f"- **结果文件** `{ROOT}/results/group_summary.csv`、`fold_metrics.csv`、`oof_predictions.csv`、`paired_contrasts.csv`、`patient_mechanism.csv`、`final_report.md`。审计文件 `{ROOT}/audits/preflight_start.json`、`training_seal.json`、`checkpoint_replay.json`、`log_format_audit.json`、`final_audit.json`。",
        f"- **失败、限制与重试** 首选目录创建返回 OSSFS EIO/errno 5，已一次切换到实际目录。日志审计修复只放在 `audits/`，训练源码封印未改。Direct-Off/Disease-Neutral 在 Best 测试预测上均未产生翻转，新增组均未同时超过 Z0 的两项主指标。",
        f"- **最终判定** `NO-GO`，科研解释为 `{result_status}`，保留 `{next_group}` 作为正式基座。历史 G4 继续作为长期固定基准。下一阶段最值得研究的问题是如何提高 Positive-DCR 直接残差在不放大误伤的前提下对最终分类 margin 的可观测影响。",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main():
    if not HANDOFF.is_file():
        raise FileNotFoundError(HANDOFF)
    original = HANDOFF.read_bytes()
    marker_bytes = MARKER.encode("utf-8")
    occurrences = original.count(marker_bytes)
    if occurrences > 1:
        raise RuntimeError("E83.8.4 handoff marker appears more than once")
    if occurrences == 0:
        summary = pd.read_csv(ROOT / "results/group_summary.csv")
        analysis = json.loads((ROOT / "audits/analysis_summary.json").read_text())
        final_audit = json.loads((ROOT / "audits/final_audit.json").read_text())
        mechanism = json.loads((ROOT / "audits/mechanism_summary.json").read_text())
        seal = json.loads((ROOT / "audits/training_seal.json").read_text())
        source_manifest = json.loads((ROOT / "source_manifest.json").read_text())
        log_audit = json.loads((ROOT / "audits/log_format_audit.json").read_text())
        replay = json.loads((ROOT / "audits/checkpoint_replay.json").read_text())
        with HANDOFF.open("ab") as stream:
            stream.write(build_section(summary, analysis, final_audit, mechanism, seal,
                                       source_manifest, log_audit, replay))
    updated = HANDOFF.read_bytes()
    if not updated.startswith(original):
        raise RuntimeError("Handoff historical prefix changed during append")
    if updated.count(marker_bytes) != 1:
        raise RuntimeError("Handoff E83.8.4 marker is not unique")
    record = {
        "experiment": "Experiment 83.8.4",
        "status": "PASS",
        "section_title": MARKER,
        "handoff_path": str(HANDOFF),
        "prefix_bytes": len(original),
        "prefix_sha256": digest(original),
        "handoff_bytes": len(updated),
        "handoff_sha256": digest(updated),
        "marker_occurrences": updated.count(marker_bytes),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = ROOT / "audits/handoff_update.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if HANDOFF.read_bytes() != updated:
        raise RuntimeError("Handoff append did not survive read-back validation")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
