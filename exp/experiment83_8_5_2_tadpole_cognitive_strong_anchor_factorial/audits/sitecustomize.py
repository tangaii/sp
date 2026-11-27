"""Audit-only serializer adapter loaded through PYTHONPATH."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code" / "common"))
import fold_checkpoint

_original_write_json = fold_checkpoint.write_json

def _sanitize_mechanism(value):
    if not isinstance(value, dict):
        return value
    result = {
        "edge_diagnostics": value.get("edge_diagnostics", []),
        "direct_scale": float(value.get("direct_scale", 0.0)),
        "positive_only_direct": bool(value.get("positive_only_direct", False)),
        "intervention": value.get("intervention"),
        "fused_delta_relative_l2": float(value.get("fused_delta_relative_l2", 0.0)),
        "audit_representation": "primitive summary of the independent Best-checkpoint DARG diagnostic pass",
    }
    gates = value.get("gates", [])
    result["gates"] = [float(item.detach().cpu()) if hasattr(item, "detach") else float(item)
                       for item in gates]
    return result

def _safe_write_json(path, value):
    if Path(path).name.startswith("fold") and Path(path).name.endswith(".json") and isinstance(value, dict):
        if "graph_mechanism" in value:
            value = dict(value)
            value["graph_mechanism"] = _sanitize_mechanism(value["graph_mechanism"])
    return _original_write_json(path, value)

fold_checkpoint.write_json = _safe_write_json

