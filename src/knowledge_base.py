"""
knowledge_base.py — deterministic KB tool for Agent 2.
Pure function: no LLM, returns only values from config thresholds.
The kbCallLog (list of these responses) is the forensic audit trail SFS-1 checks.
"""
import json
from config import KB_HPC, KB_FAN, STANDARDS


def query_kb(sensor: str, fault_type: str) -> dict:
    """
    Retrieve verified thresholds, priority rules, or procedure IDs.
    sensor:     a sensor id ('s3') or fault type ('HPC_DEG') or 'RUL'
    fault_type: one of all_thresholds | HPC_DEG | FAN_DEG | priority | procedure
    """
    ft = (fault_type or "").lower()

    if ft in ("all_thresholds", "hpc_deg"):
        return {"source": STANDARDS["hpc_thresholds"],
                "data": {k: {"name": v["name"], "normal_min": v["min"],
                             "normal_max": v["max"], "unit": v["unit"]}
                         for k, v in KB_HPC.items()}}
    if ft == "fan_deg":
        return {"source": STANDARDS["fan_thresholds"],
                "data": {k: {"name": v["name"], "normal_min": v["min"],
                             "normal_max": v["max"], "unit": v["unit"]}
                         for k, v in KB_FAN.items()}}
    if ft == "priority":
        return {"source": STANDARDS["priority"],
                "data": {"rule": "RUL<=30 CRITICAL; 31-70 HIGH; 71-130 MEDIUM; >130 LOW"}}
    if ft == "procedure":
        return {"source": STANDARDS["procedure_hpc"],
                "data": {"procedure_id": "cmapss_proc_borescope_001",
                         "steps": "borescope inspection, compressor wash, tip clearance measurement"}}
    return {"source": "", "data": {}}


# OpenAI tool spec for the function-calling loop
KB_TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "query_kb",
        "description": ("Retrieve verified KB thresholds, priority rules, or procedure IDs. "
                        "MUST be called before diagnosing. Returns JSON with exact values."),
        "parameters": {
            "type": "object",
            "properties": {
                "sensor": {"type": "string",
                           "description": "sensor id (e.g. s3) or fault type (HPC_DEG, FAN_DEG, RUL)"},
                "fault_type": {"type": "string",
                               "enum": ["all_thresholds", "HPC_DEG", "FAN_DEG", "priority", "procedure"]},
            },
            "required": ["sensor", "fault_type"],
        },
    },
}]


def kb_reference_text(kb_log: list) -> str:
    """Flatten kbCallLog responses into a string for SFS-4 embedding comparison."""
    parts = []
    for entry in kb_log:
        resp = entry.get("response", {})
        data = resp.get("data", {})
        src = resp.get("source", "")
        if isinstance(data, dict):
            for sid, v in data.items():
                if isinstance(v, dict):
                    parts.append(f"{sid} {v.get('name','')} normal range "
                                 f"{v.get('normal_min','')} to {v.get('normal_max','')} "
                                 f"{v.get('unit','')} source {src}")
                else:
                    parts.append(f"{sid} {v} {src}")
    return " ".join(parts)
