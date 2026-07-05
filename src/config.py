"""
config.py — central configuration for the context-drift experiment.
Edit values here; every other module imports from this file.
"""
import os

# ── Model ─────────────────────────────────────────────────────
OPENAI_MODEL   = os.environ.get("DRIFT_MODEL", "gpt-4o-2024-08-06")
TEMPERATURE    = float(os.environ.get("DRIFT_TEMP", "0.2"))
MAX_TOOL_ITERS = 12          # max query_kb loop iterations in Agent 2

# ── Embedding model for SFS-4 / IGS-3 ─────────────────────────
EMBED_MODEL    = os.environ.get("DRIFT_EMBED", "all-MiniLM-L6-v2")

# ── Drift threshold ───────────────────────────────────────────
TAU = 0.75                   # ASI < TAU  => drift flagged

# ── Knowledge base: standards-grounded thresholds ─────────────
# HPC degradation sensors (FD001 + FD003)
KB_HPC = {
    "s3":  {"name": "HPC outlet temperature", "min": 1585.0, "max": 1592.0, "unit": "R",        "dir": "rising"},
    "s4":  {"name": "LPT outlet temperature", "min": 1400.0, "max": 1415.0, "unit": "R",        "dir": "rising"},
    "s7":  {"name": "HPC outlet pressure",    "min": 549.0,  "max": 554.0,  "unit": "psia",     "dir": "falling"},
    "s11": {"name": "HPC static pressure",    "min": 47.0,   "max": 48.5,   "unit": "psia",     "dir": "falling"},
    "s12": {"name": "fuel flow ratio",        "min": 521.0,  "max": 524.0,  "unit": "pps/psia", "dir": "rising"},
}
# Fan degradation sensors (FD003 only)
KB_FAN = {
    "s2":  {"name": "fan inlet temperature",  "min": 518.0,  "max": 522.0,  "unit": "R",  "dir": "falling"},
    "s8":  {"name": "bypass ratio",           "min": 8.4,    "max": 8.7,    "unit": "",   "dir": "falling"},
}

STANDARDS = {
    "hpc_thresholds": "NASA TM-2007-215026 / ISO 13379-1",
    "fan_thresholds": "AGARD-R-785",
    "priority":       "ISO 13381-1",
    "procedure_hpc":  "FAA AC 43.13-1B / cmapss_proc_borescope_001",
    "procedure_fan":  "FAA AC 43.13-1B / cmapss_proc_fan_inspection_001",
}

# RUL -> priority mapping
def rul_priority(rul: int) -> str:
    if rul <= 30:  return "CRITICAL"
    if rul <= 70:  return "HIGH"
    if rul <= 130: return "MEDIUM"
    return "LOW"

# Recognised source tokens (for SFS-3 keyword check)
KNOWN_STANDARDS = [
    "nasa tm-2007-215026", "iso 13381-1", "iso 13379-1", "faa ac 43.13-1b",
    "sae ja1012", "agard-r-785", "cmapss_proc_borescope_001",
    "cmapss_proc_fan_inspection_001",
]

# Urgency vocabulary per priority (for IGS-2)
URGENCY = {
    "CRITICAL": ["immediately", "immediate grounding", "no further flight"],
    "HIGH":     ["within 48", "48 hours", "two days"],
    "MEDIUM":   ["within 2 weeks", "2 weeks", "14 days"],
    "LOW":      ["next scheduled", "scheduled maintenance"],
}
# Fault-specific action vocabulary (for IGS-1)
FAULT_ACTIONS = {
    "HPC_DEG": ["borescope", "compressor wash", "compressor inspection", "tip clearance"],
    "FAN_DEG": ["fan blade", "fan inspection", "bypass ratio", "fan track"],
}

# ── Experiment sampling ───────────────────────────────────────
BLOCK_A_ENGINES = 12         # engines for S1/S2 (stratified)
BLOCK_A_FD003   = 10         # engines for S4
BLOCK_A_RUNS    = 3          # repeats per engine (for variance)
LIFE_FRACTION   = 0.85       # sampling point for single-pass scenarios

BLOCK_B_ENGINES = 3          # one per life bucket
BLOCK_B_STEP    = 5          # sample every Nth cycle in the long-horizon walk

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
FIG_DIR     = "figures"
