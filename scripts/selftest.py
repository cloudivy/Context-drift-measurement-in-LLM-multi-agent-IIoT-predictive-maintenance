#!/usr/bin/env python3
"""
selftest.py — validate all deterministic logic WITHOUT needing the API or the
embedding model. Run this first to confirm the repo is wired correctly.

    python scripts/selftest.py

Checks: KB tool, drift validator on grounded/drifted/partial cases, data loader.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_base import query_kb, kb_reference_text
from drift_validator import validate

def approx(a, b, t=0.01): return abs(a - b) <= t

print("1. KB tool")
r = query_kb("HPC_DEG", "all_thresholds")
assert r["source"].startswith("NASA"), "KB source wrong"
assert r["data"]["s3"]["normal_max"] == 1592.0, "s3 threshold wrong"
print("   query_kb OK — s3 max =", r["data"]["s3"]["normal_max"])

print("2. Drift validator — grounded case")
kb_log = [{"sensor": "HPC_DEG", "fault_type": "all_thresholds",
           "response": query_kb("HPC_DEG", "all_thresholds")}]
diag = ("HPC_DEG confirmed. s3=1604 exceeds KB max 1592 per NASA TM-2007-215026. "
        "s4=1428 exceeds 1415. Priority CRITICAL. cmapss_proc_borescope_001.")
plan = "Based on the Diagnosis Agent findings, borescope inspection immediately — immediate grounding. compressor wash."
g = validate(diag, plan, kb_log, 18, ["HPC_DEG"])
print(f"   SFS-1..4 = {g['sfs_1']},{g['sfs_2']},{g['sfs_3']},{g['sfs_4']}  "
      f"SFS={g['sfs']:.2f} IGS={g['igs']:.2f} ASI={g['asi']:.2f} drift={g['drift']}")
assert g["sfs_1"] == 1.0 and g["sfs_2"] == 1.0 and g["sfs_3"] == 1.0, "grounded SFS keywords failed"
assert not g["drift"], "grounded case should not drift"

print("3. Drift validator — drifted case (no KB, vague)")
d2 = validate("Temperature elevated, compressor wear likely. Inspection advised.",
              "Based on findings, schedule general inspection.", [], 18, ["HPC_DEG"])
print(f"   SFS-1..4 = {d2['sfs_1']},{d2['sfs_2']},{d2['sfs_3']},{d2['sfs_4']}  "
      f"ASI={d2['asi']:.2f} drift={d2['drift']}")
assert d2["sfs_1"] == 0.0, "no-KB should give SFS-1=0"
assert d2["drift"], "drifted case should flag drift"

print("4. Drift validator — partial (KB queried, vague output)")
d3 = validate("HPC degradation noted per NASA TM-2007-215026. Temperatures rising.",
              "Based on the Diagnosis Agent findings, borescope within 48 hours.",
              kb_log, 55, ["HPC_DEG"])
print(f"   SFS-2={d3['sfs_2']} (expect 0.0, no 2+ sensors)  ASI={d3['asi']:.2f}")
assert d3["sfs_2"] == 0.0, "partial case sensor citation wrong"

print("5. Data loader (only if data present)")
try:
    from data_loader import load, stratified_engines, snapshot
    df, mx = load("FD001")
    eids = stratified_engines(mx, 6, 42)
    cyc, rul, s = snapshot(df, mx, eids[0], 0.85)
    print(f"   FD001 loaded: {df.engine_id.nunique()} engines; "
          f"sample E{eids[0]} @85% -> cycle {cyc}, RUL {rul}, s3={s['s3']}")
except FileNotFoundError:
    print("   data not downloaded yet — run scripts/get_data.py (not required for this test)")

print("\nALL DETERMINISTIC CHECKS PASSED.")
print("Next: set OPENAI_API_KEY, run scripts/get_data.py, then src/run_block_a.py --smoke")
