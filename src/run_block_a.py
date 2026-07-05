"""
run_block_a.py — single-pass scenarios with LIVE GPT-4o.
  S1  baseline        (FD001, KB on, HPC_DEG)
  S2  KB-disabled     (FD001, KB off) — the ablation
  S4  multi-fault     (FD003, KB on, HPC_DEG + FAN_DEG)

Usage:
    python run_block_a.py             # full
    python run_block_a.py --smoke     # 2 engines, 1 run
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (BLOCK_A_ENGINES, BLOCK_A_FD003, BLOCK_A_RUNS,
                    LIFE_FRACTION, RESULTS_DIR)
from data_loader import load, stratified_engines, snapshot
from agents import agent2_diagnose, agent3_plan
from drift_validator import validate
from semantic import get_backend
from scipy import stats


def run(client, runs, smoke):
    df1, m1 = load("FD001")
    df3, m3 = load("FD003")
    n1 = 2 if smoke else BLOCK_A_ENGINES
    n4 = 2 if smoke else BLOCK_A_FD003
    R = 1 if smoke else runs

    specs = [
        ("S1", df1, m1, stratified_engines(m1, n1, 42), ["HPC_DEG"], True),
        ("S2", df1, m1, stratified_engines(m1, n1, 42), ["HPC_DEG"], False),
        ("S4", df3, m3, stratified_engines(m3, n4, 44), ["HPC_DEG", "FAN_DEG"], True),
    ]
    rows = []
    for sc, df, mx, eids, faults, kb_on in specs:
        for eid in eids:
            cyc, rul, sensors = snapshot(df, mx, eid, LIFE_FRACTION)
            for r in range(R):
                conv = []
                diag, kb_log = agent2_diagnose(client, conv, sensors, rul, cyc, faults, kb_on=kb_on)
                plan = agent3_plan(client, diag, rul)
                mt = validate(diag, plan, kb_log, rul, faults)
                rows.append(dict(scenario=sc, engine_id=eid, run=r, rul=rul,
                                 diagnosis=diag[:400], plan=plan[:300], **mt))
                print(f"  {sc} E{eid} r{r} RUL={rul} KB={'Y' if kb_log else 'N'} "
                      f"SFS={mt['sfs']:.2f} IGS={mt['igs']:.2f} ASI={mt['asi']:.2f} "
                      f"{'DRIFT' if mt['drift'] else ''}")
                time.sleep(0.2)

    A = pd.DataFrame(rows)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    A.to_csv(f"{RESULTS_DIR}/block_a.csv", index=False)

    # Summary table
    print(f"\n{'Scen':<5}{'N':>4}{'SFS-1':>7}{'SFS-2':>7}{'SFS-3':>7}{'SFS-4':>7}"
          f"{'SFS':>7}{'IGS':>7}{'ASI':>7}{'Drift%':>8}")
    for sc in ["S1", "S2", "S4"]:
        d = A[A.scenario == sc]
        if len(d):
            print(f"{sc:<5}{len(d):>4}{d.sfs_1.mean():>7.3f}{d.sfs_2.mean():>7.3f}"
                  f"{d.sfs_3.mean():>7.3f}{d.sfs_4.mean():>7.3f}{d.sfs.mean():>7.3f}"
                  f"{d.igs.mean():>7.3f}{d.asi.mean():>7.3f}{100*d.drift.mean():>7.1f}%")

    # Ablation stats
    s1, s2 = A[A.scenario == "S1"], A[A.scenario == "S2"]
    if len(s1) > 1 and len(s2) > 1:
        print("\nAblation S1 vs S2 (Welch t-test):")
        for mtr in ["sfs", "igs", "asi"]:
            t, p = stats.ttest_ind(s1[mtr], s2[mtr], equal_var=False)
            pooled = np.sqrt((s1[mtr].std()**2 + s2[mtr].std()**2) / 2)
            d = (s1[mtr].mean() - s2[mtr].mean()) / pooled if pooled > 0 else float("inf")
            print(f"  {mtr.upper()}: t={t:.2f} p={p:.2e} Cohen_d={d:.2f}")

    print(f"\nSaved {RESULTS_DIR}/block_a.csv")
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=BLOCK_A_RUNS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"Embedding backend: {get_backend()}")
    if get_backend() != "embedding":
        print("  !! WARNING: TF-IDF fallback active. Do NOT call SFS-4/IGS-3 'embedding-based'.")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: set OPENAI_API_KEY"); return
    from openai import OpenAI
    client = OpenAI(api_key=key)

    print("=" * 68); print("BLOCK A — LIVE GPT-4o (S1, S2, S4)"); print("=" * 68)
    run(client, args.runs, args.smoke)


if __name__ == "__main__":
    main()
