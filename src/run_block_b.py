"""
run_block_b.py — long-horizon context drift with LIVE GPT-4o.

One engine walked cycle-by-cycle across its whole life (the horizon).
Two arms:
  accumulate : the Diagnosis Agent carries its full conversation history
               across every step — context fills up. Drift may emerge here.
  fresh      : each step starts with a clean context. Control.

The key question: does grounding (ASI) decay as context accumulates —
and does it decay MORE in the accumulate arm than the fresh arm?

Usage:
    python run_block_b.py                 # full
    python run_block_b.py --smoke         # 2 engines, coarse sampling
    python run_block_b.py --step-every 3  # finer / longer horizon
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import BLOCK_B_ENGINES, BLOCK_B_STEP, RESULTS_DIR
from data_loader import load, stratified_engines, life_walk
from agents import agent2_diagnose, agent3_plan
from drift_validator import validate
from semantic import get_backend


def run(client, step_every, smoke):
    df1, m1 = load("FD001")
    eids = stratified_engines(m1, BLOCK_B_ENGINES, 45)
    if smoke:
        eids = eids[:2]
        step_every = max(step_every, 25)

    rows = []
    for arm in ["accumulate", "fresh"]:
        for eid in eids:
            conversation = []           # persists across steps in accumulate arm
            for step, total, cyc, rul, sensors in life_walk(df1, m1, eid, step_every):
                if arm == "fresh":
                    conversation = []   # reset every step = control
                diag, kb_log = agent2_diagnose(client, conversation, sensors, rul, cyc,
                                               ["HPC_DEG"], kb_on=True, step=step, total=total)
                plan = agent3_plan(client, diag, rul)
                mt = validate(diag, plan, kb_log, rul, ["HPC_DEG"])
                rows.append(dict(arm=arm, engine_id=eid, step=step, total=total,
                                 cycle=cyc, rul=rul, context_len=len(conversation), **mt))
                print(f"  {arm[:4]} E{eid} {step}/{total} RUL={rul} ctx={len(conversation)} "
                      f"KB={'Y' if kb_log else 'N'} SFS={mt['sfs']:.2f} ASI={mt['asi']:.2f} "
                      f"{'DRIFT' if mt['drift'] else ''}")
                time.sleep(0.2)

    B = pd.DataFrame(rows)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    B.to_csv(f"{RESULTS_DIR}/block_b.csv", index=False)

    # Per-engine arm comparison
    print(f"\n{'Arm':<12}{'Engine':>7}{'Steps':>6}{'SFS':>7}{'ASI':>7}"
          f"{'Drift%':>8}{'ASI_slope':>11}{'corr(ctx,ASI)':>15}")
    for arm in ["accumulate", "fresh"]:
        for eid in eids:
            d = B[(B.arm == arm) & (B.engine_id == eid)].sort_values("step")
            if len(d) > 2:
                slope = np.polyfit(d.step, d.asi, 1)[0]
                corr = (np.corrcoef(d.context_len, d.asi)[0, 1]
                        if d.context_len.std() > 0 else float("nan"))
                print(f"{arm:<12}{eid:>7}{len(d):>6}{d.sfs.mean():>7.3f}{d.asi.mean():>7.3f}"
                      f"{100*d.drift.mean():>7.1f}%{slope:>+11.4f}{corr:>+15.3f}")

    print("\nPooled by arm:")
    for arm in ["accumulate", "fresh"]:
        d = B[B.arm == arm]
        slopes = [np.polyfit(g.step, g.asi, 1)[0]
                  for _, g in d.groupby("engine_id") if len(g) > 2]
        print(f"  {arm:<11}: ASI={d.asi.mean():.3f}  drift={100*d.drift.mean():.1f}%  "
              f"mean slope/step={np.mean(slopes):+.4f}")

    print(f"\nSaved {RESULTS_DIR}/block_b.csv")
    print("\nINTERPRET: if accumulate shows negative slope + negative corr(ctx,ASI)")
    print("while fresh stays flat, that isolates context accumulation as the drift cause.")
    print("If both stay flat, GPT-4o holds grounding over the horizon — also a valid finding.")
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-every", type=int, default=BLOCK_B_STEP)
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

    print("=" * 68); print("BLOCK B — LIVE GPT-4o LONG-HORIZON"); print("=" * 68)
    run(client, args.step_every, args.smoke)


if __name__ == "__main__":
    main()
