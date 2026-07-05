"""
analyze.py — turn results/block_a.csv and results/block_b.csv into
publication figures and a printed summary. Run after the experiments.

    python analyze.py

Produces (in figures/):
  fig_blockA_signals.png    per-signal bars across S1/S2/S4
  fig_blockB_trajectory.png ASI(t) accumulate vs fresh, representative engine
  fig_blockB_arms.png       pooled ASI distribution accumulate vs fresh
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR, FIG_DIR, TAU

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"S1": "#185FA5", "S2": "#D85A30", "S4": "#0F6E56",
     "acc": "#D85A30", "fresh": "#185FA5", "tau": "#BA7517"}


def block_a():
    p = f"{RESULTS_DIR}/block_a.csv"
    if not os.path.exists(p):
        print("no block_a.csv — skipping"); return
    A = pd.read_csv(p)
    sigs = ["sfs_1", "sfs_2", "sfs_3", "sfs_4", "igs_1", "igs_2", "igs_3"]
    labels = ["SFS-1", "SFS-2", "SFS-3", "SFS-4", "IGS-1", "IGS-2", "IGS-3"]
    scen = ["S1", "S2", "S4"]
    x = np.arange(len(sigs)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, sc in enumerate(scen):
        d = A[A.scenario == sc]
        if not len(d): continue
        ax.bar(x + (i-1)*w, [d[s].mean() for s in sigs], w, label=sc, color=C[sc], alpha=0.85)
    ax.axhline(TAU, color=C["tau"], ls="--", lw=0.8, label=f"τ={TAU}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("signal value")
    ax.legend(ncol=4, fontsize=7); ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_title("Block A — per-signal breakdown across scenarios", fontsize=9)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/fig_blockA_signals.png"); plt.close()
    print(f"wrote {FIG_DIR}/fig_blockA_signals.png")


def block_b():
    p = f"{RESULTS_DIR}/block_b.csv"
    if not os.path.exists(p):
        print("no block_b.csv — skipping"); return
    B = pd.read_csv(p)

    # Trajectory for the longest-horizon engine
    eid = B.groupby("engine_id")["step"].max().idxmax()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for arm, col in [("accumulate", C["acc"]), ("fresh", C["fresh"])]:
        d = B[(B.arm == arm) & (B.engine_id == eid)].sort_values("step")
        if len(d):
            ax.plot(d.step, d.asi, "-o", ms=2.5, lw=1.2, color=col, label=f"{arm} ASI")
    ax.axhline(TAU, color=C["tau"], ls="--", lw=0.8, label=f"τ={TAU}")
    ax.set_xlabel("horizon step"); ax.set_ylabel("ASI"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7); ax.grid(lw=0.3, alpha=0.4)
    ax.set_title(f"Block B — ASI over horizon, engine {eid}", fontsize=9)
    fig.savefig(f"{FIG_DIR}/fig_blockB_trajectory.png"); plt.close()
    print(f"wrote {FIG_DIR}/fig_blockB_trajectory.png")

    # Arm distribution
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    data = [B[B.arm == "accumulate"].asi, B[B.arm == "fresh"].asi]
    bp = ax.boxplot(data, tick_labels=["accumulate", "fresh"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [C["acc"], C["fresh"]]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(TAU, color=C["tau"], ls="--", lw=0.8)
    ax.set_ylabel("ASI"); ax.set_ylim(0, 1.02); ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_title("Block B — ASI by arm", fontsize=9)
    fig.savefig(f"{FIG_DIR}/fig_blockB_arms.png"); plt.close()
    print(f"wrote {FIG_DIR}/fig_blockB_arms.png")

    # Printed stats
    print("\nBlock B correlation(context_len, ASI) by arm:")
    for arm in ["accumulate", "fresh"]:
        d = B[B.arm == arm]
        if d.context_len.std() > 0:
            print(f"  {arm:<11}: r = {np.corrcoef(d.context_len, d.asi)[0,1]:+.3f}")


if __name__ == "__main__":
    block_a()
    block_b()
    print("\nDone. Figures in", FIG_DIR)
