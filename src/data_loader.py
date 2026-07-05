"""
data_loader.py — load CMAPSS, compute RUL, stratified sampling, snapshots.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_DIR

COLS = (["engine_id", "cycle"]
        + [f"setting{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)])


def load(subset: str):
    """subset in {'FD001','FD003'}. Returns (df_with_rul, max_cycle_series)."""
    path = Path(DATA_DIR) / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CMAPSS and place train_{subset}.txt in {DATA_DIR}/. "
            f"See README for the link.")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    maxc = df.groupby("engine_id")["cycle"].max()
    df = df.merge(maxc.rename("max_cycle"), on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    return df, maxc


def stratified_engines(maxc, n, seed):
    """Sample n engines stratified by total life (short/medium/long)."""
    rng = np.random.default_rng(seed)
    tl = {int(e): int(maxc[e]) for e in maxc.index}
    buckets = [
        [e for e, t in tl.items() if t <= 150],
        [e for e, t in tl.items() if 150 < t <= 250],
        [e for e, t in tl.items() if t > 250],
    ]
    out, per = [], max(1, n // 3)
    for b in buckets:
        if b:
            out += list(rng.choice(b, min(per, len(b)), replace=False))
    return sorted(int(e) for e in out)[:n]


def snapshot(df, maxc, eid, frac):
    """Sensor snapshot at `frac` of engine life. Returns (cycle, rul, sensors)."""
    eng = df[df.engine_id == eid]
    target = int(maxc[eid] * frac)
    row = eng.iloc[(eng["cycle"] - target).abs().argsort().iloc[0]]
    rul = max(0, int(maxc[eid] - row["cycle"]))
    sensors = {f"s{i}": round(float(row[f"s{i}"]), 4) for i in range(1, 22)}
    return int(row["cycle"]), rul, sensors


def life_walk(df, maxc, eid, step_every):
    """Yield (step, cycle, rul, sensors) walking the engine's life every Nth cycle."""
    eng = df[df.engine_id == eid].sort_values("cycle")
    life = int(maxc[eid])
    cycles = list(range(1, life + 1, step_every))
    for step, cyc in enumerate(cycles, 1):
        row = eng.iloc[(eng["cycle"] - cyc).abs().argsort().iloc[0]]
        rul = max(0, int(maxc[eid] - row["cycle"]))
        sensors = {f"s{i}": round(float(row[f"s{i}"]), 4) for i in range(1, 22)}
        yield step, len(cycles), int(row["cycle"]), rul, sensors
