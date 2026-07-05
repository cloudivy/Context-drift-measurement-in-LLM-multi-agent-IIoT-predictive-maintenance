# Measuring Context Drift in LLM-based Multi-Agent Systems for IIoT Predictive Maintenance

Experiment code for the four-agent CMAPSS predictive-maintenance pipeline and the
seven-check Drift Validator (SFS-1..4, IGS-1..3, ASI).

## What this runs

**Block A — single-pass scenarios**
- **S1** baseline (FD001, KB on)
- **S2** KB-disabled ablation (FD001, KB off) — proves the metric detects grounding, not plausibility
- **S4** multi-fault (FD003, HPC+FAN)

**Block B — long-horizon context drift**
- One engine walked cycle-by-cycle across its whole life
- Two arms: `accumulate` (context carries all prior steps) vs `fresh` (clean context each step)
- Tests whether grounding decays as context accumulates

## The checks

```
SFS = mean(SFS-1 KB queried, SFS-2 sensors cited, SFS-3 standard cited, SFS-4 KB↔diagnosis embedding sim)
IGS = mean(IGS-1 fault actions, IGS-2 urgency alignment, IGS-3 diagnosis↔plan embedding sim)
ASI = mean(SFS, IGS);  drift flagged when ASI < 0.75
```

Agents 2 and 3 are **live GPT-4o**. Agents 1 and 4 are deterministic.
SFS-4 and IGS-3 use **real sentence embeddings** (`all-MiniLM-L6-v2`).

## Setup

```bash
git clone <your-repo-url>
cd cmapss-context-drift
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt

# 1. get the data
python scripts/get_data.py

# 2. validate the deterministic core (no API/model needed)
python scripts/selftest.py

# 3. set your key
export OPENAI_API_KEY=sk-...
```

## Run

```bash
# smoke test first — tiny, ~a dollar, confirms wiring end to end
python src/run_block_a.py --smoke
python src/run_block_b.py --smoke

# full runs
python src/run_block_a.py          # S1, S2, S4
python src/run_block_b.py          # long-horizon, both arms

# figures + summary
python src/analyze.py
```

## CRITICAL — before writing "embedding-based" in the paper

Every runner prints `Embedding backend: embedding` or `tfidf` on startup.
- `embedding` → SFS-4/IGS-3 used real sentence embeddings. You may write "embedding-based".
- `tfidf` → the model failed to load (no internet / install issue). Fix it and rerun.
  Do **not** describe SFS-4/IGS-3 as embedding-based for a tfidf run.

## Reading Block B

- If **accumulate** shows a negative ASI slope and negative `corr(context_len, ASI)`
  while **fresh** stays flat → context accumulation causes drift. That's the headline.
- If **both** stay flat → GPT-4o holds grounding over the horizon. Also a valid,
  publishable finding. Let the data decide; don't pre-commit to the drift story.

## Outputs

- `results/block_a.csv`, `results/block_b.csv` — per-run metrics (send these back for analysis)
- `figures/*.png` — publication figures

## Config

All settings (model, τ, thresholds, sample sizes) live in `src/config.py`.

## Data

CMAPSS from NASA: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
`scripts/get_data.py` pulls FD001/FD003 from a public mirror.

## Cost

Block A ≈ 100 pipeline runs; Block B ≈ 2 arms × 3 engines × (life/step) steps.
Budget a few USD and 20–40 min on gpt-4o. Start with `--smoke`.

## Security

Rotate any API key you paste into a terminal or share. Never commit keys — this repo
reads `OPENAI_API_KEY` from the environment only.
