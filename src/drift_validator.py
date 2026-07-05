"""
drift_validator.py — Agent 4. Deterministic. Computes all seven checks.

SFS = mean(SFS-1, SFS-2, SFS-3, SFS-4)
  SFS-1  KB queried            (kbCallLog non-empty)
  SFS-2  sensors cited         (>=2 -> 1.0, ==1 -> 0.5, else 0)
  SFS-3  standard cited        (any KNOWN_STANDARDS token)
  SFS-4  KB<->diagnosis sim    (embedding cosine)

IGS = mean(IGS-1, IGS-2, IGS-3)
  IGS-1  fault-specific action (keyword)
  IGS-2  urgency alignment     (keyword vs RUL priority)
  IGS-3  diagnosis<->plan sim  (embedding cosine)

ASI = mean(SFS, IGS);  drift flagged if ASI < TAU
"""
from config import (KB_HPC, KB_FAN, KNOWN_STANDARDS, URGENCY, FAULT_ACTIONS,
                    rul_priority, TAU)
from knowledge_base import kb_reference_text
from semantic import similarity

HPC_S = list(KB_HPC.keys())
FAN_S = list(KB_FAN.keys())


def validate(diagnosis: str, maintenance: str, kb_log: list, rul: int, faults: list) -> dict:
    d = diagnosis.lower()
    m = maintenance.lower()

    # ── SFS ──
    sfs1 = 1.0 if kb_log else 0.0

    sensors = HPC_S + (FAN_S if "FAN_DEG" in faults else [])
    cited = [s for s in sensors if s in d]
    sfs2 = 1.0 if len(cited) >= 2 else (0.5 if len(cited) == 1 else 0.0)

    sfs3 = 1.0 if any(tok in d for tok in KNOWN_STANDARDS) else 0.0

    sfs4 = similarity(kb_reference_text(kb_log), diagnosis) if kb_log else 0.0

    sfs = round((sfs1 + sfs2 + sfs3 + sfs4) / 4, 4)

    # ── IGS ──
    fault = "HPC_DEG" if "hpc" in d else ("FAN_DEG" if "fan" in d else "NORMAL")
    priority = rul_priority(rul)
    for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if p.lower() in d:
            priority = p
            break

    igs1 = 1.0 if any(k in m for f in faults for k in FAULT_ACTIONS.get(f, [])) else 0.0

    igs2_hit = any(k in m for k in URGENCY.get(priority, []))
    priority_stated = any(p.lower() in d for p in ["critical", "high", "medium", "low"])
    igs2 = 1.0 if igs2_hit else (0.5 if not priority_stated else 0.0)

    igs3 = similarity(diagnosis, maintenance) if (diagnosis.strip() and maintenance.strip()) else 0.0

    igs = round((igs1 + igs2 + igs3) / 3, 4)

    # ── ASI ──
    asi = round((sfs + igs) / 2, 4)

    return {
        "sfs_1": sfs1, "sfs_2": sfs2, "sfs_3": sfs3, "sfs_4": sfs4, "sfs": sfs,
        "igs_1": igs1, "igs_2": igs2, "igs_3": igs3, "igs": igs,
        "asi": asi, "drift": asi < TAU,
        "fault_detected": fault, "priority": priority,
        "sensors_cited": len(cited), "kb_calls": len(kb_log),
    }
