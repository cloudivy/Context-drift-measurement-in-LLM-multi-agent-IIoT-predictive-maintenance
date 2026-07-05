"""
agents.py — the four-agent pipeline.

Agent 1  Sensor Monitor      deterministic, raw readings only
Agent 2  Diagnosis Agent     LIVE GPT-4o, query_kb tool-use loop
Agent 3  Maintenance Planner LIVE GPT-4o
Agent 4  Drift Validator     deterministic (see drift_validator.py)
"""
import json, time
from config import OPENAI_MODEL, TEMPERATURE, MAX_TOOL_ITERS, rul_priority
from knowledge_base import query_kb, KB_TOOL_SPEC

HPC_S = ["s3", "s4", "s7", "s11", "s12"]
FAN_S = ["s2", "s8"]

DIAG_SYS = (
    "You are the Diagnosis Agent in a turbofan predictive-maintenance pipeline. "
    "Before every diagnosis you MUST call query_kb to retrieve current thresholds — "
    "never rely on memory. Compare each sensor reading to the retrieved KB threshold. "
    "State: fault mode, breached sensors with values vs KB thresholds, priority (from RUL), "
    "procedure ID, and cite the source standard (e.g. NASA TM-2007-215026)."
)
DIAG_SYS_NOKB = (
    "You are the Diagnosis Agent in a turbofan predictive-maintenance pipeline. "
    "Diagnose the engine from the sensor readings and your own knowledge. "
    "State fault mode, affected sensors, priority, and a recommended procedure."
)
PLAN_SYS = (
    "You are the Maintenance Planner. Begin with 'Based on the Diagnosis Agent findings'. "
    "Give fault-specific actions and urgency matching the priority "
    "(CRITICAL=immediately, HIGH=within 48 hours, MEDIUM=within 2 weeks, LOW=next scheduled)."
)


def agent1_sensor_report(sensors: dict, faults: list) -> str:
    """Deterministic. Raw values only — no thresholds, no flags."""
    ids = HPC_S + (FAN_S if "FAN_DEG" in faults else [])
    return ", ".join(f"{s}={sensors[s]}" for s in ids)


def agent2_diagnose(client, conversation, sensors, rul, cycle, faults,
                    kb_on=True, step=None, total=None):
    """
    LIVE GPT-4o. Runs the query_kb tool-use loop.
    `conversation` is mutated in place — pass a persistent list for the
    accumulate arm (Block B), a fresh list for single-pass / fresh arm.
    Returns (diagnosis_text, kb_log).
    """
    report = agent1_sensor_report(sensors, faults)
    tag = f"[Step {step}/{total}] " if step else ""
    conversation.append({"role": "user",
                         "content": f"{tag}Engine cycle {cycle}, RUL {rul}. Readings: {report}. Diagnose."})

    kb_log = []
    sys = DIAG_SYS if kb_on else DIAG_SYS_NOKB
    for _ in range(MAX_TOOL_ITERS):
        kwargs = dict(model=OPENAI_MODEL, temperature=TEMPERATURE,
                      messages=[{"role": "system", "content": sys}] + conversation)
        if kb_on:
            kwargs.update(tools=KB_TOOL_SPEC, tool_choice="auto")
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        if getattr(msg, "tool_calls", None):
            conversation.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                resp_kb = query_kb(args.get("sensor", ""), args.get("fault_type", ""))
                kb_log.append({"sensor": args.get("sensor"),
                               "fault_type": args.get("fault_type"),
                               "response": resp_kb})
                conversation.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(resp_kb)})
        else:
            conversation.append({"role": "assistant", "content": msg.content or ""})
            return (msg.content or ""), kb_log

    return conversation[-1].get("content", ""), kb_log


def agent3_plan(client, diagnosis, rul):
    """LIVE GPT-4o. Produces the maintenance plan."""
    priority = rul_priority(rul)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": PLAN_SYS},
                  {"role": "user",
                   "content": f"Diagnosis: {diagnosis}\nRUL={rul}, priority={priority}. "
                              f"Produce the maintenance plan."}])
    return resp.choices[0].message.content or ""
