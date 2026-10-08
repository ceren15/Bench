"""
Aggregate per-step rows into the benchmark metrics and the weighted benchmark score.

Definitions (S = scored steps, T = tests):
  routing_accuracy      = |decision_ok| / |S|                              (call / no_tool / clarify)
  routing_macro_f1      = macro-F1 of the 3 decision classes                (robust to class imbalance)
  tool_selection_acc    = |tools_ok| / |S where gold = call|                (exact tool multiset)
  argument_accuracy     = Σ args_ok_cond / Σ args_total_cond                (only where the tool was selected)
  argument_acc_e2e      = Σ args_ok / Σ args_total                          (missing tool = all its args wrong)
  spurious_arg_rate     = Σ spurious_args / Σ predicted calls
  multi_tool_success    = |exact| / |S where gold has ≥2 calls|
  workflow_completion   = |T with ≥2 steps and every step exact| / |T with ≥2 steps|
  false_tool_call_rate  = |gold ∈ {no_tool, clarify} ∧ pred = call| / |S where gold ≠ call|
  missed_call_rate      = |gold = call ∧ pred ≠ call| / |S where gold = call|
  test_pass_rate        = |T where every step exact| / |T|
  ece                   = expected calibration error of action confidence vs. step exact (10 bins)
  latency p50/p95/p99   = model-call wall time per step (ms) — network included for hosted APIs
  throughput_rps        = steps / wall-clock seconds of the measured phase at the configured concurrency
  cost_per_1k_steps     = tokens × price  (or GPU-hours × hourly rate for self-hosted)
  cost_per_success      = total cost / |T passed|
  consistency           = share of (test, step) whose prediction is identical across repeats
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import median

DEFAULT_WEIGHTS = {
    "routing_accuracy": 0.20,
    "tool_selection_acc": 0.20,
    "argument_accuracy": 0.15,
    "multi_tool_success": 0.10,
    "workflow_completion": 0.15,
    "no_false_call": 0.10,   # = 1 - false_tool_call_rate
    "calibration": 0.10,     # = 1 - ece
}
DEFAULT_EFFICIENCY = {"latency_ms_best": 20, "latency_ms_worst": 3000, "cost_1k_best": 0.001, "cost_1k_worst": 5.0}
DEFAULT_GATES = {"false_tool_call_rate_max": 0.05, "routing_accuracy_min": 0.85}


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def _ratio(a, b):
    return (a / b) if b else None


def ece(conf_correct: list[tuple[float, bool]], bins=10):
    pts = [(c, ok) for c, ok in conf_correct if c is not None]
    if not pts:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pts if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if sel:
            acc = sum(ok for _, ok in sel) / len(sel)
            conf = sum(c for c, _ in sel) / len(sel)
            tot += len(sel) / len(pts) * abs(acc - conf)
    return tot


def coverage_at_precision(conf_correct, target=0.95):
    """Largest share of steps the system could handle autonomously (confidence ≥ τ) while keeping
    precision ≥ target. This is what a confidence gate buys you in production."""
    pts = sorted([(c, ok) for c, ok in conf_correct if c is not None], key=lambda x: -x[0])
    best, ok_n = 0.0, 0
    for i, (c, ok) in enumerate(pts, 1):
        ok_n += ok
        if ok_n / i >= target:
            best = i / len(pts)
    return best if pts else None


def macro_f1(rows):
    classes = ("call", "no_tool", "clarify")
    f1s = []
    for c in classes:
        tp = sum(1 for r in rows if r["exp_decision"] == c and r["pred_decision"] == c)
        fp = sum(1 for r in rows if r["exp_decision"] != c and r["pred_decision"] == c)
        fn = sum(1 for r in rows if r["exp_decision"] == c and r["pred_decision"] != c)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0)
    return sum(f1s) / len(f1s)


def compute(rows: list[dict], *, wall_s: float | None = None, cost_total: float | None = None) -> dict:
    """rows: one per (test, step, repeat) with the fields produced by runner.Runner."""
    if not rows:
        return {}
    ok_rows = [r for r in rows if r["pred_decision"] != "error"]
    call_rows = [r for r in rows if r["exp_decision"] == "call"]
    nocall_rows = [r for r in rows if r["exp_decision"] != "call"]
    multi_rows = [r for r in rows if r["n_exp_calls"] >= 2]

    by_test = defaultdict(list)
    for r in rows:
        by_test[(r["test_id"], r["repeat"])].append(r)
    tests_pass = [all(x["exact"] for x in v) for v in by_test.values()]
    wf = [all(x["exact"] for x in v) for v in by_test.values() if v[0]["n_steps"] >= 2]

    lat = [r["latency_ms"] for r in ok_rows if r["latency_ms"] is not None]
    conf_ok = [(r["confidence"], r["exact"]) for r in ok_rows]

    # consistency across repeats
    by_step = defaultdict(set)
    for r in rows:
        by_step[(r["test_id"], r["step"])].add(r["pred_signature"])
    reps = {r["repeat"] for r in rows}
    consistency = (sum(1 for s in by_step.values() if len(s) == 1) / len(by_step)) if len(reps) > 1 else None

    tok_in = sum(r.get("input_tokens", 0) for r in rows)
    m = {
        "n_steps": len(rows), "n_tests": len(by_test), "errors": len(rows) - len(ok_rows),
        "routing_accuracy": _ratio(sum(r["decision_ok"] for r in rows), len(rows)),
        "routing_macro_f1": macro_f1(rows),
        "tool_selection_acc": _ratio(sum(bool(r["tools_ok"]) for r in call_rows), len(call_rows)),
        "argument_accuracy": _ratio(sum(r["args_ok_cond"] for r in rows), sum(r["args_total_cond"] for r in rows)),
        "argument_acc_e2e": _ratio(sum(r["args_ok"] for r in rows), sum(r["args_total"] for r in rows)),
        "spurious_arg_rate": _ratio(sum(r["spurious_args"] for r in rows), sum(r["n_pred_calls"] for r in rows)),
        "multi_tool_success": _ratio(sum(r["exact"] for r in multi_rows), len(multi_rows)),
        "workflow_completion": _ratio(sum(wf), len(wf)),
        "false_tool_call_rate": _ratio(sum(r["false_call"] for r in nocall_rows), len(nocall_rows)),
        "missed_call_rate": _ratio(sum(r["missed_call"] for r in call_rows), len(call_rows)),
        "step_exact": _ratio(sum(r["exact"] for r in rows), len(rows)),
        "test_pass_rate": _ratio(sum(tests_pass), len(tests_pass)),
        "ece": ece(conf_ok),
        "coverage_at_95_precision": coverage_at_precision(conf_ok, 0.95),
        "consistency": consistency,
        "latency_ms_p50": _pct(lat, 0.5), "latency_ms_p95": _pct(lat, 0.95), "latency_ms_p99": _pct(lat, 0.99),
        "latency_ms_mean": (sum(lat) / len(lat)) if lat else None,
        "input_tokens_per_step": (tok_in / len(rows)) if rows else None,
        "throughput_rps": (len(rows) / wall_s) if wall_s else None,
    }
    if cost_total is not None:
        m["cost_total_usd"] = cost_total
        m["cost_per_1k_steps_usd"] = cost_total / len(rows) * 1000
        passed = sum(tests_pass)
        m["cost_per_success_usd"] = (cost_total / passed) if passed else None
    return m


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: compute(v) for k, v in sorted(groups.items())}


def _log_scale(v, best, worst):
    if v is None:
        return None
    v = min(max(v, best), worst)
    return (math.log(worst) - math.log(v)) / (math.log(worst) - math.log(best))


def benchmark_score(m: dict, weights=None, eff=None, gates=None, quality_share=0.8) -> dict:
    w = weights or DEFAULT_WEIGHTS
    eff = eff or DEFAULT_EFFICIENCY
    gates = gates or DEFAULT_GATES
    comp = {
        "routing_accuracy": m.get("routing_accuracy"),
        "tool_selection_acc": m.get("tool_selection_acc"),
        "argument_accuracy": m.get("argument_accuracy"),
        "multi_tool_success": m.get("multi_tool_success"),
        "workflow_completion": m.get("workflow_completion"),
        "no_false_call": None if m.get("false_tool_call_rate") is None else 1 - m["false_tool_call_rate"],
        "calibration": None if m.get("ece") is None else 1 - m["ece"],
    }
    used = {k: v for k, v in comp.items() if v is not None}
    wsum = sum(w[k] for k in used)
    quality = 100 * sum(w[k] * used[k] for k in used) / wsum if wsum else None
    lat_s = _log_scale(m.get("latency_ms_p95"), eff["latency_ms_best"], eff["latency_ms_worst"])
    cost_s = _log_scale(m.get("cost_per_1k_steps_usd"), eff["cost_1k_best"], eff["cost_1k_worst"])
    parts = [x for x in (lat_s, cost_s) if x is not None]
    efficiency = 100 * sum(parts) / len(parts) if parts else None
    overall = None
    if quality is not None:
        overall = quality if efficiency is None else quality_share * quality + (1 - quality_share) * efficiency
    failed = []
    if (m.get("false_tool_call_rate") or 0) > gates["false_tool_call_rate_max"]:
        failed.append(f"false_tool_call_rate > {gates['false_tool_call_rate_max']}")
    if (m.get("routing_accuracy") or 0) < gates["routing_accuracy_min"]:
        failed.append(f"routing_accuracy < {gates['routing_accuracy_min']}")
    return {"quality": quality, "efficiency": efficiency, "overall": overall, "components": comp,
            "production_eligible": not failed, "failed_gates": failed}


def dumps(o):
    return json.dumps(o, indent=2, ensure_ascii=False, default=lambda x: round(x, 4) if isinstance(x, float) else str(x))
