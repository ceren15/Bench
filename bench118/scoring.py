"""Per-step scoring of a Prediction against the gold expectation."""
from __future__ import annotations

from collections import Counter

from .schema import Catalog, Prediction


def norm(v):
    """Turkish-aware normalisation: 'İ'→'i', 'I'→'ı' before lower-casing; strips 'www.'."""
    if isinstance(v, str):
        s = v.strip().replace("İ", "i").replace("I", "ı").lower()
        if s.startswith("www."):
            s = s[4:]
        return s
    return v


def _arg_match(exp_args: dict, pred_args: dict) -> tuple[int, int]:
    ok = sum(1 for k, v in exp_args.items() if norm(pred_args.get(k)) == norm(v))
    return ok, len(exp_args)


def _spurious(tool: str, exp_args: dict, pred_args: dict, catalog: Catalog) -> int:
    """Count args the model filled although the user did not state them, with a value that
    differs from the declared default (a hallucinated argument)."""
    n = 0
    spec = catalog.tools.get(tool)
    if not spec:
        return 0
    for k, v in pred_args.items():
        a = spec.args.get(k)
        if k in exp_args or not a or a.kind not in ("closed", "span"):
            continue
        d = a.default
        if isinstance(d, str) and d.startswith("@"):
            continue  # dynamic default — cannot judge
        if norm(v) != norm(d):
            n += 1
    return n


def score_step(pred: Prediction, exp: dict, catalog: Catalog) -> dict:
    r = {
        "exp_decision": exp["decision"], "pred_decision": pred.decision,
        "decision_ok": pred.decision == exp["decision"],
        "n_exp_calls": len(exp["calls"]), "n_pred_calls": len(pred.calls) if pred.decision == "call" else 0,
        "tools_ok": None, "args_ok": 0, "args_total": 0, "args_ok_cond": 0, "args_total_cond": 0,
        "spurious_args": 0, "order_ok": None, "exact": False,
        "false_call": exp["decision"] != "call" and pred.decision == "call",
        "missed_call": exp["decision"] == "call" and pred.decision != "call",
        "extra_tools": 0, "missing_tools": 0,
    }
    if exp["decision"] != "call":
        r["exact"] = r["decision_ok"]
        return r
    exp_tools = Counter(c["tool"] for c in exp["calls"])
    pred_tools = Counter(c.tool for c in pred.calls) if pred.decision == "call" else Counter()
    r["tools_ok"] = exp_tools == pred_tools
    r["extra_tools"] = sum((pred_tools - exp_tools).values())
    r["missing_tools"] = sum((exp_tools - pred_tools).values())

    # greedy one-to-one matching of expected calls to predicted calls of the same tool
    pool = list(pred.calls) if pred.decision == "call" else []
    matched_idx = []
    for ec in exp["calls"]:
        best, best_ok = None, -1
        for i, pc in enumerate(pool):
            if pc is None or pc.tool != ec["tool"]:
                continue
            ok, _ = _arg_match(ec["args"], pc.args)
            if ok > best_ok:
                best, best_ok = i, ok
        n = len(ec["args"])
        r["args_total"] += n
        if best is None:
            matched_idx.append(None)
            continue
        pc = pool[best]
        pool[best] = None
        matched_idx.append(best)
        r["args_ok"] += best_ok
        r["args_ok_cond"] += best_ok
        r["args_total_cond"] += n
        r["spurious_args"] += _spurious(ec["tool"], ec["args"], pc.args, catalog)

    if exp.get("order_matters") and r["tools_ok"]:
        idx = [i for i in matched_idx if i is not None]
        r["order_ok"] = idx == sorted(idx)
    all_args = r["args_ok"] == r["args_total"]
    r["exact"] = bool(r["decision_ok"] and r["tools_ok"] and all_args and r["order_ok"] in (None, True))
    return r
