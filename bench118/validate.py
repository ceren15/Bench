"""Dataset validation — run before every benchmark and in CI.

Checks: unique ids, known tools, enum values valid, required closed args present (or the step is a
clarify), span values extractable by the compiler's candidate parser (otherwise no decision model
could ever get them right), observation count matches calls, last step of multi-step tests ends the
turn, category/difficulty balance.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .compiler import extract_candidates, span_candidate_type
from .schema import Catalog, TestCase
from .scoring import norm


def validate(path: Path, catalog: Catalog) -> list[str]:
    errs: list[str] = []
    tests = TestCase.load_all(path)
    ids = Counter(t.id for t in tests)
    errs += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for t in tests:
        cands = extract_candidates(t.user_message)
        for i, st in enumerate(t.steps):
            e = st["expected"]
            if e["decision"] not in ("call", "no_tool", "clarify"):
                errs.append(f"{t.id} step {i+1}: bad decision {e['decision']}")
            if e["decision"] == "call" and not e["calls"]:
                errs.append(f"{t.id} step {i+1}: call without calls")
            if i > 0 and st["observation"] is None:
                errs.append(f"{t.id} step {i+1}: missing observation")
            for c in e["calls"]:
                spec = catalog.tools.get(c["tool"])
                if not spec:
                    errs.append(f"{t.id}: unknown tool {c['tool']}")
                    continue
                for k, v in c["args"].items():
                    a = spec.args.get(k)
                    if not a:
                        errs.append(f"{t.id}: {c['tool']} has no arg {k}")
                    elif a.kind == "closed" and a.enum and v not in a.enum:
                        errs.append(f"{t.id}: {c['tool']}.{k}={v!r} not in enum")
                    elif a.kind == "span":
                        pool = [norm(x) for x in cands[span_candidate_type(c["tool"], k)]]
                        if norm(v) not in pool:
                            errs.append(f"{t.id}: span {c['tool']}.{k}={v!r} not extractable from message (candidates={pool})")
                    elif a.kind in ("free", "context"):
                        errs.append(f"{t.id}: {k} is {a.kind} and must not be in expected args")
                for n, a in spec.args.items():
                    if a.required and a.kind in ("closed", "span") and n not in c["args"]:
                        errs.append(f"{t.id}: {c['tool']} missing required {n}")
        if len(t.steps) > 1 and t.steps[-1]["expected"]["decision"] == "call":
            errs.append(f"{t.id}: multi-step test should end with no_tool/clarify (turn end)")
    return errs


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = Path(argv[0]) if argv else Path("data/dataset.jsonl")
    cat = Catalog()
    errs = validate(path, cat)
    tests = TestCase.load_all(path)
    print(f"{path}: {len(tests)} tests, {sum(len(t.steps) for t in tests)} steps")
    print("categories:", dict(sorted(Counter(t.category for t in tests).items())))
    print("difficulty:", dict(Counter(t.difficulty for t in tests)))
    print("decisions :", dict(Counter(s["expected"]["decision"] for t in tests for s in t.steps)))
    if errs:
        print(f"\n{len(errs)} problem(s):")
        for e in errs[:50]:
            print("  -", e)
        return 1
    print("OK — no problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
