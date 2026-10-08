"""
Self-tests for the benchmark itself.

The key one is the ORACLE test: an adapter that answers every compiled question perfectly (built
from the gold labels) must score ~100%. If it does not, the bug is in the dataset, the compiler or
the scorer — never blame a model for it. Run:  python -m pytest -q   (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench118.compiler import MULTI_VALUE_ARGS, NONE, Compiler, extract_candidates  # noqa: E402
from bench118.metrics import compute  # noqa: E402
from bench118.schema import Catalog, Prediction, TestCase  # noqa: E402
from bench118.scoring import norm, score_step  # noqa: E402

CAT = Catalog()
TESTS = TestCase.load_all(Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl")


def oracle_answers(compiler: Compiler, tc: TestCase, si: int) -> dict:
    exp = tc.steps[si]["expected"]
    cs = compiler.compile(tc, si)
    ans = {}
    tools = sorted({c["tool"] for c in exp["calls"]}, key=CAT.precedence)
    action = tools[0] if exp["decision"] == "call" else exp["decision"]
    ans["action"] = {"type": "choice", "choice": action, "confidence": 0.99}
    for qid, q in cs.questions.items():
        if qid.startswith("need."):
            ans[qid] = {"type": "noul", "noul": 1.0 if qid[5:] in tools else 0.0}
    for c in exp["calls"]:
        t = CAT.tools[c["tool"]]
        for an, a in t.args.items():
            if a.kind == "closed":
                if an in c["args"]:
                    ans[f"{t.name}.{an}"] = {"type": "choice", "choice": c["args"][an], "confidence": 0.99}
                    ans[f"{t.name}.{an}?"] = {"type": "noul", "noul": 1.0}
                else:
                    ans.setdefault(f"{t.name}.{an}", {"type": "choice", "choice": a.enum[0], "confidence": 0.5})
                    ans.setdefault(f"{t.name}.{an}?", {"type": "noul", "noul": 0.0})
            elif a.kind == "span" and f"{t.name}.{an}" in cs.questions:
                opts = cs.questions[f"{t.name}.{an}"]["criteria"]
                val = next((o for o in opts if an in c["args"] and norm(o) == norm(c["args"][an])), NONE)
                ans[f"{t.name}.{an}"] = {"type": "choice", "choice": val}
        for (tn, an) in MULTI_VALUE_ARGS:
            if tn == t.name:
                vals = {x["args"].get(an) for x in exp["calls"] if x["tool"] == tn}
                for o in t.args[an].enum:
                    ans[f"{tn}.{an}={o}"] = {"type": "noul", "noul": 1.0 if o in vals else 0.0}
    return ans


def test_oracle_scores_perfectly():
    comp = Compiler(CAT)
    rows, failures = [], []
    for tc in TESTS:
        for si in range(len(tc.steps)):
            pred = comp.decode(oracle_answers(comp, tc, si), tc)
            sc = score_step(pred, tc.steps[si]["expected"], CAT)
            rows.append({**sc, "test_id": tc.id, "step": si + 1, "n_steps": len(tc.steps), "repeat": 0,
                         "latency_ms": 1.0, "confidence": 0.99, "pred_signature": "x"})
            if not sc["exact"]:
                failures.append((tc.id, si + 1, tc.user_message, pred.to_dict()["calls"]))
    m = compute(rows)
    # Known, documented limitation: one message asking two edit_section calls with DIFFERENT
    # change_types cannot be expressed by a single change_type choice → allowed to fail.
    unexpected = [f for f in failures if not all(c["tool"] == "edit_section" for c in f[3])]
    assert not unexpected, unexpected[:5]
    assert m["routing_accuracy"] == 1.0
    assert m["false_tool_call_rate"] == 0.0


def test_span_extraction_turkish_suffixes():
    c = extract_candidates("kahvedurağı.com.tr'yi ve www.ayseyilmaz.com'u bağla, ana rengi #1E40AF'e çek, adı \"Fırın 118\" olsun")
    assert c["domain"] == ["kahvedurağı.com.tr", "ayseyilmaz.com"]
    assert c["hex"] == ["#1E40AF"]
    assert c["quoted"] == ["Fırın 118"]
    assert extract_candidates("SEO'sunu kontrol et, Kadıköy'de")["quoted"] == []


def test_turkish_normalisation():
    assert norm("İLETİŞİM") == norm("iletişim")
    assert norm("www.Koc-Merve.com") == "koc-merve.com"


def test_no_tool_scoring():
    sc = score_step(Prediction("call"), {"decision": "no_tool", "calls": [], "order_matters": False, "missing": []}, CAT)
    assert sc["false_call"] and not sc["exact"]


def test_mock_end_to_end(tmp_path=Path("/tmp/bench118_test")):
    from bench118.adapters import MockKeywordAdapter
    from bench118.runner import RunConfig, Runner
    r = Runner(CAT, TESTS, RunConfig(repeats=1, warmup=0, max_tests=40), tmp_path)
    res = asyncio.run(r.run_model(MockKeywordAdapter()))
    assert res["metrics"]["n_tests"] == 40 and res["metrics"]["errors"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
