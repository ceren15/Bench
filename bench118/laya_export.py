"""
Export the benchmark as Laya fine-tuning data ({state, questions, gold} rows, the schema read by
`laya.train.finetune` and the Kaggle notebook).

  python -m bench118 export-laya [--out data/laya_ft] [--test-share 0.3]

Split: by test (a noisy_ascii variant stays with its clean original), stratified by category,
fixed seed — so no test case, nor its noisy copy, is ever seen in training. Writes:

  train.jsonl / eval.jsonl         Laya rows (one per step) for training / held-out evaluation
  dataset_train.jsonl / dataset_test.jsonl   the same split in benchmark format (python -m bench118 run --dataset …)
  split.json                       ids per side + counts

Labels come from the gold expectation of each step, question by question:
  action        choice   gold tool(s) — uniform over them when the step calls several — or no_tool / clarify
  need.<tool>   noul     true iff the tool is called in this step
  <tool>.<arg>  choice   the stated value (uniform over values when calls of one tool differ)
  <tool>.<arg>? noul     whether the user stated the optional argument
  <tool>.<arg>=<v> noul  multi-value args: which values are asked for
  span args     choice   the matching regex candidate, or __none__
Questions about tools that are not called have no defined answer and are left unlabelled.
Every exported step is checked: decoding the argmax of its labels must reproduce the gold step.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .compiler import MULTI_VALUE_ARGS, NONE, Compiler
from .schema import Catalog, TestCase
from .scoring import norm, score_step


def split_tests(tests: list[TestCase], test_share: float, seed: int) -> tuple[list[TestCase], list[TestCase]]:
    groups: dict[str, list[TestCase]] = defaultdict(list)
    for t in tests:
        groups[t.noisy_of or t.id].append(t)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        by_cat[members[0].category].append(key)
    rng = random.Random(seed)
    test_keys: set[str] = set()
    for cat in sorted(by_cat):
        keys = sorted(by_cat[cat])
        rng.shuffle(keys)
        test_keys.update(keys[: round(len(keys) * test_share)])
    train = [t for t in tests if (t.noisy_of or t.id) not in test_keys]
    test = [t for t in tests if (t.noisy_of or t.id) in test_keys]
    return train, test


def _choice(*labels) -> dict:
    labels = list(dict.fromkeys(labels))
    return {"probabilities": {lab: 1.0 / len(labels) for lab in labels}, "label": labels[0]}


def _noul(value: bool) -> dict:
    return {"probabilities": {"true": 1.0 if value else 0.0, "false": 0.0 if value else 1.0},
            "label": "true" if value else "false"}


def gold_labels(cat: Catalog, questions: dict, expected: dict) -> dict:
    gold: dict[str, dict] = {}
    calls = expected["calls"] if expected["decision"] == "call" else []
    tools = sorted({c["tool"] for c in calls}, key=cat.precedence)
    gold["action"] = _choice(*tools) if tools else _choice(expected["decision"])
    for name in cat.tools:
        if f"need.{name}" in questions:
            gold[f"need.{name}"] = _noul(name in tools)
    for name in tools:
        t = cat.tools[name]
        tcalls = [c for c in calls if c["tool"] == name]
        for an, a in t.args.items():
            qid = f"{name}.{an}"
            vals = [c["args"][an] for c in tcalls if an in c["args"]]
            if a.json_type == "boolean":
                vals = ["true" if v else "false" for v in vals]
            if a.kind == "closed":
                if vals:
                    gold[qid] = _choice(*vals)
                if f"{qid}?" in questions:
                    gold[f"{qid}?"] = _noul(bool(vals))
                if (name, an) in MULTI_VALUE_ARGS:
                    for o in a.enum:
                        gold[f"{qid}={o}"] = _noul(o in vals)
            elif a.kind == "span" and qid in questions:
                opts = questions[qid]["criteria"]
                hit = next((o for o in opts if vals and norm(o) == norm(vals[0])), NONE)
                gold[qid] = _choice(hit)
    return {k: v for k, v in gold.items() if k in questions}


def _argmax_answers(questions: dict, gold: dict) -> dict:
    ans = {}
    for qid, g in gold.items():
        if questions[qid]["type"] == "noul":
            ans[qid] = {"type": "noul", "noul": g["probabilities"]["true"]}
        else:
            ans[qid] = {"type": "choice", "choice": g["label"], "confidence": 1.0}
    return ans


def export(tests: list[TestCase], cat: Catalog, compiler: Compiler, path: Path) -> dict:
    n_rows = n_labels = 0
    mismatches = []
    with path.open("w", encoding="utf-8") as f:
        for tc in tests:
            for si, step in enumerate(tc.steps):
                cs = compiler.compile(tc, si)
                gold = gold_labels(cat, cs.questions, step["expected"])
                pred = compiler.decode(_argmax_answers(cs.questions, gold), tc)
                if not score_step(pred, step["expected"], cat)["exact"]:
                    mismatches.append(f"{tc.id} step {si + 1}")
                qs = {k: cs.questions[k] for k in gold}
                f.write(json.dumps({"id": f"{tc.id}#{si + 1}", "category": tc.category, "state": cs.state,
                                    "questions": qs, "gold": gold}, ensure_ascii=False) + "\n")
                n_rows += 1
                n_labels += len(gold)
    return {"rows": n_rows, "labelled_questions": n_labels, "label_check_failures": mismatches}


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench118 export-laya")
    p.add_argument("--dataset", default="data/dataset.jsonl")
    p.add_argument("--out", default="data/laya_ft")
    p.add_argument("--test-share", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=118)
    p.add_argument("--question-language", choices=["tr", "en"], default="tr")
    a = p.parse_args(argv)
    cat = Catalog()
    tests = TestCase.load_all(Path(a.dataset))
    train, test = split_tests(tests, a.test_share, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comp = Compiler(cat, question_language=a.question_language)
    for name, part in (("dataset_train.jsonl", train), ("dataset_test.jsonl", test)):
        with (out / name).open("w", encoding="utf-8") as f:
            for t in part:
                f.write(json.dumps(t.__dict__, ensure_ascii=False) + "\n")
    summary = {"seed": a.seed, "test_share": a.test_share, "question_language": a.question_language,
               "train": export(train, cat, comp, out / "train.jsonl"),
               "eval": export(test, cat, comp, out / "eval.jsonl"),
               "train_tests": len(train), "test_tests": len(test),
               "train_by_category": dict(sorted(Counter(t.category for t in train).items())),
               "test_by_category": dict(sorted(Counter(t.category for t in test).items())),
               "train_ids": [t.id for t in train], "test_ids": [t.id for t in test]}
    (out / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for side in ("train", "eval"):
        s = summary[side]
        print(f"{side}: {s['rows']} rows, {s['labelled_questions']} labelled questions, "
              f"label check failures: {len(s['label_check_failures'])} {s['label_check_failures'][:5]}")
    print(f"tests: train {len(train)} / test {len(test)}")
    print("test by category:", summary["test_by_category"])
    overlap = {t.noisy_of or t.id for t in train} & {t.noisy_of or t.id for t in test}
    print("train/test group overlap:", len(overlap))


if __name__ == "__main__":
    main()
