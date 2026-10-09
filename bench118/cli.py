"""
Command line:

  python -m bench118 validate [data/dataset.jsonl]
  python -m bench118 run --config configs/models.yaml [--models jev,clef] [--categories A,B] [--max-tests 50]
  python -m bench118 compile --test A-0001 --step 1          # show exactly what a model receives
  python -m bench118 export-laya [--out data/laya_ft]        # Laya fine-tuning data + train/test split
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from . import adapters, report
from .compiler import Compiler
from .metrics import dumps
from .runner import RunConfig, Runner
from .schema import Catalog, TestCase


def _run(args):
    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rc = RunConfig(**conf.get("run", {}))
    if args.categories:
        rc.categories = args.categories.split(",")
    if args.max_tests:
        rc.max_tests = args.max_tests
    if args.dataset:
        rc.dataset = args.dataset
    if args.question_language:
        rc.question_language = args.question_language
    cat = Catalog()
    tests = TestCase.load_all(Path(rc.dataset))
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{rc.question_language}"
    out = Path(conf.get("output_dir", "results")) / run_id
    runner = Runner(cat, tests, rc, out)
    wanted = set(args.models.split(",")) if args.models else None
    specs = [m for m in conf["models"] if (wanted and m["name"] in wanted) or (not wanted and m.get("enabled", True))]
    built = []
    for spec in specs:
        if spec["adapter"] == "openai_llm":
            spec = {**spec, "catalog": cat}
        built.append(adapters.build(spec))
    meta = runner.meta(built)
    (out / "run_meta.json").write_text(dumps(meta), encoding="utf-8")
    results = []
    for a in built:
        print(f"→ {a.name}: {len(runner.tests)} tests …", flush=True)
        try:
            res = asyncio.run(runner.run_model(a))
        except Exception as e:  # one broken backend must not kill the whole run
            print(f"  ✗ {a.name} failed: {type(e).__name__}: {e}")
            continue
        s, m = res["score"], res["metrics"]
        f = lambda v, nd=1: "—" if v is None else f"{v:.{nd}f}"
        print(f"  quality={f(s['quality'])} efficiency={f(s['efficiency'])} overall={f(s['overall'])} "
              f"routing={f(m.get('routing_accuracy'), 3)} p95={f(m.get('latency_ms_p95'), 0)}ms errors={m.get('errors')}")
        results.append(res)
    if results:
        md = report.write(results, meta, out)
        print(f"report: {md}")


def _compile(args):
    cat = Catalog()
    tests = {t.id: t for t in TestCase.load_all(Path(args.dataset or "data/dataset.jsonl"))}
    tc = tests[args.test]
    cs = Compiler(cat, question_language=args.question_language or "tr").compile(tc, args.step - 1)
    print(json.dumps({"state": cs.state, "n_questions": len(cs.questions), "candidates": cs.candidates,
                      "questions": dict(list(cs.questions.items())[: args.show])}, ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench118")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("path", nargs="?", default="data/dataset.jsonl")
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/models.yaml"); r.add_argument("--models")
    r.add_argument("--categories"); r.add_argument("--max-tests", type=int); r.add_argument("--dataset")
    r.add_argument("--question-language", choices=["tr", "en"])
    c = sub.add_parser("compile")
    c.add_argument("--test", required=True); c.add_argument("--step", type=int, default=1)
    c.add_argument("--dataset"); c.add_argument("--question-language", choices=["tr", "en"]); c.add_argument("--show", type=int, default=4)
    sub.add_parser("export-laya", add_help=False)
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv[:1] == ["export-laya"]:
        from .laya_export import main as em
        return em(argv[1:])
    a = p.parse_args(argv)
    if a.cmd == "validate":
        from .validate import main as vm
        raise SystemExit(vm([a.path]))
    if a.cmd == "run":
        _run(a)
    if a.cmd == "compile":
        _compile(a)


if __name__ == "__main__":
    main()
