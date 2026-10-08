"""
Benchmark runner: dataset × models × repeats → rows → metrics → report.

Fairness / reproducibility rules implemented here:
  * Same dataset file, same tool catalog, same compiled questions for every decision model
    (sha256 of both is stored in run_meta.json).
  * Teacher forcing: step k of a multi-step test always sees the GOLD calls and the dataset's
    observations for steps < k, so one early mistake does not cascade; workflow completion is then
    "every step right". (A free-running mode needs a tool simulator — see README.)
  * Warm-up calls are excluded; latency = model call wall time only (compile/decode excluded);
    retries are counted separately.
  * Optional option-order perturbation per repeat to expose order sensitivity.
  * Seeds are fixed; every raw prediction is stored so any metric can be recomputed offline.
"""
from __future__ import annotations

import asyncio
import json
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import metrics as M
from .adapters import Adapter
from .compiler import Compiler
from .schema import Catalog, Prediction, TestCase, sha256_file
from .scoring import score_step


@dataclass
class RunConfig:
    repeats: int = 1
    warmup: int = 3
    seed: int = 118
    perturb_option_order: bool = False
    gate_threshold: float | None = None   # confidence gate (route low-confidence calls to clarify)
    need_threshold: float = 0.5
    compact_state: bool = False           # shrink state for small-context models (Laya: 512/1024 tokens)
    question_language: str = "tr"         # language of compiled question text: tr | en
    dataset: str = "data/dataset.jsonl"   # Turkish main set; data/dataset_en_control.jsonl = English control
    categories: list[str] | None = None   # e.g. ["A", "B"]
    max_tests: int | None = None
    store_answers: bool = False
    weights: dict | None = None
    efficiency: dict | None = None
    gates: dict | None = None
    extra: dict = field(default_factory=dict)


class ResourceSampler:
    """Samples this process' CPU/RSS (local adapters run in-process) and GPU via nvidia-smi if present."""

    def __init__(self, interval_s=0.5):
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t = None
        self.has_gpu = shutil.which("nvidia-smi") is not None

    def _gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2).stdout.strip().splitlines()
            u, mem = out[0].split(",")
            return float(u), float(mem)
        except Exception:
            return None, None

    def _loop(self):
        import psutil
        p = psutil.Process()
        p.cpu_percent(None)
        while not self._stop.wait(self.interval):
            s = {"cpu_pct": p.cpu_percent(None), "rss_mb": p.memory_info().rss / 2**20, "sys_cpu_pct": psutil.cpu_percent(None)}
            if self.has_gpu:
                s["gpu_util_pct"], s["gpu_mem_mb"] = self._gpu()
            self.samples.append(s)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.samples:
            return {}
        out = {}
        for k in self.samples[0]:
            vals = [s[k] for s in self.samples if s.get(k) is not None]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
                out[f"{k}_max"] = max(vals)
        return out


def _shuffle_criteria(questions: dict, rng: random.Random) -> dict:
    out = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            items = list(q["criteria"].items())
            rng.shuffle(items)
            q = {**q, "criteria": dict(items)}
        out[qid] = q
    return out


def _signature(pred: Prediction) -> str:
    return json.dumps(pred.to_dict()["decision"]) + json.dumps(
        sorted([[c.tool, sorted(c.args.items(), key=str)] for c in pred.calls], key=str), default=str)


class Runner:
    def __init__(self, catalog: Catalog, tests: list[TestCase], cfg: RunConfig, out_dir: Path):
        self.cat, self.cfg = catalog, cfg
        if cfg.categories:
            tests = [t for t in tests if t.category[0] in cfg.categories]
        if cfg.max_tests:
            tests = tests[: cfg.max_tests]
        self.tests = tests
        self.compiler = Compiler(catalog, question_language=cfg.question_language)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    async def _one(self, adapter: Adapter, tc: TestCase, si: int, rep: int, sem: asyncio.Semaphore) -> dict:
        step = tc.steps[si]
        rng = random.Random(f"{self.cfg.seed}-{tc.id}-{si}-{rep}")
        cs = self.compiler.compile(tc, si, compact_state=self.cfg.compact_state)
        qs = _shuffle_criteria(cs.questions, rng) if (self.cfg.perturb_option_order and rep > 0) else cs.questions
        err, resp = None, None
        async with sem:
            try:
                resp = await adapter.ask(cs.state, qs)
            except Exception as e:  # recorded, counted as an error row
                err = f"{type(e).__name__}: {e}"
        if resp is None:
            pred = Prediction("error", error=err)
        elif resp.prediction is not None:
            pred = resp.prediction
        else:
            pred = self.compiler.decode(resp.answers, tc, need_threshold=self.cfg.need_threshold,
                                        gate_threshold=self.cfg.gate_threshold)
        sc = score_step(pred, step["expected"], self.cat)
        row = {
            "model": adapter.name, "test_id": tc.id, "category": tc.category, "difficulty": tc.difficulty,
            "language": tc.language,
            "variant": next((v for v in ("noisy_ascii", "code_mixed", "tr_suffix") if v in tc.tags), "clean"),
            "step": si + 1, "n_steps": len(tc.steps), "repeat": rep,
            "latency_ms": resp.latency_ms if resp else None, "retries": resp.retries if resp else 0,
            "input_tokens": (resp.usage or {}).get("input_tokens", 0) if resp else 0,
            "output_tokens": (resp.usage or {}).get("output_tokens", 0) if resp else 0,
            "n_questions": len(qs), "confidence": pred.confidence,
            "prediction": pred.to_dict(), "expected": step["expected"], "pred_signature": _signature(pred),
            "error": err, **sc,
        }
        if self.cfg.store_answers and resp is not None:
            row["answers"] = resp.answers
        return row

    async def run_model(self, adapter: Adapter) -> dict:
        await adapter.setup()
        sem = asyncio.Semaphore(max(1, adapter.caps.max_concurrency))
        jobs = [(tc, si) for tc in self.tests for si in range(len(tc.steps))]
        for tc, si in jobs[: self.cfg.warmup]:  # warm-up (not recorded)
            await self._one(adapter, tc, si, 0, sem)
        rows: list[dict] = []
        with ResourceSampler() as rs:
            t0 = time.perf_counter()
            for rep in range(self.cfg.repeats):
                rows += await asyncio.gather(*(self._one(adapter, tc, si, rep, sem) for tc, si in jobs))
            wall = time.perf_counter() - t0
        await adapter.close()
        tok_in = sum(r["input_tokens"] for r in rows)
        tok_out = sum(r["output_tokens"] for r in rows)
        cost = tok_in / 1e6 * adapter.price_per_mtok_input + tok_out / 1e6 * adapter.price_per_mtok_output \
            + wall / 3600 * adapter.price_per_hour
        m = M.compute(rows, wall_s=wall, cost_total=cost)
        m["resources"] = rs.summary() if adapter.caps.local else {"note": "hosted API — resource usage not observable"}
        score = M.benchmark_score(m, self.cfg.weights, self.cfg.efficiency, self.cfg.gates)
        res = {"model": adapter.name, "adapter": adapter.describe(), "metrics": m, "score": score,
               "by_category": M.breakdown(rows, "category"), "by_difficulty": M.breakdown(rows, "difficulty"),
               "by_language": M.breakdown(rows, "language"), "by_variant": M.breakdown(rows, "variant"), "wall_s": wall}
        with (self.out / f"rows_{adapter.name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        (self.out / f"result_{adapter.name}.json").write_text(M.dumps(res), encoding="utf-8")
        return res

    def meta(self, adapters) -> dict:
        return {
            "dataset": self.cfg.dataset, "dataset_sha": sha256_file(Path(self.cfg.dataset)),
            "tools_sha": self.cat.version, "n_tests": len(self.tests),
            "n_steps": sum(len(t.steps) for t in self.tests), "config": self.cfg.__dict__,
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "adapters": [a.describe() for a in adapters], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
