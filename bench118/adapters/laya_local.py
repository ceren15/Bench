"""
In-process adapter for Laya (Convai Innovations, Apache-2.0).

    pip install laya
    agent = laya.load("convaiinnovations/laya")                          # English, 512-token window
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual")  # 100+ languages, 1,024 tokens
    agent.predict(state, questions)

Known from the model card / third-party tests (verify on your hardware):
  * "a fast base to specialise, not a zero-shot decision engine" — expect low zero-shot scores;
    fine-tune on 118.studio data before drawing conclusions.
  * Degrades beyond ~20 options and is sensitive to option order -> run with perturb_option_order.
  * Shipped temperatures are invalid; refit calibration per question type before trusting confidence.
  * `laya.load()` can hang if TensorFlow is installed (transformers probes TF) -> use a clean venv.
  * Turkish is not English: use the multilingual checkpoint (or the library's Router).

The exact shape returned by predict() is normalised defensively below; check `raw` on first run.
"""
from __future__ import annotations

import asyncio
import time

from ..schema import AdapterResponse
from .base import Adapter, Capabilities


class LayaLocalAdapter(Adapter):
    def __init__(self, name="laya", repo="convaiinnovations/laya", subfolder: str | None = "multilingual",
                 use_router: bool = False, caps: dict | None = None, price_per_hour: float = 0.0,
                 device: str | None = None, max_len: int | None = None, head_max_len: int | None = None):
        # repo may also be a local checkpoint directory (e.g. a fine-tuned model; set subfolder: null).
        # max_len / head_max_len override the checkpoint's token budgets — give the base and the
        # fine-tuned model the same values so they see identical inputs.
        self.name = name
        self.repo = repo
        self.subfolder = subfolder
        self.use_router = use_router
        self.device = device
        self.predict_kw = {k: v for k, v in (("max_len", max_len), ("head_max_len", head_max_len)) if v}
        self.caps = Capabilities(**({"local": True, "max_concurrency": 1, "context_tokens": 1024} | (caps or {})))
        self.price_per_hour = price_per_hour
        self.agent = None

    async def setup(self):
        import laya  # noqa: imported lazily so the package is optional

        def _load():
            if self.use_router:
                return laya.Router()
            kw = {"device": self.device} if self.device else {}
            if self.subfolder:
                kw["subfolder"] = self.subfolder
            return laya.load(self.repo, **kw)

        self.agent = await asyncio.to_thread(_load)

    @staticmethod
    def _normalise(res) -> dict:
        if isinstance(res, dict) and "answers" in res:
            return res["answers"]
        if isinstance(res, dict):
            return res
        if hasattr(res, "answers"):
            return dict(res.answers)
        raise TypeError(f"unexpected Laya result type {type(res)}")

    async def ask(self, state: dict, questions: dict[str, dict]) -> AdapterResponse:
        def _run():
            t0 = time.perf_counter()
            r = self.agent.predict(state, questions, **self.predict_kw)
            return r, (time.perf_counter() - t0) * 1000

        res, ms = await asyncio.to_thread(_run)
        return AdapterResponse(answers=self._normalise(res), latency_ms=ms, raw=None)
