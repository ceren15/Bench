"""
HTTP adapter for the System One contract:  POST {state, model, questions} -> {answers, usage}.

Verified contracts (Oct 2026, see docs/README):
  * Jev   — TypeSafe:   POST https://api.typesafe.ai/v1/systemone, Bearer key, model "jev-latest"
  * Clef  — Cloudflare: POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/cloudflare/clef
                         body also needs "model": "clef" | "clef-flash". Cloudflare REST responses are
                         usually wrapped in {"result": ...}; `unwrap` handles both shapes — verify on first run.
  * Kev   — local server that mirrors /v1/systemone (bound to 127.0.0.1, no auth, single request at a time).

Uses only the standard library (urllib) so it runs anywhere; calls are executed in a thread pool.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request

from ..schema import AdapterResponse
from .base import Adapter, Capabilities

RETRY_STATUS = {429, 500, 502, 503, 529}


class SystemOneHTTPAdapter(Adapter):
    def __init__(self, name: str, url: str, model: str, api_key_env: str | None = None,
                 headers: dict | None = None, unwrap: str | None = "auto", timeout_s: float = 30.0,
                 max_retries: int = 3, caps: dict | None = None,
                 price_per_mtok_input: float = 0.0, price_per_mtok_output: float = 0.0, price_per_hour: float = 0.0):
        self.name = name
        self.url = os.path.expandvars(url)
        self.model = model
        self.api_key_env = api_key_env
        self.headers = headers or {}
        self.unwrap = unwrap
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.caps = Capabilities(**(caps or {}))
        self.price_per_mtok_input = price_per_mtok_input
        self.price_per_mtok_output = price_per_mtok_output
        self.price_per_hour = price_per_hour

    def _post(self, body: dict) -> tuple[dict, float, int]:
        data = json.dumps(body, ensure_ascii=False).encode()
        hdr = {"Content-Type": "application/json", **self.headers}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {self.api_key_env} is not set")
            hdr["Authorization"] = f"Bearer {key}"
        retries = 0
        while True:
            req = urllib.request.Request(self.url, data=data, headers=hdr, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    payload = json.loads(r.read())
                return payload, (time.perf_counter() - t0) * 1000, retries
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and retries < self.max_retries:
                    retries += 1
                    ra = e.headers.get("retry-after")
                    time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else (2 ** retries) * 0.5 + random.random() * 0.25)
                    continue
                detail = e.read().decode(errors="replace")[:500]
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None

    def _unwrap(self, payload: dict) -> dict:
        if self.unwrap == "auto":
            if "answers" in payload:
                return payload
            if isinstance(payload.get("result"), dict) and "answers" in payload["result"]:
                return payload["result"]
            return payload
        if self.unwrap:
            for k in self.unwrap.split("."):
                payload = payload[k]
        return payload

    async def ask(self, state: dict, questions: dict[str, dict]) -> AdapterResponse:
        # Split very large question maps if the backend has a per-request cap.
        cap = self.caps.max_questions_per_request
        items = list(questions.items())
        chunks = [dict(items)] if not cap else [dict(items[i:i + cap]) for i in range(0, len(items), cap)]
        answers, usage, total_ms, raw, retries = {}, {"input_tokens": 0, "output_tokens": 0}, 0.0, [], 0
        for ch in chunks:
            body = {"model": self.model, "state": state, "questions": ch}
            payload, ms, r = await asyncio.to_thread(self._post, body)
            p = self._unwrap(payload)
            answers.update(p.get("answers", {}))
            for k in usage:
                usage[k] += int((p.get("usage") or {}).get(k, 0))
            total_ms += ms
            retries += r
            raw.append({"model": p.get("model")})
        return AdapterResponse(answers=answers, latency_ms=total_ms, usage=usage, raw=raw,
                               n_requests=len(chunks), retries=retries)
