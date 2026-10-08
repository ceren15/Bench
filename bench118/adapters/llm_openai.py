"""
Baseline: a generative LLM doing native function calling through any OpenAI-compatible
/chat/completions endpoint (OpenAI, OpenRouter, vLLM, Workers AI's OpenAI endpoint, Ollama...).

It skips the question compiler and returns a Prediction directly, so decision models and LLMs
are scored by the exact same scorer. Two pseudo tools (ask_user, respond_without_tools) make
"clarify" and "no_tool" explicit instead of guessing them from free text.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request

from ..schema import AdapterResponse, Call, Catalog, Prediction
from .base import Adapter, Capabilities

SYSTEM = ("You are the routing layer of 118.studio, an AI website builder. Decide the NEXT action only. "
          "Call one or more tools for actions that can run now; call ask_user if something required is missing or ambiguous; "
          "call respond_without_tools if no tool is needed or everything requested is done. Never retry a failed purchase. "
          "Session state and already-executed steps are given as JSON.")


class OpenAICompatibleLLMAdapter(Adapter):
    def __init__(self, name: str, base_url: str, model: str, api_key_env: str | None = "OPENAI_API_KEY",
                 catalog: Catalog | None = None, temperature: float = 0.0, timeout_s: float = 60.0,
                 price_per_mtok_input: float = 0.0, price_per_mtok_output: float = 0.0, caps: dict | None = None):
        self.name, self.base_url, self.model = name, base_url.rstrip("/"), model
        self.api_key_env, self.temperature, self.timeout_s = api_key_env, temperature, timeout_s
        self.catalog = catalog or Catalog()
        self.tools = self.catalog.openai_tools()
        self.caps = Capabilities(**({"kind": "llm", "max_concurrency": 4} | (caps or {})))
        self.price_per_mtok_input, self.price_per_mtok_output = price_per_mtok_input, price_per_mtok_output

    def _post(self, body):
        hdr = {"Content-Type": "application/json"}
        if self.api_key_env:
            hdr["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=hdr)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read()), (time.perf_counter() - t0) * 1000

    async def ask(self, state: dict, questions: dict[str, dict]) -> AdapterResponse:
        body = {"model": self.model, "temperature": self.temperature, "tools": self.tools, "tool_choice": "required",
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]}
        payload, ms = await asyncio.to_thread(self._post, body)
        msg = payload["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": fn.get("arguments")}
            calls.append(Call(fn["name"], args))
        names = {c.tool for c in calls}
        if not calls or names == {"respond_without_tools"}:
            pred = Prediction("no_tool")
        elif "ask_user" in names and len(names) == 1:
            pred = Prediction("clarify")
        else:
            pred = Prediction("call", [c for c in calls if c.tool not in ("ask_user", "respond_without_tools")])
        u = payload.get("usage", {})
        return AdapterResponse(answers={}, latency_ms=ms, prediction=pred,
                               usage={"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)})
