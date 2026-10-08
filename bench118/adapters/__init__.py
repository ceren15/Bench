"""Adapter registry. Add a new backend by writing an Adapter subclass and registering it here."""
from __future__ import annotations

from .base import Adapter, Capabilities
from .llm_openai import OpenAICompatibleLLMAdapter
from .mock import MockKeywordAdapter
from .systemone_http import SystemOneHTTPAdapter


def _laya(**kw):
    from .laya_local import LayaLocalAdapter
    return LayaLocalAdapter(**kw)


REGISTRY = {
    "systemone_http": SystemOneHTTPAdapter,   # Jev, Clef (Workers AI), Kev (local server)
    "laya_local": _laya,                      # Laya in-process
    "openai_llm": OpenAICompatibleLLMAdapter, # LLM baseline with native tool calling
    "mock": MockKeywordAdapter,               # pipeline dry run
}


def build(spec: dict) -> Adapter:
    spec = dict(spec)
    kind = spec.pop("adapter")
    if kind not in REGISTRY:
        raise KeyError(f"unknown adapter '{kind}'. Known: {sorted(REGISTRY)}")
    spec.pop("enabled", None)
    return REGISTRY[kind](**spec)


__all__ = ["Adapter", "Capabilities", "build", "REGISTRY"]
