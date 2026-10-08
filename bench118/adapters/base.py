"""Adapter contract. Every model — hosted API, local weights or LLM — plugs in here."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from ..schema import AdapterResponse


@dataclass
class Capabilities:
    """What the backend can actually do. The runner never assumes more than this."""
    kind: str = "decision"           # decision (answers SystemOne questions) | llm (native tool calling)
    max_choice_options: int = 255
    max_score_levels: int = 10
    max_questions_per_request: int | None = None  # None = unlimited; otherwise the runner splits
    context_tokens: int | None = None             # for state truncation warnings
    max_concurrency: int = 8
    supports_noul: bool = True
    supports_score: bool = True
    local: bool = False                            # True -> resource usage is sampled on this host
    notes: list[str] = field(default_factory=list)


class Adapter(abc.ABC):
    name: str = "adapter"
    caps: Capabilities = Capabilities()
    price_per_mtok_input: float = 0.0   # USD; used by the cost metric
    price_per_mtok_output: float = 0.0
    price_per_hour: float = 0.0         # USD; for self-hosted GPUs (cost = runtime * rate)

    async def setup(self) -> None:  # load weights, open sessions
        return None

    async def close(self) -> None:
        return None

    @abc.abstractmethod
    async def ask(self, state: dict, questions: dict[str, dict]) -> AdapterResponse:
        """Decision models: evaluate typed questions against state.
        LLM adapters may instead set AdapterResponse.prediction directly."""

    def describe(self) -> dict:
        return {"name": self.name, "class": type(self).__name__, "caps": self.caps.__dict__,
                "price_per_mtok_input": self.price_per_mtok_input, "price_per_hour": self.price_per_hour}
