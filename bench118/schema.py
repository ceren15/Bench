"""Core data types shared by the compiler, adapters, runner and scorer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PSEUDO_ACTIONS = ("no_tool", "clarify")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@dataclass
class ArgSpec:
    name: str
    kind: str  # closed | span | free | context
    required: bool
    enum: list[str] | None = None
    json_type: str = "string"
    description: str = ""
    option_descriptions: dict[str, str] = field(default_factory=dict)
    option_descriptions_tr: dict[str, str] = field(default_factory=dict)
    default: Any = None


@dataclass
class ToolSpec:
    name: str
    description: str
    precedence: int
    side_effect: str
    args: dict[str, ArgSpec]
    raw: dict  # original JSON-schema definition (for LLM adapters)
    description_tr: str = ""

    @property
    def closed_args(self):
        return {k: a for k, a in self.args.items() if a.kind == "closed"}

    @property
    def span_args(self):
        return {k: a for k, a in self.args.items() if a.kind == "span"}


class Catalog:
    """Tool catalog loaded from data/tools.json."""

    def __init__(self, path: Path = DATA / "tools.json"):
        self.path = Path(path)
        self.doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.pseudo = self.doc["pseudo_actions"]
        self.pseudo_tr = self.doc.get("pseudo_actions_tr", self.pseudo)
        self.tools: dict[str, ToolSpec] = {}
        for t in self.doc["tools"]:
            props = t["parameters"]["properties"]
            req = set(t["parameters"].get("required", []))
            args = {}
            for n, p in props.items():
                args[n] = ArgSpec(
                    name=n,
                    kind=p.get("x-arg-kind", "free"),
                    required=n in req,
                    enum=p.get("enum"),
                    json_type=p.get("type", "string"),
                    description=p.get("description", ""),
                    option_descriptions=p.get("x-option-descriptions", {}),
                    option_descriptions_tr=p.get("x-option-descriptions-tr", {}),
                    default=p.get("x-optional-default"),
                )
            self.tools[t["name"]] = ToolSpec(
                name=t["name"], description=t["description"], precedence=t.get("x-precedence", 100),
                side_effect=t.get("x-side-effect", "reversible"), args=args, raw=t,
                description_tr=t.get("x-description-tr", ""),
            )
        self.version = sha256_file(self.path)

    def precedence(self, tool: str) -> int:
        return self.tools[tool].precedence if tool in self.tools else 999

    def openai_tools(self) -> list[dict]:
        """Function-calling definitions (OpenAI format) with x- extensions stripped,
        plus two explicit pseudo tools so an LLM can express no_tool / clarify."""
        out = []
        for t in self.doc["tools"]:
            params = json.loads(json.dumps(t["parameters"]))
            for p in params["properties"].values():
                for k in [k for k in p if k.startswith("x-")]:
                    p.pop(k)
            # context args are filled by the orchestrator, never by a model
            for n, a in self.tools[t["name"]].args.items():
                if a.kind == "context":
                    params["properties"].pop(n, None)
                    if n in params.get("required", []):
                        params["required"].remove(n)
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": params}})
        out.append({"type": "function", "function": {
            "name": "ask_user", "description": self.pseudo["clarify"],
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}})
        out.append({"type": "function", "function": {
            "name": "respond_without_tools", "description": self.pseudo["no_tool"],
            "parameters": {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}}})
        return out


@dataclass
class TestCase:
    id: str
    category: str
    difficulty: str
    language: str
    user_message: str
    context: dict
    steps: list[dict]
    expected_behavior: str
    success_criteria: dict
    tags: list[str]
    free_args: list[str]
    noisy_of: str | None = None  # id of the clean Turkish original (asciified / lower-case variant)

    @classmethod
    def load_all(cls, path: Path = DATA / "dataset.jsonl") -> list["TestCase"]:
        out = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(cls(**d))
        return out


@dataclass
class Call:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    decision: str  # call | no_tool | clarify | error
    calls: list[Call] = field(default_factory=list)
    confidence: float | None = None  # confidence of the routing decision
    arg_confidence: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self):
        return {"decision": self.decision, "calls": [{"tool": c.tool, "args": c.args} for c in self.calls],
                "confidence": self.confidence, "arg_confidence": self.arg_confidence, "error": self.error}


@dataclass
class AdapterResponse:
    answers: dict[str, dict]  # SystemOne-style answers keyed by question id (decision models)
    latency_ms: float  # wall-clock of the model call(s) only
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens
    raw: Any = None
    prediction: Prediction | None = None  # set directly by LLM adapters that skip the compiler
    n_requests: int = 1
    retries: int = 0
