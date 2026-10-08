"""
Turn the tool catalog + a conversation step into System One questions, and decode answers
back into a tool-call Prediction.

Decision models (Jev, Clef, Kev, Laya) do not generate text. They answer typed questions:
  choice -> pick one option (<=255 options), score -> ordinal level, noul -> P(yes).
So "tool calling" is compiled into questions (same idea as TypeSafe's function-calling cookbook):

  action            choice  over every tool + no_tool + clarify      -> routing decision
  need.<tool>       noul    one per tool                              -> multi-tool set
  <tool>.<arg>      choice  one per closed argument                   -> argument values
  <tool>.<arg>?     noul    "did the user state this?" (optional args) -> omit vs. fill
  <tool>.<arg>=<v>  noul    per option, for args that can repeat      -> e.g. two add_page calls
  <tool>.<arg>      choice  over regex-parsed candidates + __none__   -> span args (domain, hex, names)

Question text can be Turkish or English (question_language) — benchmark both: the user
message is Turkish, but the models were mostly trained with English instructions.
Free-text args are never asked; they belong to an LLM. Context args (site_id) are filled here
from session state. Ordering of the resulting calls is deterministic (catalog precedence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Call, Catalog, Prediction, TestCase

NONE = "__none__"

# Args where one message can legitimately ask for several values -> one call per value.
MULTI_VALUE_ARGS = {("add_page", "page_type"), ("analyze_site", "analysis_type"), ("edit_section", "section")}

DOMAIN_RE = re.compile(r"(?<![@\w])(?:www\.)?((?:[a-z0-9çğıöşü](?:[a-z0-9çğıöşü-]*[a-z0-9çğıöşü])?\.)+(?:com\.tr|com|net|org|io|shop|dev|app|tr|co))\b", re.I)
HEX_RE = re.compile(r"#[0-9A-Za-z]{3,8}\b")
# Double quotes anywhere; single quotes only as real quotes: in Turkish an apostrophe usually marks a
# case suffix (SEO'sunu, kahvedurağı.com.tr'yi), so the opening quote must follow a space/start and the
# closing quote must not be followed by a letter.
QUOTED_RE = re.compile(r"[\"“«]([^\"”»]{2,60})[\"”»]|(?:(?<=\s)|^)['‘]([^'’]{2,60})['’](?![A-Za-zÇĞİÖŞÜçğıöşü])")


def extract_candidates(text: str) -> dict[str, list[str]]:
    domains = []
    for m in DOMAIN_RE.finditer(text):
        d = m.group(1).lower()
        if d not in domains:
            domains.append(d)
    hexes = list(dict.fromkeys(HEX_RE.findall(text)))
    quoted = list(dict.fromkeys((a or b).strip() for a, b in QUOTED_RE.findall(text)))
    return {"domain": domains, "hex": hexes, "quoted": quoted}


def span_candidate_type(tool: str, arg: str) -> str:
    if arg == "domain":
        return "domain"
    if arg == "primary_color":
        return "hex"
    return "quoted"  # site_name, title


@dataclass
class CompiledStep:
    state: dict
    questions: dict[str, dict]
    candidates: dict[str, list[str]]


class Compiler:
    def __init__(self, catalog: Catalog, *, include_need=True, include_args=True, max_options=255,
                 question_language: str = "tr"):
        self.cat = catalog
        self.lang = question_language
        self.include_need = include_need
        self.include_args = include_args
        self.max_options = max_options

    # ------------------------------------------------------------------ state
    @staticmethod
    def build_history(tc: TestCase, step_idx: int) -> list[dict]:
        """Teacher-forced history: gold calls of previous steps + the dataset's observations."""
        hist = []
        for i in range(step_idx):
            exp = tc.steps[i]["expected"]
            obs = tc.steps[i + 1]["observation"]
            hist.append({"step": i + 1, "assistant_called": [{"tool": c["tool"], "args": c["args"]} for c in exp["calls"]],
                         "tool_results": obs if isinstance(obs, list) else [obs]})
        return hist

    def build_state(self, tc: TestCase, step_idx: int, compact: bool = False) -> dict:
        c = tc.context
        session = {k: c.get(k) for k in ("site_exists", "current_page", "selected_section", "pages", "site_language", "domain", "published")}
        if compact:
            session = {k: v for k, v in session.items() if v not in (None, [], False)}
        st = {"user_message": tc.user_message, "session": session}
        hist = self.build_history(tc, step_idx)
        if hist:
            st["history"] = hist
        return st

    # -------------------------------------------------------------- questions
    TEXT = {
        "en": {
            "action": "What should the website-builder assistant do NEXT for `user_message`?",
            "hist": " Steps listed in `history` are already done; their results are in `tool_results`.",
            "pick": " Pick the single most important next action.",
            "no_tool_x": " Also choose this when everything the user asked for is already done in `history`, or the last tool result must simply be reported to the user (including failures that should not be retried).",
            "clarify_x": " Also choose this when a tool result in `history` shows the user must decide something before continuing.",
            "need": "Must the assistant run `{t}` in this SAME next step (it can run now, it is not already done in `history`, and it does not depend on a result that is not available yet)? `{t}`: {d}",
            "arg": "For the `{t}` action, which `{a}` does the user want? ({d})",
            "stated": "Does `user_message` say anything about `{a}` for `{t}` ({d})?",
            "multi": "Does the user want `{t}` with {a} = `{o}` ({d})?",
            "span": "Which value from the message is the `{a}` for `{t}`? ({d})",
            "none": "the user did not give a value for this",
        },
        "tr": {
            "action": "Web sitesi oluşturma asistanı `user_message` için SIRADAKİ adımda ne yapmalı?",
            "hist": " `history` içindeki adımlar zaten yapıldı; sonuçları `tool_results` içinde.",
            "pick": " Sıradaki en önemli tek eylemi seç.",
            "no_tool_x": " Kullanıcının istediği her şey `history` içinde zaten yapıldıysa ya da son araç sonucunun kullanıcıya sadece bildirilmesi gerekiyorsa (tekrar denenmemesi gereken hatalar dahil) da bunu seç.",
            "clarify_x": " `history` içindeki bir araç sonucu, devam etmeden önce kullanıcının bir karar vermesi gerektiğini gösteriyorsa da bunu seç.",
            "need": "Asistan `{t}` aracını bu AYNI sıradaki adımda çalıştırmalı mı (şu an çalıştırılabilir, `history` içinde zaten yapılmamış ve henüz elde olmayan bir sonuca bağlı değil)? `{t}`: {d}",
            "arg": "`{t}` işlemi için kullanıcı hangi `{a}` değerini istiyor? ({d})",
            "stated": "`user_message`, `{t}` için `{a}` hakkında bir şey söylüyor mu ({d})?",
            "multi": "Kullanıcı `{t}` işlemini {a} = `{o}` ({d}) ile istiyor mu?",
            "span": "Mesajdaki hangi değer `{t}` için `{a}`? ({d})",
            "none": "kullanıcı bunun için bir değer vermedi",
        },
    }

    def _tool_desc(self, t):
        return (t.description_tr or t.description) if self.lang == "tr" else t.description

    def _opt_desc(self, a, o):
        if self.lang == "tr" and a.option_descriptions_tr:
            return a.option_descriptions_tr.get(o)
        return a.option_descriptions.get(o)

    def questions(self, user_message: str, has_history: bool) -> tuple[dict, dict]:
        L = self.TEXT[self.lang]
        pseudo = self.cat.pseudo_tr if self.lang == "tr" else self.cat.pseudo
        q: dict[str, dict] = {}
        crit = {name: self._tool_desc(t) for name, t in self.cat.tools.items()}
        crit["no_tool"] = pseudo["no_tool"] + L["no_tool_x"]
        crit["clarify"] = pseudo["clarify"] + L["clarify_x"]
        q["action"] = {"type": "choice", "instructions": L["action"] + (L["hist"] if has_history else "") + L["pick"],
                       "criteria": crit}
        if self.include_need:
            for name, t in self.cat.tools.items():
                q[f"need.{name}"] = {"type": "noul", "instructions": L["need"].format(t=name, d=self._tool_desc(t))}
        cands = extract_candidates(user_message)
        if self.include_args:
            for name, t in self.cat.tools.items():
                for an, a in t.closed_args.items():
                    opts = {o: self._opt_desc(a, o) for o in a.enum}
                    if a.json_type == "boolean":
                        opts = {"true": None, "false": None}
                    q[f"{name}.{an}"] = {"type": "choice", "instructions": L["arg"].format(t=name, a=an, d=a.description or an),
                                         "criteria": opts}
                    if not a.required:
                        q[f"{name}.{an}?"] = {"type": "noul", "instructions": L["stated"].format(t=name, a=an, d=a.description or an)}
                    if (name, an) in MULTI_VALUE_ARGS:
                        for o in a.enum:
                            q[f"{name}.{an}={o}"] = {"type": "noul",
                                                    "instructions": L["multi"].format(t=name, a=an, o=o, d=self._opt_desc(a, o) or o)}
                for an, a in t.span_args.items():
                    pool = cands[span_candidate_type(name, an)]
                    if not pool:
                        continue
                    opts = {v: None for v in pool[: self.max_options - 1]}
                    opts[NONE] = L["none"]
                    q[f"{name}.{an}"] = {"type": "choice", "instructions": L["span"].format(t=name, a=an, d=a.description or an),
                                         "criteria": opts}
        return q, cands

    def compile(self, tc: TestCase, step_idx: int, compact_state: bool = False) -> CompiledStep:
        state = self.build_state(tc, step_idx, compact=compact_state)
        qs, cands = self.questions(tc.user_message, has_history=step_idx > 0)
        return CompiledStep(state=state, questions=qs, candidates=cands)

    # ----------------------------------------------------------------- decode
    def decode(self, answers: dict, tc: TestCase, *, need_threshold=0.5, stated_threshold=0.5,
               gate_threshold: float | None = None) -> Prediction:
        act = answers.get("action")
        if not act:
            return Prediction(decision="error", error="missing 'action' answer")
        choice, conf = act.get("choice"), act.get("confidence")
        if gate_threshold is not None and conf is not None and conf < gate_threshold and choice not in ("no_tool", "clarify"):
            return Prediction(decision="clarify", confidence=conf)
        if choice in ("no_tool", "clarify"):
            return Prediction(decision=choice, confidence=conf)
        tools = [choice]
        for name in self.cat.tools:
            a = answers.get(f"need.{name}")
            if name != choice and a and a.get("noul", 0) >= need_threshold:
                tools.append(name)
        calls: list[Call] = []
        argconf: dict[str, float] = {}
        for name in sorted(tools, key=self.cat.precedence):
            t = self.cat.tools[name]
            base: dict = {}
            multi_values: list[str] | None = None
            for an, a in t.args.items():
                if a.kind == "context":
                    continue  # filled by orchestrator code; not scored
                ans = answers.get(f"{name}.{an}")
                if not ans:
                    continue
                if a.kind == "closed":
                    if not a.required:
                        st = answers.get(f"{name}.{an}?")
                        if st and st.get("noul", 1) < stated_threshold:
                            continue
                    val = ans.get("choice")
                    if a.json_type == "boolean":
                        val = val == "true"
                    base[an] = val
                    if ans.get("confidence") is not None:
                        argconf[f"{name}.{an}"] = ans["confidence"]
                    if (name, an) in MULTI_VALUE_ARGS:
                        vals = [o for o in a.enum if (answers.get(f"{name}.{an}={o}") or {}).get("noul", 0) >= 0.5]
                        if len(vals) > 1 and val in vals:
                            multi_values = [val] + [v for v in vals if v != val]
                            multi_key = an
                elif a.kind == "span":
                    val = ans.get("choice")
                    if val and val != NONE:
                        base[an] = val
            if multi_values:
                for v in multi_values:
                    calls.append(Call(name, {**base, multi_key: v}))
            else:
                calls.append(Call(name, base))
        return Prediction(decision="call", calls=calls, confidence=conf, arg_confidence=argconf)
