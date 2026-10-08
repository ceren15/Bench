"""
Deterministic keyword-matching stand-in for a decision model. It does NOT read the gold labels.
Use it to dry-run the whole pipeline (compiler → runner → scorer → report) before you have API
keys or GPUs. Its scores are a floor, not a model result.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time

from ..schema import AdapterResponse
from .base import Adapter, Capabilities

SYN = {
    "create_site": "site kur yap oluştur sıfırdan build create website start new site lazım istiyorum",
    "add_page": "sayfa ekle sayfası page add ekleyelim",
    "edit_section": "bölüm kısmı kaldır değiştir section remove banner footer header hero form",
    "update_theme": "tüm bütün renk renkleri tema font palet theme colours colors whole",
    "generate_content": "yaz metin yazı açıklama slogan write copy text post blog",
    "generate_image": "görsel logo üret image generate illustration favicon kapak",
    "analyze_site": "analiz kontrol test denetim seo hız performans audit check erişilebilirlik mobil",
    "manage_domain": "domain alan adı adres boşta satın bağla .com connect register",
    "publish_site": "yayına canlı önizleme publish live preview link yayınla",
    "no_tool": "merhaba teşekkür thanks hello nasıl ne nedir sence why what how günaydın",
    "clarify": "bunu şey bir",
}


_TR = str.maketrans("çğıöşü", "cgiosu")


def _tok(s: str) -> set[str]:
    s = s.replace("İ", "i").replace("I", "ı").lower().translate(_TR)
    return set(re.findall(r"[\w#.]+", s))


def _score(text: set[str], key: str, desc: str | None) -> float:
    words = _tok(key.replace("_", " ") + " " + (desc or "") + " " + SYN.get(key, ""))
    hits = sum(1 for w in words if any(t.startswith(w[:5]) for t in text if len(w) > 2))
    return hits


def _softmax(xs, temp=1.0):
    m = max(xs)
    e = [math.exp((x - m) / temp) for x in xs]
    s = sum(e)
    return [v / s for v in e]


class MockKeywordAdapter(Adapter):
    def __init__(self, name="mock-keyword", simulated_latency_ms: float = 3.0):
        self.name = name
        self.sim = simulated_latency_ms
        self.caps = Capabilities(local=True, max_concurrency=16, notes=["keyword heuristic, not a model"])

    async def ask(self, state: dict, questions: dict[str, dict]) -> AdapterResponse:
        t0 = time.perf_counter()
        text = _tok(state.get("user_message", ""))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                sc = [_score(text, k, q["criteria"][k] if isinstance(q["criteria"][k], str) else None) for k in keys]
                if qid == "action" and max(sc) == 0:
                    sc[keys.index("clarify")] += 1
                pr = _softmax(sc, 0.7)
                i = max(range(len(keys)), key=lambda j: pr[j])
                answers[qid] = {"type": "choice", "choice": keys[i], "probabilities": dict(zip(keys, pr)), "confidence": pr[i]}
            elif q["type"] == "noul":
                h = int(hashlib.md5((qid + state.get("user_message", "")).encode()).hexdigest(), 16) % 100
                tool = qid.split(".")[1] if qid.startswith("need.") else None
                p = 0.15 + 0.01 * (h % 10)
                if tool and _score(text, tool, None) >= 2:
                    p = 0.6
                if qid.endswith("?"):
                    p = 0.55 if _score(text, qid.split(".")[1].rstrip("?"), None) else 0.3
                answers[qid] = {"type": "noul", "noul": p}
            else:
                answers[qid] = {"type": "score", "score": 0.0, "confidence": 0.5}
        await asyncio.sleep(self.sim / 1000)
        return AdapterResponse(answers=answers, latency_ms=(time.perf_counter() - t0) * 1000,
                               usage={"input_tokens": len(str(state)) // 4 + len(str(questions)) // 4, "output_tokens": 0})
