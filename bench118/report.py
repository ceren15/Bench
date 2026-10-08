"""Markdown + CSV results report (the 'Results Template', filled automatically)."""
from __future__ import annotations

import csv
from pathlib import Path


def _f(v, pct=False, nd=1):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{nd}f}%"
    return f"{v:.3f}" if abs(v) < 1 and v != 0 else f"{v:.{nd}f}"


MAIN = [  # (label, key, pct, higher_is_better)
    ("Routing Accuracy", "routing_accuracy", True), ("Routing macro-F1", "routing_macro_f1", True),
    ("Tool Selection Acc.", "tool_selection_acc", True), ("Argument Acc. (koşullu)", "argument_accuracy", True),
    ("Argument Acc. (uçtan uca)", "argument_acc_e2e", True), ("Uydurma argüman oranı", "spurious_arg_rate", True),
    ("Multi-Tool Success", "multi_tool_success", True), ("Workflow Completion", "workflow_completion", True),
    ("False Tool Call Rate", "false_tool_call_rate", True), ("Missed Call Rate", "missed_call_rate", True),
    ("Test Pass Rate", "test_pass_rate", True), ("ECE (kalibrasyon hatası)", "ece", False),
    ("%95 hassasiyette kapsama", "coverage_at_95_precision", True), ("Tutarlılık (tekrarlar)", "consistency", True),
    ("Latency p50 (ms)", "latency_ms_p50", False), ("Latency p95 (ms)", "latency_ms_p95", False),
    ("Throughput (adım/sn)", "throughput_rps", False), ("Maliyet / 1k adım (USD)", "cost_per_1k_steps_usd", False),
    ("Maliyet / başarılı test (USD)", "cost_per_success_usd", False), ("Hata (exception)", "errors", False),
]


def write(results: list[dict], meta: dict, out: Path) -> Path:
    out = Path(out)
    names = [r["model"] for r in results]
    L = [f"# 118.studio Karar Modeli Benchmark Sonuçları", "",
         f"- Veri seti: `{meta.get('dataset')}` (sha `{meta.get('dataset_sha')}`) · araç kataloğu sha `{meta.get('tools_sha')}`",
         f"- Test: {meta.get('n_tests')} · adım: {meta.get('n_steps')} · tekrar: {meta['config'].get('repeats')} · "
         f"soru dili: `{meta['config'].get('question_language')}` · seçenek sırası karıştırma: {meta['config'].get('perturb_option_order')}",
         f"- Platform: {meta.get('platform')} · başlangıç: {meta.get('started')}", "",
         "## Genel puan", "",
         "| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |",
         "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -(r["score"]["overall"] or 0)):
        s = r["score"]
        L.append(f"| {r['model']} | {_f(s['quality'])} | {_f(s['efficiency'])} | **{_f(s['overall'])}** | "
                 f"{'✅' if s['production_eligible'] else '❌'} | {', '.join(s['failed_gates']) or '—'} |")
    L += ["", "## Metrikler", "", "| Metrik | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, pct in MAIN:
        L.append(f"| {label} | " + " | ".join(_f(r["metrics"].get(key), pct and key not in ("errors",), 1 if pct else 1) for r in results) + " |")
    for dim, title in (("by_category", "Kategoriye göre adım doğruluğu (step exact)"),
                       ("by_difficulty", "Zorluğa göre adım doğruluğu"),
                       ("by_variant", "Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)")):
        keys = sorted({k for r in results for k in r[dim]})
        L += ["", f"## {title}", "", "| Grup | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(_f(r[dim].get(k, {}).get("step_exact"), True) for r in results) + " |")
    L += ["", "## Kaynak kullanımı (yalnızca yerel modeller)", "", "| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |", "|---|---|---|---|---|"]
    for r in results:
        rs = r["metrics"].get("resources", {})
        L.append(f"| {r['model']} | {_f(rs.get('cpu_pct_mean'))} | {_f(rs.get('rss_mb_max'))} | {_f(rs.get('gpu_util_pct_mean'))} | {_f(rs.get('gpu_mem_mb_max'))} |")
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "quality", "efficiency", "overall", "production_eligible"] + [k for _, k, _ in MAIN])
        for r in results:
            s = r["score"]
            w.writerow([r["model"], s["quality"], s["efficiency"], s["overall"], s["production_eligible"]]
                       + [r["metrics"].get(k) for _, k, _ in MAIN])
    return md
