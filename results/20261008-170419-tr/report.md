# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `/private/tmp/claude-502/-Users-cerenkarabag-Downloads-bench118/90c5bb7b-111e-4427-90ae-bec3357c5ceb/scratchpad/sample27.jsonl` (sha `394dc1096e8a9558`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 27 · adım: 50 · tekrar: 1 · soru dili: `tr` · seçenek sırası karıştırma: False
- Platform: macOS-26.7.1-arm64-arm-64bit · başlangıç: 2026-10-08T17:04:19+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| laya-vercel | 24.6 | 54.1 | **30.5** | ❌ | false_tool_call_rate > 0.05, routing_accuracy < 0.85 |

## Metrikler

| Metrik | laya-vercel |
|---|---|
| Routing Accuracy | 70.0% |
| Routing macro-F1 | 28.5% |
| Tool Selection Acc. | 0.0% |
| Argument Acc. (koşullu) | 34.3% |
| Argument Acc. (uçtan uca) | 34.3% |
| Uydurma argüman oranı | 0.6% |
| Multi-Tool Success | 0.0% |
| Workflow Completion | 0.0% |
| False Tool Call Rate | 80.0% |
| Missed Call Rate | 0.0% |
| Test Pass Rate | 0.0% |
| ECE (kalibrasyon hatası) | 0.656 |
| %95 hassasiyette kapsama | 0.0% |
| Tutarlılık (tekrarlar) | — |
| Latency p50 (ms) | 1018.4 |
| Latency p95 (ms) | 1989.4 |
| Throughput (adım/sn) | 0.042 |
| Maliyet / 1k adım (USD) | 0.0 |
| Maliyet / başarılı test (USD) | — |
| Hata (exception) | 3.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | laya-vercel |
|---|---|
| A_single_tool | 0.0% |
| B_similar_tools | 0.0% |
| C_argument_extraction | 0.0% |
| D_multi_tool | 0.0% |
| E_sequential | 0.0% |
| F_no_tool | 0.0% |
| G_ambiguous | 0.0% |
| H_error_recovery | 0.0% |
| I_complex_workflow | 0.0% |

## Zorluğa göre adım doğruluğu

| Grup | laya-vercel |
|---|---|
| easy | 0.0% |
| hard | 0.0% |
| medium | 0.0% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | laya-vercel |
|---|---|
| clean | 0.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| laya-vercel | — | — | — | — |
