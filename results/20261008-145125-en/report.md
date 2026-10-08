# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/dataset_en_control.jsonl` (sha `c9c1a086ec09665a`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 142 · adım: 228 · tekrar: 3 · soru dili: `en` · seçenek sırası karıştırma: True
- Platform: Linux-6.18.44-fc-v80-x86_64-with-glibc2.39 · başlangıç: 2026-10-08T14:51:25+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| mock-keyword | 46.6 | 90.7 | **55.4** | ❌ | false_tool_call_rate > 0.05, routing_accuracy < 0.85 |

## Metrikler

| Metrik | mock-keyword |
|---|---|
| Routing Accuracy | 71.1% |
| Routing macro-F1 | 39.8% |
| Tool Selection Acc. | 59.4% |
| Argument Acc. (koşullu) | 74.7% |
| Argument Acc. (uçtan uca) | 53.4% |
| Uydurma argüman oranı | 4.0% |
| Multi-Tool Success | 13.6% |
| Workflow Completion | 0.0% |
| False Tool Call Rate | 88.7% |
| Missed Call Rate | 2.9% |
| Test Pass Rate | 24.6% |
| ECE (kalibrasyon hatası) | 0.317 |
| %95 hassasiyette kapsama | 0.1% |
| Tutarlılık (tekrarlar) | 46.5% |
| Latency p50 (ms) | 28.5 |
| Latency p95 (ms) | 51.0 |
| Throughput (adım/sn) | 405.9 |
| Maliyet / 1k adım (USD) | 0.0 |
| Maliyet / başarılı test (USD) | 0.0 |
| Hata (exception) | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | mock-keyword |
|---|---|
| A_single_tool | 67.6% |
| B_similar_tools | 16.7% |
| C_argument_extraction | 0.0% |
| D_multi_tool | 30.0% |
| E_sequential | 30.4% |
| F_no_tool | 29.8% |
| G_ambiguous | 8.9% |
| H_error_recovery | 51.4% |
| I_complex_workflow | 0.0% |

## Zorluğa göre adım doğruluğu

| Grup | mock-keyword |
|---|---|
| easy | 65.8% |
| hard | 13.7% |
| medium | 28.3% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | mock-keyword |
|---|---|
| clean | 28.4% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| mock-keyword | 101.2 | 40.7 | — | — |
