# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/dataset.jsonl` (sha `a91e9dac9f1cc91e`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 637 · adım: 1017 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: Linux-6.18.44-fc-v80-x86_64-with-glibc2.39 · başlangıç: 2026-10-08T14:51:17+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| mock-keyword | 40.3 | 91.8 | **50.6** | ❌ | false_tool_call_rate > 0.05, routing_accuracy < 0.85 |

## Metrikler

| Metrik | mock-keyword |
|---|---|
| Routing Accuracy | 71.1% |
| Routing macro-F1 | 34.9% |
| Tool Selection Acc. | 49.7% |
| Argument Acc. (koşullu) | 61.2% |
| Argument Acc. (uçtan uca) | 47.1% |
| Uydurma argüman oranı | 0.0% |
| Multi-Tool Success | 2.0% |
| Workflow Completion | 0.0% |
| False Tool Call Rate | 91.1% |
| Missed Call Rate | 1.6% |
| Test Pass Rate | 18.5% |
| ECE (kalibrasyon hatası) | 0.416 |
| %95 hassasiyette kapsama | 0.0% |
| Tutarlılık (tekrarlar) | 40.9% |
| Latency p50 (ms) | 31.1 |
| Latency p95 (ms) | 45.5 |
| Throughput (adım/sn) | 386.7 |
| Maliyet / 1k adım (USD) | 0.0 |
| Maliyet / başarılı test (USD) | 0.0 |
| Hata (exception) | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | mock-keyword |
|---|---|
| A_single_tool | 46.1% |
| B_similar_tools | 26.7% |
| C_argument_extraction | 39.9% |
| D_multi_tool | 4.0% |
| E_sequential | 19.6% |
| F_no_tool | 19.1% |
| G_ambiguous | 8.8% |
| H_error_recovery | 38.4% |
| I_complex_workflow | 0.0% |

## Zorluğa göre adım doğruluğu

| Grup | mock-keyword |
|---|---|
| easy | 34.9% |
| hard | 17.4% |
| medium | 23.3% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | mock-keyword |
|---|---|
| clean | 17.3% |
| code_mixed | 56.4% |
| noisy_ascii | 20.8% |
| tr_suffix | 100.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| mock-keyword | 100.1 | 97.3 | — | — |
