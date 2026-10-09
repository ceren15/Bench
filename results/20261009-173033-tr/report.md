# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/laya_ft/dataset_test.jsonl` (sha `cde91f77bccddaec`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 192 · adım: 314 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: macOS-26.7.1-arm64-arm-64bit · başlangıç: 2026-10-09T17:30:33+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| qwen3.5-9b-modal | 72.2 | 15.3 | **60.8** | ❌ | false_tool_call_rate > 0.05 |

## Metrikler

| Metrik | qwen3.5-9b-modal |
|---|---|
| Routing Accuracy | 88.3% |
| Routing macro-F1 | 70.0% |
| Tool Selection Acc. | 90.6% |
| Argument Acc. (koşullu) | 95.7% |
| Argument Acc. (uçtan uca) | 92.2% |
| Uydurma argüman oranı | 18.8% |
| Multi-Tool Success | 45.6% |
| Workflow Completion | 25.0% |
| False Tool Call Rate | 35.0% |
| Missed Call Rate | 0.0% |
| Test Pass Rate | 68.8% |
| ECE (kalibrasyon hatası) | — |
| %95 hassasiyette kapsama | — |
| Tutarlılık (tekrarlar) | 94.3% |
| Latency p50 (ms) | 3118.6 |
| Latency p95 (ms) | 6420.4 |
| Throughput (adım/sn) | 1.5 |
| Maliyet / 1k adım (USD) | 0.372 |
| Maliyet / başarılı test (USD) | 0.001 |
| Hata (exception) | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | qwen3.5-9b-modal |
|---|---|
| A_single_tool | 100.0% |
| B_similar_tools | 95.0% |
| C_argument_extraction | 100.0% |
| D_multi_tool | 44.9% |
| E_sequential | 65.1% |
| F_no_tool | 100.0% |
| G_ambiguous | 65.0% |
| H_error_recovery | 63.7% |
| I_complex_workflow | 70.1% |

## Zorluğa göre adım doğruluğu

| Grup | qwen3.5-9b-modal |
|---|---|
| easy | 91.2% |
| hard | 72.4% |
| medium | 79.1% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | qwen3.5-9b-modal |
|---|---|
| clean | 74.6% |
| code_mixed | 100.0% |
| noisy_ascii | 76.6% |
| tr_suffix | 100.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| qwen3.5-9b-modal | — | — | — | — |
