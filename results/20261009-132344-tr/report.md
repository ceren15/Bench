# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/laya_ft/dataset_test.jsonl` (sha `cde91f77bccddaec`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 192 · adım: 314 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: macOS-26.7.1-arm64-arm-64bit · başlangıç: 2026-10-09T13:23:44+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| jev | 74.0 | 37.3 | **66.7** | ❌ | false_tool_call_rate > 0.05 |

## Metrikler

| Metrik | jev |
|---|---|
| Routing Accuracy | 93.0% |
| Routing macro-F1 | 83.7% |
| Tool Selection Acc. | 79.4% |
| Argument Acc. (koşullu) | 91.1% |
| Argument Acc. (uçtan uca) | 85.9% |
| Uydurma argüman oranı | 2.3% |
| Multi-Tool Success | 34.3% |
| Workflow Completion | 33.3% |
| False Tool Call Rate | 8.8% |
| Missed Call Rate | 1.6% |
| Test Pass Rate | 66.8% |
| ECE (kalibrasyon hatası) | 0.169 |
| %95 hassasiyette kapsama | 2.7% |
| Tutarlılık (tekrarlar) | 86.9% |
| Latency p50 (ms) | 333.3 |
| Latency p95 (ms) | 407.7 |
| Throughput (adım/sn) | 22.9 |
| Maliyet / 1k adım (USD) | 0.259 |
| Maliyet / başarılı test (USD) | 0.001 |
| Hata (exception) | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | jev |
|---|---|
| A_single_tool | 91.0% |
| B_similar_tools | 85.0% |
| C_argument_extraction | 90.3% |
| D_multi_tool | 52.9% |
| E_sequential | 69.7% |
| F_no_tool | 81.5% |
| G_ambiguous | 80.0% |
| H_error_recovery | 85.4% |
| I_complex_workflow | 42.1% |

## Zorluğa göre adım doğruluğu

| Grup | jev |
|---|---|
| easy | 89.9% |
| hard | 61.2% |
| medium | 85.8% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | jev |
|---|---|
| clean | 70.9% |
| code_mixed | 50.0% |
| noisy_ascii | 67.2% |
| tr_suffix | 100.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| jev | — | — | — | — |
