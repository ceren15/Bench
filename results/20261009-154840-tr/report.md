# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/laya_ft/dataset_test.jsonl` (sha `cde91f77bccddaec`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 192 · adım: 314 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: macOS-26.7.1-arm64-arm-64bit · başlangıç: 2026-10-09T15:48:40+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| clef | 67.0 | 8.2 | **55.2** | ❌ | false_tool_call_rate > 0.05 |

## Metrikler

| Metrik | clef |
|---|---|
| Routing Accuracy | 86.0% |
| Routing macro-F1 | 69.9% |
| Tool Selection Acc. | 84.6% |
| Argument Acc. (koşullu) | 93.2% |
| Argument Acc. (uçtan uca) | 84.9% |
| Uydurma argüman oranı | 1.2% |
| Multi-Tool Success | 13.0% |
| Workflow Completion | 19.4% |
| False Tool Call Rate | 35.4% |
| Missed Call Rate | 0.0% |
| Test Pass Rate | 59.4% |
| ECE (kalibrasyon hatası) | 0.175 |
| %95 hassasiyette kapsama | 11.2% |
| Tutarlılık (tekrarlar) | 99.4% |
| Latency p50 (ms) | 1777.8 |
| Latency p95 (ms) | 2192.2 |
| Throughput (adım/sn) | 3.7 |
| Maliyet / 1k adım (USD) | 2.1 |
| Maliyet / başarılı test (USD) | 0.006 |
| Hata (exception) | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | clef |
|---|---|
| A_single_tool | 92.3% |
| B_similar_tools | 100.0% |
| C_argument_extraction | 100.0% |
| D_multi_tool | 23.5% |
| E_sequential | 64.6% |
| F_no_tool | 77.8% |
| G_ambiguous | 40.0% |
| H_error_recovery | 61.0% |
| I_complex_workflow | 60.5% |

## Zorluğa göre adım doğruluğu

| Grup | clef |
|---|---|
| easy | 78.3% |
| hard | 67.3% |
| medium | 68.5% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | clef |
|---|---|
| clean | 66.3% |
| code_mixed | 75.0% |
| noisy_ascii | 74.0% |
| tr_suffix | 100.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| clef | — | — | — | — |
