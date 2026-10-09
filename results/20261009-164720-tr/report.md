# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/laya_ft/dataset_test.jsonl` (sha `cde91f77bccddaec`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 192 · adım: 314 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: macOS-26.7.1-arm64-arm-64bit · başlangıç: 2026-10-09T16:47:20+0300

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| kev-9b-modal | 55.3 | 18.9 | **48.0** | ❌ | false_tool_call_rate > 0.05, routing_accuracy < 0.85 |

## Metrikler

| Metrik | kev-9b-modal |
|---|---|
| Routing Accuracy | 81.3% |
| Routing macro-F1 | 62.9% |
| Tool Selection Acc. | 29.3% |
| Argument Acc. (koşullu) | 92.3% |
| Argument Acc. (uçtan uca) | 82.5% |
| Uydurma argüman oranı | 1.3% |
| Multi-Tool Success | 29.4% |
| Workflow Completion | 7.8% |
| False Tool Call Rate | 26.7% |
| Missed Call Rate | 10.5% |
| Test Pass Rate | 37.7% |
| ECE (kalibrasyon hatası) | 0.214 |
| %95 hassasiyette kapsama | 0.0% |
| Tutarlılık (tekrarlar) | 67.2% |
| Latency p50 (ms) | 1536.7 |
| Latency p95 (ms) | 2711.2 |
| Throughput (adım/sn) | 4.6 |
| Maliyet / 1k adım (USD) | 0.237 |
| Maliyet / başarılı test (USD) | 0.001 |
| Hata (exception) | 1.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | kev-9b-modal |
|---|---|
| A_single_tool | 43.6% |
| B_similar_tools | 28.3% |
| C_argument_extraction | 43.0% |
| D_multi_tool | 58.8% |
| E_sequential | 31.3% |
| F_no_tool | 92.6% |
| G_ambiguous | 53.3% |
| H_error_recovery | 53.7% |
| I_complex_workflow | 11.0% |

## Zorluğa göre adım doğruluğu

| Grup | kev-9b-modal |
|---|---|
| easy | 66.7% |
| hard | 29.0% |
| medium | 49.8% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | kev-9b-modal |
|---|---|
| clean | 39.4% |
| code_mixed | 25.0% |
| noisy_ascii | 30.8% |
| tr_suffix | 60.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| kev-9b-modal | — | — | — | — |
