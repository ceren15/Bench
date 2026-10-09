# 118.studio Karar Modeli Benchmark Sonuçları

- Veri seti: `data/laya_ft/dataset_test.jsonl` (sha `cde91f77bccddaec`) · araç kataloğu sha `be4207c37dd8628f`
- Test: 192 · adım: 314 · tekrar: 3 · soru dili: `tr` · seçenek sırası karıştırma: True
- Platform: Linux-6.18.48+-x86_64-with-glibc2.39 · başlangıç: 2026-10-09T10:44:49+0000

## Genel puan

| Model | Kalite (0–100) | Verimlilik (0–100) | Genel | Üretime uygun | Kapıdan kalan |
|---|---|---|---|---|---|
| laya-ml-ft | 81.2 | 52.1 | **75.3** | ❌ | false_tool_call_rate > 0.05 |
| laya-ml-base | 22.8 | 52.1 | **28.7** | ❌ | false_tool_call_rate > 0.05, routing_accuracy < 0.85 |

## Metrikler

| Metrik | laya-ml-base | laya-ml-ft |
|---|---|---|
| Routing Accuracy | 70.7% | 88.4% |
| Routing macro-F1 | 27.6% | 75.1% |
| Tool Selection Acc. | 0.0% | 86.7% |
| Argument Acc. (koşullu) | 46.5% | 92.6% |
| Argument Acc. (uçtan uca) | 44.7% | 88.2% |
| Uydurma argüman oranı | 0.6% | 0.2% |
| Multi-Tool Success | 0.0% | 75.5% |
| Workflow Completion | 0.0% | 56.1% |
| False Tool Call Rate | 99.6% | 17.2% |
| Missed Call Rate | 0.4% | 4.9% |
| Test Pass Rate | 0.0% | 63.7% |
| ECE (kalibrasyon hatası) | 0.833 | 0.200 |
| %95 hassasiyette kapsama | 0.0% | 0.7% |
| Tutarlılık (tekrarlar) | 3.5% | 79.3% |
| Latency p50 (ms) | 1534.2 | 1525.5 |
| Latency p95 (ms) | 2438.7 | 2429.3 |
| Throughput (adım/sn) | 0.583 | 0.586 |
| Maliyet / 1k adım (USD) | 0.0 | 0.0 |
| Maliyet / başarılı test (USD) | — | 0.0 |
| Hata (exception) | 0.0 | 0.0 |

## Kategoriye göre adım doğruluğu (step exact)

| Grup | laya-ml-base | laya-ml-ft |
|---|---|---|
| A_single_tool | 0.0% | 80.8% |
| B_similar_tools | 0.0% | 70.0% |
| C_argument_extraction | 0.0% | 61.3% |
| D_multi_tool | 0.0% | 88.2% |
| E_sequential | 0.0% | 79.5% |
| F_no_tool | 0.0% | 64.8% |
| G_ambiguous | 0.0% | 40.0% |
| H_error_recovery | 0.0% | 80.5% |
| I_complex_workflow | 0.0% | 81.1% |

## Zorluğa göre adım doğruluğu

| Grup | laya-ml-base | laya-ml-ft |
|---|---|---|
| easy | 0.0% | 81.2% |
| hard | 0.0% | 74.9% |
| medium | 0.0% | 73.0% |

## Türkçe varyanta göre adım doğruluğu (clean / noisy_ascii / code_mixed / tr_suffix)

| Grup | laya-ml-base | laya-ml-ft |
|---|---|---|
| clean | 0.0% | 73.5% |
| code_mixed | 0.0% | 66.7% |
| noisy_ascii | 0.0% | 78.1% |
| tr_suffix | 0.0% | 100.0% |

## Kaynak kullanımı (yalnızca yerel modeller)

| Model | CPU % ort. | RSS MB maks. | GPU % ort. | GPU MB maks. |
|---|---|---|---|---|
| laya-ml-base | 102.2 | 3350.7 | 97.6 | 6669.0 |
| laya-ml-ft | 102.0 | 3350.7 | 97.8 | 6669.0 |
