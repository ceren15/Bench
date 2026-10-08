# bench118 — 118.studio Karar Modeli & Tool Calling Benchmark

Jev (TypeSafe), Clef / Clef-flash (Cloudflare), Kev (açık kaynak) ve Laya (Convai) gibi **System One / karar modellerini**
— ve kıyas için bir LLM'i — 118.studio'nun gerçek araç kataloğu üzerinde, **Türkçe** kullanıcı mesajlarıyla aynı koşullarda ölçer.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install pyyaml psutil pytest          # çekirdek: yalnızca bunlar (HTTP için stdlib urllib kullanılır)
pip install laya                          # sadece Laya'yı yerelde çalıştıracaksanız (TensorFlow kurulu OLMAYAN venv'de)
```

## Hızlı başlangıç

```bash
python data/generate_dataset.py          # veri setini deterministik üretir (seed=118)
python -m bench118 validate              # veri seti kontrolü (CI'da da çalıştırın)
python tests/test_pipeline.py            # oracle testi: kusursuz cevaplar ~%100 almalı
python -m bench118 run --models mock-keyword                 # anahtar gerektirmeyen kuru çalıştırma
python -m bench118 compile --test E-0001 --step 2            # bir modelin tam olarak ne gördüğünü göster
```

Gerçek modeller: `configs/models.yaml` içinde `enabled: true` yapın, ortam değişkenlerini verin
(`TYPESAFE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_AUTH_TOKEN`, `OPENAI_API_KEY`) ve:

```bash
python -m bench118 run                                       # Türkçe ana set, Türkçe sorular
python -m bench118 run --question-language en                # Türkçe mesaj + İngilizce soru metni
python -m bench118 run --dataset data/dataset_en_control.jsonl --question-language en   # İngilizce kontrol seti
```

Her çalıştırma `results/<zaman>-<dil>/` altına `report.md`, `summary.csv`, model başına `result_*.json`,
ham satırlar `rows_*.jsonl` ve tekrar üretilebilirlik için `run_meta.json` (veri seti + katalog hash'i, config, platform) yazar.

## Dosyalar

| Yol | İçerik |
|---|---|
| `data/tools.json` | 9 araç, JSON Schema (function-calling uyumlu) + `x-arg-kind`, `x-precedence`, `x-side-effect`, Türkçe açıklamalar |
| `data/dataset.jsonl` | **637 Türkçe test / 1.017 adım**, A–I kategorileri, 3 zorluk; `noisy_ascii`, `code_mixed`, `tr_suffix` varyantları |
| `data/dataset_en_control.jsonl` | 142 İngilizce kontrol testi (Türkçe cezasını ölçmek için) |
| `data/generate_dataset.py` | Veri seti üreticisi — yeni senaryo eklemek için buraya fonksiyon ekleyin |
| `bench118/compiler.py` | Araç kataloğu → choice/noul soruları; cevaplar → tool call |
| `bench118/adapters/` | `systemone_http` (Jev, Clef, Kev), `laya_local`, `openai_llm` (LLM kıyası), `mock` |
| `bench118/scoring.py`, `metrics.py` | Adım puanlama, metrikler, ağırlıklı benchmark puanı, üretim kapıları |
| `bench118/runner.py` | Eşzamanlılık, tekrarlar, ısınma, seçenek sırası karıştırma, kaynak ölçümü |
| `tests/test_pipeline.py` | Oracle testi + Türkçe ek/normalizasyon testleri |

## Bilinen sınırlar

- Karar modelleri metin üretmez: `free` argümanlar (açıklama, prompt, talimat) bu modellerde puanlanmaz; LLM katmanına aittir.
- Çok adımlı testler **teacher forcing** ile puanlanır (her adım altın geçmişi görür). Serbest koşum için bir araç simülatörü gerekir.
- Tek mesajda farklı `change_type` ile iki `edit_section` istenmesi tek bir choice ile ifade edilemez (oracle tavanı %99,9 — D-0007).
- Clef REST yanıt sarmalayıcısı ve Laya `predict()` dönüş biçimi ilk çalıştırmada doğrulanmalı (`store_answers: true`).
