# Deploy

Ağırlıklar git'e **girmez** — 423 MB'lık checkpoint Hub'da bir model reposunda durur.
Repo zaten 62 MB tokenize veri taşıyor, üstüne modeli koymanın anlamı yok.

## Durum

| Parça | Durum |
| --- | --- |
| Tokenize veri (263.292 train / 65.000 val, 3 sınıf) | Hazır, repoda, sızıntı temizlenmiş |
| Eğitim pipeline'ı | Çalışıyor, kesintiden devam ediyor |
| Eğitilmiş checkpoint | **Hazır** — 0.8263 macro F1 |
| Hub'da model | **Yayında** — [bariskirat/duyguanalizi-berturk](https://huggingface.co/bariskirat/duyguanalizi-berturk) |
| Inference katmanı (`src/inference/predictor.py`) | Hazır, test edildi |
| Web arayüzü (`app.py`) | Hazır, yerelde çalışıyor |
| Barındırılan arayüz | **Yok** — ücretsiz seçenek kalmadı, aşağıya bak |

Sıfırdan başlıyorsan 1. adım eğitimi üretir. Elindeki modeli yayınlamak için doğrudan
"Model yayınlamak" bölümüne geçebilirsin.

## 1. Eğit

### Apple Silicon'da yerel eğitim (önerilen)

M-serisi bir Mac'te bu iş rahatça dönüyor. `bert_train.py` MPS'i tanıyor ve
Apple GPU üzerinde eğitiyor; `fp16` otomatik olarak devre dışı kalır (Apple
Silicon'da ölçülebilir bir hız kazancı yok, bkz. aşağıdaki tablo).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/models/bert/bert_train.py
```

M4 Pro (20 GPU çekirdeği, 24 GB birleşik bellek) üzerinde ölçülen değerler —
gerçek veriyle, dinamik padding devrede, her konfigürasyon ayrı süreçte:

| Ayar | Adım süresi | Örnek/s | 3 epoch |
| --- | --- | --- | --- |
| `batch_size: 16` fp32 | 222 ms | 72.1 | **~3.0 saat** |
| `batch_size: 16` fp16 | 226 ms | 70.7 | ~3.1 saat |
| `batch_size: 32` fp32 | 508 ms | 63.0 | ~3.5 saat |
| `batch_size: 32` fp16 | 489 ms | 65.5 | ~3.3 saat |

Mevcut ayar (`batch_size: 16`) en hızlısı — batch büyütmek burada işe yaramıyor,
çünkü dinamik padding ile büyük batch'te batch içi en uzun dizi de uzuyor ve
boşa giden padding artıyor (16'da ortalama 105, 32'de 127 token).

Bunlar kısa ölçümler; üç saatlik sürekli yükte laptop bir miktar throttle eder,
gerçekçi beklenti 3-4 saat. Eğitim boyunca diğer ağır uygulamaları kapat, 24 GB
birleşik bellek GPU ile paylaşılıyor.

### Colab alternatifi

Mac'i üç saat meşgul etmek istemiyorsan
[notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) hazır. Colab'da doğrudan
açmak için:

```
https://colab.research.google.com/github/bariskiratt/duyguanalizi/blob/main/notebooks/train_colab.ipynb
```

Defter repoyu klonladığı için **önce yerel düzeltmeleri push etmen gerekir.** T4'te
süre yerel M4 Pro ile kabaca aynı; kazandığın şey Mac'in serbest kalması, ödediğin
bedel oturum kopma riski ve Drive senkronizasyonu.

### Her iki durumda

Çıktı: `artifacts/bert_mlp_ckpt/best_model/` (`model.safetensors` + tokenizer).

Eğitim yarıda kesilirse scripti yeniden çalıştırmak yeterli — `artifacts/bert_mlp_ckpt`
içindeki son checkpoint'i bulup kaldığı yerden devam eder. Colab'da bunun işe
yaraması için checkpoint klasörünün Drive'a bağlı olması gerekir (defterin 2. hücresi).

Checkpoint'ler optimizer state'i içerdiği için tanesi ~1.3 GB; `save_total_limit: 2`
ile en fazla ~2.7 GB tutulur.

Eğitim bittikten sonra gerçek metrikleri üretmek için:

```bash
python src/models/bert/bert_evaluate.py
```

> `bert_evaluate.py` içindeki fallback zinciri, özel model yüklenemezse sessizce
> **eğitilmemiş** base BERT'e düşüp "✅ Base BERT model loaded (untrained)" yazar.
> Çıktıda bu satırı görürsen ürettiği metrikler anlamsızdır — checkpoint yolunu düzelt.

## 2. Ağırlıkları Hub'a yükle

```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli upload <kullanici>/duyguanalizi-berturk \
    artifacts/bert_mlp_ckpt/best_model .
```

## 3. Yerelde dene

```bash
export SENTIMENT_HF_REPO=<kullanici>/duyguanalizi-berturk
python app.py        # http://localhost:7860
```

Hub'a yüklemeden yerel checkpoint ile denemek için `SENTIMENT_HF_REPO` yerine
`SENTIMENT_CKPT=artifacts/bert_mlp_ckpt/best_model` kullan.

## 4. Space oluştur (PRO gerektirir)

> Bu adım yalnızca PRO aboneliğiyle çalışır — gerekçesi bir alttaki bölümde.
> `deploy/publish.sh` bunu otomatik dener, abonelik yoksa 402 ile düşer.

huggingface.co/new-space → SDK olarak **Gradio** seç. Sonra Space reposuna şunları koy:

```
app.py
requirements-serve.txt  →  requirements.txt adıyla
src/inference/predictor.py
src/models/bert/bert_mlp_classifier.py
src/configs/bert_hparams.yaml
```

`bert_hparams.yaml` şart: checkpoint bir `PreTrainedModel` değil, düz bir
`state_dict`. Mimari (MLP katmanları, pooling stratejisi, sınıf listesi) bu
dosyadan yeniden kurulur. Eksikse model yüklenemez.

Space'in `README.md` dosyasına şu frontmatter'ı ekle:

```yaml
---
title: Türkçe Yorum Duygu Analizi
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---
```

Space ayarlarından `SENTIMENT_HF_REPO` değişkenini model reposuna ayarla. Model
reposu private ise ayrıca `HF_TOKEN` secret'ı ekle.

## Arayüz barındırma: ücretsiz seçenek yok

**Gradio Space'leri artık ücretsiz katmanda çalışmıyor.** `create_repo` denemesi
`402 Payment Required` döndürüyor:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
> cpu-basic requires a PRO subscription.

Model reposunu yayınlamak ücretsiz; ücretli olan yalnızca çalışan arayüz. Dört seçenek
var, hepsi ölçülmüş rakamlarla:

| Seçenek | Maliyet | Gereken iş | Doğruluk |
| --- | --- | --- | --- |
| HF PRO + mevcut Gradio Space | $9/ay | yok, `publish.sh` yeterli | %82.2 |
| Static Space + tarayıcıda ONNX | ücretsiz | arayüzü JS'te yeniden yaz | %82.0 |
| Render/Railway + Docker | ~$7/ay | Dockerfile + CI | %82.2 |
| Yerel `python app.py` | ücretsiz | yok | %82.2 |

Tarayıcı seçeneği gerçekten uygulanabilir, denendi: model ONNX'e sorunsuz çıkıyor
(442 MB) ve int8 nicemlemeyle **111 MB**'a iniyor. 300 örnekte ölçüldüğünde fp32 ONNX
PyTorch ile 300/300 aynı tahmini veriyor, int8 297/300 — yani nicemlemenin bedeli
yaklaşık 1 puan. Karşılığında sonsuza dek ücretsiz barındırma, ama ziyaretçi ilk
açılışta 111 MB indiriyor ve tokenizer dahil tüm arayüzün JavaScript'te yazılması
gerekiyor.

`config`'deki `export.onnx_optimization` ve `target_size_mb: 140` ayarları bu yolu
zaten öngörmüş; ölçülen 111 MB o hedefin altında.

## Model yayınlamak (ücretsiz)

```bash
./venv/bin/huggingface-cli login     # write yetkili token
bash deploy/publish.sh <hf-kullanici-adi>
```

Script model reposunu oluşturup ağırlıkları yükler, sonra Space'i dener. Space adımı
PRO yoksa 402 ile düşer — model yüklemesi o noktada çoktan tamamlanmıştır.

Yayındaki model: <https://huggingface.co/bariskirat/duyguanalizi-berturk>

## Deploy öncesi kalan temizlik

- `data/processed/*.arrow` (66 MB) git geçmişinde duruyor; `.gitignore` artık kapsıyor
  ama geçmişten silmek `git filter-repo` ister. Space'e bu dosyalar gitmediği için
  yayını engellemez, sadece klonlamayı yavaşlatır.
- `src/analysis/encoder.py` ve `shap_error.py` checkpoint'i
  `AutoModelForSequenceClassification.from_pretrained` ile yüklemeye çalışıyor; bu
  checkpoint'te çalışmaz (`config.json` yok). `encoder.py` sessizce eğitilmemiş base
  modele düşer, `shap_error.py` hata verir. İkisi de deploy yolunda değil ama
  interpretability katmanını kullanacaksan `predictor.py`'deki yükleme biçimine
  geçirilmeleri gerekir.
- `src/saves/trained_model.pkl` (TF-IDF baseline) hangi kodla üretildiği belirsiz —
  `train_baseline.py` onu kaydetmiyor. Deploy edilecek model bu değil.
