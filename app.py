"""Turkce e-ticaret yorum duygu analizi - Gradio arayuzu.

Hugging Face Spaces bu dosyayi giris noktasi olarak calistirir.

Agirliklarin nereden geldigi iki ortam degiskeniyle belirlenir:
  SENTIMENT_HF_REPO  Hub'daki model reposu (ornek: "kullanici/duyguanalizi-berturk")
                     Verilirse checkpoint indirilir. Spaces'te bunu kullan -
                     443 MB'lik agirligi git'e koymak icin sebep yok.
  SENTIMENT_CKPT     Yerel checkpoint klasoru. Verilmezse
                     artifacts/bert_mlp_ckpt/best_model kullanilir.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src.inference.predictor import SentimentPredictor

ETIKET_GORUNUMU = {
    "pozitif": "Pozitif 🙂",
    "negatif": "Negatif 🙁",
    "notr": "Nötr 😐",
}

ORNEKLER = [
    "Ürün harika, tam beklediğim gibi. Çok memnun kaldım.",
    "Kargo geç geldi ve paket ezilmişti, hiç memnun kalmadım.",
    "Kargo hızlıydı ama ürün beklediğim gibi değil.",
    "Fiyatına göre idare eder, çok da bir beklentim yoktu.",
    "Bir haftadır kullanıyorum, şimdilik bir sorun yok.",
]


def _checkpoint_hazirla() -> str | None:
    """Hub reposu verilmisse indir, yoksa yerel yolu kullan."""
    repo = os.environ.get("SENTIMENT_HF_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN"))
    return os.environ.get("SENTIMENT_CKPT")


try:
    _predictor = SentimentPredictor(ckpt_dir=_checkpoint_hazirla())
    _yukleme_hatasi = None
except Exception as exc:  # noqa: BLE001 - hatayi arayuzde gostermek istiyoruz
    _predictor = None
    _yukleme_hatasi = str(exc)


def tahmin_et(metin: str):
    if _predictor is None:
        raise gr.Error(
            "Model yuklenemedi. Once bert_train.py ile egitip checkpoint uret, "
            f"ya da SENTIMENT_HF_REPO ayarla.\n\nDetay: {_yukleme_hatasi}"
        )
    metin = (metin or "").strip()
    if not metin:
        return {}
    sonuc = _predictor.predict(metin)
    return {ETIKET_GORUNUMU.get(k, k): v for k, v in sonuc["scores"].items()}


with gr.Blocks(title="Türkçe Yorum Duygu Analizi") as demo:
    gr.Markdown(
        "# Türkçe E-Ticaret Yorum Duygu Analizi\n"
        "BERTurk (`dbmdz/bert-base-turkish-cased`) üzerine eğitilmiş MLP başlıklı "
        "üç sınıflı duygu sınıflandırıcı. Bir ürün yorumu yazın, modelin sınıf "
        "olasılıklarını görün."
    )

    if _predictor is None:
        gr.Markdown(
            f"> ⚠️ **Model yüklenemedi.** Arayüz açık ama tahmin çalışmaz.\n>\n> `{_yukleme_hatasi}`"
        )

    with gr.Row():
        with gr.Column():
            girdi = gr.Textbox(
                label="Yorum",
                placeholder="Ürün yorumunu buraya yazın...",
                lines=4,
            )
            buton = gr.Button("Analiz et", variant="primary")
        with gr.Column():
            cikti = gr.Label(label="Tahmin", num_top_classes=3)

    gr.Examples(examples=ORNEKLER, inputs=girdi)

    gr.Markdown(
        "**Not:** `negatif` ve `notr` sınıfları modelin en çok karıştırdığı çifttir — "
        "şikâyet ve övgüyü aynı cümlede birleştiren yorumlar (\"kargo hızlıydı ama "
        "ürün beklediğim gibi değil\") en zor vakalar."
    )

    buton.click(fn=tahmin_et, inputs=girdi, outputs=cikti)
    girdi.submit(fn=tahmin_et, inputs=girdi, outputs=cikti)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
