"""Tek bir yorumdan duygu tahmini - servis edilebilir inference katmani.

Egitim ciktisi bir PreTrainedModel degil, duz nn.Module state_dict'idir; bu
yuzden `AutoModelForSequenceClassification.from_pretrained` bu checkpoint'te
calismaz. Dogru yol mimariyi config'ten yeniden kurup agirliklari yuklemektir.

Kullanim:
    from src.inference.predictor import SentimentPredictor
    p = SentimentPredictor()
    p.predict("kargo hizliydi ama urun beklendigi gibi degil")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "models" / "bert"))

from bert_mlp_classifier import (  # noqa: E402
    BertMLPClassifier,
    BertMLPConfig,
    BertMLPWithCustomPooling,
)

DEFAULT_CONFIG = _REPO_ROOT / "src" / "configs" / "bert_hparams.yaml"
DEFAULT_CKPT = _REPO_ROOT / "artifacts" / "bert_mlp_ckpt" / "best_model"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state_dict(ckpt_dir: Path) -> dict:
    safetensors = ckpt_dir / "model.safetensors"
    pytorch_bin = ckpt_dir / "pytorch_model.bin"
    if safetensors.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors))
    if pytorch_bin.exists():
        return torch.load(str(pytorch_bin), map_location="cpu")
    raise FileNotFoundError(
        f"{ckpt_dir} icinde model.safetensors veya pytorch_model.bin yok. "
        "Once bert_train.py ile egitip checkpoint uretmelisin."
    )


class SentimentPredictor:
    """Egitilmis BERTurk+MLP modelini yukleyip tek tek metin siniflandirir."""

    def __init__(
        self,
        ckpt_dir: str | os.PathLike | None = None,
        config_path: str | os.PathLike | None = None,
        device: str | torch.device | None = None,
    ):
        self.config_path = Path(config_path or os.environ.get("SENTIMENT_CONFIG", DEFAULT_CONFIG))
        self.ckpt_dir = Path(ckpt_dir or os.environ.get("SENTIMENT_CKPT", DEFAULT_CKPT))
        self.device = torch.device(device) if device else _pick_device()

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.labels = self.config["labels"]
        self.max_length = self.config["model"].get("max_length", 256)
        num_classes = self.config["model"]["num_classes"]
        if len(self.labels) != num_classes:
            raise ValueError(
                f"config tutarsiz: labels {len(self.labels)} tane ama num_classes={num_classes}"
            )

        if not self.ckpt_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint klasoru yok: {self.ckpt_dir}\n"
                "Model egitilmemis. bert_train.py calistir ya da SENTIMENT_CKPT ile "
                "indirilmis bir checkpoint'i goster."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.ckpt_dir))
        self.model = self._build_model(num_classes)
        # strict=True bilerek: eksik/fazla agirlik sessizce gecmemeli.
        self.model.load_state_dict(_load_state_dict(self.ckpt_dir))
        self.model.to(self.device).eval()

    def _build_model(self, num_classes: int):
        mlp_cfg = BertMLPConfig(
            hidden_sizes=self.config["mlp"]["hidden_sizes"],
            dropout_rate=self.config["mlp"]["dropout_rate"],
            activation=self.config["mlp"]["activation"],
            use_batch_norm=self.config["mlp"]["use_batch_norm"],
        )
        # Pooling egitimdekiyle birebir ayni olmali; havuzlamanin parametresi
        # olmadigi icin yanlis secim state_dict yuklemesinde hata vermez, sadece
        # tahminleri bozar.
        pooling = self.config["mlp"].get("pooling_strategy", "cls")
        name = self.config["model"]["name"]
        if pooling == "cls":
            return BertMLPClassifier(model_name=name, num_classes=num_classes, mlp_config=mlp_cfg)
        return BertMLPWithCustomPooling(
            model_name=name, num_classes=num_classes, mlp_config=mlp_cfg, pooling=pooling
        )

    @torch.no_grad()
    def predict_batch(self, texts: Iterable[str]) -> list[dict]:
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        probs = F.softmax(logits.float(), dim=-1).cpu()

        out = []
        for row in probs:
            scores = {label: float(row[i]) for i, label in enumerate(self.labels)}
            top = max(scores, key=scores.get)
            out.append({"label": top, "confidence": scores[top], "scores": scores})
        return out

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]


if __name__ == "__main__":
    predictor = SentimentPredictor()
    ornekler = [
        "urun harika, tam beklediğim gibi cok memnun kaldim",
        "kargo gec geldi ve paket ezilmisti, berbat",
        "kargo hizliydi ama urun bekledigim gibi degil",
        "fiyatina gore idare eder",
    ]
    for r, t in zip(predictor.predict_batch(ornekler), ornekler):
        dagilim = "  ".join(f"{k}={v:.3f}" for k, v in r["scores"].items())
        print(f"{r['label']:<8} ({r['confidence']:.3f})  {t}\n         {dagilim}")
