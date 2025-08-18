import os, yaml, math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch, numpy as np

# Config ve checkpoint ayarları
with open("src/configs/bert_hparams.yaml", "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

MODEL_NAME = _CFG["model"]["name"]
MAX_LEN_DEFAULT = _CFG["model"].get("max_length", 256)
CKPT_DIR = "artifacts/bert_ckpt/best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_text_encoder():
    """Tokenizer + sadece text encoder (sınıflandırıcı kafasız) yükle.
    - Eğer fine-tuned checkpoint varsa onu kullanır (base encoder alınır)
    - Yoksa pretrained base modeli yükler
    """
    if os.path.isdir(CKPT_DIR):
        cls_model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
        # HF modellerinde base encoder ortak isimde tutulur (ör. .base_model)
        base_encoder = cls_model.base_model
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    else:
        base_encoder = AutoModel.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer, base_encoder.to(DEVICE).eval()

# Tokenizer ve modeli bir kere yükleyip tekrar tekrar kullanacağız.
_tok, _mdl = _load_text_encoder()

def _mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    # Pad tokenlarını maskeleyerek zamansal ortalama alır.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)          # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                # [B, H]
    return summed / counts                                  # [B, H]

@torch.no_grad()
def encode(texts, max_length=MAX_LEN_DEFAULT, batch_size=64, l2_normalize=True, show_progress=True, desc="Encoding"):
    """Metin listesini dbmdz ile vektörle.
    - max_length: uzun metinlerde 256 deneyebilirsin (yavaşlar)
    - l2_normalize: cosine benzerliği için faydalıdır
    """
    out = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        total_steps = math.ceil(len(texts) / batch_size)
        iterator = tqdm(iterator, total=total_steps, desc=desc)
    for i in iterator:
        batch = texts[i:i+batch_size]
        enc = _tok(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(DEVICE)
        last = _mdl(**enc).last_hidden_state                 # [B, T, H]
        pooled = _mean_pool(last, enc["attention_mask"])     # [B, H]
        if l2_normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)                                    # [N, H]


def encode_and_save(texts, out_path, **kwargs):
    """Encode et ve .npy'ye kaydet."""
    arr = encode(texts, **kwargs)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, arr)
    return arr
    