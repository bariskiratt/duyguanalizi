#!/usr/bin/env bash
# Egitilmis modeli Hugging Face Hub'a yukler ve Space icin dosyalari toplar.
#
# Once giris yap:  ./venv/bin/huggingface-cli login    (write yetkili token)
# Sonra:           bash deploy/publish.sh <kullanici-adi>
set -euo pipefail

KULLANICI="${1:-}"
if [ -z "$KULLANICI" ]; then
  echo "kullanim: bash deploy/publish.sh <hf-kullanici-adi>" >&2
  exit 1
fi

KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$KOK/venv/bin/python"
CKPT="$KOK/artifacts/bert_mlp_ckpt/best_model"
MODEL_REPO="$KULLANICI/duyguanalizi-berturk"
SPACE_REPO="$KULLANICI/duyguanalizi"
STAGE="$KOK/deploy/.model_stage"
SPACE="$KOK/deploy/.space_stage"

[ -d "$CKPT" ] || { echo "checkpoint yok: $CKPT — once egit" >&2; exit 1; }

echo "==> giris kontrolu"
"$PY" -c "from huggingface_hub import HfApi; print('  ->', HfApi().whoami()['name'])"

# ---------- model reposu ----------
echo "==> model dosyalari hazirlaniyor"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$CKPT/model.safetensors" "$CKPT/tokenizer.json" "$CKPT/tokenizer_config.json" "$STAGE/"
[ -f "$CKPT/special_tokens_map.json" ] && cp "$CKPT/special_tokens_map.json" "$STAGE/"
# Mimari config'ten yeniden kuruldugu icin bunlar agirliklarla birlikte gitmeli
cp "$KOK/src/configs/bert_hparams.yaml" "$STAGE/"
cp "$KOK/src/models/bert/bert_mlp_classifier.py" "$STAGE/"
cp "$KOK/deploy/model_card.md" "$STAGE/README.md"
du -sh "$STAGE" | sed 's/^/  /'

echo "==> $MODEL_REPO yukleniyor"
"$PY" - "$MODEL_REPO" "$STAGE" <<'PYEOF'
import sys
from huggingface_hub import create_repo, upload_folder
repo, folder = sys.argv[1], sys.argv[2]
create_repo(repo, repo_type="model", exist_ok=True)
upload_folder(repo_id=repo, folder_path=folder,
              commit_message="BERTurk + MLP Turkish sentiment, 0.8263 macro F1")
print(f"  -> https://huggingface.co/{repo}")
PYEOF

# ---------- space ----------
echo "==> Space dosyalari hazirlaniyor"
rm -rf "$SPACE"; mkdir -p "$SPACE/src/inference" "$SPACE/src/models/bert" "$SPACE/src/configs"
cp "$KOK/app.py"                                    "$SPACE/"
cp "$KOK/requirements-serve.txt"                    "$SPACE/requirements.txt"
cp "$KOK/src/inference/predictor.py"                "$SPACE/src/inference/"
cp "$KOK/src/models/bert/bert_mlp_classifier.py"    "$SPACE/src/models/bert/"
cp "$KOK/src/configs/bert_hparams.yaml"             "$SPACE/src/configs/"

cat > "$SPACE/README.md" <<EOF
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

BERTurk tabanli uc sinifli Turkce e-ticaret yorumu duygu analizi.
Model: [$MODEL_REPO](https://huggingface.co/$MODEL_REPO)
EOF

echo "==> $SPACE_REPO yukleniyor"
"$PY" - "$SPACE_REPO" "$SPACE" "$MODEL_REPO" <<'PYEOF'
import sys
from huggingface_hub import create_repo, upload_folder, add_space_variable
repo, folder, model_repo = sys.argv[1], sys.argv[2], sys.argv[3]
create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
upload_folder(repo_id=repo, repo_type="space", folder_path=folder,
              commit_message="Gradio arayuzu")
add_space_variable(repo, "SENTIMENT_HF_REPO", model_repo)
print(f"  -> https://huggingface.co/spaces/{repo}")
PYEOF

rm -rf "$STAGE" "$SPACE"
echo
echo "Bitti. Space birkac dakikada derlenip ayaga kalkar."
