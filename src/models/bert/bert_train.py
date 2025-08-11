import yaml
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score,f1_score,classification_report
import numpy as np
from datasets import load_from_disk

def compute_metrics(eval_pred):
    predictions,labels = eval_pred
    predictions = np.argmax(predictions,axis=1)

    f1 = f1_score(labels,predictions,average="macro")
    accuracy = accuracy_score(labels,predictions)

    return{"f1":f1,"accuracy":accuracy}


def get_training_args():
    """Training parametrelerini yükle"""
    with open("src/configs/bert_hparams.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    training_args = TrainingArguments(
        output_dir="./artifacts/bert_ckpt",
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        fp16=config['training']['fp16'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        no_cuda=False,
        eval_strategy="steps",
        eval_steps=config['evaluation']['eval_steps'],
        logging_steps=config['evaluation']['logging_steps'],
        save_steps=config['evaluation']['save_steps'],
        load_best_model_at_end=config['evaluation']['load_best_model_at_end'],
        metric_for_best_model=config['early_stopping']['metric_for_best_model'],
        greater_is_better=config['early_stopping']['greater_is_better']
    )
    
    return training_args


def main():
    with open("src/configs/bert_hparams.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Cihaz ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} - {(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')}")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # Ampere (RTX 3060) hızlandırma
    except Exception:
        pass

    # Tokenizer ve model yükle
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'], 
        num_labels=config['model']['num_classes']
    ).to(device)
    
    # Dataset'leri yükle
    train_dataset = load_from_disk("data/processed/bert_train")
    val_dataset = load_from_disk("data/processed/bert_val")
    
    # Training arguments
    training_args = get_training_args()
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Eğit
    trainer.train()
    
    # En iyi modeli kaydet
    trainer.save_model("./artifacts/bert_ckpt/best_model")
    tokenizer.save_pretrained("./artifacts/bert_ckpt/best_model")

if __name__ == "__main__":
    main()
        
