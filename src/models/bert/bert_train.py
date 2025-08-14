import yaml
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
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

class WeightedLossTrainer(Trainer):
    """Custom trainer with weighted loss"""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Convert class weights to tensor
            weight_tensor = torch.tensor([self.class_weights[i] for i in range(len(self.class_weights))], 
                                       device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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
    # create the collator (pads to batch max at runtime; pad to multiple of 8 helps fp16 Tensor Cores)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )
    # Get class weights from config
    class_weights = config.get('training', {}).get('class_weights', None)
    if class_weights:
        print(f"🎯 Using class weights: {class_weights}")
    else:
        print("⚠️  No class weights specified, using standard loss")
    
    # Trainer oluştur
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
        class_weights=class_weights
    )
    
    # Eğit
    trainer.train()
    
    # En iyi modeli kaydet
    trainer.save_model("./artifacts/bert_ckpt/best_model")
    tokenizer.save_pretrained("./artifacts/bert_ckpt/best_model")

if __name__ == "__main__":
    main()
        
