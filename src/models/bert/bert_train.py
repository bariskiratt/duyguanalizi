import yaml
import torch
import time
import os
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datasets import load_from_disk, Value
from tqdm.auto import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Import your custom model
from bert_mlp_classifier import BertMLPClassifier, BertMLPConfig, BertMLPWithCustomPooling

# Initialize rich console for beautiful output
console = Console()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall metrics
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2])
    
    # Class distribution in predictions
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    pred_distribution = {f"pred_class_{i}": 0 for i in [0, 1, 2]}
    for cls, count in zip(unique_preds, pred_counts):
        pred_distribution[f"pred_class_{cls}"] = count
    
    # Check if all classes are being predicted
    missing_classes = [i for i in [0, 1, 2] if i not in unique_preds]
    
    metrics = {
        "f1": f1_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": accuracy,
        "f1_class_0": f1_per_class[0],
        "f1_class_1": f1_per_class[1], 
        "f1_class_2": f1_per_class[2],
        "missing_classes_count": len(missing_classes),
        **pred_distribution
    }
    
    # Print detailed class analysis every evaluation
    print(f"\n📊 Evaluation Metrics:")
    print(f"   F1 Macro: {f1_macro:.4f}")
    print(f"   F1 Per Class: [0: {f1_per_class[0]:.4f}, 1: {f1_per_class[1]:.4f}, 2: {f1_per_class[2]:.4f}]")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Predictions distribution: {pred_distribution}")
    if missing_classes:
        print(f"   ⚠️  Missing classes in predictions: {missing_classes}")
    else:
        print(f"   ✅ All classes being predicted!")
    
    return metrics


class WeightedLossTrainer(Trainer):
    """Custom trainer with weighted loss and enhanced progress tracking"""
    def __init__(self, class_weights=None, progress_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.progress_callback = progress_callback
        self.current_epoch = 0
        self.total_epochs = kwargs.get('args').num_train_epochs if 'args' in kwargs else 1
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))], 
                device=logits.device, dtype=logits.dtype
            )
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.current_epoch = state.epoch
        if self.progress_callback:
            self.progress_callback('epoch_start', self.current_epoch, self.total_epochs)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging"""
        if self.progress_callback and logs:
            self.progress_callback('log', logs)
            
        # Enhanced logging for class monitoring
        if logs:
            log_str = f"📉 Step {state.global_step}: Loss: {logs.get('loss', 'N/A'):.4f}"
            if 'eval_f1' in logs:
                log_str += f" | 🎯 F1: {logs['eval_f1']:.4f}"
            if 'eval_accuracy' in logs:
                log_str += f" | ✅ Acc: {logs['eval_accuracy']:.4f}"
            if 'eval_missing_classes_count' in logs:
                missing_count = logs['eval_missing_classes_count']
                if missing_count > 0:
                    log_str += f" | ⚠️  Missing {missing_count} classes"
                else:
                    log_str += f" | ✅ All classes predicted"
            console.print(log_str)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called during evaluation"""
        if self.progress_callback:
            self.progress_callback('eval_start', None)


def create_model(config):
    """Create model based on configuration with progress tracking"""
    
    with console.status("[bold green]Loading model architecture...", spinner="dots") as status:
        time.sleep(0.5)  # Brief pause for visual effect
        
        if config['mlp']['use_mlp']:
            console.print("🧠 [bold cyan]Using BERT + MLP architecture[/bold cyan]")
            
            # Create MLP config
            status.update("[bold green]Creating MLP configuration...")
            mlp_config = BertMLPConfig(
                hidden_sizes=config['mlp']['hidden_sizes'],
                dropout_rate=config['mlp']['dropout_rate'],
                activation=config['mlp']['activation'],
                use_batch_norm=config['mlp']['use_batch_norm']
            )
            
            # Choose pooling strategy
            pooling = config['mlp'].get('pooling_strategy', 'cls')
            status.update(f"[bold green]Loading BERT + MLP with {pooling} pooling...")
            
            if pooling == 'cls':
                model = BertMLPClassifier(
                    model_name=config['model']['name'],
                    num_classes=config['model']['num_classes'],
                    mlp_config=mlp_config
                )
            else:
                model = BertMLPWithCustomPooling(
                    model_name=config['model']['name'],
                    num_classes=config['model']['num_classes'],
                    mlp_config=mlp_config,
                    pooling=pooling
                )
            
            # Create configuration table
            config_table = Table(title="🧠 MLP Configuration", show_header=True, header_style="bold magenta")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Hidden layers", str(config['mlp']['hidden_sizes']))
            config_table.add_row("Dropout rate", str(config['mlp']['dropout_rate']))
            config_table.add_row("Activation", config['mlp']['activation'])
            config_table.add_row("Batch norm", str(config['mlp']['use_batch_norm']))
            config_table.add_row("Pooling", pooling)
            
            console.print(config_table)
            
        else:
            console.print("🔗 [bold yellow]Using simple BERT + Linear architecture[/bold yellow]")
            status.update("[bold green]Loading BERT with linear head...")
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                config['model']['name'], 
                num_labels=config['model']['num_classes']
            )
    
    return model


def create_progress_callback():
    """Create a progress callback for training updates"""
    training_progress = None
    
    def progress_callback(event_type, data, total=None):
        nonlocal training_progress
        
        if event_type == 'epoch_start':
            console.print(f"\n🚀 [bold green]Starting Epoch {int(data) + 1}/{total}[/bold green]")
        elif event_type == 'log' and data:
            if 'loss' in data:
                console.print(f"📉 Loss: {data['loss']:.4f}", end="")
            if 'eval_f1' in data:
                console.print(f" | 🎯 F1: {data['eval_f1']:.4f}", end="")
            if 'eval_accuracy' in data:
                console.print(f" | ✅ Acc: {data['eval_accuracy']:.4f}")
            else:
                console.print()
        elif event_type == 'eval_start':
            console.print("🔍 [yellow]Running evaluation...[/yellow]")
    
    return progress_callback


def check_gpu_availability():
    """Comprehensive GPU availability check"""
    
    console.print(Panel.fit(
        "[bold yellow]🔍 GPU Availability Check[/bold yellow]\n"
        "[dim]Checking CUDA support and GPU configuration[/dim]",
        border_style="yellow"
    ))
    
    gpu_info = {}
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    gpu_info['cuda_available'] = cuda_available
    
    if cuda_available:
        # Get GPU details
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        gpu_info.update({
            'gpu_count': gpu_count,
            'current_device': current_device,
            'gpu_name': gpu_name,
            'gpu_memory_gb': gpu_memory,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cudnn_enabled': torch.backends.cudnn.enabled
        })
        
        # Test GPU operation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            test_result = torch.matmul(test_tensor, test_tensor)
            gpu_test_passed = True
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            gpu_test_passed = False
            gpu_info['gpu_error'] = str(e)
        
        gpu_info['gpu_test_passed'] = gpu_test_passed
    
    # Create GPU status table
    gpu_table = Table(title="🔍 GPU Status Check", show_header=True, header_style="bold yellow")
    gpu_table.add_column("Component", style="cyan")
    gpu_table.add_column("Status", style="white")
    gpu_table.add_column("Details", style="dim")
    
    # CUDA availability
    cuda_status = "✅ Available" if cuda_available else "❌ Not Available"
    gpu_table.add_row("CUDA Support", cuda_status, f"PyTorch version: {torch.__version__}")
    
    if cuda_available:
        # GPU details
        gpu_table.add_row("GPU Count", str(gpu_info['gpu_count']), f"Current device: {gpu_info['current_device']}")
        gpu_table.add_row("GPU Name", gpu_info['gpu_name'], f"Memory: {gpu_info['gpu_memory_gb']:.1f} GB")
        gpu_table.add_row("CUDA Version", str(gpu_info['cuda_version']), f"cuDNN: {gpu_info['cudnn_version']}")
        
        # GPU test
        test_status = "✅ Passed" if gpu_info['gpu_test_passed'] else "❌ Failed"
        test_details = "Matrix operations work" if gpu_info['gpu_test_passed'] else gpu_info.get('gpu_error', 'Unknown error')
        gpu_table.add_row("GPU Test", test_status, test_details)
        
        # Memory status
        if gpu_info['gpu_test_passed']:
            memory_allocated = torch.cuda.memory_allocated() / 1e6
            memory_cached = torch.cuda.memory_reserved() / 1e6
            gpu_table.add_row("Memory Status", f"{memory_allocated:.1f} MB allocated", f"{memory_cached:.1f} MB cached")
    else:
        gpu_table.add_row("Recommendation", "Install CUDA PyTorch", "pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    console.print(gpu_table)
    
    # Return device recommendation
    if cuda_available and gpu_info.get('gpu_test_passed', False):
        console.print("✅ [bold green]GPU is ready for training![/bold green]")
        return torch.device("cuda")
    else:
        console.print("⚠️ [bold yellow]GPU not available, falling back to CPU[/bold yellow]")
        return torch.device("cpu")


def _coerce_labels_to_int64(ds):
    """Ensure dataset has an integer 'labels' column suitable for Trainer."""
    label_map = {'negatif': 0, 'pozitif': 1, 'notr': 2}
    cols = ds.column_names
    # Rename 'label' -> 'labels' if needed
    if 'labels' not in cols and 'label' in cols:
        ds = ds.rename_column('label', 'labels')
    # If labels are strings, map them
    dtype_str = str(ds.features['labels'].dtype) if 'labels' in ds.features else ''
    if 'string' in dtype_str:
        ds = ds.map(lambda ex: {'labels': label_map.get(ex['labels'], 2)}, desc='Map string labels to ids')
    # If labels are lists, squash to scalar
    def _squash(ex):
        v = ex['labels']
        if isinstance(v, list):
            return {'labels': v[0] if len(v) > 0 else 2}
        return {'labels': v}
    ds = ds.map(_squash, desc='Squash nested labels')
    # Cast to int64
    ds = ds.cast_column('labels', Value('int64'))
    return ds


def main():
    # Print startup banner
    console.print(Panel.fit(
        "[bold cyan]🤖 BERT + MLP Training Pipeline[/bold cyan]\n"
        "[dim]Enhanced with progress tracking and beautiful visualizations[/dim]",
        border_style="cyan"
    ))
    
    # Load configuration with progress
    with console.status("[bold green]Loading configuration...", spinner="dots"):
        with open("src/configs/bert_hparams.yaml", 'r') as f:
            config = yaml.safe_load(f)
        time.sleep(0.5)
    
    # Comprehensive GPU check
    device = check_gpu_availability()
    gpu_info = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    
    # Create device info table
    device_table = Table(title="🖥️ Hardware Configuration", show_header=True, header_style="bold green")
    device_table.add_column("Component", style="cyan")
    device_table.add_column("Details", style="white")
    device_table.add_row("Device", str(device).upper())
    device_table.add_row("GPU/CPU", gpu_info)
    device_table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        device_table.add_row("CUDA Version", torch.version.cuda)
        device_table.add_row("Memory Available", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    console.print(device_table)
    
    # Load tokenizer with progress
    with console.status("[bold green]Loading tokenizer...", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        time.sleep(0.5)
    
    console.print("✅ [green]Tokenizer loaded successfully[/green]")
    
    # Create and move model to device
    model = create_model(config)
    
    with console.status("[bold green]Moving model to device...", spinner="dots"):
        model = model.to(device)
        time.sleep(0.5)
    
    # Model parameters analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create model info table
    model_table = Table(title="📊 Model Statistics", show_header=True, header_style="bold blue")
    model_table.add_column("Metric", style="cyan")
    model_table.add_column("Value", style="white")
    model_table.add_row("Total Parameters", f"{total_params:,}")
    model_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    model_table.add_row("Model Size (approx)", f"{total_params * 4 / 1e6:.1f} MB")
    model_table.add_row("Architecture", "BERT + MLP" if config.get('mlp', {}).get('use_mlp', False) else "BERT + Linear")
    
    console.print(model_table)
    
    # Load datasets with progress
    console.print("\n📁 [bold cyan]Loading datasets...[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        
        # Load training dataset
        train_task = progress.add_task("[green]Loading training dataset...", total=100)
        train_dataset = load_from_disk("data/processed/bert_train")
        progress.update(train_task, completed=100)
        
        # Load validation dataset  
        val_task = progress.add_task("[yellow]Loading validation dataset...", total=100)
        val_dataset = load_from_disk("data/processed/bert_val")
        progress.update(val_task, completed=100)
    
    # Coerce labels to proper dtype/shape
    train_dataset = _coerce_labels_to_int64(train_dataset)
    val_dataset = _coerce_labels_to_int64(val_dataset)
    
    # Create dataset info table
    dataset_table = Table(title="📋 Dataset Information", show_header=True, header_style="bold magenta")
    dataset_table.add_column("Split", style="cyan")
    dataset_table.add_column("Size", style="white")
    dataset_table.add_column("Features", style="white")
    
    dataset_table.add_row("Training", f"{len(train_dataset):,}", str(list(train_dataset.features.keys())))
    dataset_table.add_row("Validation", f"{len(val_dataset):,}", str(list(val_dataset.features.keys())))
    
    console.print(dataset_table)
    
    # Training configuration
    with console.status("[bold green]Setting up training configuration...", spinner="dots"):
        training_args = TrainingArguments(
            output_dir="./artifacts/bert_mlp_ckpt",
            num_train_epochs=config['training']['num_epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['eval_batch_size'],
            learning_rate=float(config['training']['learning_rate']),
            warmup_steps=config['training']['warmup_steps'],
            weight_decay=config['training']['weight_decay'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            fp16=config['training']['fp16'],
            eval_strategy="steps",
            eval_steps=config['evaluation']['eval_steps'],
            logging_steps=config['evaluation']['logging_steps'],
            save_steps=config['evaluation']['save_steps'],
            load_best_model_at_end=config['evaluation']['load_best_model_at_end'],
            metric_for_best_model=config['early_stopping']['metric_for_best_model'],
            greater_is_better=config['early_stopping']['greater_is_better'],
            report_to=None,  # Disable wandb/tensorboard for cleaner output
            save_safetensors=True,  # Use safetensors for better compression
            dataloader_pin_memory=False,  # Reduce memory usage
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        time.sleep(0.5)
    
    # Training parameters table
    train_table = Table(title="🎯 Training Configuration", show_header=True, header_style="bold yellow")
    train_table.add_column("Parameter", style="cyan")
    train_table.add_column("Value", style="white")
    
    train_table.add_row("Epochs", str(config['training']['num_epochs']))
    train_table.add_row("Batch Size", str(config['training']['batch_size']))
    train_table.add_row("Learning Rate", str(config['training']['learning_rate']))
    train_table.add_row("Weight Decay", str(config['training']['weight_decay']))
    train_table.add_row("Warmup Steps", str(config['training']['warmup_steps']))
    train_table.add_row("FP16", str(config['training']['fp16']))
    train_table.add_row("Early Stopping", f"Patience: {config['early_stopping']['patience']}")
    
    console.print(train_table)
    
    # Get class weights
    class_weights = config.get('training', {}).get('class_weights', None)
    if class_weights:
        console.print(f"⚖️ [yellow]Using class weights: {class_weights}[/yellow]")
    
    # Create progress callback
    progress_callback = create_progress_callback()
    
    # Setup separate learning rates for BERT and MLP head if using MLP
    if config['mlp']['use_mlp']:
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Calculate training steps (accounting for gradient accumulation)
        effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        num_training_steps = (len(train_dataset) // effective_batch_size) * training_args.num_train_epochs
        num_warmup_steps = training_args.warmup_steps
        
        # Create optimizer with separate learning rates
        bert_lr = float(config['training']['learning_rate'])  # Use config learning rate
        mlp_lr = bert_lr * 2.5  # 2.5x higher for MLP head (more conservative)
        optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": bert_lr, "weight_decay": 0.01},
                {"params": model.mlp.parameters(), "lr": mlp_lr, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        
        # Create learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps
        )
        
        console.print(f"⚡ [yellow]Using separate learning rates: BERT={bert_lr}, MLP Head={mlp_lr}[/yellow]")
        optimizers = (optimizer, lr_scheduler)
    else:
        optimizers = None
    
    # Create trainer with progress tracking
    with console.status("[bold green]Initializing trainer...", spinner="dots"):
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            class_weights=class_weights,
            progress_callback=progress_callback,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['early_stopping']['patience'],
                early_stopping_threshold=config['early_stopping']['threshold']
            )],
            optimizers=optimizers  # Pass custom optimizer if using MLP
        )
        time.sleep(0.5)
    
    console.print("✅ [green]Trainer initialized successfully[/green]")
    
    # Start training
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]🚀 Starting Training Process[/bold green]\n"
        f"[dim]Training for {config['training']['num_epochs']} epochs with {len(train_dataset):,} samples[/dim]",
        border_style="green"
    ))
    
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        console.print(f"\n✅ [bold green]Training completed in {training_time/60:.1f} minutes![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Training interrupted by user[/yellow]")
        return
    except Exception as e:
        console.print(f"\n❌ [red]Training failed: {str(e)}[/red]")
        raise
    
    # Save model with progress
    output_dir = "./artifacts/bert_mlp_ckpt/best_model"
    os.makedirs(output_dir, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        
        # Save model
        save_task = progress.add_task("[green]Saving model...", total=100)
        trainer.save_model(output_dir)
        progress.update(save_task, completed=70)
        
        # Save tokenizer
        progress.update(save_task, description="[green]Saving tokenizer...")
        tokenizer.save_pretrained(output_dir)
        progress.update(save_task, completed=100)
    
    # Final success message
    console.print(Panel.fit(
        f"[bold green]🎉 Training completed successfully![/bold green]\n"
        f"[dim]Model saved to: {output_dir}[/dim]\n"
        f"[dim]Total training time: {training_time/60:.1f} minutes[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
