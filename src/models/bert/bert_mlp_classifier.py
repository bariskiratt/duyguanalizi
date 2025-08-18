import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm.auto import tqdm
import time

class BertMLPConfig:
    """Configuration for MLP layers"""
    def __init__(self, 
                 hidden_sizes=[512, 256], 
                 dropout_rate=0.3,
                 activation='relu',
                 use_batch_norm=True):
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm

class BertMLPClassifier(nn.Module):
    """BERT with Multi-Layer Perceptron head for classification"""
    
    def __init__(self, model_name, num_classes=3, mlp_config=None):
        super().__init__()
        
        # Load BERT backbone
        print("🔄 Loading BERT backbone...")
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_labels = num_classes
        
        # Get BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size  # Usually 768
        
        # MLP Configuration
        if mlp_config is None:
            mlp_config = BertMLPConfig()
        
        # Build MLP layers with progress
        print("🧠 Building MLP layers...")
        self.mlp = self._build_mlp(bert_hidden_size, num_classes, mlp_config)
        
        # Initialize weights with progress
        print("⚡ Initializing MLP weights...")
        self._init_weights()
        
        print(f"✅ BERT+MLP model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _build_mlp(self, input_size, num_classes, config):
        """Build the MLP head with progress tracking"""
        layers = []
        current_size = input_size
        
        # Progress bar for building layers
        total_layers = len(config.hidden_sizes) + 1  # +1 for final layer
        
        with tqdm(total=total_layers, desc="Building MLP layers", unit="layer") as pbar:
            # Hidden layers
            for i, hidden_size in enumerate(config.hidden_sizes):
                layer_desc = f"Hidden layer {i+1} ({current_size}→{hidden_size})"
                pbar.set_description(layer_desc)
                
                # Linear layer
                layers.append(nn.Linear(current_size, hidden_size))
                
                # Activation function
                if config.activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif config.activation.lower() == 'gelu':
                    layers.append(nn.GELU())
                elif config.activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                
                # Dropout
                layers.append(nn.Dropout(config.dropout_rate))
                
                # Batch normalization (optional)
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
                current_size = hidden_size
                pbar.update(1)
                time.sleep(0.1)  # Small delay for visual effect
            
            # Final classification layer
            pbar.set_description(f"Output layer ({current_size}→{num_classes})")
            layers.append(nn.Linear(current_size, num_classes))
            pbar.update(1)
            time.sleep(0.1)
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize MLP weights with progress tracking"""
        linear_modules = [module for module in self.mlp.modules() if isinstance(module, nn.Linear)]
        
        with tqdm(linear_modules, desc="Initializing weights", unit="layer") as pbar:
            for module in pbar:
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                
                # Update progress description with layer info
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                pbar.set_description(f"Initializing {in_features}→{out_features}")
                time.sleep(0.05)  # Small delay for visual effect
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass"""
        
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation (pooled output)
        pooled_output = bert_outputs.pooler_output  # Shape: (batch_size, 768)
        
        # Pass through MLP
        logits = self.mlp(pooled_output)  # Shape: (batch_size, num_classes)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return in HuggingFace format for compatibility
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )

# Alternative pooling strategies (advanced)
class BertMLPWithCustomPooling(BertMLPClassifier):
    """BERT+MLP with different pooling strategies"""
    
    def __init__(self, model_name, num_classes=3, mlp_config=None, pooling='cls'):
        self.pooling_strategy = pooling
        print(f"🔄 Initializing BERT+MLP with {pooling} pooling...")
        super().__init__(model_name, num_classes, mlp_config)
        print(f"✅ Custom pooling ({pooling}) configured successfully")
    
    def _custom_pooling(self, sequence_output, attention_mask):
        """Apply different pooling strategies"""
        
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (default)
            return sequence_output[:, 0]  # First token
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling over all tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            return sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        elif self.pooling_strategy == 'max':
            # Max pooling over all tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sequence_output[input_mask_expanded == 0] = -1e9  # Set padding tokens to very low value
            return torch.max(sequence_output, 1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with custom pooling"""
        
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply custom pooling
        pooled_output = self._custom_pooling(bert_outputs.last_hidden_state, attention_mask)
        
        # Pass through MLP
        logits = self.mlp(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )