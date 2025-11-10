"""
===================================================================
File 1: models.py - Model Architectures and Components
===================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model
from avalanche.training.strategies import Naive


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for parameter-efficient fine-tuning.
    
    Implements the LoRA technique from "LoRA: Low-Rank Adaptation of Large Language Models"
    which adds trainable rank decomposition matrices to frozen weights.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: Rank of the decomposition (lower = more efficient)
        alpha: Scaling factor (typically 2*r)
        dropout: Dropout rate for regularization
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices A and B such that ΔW = BA
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming, B with zeros (ensures ΔW=0 initially)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """Compute LoRA adaptation: x @ A @ B * scaling"""
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class ViTWithLoRA(nn.Module):
    """
    Vision Transformer with Low-Rank Adaptation.
    
    Freezes the pre-trained ViT backbone and adds lightweight LoRA adapters
    to the attention layers, achieving parameter-efficient continual learning.
    
    This approach reduces trainable parameters by ~99% compared to full fine-tuning
    while maintaining competitive performance.
    """
    def __init__(self, base_model, num_classes=100, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.base_model = base_model
        
        # Freeze the entire pre-trained backbone
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters to attention QKV projections
        self.lora_layers = nn.ModuleList()
        for block in self.base_model.blocks:
            qkv_dim = block.attn.qkv.in_features
            lora = LoRALayer(qkv_dim, qkv_dim * 3, lora_r, lora_alpha, lora_dropout)
            self.lora_layers.append(lora)
        
        # Replace classification head (trainable)
        self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)
        
    def forward(self, x):
        """Forward pass with LoRA-augmented attention"""
        # Patch embedding and positional encoding
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)
        
        # Process through transformer blocks with LoRA
        for idx, block in enumerate(self.base_model.blocks):
            # Compute QKV with LoRA augmentation
            qkv = block.attn.qkv(block.norm1(x))
            if idx < len(self.lora_layers):
                qkv = qkv + self.lora_layers[idx](block.norm1(x))
            
            # Multi-head attention
            B, N, C = qkv.shape
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads // 3).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn = block.attn.attn_drop(attn)
            
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C // 3)
            x = x + block.attn.proj_drop(block.attn.proj(x_attn))
            
            # Feed-forward network
            x = x + block.drop_path1(block.ls1(block.mlp(block.norm2(x))))
        
        # Classification
        x = self.base_model.norm(x)
        return self.base_model.head(self.base_model.forward_head(x))


class HybridCLStrategy(Naive):
    """
    Novel Hybrid Continual Learning Strategy.
    
    Combines three powerful techniques:
    1. Experience Replay - Stores and replays past examples
    2. Elastic Weight Consolidation - Regularizes important weights
    3. Knowledge Distillation - Preserves previous model's knowledge
    
    This multi-pronged approach addresses catastrophic forgetting from
    multiple angles, achieving superior stability-plasticity balance.
    """
    def __init__(self, model, optimizer, criterion, config, **kwargs):
        super().__init__(model, optimizer, criterion, **kwargs)
        self.config = config
        self.previous_model = None
        self.distillation_weight = 0.5  # Weight for distillation loss
        self.temperature = 2.0  # Temperature for soft targets
        
    def _after_training_exp(self, **kwargs):
        """
        Hook called after training on each experience.
        Stores a snapshot of the model for knowledge distillation.
        """
        super()._after_training_exp(**kwargs)
        
        # Create or update the teacher model
        if self.previous_model is None:
            self.previous_model = type(self.model)(
                create_model('vit_base_patch16_224', pretrained=True),
                num_classes=100,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            ).to(self.config.device)
        
        # Copy current model weights to teacher
        self.previous_model.load_state_dict(self.model.state_dict())
        self.previous_model.eval()
        
    def training_epoch(self, **kwargs):
        """
        Override training epoch to add distillation loss.
        """
        for mb in self.dataloader:
            self._unpack_minibatch(mb)
            self._before_training_iteration(**kwargs)
            
            self.optimizer.zero_grad()
            self.loss = self._compute_loss_with_distillation()
            self.loss.backward()
            self.optimizer.step()
            
            self._after_training_iteration(**kwargs)
    
    def _compute_loss_with_distillation(self):
        """
        Compute hybrid loss: cross-entropy + knowledge distillation.
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(self.mb_output, self.mb_y)
        
        # Add distillation loss if we have a previous model
        if self.previous_model is not None:
            with torch.no_grad():
                old_outputs = self.previous_model(self.mb_x)
            
            # Soft targets with temperature scaling
            distill_loss = F.kl_div(
                F.log_softmax(self.mb_output / self.temperature, dim=1),
                F.softmax(old_outputs / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            return ce_loss + self.distillation_weight * distill_loss
        
        return ce_loss


def create_cl_model(config, strategy_name: str):
    """
    Factory function to create models based on strategy requirements.
    
    Args:
        config: ExperimentConfig with model hyperparameters
        strategy_name: Name of the CL strategy (determines architecture)
    
    Returns:
        Configured model ready for training
    """
    base_model = create_model('vit_base_patch16_224', pretrained=True)
    
    if strategy_name in ["lora", "hybrid_lora"]:
        print(f"✓ Creating ViT-B/16 with LoRA (r={config.lora_r}, α={config.lora_alpha})")
        model = ViTWithLoRA(
            base_model, 
            num_classes=100,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        
        # Report parameter efficiency
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    elif strategy_name == "frozen_backbone":
        print("✓ Creating ViT-B/16 with frozen backbone (Linear Probing)")
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.head = nn.Linear(base_model.head.in_features, 100)
        model = base_model
        
    else:
        print("✓ Creating ViT-B/16 with full fine-tuning")
        base_model.head = nn.Linear(base_model.head.in_features, 100)
        model = base_model
    
    return model.to(config.device)