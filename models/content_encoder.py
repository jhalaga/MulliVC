"""
Content Encoder based on WavLM for extracting linguistic content features.
"""
import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig
from typing import Optional, Tuple


class ContentEncoder(nn.Module):
    """
    Content Encoder that uses WavLM to extract speaker-independent
    linguistic content features.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        hidden_size: int = 768,
        output_dim: int = 256,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # Load the pretrained WavLM model
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        if freeze_backbone:
            # Freeze backbone parameters
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Projection to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Normalization to stabilize training
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self, 
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes audio to extract content features.

        Args:
            audio: Tensor of shape (batch_size, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            content_features: Tensor of shape (batch_size, seq_len, output_dim).
            pooled_features: Tensor of shape (batch_size, output_dim).
        """
        # Extract features with WavLM
        outputs = self.wavlm(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last-layer hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Project to the output space
        content_features = self.projection(hidden_states)
        content_features = self.layer_norm(content_features)
        
        # Global pooling to obtain a global representation
        if attention_mask is not None:
            # Mask padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(content_features)
            masked_features = content_features * mask
            pooled_features = masked_features.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_features = content_features.mean(dim=1)
        
        return content_features, pooled_features
    
    def extract_content_features(
        self, 
        audio: torch.Tensor,
        return_pooled: bool = True
    ) -> torch.Tensor:
        """
        Simplified method to extract only content features.

        Args:
            audio: Tensor of shape (batch_size, seq_len).
            return_pooled: If True, returns pooled features, otherwise sequential features.

        Returns:
            features: Tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim).
        """
        content_features, pooled_features = self.forward(audio)
        
        if return_pooled:
            return pooled_features
        else:
            return content_features
    
    def get_attention_weights(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Retrieves attention weights from the WavLM model.

        Args:
            audio: Tensor of shape (batch_size, seq_len).

        Returns:
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        outputs = self.wavlm(
            input_values=audio,
            output_attentions=True
        )
        
        # Retrieve the last-layer attentions
        attention_weights = outputs.attentions[-1]
        
        return attention_weights


class ContentVecEncoder(nn.Module):
    """
    Alternative Content Encoder using ContentVec.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/contentvec",
        output_dim: int = 256
    ):
        super().__init__()
        
        # Load ContentVec (replace with the real implementation)
        # For now, a placeholder is used
        self.contentvec = None  # To be implemented with the real ContentVec
        
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes audio with ContentVec.

        Args:
            audio: Tensor of shape (batch_size, seq_len).

        Returns:
            content_features: Tensor of shape (batch_size, seq_len, output_dim).
        """
        # Placeholder - to be implemented with the real ContentVec
        batch_size, seq_len = audio.shape
        hidden_size = 768
        
        # Simulate ContentVec features
        contentvec_features = torch.randn(batch_size, seq_len, hidden_size)
        
        # Projection
        content_features = self.projection(contentvec_features)
        
        return content_features


class MultiScaleContentEncoder(nn.Module):
    """
    Multi-scale Content Encoder that combines several representation levels.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        output_dim: int = 256,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.num_scales = num_scales
        self.output_dim = output_dim
        
        # Projections for different scales
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * num_scales, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes audio with multiple scales.

        Args:
            audio: Tensor of shape (batch_size, seq_len).
            
        Returns:
            multi_scale_features: Tensor of shape (batch_size, seq_len, output_dim).
        """
        outputs = self.wavlm(
            input_values=audio,
            output_hidden_states=True
        )
        
        # Use hidden states from different layers
        hidden_states = outputs.hidden_states[-self.num_scales:]
        
        # Projection for each scale
        scale_features = []
        for i, hidden_state in enumerate(hidden_states):
            scale_feat = self.scale_projections[i](hidden_state)
            scale_features.append(scale_feat)
        
        # Scale fusion
        concatenated = torch.cat(scale_features, dim=-1)
        multi_scale_features = self.fusion(concatenated)
        
        return multi_scale_features
