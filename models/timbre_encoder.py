"""
Timbre Encoder for extracting speaker timbre features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TimbreEncoder(nn.Module):
    """
    Timbre Encoder that extracts global speaker timbre features
    from mel spectrograms.
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # mel channels
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Convolution layers to extract local features
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1 if i == 0 else hidden_dim // (2 ** (i-1)),
                    out_channels=hidden_dim // (2 ** i),
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm2d(hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.Dropout2d(dropout)
            ) for i in range(num_layers)
        ])
        
        # Global pooling to obtain a global representation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for the final projection
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim // (2 ** (num_layers - 1)), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Attention to focus on important parts
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self, 
        mel_spec: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes a mel spectrogram to extract timbre features.

        Args:
            mel_spec: Tensor of shape (batch_size, n_mel_channels, time_frames).
            return_attention: If True, returns attention weights.

        Returns:
            timbre_features: Tensor of shape (batch_size, output_dim).
            attention_weights: Attention weights if return_attention=True.
        """
        batch_size = mel_spec.shape[0]
        
        # Add a channel dimension for 2D convolutions
        x = mel_spec.unsqueeze(1)  # (batch_size, 1, n_mel_channels, time_frames)
        
        # Apply convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, channels, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, channels)
        
        # Final projection
        timbre_features = self.fc_layers(x)  # (batch_size, output_dim)
        
        # Apply attention if needed
        if return_attention:
            # Reshape for attention
            timbre_features_reshaped = timbre_features.unsqueeze(1)  # (batch_size, 1, output_dim)
            
            # Self-attention
            attended_features, attention_weights = self.attention(
                timbre_features_reshaped,
                timbre_features_reshaped,
                timbre_features_reshaped
            )
            
            attended_features = attended_features.squeeze(1)  # (batch_size, output_dim)
            
            return attended_features, attention_weights
        else:
            return timbre_features, None
    
    def extract_timbre_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Simplified method to extract only timbre features.

        Args:
            mel_spec: Tensor of shape (batch_size, n_mel_channels, time_frames).

        Returns:
            timbre_features: Tensor of shape (batch_size, output_dim).
        """
        timbre_features, _ = self.forward(mel_spec, return_attention=False)
        return timbre_features


class MultiScaleTimbreEncoder(nn.Module):
    """
    Multi-scale Timbre Encoder that captures features at different resolutions.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.output_dim = output_dim
        
        # Encoders for different scales
        self.scale_encoders = nn.ModuleList([
            TimbreEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim // num_scales
            ) for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encodes timbre at multiple scales.

        Args:
            mel_spec: Tensor of shape (batch_size, n_mel_channels, time_frames).

        Returns:
            multi_scale_timbre: Tensor of shape (batch_size, output_dim).
        """
        scale_features = []
        
        # Extract features at different scales
        for i, encoder in enumerate(self.scale_encoders):
            # Downsample for different scales
            if i > 0:
                scale_factor = 2 ** i
                downsampled = F.avg_pool2d(
                    mel_spec.unsqueeze(1), 
                    kernel_size=(1, scale_factor)
                ).squeeze(1)
            else:
                downsampled = mel_spec
            
            # Encode this scale
            scale_feat = encoder.extract_timbre_features(downsampled)
            scale_features.append(scale_feat)
        
        # Concatenate and fuse
        concatenated = torch.cat(scale_features, dim=-1)
        multi_scale_timbre = self.fusion(concatenated)
        
        return multi_scale_timbre


class SpeakerEmbeddingEncoder(nn.Module):
    """
    Encoder specialized for speaker embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        embedding_dim: int = 256,
        num_speakers: int = 1000
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        # Base encoder
        self.base_encoder = TimbreEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim
        )
        
        # Learned speaker embeddings
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        
        # Embedding fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(
        self, 
        mel_spec: torch.Tensor, 
        speaker_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes timbre with speaker embeddings.

        Args:
            mel_spec: Tensor of shape (batch_size, n_mel_channels, time_frames).
            speaker_ids: Tensor of shape (batch_size,) containing speaker IDs.

        Returns:
            speaker_timbre: Tensor of shape (batch_size, embedding_dim).
        """
        # Extract base features
        base_features = self.base_encoder.extract_timbre_features(mel_spec)
        
        # Retrieve speaker embeddings
        speaker_emb = self.speaker_embeddings(speaker_ids)
        
        # Fuse
        combined = torch.cat([base_features, speaker_emb], dim=-1)
        speaker_timbre = self.fusion(combined)
        
        return speaker_timbre
