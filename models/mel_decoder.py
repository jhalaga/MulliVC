"""
Mel Decoder for generating mel spectrograms from content and timbre features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MelDecoder(nn.Module):
    """
    Mel Decoder that generates mel spectrograms from content
    and timbre features.
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # content + timbre features
        hidden_dim: int = 512,
        output_dim: int = 80,  # mel channels
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformation layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Convolution layers for generation
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim if i == 0 else hidden_dim // (2 ** (i-1)),
                    out_channels=hidden_dim // (2 ** i),
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm1d(hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim // (2 ** (num_layers - 1)), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(
        self, 
        content_features: torch.Tensor,
        timbre_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates mel spectrograms from features.

        Args:
            content_features: Tensor of shape (batch_size, seq_len, content_dim).
            timbre_features: Tensor of shape (batch_size, seq_len, timbre_dim).
            mask: Optional attention mask.

        Returns:
            mel_spec: Tensor of shape (batch_size, seq_len, n_mel_channels).
        """
        # Concatenate content and timbre features
        combined_features = torch.cat([content_features, timbre_features], dim=-1)
        
        # Input projection
        x = self.input_projection(combined_features)
        
        # Apply transformation layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, src_key_padding_mask=mask)
        
        # Apply convolution layers
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Output projection
        mel_spec = self.output_projection(x)
        
        return mel_spec


class ConditionalMelDecoder(nn.Module):
    """
    Conditional Mel Decoder that uses additional information
    such as pitch and speaker characteristics.
    """
    
    def __init__(
        self,
        content_dim: int = 256,
        timbre_dim: int = 256,
        pitch_dim: int = 1,
        speaker_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 80,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.content_dim = content_dim
        self.timbre_dim = timbre_dim
        self.pitch_dim = pitch_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projections for each feature type
        self.content_projection = nn.Linear(content_dim, hidden_dim // 4)
        self.timbre_projection = nn.Linear(timbre_dim, hidden_dim // 4)
        self.pitch_projection = nn.Linear(pitch_dim, hidden_dim // 4)
        self.speaker_projection = nn.Linear(speaker_dim, hidden_dim // 4)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main decoder
        self.decoder = MelDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
    
    def forward(
        self,
        content_features: torch.Tensor,
        timbre_features: torch.Tensor,
        pitch_features: torch.Tensor,
        speaker_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates mel spectrograms with additional conditions.

        Args:
            content_features: Tensor of shape (batch_size, seq_len, content_dim).
            timbre_features: Tensor of shape (batch_size, seq_len, timbre_dim).
            pitch_features: Tensor of shape (batch_size, seq_len, pitch_dim).
            speaker_features: Tensor of shape (batch_size, speaker_dim).
            mask: Optional attention mask.

        Returns:
            mel_spec: Tensor of shape (batch_size, seq_len, n_mel_channels).
        """
        # Projections
        content_proj = self.content_projection(content_features)
        timbre_proj = self.timbre_projection(timbre_features)
        pitch_proj = self.pitch_projection(pitch_features)
        
        # Expand speaker features
        speaker_proj = self.speaker_projection(speaker_features)
        speaker_proj = speaker_proj.unsqueeze(1).expand(-1, content_features.shape[1], -1)
        
        # Merge all features
        combined_features = torch.cat([
            content_proj, timbre_proj, pitch_proj, speaker_proj
        ], dim=-1)
        
        # Fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Decode
        mel_spec = self.decoder(fused_features, fused_features, mask)
        
        return mel_spec


class MultiScaleMelDecoder(nn.Module):
    """
    Multi-scale Mel Decoder that generates spectrograms at different resolutions.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 80,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Decoders for different scales
        self.scale_decoders = nn.ModuleList([
            MelDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=4
            ) for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(output_dim * num_scales, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(
        self,
        content_features: torch.Tensor,
        timbre_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates multi-scale mel spectrograms.

        Args:
            content_features: Tensor of shape (batch_size, seq_len, content_dim).
            timbre_features: Tensor of shape (batch_size, seq_len, timbre_dim).
            mask: Optional attention mask.

        Returns:
            mel_spec: Tensor of shape (batch_size, seq_len, n_mel_channels).
        """
        scale_outputs = []
        
        # Generate at different scales
        for i, decoder in enumerate(self.scale_decoders):
            if i > 0:
                # Downsample for smaller scales
                scale_factor = 2 ** i
                downsampled_content = F.avg_pool1d(
                    content_features.transpose(1, 2),
                    kernel_size=scale_factor
                ).transpose(1, 2)
                downsampled_timbre = F.avg_pool1d(
                    timbre_features.transpose(1, 2),
                    kernel_size=scale_factor
                ).transpose(1, 2)
            else:
                downsampled_content = content_features
                downsampled_timbre = timbre_features
            
            # Decode this scale
            scale_output = decoder(downsampled_content, downsampled_timbre, mask)
            
            # Upsample if needed
            if i > 0:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=content_features.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(scale_output)
        
        # Merge scales
        concatenated = torch.cat(scale_outputs, dim=-1)
        mel_spec = self.scale_fusion(concatenated)
        
        return mel_spec


class AdversarialMelDecoder(nn.Module):
    """
    Mel Decoder with an adversarial component to improve quality.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 80,
        num_layers: int = 4
    ):
        super().__init__()
        
        # Main decoder
        self.decoder = MelDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        # Adversarial component
        self.adversarial_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        content_features: torch.Tensor,
        timbre_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_adversarial: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generates mel spectrograms with an adversarial score.

        Args:
            content_features: Tensor of shape (batch_size, seq_len, content_dim).
            timbre_features: Tensor of shape (batch_size, seq_len, timbre_dim).
            mask: Optional attention mask.
            return_adversarial: If True, returns the adversarial score.

        Returns:
            mel_spec: Tensor of shape (batch_size, seq_len, n_mel_channels).
            adversarial_score: Adversarial score if return_adversarial=True.
        """
        # Generate the mel spectrogram
        mel_spec = self.decoder(content_features, timbre_features, mask)
        
        adversarial_score = None
        if return_adversarial:
            # Compute the adversarial score
            adversarial_score = self.adversarial_head(mel_spec)
            adversarial_score = adversarial_score.mean(dim=1)  # (batch_size, 1)
        
        return mel_spec, adversarial_score
