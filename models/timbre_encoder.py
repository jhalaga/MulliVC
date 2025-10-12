"""
Timbre Encoder pour extraire les caractéristiques de timbre du locuteur
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TimbreEncoder(nn.Module):
    """
    Timbre Encoder qui extrait les caractéristiques globales de timbre du locuteur
    à partir des mél-spectrogrammes
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
        
        # Couches de convolution pour extraire les caractéristiques locales
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_dim if i == 0 else hidden_dim // (2 ** (i-1)),
                    out_channels=hidden_dim // (2 ** i),
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm2d(hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.Dropout2d(dropout)
            ) for i in range(num_layers)
        ])
        
        # Pooling global pour obtenir une représentation globale
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Couches fully connected pour la projection finale
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim // (2 ** (num_layers - 1)), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Attention pour se concentrer sur les parties importantes
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
        Encode le mél-spectrogramme pour extraire les caractéristiques de timbre
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            return_attention: Si True, retourne les poids d'attention
            
        Returns:
            timbre_features: Tensor de forme (batch_size, output_dim)
            attention_weights: Poids d'attention si return_attention=True
        """
        batch_size = mel_spec.shape[0]
        
        # Ajouter une dimension de canal pour les convolutions 2D
        x = mel_spec.unsqueeze(1)  # (batch_size, 1, n_mel_channels, time_frames)
        
        # Appliquer les couches de convolution
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Pooling global
        x = self.global_pool(x)  # (batch_size, channels, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, channels)
        
        # Projection finale
        timbre_features = self.fc_layers(x)  # (batch_size, output_dim)
        
        # Appliquer l'attention si nécessaire
        if return_attention:
            # Reshape pour l'attention
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
        Méthode simplifiée pour extraire seulement les caractéristiques de timbre
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            
        Returns:
            timbre_features: Tensor de forme (batch_size, output_dim)
        """
        timbre_features, _ = self.forward(mel_spec, return_attention=False)
        return timbre_features


class MultiScaleTimbreEncoder(nn.Module):
    """
    Timbre Encoder multi-échelle qui capture les caractéristiques à différentes résolutions
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
        
        # Encoders pour différentes échelles
        self.scale_encoders = nn.ModuleList([
            TimbreEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim // num_scales
            ) for _ in range(num_scales)
        ])
        
        # Fusion des échelles
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode le timbre à plusieurs échelles
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            
        Returns:
            multi_scale_timbre: Tensor de forme (batch_size, output_dim)
        """
        scale_features = []
        
        # Extraire les caractéristiques à différentes échelles
        for i, encoder in enumerate(self.scale_encoders):
            # Downsample pour différentes échelles
            if i > 0:
                scale_factor = 2 ** i
                downsampled = F.avg_pool2d(
                    mel_spec.unsqueeze(1), 
                    kernel_size=(1, scale_factor)
                ).squeeze(1)
            else:
                downsampled = mel_spec
            
            # Encoder cette échelle
            scale_feat = encoder.extract_timbre_features(downsampled)
            scale_features.append(scale_feat)
        
        # Concaténer et fusionner
        concatenated = torch.cat(scale_features, dim=-1)
        multi_scale_timbre = self.fusion(concatenated)
        
        return multi_scale_timbre


class SpeakerEmbeddingEncoder(nn.Module):
    """
    Encoder spécialisé pour les embeddings de locuteur
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
        
        # Encoder de base
        self.base_encoder = TimbreEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim
        )
        
        # Embeddings de locuteur appris
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        
        # Fusion des embeddings
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
        Encode le timbre avec les embeddings de locuteur
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            speaker_ids: Tensor de forme (batch_size,) avec les IDs des locuteurs
            
        Returns:
            speaker_timbre: Tensor de forme (batch_size, embedding_dim)
        """
        # Extraire les caractéristiques de base
        base_features = self.base_encoder.extract_timbre_features(mel_spec)
        
        # Récupérer les embeddings de locuteur
        speaker_emb = self.speaker_embeddings(speaker_ids)
        
        # Fusionner
        combined = torch.cat([base_features, speaker_emb], dim=-1)
        speaker_timbre = self.fusion(combined)
        
        return speaker_timbre
