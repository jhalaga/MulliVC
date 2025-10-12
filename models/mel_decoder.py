"""
Mel Decoder pour générer les mél-spectrogrammes à partir des features de contenu et de timbre
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MelDecoder(nn.Module):
    """
    Mel Decoder qui génère les mél-spectrogrammes à partir des features
    de contenu et de timbre
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
        
        # Couches de transformation
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Couches de convolution pour la génération
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
        
        # Activation finale
        self.output_activation = nn.Tanh()
    
    def forward(
        self, 
        content_features: torch.Tensor,
        timbre_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Génère les mél-spectrogrammes à partir des features
        
        Args:
            content_features: Tensor de forme (batch_size, seq_len, content_dim)
            timbre_features: Tensor de forme (batch_size, seq_len, timbre_dim)
            mask: Masque d'attention optionnel
            
        Returns:
            mel_spec: Tensor de forme (batch_size, seq_len, n_mel_channels)
        """
        # Concaténer les features de contenu et de timbre
        combined_features = torch.cat([content_features, timbre_features], dim=-1)
        
        # Input projection
        x = self.input_projection(combined_features)
        
        # Appliquer les couches de transformation
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, src_key_padding_mask=mask)
        
        # Appliquer les couches de convolution
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Output projection
        mel_spec = self.output_projection(x)
        mel_spec = self.output_activation(mel_spec)
        
        return mel_spec


class ConditionalMelDecoder(nn.Module):
    """
    Mel Decoder conditionnel qui utilise des informations supplémentaires
    comme le pitch et les caractéristiques du locuteur
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
        
        # Projections pour chaque type de feature
        self.content_projection = nn.Linear(content_dim, hidden_dim // 4)
        self.timbre_projection = nn.Linear(timbre_dim, hidden_dim // 4)
        self.pitch_projection = nn.Linear(pitch_dim, hidden_dim // 4)
        self.speaker_projection = nn.Linear(speaker_dim, hidden_dim // 4)
        
        # Fusion des features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder principal
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
        Génère les mél-spectrogrammes avec des conditions supplémentaires
        
        Args:
            content_features: Tensor de forme (batch_size, seq_len, content_dim)
            timbre_features: Tensor de forme (batch_size, seq_len, timbre_dim)
            pitch_features: Tensor de forme (batch_size, seq_len, pitch_dim)
            speaker_features: Tensor de forme (batch_size, speaker_dim)
            mask: Masque d'attention optionnel
            
        Returns:
            mel_spec: Tensor de forme (batch_size, seq_len, n_mel_channels)
        """
        # Projections
        content_proj = self.content_projection(content_features)
        timbre_proj = self.timbre_projection(timbre_features)
        pitch_proj = self.pitch_projection(pitch_features)
        
        # Étendre les features de locuteur
        speaker_proj = self.speaker_projection(speaker_features)
        speaker_proj = speaker_proj.unsqueeze(1).expand(-1, content_features.shape[1], -1)
        
        # Fusionner toutes les features
        combined_features = torch.cat([
            content_proj, timbre_proj, pitch_proj, speaker_proj
        ], dim=-1)
        
        # Fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Décoder
        mel_spec = self.decoder(fused_features, fused_features, mask)
        
        return mel_spec


class MultiScaleMelDecoder(nn.Module):
    """
    Mel Decoder multi-échelle qui génère des spectrogrammes à différentes résolutions
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
        
        # Decoders pour différentes échelles
        self.scale_decoders = nn.ModuleList([
            MelDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=4
            ) for _ in range(num_scales)
        ])
        
        # Fusion des échelles
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
        Génère des mél-spectrogrammes multi-échelles
        
        Args:
            content_features: Tensor de forme (batch_size, seq_len, content_dim)
            timbre_features: Tensor de forme (batch_size, seq_len, timbre_dim)
            mask: Masque d'attention optionnel
            
        Returns:
            mel_spec: Tensor de forme (batch_size, seq_len, n_mel_channels)
        """
        scale_outputs = []
        
        # Générer à différentes échelles
        for i, decoder in enumerate(self.scale_decoders):
            if i > 0:
                # Downsample pour les échelles plus petites
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
            
            # Décoder cette échelle
            scale_output = decoder(downsampled_content, downsampled_timbre, mask)
            
            # Upsample si nécessaire
            if i > 0:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=content_features.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(scale_output)
        
        # Fusionner les échelles
        concatenated = torch.cat(scale_outputs, dim=-1)
        mel_spec = self.scale_fusion(concatenated)
        
        return mel_spec


class AdversarialMelDecoder(nn.Module):
    """
    Mel Decoder avec composant adversarial pour améliorer la qualité
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 80,
        num_layers: int = 4
    ):
        super().__init__()
        
        # Decoder principal
        self.decoder = MelDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        # Composant adversarial
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
        Génère les mél-spectrogrammes avec score adversarial
        
        Args:
            content_features: Tensor de forme (batch_size, seq_len, content_dim)
            timbre_features: Tensor de forme (batch_size, seq_len, timbre_dim)
            mask: Masque d'attention optionnel
            return_adversarial: Si True, retourne le score adversarial
            
        Returns:
            mel_spec: Tensor de forme (batch_size, seq_len, n_mel_channels)
            adversarial_score: Score adversarial si return_adversarial=True
        """
        # Générer le mél-spectrogramme
        mel_spec = self.decoder(content_features, timbre_features, mask)
        
        adversarial_score = None
        if return_adversarial:
            # Calculer le score adversarial
            adversarial_score = self.adversarial_head(mel_spec)
            adversarial_score = adversarial_score.mean(dim=1)  # (batch_size, 1)
        
        return mel_spec, adversarial_score
