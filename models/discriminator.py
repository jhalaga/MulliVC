"""
Discriminateur PatchGAN pour l'entraînement adversarial
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchGANDiscriminator(nn.Module):
    """
    Discriminateur PatchGAN pour discriminer les mél-spectrogrammes réels et générés
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # mel channels
        hidden_dim: int = 512,
        num_layers: int = 4,
        patch_size: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        
        # Couches de convolution
        self.conv_layers = nn.ModuleList()
        
        # Première couche
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=hidden_dim // 8,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            )
        )
        
        # Couches intermédiaires
        for i in range(1, num_layers - 1):
            in_channels = hidden_dim // (2 ** (4 - i))
            out_channels = hidden_dim // (2 ** (3 - i))
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(dropout)
                )
            )
        
        # Dernière couche
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim // 2,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1
                ),
                nn.Sigmoid()
            )
        )
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialise les poids du modèle"""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        mel_spec: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass du discriminateur
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            return_features: Si True, retourne les features intermédiaires
            
        Returns:
            patch_scores: Tensor de forme (batch_size, 1, height, width)
            features: Features intermédiaires si return_features=True
        """
        # Ajouter une dimension de canal
        x = mel_spec.unsqueeze(1)  # (batch_size, 1, n_mel_channels, time_frames)
        
        features = []
        
        # Appliquer les couches de convolution
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            if return_features and i < len(self.conv_layers) - 1:
                features.append(x)
        
        if return_features:
            return x, features
        else:
            return x, None
    
    def get_patch_scores(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Récupère les scores de patch pour chaque région
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            
        Returns:
            patch_scores: Tensor de forme (batch_size, num_patches)
        """
        patch_scores, _ = self.forward(mel_spec, return_features=False)
        
        # Reshape pour obtenir les scores par patch
        batch_size = patch_scores.shape[0]
        patch_scores = patch_scores.view(batch_size, -1)
        
        return patch_scores


class MultiScaleDiscriminator(nn.Module):
    """
    Discriminateur multi-échelle pour capturer les patterns à différentes résolutions
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 512,
        num_scales: int = 3,
        patch_size: int = 16
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Discriminateurs pour différentes échelles
        self.scale_discriminators = nn.ModuleList([
            PatchGANDiscriminator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                patch_size=patch_size
            ) for _ in range(num_scales)
        ])
        
        # Fusion des scores multi-échelles
        self.score_fusion = nn.Sequential(
            nn.Linear(num_scales, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass du discriminateur multi-échelle
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            
        Returns:
            final_score: Score final fusionné
            scale_scores: Scores pour chaque échelle
        """
        scale_scores = []
        
        # Évaluer à différentes échelles
        for i, discriminator in enumerate(self.scale_discriminators):
            if i > 0:
                # Downsample pour les échelles plus petites
                scale_factor = 2 ** i
                downsampled = F.avg_pool2d(
                    mel_spec.unsqueeze(1),
                    kernel_size=(1, scale_factor)
                ).squeeze(1)
            else:
                downsampled = mel_spec
            
            # Score pour cette échelle
            scale_score, _ = discriminator(downsampled)
            scale_scores.append(scale_score.mean(dim=(2, 3)))  # (batch_size, 1)
        
        # Concaténer les scores
        concatenated_scores = torch.cat(scale_scores, dim=1)  # (batch_size, num_scales)
        
        # Fusionner les scores
        final_score = self.score_fusion(concatenated_scores)
        
        return final_score, torch.cat(scale_scores, dim=1)


class ConditionalDiscriminator(nn.Module):
    """
    Discriminateur conditionnel qui utilise des informations sur le locuteur
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        speaker_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        
        # Encoder pour les features de locuteur
        self.speaker_encoder = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Discriminateur principal
        self.discriminator = PatchGANDiscriminator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Fusion des features audio et speaker
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        mel_spec: torch.Tensor, 
        speaker_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass du discriminateur conditionnel
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            speaker_features: Tensor de forme (batch_size, speaker_dim)
            
        Returns:
            score: Score de discrimination
        """
        # Encoder les features de locuteur
        speaker_encoded = self.speaker_encoder(speaker_features)
        
        # Obtenir les features audio
        audio_features, _ = self.discriminator(mel_spec, return_features=True)
        
        # Pooling global des features audio
        audio_pooled = F.adaptive_avg_pool2d(audio_features, (1, 1))
        audio_pooled = audio_pooled.view(audio_pooled.shape[0], -1)
        
        # Fusionner les features
        combined_features = torch.cat([audio_pooled, speaker_encoded], dim=-1)
        
        # Score final
        score = self.feature_fusion(combined_features)
        
        return score


class SpectralDiscriminator(nn.Module):
    """
    Discriminateur qui analyse les caractéristiques spectrales
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Couches de convolution 1D pour l'analyse spectrale
        self.conv_layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - 1 - i))
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                )
            )
            in_channels = out_channels
        
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du discriminateur spectral
        
        Args:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
            
        Returns:
            score: Score de discrimination
        """
        x = mel_spec
        
        # Appliquer les couches de convolution
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Pooling global
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.shape[0], -1)
        
        # Classification
        score = self.classifier(x)
        
        return score
