"""
Content Encoder basé sur WavLM pour extraire les caractéristiques de contenu linguistique
"""
import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig
from typing import Optional, Tuple


class ContentEncoder(nn.Module):
    """
    Content Encoder qui utilise WavLM pour extraire les caractéristiques de contenu
    linguistique indépendantes du locuteur
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
        
        # Charger le modèle WavLM pré-entraîné
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        if freeze_backbone:
            # Geler les paramètres du backbone
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Projection pour réduire la dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Normalisation pour stabiliser l'entraînement
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self, 
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode l'audio pour extraire les caractéristiques de contenu
        
        Args:
            audio: Tensor de forme (batch_size, seq_len)
            attention_mask: Masque d'attention optionnel
            
        Returns:
            content_features: Tensor de forme (batch_size, seq_len, output_dim)
            pooled_features: Tensor de forme (batch_size, output_dim)
        """
        # Extraire les caractéristiques avec WavLM
        outputs = self.wavlm(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Utiliser les hidden states de la dernière couche
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Projection vers l'espace de sortie
        content_features = self.projection(hidden_states)
        content_features = self.layer_norm(content_features)
        
        # Pooling global pour obtenir une représentation globale
        if attention_mask is not None:
            # Masquer les tokens de padding
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
        Méthode simplifiée pour extraire seulement les caractéristiques de contenu
        
        Args:
            audio: Tensor de forme (batch_size, seq_len)
            return_pooled: Si True, retourne les features pooled, sinon les features séquentielles
            
        Returns:
            features: Tensor de forme (batch_size, output_dim) ou (batch_size, seq_len, output_dim)
        """
        content_features, pooled_features = self.forward(audio)
        
        if return_pooled:
            return pooled_features
        else:
            return content_features
    
    def get_attention_weights(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Récupère les poids d'attention du modèle WavLM
        
        Args:
            audio: Tensor de forme (batch_size, seq_len)
            
        Returns:
            attention_weights: Tensor de forme (batch_size, num_heads, seq_len, seq_len)
        """
        outputs = self.wavlm(
            input_values=audio,
            output_attentions=True
        )
        
        # Récupérer les attentions de la dernière couche
        attention_weights = outputs.attentions[-1]
        
        return attention_weights


class ContentVecEncoder(nn.Module):
    """
    Alternative Content Encoder utilisant ContentVec
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/contentvec",
        output_dim: int = 256
    ):
        super().__init__()
        
        # Charger ContentVec (remplacer par l'implémentation réelle)
        # Pour l'instant, on utilise un placeholder
        self.contentvec = None  # À implémenter avec le vrai ContentVec
        
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode l'audio avec ContentVec
        
        Args:
            audio: Tensor de forme (batch_size, seq_len)
            
        Returns:
            content_features: Tensor de forme (batch_size, seq_len, output_dim)
        """
        # Placeholder - à implémenter avec le vrai ContentVec
        batch_size, seq_len = audio.shape
        hidden_size = 768
        
        # Simulation des features ContentVec
        contentvec_features = torch.randn(batch_size, seq_len, hidden_size)
        
        # Projection
        content_features = self.projection(contentvec_features)
        
        return content_features


class MultiScaleContentEncoder(nn.Module):
    """
    Content Encoder multi-échelle qui combine plusieurs niveaux de représentation
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
        
        # Projections pour différentes échelles
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_scales)
        ])
        
        # Fusion des échelles
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * num_scales, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode l'audio avec plusieurs échelles
        
        Args:
            audio: Tensor de forme (batch_size, seq_len)
            
        Returns:
            multi_scale_features: Tensor de forme (batch_size, seq_len, output_dim)
        """
        outputs = self.wavlm(
            input_values=audio,
            output_hidden_states=True
        )
        
        # Utiliser les hidden states de différentes couches
        hidden_states = outputs.hidden_states[-self.num_scales:]
        
        # Projection pour chaque échelle
        scale_features = []
        for i, hidden_state in enumerate(hidden_states):
            scale_feat = self.scale_projections[i](hidden_state)
            scale_features.append(scale_feat)
        
        # Fusion des échelles
        concatenated = torch.cat(scale_features, dim=-1)
        multi_scale_features = self.fusion(concatenated)
        
        return multi_scale_features
