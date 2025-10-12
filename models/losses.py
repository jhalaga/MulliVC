"""
Fonctions de perte pour MulliVC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class ReconstructionLoss(nn.Module):
    """Perte de reconstruction L1 et L2"""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.reduction = reduction
        
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte de reconstruction
        
        Args:
            predicted: Tensor prédit
            target: Tensor cible
            
        Returns:
            loss: Perte de reconstruction
        """
        l1_loss = self.l1_loss(predicted, target)
        l2_loss = self.l2_loss(predicted, target)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        return total_loss


class TimbreLoss(nn.Module):
    """Perte de timbre basée sur la similarité cosinus"""
    
    def __init__(
        self,
        margin: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        predicted_timbre: torch.Tensor,
        target_timbre: torch.Tensor,
        negative_timbre: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule la perte de timbre
        
        Args:
            predicted_timbre: Timbre prédit (batch_size, timbre_dim)
            target_timbre: Timbre cible (batch_size, timbre_dim)
            negative_timbre: Timbre négatif pour contraste (batch_size, timbre_dim)
            
        Returns:
            loss: Perte de timbre
        """
        # Normaliser les vecteurs
        predicted_norm = F.normalize(predicted_timbre, p=2, dim=1)
        target_norm = F.normalize(target_timbre, p=2, dim=1)
        
        # Similarité cosinus positive
        positive_sim = F.cosine_similarity(predicted_norm, target_norm, dim=1)
        
        if negative_timbre is not None:
            # Similarité cosinus négative
            negative_norm = F.normalize(negative_timbre, p=2, dim=1)
            negative_sim = F.cosine_similarity(predicted_norm, negative_norm, dim=1)
            
            # Triplet loss
            loss = F.relu(self.margin - positive_sim + negative_sim).mean()
        else:
            # Perte de similarité simple
            loss = (1 - positive_sim).mean()
        
        return loss


class PitchLoss(nn.Module):
    """Perte de pitch (F0)"""
    
    def __init__(
        self,
        pitch_weight: float = 1.0,
        voiced_weight: float = 0.5
    ):
        super().__init__()
        self.pitch_weight = pitch_weight
        self.voiced_weight = voiced_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        predicted_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        predicted_voiced: torch.Tensor,
        target_voiced: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte de pitch
        
        Args:
            predicted_pitch: Pitch prédit (batch_size, seq_len)
            target_pitch: Pitch cible (batch_size, seq_len)
            predicted_voiced: Voiced prédit (batch_size, seq_len)
            target_voiced: Voiced cible (batch_size, seq_len)
            
        Returns:
            loss: Perte de pitch
        """
        # Perte de pitch pour les segments voiced
        voiced_mask = target_voiced > 0.5
        if voiced_mask.any():
            pitch_loss = self.mse_loss(
                predicted_pitch[voiced_mask],
                target_pitch[voiced_mask]
            )
        else:
            pitch_loss = torch.tensor(0.0, device=predicted_pitch.device)
        
        # Perte de classification voiced/unvoiced
        voiced_loss = self.bce_loss(predicted_voiced, target_voiced)
        
        total_loss = (
            self.pitch_weight * pitch_loss + 
            self.voiced_weight * voiced_loss
        )
        
        return total_loss


class ASRLoss(nn.Module):
    """Perte ASR pour forcer la préservation du contenu"""
    
    def __init__(
        self,
        asr_model: Optional[nn.Module] = None,
        weight: float = 1.0
    ):
        super().__init__()
        self.asr_model = asr_model
        self.weight = weight
        
        # Si pas de modèle ASR fourni, utiliser une perte de contenu simple
        if asr_model is None:
            self.content_loss = nn.MSELoss()
    
    def forward(
        self,
        generated_audio: torch.Tensor,
        target_audio: torch.Tensor,
        target_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Calcule la perte ASR
        
        Args:
            generated_audio: Audio généré
            target_audio: Audio cible
            target_text: Texte cible optionnel
            
        Returns:
            loss: Perte ASR
        """
        if self.asr_model is not None:
            # Utiliser le modèle ASR pour extraire les features de contenu
            with torch.no_grad():
                target_features = self.asr_model.extract_features(target_audio)
            
            generated_features = self.asr_model.extract_features(generated_audio)
            
            # Perte de contenu
            loss = F.mse_loss(generated_features, target_features)
        else:
            # Perte de contenu simple
            loss = self.content_loss(generated_audio, target_audio)
        
        return self.weight * loss


class AdversarialLoss(nn.Module):
    """Perte adversarial pour l'entraînement GAN"""
    
    def __init__(
        self,
        gan_mode: str = 'lsgan',  # 'lsgan', 'wgangp', 'hinge'
        weight: float = 1.0
    ):
        super().__init__()
        self.gan_mode = gan_mode
        self.weight = weight
    
    def forward(
        self,
        discriminator_output: torch.Tensor,
        is_real: bool = True
    ) -> torch.Tensor:
        """
        Calcule la perte adversarial
        
        Args:
            discriminator_output: Sortie du discriminateur
            is_real: Si True, cible pour les vrais échantillons
            
        Returns:
            loss: Perte adversarial
        """
        if self.gan_mode == 'lsgan':
            if is_real:
                target = torch.ones_like(discriminator_output)
            else:
                target = torch.zeros_like(discriminator_output)
            loss = F.mse_loss(discriminator_output, target)
        
        elif self.gan_mode == 'wgangp':
            if is_real:
                loss = -discriminator_output.mean()
            else:
                loss = discriminator_output.mean()
        
        elif self.gan_mode == 'hinge':
            if is_real:
                loss = F.relu(1 - discriminator_output).mean()
            else:
                loss = F.relu(1 + discriminator_output).mean()
        
        else:
            raise ValueError(f"Mode GAN non supporté: {self.gan_mode}")
        
        return self.weight * loss


class PerceptualLoss(nn.Module):
    """Perte perceptuelle basée sur un modèle pré-entraîné"""
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        layers: list = [0, 1, 2, 3],
        weights: list = [1.0, 1.0, 1.0, 1.0]
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers
        self.weights = weights
        
        # Geler les paramètres du feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte perceptuelle
        
        Args:
            predicted: Tensor prédit
            target: Tensor cible
            
        Returns:
            loss: Perte perceptuelle
        """
        # Extraire les features
        pred_features = self.feature_extractor(predicted)
        target_features = self.feature_extractor(target)
        
        # Calculer la perte pour chaque couche
        total_loss = 0.0
        for i, (layer, weight) in enumerate(zip(self.layers, self.weights)):
            if layer < len(pred_features):
                layer_loss = F.mse_loss(
                    pred_features[layer],
                    target_features[layer]
                )
                total_loss += weight * layer_loss
        
        return total_loss


class CycleConsistencyLoss(nn.Module):
    """Perte de cohérence cyclique pour l'entraînement cross-lingual"""
    
    def __init__(
        self,
        weight: float = 1.0
    ):
        super().__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte de cohérence cyclique
        
        Args:
            reconstructed: Tensor reconstruit
            original: Tensor original
            
        Returns:
            loss: Perte de cohérence cyclique
        """
        loss = self.l1_loss(reconstructed, original)
        return self.weight * loss


class MultiScaleLoss(nn.Module):
    """Perte multi-échelle"""
    
    def __init__(
        self,
        base_loss: nn.Module,
        scales: list = [1, 2, 4],
        weights: list = [1.0, 0.5, 0.25]
    ):
        super().__init__()
        self.base_loss = base_loss
        self.scales = scales
        self.weights = weights
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte multi-échelle
        
        Args:
            predicted: Tensor prédit
            target: Tensor cible
            
        Returns:
            loss: Perte multi-échelle
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale > 1:
                # Downsample
                pred_scaled = F.avg_pool2d(predicted, kernel_size=scale)
                target_scaled = F.avg_pool2d(target, kernel_size=scale)
            else:
                pred_scaled = predicted
                target_scaled = target
            
            scale_loss = self.base_loss(pred_scaled, target_scaled)
            total_loss += weight * scale_loss
        
        return total_loss


class CombinedLoss(nn.Module):
    """Combinaison de toutes les pertes pour MulliVC"""
    
    def __init__(
        self,
        config: dict,
        asr_model: Optional[nn.Module] = None,
        feature_extractor: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.config = config
        self.lambda_adv = config['training']['lambda_adv']
        self.lambda_rec = config['training']['lambda_rec']
        self.lambda_timbre = config['training']['lambda_timbre']
        self.lambda_pitch = config['training']['lambda_pitch']
        self.lambda_asr = config['training']['lambda_asr']
        
        # Initialiser les pertes
        self.reconstruction_loss = ReconstructionLoss()
        self.timbre_loss = TimbreLoss()
        self.pitch_loss = PitchLoss()
        self.asr_loss = ASRLoss(asr_model=asr_model, weight=self.lambda_asr)
        self.adversarial_loss = AdversarialLoss(weight=self.lambda_adv)
        self.cycle_loss = CycleConsistencyLoss()
        
        # Perte perceptuelle si disponible
        if feature_extractor is not None:
            self.perceptual_loss = PerceptualLoss(feature_extractor)
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
        predicted_timbre: torch.Tensor,
        target_timbre: torch.Tensor,
        predicted_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        predicted_voiced: torch.Tensor,
        target_voiced: torch.Tensor,
        discriminator_output: torch.Tensor,
        is_real: bool = True,
        generated_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule toutes les pertes combinées
        
        Returns:
            losses: Dictionnaire contenant toutes les pertes
        """
        losses = {}
        
        # Perte de reconstruction
        losses['reconstruction'] = self.lambda_rec * self.reconstruction_loss(
            predicted_mel, target_mel
        )
        
        # Perte de timbre
        losses['timbre'] = self.lambda_timbre * self.timbre_loss(
            predicted_timbre, target_timbre
        )
        
        # Perte de pitch
        losses['pitch'] = self.lambda_pitch * self.pitch_loss(
            predicted_pitch, target_pitch, predicted_voiced, target_voiced
        )
        
        # Perte adversarial
        losses['adversarial'] = self.adversarial_loss(
            discriminator_output, is_real
        )
        
        # Perte ASR si audio fourni
        if generated_audio is not None and target_audio is not None:
            losses['asr'] = self.asr_loss(generated_audio, target_audio)
        else:
            losses['asr'] = torch.tensor(0.0, device=predicted_mel.device)
        
        # Perte perceptuelle si disponible
        if self.perceptual_loss is not None:
            losses['perceptual'] = self.perceptual_loss(predicted_mel, target_mel)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=predicted_mel.device)
        
        # Perte totale
        losses['total'] = sum(losses.values())
        
        return losses
