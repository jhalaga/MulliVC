"""
Loss functions for MulliVC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class ReconstructionLoss(nn.Module):
    """L1 and L2 reconstruction loss."""
    
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
        Computes reconstruction loss.

        Args:
            predicted: Predicted tensor.
            target: Target tensor.

        Returns:
            loss: Reconstruction loss.
        """
        if predicted.dim() == 3 and target.dim() == 3:
            if target.shape[1] != predicted.shape[1] and target.shape[2] == predicted.shape[1]:
                target = target.transpose(1, 2)

            if target.shape[-1] != predicted.shape[-1]:
                target = F.interpolate(
                    target,
                    size=predicted.shape[-1],
                    mode='linear',
                    align_corners=False
                )

        l1_loss = self.l1_loss(predicted, target)
        l2_loss = self.l2_loss(predicted, target)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        return total_loss


class TimbreLoss(nn.Module):
    """Timbre loss based on cosine similarity."""
    
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
        Computes timbre loss.

        Args:
            predicted_timbre: Predicted timbre of shape (batch_size, timbre_dim).
            target_timbre: Target timbre of shape (batch_size, timbre_dim).
            negative_timbre: Negative timbre for contrastive learning.

        Returns:
            loss: Timbre loss.
        """
        # Normalize vectors
        predicted_norm = F.normalize(predicted_timbre, p=2, dim=1)
        target_norm = F.normalize(target_timbre, p=2, dim=1)
        
        # Positive cosine similarity
        positive_sim = F.cosine_similarity(predicted_norm, target_norm, dim=1)
        
        if negative_timbre is not None:
            # Negative cosine similarity
            negative_norm = F.normalize(negative_timbre, p=2, dim=1)
            negative_sim = F.cosine_similarity(predicted_norm, negative_norm, dim=1)
            
            # Triplet loss
            loss = F.relu(self.margin - positive_sim + negative_sim).mean()
        else:
            # Simple similarity loss
            loss = (1 - positive_sim).mean()
        
        return loss


class PitchLoss(nn.Module):
    """Pitch (F0) loss."""
    
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
        Computes pitch loss.

        Args:
            predicted_pitch: Predicted pitch of shape (batch_size, seq_len).
            target_pitch: Target pitch of shape (batch_size, seq_len).
            predicted_voiced: Predicted voiced mask.
            target_voiced: Target voiced mask.

        Returns:
            loss: Pitch loss.
        """
        # Pitch loss for voiced segments
        voiced_mask = target_voiced > 0.5
        if voiced_mask.any():
            pitch_loss = self.mse_loss(
                predicted_pitch[voiced_mask],
                target_pitch[voiced_mask]
            )
        else:
            pitch_loss = torch.tensor(0.0, device=predicted_pitch.device)
        
        # Voiced/unvoiced classification loss
        voiced_loss = self.bce_loss(predicted_voiced, target_voiced)
        
        total_loss = (
            self.pitch_weight * pitch_loss + 
            self.voiced_weight * voiced_loss
        )
        
        return total_loss


class ASRLoss(nn.Module):
    """ASR loss to enforce content preservation."""
    
    def __init__(
        self,
        asr_model: Optional[nn.Module] = None,
        weight: float = 1.0
    ):
        super().__init__()
        self.asr_model = asr_model
        self.weight = weight
        
        # If no ASR model is provided, use a simple content loss
        if asr_model is None:
            self.content_loss = nn.MSELoss()
    
    def forward(
        self,
        generated_audio: torch.Tensor,
        target_audio: torch.Tensor,
        target_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Computes ASR loss.

        Args:
            generated_audio: Generated audio.
            target_audio: Target audio.
            target_text: Optional target text.

        Returns:
            loss: ASR loss.
        """
        if self.asr_model is not None:
            # Use the ASR model to extract content features
            with torch.no_grad():
                target_features = self.asr_model.extract_features(target_audio)
            
            generated_features = self.asr_model.extract_features(generated_audio)
            
            # Content loss
            loss = F.mse_loss(generated_features, target_features)
        else:
            # Simple content loss
            loss = self.content_loss(generated_audio, target_audio)
        
        return self.weight * loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
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
        Computes adversarial loss.

        Args:
            discriminator_output: Discriminator output.
            is_real: If True, uses the target for real samples.

        Returns:
            loss: Adversarial loss.
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
            raise ValueError(f"Unsupported GAN mode: {self.gan_mode}")
        
        return self.weight * loss


class PerceptualLoss(nn.Module):
    """Perceptual loss based on a pretrained model."""
    
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
        
        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes perceptual loss.

        Args:
            predicted: Predicted tensor.
            target: Target tensor.

        Returns:
            loss: Perceptual loss.
        """
        # Extract features
        pred_features = self.feature_extractor(predicted)
        target_features = self.feature_extractor(target)
        
        # Compute loss for each layer
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
    """Cycle consistency loss for cross-lingual training."""
    
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
        Computes cycle consistency loss.

        Args:
            reconstructed: Reconstructed tensor.
            original: Original tensor.

        Returns:
            loss: Cycle consistency loss.
        """
        loss = self.l1_loss(reconstructed, original)
        return self.weight * loss


class MultiScaleLoss(nn.Module):
    """Multi-scale loss."""
    
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
        Computes multi-scale loss.

        Args:
            predicted: Predicted tensor.
            target: Target tensor.

        Returns:
            loss: Multi-scale loss.
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
    """Combination of all losses for MulliVC."""
    
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
        
        # Initialize losses
        self.reconstruction_loss = ReconstructionLoss()
        self.timbre_loss = TimbreLoss()
        self.pitch_loss = PitchLoss()
        self.asr_loss = ASRLoss(asr_model=asr_model, weight=self.lambda_asr)
        self.adversarial_loss = AdversarialLoss(weight=self.lambda_adv)
        self.cycle_loss = CycleConsistencyLoss()
        
        # Perceptual loss if available
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
        Computes all combined losses.

        Returns:
            losses: Dictionary containing all losses.
        """
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = self.lambda_rec * self.reconstruction_loss(
            predicted_mel, target_mel
        )
        
        # Timbre loss
        losses['timbre'] = self.lambda_timbre * self.timbre_loss(
            predicted_timbre, target_timbre
        )
        
        # Pitch loss
        if (
            predicted_pitch is not None and
            target_pitch is not None and
            predicted_voiced is not None and
            target_voiced is not None
        ):
            losses['pitch'] = self.lambda_pitch * self.pitch_loss(
                predicted_pitch, target_pitch, predicted_voiced, target_voiced
            )
        else:
            losses['pitch'] = torch.tensor(0.0, device=predicted_mel.device)
        
        # Adversarial loss
        losses['adversarial'] = self.adversarial_loss(
            discriminator_output, is_real
        )
        
        # ASR loss if audio is provided
        if generated_audio is not None and target_audio is not None:
            losses['asr'] = self.asr_loss(generated_audio, target_audio)
        else:
            losses['asr'] = torch.tensor(0.0, device=predicted_mel.device)
        
        # Perceptual loss if available
        if self.perceptual_loss is not None:
            losses['perceptual'] = self.perceptual_loss(predicted_mel, target_mel)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=predicted_mel.device)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
