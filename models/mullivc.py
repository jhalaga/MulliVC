"""
Modèle principal MulliVC
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import yaml

from .content_encoder import ContentEncoder
from .timbre_encoder import TimbreEncoder
from .fine_grained_conformer import FineGrainedTimbreConformer
from .mel_decoder import MelDecoder
from .discriminator import PatchGANDiscriminator
from .losses import CombinedLoss


class MulliVC(nn.Module):
    """
    Modèle principal MulliVC pour la conversion vocale multilingue
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Initialiser les composants
        self.content_encoder = ContentEncoder(
            model_name=config['model']['content_encoder']['model_name'],
            hidden_size=config['model']['content_encoder']['hidden_size'],
            output_dim=config['model']['content_encoder']['output_dim']
        )
        
        self.timbre_encoder = TimbreEncoder(
            input_dim=config['model']['timbre_encoder']['input_dim'],
            hidden_dim=config['model']['timbre_encoder']['hidden_dim'],
            output_dim=config['model']['timbre_encoder']['output_dim'],
            num_layers=config['model']['timbre_encoder']['num_layers']
        )
        
        self.fine_grained_conformer = FineGrainedTimbreConformer(
            input_dim=config['model']['conformer']['input_dim'],
            num_heads=config['model']['conformer']['num_heads'],
            num_layers=config['model']['conformer']['num_layers'],
            conv_kernel_size=config['model']['conformer']['conv_kernel_size'],
            dropout=config['model']['conformer']['dropout']
        )
        
        self.mel_decoder = MelDecoder(
            input_dim=config['model']['mel_decoder']['input_dim'],
            hidden_dim=config['model']['mel_decoder']['hidden_dim'],
            output_dim=config['model']['mel_decoder']['output_dim'],
            num_layers=config['model']['mel_decoder']['num_layers']
        )
        
        self.discriminator = PatchGANDiscriminator(
            input_dim=config['model']['discriminator']['input_dim'],
            hidden_dim=config['model']['discriminator']['hidden_dim'],
            num_layers=config['model']['discriminator']['num_layers'],
            patch_size=config['model']['discriminator']['patch_size']
        )
        
        # Loss function
        self.loss_fn = CombinedLoss(config)
    
    def forward(
        self,
        source_audio: torch.Tensor,
        target_timbre_audio: torch.Tensor,
        source_mel: Optional[torch.Tensor] = None,
        target_timbre_mel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass du modèle MulliVC
        
        Args:
            source_audio: Audio source (batch_size, samples)
            target_timbre_audio: Audio cible pour le timbre (batch_size, samples)
            source_mel: Mél-spectrogramme source optionnel
            target_timbre_mel: Mél-spectrogramme cible optionnel
            
        Returns:
            outputs: Dictionnaire contenant toutes les sorties
        """
        # Encoder le contenu de l'audio source
        content_features, content_pooled = self.content_encoder(source_audio)
        
        # Encoder le timbre de l'audio cible
        if target_timbre_mel is not None:
            timbre_features = self.timbre_encoder.extract_timbre_features(target_timbre_mel)
        else:
            # Générer le mél-spectrogramme si non fourni
            target_timbre_mel = self._audio_to_mel(target_timbre_audio)
            timbre_features = self.timbre_encoder.extract_timbre_features(target_timbre_mel)
        
        # Fine-grained timbre processing
        fine_grained_timbre, global_timbre, attention_weights = self.fine_grained_conformer(
            timbre_features.unsqueeze(1).expand(-1, content_features.shape[1], -1),
            content_features
        )
        
        # Générer le mél-spectrogramme
        generated_mel = self.mel_decoder(content_features, fine_grained_timbre)
        
        # Discriminateur
        discriminator_output, _ = self.discriminator(generated_mel)
        
        outputs = {
            'generated_mel': generated_mel,
            'content_features': content_features,
            'timbre_features': timbre_features,
            'fine_grained_timbre': fine_grained_timbre,
            'global_timbre': global_timbre,
            'discriminator_output': discriminator_output,
            'attention_weights': attention_weights
        }
        
        return outputs
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convertit l'audio en mél-spectrogramme"""
        # Cette fonction devrait utiliser le même transformateur que dans audio_utils
        # Pour l'instant, on utilise un placeholder
        batch_size, samples = audio.shape
        n_mel_channels = self.config['data']['n_mel_channels']
        time_frames = samples // 256  # Approximation
        
        # Placeholder - à remplacer par le vrai transformateur
        mel_spec = torch.randn(batch_size, n_mel_channels, time_frames, device=audio.device)
        
        return mel_spec
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_real: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule toutes les pertes
        
        Args:
            outputs: Sorties du modèle
            targets: Cibles pour le calcul des pertes
            is_real: Si True, cible pour les vrais échantillons
            
        Returns:
            losses: Dictionnaire des pertes
        """
        losses = self.loss_fn(
            predicted_mel=outputs['generated_mel'],
            target_mel=targets.get('target_mel'),
            predicted_timbre=outputs['global_timbre'],
            target_timbre=targets.get('target_timbre'),
            predicted_pitch=outputs.get('predicted_pitch'),
            target_pitch=targets.get('target_pitch'),
            predicted_voiced=outputs.get('predicted_voiced'),
            target_voiced=targets.get('target_voiced'),
            discriminator_output=outputs['discriminator_output'],
            is_real=is_real,
            generated_audio=outputs.get('generated_audio'),
            target_audio=targets.get('target_audio')
        )
        
        return losses
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """
        Une étape d'entraînement complète avec les 3 sous-étapes
        
        Args:
            batch: Batch de données
            step: Numéro d'étape
            
        Returns:
            losses: Dictionnaire des pertes
        """
        # Étape 1: Entraînement standard (monolingue)
        step1_losses = self._training_step_1(batch)
        
        # Étape 2: Conversion croisée simulée
        step2_losses = self._training_step_2(batch)
        
        # Étape 3: Cohérence cyclique
        step3_losses = self._training_step_3(batch)
        
        # Combiner toutes les pertes
        total_losses = {}
        for key in step1_losses.keys():
            total_losses[key] = (
                step1_losses[key] + 
                step2_losses[key] + 
                step3_losses[key]
            ) / 3
        
        return total_losses
    
    def _training_step_1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Étape 1: Entraînement standard (monolingue)"""
        # Utiliser le même locuteur pour contenu et timbre
        source_audio = batch['audio']
        target_audio = batch['audio']  # Même audio
        
        # Forward pass
        outputs = self.forward(source_audio, target_audio)
        
        # Cibles pour la reconstruction
        targets = {
            'target_mel': self._audio_to_mel(target_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Calculer les pertes
        losses = self.compute_losses(outputs, targets, is_real=True)
        
        return losses
    
    def _training_step_2(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Étape 2: Conversion croisée simulée"""
        # Utiliser des locuteurs différents
        source_audio = batch['audio']
        # Simuler un audio de locuteur différent
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)  # Simple simulation
        
        # Forward pass
        outputs = self.forward(source_audio, target_timbre_audio)
        
        # Cibles pour la conversion
        targets = {
            'target_mel': self._audio_to_mel(target_timbre_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Calculer les pertes
        losses = self.compute_losses(outputs, targets, is_real=False)
        
        return losses
    
    def _training_step_3(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Étape 3: Cohérence cyclique"""
        # Utiliser l'audio généré de l'étape 2
        source_audio = batch['audio']
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)
        
        # Forward pass
        outputs = self.forward(source_audio, target_timbre_audio)
        
        # Reconstruction cyclique
        reconstructed_outputs = self.forward(
            outputs['generated_mel'],  # Utiliser le mél généré
            source_audio  # Timbre original
        )
        
        # Cibles pour la cohérence cyclique
        targets = {
            'target_mel': self._audio_to_mel(source_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Calculer les pertes
        losses = self.compute_losses(reconstructed_outputs, targets, is_real=True)
        
        return losses
    
    def inference(
        self,
        source_audio: torch.Tensor,
        target_speaker_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Inférence pour la conversion vocale
        
        Args:
            source_audio: Audio source
            target_speaker_audio: Audio du locuteur cible
            
        Returns:
            generated_mel: Mél-spectrogramme généré
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(source_audio, target_speaker_audio)
            generated_mel = outputs['generated_mel']
        
        return generated_mel
    
    def save_checkpoint(self, path: str, epoch: int, step: int):
        """Sauvegarde le checkpoint du modèle"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Charge un checkpoint du modèle"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['step']


def create_mullivc_model(config_path: str) -> MulliVC:
    """Crée un modèle MulliVC à partir d'un fichier de configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = MulliVC(config)
    return model
