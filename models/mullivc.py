"""
Main MulliVC model.
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
    Main MulliVC model for multilingual voice conversion.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Initialize components
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
        Forward pass of the MulliVC model.

        Args:
            source_audio: Source audio of shape (batch_size, samples).
            target_timbre_audio: Target audio for timbre of shape (batch_size, samples).
            source_mel: Optional source mel spectrogram.
            target_timbre_mel: Optional target mel spectrogram.

        Returns:
            outputs: Dictionary containing all outputs.
        """
        # Encode source audio content
        content_features, content_pooled = self.content_encoder(source_audio)
        
        # Encode target audio timbre
        if target_timbre_mel is not None:
            timbre_features = self.timbre_encoder.extract_timbre_features(target_timbre_mel)
        else:
            # Generate the mel spectrogram if not provided
            target_timbre_mel = self._audio_to_mel(target_timbre_audio)
            timbre_features = self.timbre_encoder.extract_timbre_features(target_timbre_mel)
        
        # Fine-grained timbre processing
        fine_grained_timbre, global_timbre, attention_weights = self.fine_grained_conformer(
            timbre_features.unsqueeze(1).expand(-1, content_features.shape[1], -1),
            content_features
        )
        
        # Generate the mel spectrogram
        generated_mel = self.mel_decoder(content_features, fine_grained_timbre)
        
        # Discriminator
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
        """Converts audio to a mel spectrogram."""
        # This function should use the same transform as in audio_utils
        # For now, a placeholder is used
        batch_size, samples = audio.shape
        n_mel_channels = self.config['data']['n_mel_channels']
        time_frames = samples // 256  # Approximation
        
        # Placeholder - to be replaced with the real transform
        mel_spec = torch.randn(batch_size, n_mel_channels, time_frames, device=audio.device)
        
        return mel_spec
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_real: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Computes all losses.

        Args:
            outputs: Model outputs.
            targets: Targets for loss computation.
            is_real: If True, uses the target for real samples.

        Returns:
            losses: Dictionary of losses.
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
        A full training step with the three sub-steps.

        Args:
            batch: Data batch.
            step: Step number.

        Returns:
            losses: Dictionary of losses.
        """
        # Stage 1: Standard training (monolingual)
        step1_losses = self._training_step_1(batch)
        
        # Stage 2: Simulated cross conversion
        step2_losses = self._training_step_2(batch)
        
        # Stage 3: Cycle consistency
        step3_losses = self._training_step_3(batch)
        
        # Combine all losses
        total_losses = {}
        for key in step1_losses.keys():
            total_losses[key] = (
                step1_losses[key] + 
                step2_losses[key] + 
                step3_losses[key]
            ) / 3
        
        return total_losses
    
    def _training_step_1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 1: Standard training (monolingual)."""
        # Use the same speaker for content and timbre
        source_audio = batch['audio']
        target_audio = batch['audio']  # Same audio
        
        # Forward pass
        outputs = self.forward(source_audio, target_audio)
        
        # Targets for reconstruction
        targets = {
            'target_mel': self._audio_to_mel(target_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Compute losses
        losses = self.compute_losses(outputs, targets, is_real=True)
        
        return losses
    
    def _training_step_2(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 2: Simulated cross conversion."""
        # Use different speakers
        source_audio = batch['audio']
        # Simulate audio from a different speaker
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)  # Simple simulation
        
        # Forward pass
        outputs = self.forward(source_audio, target_timbre_audio)
        
        # Targets for conversion
        targets = {
            'target_mel': self._audio_to_mel(target_timbre_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Compute losses
        losses = self.compute_losses(outputs, targets, is_real=False)
        
        return losses
    
    def _training_step_3(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 3: Cycle consistency."""
        # Use the audio generated in stage 2
        source_audio = batch['audio']
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)
        
        # Forward pass
        outputs = self.forward(source_audio, target_timbre_audio)
        
        # Cycle reconstruction
        reconstructed_outputs = self.forward(
            outputs['generated_mel'],  # Use the generated mel
            source_audio  # Original timbre
        )
        
        # Targets for cycle consistency
        targets = {
            'target_mel': self._audio_to_mel(source_audio),
            'target_timbre': outputs['timbre_features']
        }
        
        # Compute losses
        losses = self.compute_losses(reconstructed_outputs, targets, is_real=True)
        
        return losses
    
    def inference(
        self,
        source_audio: torch.Tensor,
        target_speaker_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference for voice conversion.

        Args:
            source_audio: Source audio.
            target_speaker_audio: Target speaker audio.

        Returns:
            generated_mel: Generated mel spectrogram.
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(source_audio, target_speaker_audio)
            generated_mel = outputs['generated_mel']
        
        return generated_mel
    
    def save_checkpoint(self, path: str, epoch: int, step: int):
        """Saves the model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Loads a model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['step']


def create_mullivc_model(config_path: str) -> MulliVC:
    """Creates a MulliVC model from a configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = MulliVC(config)
    return model
