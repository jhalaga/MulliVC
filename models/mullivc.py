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
from utils.audio_utils import AudioProcessor


class MulliVC(nn.Module):
    """
    Main MulliVC model for multilingual voice conversion.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.enable_cycle_consistency = config.get('training', {}).get(
            'enable_cycle_consistency',
            True
        )
        self.audio_processor = AudioProcessor(config)
        
        # Initialize components
        self.content_encoder = ContentEncoder(
            model_name=config['model']['content_encoder']['model_name'],
            hidden_size=config['model']['content_encoder']['hidden_size'],
            output_dim=config['model']['content_encoder']['output_dim'],
            input_sample_rate=config['data']['sample_rate'],
            target_sample_rate=16000
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
        if source_audio.dim() == 3:
            source_audio = self._mel_to_audio(source_audio, allow_grad=False)

        if target_timbre_audio.dim() == 3:
            target_timbre_audio = self._mel_to_audio(target_timbre_audio, allow_grad=False)

        # Encode source audio content
        content_features, content_pooled = self.content_encoder(source_audio)

        # Upsample content features from WavLM's ~50 fps to the mel-spectrogram
        # frame rate (sample_rate / hop_length). Without this, the generated mel
        # is time-compressed by a factor of (WavLM_fps / mel_fps), producing
        # audio that is roughly 0.58x the source length at sample_rate=22050,
        # hop_length=256.
        hop_length = self.audio_processor.hop_length
        audio_samples = source_audio.shape[-1]
        # torchaudio.MelSpectrogram with center=True (default) produces this many frames:
        expected_mel_len = audio_samples // hop_length + 1
        if content_features.shape[1] != expected_mel_len:
            content_features = torch.nn.functional.interpolate(
                content_features.transpose(1, 2),
                size=expected_mel_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

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
        generated_mel = generated_mel.transpose(1, 2)
        
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
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return self.audio_processor.audio_to_mel(audio)

    def _mel_to_audio(self, mel_spec: torch.Tensor, allow_grad: bool = False) -> torch.Tensor:
        """Converts mel spectrograms into waveform audio with the configured vocoder."""
        return self.audio_processor.mel_to_audio(
            mel_spec,
            allow_grad=allow_grad
        )

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
            target_audio=targets.get('target_audio'),
            content_encoder=self.content_encoder,
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
        stage_losses = [
            self._training_step_1(batch),
            self._training_step_2(batch)
        ]

        if self.enable_cycle_consistency:
            stage_losses.append(self._training_step_3(batch))

        # Combine all enabled stage losses
        total_losses = {}
        for key in stage_losses[0].keys():
            total_losses[key] = sum(losses[key] for losses in stage_losses) / len(stage_losses)
        
        return total_losses
    
    def _training_step_1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 1: Standard training (monolingual)."""
        # Use the same speaker for content and timbre
        source_audio = batch['audio']
        target_audio = batch['audio']  # Same audio

        # Forward pass
        outputs = self.forward(source_audio, target_audio)
        # NOTE: do NOT run HiFi-GAN with grad in stage 1. Since target_audio
        # == source_audio, the mel reconstruction loss already supervises
        # content preservation 1:1; the ASR / content-preservation loss is
        # redundant here but would cost a large HiFi-GAN backward pass.

        # Targets for reconstruction.
        # target_timbre uses the timbre_encoder's own embedding of the source
        # (detached) so predicted global_timbre -- which has been through the
        # conformer -- is pulled back toward the raw speaker embedding rather
        # than being self-referential to its own graph.
        targets = {
            'target_mel': self._audio_to_mel(target_audio),
            'target_timbre': outputs['timbre_features'].detach(),
            # No target_audio: ASR loss is skipped in stage 1.
        }

        # Compute losses
        losses = self.compute_losses(outputs, targets, is_real=True)

        return losses
    
    def _training_step_2(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 2: Simulated cross conversion.

        The generator is given the *source* content but a *different* speaker
        timbre. The training target is the *source* mel (content X, speaker A)
        because we cannot supervise "speaker B saying X" without ground truth.
        This teaches the model to preserve content when timbre input varies,
        while adversarial / timbre losses push the generator's global timbre
        toward the supplied target speaker.
        """
        source_audio = batch['audio']
        # Simulate audio from a different speaker (paired within the batch)
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)

        # Forward pass with source content + target speaker timbre input
        outputs = self.forward(source_audio, target_timbre_audio)
        outputs['generated_audio'] = self._mel_to_audio(
            outputs['generated_mel'],
            allow_grad=True
        )

        # Target: reconstruct SOURCE (content X, speaker A).
        # Target timbre: the target speaker's embedding, so the model learns
        # to consume the provided timbre signal.
        target_timbre_mel = self._audio_to_mel(target_timbre_audio)
        target_speaker_timbre = self.timbre_encoder.extract_timbre_features(target_timbre_mel).detach()
        targets = {
            'target_mel': self._audio_to_mel(source_audio),
            'target_timbre': target_speaker_timbre,
            # Content of generated should still match source content.
            'target_audio': source_audio,
        }

        losses = self.compute_losses(outputs, targets, is_real=True)

        return losses
    
    def _training_step_3(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 3: Cycle consistency (A -> B -> A should recover A)."""
        source_audio = batch['audio']
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)

        # Forward pass 1: source content + target timbre.
        # We need a waveform to feed back into the cycle, but we detach it
        # so gradients do not propagate through the first HiFi-GAN pass.
        # The content-preservation loss is delivered by stage 2 on this
        # same pairing.
        outputs = self.forward(source_audio, target_timbre_audio)
        with torch.no_grad():
            generated_audio = self._mel_to_audio(
                outputs['generated_mel'],
                allow_grad=False
            )

        # Forward pass 2: cycle back using source as the timbre reference.
        # Mel-level reconstruction against source_mel trains the cycle; we do
        # NOT run HiFi-GAN with grad on the cycle output -- that would
        # quadruple per-step cost for marginal benefit.
        reconstructed_outputs = self.forward(
            generated_audio,
            source_audio
        )

        # Cycle target: reconstruct the source; matching timbre is source speaker's
        source_mel = self._audio_to_mel(source_audio)
        source_speaker_timbre = self.timbre_encoder.extract_timbre_features(source_mel).detach()
        targets = {
            'target_mel': source_mel,
            'target_timbre': source_speaker_timbre,
            # No target_audio: ASR loss is skipped in cycle stage.
        }

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
    from utils.data_utils import load_config

    config = load_config(config_path)
    
    model = MulliVC(config)
    return model
