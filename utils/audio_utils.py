"""
Utilities for audio processing.
"""
import os
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import soundfile as sf


class SpeechBrainHiFiGANVocoder:
    """Lazy wrapper around a pretrained SpeechBrain HiFi-GAN vocoder."""

    def __init__(self, source: str, savedir: str, sample_rate: int):
        self.source = source
        self.savedir = savedir
        self.sample_rate = sample_rate
        self._vocoder = None
        self._device = None
        self._generator_ready = False

    def _load(self, device: torch.device):
        from speechbrain.inference.vocoders import HIFIGAN

        if self._vocoder is None:
            run_opts = {'device': str(device)}
            self._vocoder = HIFIGAN.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts=run_opts
            )
            self._device = device
        elif self._device != device:
            self._vocoder = self._vocoder.to(device)
            self._device = device

        if not self._generator_ready:
            generator = self._vocoder.hparams.generator
            generator.remove_weight_norm()
            generator.eval()
            for param in generator.parameters():
                param.requires_grad = False
            self._vocoder.first_call = False
            self._generator_ready = True

        return self._vocoder

    def mel_to_audio(
        self,
        mel_spec: torch.Tensor,
        hop_length: int,
        allow_grad: bool = False,
        mel_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Converts a batch of mel spectrograms into waveform audio."""
        input_was_single = mel_spec.dim() == 2
        if input_was_single:
            mel_spec = mel_spec.unsqueeze(0)

        if mel_spec.dim() != 3:
            raise ValueError(f"Unsupported mel format: {mel_spec.shape}")

        device = mel_spec.device
        vocoder = self._load(device)
        mel_spec = mel_spec.to(device)
        mel_lens = mel_lens.to(device) if mel_lens is not None else None

        if allow_grad:
            generator = vocoder.hparams.generator
            padding = getattr(generator, 'inference_padding', 0)
            if padding > 0:
                mel_spec = F.pad(mel_spec, (padding, padding), mode='replicate')
            waveform = generator(mel_spec)
            if mel_lens is not None:
                waveform = vocoder.mask_noise(waveform, mel_lens, hop_length)
        else:
            waveform = vocoder.decode_batch(
                mel_spec,
                mel_lens=mel_lens,
                hop_len=hop_length
            )

        waveform = waveform.squeeze(1)
        if input_was_single:
            return waveform.squeeze(0)
        return waveform


class MelSpectrogram:
    """Class for computing mel spectrograms."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mel_channels: int = 80,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax or sample_rate // 2
        
        # Create the mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            f_min=mel_fmin,
            f_max=self.mel_fmax,
            n_mels=n_mel_channels,
            power=1.0
        )
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Computes the mel spectrogram of an audio signal.

        Args:
            audio: Tensor of shape (batch_size, samples) or (samples,).

        Returns:
            mel_spec: Tensor of shape (batch_size, n_mel_channels, time_frames).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        self.mel_transform = self.mel_transform.to(audio.device)
        
        # Compute the mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec


class AudioProcessor:
    """Audio processor for MulliVC."""
    
    def __init__(self, config: dict):
        self.sample_rate = config['data']['sample_rate']
        self.hop_length = config['data']['hop_length']
        self.win_length = config['data']['win_length']
        self.n_mel_channels = config['data']['n_mel_channels']
        self.mel_fmin = config['data']['mel_fmin']
        self.mel_fmax = config['data']['mel_fmax']
        self.vocoder_config = config.get('vocoder', {})
        self.griffin_lim_fallback = self.vocoder_config.get('fallback_to_griffin_lim', False)
        
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mel_channels=self.n_mel_channels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax
        )

        self.vocoder = None
        backend = self.vocoder_config.get('backend', 'speechbrain_hifigan')
        if backend == 'speechbrain_hifigan':
            source = self.vocoder_config.get(
                'source',
                'speechbrain/tts-hifigan-libritts-22050Hz'
            )
            savedir = self.vocoder_config.get(
                'savedir',
                os.path.join(
                    self.vocoder_config.get('cache_dir', 'pretrained_models'),
                    source.split('/')[-1]
                )
            )
            self.vocoder = SpeechBrainHiFiGANVocoder(source, savedir, self.sample_rate)
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Loads an audio file."""
        audio, sr = sf.read(file_path, always_2d=False)
        audio = torch.from_numpy(audio).float()

        if audio.dim() > 1:
            audio = audio.transpose(0, 1)
        else:
            audio = audio.unsqueeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio.squeeze(0)  # Return (samples,)

    def save_audio(self, file_path: str, audio: torch.Tensor):
        """Saves an audio tensor without requiring torchcodec."""
        audio = audio.detach().cpu()

        if audio.dim() == 1:
            audio_np = audio.numpy()
        elif audio.dim() == 2:
            audio_np = audio.transpose(0, 1).numpy()
        else:
            raise ValueError(f"Unsupported audio format for saving: {audio.shape}")

        sf.write(file_path, audio_np, self.sample_rate)
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocesses audio by normalizing and trimming it."""
        # Normalize
        audio = audio / (torch.abs(audio).max() + 1e-8)
        
        # Trim silence
        audio = self._trim_silence(audio)
        
        return audio
    
    def _trim_silence(self, audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Removes silence at the beginning and end."""
        # Find non-silent indices
        non_silent = torch.abs(audio) > threshold
        
        if non_silent.any():
            start_idx = non_silent.nonzero()[0].item()
            end_idx = non_silent.nonzero()[-1].item() + 1
            return audio[start_idx:end_idx]
        
        return audio
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Converts audio to a mel spectrogram."""
        return self.mel_transform(audio)

    def _griffin_lim_mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Fallback waveform reconstruction using inverse mel + Griffin-Lim."""
        mel_spec = torch.exp(mel_spec)
        
        # Inverse mel transform
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=(self.win_length // 2 + 1),
            n_mels=self.n_mel_channels,
            sample_rate=self.sample_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax
        ).to(mel_spec.device)
        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length
        ).to(mel_spec.device)
        
        stft = inverse_mel(mel_spec)
        audio = griffin_lim(stft)
        
        return audio

    def mel_to_audio(
        self,
        mel_spec: torch.Tensor,
        allow_grad: bool = False,
        mel_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Converts mel spectrograms to waveform audio using HiFi-GAN by default."""
        if self.vocoder is not None:
            return self.vocoder.mel_to_audio(
                mel_spec,
                hop_length=self.hop_length,
                allow_grad=allow_grad,
                mel_lens=mel_lens
            )

        if self.griffin_lim_fallback:
            input_was_single = mel_spec.dim() == 2
            if input_was_single:
                mel_spec = mel_spec.unsqueeze(0)

            audio = self._griffin_lim_mel_to_audio(mel_spec)
            if input_was_single:
                return audio.squeeze(0)
            return audio

        raise RuntimeError(
            'No HiFi-GAN vocoder is configured. '
            'Set vocoder.backend=speechbrain_hifigan or enable fallback_to_griffin_lim.'
        )


def extract_pitch(audio: torch.Tensor, sample_rate: int = 22050) -> torch.Tensor:
    """
    Extracts pitch (F0) from an audio signal.

    Args:
        audio: Audio signal of shape (samples,).
        sample_rate: Sampling rate.

    Returns:
        pitch: Tensor of shape (time_frames,).
    """
    # Convert to numpy for librosa
    audio_np = audio.numpy()
    
    # Extract F0 with librosa
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_np,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=256
    )
    
    # Replace NaN values with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return torch.from_numpy(f0).float()


def compute_spectral_centroid(audio: torch.Tensor, sample_rate: int = 22050) -> torch.Tensor:
    """Computes the spectral centroid."""
    audio_np = audio.numpy()
    centroid = librosa.feature.spectral_centroid(
        y=audio_np, sr=sample_rate, hop_length=256
    )[0]
    return torch.from_numpy(centroid).float()
