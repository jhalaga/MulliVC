"""
Utilitaires pour le traitement audio
"""
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import soundfile as sf


class MelSpectrogram:
    """Classe pour calculer les mél-spectrogrammes"""
    
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
        
        # Créer le transformateur mel
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
            win_length=win_length,
            mel_fmin=mel_fmin,
            mel_fmax=self.mel_fmax,
            power=1.0
        )
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Calcule le mél-spectrogramme d'un signal audio
        
        Args:
            audio: Tensor de forme (batch_size, samples) ou (samples,)
            
        Returns:
            mel_spec: Tensor de forme (batch_size, n_mel_channels, time_frames)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Calculer le mél-spectrogramme
        mel_spec = self.mel_transform(audio)
        
        # Convertir en log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec


class AudioProcessor:
    """Processeur audio pour MulliVC"""
    
    def __init__(self, config: dict):
        self.sample_rate = config['data']['sample_rate']
        self.hop_length = config['data']['hop_length']
        self.win_length = config['data']['win_length']
        self.n_mel_channels = config['data']['n_mel_channels']
        self.mel_fmin = config['data']['mel_fmin']
        self.mel_fmax = config['data']['mel_fmax']
        
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mel_channels=self.n_mel_channels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax
        )
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Charge un fichier audio"""
        audio, sr = torchaudio.load(file_path)
        
        # Resample si nécessaire
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convertir en mono si stéréo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio.squeeze(0)  # Retourner (samples,)
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Préprocesse l'audio (normalisation, trimming)"""
        # Normaliser
        audio = audio / (torch.abs(audio).max() + 1e-8)
        
        # Trimmer le silence
        audio = self._trim_silence(audio)
        
        return audio
    
    def _trim_silence(self, audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Supprime le silence au début et à la fin"""
        # Trouver les indices non-silencieux
        non_silent = torch.abs(audio) > threshold
        
        if non_silent.any():
            start_idx = non_silent.nonzero()[0].item()
            end_idx = non_silent.nonzero()[-1].item() + 1
            return audio[start_idx:end_idx]
        
        return audio
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convertit l'audio en mél-spectrogramme"""
        return self.mel_transform(audio)
    
    def mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convertit le mél-spectrogramme en audio (approximation)"""
        # Cette fonction est une approximation simple
        # En pratique, on utiliserait un vocoder comme HiFi-GAN
        mel_spec = torch.exp(mel_spec)
        
        # Inverse mel transform
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=(self.win_length // 2 + 1),
            n_mels=self.n_mel_channels,
            sample_rate=self.sample_rate,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax
        )
        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        stft = inverse_mel(mel_spec)
        audio = griffin_lim(stft)
        
        return audio


def extract_pitch(audio: torch.Tensor, sample_rate: int = 22050) -> torch.Tensor:
    """
    Extrait le pitch (F0) d'un signal audio
    
    Args:
        audio: Signal audio de forme (samples,)
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        pitch: Tensor de forme (time_frames,)
    """
    # Convertir en numpy pour librosa
    audio_np = audio.numpy()
    
    # Extraire F0 avec librosa
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_np,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=256
    )
    
    # Remplacer NaN par 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return torch.from_numpy(f0).float()


def compute_spectral_centroid(audio: torch.Tensor, sample_rate: int = 22050) -> torch.Tensor:
    """Calcule le centroïde spectral"""
    audio_np = audio.numpy()
    centroid = librosa.feature.spectral_centroid(
        y=audio_np, sr=sample_rate, hop_length=256
    )[0]
    return torch.from_numpy(centroid).float()
