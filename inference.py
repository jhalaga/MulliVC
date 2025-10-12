"""
Script d'inférence pour MulliVC
"""
import torch
import torchaudio
import numpy as np
import argparse
import os
from typing import Optional, Tuple
import yaml

from models.mullivc import MulliVC, create_mullivc_model
from utils.audio_utils import AudioProcessor
from utils.data_utils import load_config


class MulliVCInference:
    """Pipeline d'inférence pour MulliVC"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le modèle
        self.model = create_mullivc_model(config_path).to(self.device)
        self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Charger le vocoder HiFi-GAN (placeholder)
        self.vocoder = self._load_vocoder()
    
    def _load_vocoder(self):
        """Charge le vocoder HiFi-GAN"""
        # Placeholder - à implémenter avec le vrai HiFi-GAN
        print("Attention: Vocoder HiFi-GAN non implémenté, utilisation d'un placeholder")
        return None
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Charge un fichier audio"""
        audio, sr = torchaudio.load(audio_path)
        
        # Resample si nécessaire
        if sr != self.config['data']['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, self.config['data']['sample_rate'])
            audio = resampler(audio)
        
        # Convertir en mono si stéréo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio.squeeze(0)  # (samples,)
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Préprocesse l'audio"""
        # Normaliser
        audio = audio / (torch.abs(audio).max() + 1e-8)
        
        # Trimmer le silence
        audio = self.audio_processor._trim_silence(audio)
        
        return audio
    
    def convert_voice(
        self,
        source_audio_path: str,
        target_speaker_audio_path: str,
        output_path: str,
        target_speaker_id: Optional[str] = None
    ) -> str:
        """
        Convertit la voix d'un audio source vers le timbre d'un locuteur cible
        
        Args:
            source_audio_path: Chemin vers l'audio source
            target_speaker_audio_path: Chemin vers l'audio du locuteur cible
            output_path: Chemin de sortie pour l'audio converti
            target_speaker_id: ID du locuteur cible (optionnel)
            
        Returns:
            output_path: Chemin vers l'audio converti
        """
        # Charger les audios
        source_audio = self.load_audio(source_audio_path)
        target_speaker_audio = self.load_audio(target_speaker_audio_path)
        
        # Préprocesser
        source_audio = self.preprocess_audio(source_audio)
        target_speaker_audio = self.preprocess_audio(target_speaker_audio)
        
        # Ajouter dimension batch
        source_audio = source_audio.unsqueeze(0)  # (1, samples)
        target_speaker_audio = target_speaker_audio.unsqueeze(0)  # (1, samples)
        
        # Conversion
        with torch.no_grad():
            generated_mel = self.model.inference(source_audio, target_speaker_audio)
        
        # Convertir le mél-spectrogramme en audio
        generated_audio = self._mel_to_audio(generated_mel)
        
        # Sauvegarder
        self._save_audio(generated_audio, output_path)
        
        return output_path
    
    def _mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convertit le mél-spectrogramme en audio"""
        if self.vocoder is not None:
            # Utiliser le vrai vocoder HiFi-GAN
            with torch.no_grad():
                audio = self.vocoder(mel_spec)
        else:
            # Utiliser l'approximation simple
            audio = self.audio_processor.mel_to_audio(mel_spec.squeeze(0))
            audio = audio.unsqueeze(0)  # (1, samples)
        
        return audio
    
    def _save_audio(self, audio: torch.Tensor, output_path: str):
        """Sauvegarde l'audio"""
        # Créer le dossier de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sauvegarder
        torchaudio.save(
            output_path,
            audio.cpu(),
            self.config['data']['sample_rate']
        )
    
    def batch_convert(
        self,
        source_audio_paths: list,
        target_speaker_audio_path: str,
        output_dir: str
    ) -> list:
        """
        Convertit plusieurs audios en batch
        
        Args:
            source_audio_paths: Liste des chemins vers les audios sources
            target_speaker_audio_path: Chemin vers l'audio du locuteur cible
            output_dir: Dossier de sortie
            
        Returns:
            output_paths: Liste des chemins vers les audios convertis
        """
        output_paths = []
        
        for i, source_path in enumerate(source_audio_paths):
            # Nom de fichier de sortie
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            output_path = os.path.join(output_dir, f"{source_name}_converted.wav")
            
            # Conversion
            try:
                self.convert_voice(source_path, target_speaker_audio_path, output_path)
                output_paths.append(output_path)
                print(f"Converti: {source_path} -> {output_path}")
            except Exception as e:
                print(f"Erreur lors de la conversion de {source_path}: {e}")
        
        return output_paths
    
    def cross_lingual_convert(
        self,
        source_audio_path: str,
        target_speaker_audio_path: str,
        output_path: str,
        source_language: str = "en",
        target_language: str = "fongbe"
    ) -> str:
        """
        Conversion cross-linguale
        
        Args:
            source_audio_path: Chemin vers l'audio source
            target_speaker_audio_path: Chemin vers l'audio du locuteur cible
            output_path: Chemin de sortie
            source_language: Langue source
            target_language: Langue cible
            
        Returns:
            output_path: Chemin vers l'audio converti
        """
        print(f"Conversion cross-linguale: {source_language} -> {target_language}")
        
        # Utiliser la conversion standard
        return self.convert_voice(
            source_audio_path,
            target_speaker_audio_path,
            output_path
        )
    
    def evaluate_conversion(
        self,
        source_audio_path: str,
        target_speaker_audio_path: str,
        converted_audio_path: str
    ) -> dict:
        """
        Évalue la qualité de la conversion
        
        Args:
            source_audio_path: Chemin vers l'audio source
            target_speaker_audio_path: Chemin vers l'audio du locuteur cible
            converted_audio_path: Chemin vers l'audio converti
            
        Returns:
            metrics: Dictionnaire des métriques d'évaluation
        """
        # Charger les audios
        source_audio = self.load_audio(source_audio_path)
        target_audio = self.load_audio(target_speaker_audio_path)
        converted_audio = self.load_audio(converted_audio_path)
        
        # Calculer les métriques
        metrics = {}
        
        # Similarité du locuteur (placeholder)
        metrics['speaker_similarity'] = self._compute_speaker_similarity(
            converted_audio, target_audio
        )
        
        # Préservation du contenu (placeholder)
        metrics['content_preservation'] = self._compute_content_preservation(
            source_audio, converted_audio
        )
        
        # Qualité audio (placeholder)
        metrics['audio_quality'] = self._compute_audio_quality(converted_audio)
        
        return metrics
    
    def _compute_speaker_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> float:
        """Calcule la similarité du locuteur entre deux audios"""
        # Placeholder - à implémenter avec un modèle de vérification vocale
        return 0.8  # Valeur simulée
    
    def _compute_content_preservation(self, source: torch.Tensor, converted: torch.Tensor) -> float:
        """Calcule la préservation du contenu"""
        # Placeholder - à implémenter avec un modèle ASR
        return 0.9  # Valeur simulée
    
    def _compute_audio_quality(self, audio: torch.Tensor) -> float:
        """Calcule la qualité audio"""
        # Placeholder - à implémenter avec des métriques objectives
        return 0.85  # Valeur simulée


def main():
    parser = argparse.ArgumentParser(description='Inférence MulliVC')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--source_audio', type=str, required=True,
                       help='Chemin vers l\'audio source')
    parser.add_argument('--target_speaker_audio', type=str, required=True,
                       help='Chemin vers l\'audio du locuteur cible')
    parser.add_argument('--output', type=str, required=True,
                       help='Chemin de sortie pour l\'audio converti')
    parser.add_argument('--source_language', type=str, default='en',
                       help='Langue source')
    parser.add_argument('--target_language', type=str, default='fongbe',
                       help='Langue cible')
    parser.add_argument('--evaluate', action='store_true',
                       help='Évaluer la qualité de la conversion')
    
    args = parser.parse_args()
    
    # Créer l'inférence
    inference = MulliVCInference(args.config, args.checkpoint)
    
    # Conversion
    if args.source_language != args.target_language:
        output_path = inference.cross_lingual_convert(
            args.source_audio,
            args.target_speaker_audio,
            args.output,
            args.source_language,
            args.target_language
        )
    else:
        output_path = inference.convert_voice(
            args.source_audio,
            args.target_speaker_audio,
            args.output
        )
    
    print(f"Conversion terminée: {output_path}")
    
    # Évaluation si demandée
    if args.evaluate:
        metrics = inference.evaluate_conversion(
            args.source_audio,
            args.target_speaker_audio,
            output_path
        )
        
        print("Métriques d'évaluation:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


if __name__ == '__main__':
    main()
