"""
Métriques d'évaluation pour MulliVC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
from scipy.spatial.distance import cosine
import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier


class SpeakerSimilarityMetric:
    """Métrique de similarité du locuteur"""
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
    
    def compute_similarity(
        self, 
        audio1: torch.Tensor, 
        audio2: torch.Tensor,
        sample_rate: int = 16000
    ) -> float:
        """
        Calcule la similarité du locuteur entre deux audios
        
        Args:
            audio1: Premier audio (samples,)
            audio2: Deuxième audio (samples,)
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            similarity: Score de similarité (0-1)
        """
        # Resample si nécessaire
        if sample_rate != 16000:
            audio1 = self._resample_audio(audio1, sample_rate, 16000)
            audio2 = self._resample_audio(audio2, sample_rate, 16000)
        
        # Extraire les embeddings
        with torch.no_grad():
            embedding1 = self.classifier.encode_batch(audio1.unsqueeze(0))
            embedding2 = self.classifier.encode_batch(audio2.unsqueeze(0))
        
        # Calculer la similarité cosinus
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1).item()
        
        return similarity
    
    def _resample_audio(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample l'audio"""
        if orig_sr == target_sr:
            return audio
        
        # Utiliser librosa pour le resampling
        audio_np = audio.numpy()
        resampled = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
        
        return torch.from_numpy(resampled).float()
    
    def compute_speaker_verification_accuracy(
        self,
        genuine_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        impostor_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calcule l'accuracy de vérification du locuteur
        
        Args:
            genuine_pairs: Paires de vrais locuteurs
            impostor_pairs: Paires d'imposteurs
            threshold: Seuil de décision
            
        Returns:
            metrics: Dictionnaire des métriques
        """
        # Vrais positifs
        tp = 0
        for audio1, audio2 in genuine_pairs:
            similarity = self.compute_similarity(audio1, audio2)
            if similarity >= threshold:
                tp += 1
        
        # Faux négatifs
        fn = len(genuine_pairs) - tp
        
        # Vrais négatifs
        tn = 0
        for audio1, audio2 in impostor_pairs:
            similarity = self.compute_similarity(audio1, audio2)
            if similarity < threshold:
                tn += 1
        
        # Faux positifs
        fp = len(impostor_pairs) - tn
        
        # Calculer les métriques
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'eer': self._compute_eer(genuine_pairs, impostor_pairs)
        }
    
    def _compute_eer(self, genuine_pairs: List, impostor_pairs: List) -> float:
        """Calcule l'Equal Error Rate"""
        # Calculer les scores pour les vrais locuteurs
        genuine_scores = []
        for audio1, audio2 in genuine_pairs:
            score = self.compute_similarity(audio1, audio2)
            genuine_scores.append(score)
        
        # Calculer les scores pour les imposteurs
        impostor_scores = []
        for audio1, audio2 in impostor_pairs:
            score = self.compute_similarity(audio1, audio2)
            impostor_scores.append(score)
        
        # Trouver le seuil EER
        all_scores = genuine_scores + impostor_scores
        eer = 0.5  # Placeholder - à implémenter correctement
        
        return eer


class ASRMetric:
    """Métrique ASR pour évaluer la préservation du contenu"""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        self.model = whisper.load_model("base")
    
    def compute_wer(
        self, 
        reference_text: str, 
        predicted_text: str
    ) -> float:
        """
        Calcule le Word Error Rate
        
        Args:
            reference_text: Texte de référence
            predicted_text: Texte prédit
            
        Returns:
            wer: Word Error Rate
        """
        # Tokeniser les textes
        ref_words = reference_text.lower().split()
        pred_words = predicted_text.lower().split()
        
        # Calculer la distance d'édition
        distance = self._levenshtein_distance(ref_words, pred_words)
        
        # WER = distance / nombre de mots de référence
        wer = distance / len(ref_words) if len(ref_words) > 0 else 0
        
        return wer
    
    def compute_cer(
        self, 
        reference_text: str, 
        predicted_text: str
    ) -> float:
        """
        Calcule le Character Error Rate
        
        Args:
            reference_text: Texte de référence
            predicted_text: Texte prédit
            
        Returns:
            cer: Character Error Rate
        """
        # Calculer la distance d'édition au niveau des caractères
        distance = self._levenshtein_distance(
            list(reference_text.lower()),
            list(predicted_text.lower())
        )
        
        # CER = distance / nombre de caractères de référence
        cer = distance / len(reference_text) if len(reference_text) > 0 else 0
        
        return cer
    
    def transcribe_audio(self, audio: torch.Tensor, sample_rate: int = 16000) -> str:
        """
        Transcrit un audio en texte
        
        Args:
            audio: Audio à transcrire
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            text: Texte transcrit
        """
        # Convertir en numpy
        audio_np = audio.numpy()
        
        # Transcrir avec Whisper
        result = self.model.transcribe(audio_np)
        
        return result["text"]
    
    def _levenshtein_distance(self, s1: List, s2: List) -> int:
        """Calcule la distance de Levenshtein"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class AudioQualityMetric:
    """Métriques de qualité audio"""
    
    def __init__(self):
        pass
    
    def compute_snr(self, signal: torch.Tensor, noise: torch.Tensor) -> float:
        """
        Calcule le Signal-to-Noise Ratio
        
        Args:
            signal: Signal
            noise: Bruit
            
        Returns:
            snr: SNR en dB
        """
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    def compute_spectral_centroid(self, audio: torch.Tensor, sample_rate: int = 22050) -> float:
        """
        Calcule le centroïde spectral
        
        Args:
            audio: Audio
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            centroid: Centroïde spectral
        """
        audio_np = audio.numpy()
        centroid = librosa.feature.spectral_centroid(
            y=audio_np, sr=sample_rate
        )[0]
        
        return np.mean(centroid)
    
    def compute_spectral_rolloff(self, audio: torch.Tensor, sample_rate: int = 22050) -> float:
        """
        Calcule le spectral rolloff
        
        Args:
            audio: Audio
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            rolloff: Spectral rolloff
        """
        audio_np = audio.numpy()
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_np, sr=sample_rate
        )[0]
        
        return np.mean(rolloff)
    
    def compute_zero_crossing_rate(self, audio: torch.Tensor) -> float:
        """
        Calcule le taux de passage par zéro
        
        Args:
            audio: Audio
            
        Returns:
            zcr: Taux de passage par zéro
        """
        audio_np = audio.numpy()
        zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
        
        return np.mean(zcr)
    
    def compute_mfcc_similarity(
        self, 
        audio1: torch.Tensor, 
        audio2: torch.Tensor,
        sample_rate: int = 22050
    ) -> float:
        """
        Calcule la similarité MFCC entre deux audios
        
        Args:
            audio1: Premier audio
            audio2: Deuxième audio
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            similarity: Similarité MFCC
        """
        # Extraire les MFCCs
        mfcc1 = librosa.feature.mfcc(
            y=audio1.numpy(), sr=sample_rate, n_mfcc=13
        )
        mfcc2 = librosa.feature.mfcc(
            y=audio2.numpy(), sr=sample_rate, n_mfcc=13
        )
        
        # Calculer la similarité cosinus
        mfcc1_flat = mfcc1.flatten()
        mfcc2_flat = mfcc2.flatten()
        
        similarity = 1 - cosine(mfcc1_flat, mfcc2_flat)
        
        return similarity


class ComprehensiveEvaluator:
    """Évaluateur complet pour MulliVC"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialiser les métriques
        self.speaker_metric = SpeakerSimilarityMetric()
        self.asr_metric = ASRMetric()
        self.quality_metric = AudioQualityMetric()
    
    def evaluate_conversion(
        self,
        source_audio: torch.Tensor,
        target_speaker_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        reference_text: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Évalue complètement une conversion
        
        Args:
            source_audio: Audio source
            target_speaker_audio: Audio du locuteur cible
            converted_audio: Audio converti
            reference_text: Texte de référence (optionnel)
            
        Returns:
            metrics: Dictionnaire des métriques
        """
        metrics = {}
        
        # Similarité du locuteur
        metrics['speaker_similarity'] = self.speaker_metric.compute_similarity(
            converted_audio, target_speaker_audio
        )
        
        # Préservation du contenu
        if reference_text is not None:
            # Transcrire l'audio converti
            predicted_text = self.asr_metric.transcribe_audio(converted_audio)
            
            # Calculer WER et CER
            metrics['wer'] = self.asr_metric.compute_wer(reference_text, predicted_text)
            metrics['cer'] = self.asr_metric.compute_cer(reference_text, predicted_text)
        
        # Qualité audio
        metrics['spectral_centroid'] = self.quality_metric.compute_spectral_centroid(converted_audio)
        metrics['spectral_rolloff'] = self.quality_metric.compute_spectral_rolloff(converted_audio)
        metrics['zero_crossing_rate'] = self.quality_metric.compute_zero_crossing_rate(converted_audio)
        
        # Similarité MFCC avec l'original
        metrics['mfcc_similarity'] = self.quality_metric.compute_mfcc_similarity(
            source_audio, converted_audio
        )
        
        return metrics
    
    def evaluate_batch(
        self,
        source_audios: List[torch.Tensor],
        target_speaker_audios: List[torch.Tensor],
        converted_audios: List[torch.Tensor],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Évalue un batch de conversions
        
        Args:
            source_audios: Liste des audios sources
            target_speaker_audios: Liste des audios des locuteurs cibles
            converted_audios: Liste des audios convertis
            reference_texts: Liste des textes de référence (optionnel)
            
        Returns:
            metrics: Métriques moyennes
        """
        all_metrics = []
        
        for i in range(len(source_audios)):
            ref_text = reference_texts[i] if reference_texts else None
            
            metrics = self.evaluate_conversion(
                source_audios[i],
                target_speaker_audios[i],
                converted_audios[i],
                ref_text
            )
            
            all_metrics.append(metrics)
        
        # Moyenner les métriques
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
