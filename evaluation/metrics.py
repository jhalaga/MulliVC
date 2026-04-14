"""
Evaluation metrics for MulliVC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference import EncoderClassifier


class SpeakerSimilarityMetric:
    """Speaker similarity metric."""
    
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
        Computes speaker similarity between two audio samples.

        Args:
            audio1: First audio sample of shape (samples,).
            audio2: Second audio sample of shape (samples,).
            sample_rate: Sampling rate.

        Returns:
            similarity: Similarity score in the range [0, 1].
        """
        # Resample if needed
        if sample_rate != 16000:
            audio1 = self._resample_audio(audio1, sample_rate, 16000)
            audio2 = self._resample_audio(audio2, sample_rate, 16000)
        
        # Extract embeddings
        with torch.no_grad():
            embedding1 = self.classifier.encode_batch(audio1.unsqueeze(0))
            embedding2 = self.classifier.encode_batch(audio2.unsqueeze(0))
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1).item()
        
        return similarity
    
    def _resample_audio(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resamples audio."""
        if orig_sr == target_sr:
            return audio
        
        # Use librosa for resampling
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
        Computes speaker verification accuracy.

        Args:
            genuine_pairs: Pairs from the same speaker.
            impostor_pairs: Pairs from different speakers.
            threshold: Decision threshold.

        Returns:
            metrics: Dictionary of verification metrics.
        """
        # True positives
        tp = 0
        for audio1, audio2 in genuine_pairs:
            similarity = self.compute_similarity(audio1, audio2)
            if similarity >= threshold:
                tp += 1
        
        # False negatives
        fn = len(genuine_pairs) - tp
        
        # True negatives
        tn = 0
        for audio1, audio2 in impostor_pairs:
            similarity = self.compute_similarity(audio1, audio2)
            if similarity < threshold:
                tn += 1
        
        # False positives
        fp = len(impostor_pairs) - tn
        
        # Compute metrics
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
        """Computes the Equal Error Rate."""
        # Compute scores for genuine speakers
        genuine_scores = []
        for audio1, audio2 in genuine_pairs:
            score = self.compute_similarity(audio1, audio2)
            genuine_scores.append(score)
        
        # Compute scores for impostors
        impostor_scores = []
        for audio1, audio2 in impostor_pairs:
            score = self.compute_similarity(audio1, audio2)
            impostor_scores.append(score)
        
        # Find the EER threshold
        all_scores = genuine_scores + impostor_scores
        eer = 0.5  # Placeholder - to be implemented properly
        
        return eer


class ASRMetric:
    """ASR metric for evaluating content preservation."""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for ASR metrics. Install the local Python 3.13 requirements to enable Whisper-based evaluation."
            ) from exc

        self.model = whisper.load_model("base")
    
    def compute_wer(
        self, 
        reference_text: str, 
        predicted_text: str
    ) -> float:
        """
        Computes the Word Error Rate.

        Args:
            reference_text: Reference text.
            predicted_text: Predicted text.

        Returns:
            wer: Word Error Rate.
        """
        # Tokenize texts
        ref_words = reference_text.lower().split()
        pred_words = predicted_text.lower().split()
        
        # Compute edit distance
        distance = self._levenshtein_distance(ref_words, pred_words)
        
        # WER = distance / number of reference words
        wer = distance / len(ref_words) if len(ref_words) > 0 else 0
        
        return wer
    
    def compute_cer(
        self, 
        reference_text: str, 
        predicted_text: str
    ) -> float:
        """
        Computes the Character Error Rate.

        Args:
            reference_text: Reference text.
            predicted_text: Predicted text.

        Returns:
            cer: Character Error Rate.
        """
        # Compute edit distance at the character level
        distance = self._levenshtein_distance(
            list(reference_text.lower()),
            list(predicted_text.lower())
        )
        
        # CER = distance / number of reference characters
        cer = distance / len(reference_text) if len(reference_text) > 0 else 0
        
        return cer
    
    def transcribe_audio(self, audio: torch.Tensor, sample_rate: int = 16000) -> str:
        """
        Transcribes audio into text.

        Args:
            audio: Audio to transcribe.
            sample_rate: Sampling rate.

        Returns:
            text: Transcribed text.
        """
        # Convert to numpy
        audio_np = audio.numpy()
        
        # Transcribe with Whisper
        result = self.model.transcribe(audio_np)
        
        return result["text"]
    
    def _levenshtein_distance(self, s1: List, s2: List) -> int:
        """Computes Levenshtein distance."""
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
    """Audio quality metrics."""
    
    def __init__(self):
        pass
    
    def compute_snr(self, signal: torch.Tensor, noise: torch.Tensor) -> float:
        """
        Computes the Signal-to-Noise Ratio.

        Args:
            signal: Signal.
            noise: Noise.

        Returns:
            snr: SNR in dB.
        """
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    def compute_spectral_centroid(self, audio: torch.Tensor, sample_rate: int = 22050) -> float:
        """
        Computes the spectral centroid.

        Args:
            audio: Audio.
            sample_rate: Sampling rate.

        Returns:
            centroid: Spectral centroid.
        """
        audio_np = audio.numpy()
        centroid = librosa.feature.spectral_centroid(
            y=audio_np, sr=sample_rate
        )[0]
        
        return np.mean(centroid)
    
    def compute_spectral_rolloff(self, audio: torch.Tensor, sample_rate: int = 22050) -> float:
        """
        Computes spectral rolloff.

        Args:
            audio: Audio.
            sample_rate: Sampling rate.

        Returns:
            rolloff: Spectral rolloff.
        """
        audio_np = audio.numpy()
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_np, sr=sample_rate
        )[0]
        
        return np.mean(rolloff)
    
    def compute_zero_crossing_rate(self, audio: torch.Tensor) -> float:
        """
        Computes the zero-crossing rate.

        Args:
            audio: Audio.

        Returns:
            zcr: Zero-crossing rate.
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
        Computes MFCC similarity between two audio samples.

        Args:
            audio1: First audio sample.
            audio2: Second audio sample.
            sample_rate: Sampling rate.

        Returns:
            similarity: MFCC similarity.
        """
        # Extract MFCCs
        mfcc1 = librosa.feature.mfcc(
            y=audio1.numpy(), sr=sample_rate, n_mfcc=13
        )
        mfcc2 = librosa.feature.mfcc(
            y=audio2.numpy(), sr=sample_rate, n_mfcc=13
        )
        
        # Compute cosine similarity
        mfcc1_flat = mfcc1.flatten()
        mfcc2_flat = mfcc2.flatten()
        
        similarity = 1 - cosine(mfcc1_flat, mfcc2_flat)
        
        return similarity


class ComprehensiveEvaluator:
    """Comprehensive evaluator for MulliVC."""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize metrics
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
        Fully evaluates a conversion.

        Args:
            source_audio: Source audio.
            target_speaker_audio: Target speaker audio.
            converted_audio: Converted audio.
            reference_text: Reference text, if available.

        Returns:
            metrics: Dictionary of metrics.
        """
        metrics = {}
        
        # Speaker similarity
        metrics['speaker_similarity'] = self.speaker_metric.compute_similarity(
            converted_audio, target_speaker_audio
        )
        
        # Content preservation
        if reference_text is not None:
            # Transcribe the converted audio
            predicted_text = self.asr_metric.transcribe_audio(converted_audio)
            
            # Compute WER and CER
            metrics['wer'] = self.asr_metric.compute_wer(reference_text, predicted_text)
            metrics['cer'] = self.asr_metric.compute_cer(reference_text, predicted_text)
        
        # Audio quality
        metrics['spectral_centroid'] = self.quality_metric.compute_spectral_centroid(converted_audio)
        metrics['spectral_rolloff'] = self.quality_metric.compute_spectral_rolloff(converted_audio)
        metrics['zero_crossing_rate'] = self.quality_metric.compute_zero_crossing_rate(converted_audio)
        
        # MFCC similarity with the original
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
        Evaluates a batch of conversions.

        Args:
            source_audios: List of source audios.
            target_speaker_audios: List of target speaker audios.
            converted_audios: List of converted audios.
            reference_texts: List of reference texts, if available.

        Returns:
            metrics: Average metrics.
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
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
