"""
Inference script for MulliVC
"""
import torch
import numpy as np
import argparse
import os
from typing import Optional, Tuple
import yaml

from models.mullivc import MulliVC, create_mullivc_model
from utils.audio_utils import AudioProcessor
from utils.data_utils import load_config
from utils.model_utils import get_runtime_device


class MulliVCInference:
    """Inference pipeline for MulliVC."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = get_runtime_device()
        
        # Load the model
        self.model = create_mullivc_model(config_path).to(self.device)
        self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads an audio file."""
        return self.audio_processor.load_audio(audio_path)
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocesses audio."""
        # Normalize
        audio = audio / (torch.abs(audio).max() + 1e-8)
        
        # Trim silence
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
        Converts a source voice to the timbre of a target speaker.

        Args:
            source_audio_path: Path to the source audio.
            target_speaker_audio_path: Path to the target speaker audio.
            output_path: Output path for the converted audio.
            target_speaker_id: Target speaker ID, if available.

        Returns:
            output_path: Path to the converted audio.
        """
        # Load audio
        source_audio = self.load_audio(source_audio_path)
        target_speaker_audio = self.load_audio(target_speaker_audio_path)
        
        # Preprocess
        source_audio = self.preprocess_audio(source_audio)
        target_speaker_audio = self.preprocess_audio(target_speaker_audio)
        
        # Add batch dimension
        source_audio = source_audio.unsqueeze(0)  # (1, samples)
        target_speaker_audio = target_speaker_audio.unsqueeze(0)  # (1, samples)
        
        # Convert
        with torch.no_grad():
            generated_mel = self.model.inference(source_audio, target_speaker_audio)
        
        # Convert the mel spectrogram to audio
        generated_audio = self._mel_to_audio(generated_mel)
        
        # Save
        self._save_audio(generated_audio, output_path)
        
        return output_path
    
    def _mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Converts a mel spectrogram to audio."""
        return self.audio_processor.mel_to_audio(mel_spec)
    
    def _save_audio(self, audio: torch.Tensor, output_path: str):
        """Saves audio."""
        # Create the output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.audio_processor.save_audio(output_path, audio.cpu())
    
    def batch_convert(
        self,
        source_audio_paths: list,
        target_speaker_audio_path: str,
        output_dir: str
    ) -> list:
        """
        Converts multiple audio files in a batch.

        Args:
            source_audio_paths: List of source audio paths.
            target_speaker_audio_path: Path to the target speaker audio.
            output_dir: Output directory.

        Returns:
            output_paths: List of converted audio paths.
        """
        output_paths = []
        
        for i, source_path in enumerate(source_audio_paths):
            # Output file name
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            output_path = os.path.join(output_dir, f"{source_name}_converted.wav")
            
            # Convert
            try:
                self.convert_voice(source_path, target_speaker_audio_path, output_path)
                output_paths.append(output_path)
                print(f"Converted: {source_path} -> {output_path}")
            except Exception as e:
                print(f"Error converting {source_path}: {e}")
        
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
        Performs cross-lingual conversion.

        Args:
            source_audio_path: Path to the source audio.
            target_speaker_audio_path: Path to the target speaker audio.
            output_path: Output path.
            source_language: Source language.
            target_language: Target language.

        Returns:
            output_path: Path to the converted audio.
        """
        print(f"Cross-lingual conversion: {source_language} -> {target_language}")
        
        # Use standard conversion
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
        Evaluates conversion quality.

        Args:
            source_audio_path: Path to the source audio.
            target_speaker_audio_path: Path to the target speaker audio.
            converted_audio_path: Path to the converted audio.

        Returns:
            metrics: Dictionary of evaluation metrics.
        """
        # Load audio
        source_audio = self.load_audio(source_audio_path)
        target_audio = self.load_audio(target_speaker_audio_path)
        converted_audio = self.load_audio(converted_audio_path)
        
        # Compute metrics
        metrics = {}
        
        # Speaker similarity (placeholder)
        metrics['speaker_similarity'] = self._compute_speaker_similarity(
            converted_audio, target_audio
        )
        
        # Content preservation (placeholder)
        metrics['content_preservation'] = self._compute_content_preservation(
            source_audio, converted_audio
        )
        
        # Audio quality (placeholder)
        metrics['audio_quality'] = self._compute_audio_quality(converted_audio)
        
        return metrics
    
    def _compute_speaker_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> float:
        """Computes speaker similarity between two audio samples."""
        # Placeholder - to be implemented with a speaker verification model
        return 0.8  # Simulated value
    
    def _compute_content_preservation(self, source: torch.Tensor, converted: torch.Tensor) -> float:
        """Computes content preservation."""
        # Placeholder - to be implemented with an ASR model
        return 0.9  # Simulated value
    
    def _compute_audio_quality(self, audio: torch.Tensor) -> float:
        """Computes audio quality."""
        # Placeholder - to be implemented with objective metrics
        return 0.85  # Simulated value


def main():
    parser = argparse.ArgumentParser(description='MulliVC inference')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--source_audio', type=str, required=True,
                       help='Path to the source audio')
    parser.add_argument('--target_speaker_audio', type=str, required=True,
                       help='Path to the target speaker audio')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for the converted audio')
    parser.add_argument('--source_language', type=str, default='en',
                       help='Source language')
    parser.add_argument('--target_language', type=str, default='fongbe',
                       help='Target language')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate conversion quality')
    
    args = parser.parse_args()
    
    # Create the inference pipeline
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
    
    print(f"Conversion completed: {output_path}")
    
    # Evaluate if requested
    if args.evaluate:
        metrics = inference.evaluate_conversion(
            args.source_audio,
            args.target_speaker_audio,
            output_path
        )
        
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


if __name__ == '__main__':
    main()
