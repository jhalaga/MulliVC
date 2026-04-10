"""
Evaluation script for MulliVC.
"""
import torch
import argparse
import os
import json
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from models.mullivc import MulliVC, create_mullivc_model
from utils.data_utils import create_dataloader, load_config
from utils.audio_utils import AudioProcessor
from utils.model_utils import get_runtime_device
from evaluation.metrics import ComprehensiveEvaluator


class MulliVCEvaluator:
    """Evaluator for MulliVC."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = get_runtime_device()
        
        # Load the model
        self.model = create_mullivc_model(config_path).to(self.device)
        self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Evaluator
        self.evaluator = ComprehensiveEvaluator(self.config)
    
    def evaluate_dataset(
        self,
        dataloader,
        output_dir: str,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluates the model on a dataset.

        Args:
            dataloader: Dataset dataloader.
            output_dir: Output directory for results.
            num_samples: Number of samples to evaluate.

        Returns:
            metrics: Evaluation metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        source_audios = []
        target_audios = []
        converted_audios = []
        reference_texts = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
                if i >= num_samples:
                    break
                
                # Move to the device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Conversion
                source_audio = batch['audio']
                target_audio = batch['audio']  # Same audio for evaluation
                
                # Generate converted audio
                converted_mel = self.model.inference(source_audio, target_audio)
                converted_audio = self.audio_processor.mel_to_audio(converted_mel.squeeze(0))
                
                # Store for evaluation
                source_audios.append(source_audio.squeeze(0).cpu())
                target_audios.append(target_audio.squeeze(0).cpu())
                converted_audios.append(converted_audio.cpu())
                reference_texts.append(batch['text'][0] if 'text' in batch else None)
                
                # Save converted audio
                output_path = os.path.join(output_dir, f"converted_{i:04d}.wav")
                self.audio_processor.save_audio(output_path, converted_audio)
        
            # Evaluate the batch
        metrics = self.evaluator.evaluate_batch(
            source_audios,
            target_audios,
            converted_audios,
            reference_texts
        )
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def evaluate_cross_lingual(
        self,
        source_dataloader,
        target_dataloader,
        output_dir: str,
        num_samples: int = 50
    ) -> Dict[str, float]:
        """
        Evaluates cross-lingual conversion.

        Args:
            source_dataloader: Dataloader for the source language.
            target_dataloader: Dataloader for the target language.
            output_dir: Output directory.
            num_samples: Number of samples.

        Returns:
            metrics: Cross-lingual metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Cross-lingual evaluation"):
                # Get a source sample
                source_batch = next(iter(source_dataloader))
                source_audio = source_batch['audio'].to(self.device)
                source_text = source_batch['text'][0] if 'text' in source_batch else None
                
                # Get a target sample
                target_batch = next(iter(target_dataloader))
                target_audio = target_batch['audio'].to(self.device)
                
                # Cross-lingual conversion
                converted_mel = self.model.inference(source_audio, target_audio)
                converted_audio = self.audio_processor.mel_to_audio(converted_mel.squeeze(0))
                
                # Evaluate
                metrics = self.evaluator.evaluate_conversion(
                    source_audio.squeeze(0).cpu(),
                    target_audio.squeeze(0).cpu(),
                    converted_audio.cpu(),
                    source_text
                )
                
                all_metrics.append(metrics)
                
                # Save
                output_path = os.path.join(output_dir, f"cross_lingual_{i:04d}.wav")
                self.audio_processor.save_audio(output_path, converted_audio)
        
            # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Save
        metrics_path = os.path.join(output_dir, "cross_lingual_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        return avg_metrics
    
    def evaluate_speaker_verification(
        self,
        genuine_pairs: List[tuple],
        impostor_pairs: List[tuple]
    ) -> Dict[str, float]:
        """
        Evaluates speaker verification.

        Args:
            genuine_pairs: Genuine speaker pairs.
            impostor_pairs: Impostor pairs.

        Returns:
            metrics: Verification metrics.
        """
        return self.evaluator.speaker_metric.compute_speaker_verification_accuracy(
            genuine_pairs, impostor_pairs
        )
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        output_path: str
    ):
        """
        Generates an evaluation report.

        Args:
            metrics: Evaluation metrics.
            output_path: Path to the report.
        """
        report = f"""
    # MulliVC Evaluation Report

    ## Quality Metrics

    ### Speaker Similarity
    - Similarity score: {metrics.get('speaker_similarity', 0):.3f}

    ### Content Preservation
- Word Error Rate (WER): {metrics.get('wer', 0):.3f}
- Character Error Rate (CER): {metrics.get('cer', 0):.3f}

    ### Audio Quality
    - Spectral centroid: {metrics.get('spectral_centroid', 0):.3f}
- Spectral rolloff: {metrics.get('spectral_rolloff', 0):.3f}
    - Zero-crossing rate: {metrics.get('zero_crossing_rate', 0):.3f}
    - MFCC similarity: {metrics.get('mfcc_similarity', 0):.3f}

    ## Interpretation

    - **Speaker similarity** (>0.8): Excellent timbre conversion
    - **WER** (<0.1): Excellent content preservation
    - **CER** (<0.05): Excellent content preservation
    - **Audio quality**: Objective signal-quality metrics

    ## Recommendations

    - If speaker similarity is low, adjust the timbre encoder settings
    - If WER/CER is high, improve content preservation
    - If audio quality is low, optimize the vocoder
"""
        
        with open(output_path, 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description='MulliVC evaluation')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--cross_lingual', action='store_true',
                       help='Evaluate cross-lingual conversion')
    
    args = parser.parse_args()
    
    # Create the evaluator
    evaluator = MulliVCEvaluator(args.config, args.checkpoint)
    
    # Create dataloaders
    train_dataloader = create_dataloader(evaluator.config, split='train')
    val_dataloader = create_dataloader(evaluator.config, split='validation')
    
    # Standard evaluation
    print("Running standard evaluation...")
    metrics = evaluator.evaluate_dataset(
        val_dataloader,
        os.path.join(args.output_dir, 'standard'),
        args.num_samples
    )
    
    # Cross-lingual evaluation if requested
    if args.cross_lingual:
        print("Running cross-lingual evaluation...")
        cross_lingual_metrics = evaluator.evaluate_cross_lingual(
            train_dataloader,
            val_dataloader,
            os.path.join(args.output_dir, 'cross_lingual'),
            args.num_samples // 2
        )
        
        # Combine metrics
        metrics.update(cross_lingual_metrics)
    
    # Generate the report
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    evaluator.generate_evaluation_report(metrics, report_path)
    
    print(f"Evaluation completed. Results saved in {args.output_dir}")
    print(f"Report generated: {report_path}")


if __name__ == '__main__':
    main()
