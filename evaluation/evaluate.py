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
            for i, batch in enumerate(tqdm(dataloader, desc="Évaluation")):
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
            for i in tqdm(range(num_samples), desc="Évaluation cross-linguale"):
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
# Rapport d'évaluation MulliVC

## Métriques de qualité

### Similarité du locuteur
- Score de similarité: {metrics.get('speaker_similarity', 0):.3f}

### Préservation du contenu
- Word Error Rate (WER): {metrics.get('wer', 0):.3f}
- Character Error Rate (CER): {metrics.get('cer', 0):.3f}

### Qualité audio
- Centroïde spectral: {metrics.get('spectral_centroid', 0):.3f}
- Spectral rolloff: {metrics.get('spectral_rolloff', 0):.3f}
- Taux de passage par zéro: {metrics.get('zero_crossing_rate', 0):.3f}
- Similarité MFCC: {metrics.get('mfcc_similarity', 0):.3f}

## Interprétation

- **Similarité du locuteur** (>0.8): Excellente conversion du timbre
- **WER** (<0.1): Excellente préservation du contenu
- **CER** (<0.05): Excellente préservation du contenu
- **Qualité audio**: Métriques objectives de la qualité du signal

## Recommandations

- Si la similarité du locuteur est faible, ajuster les paramètres du timbre encoder
- Si le WER/CER est élevé, améliorer la préservation du contenu
- Si la qualité audio est faible, optimiser le vocoder
"""
        
        with open(output_path, 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Évaluation MulliVC')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Nombre d\'échantillons à évaluer')
    parser.add_argument('--cross_lingual', action='store_true',
                       help='Évaluer la conversion cross-linguale')
    
    args = parser.parse_args()
    
    # Create the evaluator
    evaluator = MulliVCEvaluator(args.config, args.checkpoint)
    
    # Create dataloaders
    train_dataloader = create_dataloader(evaluator.config, split='train')
    val_dataloader = create_dataloader(evaluator.config, split='validation')
    
    # Standard evaluation
    print("Évaluation standard...")
    metrics = evaluator.evaluate_dataset(
        val_dataloader,
        os.path.join(args.output_dir, 'standard'),
        args.num_samples
    )
    
    # Cross-lingual evaluation if requested
    if args.cross_lingual:
        print("Évaluation cross-linguale...")
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
    
    print(f"Évaluation terminée. Résultats dans {args.output_dir}")
    print(f"Rapport généré: {report_path}")


if __name__ == '__main__':
    main()
