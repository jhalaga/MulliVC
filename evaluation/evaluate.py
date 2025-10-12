"""
Script d'évaluation pour MulliVC
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
from evaluation.metrics import ComprehensiveEvaluator


class MulliVCEvaluator:
    """Évaluateur pour MulliVC"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le modèle
        self.model = create_mullivc_model(config_path).to(self.device)
        self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Évaluateur
        self.evaluator = ComprehensiveEvaluator(self.config)
    
    def evaluate_dataset(
        self,
        dataloader,
        output_dir: str,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un dataset
        
        Args:
            dataloader: DataLoader du dataset
            output_dir: Dossier de sortie pour les résultats
            num_samples: Nombre d'échantillons à évaluer
            
        Returns:
            metrics: Métriques d'évaluation
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
                
                # Déplacer sur le device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Conversion
                source_audio = batch['audio']
                target_audio = batch['audio']  # Même audio pour l'évaluation
                
                # Générer l'audio converti
                converted_mel = self.model.inference(source_audio, target_audio)
                converted_audio = self.audio_processor.mel_to_audio(converted_mel.squeeze(0))
                
                # Stocker pour l'évaluation
                source_audios.append(source_audio.squeeze(0).cpu())
                target_audios.append(target_audio.squeeze(0).cpu())
                converted_audios.append(converted_audio.cpu())
                reference_texts.append(batch['text'][0] if 'text' in batch else None)
                
                # Sauvegarder l'audio converti
                output_path = os.path.join(output_dir, f"converted_{i:04d}.wav")
                torchaudio.save(output_path, converted_audio, self.config['data']['sample_rate'])
        
        # Évaluer le batch
        metrics = self.evaluator.evaluate_batch(
            source_audios,
            target_audios,
            converted_audios,
            reference_texts
        )
        
        # Sauvegarder les métriques
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
        Évalue la conversion cross-linguale
        
        Args:
            source_dataloader: DataLoader pour la langue source
            target_dataloader: DataLoader pour la langue cible
            output_dir: Dossier de sortie
            num_samples: Nombre d'échantillons
            
        Returns:
            metrics: Métriques cross-linguales
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Évaluation cross-linguale"):
                # Obtenir un échantillon source
                source_batch = next(iter(source_dataloader))
                source_audio = source_batch['audio'].to(self.device)
                source_text = source_batch['text'][0] if 'text' in source_batch else None
                
                # Obtenir un échantillon cible
                target_batch = next(iter(target_dataloader))
                target_audio = target_batch['audio'].to(self.device)
                
                # Conversion cross-linguale
                converted_mel = self.model.inference(source_audio, target_audio)
                converted_audio = self.audio_processor.mel_to_audio(converted_mel.squeeze(0))
                
                # Évaluer
                metrics = self.evaluator.evaluate_conversion(
                    source_audio.squeeze(0).cpu(),
                    target_audio.squeeze(0).cpu(),
                    converted_audio.cpu(),
                    source_text
                )
                
                all_metrics.append(metrics)
                
                # Sauvegarder
                output_path = os.path.join(output_dir, f"cross_lingual_{i:04d}.wav")
                torchaudio.save(output_path, converted_audio, self.config['data']['sample_rate'])
        
        # Moyenner les métriques
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Sauvegarder
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
        Évalue la vérification du locuteur
        
        Args:
            genuine_pairs: Paires de vrais locuteurs
            impostor_pairs: Paires d'imposteurs
            
        Returns:
            metrics: Métriques de vérification
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
        Génère un rapport d'évaluation
        
        Args:
            metrics: Métriques d'évaluation
            output_path: Chemin vers le rapport
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
    
    # Créer l'évaluateur
    evaluator = MulliVCEvaluator(args.config, args.checkpoint)
    
    # Créer les dataloaders
    train_dataloader = create_dataloader(evaluator.config, split='train')
    val_dataloader = create_dataloader(evaluator.config, split='validation')
    
    # Évaluation standard
    print("Évaluation standard...")
    metrics = evaluator.evaluate_dataset(
        val_dataloader,
        os.path.join(args.output_dir, 'standard'),
        args.num_samples
    )
    
    # Évaluation cross-linguale si demandée
    if args.cross_lingual:
        print("Évaluation cross-linguale...")
        cross_lingual_metrics = evaluator.evaluate_cross_lingual(
            train_dataloader,
            val_dataloader,
            os.path.join(args.output_dir, 'cross_lingual'),
            args.num_samples // 2
        )
        
        # Combiner les métriques
        metrics.update(cross_lingual_metrics)
    
    # Générer le rapport
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    evaluator.generate_evaluation_report(metrics, report_path)
    
    print(f"Évaluation terminée. Résultats dans {args.output_dir}")
    print(f"Rapport généré: {report_path}")


if __name__ == '__main__':
    main()
