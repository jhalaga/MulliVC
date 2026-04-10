"""
Demo script for MulliVC
"""
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from typing import Optional
import yaml

from models.mullivc import MulliVC, create_mullivc_model
from utils.data_utils import load_config
from utils.audio_utils import AudioProcessor
from utils.model_utils import print_model_summary, count_parameters, get_runtime_device


class MulliVCDemo:
    """MulliVC demo."""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.device = get_runtime_device()
        
        # Load the model
        self.model = create_mullivc_model(config_path).to(self.device)
        
        if checkpoint_path:
            self.model.load_checkpoint(checkpoint_path)
            print(f"Modèle chargé depuis {checkpoint_path}")
        else:
            print("Modèle initialisé sans checkpoint (mode démonstration)")
        
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
    
    def load_demo_audio(self, audio_path: str) -> torch.Tensor:
        """Loads demo audio."""
        if not os.path.exists(audio_path):
            print(f"Fichier audio non trouvé: {audio_path}")
            print("Génération d'un audio de démonstration...")
            return self._generate_demo_audio()

        return self.audio_processor.load_audio(audio_path)
    
    def _generate_demo_audio(self, duration: float = 3.0) -> torch.Tensor:
        """Generates demo audio."""
        sample_rate = self.config['data']['sample_rate']
        samples = int(duration * sample_rate)
        
        # Generate a sinusoidal signal with modulation
        t = torch.linspace(0, duration, samples)
        frequency = 440.0  # The A4 note
        audio = torch.sin(2 * np.pi * frequency * t)
        
        # Add modulation
        modulation = 0.1 * torch.sin(2 * np.pi * 5 * t)
        audio = audio * (1 + modulation)
        
        # Normalize
        audio = audio / torch.abs(audio).max()
        
        return audio
    
    def demonstrate_conversion(
        self,
        source_audio_path: str,
        target_speaker_audio_path: str,
        output_dir: str = "demo_output"
    ):
        """Demonstrates voice conversion."""
        print("=" * 60)
        print("DÉMONSTRATION DE CONVERSION VOCALE MULLIVC")
        print("=" * 60)
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        print("Chargement des audios...")
        source_audio = self.load_demo_audio(source_audio_path)
        target_audio = self.load_demo_audio(target_speaker_audio_path)
        
        print(f"Audio source: {source_audio.shape}")
        print(f"Audio cible: {target_audio.shape}")
        
        # Add batch dimension
        source_audio = source_audio.unsqueeze(0).to(self.device)
        target_audio = target_audio.unsqueeze(0).to(self.device)
        
        # Conversion
        print("\nConversion en cours...")
        with torch.no_grad():
            outputs = self.model.forward(source_audio, target_audio)
            generated_mel = outputs['generated_mel']
        
        print(f"Mél-spectrogramme généré: {generated_mel.shape}")
        
        # Convert to audio
        print("Conversion en audio...")
        generated_audio = self.audio_processor.mel_to_audio(generated_mel.squeeze(0))
        
        # Save results
        print("Sauvegarde des résultats...")
        
        # Audio source
        source_path = os.path.join(output_dir, "source.wav")
        self.audio_processor.save_audio(source_path, source_audio.squeeze(0).cpu())
        
        # Target audio
        target_path = os.path.join(output_dir, "target.wav")
        self.audio_processor.save_audio(target_path, target_audio.squeeze(0).cpu())
        
        # Converted audio
        converted_path = os.path.join(output_dir, "converted.wav")
        self.audio_processor.save_audio(converted_path, generated_audio.cpu())
        
        # Visualizations
        self._create_visualizations(
            source_audio.squeeze(0).cpu(),
            target_audio.squeeze(0).cpu(),
            generated_audio.cpu(),
            outputs,
            output_dir
        )
        
        print(f"\nDémonstration terminée! Résultats dans {output_dir}")
        print(f"- Audio source: {source_path}")
        print(f"- Audio cible: {target_path}")
        print(f"- Audio converti: {converted_path}")
    
    def _create_visualizations(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        outputs: dict,
        output_dir: str
    ):
        """Creates visualizations."""
        print("Création des visualisations...")
        
        # Waveforms
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Audio source
        axes[0].plot(source_audio.numpy())
        axes[0].set_title('Audio Source')
        axes[0].set_ylabel('Amplitude')
        
        # Target audio
        axes[1].plot(target_audio.numpy())
        axes[1].set_title('Audio Cible (Timbre)')
        axes[1].set_ylabel('Amplitude')
        
        # Converted audio
        axes[2].plot(converted_audio.numpy())
        axes[2].set_title('Audio Converti')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_xlabel('Échantillons')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'waveforms.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mel spectrograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Source
        source_mel = self.audio_processor.audio_to_mel(source_audio.unsqueeze(0))
        im1 = axes[0].imshow(source_mel.squeeze(0).numpy(), aspect='auto', origin='lower')
        axes[0].set_title('Mél-spectrogramme Source')
        axes[0].set_ylabel('Bandes de fréquence')
        
        # Target
        target_mel = self.audio_processor.audio_to_mel(target_audio.unsqueeze(0))
        im2 = axes[1].imshow(target_mel.squeeze(0).numpy(), aspect='auto', origin='lower')
        axes[1].set_title('Mél-spectrogramme Cible')
        
        # Converted
        im3 = axes[2].imshow(outputs['generated_mel'].squeeze(0).cpu().numpy(), aspect='auto', origin='lower')
        axes[2].set_title('Mél-spectrogramme Converti')
        axes[2].set_xlabel('Frames temporelles')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectrograms.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualisations sauvegardées!")
    
    def demonstrate_cross_lingual(
        self,
        source_audio_path: str,
        target_audio_path: str,
        output_dir: str = "demo_cross_lingual"
    ):
        """Demonstrates cross-lingual conversion."""
        print("=" * 60)
        print("DÉMONSTRATION DE CONVERSION CROSS-LINGUALE")
        print("=" * 60)
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        print("Chargement des audios...")
        source_audio = self.load_demo_audio(source_audio_path)
        target_audio = self.load_demo_audio(target_audio_path)
        
        # Conversion cross-linguale
        print("Conversion cross-linguale...")
        with torch.no_grad():
            outputs = self.model.forward(
                source_audio.unsqueeze(0).to(self.device),
                target_audio.unsqueeze(0).to(self.device)
            )
            generated_mel = outputs['generated_mel']
        
        # Convert to audio
        generated_audio = self.audio_processor.mel_to_audio(generated_mel.squeeze(0))
        
        # Save
        converted_path = os.path.join(output_dir, "cross_lingual_converted.wav")
        self.audio_processor.save_audio(converted_path, generated_audio.cpu())
        
        print(f"Conversion cross-linguale terminée: {converted_path}")
    
    def analyze_model(self):
        """Analyzes the model."""
        print("=" * 60)
        print("ANALYSE DU MODÈLE MULLIVC")
        print("=" * 60)
        
        # Model summary
        print_model_summary(self.model)
        
        # Analyze each component
        print("\n" + "=" * 60)
        print("ANALYSE DES COMPOSANTS")
        print("=" * 60)
        
        components = {
            'Content Encoder': self.model.content_encoder,
            'Timbre Encoder': self.model.timbre_encoder,
            'Fine-grained Conformer': self.model.fine_grained_conformer,
            'Mel Decoder': self.model.mel_decoder,
            'Discriminator': self.model.discriminator
        }
        
        for name, component in components.items():
            params = count_parameters(component)
            print(f"\n{name}:")
            print(f"  Paramètres totaux: {params['total']:,}")
            print(f"  Paramètres entraînables: {params['trainable']:,}")
    
    def benchmark_inference(self, num_runs: int = 10):
        """Benchmarks inference."""
        print("=" * 60)
        print("BENCHMARK D'INFÉRENCE")
        print("=" * 60)
        
        # Generate test audio
        source_audio = self._generate_demo_audio(2.0).unsqueeze(0).to(self.device)
        target_audio = self._generate_demo_audio(2.0).unsqueeze(0).to(self.device)
        
        # Measure inference time
        import time
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.forward(source_audio, target_audio)
            
            end_time = time.time()
            inference_time = end_time - start_time
            times.append(inference_time)
            
            print(f"Run {i+1}/{num_runs}: {inference_time:.3f}s")
        
        # Statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\nRésultats du benchmark:")
        print(f"  Temps moyen: {avg_time:.3f}s")
        print(f"  Écart-type: {std_time:.3f}s")
        print(f"  Temps minimum: {min_time:.3f}s")
        print(f"  Temps maximum: {max_time:.3f}s")
        print(f"  FPS: {1/avg_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Démonstration MulliVC')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--source_audio', type=str, default=None,
                       help='Chemin vers l\'audio source')
    parser.add_argument('--target_audio', type=str, default=None,
                       help='Chemin vers l\'audio cible')
    parser.add_argument('--output_dir', type=str, default='demo_output',
                       help='Dossier de sortie')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyser le modèle')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark de l\'inférence')
    parser.add_argument('--cross_lingual', action='store_true',
                       help='Démonstration cross-linguale')
    
    args = parser.parse_args()
    
    # Create the demo
    demo = MulliVCDemo(args.config, args.checkpoint)
    
    # Analyze the model if requested
    if args.analyze:
        demo.analyze_model()
    
    # Benchmark if requested
    if args.benchmark:
        demo.benchmark_inference()
    
    # Conversion demo
    if args.source_audio or args.target_audio:
        source_path = args.source_audio or "demo_source.wav"
        target_path = args.target_audio or "demo_target.wav"
        
        if args.cross_lingual:
            demo.demonstrate_cross_lingual(source_path, target_path, args.output_dir)
        else:
            demo.demonstrate_conversion(source_path, target_path, args.output_dir)
    else:
        print("Aucun audio spécifié, génération d'audios de démonstration...")
        demo.demonstrate_conversion("demo_source.wav", "demo_target.wav", args.output_dir)


if __name__ == '__main__':
    main()
