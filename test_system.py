"""
Script de test pour vérifier que le système MulliVC fonctionne correctement
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Dict, Any

# Ajouter le répertoire racine au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mullivc import MulliVC, create_mullivc_model
from utils.data_utils import load_config, create_dataloader
from utils.audio_utils import AudioProcessor
from utils.model_utils import count_parameters, print_model_summary


def test_imports():
    """Teste que tous les imports fonctionnent"""
    print("Test des imports...")
    
    try:
        from models.content_encoder import ContentEncoder
        from models.timbre_encoder import TimbreEncoder
        from models.fine_grained_conformer import FineGrainedTimbreConformer
        from models.mel_decoder import MelDecoder
        from models.discriminator import PatchGANDiscriminator
        from models.losses import CombinedLoss
        print("✓ Imports des modèles réussis")
    except Exception as e:
        print(f"✗ Erreur lors des imports des modèles: {e}")
        return False
    
    try:
        from utils.audio_utils import MelSpectrogram, AudioProcessor
        from utils.data_utils import MulliVCDataset, create_dataloader
        from utils.model_utils import load_pretrained_models, save_checkpoint
        print("✓ Imports des utilitaires réussis")
    except Exception as e:
        print(f"✗ Erreur lors des imports des utilitaires: {e}")
        return False
    
    try:
        from evaluation.metrics import SpeakerSimilarityMetric, ASRMetric, AudioQualityMetric
        print("✓ Imports des métriques réussis")
    except Exception as e:
        print(f"✗ Erreur lors des imports des métriques: {e}")
        return False
    
    return True


def test_config_loading():
    """Teste le chargement de la configuration"""
    print("\nTest du chargement de configuration...")
    
    try:
        config = load_config('configs/mullivc_config.yaml')
        print("✓ Configuration chargée avec succès")
        
        # Vérifier les sections principales
        required_sections = ['data', 'model', 'training', 'paths']
        for section in required_sections:
            if section in config:
                print(f"✓ Section '{section}' présente")
            else:
                print(f"✗ Section '{section}' manquante")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du chargement de la configuration: {e}")
        return False


def test_model_creation():
    """Teste la création du modèle"""
    print("\nTest de création du modèle...")
    
    try:
        # Charger la configuration
        config = load_config('configs/mullivc_config.yaml')
        
        # Créer le modèle
        model = create_mullivc_model('configs/mullivc_config.yaml')
        print("✓ Modèle créé avec succès")
        
        # Vérifier les composants
        components = [
            'content_encoder',
            'timbre_encoder', 
            'fine_grained_conformer',
            'mel_decoder',
            'discriminator'
        ]
        
        for component in components:
            if hasattr(model, component):
                print(f"✓ Composant '{component}' présent")
            else:
                print(f"✗ Composant '{component}' manquant")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors de la création du modèle: {e}")
        return False


def test_model_forward():
    """Teste le forward pass du modèle"""
    print("\nTest du forward pass...")
    
    try:
        # Créer le modèle
        model = create_mullivc_model('configs/mullivc_config.yaml')
        model.eval()
        
        # Créer des données de test
        batch_size = 2
        seq_len = 1000
        n_mel_channels = 80
        time_frames = 100
        
        # Audio de test
        source_audio = torch.randn(batch_size, seq_len)
        target_audio = torch.randn(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            outputs = model.forward(source_audio, target_audio)
        
        # Vérifier les sorties
        expected_outputs = [
            'generated_mel',
            'content_features', 
            'timbre_features',
            'fine_grained_timbre',
            'global_timbre',
            'discriminator_output'
        ]
        
        for output in expected_outputs:
            if output in outputs:
                print(f"✓ Sortie '{output}' générée: {outputs[output].shape}")
            else:
                print(f"✗ Sortie '{output}' manquante")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du forward pass: {e}")
        return False


def test_audio_processing():
    """Teste le traitement audio"""
    print("\nTest du traitement audio...")
    
    try:
        config = load_config('configs/mullivc_config.yaml')
        audio_processor = AudioProcessor(config)
        
        # Créer un audio de test
        sample_rate = config['data']['sample_rate']
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = torch.randn(samples)
        
        # Test de conversion en mél-spectrogramme
        mel_spec = audio_processor.audio_to_mel(audio)
        print(f"✓ Mél-spectrogramme généré: {mel_spec.shape}")
        
        # Test de conversion inverse
        reconstructed_audio = audio_processor.mel_to_audio(mel_spec)
        print(f"✓ Audio reconstruit: {reconstructed_audio.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du traitement audio: {e}")
        return False


def test_loss_computation():
    """Teste le calcul des pertes"""
    print("\nTest du calcul des pertes...")
    
    try:
        config = load_config('configs/mullivc_config.yaml')
        model = create_mullivc_model('configs/mullivc_config.yaml')
        
        # Créer des données de test
        batch_size = 2
        seq_len = 1000
        n_mel_channels = 80
        time_frames = 100
        
        source_audio = torch.randn(batch_size, seq_len)
        target_audio = torch.randn(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            outputs = model.forward(source_audio, target_audio)
        
        # Créer les cibles
        targets = {
            'target_mel': torch.randn(batch_size, n_mel_channels, time_frames),
            'target_timbre': torch.randn(batch_size, 256),
            'target_pitch': torch.randn(batch_size, time_frames),
            'target_voiced': torch.randn(batch_size, time_frames)
        }
        
        # Calculer les pertes
        losses = model.compute_losses(outputs, targets, is_real=True)
        
        # Vérifier les pertes
        expected_losses = ['total', 'reconstruction', 'timbre', 'pitch', 'adversarial', 'asr']
        for loss_name in expected_losses:
            if loss_name in losses:
                print(f"✓ Perte '{loss_name}': {losses[loss_name].item():.4f}")
            else:
                print(f"✗ Perte '{loss_name}' manquante")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du calcul des pertes: {e}")
        return False


def test_model_parameters():
    """Teste les paramètres du modèle"""
    print("\nTest des paramètres du modèle...")
    
    try:
        model = create_mullivc_model('configs/mullivc_config.yaml')
        
        # Compter les paramètres
        param_counts = count_parameters(model)
        print(f"✓ Paramètres totaux: {param_counts['total']:,}")
        print(f"✓ Paramètres entraînables: {param_counts['trainable']:,}")
        print(f"✓ Paramètres non-entraînables: {param_counts['non_trainable']:,}")
        
        # Vérifier que le modèle a des paramètres
        if param_counts['total'] > 0:
            print("✓ Modèle a des paramètres")
        else:
            print("✗ Modèle n'a pas de paramètres")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test des paramètres: {e}")
        return False


def test_gradient_flow():
    """Teste le flux des gradients"""
    print("\nTest du flux des gradients...")
    
    try:
        model = create_mullivc_model('configs/mullivc_config.yaml')
        model.train()
        
        # Créer des données de test
        source_audio = torch.randn(2, 1000, requires_grad=True)
        target_audio = torch.randn(2, 1000, requires_grad=True)
        
        # Forward pass
        outputs = model.forward(source_audio, target_audio)
        
        # Calculer une perte simple
        loss = outputs['generated_mel'].mean()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients existent
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Gradients calculés avec succès")
        else:
            print("✗ Aucun gradient calculé")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test des gradients: {e}")
        return False


def run_all_tests():
    """Exécute tous les tests"""
    print("=" * 60)
    print("TESTS DU SYSTÈME MULLIVC")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_loading),
        ("Création du modèle", test_model_creation),
        ("Forward pass", test_model_forward),
        ("Traitement audio", test_audio_processing),
        ("Calcul des pertes", test_loss_computation),
        ("Paramètres du modèle", test_model_parameters),
        ("Flux des gradients", test_gradient_flow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✓ {test_name} réussi")
                passed += 1
            else:
                print(f"✗ {test_name} échoué")
        except Exception as e:
            print(f"✗ {test_name} échoué avec erreur: {e}")
    
    print("\n" + "=" * 60)
    print(f"RÉSULTATS: {passed}/{total} tests réussis")
    print("=" * 60)
    
    if passed == total:
        print("🎉 Tous les tests sont passés! Le système MulliVC est prêt.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
