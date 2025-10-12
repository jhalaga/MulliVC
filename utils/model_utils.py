"""
Utilitaires pour les modèles
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os
import yaml


def load_pretrained_models(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    """
    Charge les modèles pré-entraînés nécessaires
    
    Args:
        config: Configuration du modèle
        
    Returns:
        models: Dictionnaire des modèles chargés
    """
    models = {}
    
    # Charger WavLM pour le content encoder
    try:
        from transformers import WavLMModel
        models['wavlm'] = WavLMModel.from_pretrained("microsoft/wavlm-base")
        print("WavLM chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement de WavLM: {e}")
    
    # Charger le modèle de vérification vocale
    try:
        import speechbrain as sb
        from speechbrain.pretrained import EncoderClassifier
        models['speaker_verification'] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("Modèle de vérification vocale chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de vérification: {e}")
    
    # Charger Whisper pour l'ASR
    try:
        import whisper
        models['whisper'] = whisper.load_model("base")
        print("Whisper chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement de Whisper: {e}")
    
    return models


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    config: Optional[Dict[str, Any]] = None
):
    """
    Sauvegarde un checkpoint du modèle
    
    Args:
        model: Modèle à sauvegarder
        optimizer: Optimiseur
        epoch: Numéro d'époque
        step: Numéro d'étape
        loss: Perte actuelle
        path: Chemin de sauvegarde
        config: Configuration optionnelle
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarder
    torch.save(checkpoint, path)
    print(f"Checkpoint sauvegardé: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str
) -> Dict[str, Any]:
    """
    Charge un checkpoint du modèle
    
    Args:
        model: Modèle à charger
        optimizer: Optimiseur optionnel
        path: Chemin du checkpoint
        
    Returns:
        checkpoint_info: Informations du checkpoint
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Charger l'état du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Charger l'état de l'optimiseur si fourni
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'config': checkpoint.get('config', None)
    }
    
    print(f"Checkpoint chargé: {path}")
    print(f"Époque: {checkpoint_info['epoch']}, Étape: {checkpoint_info['step']}")
    
    return checkpoint_info


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Compte le nombre de paramètres du modèle
    
    Args:
        model: Modèle à analyser
        
    Returns:
        param_counts: Dictionnaire des comptes de paramètres
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_counts = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }
    
    return param_counts


def print_model_summary(model: nn.Module):
    """
    Affiche un résumé du modèle
    
    Args:
        model: Modèle à analyser
    """
    print("=" * 50)
    print("RÉSUMÉ DU MODÈLE")
    print("=" * 50)
    
    # Compter les paramètres
    param_counts = count_parameters(model)
    
    print(f"Paramètres totaux: {param_counts['total']:,}")
    print(f"Paramètres entraînables: {param_counts['trainable']:,}")
    print(f"Paramètres non-entraînables: {param_counts['non_trainable']:,}")
    
    print("\n" + "=" * 50)
    print("ARCHITECTURE")
    print("=" * 50)
    
    # Afficher l'architecture
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {module_params:,} paramètres")
    
    print("=" * 50)


def freeze_parameters(model: nn.Module, freeze: bool = True):
    """
    Gèle ou dégèle les paramètres du modèle
    
    Args:
        model: Modèle à modifier
        freeze: Si True, gèle les paramètres
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    status = "gelés" if freeze else "dégelés"
    print(f"Paramètres {status}")


def initialize_weights(model: nn.Module, init_type: str = 'xavier'):
    """
    Initialise les poids du modèle
    
    Args:
        model: Modèle à initialiser
        init_type: Type d'initialisation ('xavier', 'kaiming', 'normal')
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Conv2d):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    print(f"Poids initialisés avec {init_type}")


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calcule la taille du modèle en MB
    
    Args:
        model: Modèle à analyser
        
    Returns:
        size_info: Informations sur la taille
    """
    # Compter les paramètres
    param_counts = count_parameters(model)
    
    # Estimer la taille (4 bytes par paramètre float32)
    size_bytes = param_counts['total'] * 4
    
    size_info = {
        'bytes': size_bytes,
        'kb': size_bytes / 1024,
        'mb': size_bytes / (1024 * 1024),
        'gb': size_bytes / (1024 * 1024 * 1024)
    }
    
    return size_info


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    Compare deux modèles
    
    Args:
        model1: Premier modèle
        model2: Deuxième modèle
        
    Returns:
        comparison: Résultats de la comparaison
    """
    comparison = {}
    
    # Compter les paramètres
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    comparison['parameters'] = {
        'model1': params1,
        'model2': params2,
        'difference': params1['total'] - params2['total']
    }
    
    # Comparer les tailles
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    comparison['size'] = {
        'model1': size1,
        'model2': size2,
        'difference_mb': size1['mb'] - size2['mb']
    }
    
    return comparison
