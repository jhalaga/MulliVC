"""
Utilities for models.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os
import yaml


def get_runtime_device(prefer_cuda: bool = True) -> torch.device:
    """Returns a usable runtime device, falling back to CPU when CUDA is unsupported."""
    if prefer_cuda and torch.cuda.is_available():
        try:
            # Try to allocate a small tensor on CUDA. This is the definitive
            # test: if it works, the GPU is usable (even for architectures
            # not explicitly in get_arch_list() — PTX forward-compatibility
            # handles newer GPUs like Ada sm_89 on sm_80-compiled builds).
            torch.empty(1, device='cuda')
            return torch.device('cuda')
        except Exception as exc:
            print(f"CUDA unavailable at runtime: {exc}. Using CPU instead.")

    return torch.device('cpu')


def load_pretrained_models(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    """
    Loads the required pretrained models.

    Args:
        config: Model configuration.

    Returns:
        models: Dictionary of loaded models.
    """
    models = {}
    
    # Load WavLM for the content encoder
    try:
        from transformers import WavLMModel
        models['wavlm'] = WavLMModel.from_pretrained("microsoft/wavlm-base")
        print("WavLM loaded successfully")
    except Exception as e:
        print(f"Error loading WavLM: {e}")
    
    # Load the speaker verification model
    try:
        from speechbrain.inference import EncoderClassifier
        models['speaker_verification'] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("Speaker verification model loaded successfully")
    except Exception as e:
        print(f"Error loading the speaker verification model: {e}")
    
    # Load Whisper for ASR
    try:
        import whisper
        models['whisper'] = whisper.load_model("base")
        print("Whisper loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper: {e}")
    
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
    Saves a model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer.
        epoch: Epoch number.
        step: Step number.
        loss: Current loss.
        path: Save path.
        config: Optional configuration.
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Create the directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str
) -> Dict[str, Any]:
    """
    Loads a model checkpoint.

    Args:
        model: Model to load.
        optimizer: Optional optimizer.
        path: Checkpoint path.

    Returns:
        checkpoint_info: Checkpoint information.
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'config': checkpoint.get('config', None)
    }
    
    print(f"Checkpoint loaded: {path}")
    print(f"Epoch: {checkpoint_info['epoch']}, Step: {checkpoint_info['step']}")
    
    return checkpoint_info


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Counts the number of model parameters.

    Args:
        model: Model to analyze.

    Returns:
        param_counts: Dictionary of parameter counts.
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
    Prints a model summary.

    Args:
        model: Model to analyze.
    """
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    
    # Count parameters
    param_counts = count_parameters(model)
    
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    
    print("\n" + "=" * 50)
    print("ARCHITECTURE")
    print("=" * 50)
    
    # Display the architecture
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {module_params:,} parameters")
    
    print("=" * 50)


def freeze_parameters(model: nn.Module, freeze: bool = True):
    """
    Freezes or unfreezes model parameters.

    Args:
        model: Model to modify.
        freeze: If True, freezes parameters.
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Parameters {status}")


def initialize_weights(model: nn.Module, init_type: str = 'xavier'):
    """
    Initializes model weights.

    Args:
        model: Model to initialize.
        init_type: Initialization type ('xavier', 'kaiming', 'normal').
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
    print(f"Weights initialized with {init_type}")


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Computes model size in MB.

    Args:
        model: Model to analyze.

    Returns:
        size_info: Size information.
    """
    # Count parameters
    param_counts = count_parameters(model)
    
    # Estimate size (4 bytes per float32 parameter)
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
    Compares two models.

    Args:
        model1: First model.
        model2: Second model.

    Returns:
        comparison: Comparison results.
    """
    comparison = {}
    
    # Count parameters
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    comparison['parameters'] = {
        'model1': params1,
        'model2': params2,
        'difference': params1['total'] - params2['total']
    }
    
    # Compare sizes
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    comparison['size'] = {
        'model1': size1,
        'model2': size2,
        'difference_mb': size1['mb'] - size2['mb']
    }
    
    return comparison
