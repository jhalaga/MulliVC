"""
Utilitaires pour le chargement et le traitement des données
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import random
from datasets import load_dataset
import yaml


class MulliVCDataset(Dataset):
    """Dataset pour MulliVC avec support multilingue"""
    
    def __init__(
        self,
        config: dict,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.config = config
        self.split = split
        self.max_samples = max_samples
        
        # Charger les datasets
        self.libritts_data = self._load_libritts()
        self.fongbe_data = self._load_fongbe()
        
        # Créer les listes d'échantillons
        self.samples = self._create_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_libritts(self) -> List[Dict]:
        """Charge le dataset LibriTTS"""
        try:
            # Charger depuis HuggingFace
            dataset = load_dataset(
                "mythicinfinity/libritts",
                split="train.clean.360",
                streaming=True
            )
            
            # Convertir en liste pour faciliter l'accès
            libritts_samples = []
            for i, sample in enumerate(dataset):
                if i >= 1000:  # Limiter pour les tests
                    break
                libritts_samples.append({
                    'audio': sample['audio'],
                    'text': sample['text_normalized'],
                    'speaker_id': sample['speaker_id'],
                    'language': 'en'
                })
            
            return libritts_samples
            
        except Exception as e:
            print(f"Erreur lors du chargement de LibriTTS: {e}")
            return []
    
    def _load_fongbe(self) -> List[Dict]:
        """Charge le dataset Fongbe"""
        try:
            # Charger depuis HuggingFace
            dataset = load_dataset(
                "beethogedeon/fongbe-speech",
                split="train",
                streaming=True
            )
            
            # Convertir en liste
            fongbe_samples = []
            for i, sample in enumerate(dataset):
                if i >= 1000:  # Limiter pour les tests
                    break
                fongbe_samples.append({
                    'audio': sample['audio'],
                    'text': sample['text'],
                    'speaker_id': sample['speaker_id'],
                    'language': 'fongbe'
                })
            
            return fongbe_samples
            
        except Exception as e:
            print(f"Erreur lors du chargement de Fongbe: {e}")
            return []
    
    def _create_samples(self) -> List[Dict]:
        """Crée la liste des échantillons pour l'entraînement"""
        samples = []
        
        # Ajouter les échantillons LibriTTS
        for sample in self.libritts_data:
            samples.append({
                'audio': sample['audio'],
                'text': sample['text'],
                'speaker_id': sample['speaker_id'],
                'language': sample['language'],
                'dataset': 'libritts'
            })
        
        # Ajouter les échantillons Fongbe
        for sample in self.fongbe_data:
            samples.append({
                'audio': sample['audio'],
                'text': sample['text'],
                'speaker_id': sample['speaker_id'],
                'language': sample['language'],
                'dataset': 'fongbe'
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Récupère un échantillon du dataset"""
        sample = self.samples[idx]
        
        # Charger l'audio
        audio = sample['audio']['array']
        audio_tensor = torch.from_numpy(audio).float()
        
        # Normaliser l'audio
        audio_tensor = audio_tensor / (torch.abs(audio_tensor).max() + 1e-8)
        
        return {
            'audio': audio_tensor,
            'text': sample['text'],
            'speaker_id': sample['speaker_id'],
            'language': sample['language'],
            'dataset': sample['dataset']
        }
    
    def get_speaker_samples(self, speaker_id: str, language: str) -> List[Dict]:
        """Récupère tous les échantillons d'un locuteur dans une langue donnée"""
        speaker_samples = []
        for sample in self.samples:
            if (sample['speaker_id'] == speaker_id and 
                sample['language'] == language):
                speaker_samples.append(sample)
        return speaker_samples
    
    def get_random_speaker(self, language: str) -> str:
        """Récupère un ID de locuteur aléatoire pour une langue donnée"""
        language_samples = [s for s in self.samples if s['language'] == language]
        if not language_samples:
            return None
        
        speaker_ids = list(set([s['speaker_id'] for s in language_samples]))
        return random.choice(speaker_ids)


def create_dataloader(
    config: dict,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Crée un DataLoader pour MulliVC"""
    
    dataset = MulliVCDataset(config, split=split)
    
    batch_size = batch_size or config['data']['batch_size']
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict:
    """Fonction de collation pour le DataLoader"""
    # Séparer les éléments du batch
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    languages = [item['language'] for item in batch]
    datasets = [item['dataset'] for item in batch]
    
    # Pad les audios à la même longueur
    max_length = max([len(audio) for audio in audios])
    padded_audios = []
    
    for audio in audios:
        if len(audio) < max_length:
            padding = torch.zeros(max_length - len(audio))
            padded_audio = torch.cat([audio, padding])
        else:
            padded_audio = audio
        padded_audios.append(padded_audio)
    
    return {
        'audio': torch.stack(padded_audios),
        'text': texts,
        'speaker_id': speaker_ids,
        'language': languages,
        'dataset': datasets
    }


def create_cross_lingual_pairs(
    dataset: MulliVCDataset,
    num_pairs: int = 100
) -> List[Tuple[Dict, Dict]]:
    """
    Crée des paires cross-linguales pour l'entraînement
    
    Args:
        dataset: Dataset MulliVC
        num_pairs: Nombre de paires à créer
        
    Returns:
        List de tuples (sample_A, sample_B) où A et B sont de langues différentes
    """
    pairs = []
    
    # Obtenir les locuteurs par langue
    en_speakers = list(set([
        s['speaker_id'] for s in dataset.samples 
        if s['language'] == 'en'
    ]))
    fongbe_speakers = list(set([
        s['speaker_id'] for s in dataset.samples 
        if s['language'] == 'fongbe'
    ]))
    
    for _ in range(num_pairs):
        # Choisir un locuteur anglais
        en_speaker = random.choice(en_speakers)
        en_samples = dataset.get_speaker_samples(en_speaker, 'en')
        if not en_samples:
            continue
        
        # Choisir un locuteur fongbe
        fongbe_speaker = random.choice(fongbe_speakers)
        fongbe_samples = dataset.get_speaker_samples(fongbe_speaker, 'fongbe')
        if not fongbe_samples:
            continue
        
        # Créer une paire
        en_sample = random.choice(en_samples)
        fongbe_sample = random.choice(fongbe_samples)
        
        pairs.append((en_sample, fongbe_sample))
    
    return pairs


def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
